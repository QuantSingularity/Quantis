from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from models.train_model import TemporalFusionTransformer, train_model


def test_model_with_different_input_sizes() -> Any:
    min_model = TemporalFusionTransformer(input_size=64)
    min_input = torch.randn(1, 64)
    min_output = min_model(min_input)
    assert min_output.shape == (1, 1)
    max_model = TemporalFusionTransformer(input_size=256)
    max_input = torch.randn(1, 256)
    max_output = max_model(max_input)
    assert max_output.shape == (1, 1)


def test_model_with_different_batch_sizes(sample_model: Any) -> Any:
    small_batch = torch.randn(5, 128)
    small_output = sample_model(small_batch)
    assert small_output.shape == (5, 1)
    large_batch = torch.randn(100, 128)
    large_output = sample_model(large_batch)
    assert large_output.shape == (100, 1)
    single_sample = torch.randn(1, 128)
    single_output = sample_model(single_sample)
    assert single_output.shape == (1, 1)


def test_model_with_different_data_types(sample_model: Any) -> Any:
    float32_input = torch.randn(1, 128, dtype=torch.float32)
    float32_output = sample_model(float32_input)
    assert float32_output.dtype == torch.float32
    float64_input = torch.randn(1, 128, dtype=torch.float64)
    with pytest.raises(Exception):
        sample_model(float64_input)


def test_model_with_extreme_values(sample_model: Any) -> Any:
    large_values = torch.ones(1, 128) * 10000000000.0
    large_output = sample_model(large_values)
    assert not torch.isnan(large_output).any()
    small_values = torch.ones(1, 128) * 1e-10
    small_output = sample_model(small_values)
    assert not torch.isnan(small_output).any()
    mixed_values = torch.ones(1, 128)
    mixed_values[0, ::2] = -1
    mixed_output = sample_model(mixed_values)
    assert not torch.isnan(mixed_output).any()


@patch("torch.optim.Adam")
def test_model_training_with_different_optimizers(
    mock_adam: Any, sample_model: Any, mock_mlflow: Any
) -> Any:
    mock_optimizer = MagicMock()
    mock_adam.return_value = mock_optimizer
    X_train = torch.randn(100, 128)
    y_train = torch.randn(100, 1)
    trained_model = train_model(sample_model, X_train, y_train, mock_mlflow)
    mock_adam.assert_called_once()
    assert isinstance(trained_model, TemporalFusionTransformer)


def test_model_robustness_to_noise(sample_model: Any) -> Any:
    base_input = torch.randn(1, 128)
    base_output = sample_model(base_input)
    noise_levels = [0.01, 0.1, 0.5]
    for noise_level in noise_levels:
        noisy_input = base_input + torch.randn(1, 128) * noise_level
        noisy_output = sample_model(noisy_input)
        output_diff = torch.norm(base_output - noisy_output).item()
        assert output_diff < noise_level * 10


def test_model_with_adversarial_inputs(sample_model: Any) -> Any:
    base_input = torch.randn(1, 128, requires_grad=True)
    output = sample_model(base_input)
    loss = output.mean()
    loss.backward()
    epsilon = 0.1
    adversarial_input = base_input + epsilon * base_input.grad.sign()
    adversarial_output = sample_model(adversarial_input.detach())
    assert not torch.isnan(adversarial_output).any()


def test_model_serialization_formats(sample_model: Any, tmp_path: Any) -> Any:
    torch_path = tmp_path / "model.pt"
    torch.save(sample_model, torch_path)
    loaded_model = torch.load(torch_path)
    assert isinstance(loaded_model, TemporalFusionTransformer)
    state_dict_path = tmp_path / "model_state_dict.pt"
    torch.save(sample_model.state_dict(), state_dict_path)
    new_model = TemporalFusionTransformer(input_size=128)
    new_model.load_state_dict(torch.load(state_dict_path))
    test_input = torch.randn(1, 128)
    original_output = sample_model(test_input)
    loaded_output = new_model(test_input)
    assert torch.allclose(original_output, loaded_output)


def test_model_performance_metrics(sample_model: Any) -> Any:
    num_samples = 100
    X_test = torch.randn(num_samples, 128)
    y_test = torch.randn(num_samples, 1)
    y_pred = sample_model(X_test)
    mse = torch.mean((y_pred - y_test) ** 2).item()
    mae = torch.mean(torch.abs(y_pred - y_test)).item()
    y_mean = torch.mean(y_test)
    ss_tot = torch.sum((y_test - y_mean) ** 2)
    ss_res = torch.sum((y_test - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert 0 <= mse
    assert 0 <= mae
    assert r2 <= 1.0


def test_model_with_time_features(sample_model: Any) -> Any:
    batch_size = 10
    seq_len = 128
    time_features = torch.zeros(batch_size, seq_len)
    for i in range(seq_len):
        time_features[:, i] = i % 24
    time_features = time_features / 24.0
    output = sample_model(time_features)
    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any()


def test_model_with_missing_values(sample_model: Any) -> Any:
    input_with_nans = torch.randn(1, 128)
    input_with_nans[0, 10:20] = float("nan")
    input_fixed = torch.nan_to_num(input_with_nans, nan=0.0)
    output = sample_model(input_fixed)
    assert not torch.isnan(output).any()


def test_model_with_different_learning_rates(
    sample_model: Any, mock_mlflow: Any
) -> Any:
    X_train = torch.randn(100, 128)
    y_train = torch.randn(100, 1)
    learning_rates = [0.001, 0.01, 0.1]
    for lr in learning_rates:
        sample_model.learning_rate = lr
        trained_model = train_model(sample_model, X_train, y_train, mock_mlflow)
        assert isinstance(trained_model, TemporalFusionTransformer)
        assert trained_model.learning_rate == lr
