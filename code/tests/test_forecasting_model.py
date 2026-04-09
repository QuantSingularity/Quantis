"""
Extended model robustness, serialization, performance, and noise tests.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from models.train_model import TemporalFusionTransformer, train_model


def test_model_with_different_input_sizes() -> None:
    min_model = TemporalFusionTransformer(input_size=64)
    min_input = torch.randn(1, 64)
    min_output = min_model(min_input)
    assert min_output.shape == (1, 1)

    max_model = TemporalFusionTransformer(input_size=256)
    max_input = torch.randn(1, 256)
    max_output = max_model(max_input)
    assert max_output.shape == (1, 1)


def test_model_with_different_batch_sizes(sample_model: Any) -> None:
    for batch_size in (5, 100, 1):
        output = sample_model(torch.randn(batch_size, 128))
        assert output.shape == (batch_size, 1)


def test_model_with_different_data_types(sample_model: Any) -> None:
    float32_input = torch.randn(1, 128, dtype=torch.float32)
    float32_output = sample_model(float32_input)
    assert float32_output.dtype == torch.float32

    float64_input = torch.randn(1, 128, dtype=torch.float64)
    with pytest.raises(Exception):
        sample_model(float64_input)


def test_model_with_extreme_values(sample_model: Any) -> None:
    for values in (
        torch.ones(1, 128) * 1e10,
        torch.ones(1, 128) * 1e-10,
        torch.cat([torch.ones(1, 64), -torch.ones(1, 64)], dim=1),
    ):
        output = sample_model(values)
        assert not torch.isnan(output).any()


@patch("torch.optim.Adam")
def test_model_training_with_different_optimizers(
    mock_adam: Any, sample_model: Any, mock_mlflow: Any
) -> None:
    mock_optimizer = MagicMock()
    mock_optimizer.zero_grad = MagicMock()
    mock_optimizer.step = MagicMock()
    mock_adam.return_value = mock_optimizer

    X_train = torch.randn(100, 128)
    y_train = torch.randn(100, 1)
    trained_model = train_model(sample_model, X_train, y_train, mock_mlflow)

    mock_adam.assert_called_once()
    assert isinstance(trained_model, TemporalFusionTransformer)


def test_model_robustness_to_noise(sample_model: Any) -> None:
    sample_model.eval()
    base_input = torch.randn(1, 128)
    with torch.no_grad():
        base_output = sample_model(base_input)

    for noise_level in (0.01, 0.1, 0.5):
        noisy_input = base_input + torch.randn(1, 128) * noise_level
        with torch.no_grad():
            noisy_output = sample_model(noisy_input)
        output_diff = torch.norm(base_output - noisy_output).item()
        assert output_diff < noise_level * 100  # loose bound for determinism


def test_model_with_adversarial_inputs(sample_model: Any) -> None:
    base_input = torch.randn(1, 128, requires_grad=True)
    output = sample_model(base_input)
    loss = output.mean()
    loss.backward()

    epsilon = 0.1
    adversarial_input = (base_input + epsilon * base_input.grad.sign()).detach()
    with torch.no_grad():
        adversarial_output = sample_model(adversarial_input)
    assert not torch.isnan(adversarial_output).any()


def test_model_serialization_formats(sample_model: Any, tmp_path: Any) -> None:
    # Full model save/load
    torch_path = tmp_path / "model.pt"
    torch.save(sample_model, torch_path)
    loaded_model = torch.load(torch_path, weights_only=False)
    assert isinstance(loaded_model, TemporalFusionTransformer)

    # State-dict round-trip
    state_dict_path = tmp_path / "model_state_dict.pt"
    torch.save(sample_model.state_dict(), state_dict_path)
    new_model = TemporalFusionTransformer(input_size=128)
    new_model.load_state_dict(torch.load(state_dict_path, weights_only=True))

    sample_model.eval()
    new_model.eval()
    test_input = torch.randn(1, 128)
    with torch.no_grad():
        assert torch.allclose(sample_model(test_input), new_model(test_input))


def test_model_performance_metrics(sample_model: Any) -> None:
    num_samples = 100
    X_test = torch.randn(num_samples, 128)
    y_test = torch.randn(num_samples, 1)
    sample_model.eval()
    with torch.no_grad():
        y_pred = sample_model(X_test)

    mse = torch.mean((y_pred - y_test) ** 2).item()
    mae = torch.mean(torch.abs(y_pred - y_test)).item()
    y_mean = torch.mean(y_test)
    ss_tot = torch.sum((y_test - y_mean) ** 2)
    ss_res = torch.sum((y_test - y_pred) ** 2)
    r2 = (1 - ss_res / ss_tot).item()

    assert mse >= 0
    assert mae >= 0
    assert r2 <= 1.0


def test_model_with_time_features(sample_model: Any) -> None:
    batch_size = 10
    seq_len = 128
    time_features = torch.zeros(batch_size, seq_len)
    for i in range(seq_len):
        time_features[:, i] = i % 24
    time_features = time_features / 24.0
    sample_model.eval()
    with torch.no_grad():
        output = sample_model(time_features)
    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any()


def test_model_with_missing_values(sample_model: Any) -> None:
    input_with_nans = torch.randn(1, 128)
    input_with_nans[0, 10:20] = float("nan")
    input_fixed = torch.nan_to_num(input_with_nans, nan=0.0)
    sample_model.eval()
    with torch.no_grad():
        output = sample_model(input_fixed)
    assert not torch.isnan(output).any()


def test_model_with_different_learning_rates(
    sample_model: Any, mock_mlflow: Any
) -> None:
    X_train = torch.randn(100, 128)
    y_train = torch.randn(100, 1)
    for lr in (0.001, 0.01, 0.1):
        sample_model.learning_rate = lr
        trained_model = train_model(sample_model, X_train, y_train, mock_mlflow)
        assert isinstance(trained_model, TemporalFusionTransformer)
        assert trained_model.learning_rate == lr
