"""
Tests for TemporalFusionTransformer model: initialization, forward pass,
training, serialization, inference, gradient flow, hyperparameters, and error handling.
"""

from typing import Any

import pytest
import torch
from models.mlflow_tracking import log_metrics
from models.train_model import TemporalFusionTransformer, train_model


def test_model_loading(tmp_path: Any) -> Any:
    """Model weights can be saved and reloaded correctly."""
    model = TemporalFusionTransformer(input_size=128)
    save_path = tmp_path / "tft_model.pt"
    torch.save(model.state_dict(), save_path)

    loaded = TemporalFusionTransformer(input_size=128)
    loaded.load_state_dict(torch.load(save_path))
    loaded.eval()

    inp = torch.randn(1, 128)
    with torch.no_grad():
        assert loaded(inp).shape == (1, 1)


def test_api_endpoint(test_client: Any) -> Any:
    """Health endpoint is reachable."""
    response = test_client.get("/health")
    assert response.status_code == 200


def test_model_initialization(sample_model: Any) -> Any:
    assert isinstance(sample_model, TemporalFusionTransformer)
    assert sample_model.input_size == 128


def test_model_forward_pass(sample_model: Any) -> Any:
    input_tensor = torch.randn(1, 128)
    output = sample_model(input_tensor)
    assert output.shape == (1, 1)
    assert not torch.isnan(output).any()


def test_model_training(sample_model: Any, mock_mlflow: Any) -> Any:
    X_train = torch.randn(100, 128)
    y_train = torch.randn(100, 1)
    trained_model = train_model(sample_model, X_train, y_train, mock_mlflow)
    assert isinstance(trained_model, TemporalFusionTransformer)
    assert len(mock_mlflow.metrics) > 0
    assert "loss" in mock_mlflow.metrics


def test_model_save_load(sample_model: Any, tmp_path: Any) -> Any:
    save_path = tmp_path / "test_model.pt"
    torch.save(sample_model.state_dict(), save_path)
    loaded_model = TemporalFusionTransformer(input_size=128)
    loaded_model.load_state_dict(torch.load(save_path))
    assert isinstance(loaded_model, TemporalFusionTransformer)
    input_tensor = torch.randn(1, 128)
    sample_model.eval()
    loaded_model.eval()
    with torch.no_grad():
        assert torch.allclose(sample_model(input_tensor), loaded_model(input_tensor))


def test_mlflow_tracking(mock_mlflow: Any) -> Any:
    metrics = {"accuracy": 0.95, "loss": 0.1}
    log_metrics(metrics, mock_mlflow)
    assert mock_mlflow.metrics["accuracy"] == 0.95
    assert mock_mlflow.metrics["loss"] == 0.1


def test_model_inference(sample_model: Any) -> Any:
    input_tensor = torch.randn(1, 128)
    sample_model.eval()
    with torch.no_grad():
        prediction = sample_model(input_tensor)
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape == (1, 1)
    assert not torch.isnan(prediction).any()
    assert not torch.isinf(prediction).any()


def test_model_batch_processing(sample_model: Any) -> Any:
    batch_size = 32
    input_batch = torch.randn(batch_size, 128)
    sample_model.eval()
    with torch.no_grad():
        output_batch = sample_model(input_batch)
    assert output_batch.shape == (batch_size, 1)
    assert not torch.isnan(output_batch).any()
    assert not torch.isinf(output_batch).any()


def test_model_gradient_flow(sample_model: Any) -> Any:
    input_tensor = torch.randn(1, 128, requires_grad=True)
    output = sample_model(input_tensor)
    loss = output.mean()
    loss.backward()
    assert input_tensor.grad is not None
    assert not torch.isnan(input_tensor.grad).any()


def test_model_hyperparameters(sample_model: Any) -> Any:
    assert hasattr(sample_model, "learning_rate")
    assert hasattr(sample_model, "batch_size")
    assert hasattr(sample_model, "num_epochs")
    assert sample_model.learning_rate > 0
    assert sample_model.batch_size > 0
    assert sample_model.num_epochs > 0


def test_model_regularization(sample_model: Any) -> Any:
    """Dropout causes different outputs for same input in train mode."""
    sample_model.train()
    input_tensor = torch.randn(1, 128)
    # Run several times; at least one pair should differ due to dropout
    outputs = [sample_model(input_tensor) for _ in range(10)]
    all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
    # With dropout p=0.2 over 10 runs, outputs should not all be identical
    assert not all_same


def test_model_memory_efficiency(sample_model: Any) -> Any:
    """Model handles large batch without OOM (CPU)."""
    batch_size = 1024
    input_batch = torch.randn(batch_size, 128)
    sample_model.eval()
    with torch.no_grad():
        output = sample_model(input_batch)
    assert output.shape == (batch_size, 1)

    # Only check CUDA stats when CUDA is actually available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        cuda_input = input_batch.cuda()
        sample_model.cuda()
        with torch.no_grad():
            _ = sample_model(cuda_input)
        final_memory = torch.cuda.memory_allocated()
        memory_used = final_memory - initial_memory
        assert memory_used < 1_000_000_000


def test_model_error_handling(sample_model: Any) -> Any:
    """Model raises ValueError for wrong input shape."""
    with pytest.raises(ValueError):
        sample_model(torch.randn(1, 64))  # wrong feature dim
    with pytest.raises(ValueError):
        sample_model(torch.randn(1, 128, 1))  # wrong number of dims
