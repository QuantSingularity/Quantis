from typing import Any
import pytest
import torch
from models.mlflow_tracking import log_metrics
from models.train_model import TemporalFusionTransformer, train_model


def test_model_loading() -> Any:
    model = TemporalFusionTransformer(input_size=128)
    model.load_state_dict(torch.load("tft_model.pt"))
    assert model(torch.randn(1, 128)).shape == (1, 1)


def test_api_endpoint(test_client: Any) -> Any:
    response = test_client.post(
        "/predict", json={"features": [0.1, 0.5, ...], "api_key": "valid_key"}
    )
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
    assert torch.allclose(sample_model(input_tensor), loaded_model(input_tensor))


def test_mlflow_tracking(mock_mlflow: Any) -> Any:
    metrics = {"accuracy": 0.95, "loss": 0.1}
    log_metrics(metrics, mock_mlflow)
    assert mock_mlflow.metrics["accuracy"] == 0.95
    assert mock_mlflow.metrics["loss"] == 0.1


def test_model_inference(sample_model: Any) -> Any:
    input_tensor = torch.randn(1, 128)
    prediction = sample_model(input_tensor)
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape == (1, 1)
    assert not torch.isnan(prediction).any()
    assert not torch.isinf(prediction).any()


def test_model_batch_processing(sample_model: Any) -> Any:
    batch_size = 32
    input_batch = torch.randn(batch_size, 128)
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
    input_tensor = torch.randn(1, 128)
    output1 = sample_model(input_tensor)
    noisy_input = input_tensor + torch.randn_like(input_tensor) * 0.1
    output2 = sample_model(noisy_input)
    assert not torch.allclose(output1, output2)
    assert torch.norm(output1 - output2) < 1.0


def test_model_memory_efficiency(sample_model: Any) -> Any:
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    batch_size = 1024
    input_batch = torch.randn(batch_size, 128)
    _ = sample_model(input_batch)
    final_memory = torch.cuda.memory_allocated()
    memory_used = final_memory - initial_memory
    assert memory_used < 1000000000.0


def test_model_error_handling(sample_model: Any) -> Any:
    with pytest.raises(ValueError):
        sample_model(torch.randn(1, 64))
    with pytest.raises(ValueError):
        sample_model(torch.randn(1, 128, 1))
