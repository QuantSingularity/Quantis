from typing import Any

from sagemaker.pytorch import PyTorchModel


def deploy_to_sagemaker() -> Any:
    model = PyTorchModel(
        entry_point="inference.py",
        role="SageMakerRole",
        framework_version="2.0",
        model_data="s3://models/tft_model.tar.gz",
    )
    predictor = model.deploy(instance_type="ml.g4dn.xlarge", initial_instance_count=1)
    return predictor.endpoint
