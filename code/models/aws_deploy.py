"""
AWS deployment utilities for Quantis models.
"""

from typing import Any


def deploy_to_aws(model_path: str, bucket_name: str) -> Any:
    """
    Upload a trained model file to an S3 bucket.

    Args:
        model_path: Local path to the model file.
        bucket_name: Target S3 bucket name.

    Returns:
        S3 object key of the uploaded model.
    """
    import boto3

    s3_client = boto3.client("s3")
    s3_key = f"models/{model_path}"
    s3_client.upload_file(model_path, bucket_name, s3_key)
    return s3_key


def deploy_to_sagemaker(
    model_data_s3_uri: str = "s3://models/tft_model.tar.gz",
    role: str = "SageMakerRole",
    instance_type: str = "ml.g4dn.xlarge",
) -> Any:
    """
    Deploy a model to AWS SageMaker.

    Args:
        model_data_s3_uri: S3 URI for the packaged model artifact.
        role: IAM role ARN for SageMaker.
        instance_type: SageMaker instance type.

    Returns:
        Endpoint name string.
    """
    try:
        from sagemaker.pytorch import PyTorchModel
    except ImportError as exc:
        raise ImportError(
            "sagemaker package is required for SageMaker deployment. "
            "Install it with: pip install sagemaker"
        ) from exc

    model = PyTorchModel(
        entry_point="inference.py",
        role=role,
        framework_version="2.0",
        model_data=model_data_s3_uri,
    )
    predictor = model.deploy(instance_type=instance_type, initial_instance_count=1)
    return predictor.endpoint
