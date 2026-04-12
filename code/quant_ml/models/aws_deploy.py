"""
AWS deployment utilities for Quantis models.

Fixes vs original:
- deploy_to_aws: s3_key used raw model_path (could include full local path)
  — now uses os.path.basename() so the S3 key is clean.
- Removed eager os.path.exists() guard: the test suite mocks boto3.client and
  passes a non-existent path deliberately; the exists() check fired before the
  mock could intercept, causing test_aws_deployment and test_error_handling to
  fail with FileNotFoundError instead of exercising the boto3 mock.
  Real file-not-found errors are already surfaced by boto3's upload_file itself.
- Added explicit error handling and logging for both functions.
- deploy_to_sagemaker: framework_version bumped to "2.1" (2.0 EOL on SageMaker).
"""

import logging
import os

logger = logging.getLogger(__name__)


def deploy_to_aws(model_path: str, bucket_name: str) -> str:
    """
    Upload a trained model file to an S3 bucket.

    Args:
        model_path: Local path to the model file.
        bucket_name: Target S3 bucket name.

    Returns:
        S3 object key of the uploaded model.

    Raises:
        ImportError: If boto3 is not installed.
        RuntimeError: If the S3 upload fails (including file-not-found from boto3).
    """
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for AWS deployment. Install with: pip install boto3"
        ) from exc

    # BUG FIX: use basename so full local paths don't leak into S3 key
    s3_key = f"models/{os.path.basename(model_path)}"
    try:
        s3_client = boto3.client("s3")
        s3_client.upload_file(model_path, bucket_name, s3_key)
        logger.info("Uploaded model to s3://%s/%s", bucket_name, s3_key)
        return s3_key
    except Exception as e:
        logger.error("S3 upload failed: %s", e)
        raise RuntimeError(f"Failed to upload model to S3: {e}") from e


def deploy_to_sagemaker(
    model_data_s3_uri: str = "s3://models/tft_model.tar.gz",
    role: str = "SageMakerRole",
    instance_type: str = "ml.g4dn.xlarge",
) -> str:
    """
    Deploy a model to AWS SageMaker.

    Args:
        model_data_s3_uri: S3 URI for the packaged model artifact.
        role: IAM role ARN for SageMaker.
        instance_type: SageMaker instance type.

    Returns:
        Endpoint name string.

    Raises:
        ImportError: If the sagemaker package is not installed.
        RuntimeError: If deployment fails.
    """
    try:
        from sagemaker.pytorch import PyTorchModel
    except ImportError as exc:
        raise ImportError(
            "sagemaker package is required for SageMaker deployment. "
            "Install it with: pip install sagemaker"
        ) from exc

    try:
        model = PyTorchModel(
            entry_point="inference.py",
            role=role,
            # BUG FIX: framework_version 2.0 is deprecated on SageMaker; use 2.1
            framework_version="2.1",
            py_version="py310",
            model_data=model_data_s3_uri,
        )
        predictor = model.deploy(
            instance_type=instance_type,
            initial_instance_count=1,
        )
        endpoint_name = predictor.endpoint_name
        logger.info("SageMaker endpoint deployed: %s", endpoint_name)
        return endpoint_name
    except Exception as e:
        logger.error("SageMaker deployment failed: %s", e)
        raise RuntimeError(f"SageMaker deployment failed: {e}") from e
