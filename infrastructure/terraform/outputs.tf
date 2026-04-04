output "s3_bucket_name" {
  description = "Name of the S3 bucket for application data"
  value       = aws_s3_bucket.app_data.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.app_data.arn
}

output "ecr_repository_url" {
  description = "URL of the ECR repository for model images"
  value       = aws_ecr_repository.model.repository_url
}

output "ecr_repository_arn" {
  description = "ARN of the ECR repository"
  value       = aws_ecr_repository.model.arn
}

output "sagemaker_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker.arn
}

output "sagemaker_model_name" {
  description = "Name of the SageMaker model"
  value       = aws_sagemaker_model.main.name
}
