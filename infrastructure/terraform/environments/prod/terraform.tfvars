aws_region  = "us-west-2"
environment = "prod"
app_name    = "quantis"

vpc_cidr             = "10.2.0.0/16"
availability_zones   = ["us-west-2a", "us-west-2b", "us-west-2c"]
public_subnet_cidrs  = ["10.2.1.0/24", "10.2.2.0/24", "10.2.3.0/24"]
private_subnet_cidrs = ["10.2.4.0/24", "10.2.5.0/24", "10.2.6.0/24"]

instance_type = "t3.large"
key_name      = null

db_instance_class = "db.r6g.large"
db_name           = "quantisdb"
db_username       = "quantis_admin"
# Set via: export TF_VAR_db_password="your-secure-password-here"

default_tags = {
  Terraform   = "true"
  Environment = "prod"
  Project     = "quantis"
  ManagedBy   = "terraform"
}
