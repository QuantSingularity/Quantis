.PHONY: help up down build logs ps clean tf-init tf-plan tf-apply tf-destroy k8s-apply k8s-delete

ENVIRONMENT ?= dev
COMPOSE = docker compose
TF_DIR = terraform/environments/$(ENVIRONMENT)

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

## Docker Compose
up: ## Start all services (copy .env.example to .env first)
	$(COMPOSE) up -d

down: ## Stop all services
	$(COMPOSE) down

build: ## Build all Docker images
	$(COMPOSE) build

logs: ## Tail logs for all services
	$(COMPOSE) logs -f

ps: ## Show running containers
	$(COMPOSE) ps

clean: ## Stop services and remove volumes (DESTRUCTIVE)
	$(COMPOSE) down -v --remove-orphans

restart: ## Restart all services
	$(COMPOSE) restart

## Terraform
tf-init: ## Initialize Terraform for ENVIRONMENT (default: dev)
	cd $(TF_DIR) && terraform init

tf-plan: ## Plan Terraform changes for ENVIRONMENT
	cd $(TF_DIR) && terraform plan -var-file=terraform.tfvars

tf-apply: ## Apply Terraform changes for ENVIRONMENT
	cd $(TF_DIR) && terraform apply -var-file=terraform.tfvars

tf-destroy: ## Destroy Terraform resources for ENVIRONMENT (DESTRUCTIVE)
	cd $(TF_DIR) && terraform destroy -var-file=terraform.tfvars

tf-fmt: ## Format all Terraform files
	terraform fmt -recursive terraform/

tf-validate: ## Validate all Terraform configurations
	cd terraform && terraform validate

## Kubernetes
k8s-apply: ## Apply all Kubernetes base manifests
	kubectl apply -f kubernetes/base/

k8s-delete: ## Delete all Kubernetes base manifests
	kubectl delete -f kubernetes/base/

k8s-status: ## Show status of all Quantis pods
	kubectl get pods,svc,ingress -l app.kubernetes.io/part-of=quantis

## Ansible
ansible-ping: ## Test Ansible connectivity to all hosts
	ansible all -i ansible/inventory/hosts.yml -m ping

ansible-deploy: ## Run full Ansible deployment
	ansible-playbook -i ansible/inventory/hosts.yml ansible/playbooks/main.yml --ask-vault-pass
