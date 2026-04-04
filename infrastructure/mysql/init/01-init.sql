-- Create MLflow database
CREATE DATABASE IF NOT EXISTS mlflow CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Grant app user access to mlflow db
GRANT ALL PRIVILEGES ON mlflow.* TO 'appuser'@'%';

-- Ensure quantisdb exists with correct charset
ALTER DATABASE quantisdb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

FLUSH PRIVILEGES;
