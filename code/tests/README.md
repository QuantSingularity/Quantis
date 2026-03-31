# Tests Directory

## Overview

The `tests` directory contains a comprehensive suite of automated tests for the Quantis platform. These tests ensure code quality, verify functionality, prevent regressions during development, and maintain the reliability of the platform across all components. The testing framework is designed to provide thorough coverage of the codebase, from unit tests of individual components to integration tests of the entire system.

## Directory Structure

The tests directory is organized into the following key files:

- **conftest.py**: Configuration file for pytest, defining fixtures and test setup for consistent test environments
- **requirements-test.txt**: Python package dependencies required specifically for running the test suite
- **test_api.py**: Tests for the general API functionality, endpoints, request handling, and response validation
- **test_auth.py**: Tests for authentication and authorization mechanisms, including token handling and security
- **test_forecasting_api.py**: Tests for the forecasting API endpoints, focusing on time series prediction functionality
- **test_forecasting_model.py**: Tests for the forecasting machine learning models, including training and inference
- **test_infrastructure.py**: Tests for infrastructure components, configurations, and deployment processes
- **test_model.py**: Tests for general machine learning model functionality, including loading, saving, and evaluation

## Testing Framework

The Quantis testing infrastructure is built on pytest, a powerful and flexible testing framework for Python. Key components of the testing framework include:

### Test Configuration (conftest.py)

The `conftest.py` file defines shared fixtures and configuration for the test suite, including:

```python
import pytest
from fastapi.testclient import TestClient
from api.app import app
from models.train_model import TemporalFusionTransformer

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def sample_model():
    model = TemporalFusionTransformer(input_size=128)
    return model
```

These fixtures provide:

- A consistent test client for API testing
- Sample model instances for model testing
- Mock data for consistent test inputs
- Environment variable overrides for testing
- Database fixtures with test data

### Test Categories

#### API Tests

Tests in `test_api.py` and `test_forecasting_api.py` verify that API endpoints function correctly, handle various input scenarios, and return appropriate responses. These tests ensure the API contract is maintained and that all endpoints behave as expected.

Key aspects tested include:

- Endpoint availability and response codes
- Request validation and error handling
- Response format and content validation
- Edge cases and boundary conditions
- Performance under various loads

Example API test:

```python
def test_predict_endpoint_success(test_client, sample_data):
    response = test_client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_endpoint_invalid_data(test_client):
    response = test_client.post(
        "/predict", json={"features": [0.1], "api_key": "test_key"}
    )
    assert response.status_code == 422  # Validation error
```

#### Authentication Tests

Tests in `test_auth.py` verify user authentication, authorization, token handling, and security mechanisms to ensure proper access control throughout the platform.

These tests cover:

- User authentication flows
- Token generation and validation
- Permission checking and role-based access
- Security measures against common attacks
- Rate limiting and throttling

Example authentication test:

```python
@pytest.fixture
def mock_env_api_key(monkeypatch):
    monkeypatch.setenv("API_SECRET", "test_key")
    monkeypatch.setenv("JWT_SECRET", "test_jwt_secret")

def test_validate_api_key_success(mock_env_api_key):
    result = validate_api_key("test_key")
    assert result is True

def test_validate_api_key_failure(mock_env_api_key):
    result = validate_api_key("invalid_key")
    assert result is False
```

#### Model Tests

Tests in `test_model.py` and `test_forecasting_model.py` validate machine learning model behavior, including training, inference, and performance metrics. These tests ensure models meet accuracy and reliability requirements.

These tests cover:

- Model initialization with different parameters
- Forward and backward passes
- Training procedures
- Inference with various inputs
- Model saving and loading
- Performance metrics calculation

Example model test:

```python
def test_model_with_different_input_sizes():
    # Test with minimum input size
    min_model = TemporalFusionTransformer(input_size=64)
    min_input = torch.randn(1, 64)
    min_output = min_model(min_input)
    assert min_output.shape == (1, 1)

    # Test with maximum input size
    max_model = TemporalFusionTransformer(input_size=256)
    max_input = torch.randn(1, 256)
    max_output = max_model(max_input)
    assert max_output.shape == (1, 1)
```

#### Infrastructure Tests

Tests in `test_infrastructure.py` verify that infrastructure components are correctly configured and functioning as expected, ensuring reliable deployment and operation.

These tests cover:

- Deployment procedures
- Configuration validation
- Resource provisioning
- Service discovery
- Monitoring integration

Example infrastructure test:

```python
@pytest.fixture
def mock_s3():
    with patch("boto3.client") as mock:
        mock_s3 = Mock()
        mock.return_value = mock_s3
        yield mock_s3

def test_model_deployment_to_s3(mock_s3):
    model_path = "models/test_model.pt"
    bucket = "quantis-models"
    key = "production/test_model.pt"

    deploy_to_aws(model_path, bucket, key)

    mock_s3.upload_file.assert_called_once_with(
        model_path, bucket, key
    )
```

## Test Coverage

The test suite aims for comprehensive coverage of the codebase, with specific targets for different components:

- **Core API**: 95% code coverage
- **Authentication**: 100% code coverage
- **ML Models**: 90% code coverage
- **Infrastructure**: 85% code coverage
- **Utilities**: 80% code coverage

Coverage is measured using pytest-cov and reported during CI/CD pipeline execution. Coverage reports are generated in HTML and XML formats for easy visualization and integration with code quality tools.

## Running Tests

### Prerequisites

Before running tests, ensure all test dependencies are installed:

```bash
pip install -r requirements-test.txt
```

The requirements file includes:

- pytest and pytest plugins
- Testing utilities like mock
- Coverage reporting tools
- Test-specific dependencies

### Running All Tests

To run the complete test suite:

```bash
pytest
```

This will discover and execute all test files in the directory.

### Running Specific Test Files

To run tests from a specific file:

```bash
pytest test_api.py
```

### Running Tests by Category

To run tests by category or marker:

```bash
pytest -m "api"  # Run all API tests
pytest -m "model"  # Run all model tests
```

### Running with Coverage Report

To generate a coverage report:

```bash
pytest --cov=quantis
```

For a detailed HTML coverage report:

```bash
pytest --cov=quantis --cov-report=html
```

### Test Output and Reporting

Test results are displayed in the console with detailed information about passing and failing tests. For CI/CD integration, JUnit XML reports can be generated:

```bash
pytest --junitxml=test-results.xml
```

## Continuous Integration

The test suite is integrated into the CI/CD pipeline defined in the `.github` directory. Tests are automatically run on:

- Pull request creation
- Commits to development branches
- Release preparation

The CI pipeline performs the following steps:

1. Set up the test environment
2. Install dependencies
3. Run the test suite
4. Generate coverage reports
5. Validate code quality
6. Report results

Tests must pass before code can be merged into the main branches, ensuring code quality and preventing regressions.

## Test Development Guidelines

When adding new features or fixing bugs, follow these guidelines for test development:

### Test Structure

1. **Arrange**: Set up the test environment and inputs
2. **Act**: Execute the code being tested
3. **Assert**: Verify the results match expectations

Example:

```python
def test_feature_calculation():
    # Arrange
    data = [1, 2, 3, 4, 5]

    # Act
    result = calculate_feature(data)

    # Assert
    assert result == 15
```

### Test Isolation

Ensure tests are isolated and do not depend on:

- External services (use mocks instead)
- Other test results
- Specific environment configurations
- Global state

Use fixtures defined in `conftest.py` for common test setup to maintain consistency.

### Mocking

Use mocking to isolate the code being tested from external dependencies:

```python
@patch('requests.get')
def test_external_api_call(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'data': 'test'}
    mock_get.return_value = mock_response

    result = call_external_api()

    assert result == {'data': 'test'}
    mock_get.assert_called_once_with('https://api.example.com/data')
```

### Parameterized Tests

Use parameterized tests for testing multiple input combinations:

```python
@pytest.mark.parametrize("input_value,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square_function(input_value, expected):
    assert square(input_value) == expected
```

### Error Case Testing

Always include tests for error cases and edge conditions:

```python
def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)
```

## Test Data Management

The test suite uses several approaches for test data management:

### Fixtures

Fixtures in `conftest.py` provide consistent test data:

```python
@pytest.fixture
def sample_time_series_data():
    return [
        {"timestamp": "2023-01-01T00:00:00Z", "value": 10.5},
        {"timestamp": "2023-01-02T00:00:00Z", "value": 11.2},
        {"timestamp": "2023-01-03T00:00:00Z", "value": 10.8}
    ]
```

### Factory Functions

Factory functions generate test data with specific characteristics:

```python
def create_test_user(username="testuser", is_admin=False):
    return {
        "username": username,
        "email": f"{username}@example.com",
        "is_admin": is_admin
    }
```

### Test Databases

For database-dependent tests, the suite uses:

- In-memory SQLite databases for unit tests
- Test-specific PostgreSQL databases for integration tests
- Database fixtures with predefined test data

## Performance Testing

The test suite includes performance tests to ensure the system meets performance requirements:

### Response Time Tests

```python
def test_api_response_time():
    start_time = time.time()
    response = test_client.get("/api/data")
    end_time = time.time()

    assert response.status_code == 200
    assert end_time - start_time < 0.1  # Response time under 100ms
```

### Load Tests

Load tests verify system behavior under various load conditions:

```python
def test_concurrent_requests():
    results = []

    def make_request():
        response = test_client.get("/api/data")
        results.append(response.status_code)

    threads = []
    for _ in range(100):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    assert all(status == 200 for status in results)
```

## Troubleshooting Tests

Common issues and solutions when working with the test suite:

### Test Failures

1. **Inconsistent Test Results**:
   - Check for test isolation issues
   - Verify fixture cleanup
   - Look for shared state between tests

2. **Mock Configuration Issues**:
   - Ensure mocks are correctly configured
   - Verify mock return values
   - Check mock assertion parameters

3. **Environment Dependencies**:
   - Verify environment variables
   - Check for external service dependencies
   - Ensure consistent test environment

### Debugging Techniques

1. **Verbose Output**:

   ```bash
   pytest -v
   ```

2. **Print Debugging**:

   ```python
   def test_complex_function():
       result = complex_function()
       print(f"Debug: result = {result}")
       assert result == expected_value
   ```

3. **Interactive Debugging**:

   ```bash
   pytest --pdb
   ```

4. **Trace Function Calls**:
   ```bash
   pytest --trace
   ```

## Integration with Development Workflow

The test suite is integrated into the development workflow:

### Pre-commit Hooks

Pre-commit hooks run tests before allowing commits:

```bash
# .git/hooks/pre-commit
#!/bin/sh
pytest test_api.py test_auth.py
```

### Code Review Requirements

Pull requests must include:

- Tests for new functionality
- Tests for fixed bugs
- Passing CI build with tests

### Test-Driven Development

The development process encourages TDD:

1. Write failing tests for new functionality
2. Implement the functionality
3. Verify tests pass
4. Refactor while maintaining passing tests

## Future Test Improvements

Planned improvements to the test suite include:

1. **Property-Based Testing**:
   - Implementing hypothesis for property-based tests
   - Generating random inputs to find edge cases

2. **Mutation Testing**:
   - Using mutation testing to evaluate test quality
   - Identifying untested code paths

3. **Visual Regression Testing**:
   - Adding screenshot comparison for UI tests
   - Automating visual verification

4. **Contract Testing**:
   - Implementing consumer-driven contract tests
   - Ensuring API compatibility between services

5. **Chaos Testing**:
   - Introducing controlled failures
   - Testing system resilience

## Contributing to Tests

When contributing to the test suite:

1. **Follow Existing Patterns**:
   - Maintain consistent test structure
   - Use existing fixtures where appropriate
   - Follow naming conventions

2. **Document Test Purpose**:
   - Include docstrings explaining test purpose
   - Document complex test setups
   - Explain non-obvious assertions

3. **Maintain Test Quality**:
   - Ensure tests are deterministic
   - Avoid flaky tests
   - Keep tests fast and focused

4. **Review Test Coverage**:
   - Identify uncovered code paths
   - Add tests for edge cases
   - Ensure comprehensive coverage

This comprehensive test suite ensures the reliability, performance, and correctness of the Quantis platform across all components and use cases.
