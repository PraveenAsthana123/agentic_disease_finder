# Contributing to AgenticFinder

Thank you for your interest in contributing to AgenticFinder! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.8+
- pip or conda
- Git

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/praveenairesearch/agenticfinder.git
cd agenticfinder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black isort flake8 mypy
```

## Code Style

We follow PEP 8 style guidelines with some modifications:

- Line length: 100 characters
- Use type hints for all function signatures
- Use docstrings (Google style) for all public functions/classes

### Formatting

```bash
# Format code
black --line-length 100 .

# Sort imports
isort .

# Check style
flake8 --max-line-length 100 .

# Type checking
mypy scripts/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=html

# Run specific test file
pytest tests/test_feature_extraction.py -v

# Run specific test
pytest tests/test_model.py::TestUltraStackingEnsemble::test_training -v
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Name test classes as `Test*`
- Name test methods as `test_*`
- Use fixtures from `conftest.py` when possible

Example:
```python
import pytest
import numpy as np

class TestMyFeature:
    @pytest.fixture
    def sample_data(self):
        return np.random.randn(100, 47)

    def test_basic_functionality(self, sample_data):
        # Test implementation
        assert sample_data.shape == (100, 47)
```

## Project Structure

```
agenticfinder/
├── scripts/                 # Main scripts
│   ├── train.py            # Training pipeline
│   ├── evaluate.py         # Evaluation metrics
│   ├── predict.py          # Inference
│   ├── generate_figures.py # Figure generation
│   └── generate_sample_data.py  # Data generation
├── responsible_ai/          # RAI framework modules
├── tests/                   # Unit tests
├── paper/                   # Journal paper and figures
├── models/                  # Saved models
└── data/                    # Datasets
```

## Adding New Features

### Adding a New Disease

1. Add disease profile in `scripts/generate_sample_data.py`:
```python
DISEASE_PROFILES['new_disease'] = {
    'delta': 0.20, 'theta': 0.25, ...
}
```

2. Update performance data in `scripts/generate_figures.py`

3. Add tests for the new disease

4. Update README.md with new disease information

### Adding New Features to Extraction

1. Modify `EEGFeatureExtractor` in `scripts/train.py`
2. Update feature count (currently 47)
3. Add corresponding tests
4. Update documentation

### Adding New Classifiers

1. Modify `UltraStackingEnsemble` in `scripts/train.py`
2. Add to `_create_base_estimators()` method
3. Test with synthetic data
4. Benchmark performance

## Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Make** your changes
4. **Test** your changes: `pytest tests/ -v`
5. **Format** code: `black . && isort .`
6. **Commit** with clear message: `git commit -m "Add feature X"`
7. **Push** to your fork: `git push origin feature/my-feature`
8. **Open** a Pull Request

### PR Requirements

- [ ] All tests pass
- [ ] Code is formatted (black, isort)
- [ ] Type hints added for new functions
- [ ] Docstrings added for public functions
- [ ] README updated if needed
- [ ] CHANGELOG updated

## Reporting Issues

When reporting issues, please include:

1. **Description** of the problem
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Environment** (OS, Python version, package versions)
6. **Error messages** (full traceback)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
