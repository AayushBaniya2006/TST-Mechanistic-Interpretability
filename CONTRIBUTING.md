# Contributing to TST Interpretability

Thank you for your interest in contributing!

## Setup

```bash
# Clone the repository
git clone https://github.com/AayushBaniya2006/TST-Mechanistic-Interpretability.git
cd TST-Mechanistic-Interpretability

# Install in development mode
pip install -e ".[dev]"

# Verify installation
pytest tests/ -v
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=Utilities --cov-report=html

# Run specific test file
pytest tests/test_statistics.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## Code Style

### General Guidelines
- Follow existing patterns in the codebase
- Use type hints for all public functions
- Include Google-style docstrings

### Docstring Format
```python
def compute_something(data: np.ndarray, threshold: float = 0.5) -> float:
    """
    Brief description of what the function does.

    Args:
        data: Description of the data parameter
        threshold: Description with default value noted

    Returns:
        Description of return value

    Example:
        >>> result = compute_something(np.array([1, 2, 3]))
        >>> print(result)
        2.0
    """
```

### Import Organization
```python
# Standard library
import os
from typing import Optional, Tuple

# Third party
import numpy as np
import torch

# Local
from Utilities.utils import sweep_heads
```

## Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Write** tests for new code
4. **Run** the full test suite (`pytest tests/ -v`)
5. **Commit** with clear messages
6. **Push** to your fork
7. **Submit** a pull request

## Adding New Features

### New Statistical Methods
Add to `Utilities/statistics.py` with:
- Type hints
- Docstring with references
- Unit tests in `tests/test_statistics.py`

### New Perturbation Types
Add to `Utilities/perturbations.py` with:
- Label-preserving validation
- Tests in `tests/test_perturbations.py`

### New Metrics
Add to `Utilities/metrics.py` with:
- Clear mathematical definition in docstring
- Tests in `tests/test_metrics.py`

## Questions?

Open an issue on GitHub with the `question` label.
