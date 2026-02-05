"""
Pytest fixtures and configuration for TST Mechanistic Interpretability tests.

Provides:
- Small model fixtures for fast testing
- Sample data fixtures
- Device configuration
- Seed management for reproducibility
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# SEED MANAGEMENT
# =============================================================================

@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds before each test for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

@pytest.fixture
def device():
    """Get available device (CPU for CI, GPU if available locally)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# MODEL FIXTURES
# =============================================================================

@pytest.fixture
def small_model(device):
    """
    Create a small TimeSeriesTransformer for testing.

    Uses minimal dimensions for fast testing:
    - 2 layers, 4 heads
    - d_model=32, d_ff=64
    - input_dim=4, num_classes=3, seq_len=10
    """
    from Utilities.TST_trainer import TimeSeriesTransformer

    model = TimeSeriesTransformer(
        input_dim=4,
        num_classes=3,
        seq_len=10,
        d_model=32,
        n_head=4,
        num_encoder_layers=2,
        dim_feedforward=64,
        dropout=0.0  # No dropout for deterministic testing
    )
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def tiny_model(device):
    """
    Even smaller model for very fast tests.

    - 1 layer, 2 heads
    - d_model=16, d_ff=32
    """
    from Utilities.TST_trainer import TimeSeriesTransformer

    model = TimeSeriesTransformer(
        input_dim=2,
        num_classes=2,
        seq_len=5,
        d_model=16,
        n_head=2,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.0
    )
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def standard_model(device):
    """
    Standard-sized model matching paper configuration.

    - 3 layers, 8 heads
    - d_model=128, d_ff=256
    """
    from Utilities.TST_trainer import TimeSeriesTransformer

    model = TimeSeriesTransformer(
        input_dim=12,
        num_classes=9,
        seq_len=29,
        d_model=128,
        n_head=8,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.0
    )
    model = model.to(device)
    model.eval()
    return model


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_input(device):
    """Generate a random sample input tensor."""
    return torch.randn(1, 10, 4, device=device)


@pytest.fixture
def sample_pair(device):
    """Generate a clean-corrupt sample pair."""
    clean = torch.randn(1, 10, 4, device=device)
    corrupt = torch.randn(1, 10, 4, device=device)
    return clean, corrupt


@pytest.fixture
def batch_input(device):
    """Generate a batch of sample inputs."""
    return torch.randn(8, 10, 4, device=device)


@pytest.fixture
def tiny_input(device):
    """Input matching tiny_model dimensions."""
    return torch.randn(1, 5, 2, device=device)


@pytest.fixture
def tiny_pair(device):
    """Sample pair for tiny model."""
    clean = torch.randn(1, 5, 2, device=device)
    corrupt = torch.randn(1, 5, 2, device=device)
    return clean, corrupt


@pytest.fixture
def standard_input(device):
    """Input matching standard_model (JapaneseVowels) dimensions."""
    return torch.randn(1, 29, 12, device=device)


@pytest.fixture
def standard_pair(device):
    """Sample pair for standard model."""
    clean = torch.randn(1, 29, 12, device=device)
    corrupt = torch.randn(1, 29, 12, device=device)
    return clean, corrupt


# =============================================================================
# PROBABILITY FIXTURES
# =============================================================================

@pytest.fixture
def sample_probs():
    """Generate sample probability distributions."""
    probs = torch.softmax(torch.randn(1, 5), dim=-1)
    return probs


@pytest.fixture
def prob_pair():
    """Generate a pair of probability distributions."""
    probs_a = torch.softmax(torch.randn(1, 5), dim=-1)
    probs_b = torch.softmax(torch.randn(1, 5), dim=-1)
    return probs_a, probs_b


# =============================================================================
# NUMPY FIXTURES
# =============================================================================

@pytest.fixture
def importance_matrix():
    """Generate a sample head importance matrix."""
    return np.random.rand(3, 8)  # 3 layers, 8 heads


@pytest.fixture
def importance_pair():
    """Generate a pair of importance matrices for comparison."""
    baseline = np.random.rand(3, 8)
    perturbed = baseline + np.random.randn(3, 8) * 0.1
    return baseline, perturbed


# =============================================================================
# STATISTICAL DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample data for statistical tests."""
    return np.random.randn(100)


@pytest.fixture
def paired_samples():
    """Generate paired samples for statistical tests."""
    treatment = np.random.randn(50)
    control = treatment + np.random.randn(50) * 0.5 + 0.3  # Small effect
    return treatment, control


@pytest.fixture
def effect_size_samples():
    """Generate samples with known effect size."""
    # Cohen's d ≈ 0.5 (medium effect)
    control = np.random.randn(100)
    treatment = np.random.randn(100) + 0.5
    return treatment, control


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Get default experiment configuration."""
    from Utilities.config import ExperimentConfig
    return ExperimentConfig(
        name="test_experiment",
        seed=42,
        device="cpu"
    )


@pytest.fixture
def quick_config():
    """Get quick test configuration."""
    from Utilities.config import get_quick_test_config
    return get_quick_test_config()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_tensor_close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8):
    """Assert two tensors are close."""
    assert torch.allclose(a, b, rtol=rtol, atol=atol), \
        f"Tensors not close: max diff = {(a - b).abs().max().item()}"


def assert_valid_probs(probs: torch.Tensor, dim: int = -1):
    """Assert tensor represents valid probabilities."""
    assert torch.all(probs >= 0), "Probabilities must be non-negative"
    assert torch.all(probs <= 1), "Probabilities must be <= 1"
    sums = probs.sum(dim=dim)
    assert torch.allclose(sums, torch.ones_like(sums)), "Probabilities must sum to 1"


def assert_array_close(a: np.ndarray, b: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8):
    """Assert two arrays are close."""
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
