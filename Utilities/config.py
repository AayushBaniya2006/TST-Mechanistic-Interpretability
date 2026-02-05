"""
Configuration management for reproducible experiments.

This module provides:
- Dataclass-based configuration with validation
- Seed management for full reproducibility
- YAML loading/saving support
- Experiment metadata tracking

Usage:
    >>> config = ExperimentConfig.from_yaml("configs/experiment.yaml")
    >>> set_all_seeds(config.seed)
    >>> # Run experiments...
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
import hashlib
from datetime import datetime
import platform
import os

import yaml
import torch
import numpy as np
import random


# =============================================================================
# SEED MANAGEMENT
# =============================================================================

def set_all_seeds(seed: int) -> None:
    """
    Set all random seeds for complete reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDA deterministic mode

    Args:
        seed: Random seed value

    Example:
        >>> set_all_seeds(42)
        >>> # All subsequent random operations are reproducible
    """
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For some operations (slower but reproducible)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_seed_state() -> Dict[str, Any]:
    """
    Get current random state for saving/restoring.

    Returns:
        Dict with random states for all libraries
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()

    return state


def set_seed_state(state: Dict[str, Any]) -> None:
    """
    Restore random state from saved state.

    Args:
        state: State dict from get_seed_state()
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    d_model: int = 128
    n_head: int = 8
    num_encoder_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1

    def __post_init__(self):
        # Validation
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_training_runs: int = 5  # Multiple runs for variance estimation
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")


@dataclass
class PatchingConfig:
    """Activation patching configuration."""
    threshold: float = 0.01
    max_pairs_per_class: int = 20
    metrics: Tuple[str, ...] = ("delta_p", "logit_diff", "kl_divergence")
    directions: Tuple[str, ...] = ("denoise", "noise")
    progressive_k_values: Tuple[int, ...] = (1, 2, 3, 5, 10)

    def __post_init__(self):
        valid_metrics = {"delta_p", "logit_diff", "kl_divergence", "cross_entropy"}
        for m in self.metrics:
            if m not in valid_metrics:
                raise ValueError(f"Invalid metric: {m}. Must be one of {valid_metrics}")


@dataclass
class PerturbationConfig:
    """Perturbation settings for stability testing."""
    gaussian_sigmas: Tuple[float, ...] = (0.05, 0.10, 0.20)
    time_warp_factors: Tuple[float, ...] = (0.05, 0.10, 0.20)
    phase_shift_max: Tuple[int, ...] = (1, 2, 3)
    n_perturbation_samples: int = 10
    max_accuracy_drop: float = 0.05


@dataclass
class StatisticsConfig:
    """Statistical analysis settings."""
    confidence_level: float = 0.95
    bootstrap_iterations: int = 10000
    fdr_method: str = "benjamini_hochberg"
    alpha: float = 0.05

    def __post_init__(self):
        valid_methods = {"benjamini_hochberg", "bonferroni", "holm"}
        if self.fdr_method not in valid_methods:
            raise ValueError(f"Invalid fdr_method: {self.fdr_method}. Must be one of {valid_methods}")
        if not 0 < self.confidence_level < 1:
            raise ValueError(f"confidence_level must be in (0, 1), got {self.confidence_level}")


@dataclass
class ExperimentConfig:
    """
    Master configuration for experiments.

    Combines all sub-configurations into a single validated config.
    Supports YAML serialization and hash-based integrity checking.

    Example:
        >>> config = ExperimentConfig(name="my_experiment", seed=42)
        >>> config.save("configs/my_experiment.yaml")
        >>>
        >>> # Later...
        >>> config = ExperimentConfig.from_yaml("configs/my_experiment.yaml")
        >>> set_all_seeds(config.seed)
    """
    # Experiment metadata
    name: str = "tst_patching_experiment"
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "Results"

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)

    # Datasets to run (empty = use default 3)
    datasets: List[str] = field(default_factory=lambda: [
        "JapaneseVowels",
        "PenDigits",
        "LSST"
    ])

    def __post_init__(self):
        # Ensure device is valid
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Ensure sub-configs are proper types
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.patching, dict):
            self.patching = PatchingConfig(**self.patching)
        if isinstance(self.perturbation, dict):
            self.perturbation = PerturbationConfig(**self.perturbation)
        if isinstance(self.statistics, dict):
            self.statistics = StatisticsConfig(**self.statistics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary."""
        def convert_tuples(obj):
            """Recursively convert tuples to lists for YAML compatibility."""
            if isinstance(obj, dict):
                return {k: convert_tuples(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_tuples(item) for item in obj]
            return obj

        return convert_tuples({
            'name': self.name,
            'seed': self.seed,
            'device': self.device,
            'output_dir': self.output_dir,
            'datasets': self.datasets,
            'model': asdict(self.model),
            'training': asdict(self.training),
            'patching': asdict(self.patching),
            'perturbation': asdict(self.perturbation),
            'statistics': asdict(self.statistics),
        })

    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**data)

    def get_hash(self) -> str:
        """
        Compute deterministic hash of configuration.

        Useful for detecting configuration changes between runs.
        """
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_run_id(self) -> str:
        """Generate unique run ID based on config hash and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.name}_{timestamp}_{self.get_hash()[:8]}"


# =============================================================================
# EXPERIMENT METADATA
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for reproducibility documentation.

    Returns:
        Dict with Python version, PyTorch version, CUDA info, etc.
    """
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()

    return info


def save_experiment_metadata(
    config: ExperimentConfig,
    output_dir: str,
    extra_info: Optional[Dict] = None
) -> str:
    """
    Save complete experiment metadata for reproducibility.

    Args:
        config: Experiment configuration
        output_dir: Directory to save metadata
        extra_info: Optional additional information

    Returns:
        Path to saved metadata file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        'config': config.to_dict(),
        'config_hash': config.get_hash(),
        'system': get_system_info(),
    }

    if extra_info:
        metadata['extra'] = extra_info

    path = output_dir / f"experiment_metadata_{config.get_run_id()}.json"

    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    return str(path)


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Current datasets
    'JapaneseVowels': {
        'input_dim': 12,
        'num_classes': 9,
        'seq_len': 29,
        'domain': 'audio',
        'difficulty': 'easy'
    },
    'PenDigits': {
        'input_dim': 2,
        'num_classes': 10,
        'seq_len': 8,
        'domain': 'gesture',
        'difficulty': 'easy'
    },
    'LSST': {
        'input_dim': 6,
        'num_classes': 14,
        'seq_len': 36,
        'domain': 'astronomy',
        'difficulty': 'hard'
    },
    # Future datasets (for expansion)
    'ECG200': {
        'input_dim': 1,
        'num_classes': 2,
        'seq_len': 96,
        'domain': 'medical',
        'difficulty': 'medium'
    },
    'FordA': {
        'input_dim': 1,
        'num_classes': 2,
        'seq_len': 500,
        'domain': 'industrial',
        'difficulty': 'medium'
    },
    'Epilepsy': {
        'input_dim': 3,
        'num_classes': 4,
        'seq_len': 206,
        'domain': 'medical',
        'difficulty': 'medium'
    },
    'UWaveGestureLibrary': {
        'input_dim': 3,
        'num_classes': 8,
        'seq_len': 315,
        'domain': 'gesture',
        'difficulty': 'medium'
    },
}


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get configuration for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dict with input_dim, num_classes, seq_len, domain, difficulty
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name].copy()


# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


def get_quick_test_config() -> ExperimentConfig:
    """Get minimal configuration for quick testing."""
    return ExperimentConfig(
        name="quick_test",
        training=TrainingConfig(epochs=5, n_training_runs=1),
        patching=PatchingConfig(max_pairs_per_class=2),
        statistics=StatisticsConfig(bootstrap_iterations=100),
        datasets=["JapaneseVowels"]
    )


def get_full_experiment_config() -> ExperimentConfig:
    """Get comprehensive configuration for full experiments."""
    return ExperimentConfig(
        name="full_experiment",
        training=TrainingConfig(epochs=100, n_training_runs=5),
        patching=PatchingConfig(max_pairs_per_class=20),
        statistics=StatisticsConfig(bootstrap_iterations=10000),
        datasets=[
            "JapaneseVowels",
            "PenDigits",
            "LSST",
            "ECG200",
            "FordA",
            "Epilepsy",
            "UWaveGestureLibrary"
        ]
    )
