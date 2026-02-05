"""
Tests for Utilities/config.py

Tests configuration management including:
- Seed management
- Configuration dataclasses
- YAML serialization
- Experiment metadata
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path


class TestSetAllSeeds:
    """Tests for seed management."""

    def test_sets_torch_seed(self):
        """set_all_seeds should make torch reproducible."""
        from Utilities.config import set_all_seeds

        set_all_seeds(42)
        a = torch.randn(10)

        set_all_seeds(42)
        b = torch.randn(10)

        torch.testing.assert_close(a, b)

    def test_sets_numpy_seed(self):
        """set_all_seeds should make numpy reproducible."""
        from Utilities.config import set_all_seeds

        set_all_seeds(42)
        a = np.random.randn(10)

        set_all_seeds(42)
        b = np.random.randn(10)

        np.testing.assert_array_equal(a, b)

    def test_sets_python_seed(self):
        """set_all_seeds should make random module reproducible."""
        from Utilities.config import set_all_seeds
        import random

        set_all_seeds(42)
        a = [random.random() for _ in range(10)]

        set_all_seeds(42)
        b = [random.random() for _ in range(10)]

        assert a == b

    def test_different_seeds_different_results(self):
        """Different seeds should produce different results."""
        from Utilities.config import set_all_seeds

        set_all_seeds(42)
        a = torch.randn(10)

        set_all_seeds(123)
        b = torch.randn(10)

        assert not torch.allclose(a, b)


class TestSeedState:
    """Tests for seed state save/restore."""

    def test_save_restore_cycle(self):
        """Should be able to save and restore random state."""
        from Utilities.config import set_all_seeds, get_seed_state, set_seed_state

        set_all_seeds(42)

        # Generate some random numbers
        _ = torch.randn(10)
        _ = np.random.randn(10)

        # Save state
        state = get_seed_state()

        # Generate more numbers
        a1 = torch.randn(10)
        a2 = np.random.randn(10)

        # Restore state
        set_seed_state(state)

        # Should get same numbers
        b1 = torch.randn(10)
        b2 = np.random.randn(10)

        torch.testing.assert_close(a1, b1)
        np.testing.assert_array_equal(a2, b2)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """ModelConfig should have sensible defaults."""
        from Utilities.config import ModelConfig

        config = ModelConfig()

        assert config.d_model == 128
        assert config.n_head == 8
        assert config.num_encoder_layers == 3
        assert config.dim_feedforward == 256
        assert 0 <= config.dropout <= 1

    def test_validation_d_model_divisible(self):
        """d_model must be divisible by n_head."""
        from Utilities.config import ModelConfig

        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(d_model=128, n_head=7)

    def test_validation_dropout_range(self):
        """dropout must be in [0, 1]."""
        from Utilities.config import ModelConfig

        with pytest.raises(ValueError, match="dropout"):
            ModelConfig(dropout=1.5)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """TrainingConfig should have sensible defaults."""
        from Utilities.config import TrainingConfig

        config = TrainingConfig()

        assert config.epochs > 0
        assert config.batch_size > 0
        assert config.lr > 0
        assert config.n_training_runs >= 1

    def test_validation_epochs(self):
        """epochs must be >= 1."""
        from Utilities.config import TrainingConfig

        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(epochs=0)

    def test_validation_lr(self):
        """lr must be > 0."""
        from Utilities.config import TrainingConfig

        with pytest.raises(ValueError, match="lr"):
            TrainingConfig(lr=-0.001)


class TestPatchingConfig:
    """Tests for PatchingConfig dataclass."""

    def test_default_values(self):
        """PatchingConfig should have sensible defaults."""
        from Utilities.config import PatchingConfig

        config = PatchingConfig()

        assert config.threshold > 0
        assert config.max_pairs_per_class > 0
        assert len(config.metrics) > 0

    def test_validation_invalid_metric(self):
        """Invalid metric should raise error."""
        from Utilities.config import PatchingConfig

        with pytest.raises(ValueError, match="Invalid metric"):
            PatchingConfig(metrics=("invalid_metric",))


class TestStatisticsConfig:
    """Tests for StatisticsConfig dataclass."""

    def test_default_values(self):
        """StatisticsConfig should have sensible defaults."""
        from Utilities.config import StatisticsConfig

        config = StatisticsConfig()

        assert 0 < config.confidence_level < 1
        assert config.bootstrap_iterations > 0
        assert config.alpha > 0

    def test_validation_confidence_level(self):
        """confidence_level must be in (0, 1)."""
        from Utilities.config import StatisticsConfig

        with pytest.raises(ValueError, match="confidence_level"):
            StatisticsConfig(confidence_level=1.5)

    def test_validation_fdr_method(self):
        """fdr_method must be valid."""
        from Utilities.config import StatisticsConfig

        with pytest.raises(ValueError, match="fdr_method"):
            StatisticsConfig(fdr_method="invalid_method")


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_creation(self):
        """ExperimentConfig should create with defaults."""
        from Utilities.config import ExperimentConfig

        config = ExperimentConfig()

        assert config.name == "tst_patching_experiment"
        assert config.seed == 42
        assert len(config.datasets) > 0

    def test_nested_configs(self):
        """Should properly create nested configs."""
        from Utilities.config import ExperimentConfig, ModelConfig

        config = ExperimentConfig()

        assert isinstance(config.model, ModelConfig)
        assert config.model.d_model == 128

    def test_from_dict(self):
        """Should create from dictionary."""
        from Utilities.config import ExperimentConfig

        data = {
            'name': 'test_exp',
            'seed': 123,
            'model': {'d_model': 64, 'n_head': 4}
        }

        config = ExperimentConfig.from_dict(data)

        assert config.name == 'test_exp'
        assert config.seed == 123
        assert config.model.d_model == 64

    def test_to_dict(self):
        """Should convert to dictionary."""
        from Utilities.config import ExperimentConfig

        config = ExperimentConfig(name='test', seed=99)
        data = config.to_dict()

        assert isinstance(data, dict)
        assert data['name'] == 'test'
        assert data['seed'] == 99
        assert isinstance(data['model'], dict)

    def test_yaml_round_trip(self):
        """Should save and load from YAML."""
        from Utilities.config import ExperimentConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"

            original = ExperimentConfig(name='yaml_test', seed=777)
            original.save(str(path))

            loaded = ExperimentConfig.from_yaml(str(path))

            assert loaded.name == 'yaml_test'
            assert loaded.seed == 777
            assert loaded.model.d_model == original.model.d_model

    def test_get_hash(self):
        """Hash should be deterministic."""
        from Utilities.config import ExperimentConfig

        config1 = ExperimentConfig(name='test', seed=42)
        config2 = ExperimentConfig(name='test', seed=42)
        config3 = ExperimentConfig(name='test', seed=43)

        assert config1.get_hash() == config2.get_hash()
        assert config1.get_hash() != config3.get_hash()

    def test_get_run_id(self):
        """Run ID should contain name and hash."""
        from Utilities.config import ExperimentConfig

        config = ExperimentConfig(name='my_experiment')
        run_id = config.get_run_id()

        assert 'my_experiment' in run_id
        assert config.get_hash()[:8] in run_id


class TestDatasetConfigs:
    """Tests for dataset configuration utilities."""

    def test_known_datasets(self):
        """Should return config for known datasets."""
        from Utilities.config import get_dataset_config

        config = get_dataset_config('JapaneseVowels')

        assert config['input_dim'] == 12
        assert config['num_classes'] == 9

    def test_unknown_dataset_raises(self):
        """Unknown dataset should raise ValueError."""
        from Utilities.config import get_dataset_config

        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_config('NonExistentDataset')

    def test_returns_copy(self):
        """Should return copy to prevent mutation."""
        from Utilities.config import get_dataset_config

        config1 = get_dataset_config('JapaneseVowels')
        config2 = get_dataset_config('JapaneseVowels')

        config1['input_dim'] = 999

        assert config2['input_dim'] == 12


class TestPresetConfigs:
    """Tests for preset configuration functions."""

    def test_quick_test_config(self):
        """Quick test config should have reduced values."""
        from Utilities.config import get_quick_test_config

        config = get_quick_test_config()

        assert config.training.epochs <= 10
        assert config.training.n_training_runs == 1

    def test_full_experiment_config(self):
        """Full experiment config should have comprehensive values."""
        from Utilities.config import get_full_experiment_config

        config = get_full_experiment_config()

        assert config.training.epochs >= 100
        assert config.training.n_training_runs >= 5
        assert len(config.datasets) >= 3


class TestSystemInfo:
    """Tests for system info collection."""

    def test_get_system_info(self):
        """Should return system information dict."""
        from Utilities.config import get_system_info

        info = get_system_info()

        assert 'python_version' in info
        assert 'pytorch_version' in info
        assert 'cuda_available' in info
        assert 'timestamp' in info


class TestExperimentMetadata:
    """Tests for experiment metadata saving."""

    def test_saves_metadata(self):
        """Should save metadata to file."""
        from Utilities.config import ExperimentConfig, save_experiment_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(name='metadata_test')

            path = save_experiment_metadata(config, tmpdir)

            assert Path(path).exists()

    def test_metadata_contains_config(self):
        """Saved metadata should contain config."""
        from Utilities.config import ExperimentConfig, save_experiment_metadata
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(name='metadata_test', seed=999)

            path = save_experiment_metadata(config, tmpdir)

            with open(path) as f:
                data = json.load(f)

            assert data['config']['name'] == 'metadata_test'
            assert data['config']['seed'] == 999
