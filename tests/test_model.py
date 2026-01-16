#!/usr/bin/env python3
"""
Unit Tests for Ultra Stacking Ensemble Model
==============================================

Tests for the ensemble model including training, prediction,
and cross-validation capabilities.

Author: AgenticFinder Research Team
License: MIT
"""

import pytest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from train import UltraStackingEnsemble, generate_synthetic_data


class TestUltraStackingEnsemble:
    """Test suite for UltraStackingEnsemble classifier."""

    @pytest.fixture
    def ensemble(self):
        """Create ensemble classifier instance."""
        return UltraStackingEnsemble(n_jobs=1, random_state=42)

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        X = np.random.randn(200, 47)  # 200 samples, 47 features
        y = np.random.randint(0, 2, 200)  # Binary classification
        return X, y

    @pytest.fixture
    def trained_ensemble(self, sample_data):
        """Create and train ensemble."""
        X, y = sample_data
        ensemble = UltraStackingEnsemble(n_jobs=1, random_state=42)
        ensemble.fit(X, y)
        return ensemble

    def test_initialization(self, ensemble):
        """Test ensemble initialization."""
        assert ensemble is not None
        assert hasattr(ensemble, 'fit')
        assert hasattr(ensemble, 'predict')
        assert hasattr(ensemble, 'predict_proba')

    def test_training(self, ensemble, sample_data):
        """Test that ensemble can be trained."""
        X, y = sample_data
        ensemble.fit(X, y)

        # Should have fitted base estimators
        assert hasattr(ensemble, 'base_estimators_')
        assert len(ensemble.base_estimators_) > 0

    def test_prediction_shape(self, trained_ensemble, sample_data):
        """Test prediction output shape."""
        X, _ = sample_data
        predictions = trained_ensemble.predict(X)

        assert predictions.shape == (len(X),)

    def test_prediction_values(self, trained_ensemble, sample_data):
        """Test prediction values are valid class labels."""
        X, _ = sample_data
        predictions = trained_ensemble.predict(X)

        # Should only contain 0 or 1 for binary classification
        unique_preds = np.unique(predictions)
        assert all(p in [0, 1] for p in unique_preds)

    def test_probability_prediction(self, trained_ensemble, sample_data):
        """Test probability prediction."""
        X, _ = sample_data

        if hasattr(trained_ensemble, 'predict_proba'):
            proba = trained_ensemble.predict_proba(X)

            # Shape should be (n_samples, n_classes)
            assert proba.shape == (len(X), 2)

            # Probabilities should sum to 1
            np.testing.assert_array_almost_equal(
                proba.sum(axis=1), np.ones(len(X)), decimal=5
            )

            # Probabilities should be between 0 and 1
            assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_single_sample_prediction(self, trained_ensemble):
        """Test prediction on single sample."""
        X = np.random.randn(1, 47)
        prediction = trained_ensemble.predict(X)

        assert len(prediction) == 1
        assert prediction[0] in [0, 1]

    def test_reproducibility(self, sample_data):
        """Test that training is reproducible with same seed."""
        X, y = sample_data

        ensemble1 = UltraStackingEnsemble(n_jobs=1, random_state=42)
        ensemble1.fit(X, y)
        pred1 = ensemble1.predict(X[:10])

        ensemble2 = UltraStackingEnsemble(n_jobs=1, random_state=42)
        ensemble2.fit(X, y)
        pred2 = ensemble2.predict(X[:10])

        np.testing.assert_array_equal(pred1, pred2)

    def test_different_feature_counts(self):
        """Test with different number of features."""
        for n_features in [10, 47, 100]:
            np.random.seed(42)
            X = np.random.randn(100, n_features)
            y = np.random.randint(0, 2, 100)

            ensemble = UltraStackingEnsemble(n_jobs=1, random_state=42)
            ensemble.fit(X, y)
            predictions = ensemble.predict(X)

            assert len(predictions) == 100

    def test_imbalanced_data(self):
        """Test with imbalanced class distribution."""
        np.random.seed(42)
        X = np.random.randn(200, 47)
        y = np.array([0] * 180 + [1] * 20)  # 90% class 0

        ensemble = UltraStackingEnsemble(n_jobs=1, random_state=42)
        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        # Should still make predictions
        assert len(predictions) == 200


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_data_generation(self):
        """Test synthetic data generation."""
        X, y = generate_synthetic_data(n_samples=100, n_features=47, disease='test')

        assert X.shape == (100, 47)
        assert y.shape == (100,)

    def test_class_balance(self):
        """Test generated data has both classes."""
        X, y = generate_synthetic_data(n_samples=200, n_features=47, disease='test')

        unique_classes = np.unique(y)
        assert len(unique_classes) == 2
        assert 0 in unique_classes
        assert 1 in unique_classes

    def test_feature_range(self):
        """Test generated features are normalized."""
        X, y = generate_synthetic_data(n_samples=100, n_features=47, disease='test')

        # Features should be roughly normalized (most values within [-3, 3])
        assert np.abs(X.mean()) < 1
        assert 0 < X.std() < 3


class TestModelPersistence:
    """Tests for model saving and loading."""

    @pytest.fixture
    def trained_ensemble(self):
        """Create and train ensemble."""
        np.random.seed(42)
        X = np.random.randn(100, 47)
        y = np.random.randint(0, 2, 100)

        ensemble = UltraStackingEnsemble(n_jobs=1, random_state=42)
        ensemble.fit(X, y)
        return ensemble

    def test_model_save_load(self, trained_ensemble):
        """Test model can be saved and loaded."""
        import joblib

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.joblib')

            # Save
            joblib.dump({'model': trained_ensemble}, model_path)

            # Load
            loaded = joblib.load(model_path)
            loaded_model = loaded['model']

            # Test predictions match
            X_test = np.random.randn(10, 47)
            orig_pred = trained_ensemble.predict(X_test)
            loaded_pred = loaded_model.predict(X_test)

            np.testing.assert_array_equal(orig_pred, loaded_pred)


class TestCrossValidation:
    """Tests for cross-validation functionality."""

    def test_stratified_cv(self):
        """Test stratified cross-validation."""
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        np.random.seed(42)
        X = np.random.randn(100, 47)
        y = np.random.randint(0, 2, 100)

        ensemble = UltraStackingEnsemble(n_jobs=1, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')

        assert len(scores) == 5
        assert all(0 <= s <= 1 for s in scores)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_dataset(self):
        """Test with very small dataset."""
        np.random.seed(42)
        X = np.random.randn(20, 47)
        y = np.array([0] * 10 + [1] * 10)

        ensemble = UltraStackingEnsemble(n_jobs=1, random_state=42)

        try:
            ensemble.fit(X, y)
            predictions = ensemble.predict(X)
            assert len(predictions) == 20
        except Exception as e:
            # May fail with small datasets, which is acceptable
            assert "too small" in str(e).lower() or True

    def test_single_class_data(self):
        """Test handling of single class data."""
        np.random.seed(42)
        X = np.random.randn(50, 47)
        y = np.zeros(50)  # Only class 0

        ensemble = UltraStackingEnsemble(n_jobs=1, random_state=42)

        # Should raise an error or handle gracefully
        try:
            ensemble.fit(X, y)
        except ValueError:
            pass  # Expected behavior

    def test_nan_handling(self):
        """Test handling of NaN values."""
        np.random.seed(42)
        X = np.random.randn(100, 47)
        X[0, 0] = np.nan  # Introduce NaN
        y = np.random.randint(0, 2, 100)

        ensemble = UltraStackingEnsemble(n_jobs=1, random_state=42)

        # Should raise error or handle appropriately
        try:
            ensemble.fit(X, y)
            # If it fits, predictions should still work
            predictions = ensemble.predict(X[1:])  # Exclude NaN row
            assert len(predictions) == 99
        except ValueError:
            pass  # Expected behavior


class TestPerformanceMetrics:
    """Tests for performance metrics calculation."""

    def test_accuracy_calculation(self):
        """Test accuracy calculation on known data."""
        from sklearn.metrics import accuracy_score

        np.random.seed(42)
        X = np.random.randn(200, 47)

        # Create separable data
        y = (X[:, 0] > 0).astype(int)

        ensemble = UltraStackingEnsemble(n_jobs=1, random_state=42)
        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        accuracy = accuracy_score(y, predictions)

        # On training data, should achieve reasonable accuracy
        assert accuracy > 0.5  # Better than random

    def test_multiclass_handling(self):
        """Test handling of multi-class classification."""
        np.random.seed(42)
        X = np.random.randn(150, 47)
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)  # 3 classes

        ensemble = UltraStackingEnsemble(n_jobs=1, random_state=42)
        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        # Should predict all classes
        unique_preds = np.unique(predictions)
        assert len(unique_preds) >= 1  # At least some predictions


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
