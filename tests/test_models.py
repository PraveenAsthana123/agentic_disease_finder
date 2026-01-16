"""
Unit Tests for Model Inference Module
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.inference import (
    AlzheimerInferenceModel,
    ParkinsonInferenceModel,
    SchizophreniaInferenceModel,
    MultiDiseaseInferenceEngine,
    PredictionResult
)


class TestAlzheimerInferenceModel:
    """Test Alzheimer's inference model"""

    @pytest.fixture
    def model(self):
        model = AlzheimerInferenceModel()
        model.initialize_default_model()
        return model

    @pytest.fixture
    def sample_features(self):
        np.random.seed(42)
        return np.random.rand(20)

    def test_initialization(self, model):
        assert model is not None
        assert model.classes == ['CN', 'MCI', 'AD']
        assert model.is_loaded

    def test_predict(self, model, sample_features):
        result = model.predict(sample_features)

        assert isinstance(result, PredictionResult)
        assert result.class_name in ['CN', 'MCI', 'AD']
        assert 0 <= result.confidence <= 1

    def test_probabilities_sum_to_one(self, model, sample_features):
        result = model.predict(sample_features)

        prob_sum = sum(result.probabilities.values())
        assert abs(prob_sum - 1.0) < 0.01

    def test_batch_prediction(self, model):
        np.random.seed(42)
        batch = np.random.rand(10, 20)

        # Predict each sample
        predictions = [model.predict(batch[i]) for i in range(10)]

        assert len(predictions) == 10
        assert all(isinstance(p, PredictionResult) for p in predictions)


class TestParkinsonInferenceModel:
    """Test Parkinson's inference model"""

    @pytest.fixture
    def model(self):
        model = ParkinsonInferenceModel()
        model.initialize_default_model()
        return model

    @pytest.fixture
    def sample_features(self):
        np.random.seed(42)
        return np.random.rand(26)

    def test_initialization(self, model):
        assert model is not None
        assert model.classes == ['HC', 'PD']
        assert model.is_loaded

    def test_predict(self, model, sample_features):
        result = model.predict(sample_features)

        assert isinstance(result, PredictionResult)
        assert result.class_name in ['HC', 'PD']
        assert 0 <= result.confidence <= 1

    def test_model_info(self, model, sample_features):
        result = model.predict(sample_features)

        assert result.model_info['disease'] == 'Parkinson'
        assert result.model_info['num_classes'] == 2


class TestSchizophreniaInferenceModel:
    """Test Schizophrenia inference model"""

    @pytest.fixture
    def model(self):
        model = SchizophreniaInferenceModel()
        model.initialize_default_model()
        return model

    @pytest.fixture
    def sample_features(self):
        np.random.seed(42)
        return np.random.rand(30)

    def test_initialization(self, model):
        assert model is not None
        assert model.classes == ['HC', 'SZ']
        assert model.is_loaded

    def test_predict(self, model, sample_features):
        result = model.predict(sample_features)

        assert isinstance(result, PredictionResult)
        assert result.class_name in ['HC', 'SZ']
        assert 0 <= result.confidence <= 1


class TestMultiDiseaseInferenceEngine:
    """Test multi-disease inference engine"""

    @pytest.fixture
    def engine(self):
        engine = MultiDiseaseInferenceEngine()
        engine.load_all_models()
        return engine

    @pytest.fixture
    def sample_features_dict(self):
        np.random.seed(42)
        return {
            'alzheimer': np.random.rand(20),
            'parkinson': np.random.rand(26),
            'schizophrenia': np.random.rand(30)
        }

    def test_initialization(self, engine):
        assert engine is not None
        assert 'alzheimer' in engine.models
        assert 'parkinson' in engine.models
        assert 'schizophrenia' in engine.models

    def test_predict_single(self, engine, sample_features_dict):
        result = engine.predict('alzheimer', sample_features_dict['alzheimer'])

        assert isinstance(result, PredictionResult)

    def test_screen_all_diseases(self, engine, sample_features_dict):
        results = engine.screen_all_diseases(sample_features_dict)

        assert 'alzheimer' in results
        assert 'parkinson' in results
        assert 'schizophrenia' in results

    def test_risk_assessment(self, engine, sample_features_dict):
        results = engine.screen_all_diseases(sample_features_dict)
        assessment = engine.get_risk_assessment(results)

        assert 'diseases_screened' in assessment
        assert 'risk_levels' in assessment
        assert 'recommendations' in assessment
        assert 'overall_risk' in assessment

    def test_invalid_disease(self, engine):
        with pytest.raises(ValueError):
            engine.predict('invalid_disease', np.random.rand(20))


class TestModelTrainingAndSaving:
    """Test model training and persistence"""

    def test_train_alzheimer_model(self, tmp_path):
        model = AlzheimerInferenceModel()
        model.initialize_default_model()

        # Generate training data
        np.random.seed(42)
        X = np.random.rand(100, 20)
        y = np.random.randint(0, 3, 100)

        model.train(X, y)

        # Save model
        save_path = tmp_path / "test_model.pkl"
        model.save(str(save_path))

        assert save_path.exists()

    def test_load_saved_model(self, tmp_path):
        # Create and save model
        model = AlzheimerInferenceModel()
        model.initialize_default_model()

        np.random.seed(42)
        X = np.random.rand(100, 20)
        y = np.random.randint(0, 3, 100)
        model.train(X, y)

        save_path = tmp_path / "test_model.pkl"
        model.save(str(save_path))

        # Load in new model
        new_model = AlzheimerInferenceModel(str(save_path))
        loaded = new_model.load()

        assert loaded
        assert new_model.is_loaded

        # Test prediction
        result = new_model.predict(X[0])
        assert isinstance(result, PredictionResult)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
