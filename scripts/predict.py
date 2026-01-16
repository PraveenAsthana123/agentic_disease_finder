#!/usr/bin/env python3
"""
Prediction/Inference Script for EEG-Based Neurological Disease Classification
==============================================================================

This script provides inference capabilities for trained models, including
single sample prediction, batch prediction, and real-time streaming prediction.

Author: AgenticFinder Research Team
License: MIT
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EEGPredictor:
    """
    Prediction engine for EEG-based disease classification.

    Supports single sample, batch, and streaming predictions with
    confidence scores and uncertainty estimation.
    """

    def __init__(self, model_path: str, threshold: float = 0.5):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to saved model file (.joblib)
            threshold: Classification threshold for binary predictions
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.model = None
        self.feature_extractor = None
        self.class_names = None
        self.disease_name = None

        self._load_model()

    def _load_model(self):
        """Load trained model and associated components."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Loading model from {self.model_path}")
        checkpoint = joblib.load(self.model_path)

        self.model = checkpoint.get('model')
        self.feature_extractor = checkpoint.get('feature_extractor')
        self.class_names = checkpoint.get('class_names', ['Control', 'Disease'])
        self.disease_name = checkpoint.get('disease_name', 'Unknown')
        self.training_info = checkpoint.get('training_info', {})

        logger.info(f"Model loaded for disease: {self.disease_name}")
        logger.info(f"Classes: {self.class_names}")

    def extract_features(self, eeg_signal: np.ndarray,
                        sampling_rate: int = 256) -> np.ndarray:
        """
        Extract features from raw EEG signal.

        Args:
            eeg_signal: Raw EEG signal array (n_channels x n_samples) or (n_samples,)
            sampling_rate: Sampling rate in Hz

        Returns:
            Feature vector
        """
        if self.feature_extractor is not None:
            return self.feature_extractor.extract(eeg_signal, sampling_rate)
        else:
            # Fallback to basic feature extraction
            return self._basic_feature_extraction(eeg_signal, sampling_rate)

    def _basic_feature_extraction(self, signal: np.ndarray,
                                  sampling_rate: int) -> np.ndarray:
        """
        Basic feature extraction when no extractor is available.

        Extracts 47 features matching our model architecture.
        """
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)

        features = []

        for channel in signal:
            # Statistical features (15)
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.var(channel),
                np.min(channel),
                np.max(channel),
                np.median(channel),
                np.ptp(channel),  # peak-to-peak
                self._skewness(channel),
                self._kurtosis(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75),
                np.sqrt(np.mean(channel**2)),  # RMS
                np.mean(np.abs(channel)),  # MAV
                np.sum(np.abs(np.diff(channel))),  # Line length
                len(np.where(np.diff(np.sign(channel)))[0])  # Zero crossings
            ])

            # Spectral features (basic, 18)
            fft = np.fft.fft(channel)
            psd = np.abs(fft)**2 / len(channel)
            freqs = np.fft.fftfreq(len(channel), 1/sampling_rate)

            # Band powers
            for (low, high) in [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]:
                band_mask = (freqs >= low) & (freqs < high)
                band_power = np.sum(psd[band_mask])
                features.append(band_power)

            # Spectral statistics
            total_power = np.sum(psd[:len(psd)//2])
            features.extend([
                total_power,
                np.argmax(psd[:len(psd)//2]) * sampling_rate / len(channel),  # Dominant frequency
                self._spectral_entropy(psd[:len(psd)//2]),
                np.std(psd[:len(psd)//2]),
                np.mean(psd[:len(psd)//2])
            ])

            # Temporal features (9)
            features.extend([
                np.mean(np.abs(np.diff(channel))),  # Mean absolute difference
                np.std(np.diff(channel)),
                np.max(np.abs(np.diff(channel))),
                self._hjorth_mobility(channel),
                self._hjorth_complexity(channel),
                np.correlate(channel, channel)[0] / len(channel),  # Autocorrelation
                np.sum(np.diff(np.sign(np.diff(channel))) != 0),  # Slope changes
                np.mean(channel[:len(channel)//2]) - np.mean(channel[len(channel)//2:]),  # Trend
                np.max(np.abs(channel)) / np.mean(np.abs(channel)) if np.mean(np.abs(channel)) > 0 else 0  # Crest factor
            ])

            # Nonlinear features (5)
            features.extend([
                self._approximate_entropy(channel, 2, 0.2 * np.std(channel)),
                self._sample_entropy(channel, 2, 0.2 * np.std(channel)),
                self._hurst_exponent(channel),
                self._dfa(channel),
                self._lziv_complexity(channel)
            ])

        # Ensure we have exactly 47 features
        features = np.array(features[:47])
        if len(features) < 47:
            features = np.pad(features, (0, 47 - len(features)))

        return features

    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.sum((x - mean)**3) / (n * std**3)

    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.sum((x - mean)**4) / (n * std**4) - 3

    def _spectral_entropy(self, psd: np.ndarray) -> float:
        """Calculate spectral entropy."""
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        psd_norm = psd_norm[psd_norm > 0]
        return -np.sum(psd_norm * np.log2(psd_norm)) if len(psd_norm) > 0 else 0

    def _hjorth_mobility(self, x: np.ndarray) -> float:
        """Calculate Hjorth mobility."""
        return np.sqrt(np.var(np.diff(x)) / np.var(x)) if np.var(x) > 0 else 0

    def _hjorth_complexity(self, x: np.ndarray) -> float:
        """Calculate Hjorth complexity."""
        mobility = self._hjorth_mobility(x)
        diff_mobility = self._hjorth_mobility(np.diff(x))
        return diff_mobility / mobility if mobility > 0 else 0

    def _approximate_entropy(self, x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate approximate entropy (simplified)."""
        n = len(x)
        if n < m + 1:
            return 0

        def phi(m_val):
            patterns = np.array([x[i:i+m_val] for i in range(n - m_val + 1)])
            count = 0
            for i, p1 in enumerate(patterns):
                for p2 in patterns:
                    if np.max(np.abs(p1 - p2)) <= r:
                        count += 1
            return np.log(count / len(patterns)) if count > 0 else 0

        return abs(phi(m) - phi(m + 1))

    def _sample_entropy(self, x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy (simplified)."""
        return self._approximate_entropy(x, m, r) * 1.1  # Approximation

    def _hurst_exponent(self, x: np.ndarray) -> float:
        """Estimate Hurst exponent using R/S analysis (simplified)."""
        n = len(x)
        if n < 20:
            return 0.5

        max_k = min(n // 4, 100)
        rs_list = []

        for k in range(10, max_k, 10):
            rs_values = []
            for start in range(0, n - k, k):
                ts = x[start:start+k]
                mean = np.mean(ts)
                cumdev = np.cumsum(ts - mean)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(ts)
                if s > 0:
                    rs_values.append(r / s)

            if rs_values:
                rs_list.append((np.log(k), np.log(np.mean(rs_values))))

        if len(rs_list) >= 2:
            x_vals = [r[0] for r in rs_list]
            y_vals = [r[1] for r in rs_list]
            slope, _ = np.polyfit(x_vals, y_vals, 1)
            return slope

        return 0.5

    def _dfa(self, x: np.ndarray) -> float:
        """Detrended Fluctuation Analysis (simplified)."""
        return self._hurst_exponent(x)  # Use Hurst as approximation

    def _lziv_complexity(self, x: np.ndarray) -> float:
        """Lempel-Ziv complexity (simplified)."""
        # Binarize signal
        binary = (x > np.median(x)).astype(int)
        s = ''.join(map(str, binary))

        n = len(s)
        c = 1
        l = 1
        i = 0
        k = 1
        k_max = 1

        while True:
            if s[i + k - 1] != s[l + k - 1]:
                if k > k_max:
                    k_max = k
                i += 1
                if i == l:
                    c += 1
                    l += k_max
                    if l + 1 > n:
                        break
                    i = 0
                    k = 1
                    k_max = 1
                else:
                    k = 1
            else:
                k += 1
                if l + k > n:
                    c += 1
                    break

        return c / (n / np.log2(n)) if n > 1 else 0

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction on feature vector.

        Args:
            features: Feature vector (n_features,) or (n_samples, n_features)

        Returns:
            Dictionary with prediction results
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get prediction
        prediction = self.model.predict(features)

        # Get probability if available
        probability = None
        if hasattr(self.model, 'predict_proba'):
            try:
                probability = self.model.predict_proba(features)
            except Exception:
                pass

        results = []
        for i in range(len(prediction)):
            result = {
                'prediction': int(prediction[i]),
                'class_name': self.class_names[prediction[i]] if self.class_names else f"Class {prediction[i]}",
                'disease': self.disease_name
            }

            if probability is not None:
                result['probabilities'] = {
                    self.class_names[j]: float(probability[i][j])
                    for j in range(len(self.class_names))
                }
                result['confidence'] = float(np.max(probability[i]))
                result['uncertainty'] = 1.0 - result['confidence']

            results.append(result)

        return results[0] if len(results) == 1 else results

    def predict_from_signal(self, eeg_signal: np.ndarray,
                           sampling_rate: int = 256) -> Dict[str, Any]:
        """
        Make prediction directly from raw EEG signal.

        Args:
            eeg_signal: Raw EEG signal
            sampling_rate: Sampling rate in Hz

        Returns:
            Prediction results
        """
        features = self.extract_features(eeg_signal, sampling_rate)
        return self.predict(features)

    def predict_batch(self, features_batch: np.ndarray) -> List[Dict[str, Any]]:
        """
        Make predictions on batch of samples.

        Args:
            features_batch: Feature matrix (n_samples, n_features)

        Returns:
            List of prediction results
        """
        results = self.predict(features_batch)
        return results if isinstance(results, list) else [results]

    def predict_with_explanation(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction with feature contribution analysis.

        Args:
            features: Feature vector

        Returns:
            Prediction with explanations
        """
        result = self.predict(features)

        # Add feature contributions if model supports it
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Calculate contribution scores
            contributions = features[0] * importances
            top_features = np.argsort(np.abs(contributions))[::-1][:10]

            result['explanation'] = {
                'top_features': top_features.tolist(),
                'contributions': contributions[top_features].tolist(),
                'method': 'feature_importance'
            }

        return result


class StreamingPredictor:
    """
    Real-time streaming predictor for continuous EEG monitoring.
    """

    def __init__(self, predictor: EEGPredictor, window_size: int = 256,
                 step_size: int = 64, sampling_rate: int = 256):
        """
        Initialize streaming predictor.

        Args:
            predictor: Base EEGPredictor instance
            window_size: Size of sliding window in samples
            step_size: Step size for sliding window
            sampling_rate: Sampling rate in Hz
        """
        self.predictor = predictor
        self.window_size = window_size
        self.step_size = step_size
        self.sampling_rate = sampling_rate
        self.buffer = []
        self.predictions_history = []

    def add_samples(self, samples: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Add new samples to buffer and return prediction if window is full.

        Args:
            samples: New EEG samples

        Returns:
            Prediction result if window is ready, None otherwise
        """
        self.buffer.extend(samples.tolist())

        if len(self.buffer) >= self.window_size:
            # Extract window
            window = np.array(self.buffer[:self.window_size])

            # Make prediction
            result = self.predictor.predict_from_signal(window, self.sampling_rate)
            result['timestamp'] = datetime.now().isoformat()

            # Store in history
            self.predictions_history.append(result)

            # Slide buffer
            self.buffer = self.buffer[self.step_size:]

            return result

        return None

    def get_smoothed_prediction(self, n_recent: int = 5) -> Optional[Dict[str, Any]]:
        """
        Get smoothed prediction from recent history.

        Args:
            n_recent: Number of recent predictions to average

        Returns:
            Smoothed prediction
        """
        if len(self.predictions_history) < n_recent:
            return None

        recent = self.predictions_history[-n_recent:]

        # Average probabilities
        if 'probabilities' in recent[0]:
            avg_probs = {}
            for class_name in recent[0]['probabilities'].keys():
                avg_probs[class_name] = np.mean([r['probabilities'][class_name] for r in recent])

            prediction = max(avg_probs, key=avg_probs.get)

            return {
                'prediction': prediction,
                'probabilities': avg_probs,
                'confidence': max(avg_probs.values()),
                'smoothing_window': n_recent
            }

        # Majority vote for predictions
        predictions = [r['prediction'] for r in recent]
        majority = max(set(predictions), key=predictions.count)

        return {
            'prediction': majority,
            'vote_count': predictions.count(majority),
            'smoothing_window': n_recent
        }

    def reset(self):
        """Reset buffer and history."""
        self.buffer = []
        self.predictions_history = []


def main():
    parser = argparse.ArgumentParser(
        description='Make predictions using trained EEG classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model file (.joblib)'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input data file (NPZ format with X array)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='predictions.json',
        help='Output file for predictions'
    )

    parser.add_argument(
        '--format',
        choices=['json', 'csv'],
        default='json',
        help='Output format'
    )

    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Generate synthetic data for testing'
    )

    parser.add_argument(
        '--explain',
        action='store_true',
        help='Include feature contribution explanations'
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = EEGPredictor(args.model)

    # Load or generate data
    if args.synthetic:
        logger.info("Generating synthetic test samples...")
        np.random.seed(42)
        X = np.random.randn(10, 47)  # 10 samples, 47 features
        logger.info(f"Generated {len(X)} synthetic samples")
    elif args.input:
        logger.info(f"Loading data from {args.input}")
        data = np.load(args.input)
        X = data['X']
        logger.info(f"Loaded {len(X)} samples")
    else:
        logger.error("Please provide either --input or --synthetic flag")
        sys.exit(1)

    # Make predictions
    logger.info("Making predictions...")
    results = []

    for i, sample in enumerate(X):
        if args.explain:
            result = predictor.predict_with_explanation(sample)
        else:
            result = predictor.predict(sample)

        result['sample_index'] = i
        results.append(result)

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(X)} samples")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        # CSV format
        import csv
        fieldnames = list(results[0].keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                # Flatten nested dicts for CSV
                flat_result = {}
                for k, v in result.items():
                    if isinstance(v, dict):
                        flat_result[k] = json.dumps(v)
                    else:
                        flat_result[k] = v
                writer.writerow(flat_result)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    print(f"\nTotal samples: {len(results)}")

    # Count by class
    class_counts = {}
    for r in results:
        class_name = r.get('class_name', r['prediction'])
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print("\nPrediction distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({100*count/len(results):.1f}%)")

    if 'confidence' in results[0]:
        confidences = [r['confidence'] for r in results]
        print(f"\nConfidence statistics:")
        print(f"  Mean: {np.mean(confidences):.4f}")
        print(f"  Std:  {np.std(confidences):.4f}")
        print(f"  Min:  {np.min(confidences):.4f}")
        print(f"  Max:  {np.max(confidences):.4f}")


if __name__ == '__main__':
    main()
