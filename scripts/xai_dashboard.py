#!/usr/bin/env python3
"""
Explainable AI (XAI) Dashboard — Captum + LIME + SHAP on real EEG features
==========================================================================

Applies real explainability methods to a trained EEG classifier:
  - Captum Integrated Gradients (Sundararajan et al. 2017)
  - LIME (Ribeiro et al. 2016)
  - SHAP (Lundberg & Lee, 2017)

Input: real EEG features extracted from CHB-MIT EDF recordings via MNE.
Model: a small PyTorch classifier trained on spectral/entropy features.

Clinical relevance:
  XAI for EEG-AI is mandated by EU AI Act Art. 86 for clinical decision
  support.  It answers 'why did the model flag this EEG segment?' by
  attributing the prediction to specific EEG channels / frequency bands /
  entropy metrics.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent.parent

_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _find_edf() -> Optional[Path]:
    """Return the first available real EDF file."""
    dirs = [
        ROOT / "data/real_eeg/epilepsy_physionet",
        ROOT / "data/real_eeg/depression_figshare",
        ROOT / "data/real_eeg",
    ]
    for d in dirs:
        if d.is_dir():
            edfs = sorted(d.glob("*.edf"))
            if edfs:
                return edfs[0]
    return None


def _extract_features(file=None, seconds=10.0):
    """Extract real EEG features from EDF for XAI analysis.

    Returns (feature_matrix, feature_names, channel_names, labels, metadata).
    """
    import mne
    mne.set_log_level("ERROR")

    if file:
        p = ROOT / file if not Path(file).is_absolute() else Path(file)
    else:
        p = _find_edf()
    if not p or not p.exists():
        return None, None, None, None, None

    raw = mne.io.read_raw_edf(str(p), preload=True, verbose=False)
    raw.filter(0.5, 45.0, verbose=False)
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names[:8]
    raw.pick(ch_names)

    epoch_len = 2.0
    n_samples_epoch = int(epoch_len * sfreq)
    total_samples = min(int(seconds * sfreq), raw.n_times)
    data = raw.get_data(start=0, stop=total_samples)

    n_epochs = total_samples // n_samples_epoch
    if n_epochs < 2:
        n_epochs = 1

    feature_names = []
    for ch in ch_names:
        for band in _BANDS:
            feature_names.append(f"{ch}_{band}_power")
        feature_names.append(f"{ch}_line_length")
        feature_names.append(f"{ch}_variance")
        feature_names.append(f"{ch}_kurtosis")

    n_features = len(feature_names)
    X = np.zeros((n_epochs, n_features), dtype=np.float32)

    from scipy.signal import welch
    from scipy.stats import kurtosis as sp_kurtosis

    for ep in range(n_epochs):
        start = ep * n_samples_epoch
        end = start + n_samples_epoch
        seg = data[:, start:end]

        idx = 0
        for ci in range(len(ch_names)):
            ch_data = seg[ci]
            freqs, psd = welch(ch_data, fs=sfreq, nperseg=min(256, n_samples_epoch))
            for band, (lo, hi) in _BANDS.items():
                mask = (freqs >= lo) & (freqs <= hi)
                X[ep, idx] = np.log1p(np.sum(psd[mask]))
                idx += 1
            X[ep, idx] = np.mean(np.abs(np.diff(ch_data)))
            idx += 1
            X[ep, idx] = np.var(ch_data)
            idx += 1
            X[ep, idx] = float(sp_kurtosis(ch_data, fisher=True))
            idx += 1

    # Binary labels based on high-frequency power threshold
    gamma_cols = [i for i, name in enumerate(feature_names) if "gamma_power" in name]
    gamma_mean = np.mean(X[:, gamma_cols], axis=1)
    threshold = np.median(gamma_mean)
    labels = (gamma_mean > threshold).astype(int)

    meta = {
        "file": str(p.name),
        "sfreq": float(sfreq),
        "n_channels": len(ch_names),
        "n_epochs": int(n_epochs),
        "n_features": n_features,
        "epoch_duration_s": epoch_len,
        "seconds_analyzed": float(n_epochs * epoch_len),
    }
    return X, feature_names, ch_names, labels, meta


def _train_model(X, labels, feature_names):
    """Train a small PyTorch classifier for XAI analysis."""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_features = X.shape[1]
    class_names = ["Normal", "Abnormal"]

    class EEGClassifier(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 2)
            )

        def forward(self, x):
            return self.net(x)

    model = EEGClassifier(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.long)

    model.train()
    for _epoch in range(100):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).numpy()
    accuracy = float(np.mean(preds == labels))

    return model, scaler, accuracy, class_names


def _json_clean(obj):
    """Make numpy types JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return round(v, 6)
    if isinstance(obj, np.ndarray):
        return [_json_clean(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(x) for x in obj]
    return obj


def xai_overview(file=None, seconds=10.0):
    """Summary of XAI analysis: model info, methods available, top features."""
    X, feat_names, ch_names, labels, meta = _extract_features(file, seconds)
    if X is None:
        return {"available": False, "note": "No EDF data found for XAI analysis."}

    model, scaler, accuracy, class_names = _train_model(X, labels, feat_names)

    from sklearn.ensemble import GradientBoostingClassifier
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    X_scaled = scaler.transform(X)
    gb.fit(X_scaled, labels)

    import shap
    explainer = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        sv = np.abs(shap_values[1])
    else:
        sv = np.abs(shap_values)

    mean_shap = np.mean(sv, axis=0)
    top_idx = np.argsort(mean_shap)[::-1][:10]

    top_features = [
        {"feature": feat_names[i], "importance": round(float(mean_shap[i]), 6)}
        for i in top_idx
    ]

    return _json_clean({
        "available": True,
        "model": {
            "type": "PyTorch 2-layer MLP (32-16-2)",
            "accuracy": accuracy,
            "n_params": sum(p.numel() for p in model.parameters()),
            "class_names": class_names,
        },
        "data": meta,
        "methods": [
            {"name": "Captum Integrated Gradients", "library": "captum", "version": "0.9.0",
             "type": "gradient-based", "scope": "local"},
            {"name": "LIME", "library": "lime", "version": "0.2.0",
             "type": "surrogate-model", "scope": "local"},
            {"name": "SHAP TreeExplainer", "library": "shap",
             "type": "shapley-value", "scope": "global+local"},
            {"name": "Grad-CAM", "library": "pytorch-grad-cam", "version": "1.5.5",
             "type": "gradient-activation", "scope": "local"},
        ],
        "top_features_shap": top_features,
        "channels_analyzed": ch_names,
        "feature_groups": {
            "spectral_power": [f for f in feat_names if "power" in f],
            "line_length": [f for f in feat_names if "line_length" in f],
            "variance": [f for f in feat_names if "variance" in f],
            "kurtosis": [f for f in feat_names if "kurtosis" in f],
        },
    })


def xai_captum(file=None, seconds=10.0):
    """Captum Integrated Gradients + Feature Ablation attributions."""
    import torch
    from captum.attr import IntegratedGradients, FeatureAblation

    X, feat_names, ch_names, labels, meta = _extract_features(file, seconds)
    if X is None:
        return {"available": False, "note": "No EDF data found."}

    model, scaler, accuracy, class_names = _train_model(X, labels, feat_names)
    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(X_t[0:1])

    ig_attrs = []
    for i in range(min(len(X_t), 20)):
        attr = ig.attribute(X_t[i:i+1], baselines=baseline, target=int(labels[i]),
                            n_steps=50)
        ig_attrs.append(attr.detach().numpy()[0])
    ig_attrs = np.array(ig_attrs)
    mean_ig = np.mean(np.abs(ig_attrs), axis=0)

    fa = FeatureAblation(model)
    fa_attrs = []
    for i in range(min(len(X_t), 20)):
        attr = fa.attribute(X_t[i:i+1], target=int(labels[i]))
        fa_attrs.append(attr.detach().numpy()[0])
    fa_attrs = np.array(fa_attrs)
    mean_fa = np.mean(np.abs(fa_attrs), axis=0)

    top_ig_idx = np.argsort(mean_ig)[::-1][:15]
    top_fa_idx = np.argsort(mean_fa)[::-1][:15]

    n_feat_per_ch = len(feat_names) // len(ch_names)
    channel_importance_ig = []
    for ci, ch in enumerate(ch_names):
        start = ci * n_feat_per_ch
        end = start + n_feat_per_ch
        channel_importance_ig.append({
            "channel": ch,
            "importance": round(float(np.mean(mean_ig[start:end])), 6),
        })
    channel_importance_ig.sort(key=lambda x: x["importance"], reverse=True)

    return _json_clean({
        "available": True,
        "method": "Captum (Kokhlikyan et al. 2020)",
        "integrated_gradients": {
            "description": "Attributes prediction to each feature by integrating gradients along path from baseline to input",
            "top_features": [
                {"feature": feat_names[i], "attribution": round(float(mean_ig[i]), 6)}
                for i in top_ig_idx
            ],
            "per_channel": channel_importance_ig,
        },
        "feature_ablation": {
            "description": "Measures prediction change when each feature is replaced with baseline value",
            "top_features": [
                {"feature": feat_names[i], "attribution": round(float(mean_fa[i]), 6)}
                for i in top_fa_idx
            ],
        },
        "n_samples_explained": min(len(X_t), 20),
        "model_accuracy": accuracy,
    })


def xai_lime(file=None, seconds=10.0):
    """LIME explanations for sampled instances."""
    import lime.lime_tabular
    import torch

    X, feat_names, ch_names, labels, meta = _extract_features(file, seconds)
    if X is None:
        return {"available": False, "note": "No EDF data found."}

    model, scaler, accuracy, class_names = _train_model(X, labels, feat_names)
    X_scaled = scaler.transform(X)

    def predict_fn(x):
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            logits = model(t)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_scaled, feature_names=feat_names, class_names=class_names,
        mode="classification", random_state=42,
    )

    n_explain = min(10, len(X_scaled))
    all_weights = np.zeros(len(feat_names))
    sample_explanations = []

    for i in range(n_explain):
        exp = explainer.explain_instance(
            X_scaled[i], predict_fn, num_features=len(feat_names),
            top_labels=1, num_samples=200
        )
        label = exp.available_labels()[0]
        weights = dict(exp.as_list(label=label))

        for fi, fname in enumerate(feat_names):
            w = weights.get(fname, 0.0)
            all_weights[fi] += abs(w)

        if i < 3:
            top_exp = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            sample_explanations.append({
                "sample_idx": i,
                "predicted_class": class_names[label],
                "true_class": class_names[int(labels[i])],
                "top_contributions": [
                    {"feature": f, "weight": round(w, 6)} for f, w in top_exp
                ],
            })

    all_weights /= n_explain
    top_idx = np.argsort(all_weights)[::-1][:15]

    return _json_clean({
        "available": True,
        "method": "LIME (Ribeiro et al. 2016)",
        "description": "Perturbs input features, fits local linear model to approximate prediction boundary",
        "global_feature_importance": [
            {"feature": feat_names[i], "mean_abs_weight": round(float(all_weights[i]), 6)}
            for i in top_idx
        ],
        "sample_explanations": sample_explanations,
        "n_samples_explained": n_explain,
        "n_perturbations_per_sample": 200,
        "model_accuracy": accuracy,
    })


def xai_shap(file=None, seconds=10.0):
    """SHAP values — global + local feature importance."""
    try:
        import shap
    except (ImportError, Exception) as e:
        return {"available": False, "note": f"SHAP unavailable: {e}"}
    from sklearn.ensemble import GradientBoostingClassifier

    X, feat_names, ch_names, labels, meta = _extract_features(file, seconds)
    if X is None:
        return {"available": False, "note": "No EDF data found."}

    model_pt, scaler, accuracy, class_names = _train_model(X, labels, feat_names)
    X_scaled = scaler.transform(X)

    gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    gb.fit(X_scaled, labels)
    gb_accuracy = float(gb.score(X_scaled, labels))

    explainer = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs_shap = np.mean(np.abs(sv), axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:15]

    n_feat_per_ch = len(feat_names) // len(ch_names)
    channel_shap = []
    for ci, ch in enumerate(ch_names):
        start = ci * n_feat_per_ch
        end = start + n_feat_per_ch
        channel_shap.append({
            "channel": ch,
            "mean_abs_shap": round(float(np.mean(mean_abs_shap[start:end])), 6),
        })
    channel_shap.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

    band_shap = {}
    for band in _BANDS:
        band_cols = [i for i, f in enumerate(feat_names) if f"{band}_power" in f]
        if band_cols:
            band_shap[band] = round(float(np.mean(mean_abs_shap[band_cols])), 6)

    waterfall_samples = []
    for i in range(min(3, len(sv))):
        sorted_idx = np.argsort(np.abs(sv[i]))[::-1][:8]
        exp_val = explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            base = float(exp_val[1])
        else:
            base = float(exp_val)
        waterfall_samples.append({
            "sample_idx": i,
            "predicted_class": class_names[int(gb.predict(X_scaled[i:i+1])[0])],
            "true_class": class_names[int(labels[i])],
            "base_value": round(base, 6),
            "contributions": [
                {"feature": feat_names[j], "shap_value": round(float(sv[i, j]), 6)}
                for j in sorted_idx
            ],
        })

    return _json_clean({
        "available": True,
        "method": "SHAP TreeExplainer (Lundberg and Lee 2017)",
        "explainer_model": "GradientBoosting (50 trees, depth 3)",
        "explainer_accuracy": gb_accuracy,
        "global_importance": [
            {"feature": feat_names[i], "mean_abs_shap": round(float(mean_abs_shap[i]), 6)}
            for i in top_idx
        ],
        "per_channel": channel_shap,
        "per_band": band_shap,
        "waterfall_samples": waterfall_samples,
        "n_samples": len(X_scaled),
    })


def xai_comparison(file=None, seconds=10.0):
    """Side-by-side ranking comparison across Captum IG, LIME, and SHAP."""
    import torch
    import shap
    from captum.attr import IntegratedGradients
    import lime.lime_tabular
    from sklearn.ensemble import GradientBoostingClassifier

    X, feat_names, ch_names, labels, meta = _extract_features(file, seconds)
    if X is None:
        return {"available": False, "note": "No EDF data found."}

    model, scaler, accuracy, class_names = _train_model(X, labels, feat_names)
    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    # Captum IG
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(X_t[0:1])
    ig_attrs = []
    for i in range(min(len(X_t), 15)):
        attr = ig.attribute(X_t[i:i+1], baselines=baseline, target=int(labels[i]), n_steps=30)
        ig_attrs.append(np.abs(attr.detach().numpy()[0]))
    mean_ig = np.mean(ig_attrs, axis=0)

    # LIME
    def predict_fn(x):
        with torch.no_grad():
            return torch.softmax(model(torch.tensor(x, dtype=torch.float32)), dim=1).numpy()

    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        X_scaled, feature_names=feat_names, class_names=class_names,
        mode="classification", random_state=42)
    lime_weights = np.zeros(len(feat_names))
    for i in range(min(10, len(X_scaled))):
        exp = lime_exp.explain_instance(X_scaled[i], predict_fn, num_features=len(feat_names),
                                        top_labels=1, num_samples=150)
        label = exp.available_labels()[0]
        for fname, w in exp.as_list(label=label):
            if fname in feat_names:
                fi = feat_names.index(fname)
                lime_weights[fi] += abs(w)
    lime_weights /= min(10, len(X_scaled))

    # SHAP
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    gb.fit(X_scaled, labels)
    sv = shap.TreeExplainer(gb).shap_values(X_scaled)
    if isinstance(sv, list):
        sv = sv[1]
    mean_shap = np.mean(np.abs(sv), axis=0)

    def _norm(arr):
        mx = np.max(arr)
        return arr / mx if mx > 0 else arr

    ig_n = _norm(mean_ig)
    lime_n = _norm(lime_weights)
    shap_n = _norm(mean_shap)

    consensus = (ig_n + lime_n + shap_n) / 3.0
    top_idx = np.argsort(consensus)[::-1][:15]

    comparison = []
    for rank, i in enumerate(top_idx):
        comparison.append({
            "rank": rank + 1,
            "feature": feat_names[i],
            "captum_ig": round(float(ig_n[i]), 4),
            "lime": round(float(lime_n[i]), 4),
            "shap": round(float(shap_n[i]), 4),
            "consensus": round(float(consensus[i]), 4),
        })

    from scipy.stats import kendalltau
    tau_ig_shap = float(kendalltau(ig_n, shap_n).statistic)
    tau_ig_lime = float(kendalltau(ig_n, lime_n).statistic)
    tau_lime_shap = float(kendalltau(lime_n, shap_n).statistic)

    return _json_clean({
        "available": True,
        "comparison": comparison,
        "agreement": {
            "captum_vs_shap": round(tau_ig_shap, 4),
            "captum_vs_lime": round(tau_ig_lime, 4),
            "lime_vs_shap": round(tau_lime_shap, 4),
            "note": "Kendall tau rank correlation (-1 to 1); higher means more agreement",
        },
        "model_accuracy": accuracy,
    })


def xai_gradcam(file=None, seconds=10.0):
    """Grad-CAM visual explanations using a 2D CNN on EEG features.

    Reshapes per-channel EEG features into a 2D grid (channels x feature_types),
    applies a small Conv2d network, then computes Grad-CAM (Selvaraju et al. 2017)
    heatmaps highlighting which channels and feature groups drive the prediction.
    Uses the pytorch-grad-cam library for correct gradient-weighted activation mapping.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pytorch_grad_cam import GradCAM as GradCAMImpl
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from sklearn.preprocessing import StandardScaler

    X, feat_names, ch_names, labels, meta = _extract_features(file, seconds)
    if X is None:
        return {"available": False, "note": "No EDF data found for Grad-CAM analysis."}

    n_channels = len(ch_names)
    n_feat_per_ch = len(feat_names) // n_channels  # 8 features per channel

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Reshape to 2D pseudo-image: (batch, 1, n_channels, n_feat_per_ch)
    X_2d = X_scaled.reshape(-1, n_channels, n_feat_per_ch)
    X_img = torch.tensor(X_2d, dtype=torch.float32).unsqueeze(1)  # add channel dim
    y_t = torch.tensor(labels, dtype=torch.long)

    class EEG_CNN2D(nn.Module):
        def __init__(self, h, w):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 2)

        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = EEG_CNN2D(n_channels, n_feat_per_ch)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(120):
        optimizer.zero_grad()
        loss = criterion(model(X_img), y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_img).argmax(dim=1).numpy()
    accuracy = float(np.mean(preds == labels))

    # Grad-CAM on conv2 (last conv layer) — pytorch-grad-cam handles 2D inputs
    cam = GradCAMImpl(model=model, target_layers=[model.conv2])

    n_explain = min(len(X_img), 20)
    all_heatmaps = np.zeros((n_explain, n_channels, n_feat_per_ch))
    sample_heatmaps = []

    for i in range(n_explain):
        inp = X_img[i:i+1]
        target = [ClassifierOutputTarget(int(labels[i]))]
        grayscale_cam = cam(input_tensor=inp, targets=target)  # (1, H, W)
        hm = grayscale_cam[0]  # (n_channels, n_feat_per_ch)
        all_heatmaps[i] = hm

        if i < 5:
            ch_scores = hm.mean(axis=1)  # average across feature types per channel
            sample_heatmaps.append({
                "sample_idx": i,
                "predicted_class": ["Normal", "Abnormal"][int(preds[i])],
                "true_class": ["Normal", "Abnormal"][int(labels[i])],
                "channel_activations": [
                    {"channel": ch_names[ci], "activation": round(float(ch_scores[ci]), 6)}
                    for ci in range(n_channels)
                ],
            })

    # Aggregate: channel importance (mean across feat types and samples)
    mean_hm = np.mean(all_heatmaps, axis=0)  # (n_channels, n_feat_per_ch)
    channel_scores = mean_hm.mean(axis=1)
    channel_importance = sorted(
        [{"channel": ch_names[ci], "grad_cam_score": round(float(channel_scores[ci]), 6)}
         for ci in range(n_channels)],
        key=lambda x: x["grad_cam_score"], reverse=True)

    # Per-class channel heatmaps
    class_heatmaps = {}
    for cls_idx, cls_name in enumerate(["Normal", "Abnormal"]):
        mask = labels[:n_explain] == cls_idx
        if np.any(mask):
            cls_mean = np.mean(all_heatmaps[mask], axis=0).mean(axis=1)
            class_heatmaps[cls_name] = [
                {"channel": ch_names[ci], "activation": round(float(cls_mean[ci]), 6)}
                for ci in range(n_channels)
            ]

    # Feature-type sensitivity (column-wise mean of heatmap)
    feat_group_names = ["delta_power", "theta_power", "alpha_power", "beta_power",
                        "gamma_power", "line_length", "variance", "kurtosis"]
    feat_scores = mean_hm.mean(axis=0)  # (n_feat_per_ch,)
    feat_groups = sorted(
        [{"feature_type": feat_group_names[fi], "sensitivity": round(float(feat_scores[fi]), 6)}
         for fi in range(n_feat_per_ch)],
        key=lambda x: x["sensitivity"], reverse=True)

    return _json_clean({
        "available": True,
        "method": "Grad-CAM (Selvaraju et al. 2017)",
        "description": "Gradient-weighted Class Activation Mapping on a 2D CNN treating "
                       "EEG channels x feature types as a pseudo-image. Highlights which "
                       "channels and feature groups drive the classification decision.",
        "model": {
            "type": "2D CNN (Conv2d 1->16->32, AdaptiveAvgPool2d, FC->2)",
            "accuracy": accuracy,
            "n_params": sum(p.numel() for p in model.parameters()),
        },
        "channel_importance": channel_importance,
        "per_class_heatmaps": class_heatmaps,
        "feature_type_sensitivity": feat_groups,
        "sample_heatmaps": sample_heatmaps,
        "n_samples_explained": n_explain,
        "data": meta,
    })


def xai_definitions():
    """Method definitions and references for the XAI dashboard."""
    return {
        "methods": [
            {
                "name": "Integrated Gradients (Captum)",
                "citation": "Sundararajan, Taly and Yan (2017). Axiomatic Attribution for Deep Networks. ICML.",
                "principle": "Integrates gradients of the output w.r.t. input features along a straight-line path from a baseline to the input. Satisfies completeness axiom.",
                "strengths": ["Theoretically grounded (axioms)", "Works with any differentiable model", "No model retraining needed"],
                "limitations": ["Requires differentiable model", "Choice of baseline affects results", "Computationally expensive with many integration steps"],
                "clinical_use": "Identifies which EEG channels and frequency bands most influenced the classification for a specific patient epoch.",
            },
            {
                "name": "LIME",
                "citation": "Ribeiro, Singh and Guestrin (2016). Why Should I Trust You?: Explaining the Predictions of Any Classifier. KDD.",
                "principle": "Generates perturbed samples near the input, gets model predictions, fits a weighted sparse linear model to approximate the local decision boundary.",
                "strengths": ["Model-agnostic", "Human-interpretable linear weights", "Works with black-box APIs"],
                "limitations": ["Stochastic perturbation (results may vary)", "Fidelity depends on perturbation count", "Assumes local linearity"],
                "clinical_use": "Provides feature-level explanation for individual EEG epochs, showing which spectral features pushed toward normal vs abnormal.",
            },
            {
                "name": "SHAP",
                "citation": "Lundberg and Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.",
                "principle": "Computes Shapley values from cooperative game theory. TreeExplainer uses polynomial-time exact computation for tree-based models.",
                "strengths": ["Theoretically optimal (Shapley axioms)", "Global + local explanations", "TreeExplainer is exact and fast"],
                "limitations": ["KernelExplainer is slow for high-dim", "Correlated features can split importance", "Requires background dataset"],
                "clinical_use": "Global importance reveals which EEG features are most predictive overall; per-sample waterfall plots trace each classification.",
            },
            {
                "name": "Grad-CAM",
                "citation": "Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.",
                "principle": "Computes gradient of the target class score w.r.t. feature maps of a convolutional layer, then produces a weighted combination of those maps as a heatmap highlighting important spatial regions.",
                "strengths": ["No model retraining needed", "Produces intuitive visual heatmaps", "Works with any CNN architecture", "Class-discriminative"],
                "limitations": ["Requires convolutional layers", "Resolution limited to feature map size", "Can miss fine-grained details"],
                "clinical_use": "Visualizes which EEG channels (spatial regions) the CNN focuses on when classifying normal vs abnormal segments, aiding clinician trust and review.",
            },
        ],
        "eu_ai_act": {
            "article": "Art. 86 — Right to explanation of individual decision-making",
            "requirement": "Users affected by an AI decision have the right to obtain a clear and meaningful explanation.",
            "how_this_dashboard_helps": "Provides three independent explanation methods so clinicians can triangulate which EEG features drove the classification.",
        },
    }


if __name__ == "__main__":
    print("Testing XAI Dashboard...")
    ov = xai_overview()
    print(json.dumps(ov, indent=2, default=str)[:500])
    if ov.get("available"):
        print(f"\nModel accuracy: {ov['model']['accuracy']:.2%}")
        print(f"Top feature: {ov['top_features_shap'][0]['feature']}")
        print(f"Methods: {len(ov['methods'])}")
