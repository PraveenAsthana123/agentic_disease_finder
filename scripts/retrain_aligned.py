#!/usr/bin/env python3
"""Fix train/serve skew: retrain epilepsy model + rebuild reference npz using the SAME
feature extractor that SERVES (eeg_analysis_pipeline.extract_features), so training and
inference distributions match (drift drops). Subject-wise CV (no leakage). Honest accuracy.

CAVEAT (disclosed in metrics): control class = healthy motor-imagery recordings, which
differ from CHB-MIT epilepsy in montage/sfreq/channel-0 — so the model partly separates
DATASET, not only epilepsy. The PRIMARY goal here is pipeline ALIGNMENT (kill the skew);
clinically meaningful accuracy needs same-setup ictal/interictal data (follow-up)."""
from __future__ import annotations
import glob, re, sys, warnings
from datetime import datetime, timezone
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def subj(fp, kind):
    b = Path(fp).name
    if kind == "epi":
        m = re.match(r"(chb\d+)", b); return m.group(1) if m else b
    m = re.match(r"(S\d+|PN\d+)", b); return m.group(1) if m else b


def collect(files, kind, label, per_subj, extractor, cap):
    import numpy as np
    seen, X, y, S = {}, [], [], []
    for fp in files:
        s = subj(fp, kind)
        if seen.get(s, 0) >= per_subj:
            continue
        try:
            data, sf, _ = extractor_parse(fp)
            feat = extractor(data, sf)
            if feat is None or len(feat) != 47:
                continue
            X.append(np.asarray(feat, float)); y.append(label); S.append(s)
            seen[s] = seen.get(s, 0) + 1
        except Exception:
            continue
        if len(X) >= cap:
            break
    return X, y, S


def extractor_parse(fp):
    import eeg_analysis_pipeline as eeg
    return eeg.parse_eeg(fp)


def main():
    import numpy as np, joblib
    import eeg_analysis_pipeline as eeg
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    epi = sorted(glob.glob(str(ROOT / "data/real_eeg/epilepsy_physionet/*.edf"))) + \
          sorted(glob.glob(str(ROOT / "data/real_eeg/siena_epilepsy_physionet/**/*.edf"), recursive=True))
    ctl = sorted(glob.glob(str(ROOT / "data/real_eeg/motor_imagery_physionet/**/*.edf"), recursive=True))
    print(f"available: {len(epi)} epilepsy, {len(ctl)} control EDFs")

    cache = Path("/tmp/align_feats.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True); X, y, groups = z["X"], z["y"], z["groups"]
        print("loaded cached features", X.shape)
    else:
        Xe, ye, Se = collect(epi, "epi", 1, 10, eeg.extract_features, 60)
        Xc, yc, Sc = collect(ctl, "ctl", 0, 6, eeg.extract_features, 60)
        X = np.array(Xe + Xc); y = np.array(ye + yc)
        groups = np.array([f"e_{s}" for s in Se] + [f"c_{s}" for s in Sc])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        np.savez(cache, X=X, y=y, groups=groups)
    print(f"dataset: {X.shape}, epilepsy={sum(y)}, control={len(y)-sum(y)}, subjects={len(set(groups))}")
    if len(set(y)) < 2 or X.shape[0] < 20:
        print("insufficient data"); return 1

    base = lambda: VotingClassifier(estimators=[
        ("et", ExtraTreesClassifier(n_estimators=200, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("gb", GradientBoostingClassifier(random_state=42))], voting="soft")

    n_split = min(5, min(int((y==0).sum()), int((y==1).sum()), len(set(groups))))
    n_split = max(2, n_split)
    gkf = StratifiedGroupKFold(n_splits=n_split)
    yp = cross_val_predict(base(), X, y, groups=groups, cv=gkf, method="predict")
    ypp = cross_val_predict(base(), X, y, groups=groups, cv=gkf, method="predict_proba")[:, 1]
    acc = accuracy_score(y, yp); f1 = f1_score(y, yp); auc = roc_auc_score(y, ypp)
    print(f"SUBJECT-WISE CV: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

    # final fit on all data
    model = base().fit(X, y)
    feat_names = ["mean","std","var","min","max","median","ptp","skewness","kurtosis","q25","q75","rms","mav",
                  "line_length","zero_crossings","delta_power","theta_power","alpha_power","beta_power","gamma_power",
                  "total_power","dominant_freq","spectral_entropy","psd_std","psd_mean","psd_median","psd_q10","psd_q90",
                  "peak_ratio","spectral_flatness","spectral_centroid","spectral_bandwidth","spectral_rolloff",
                  "mean_abs_diff","std_diff","max_diff","hjorth_mobility","hjorth_complexity","autocorr","slope_changes",
                  "trend","crest_factor","approx_entropy","sample_entropy","hurst_exponent","dfa_alpha","lz_complexity"]
    bundle = {
        "model": model, "class_names": ["Control", "Epilepsy"], "n_features": 47,
        "training_date": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "metrics": {"subject_wise_cv_accuracy": round(float(acc), 4), "f1": round(float(f1), 4),
                    "auc": round(float(auc), 4), "n_samples": int(X.shape[0]),
                    "n_subjects": int(len(set(groups))), "cv": f"GroupKFold-{n_split} (subject-wise)"},
        "caveat": "Control=healthy motor-imagery (different montage/sfreq). Model partly separates dataset, "
                  "not only epilepsy. Pipeline-alignment fix; clinically meaningful accuracy needs same-setup "
                  "ictal/interictal data. Extractor=eeg_analysis_pipeline.extract_features (matches serving).",
    }
    out = ROOT / "models" / "epilepsy_model.joblib"
    out.parent.mkdir(exist_ok=True)
    # back up old model
    if out.exists():
        out.rename(out.with_suffix(".joblib.pre_align_bak"))
    joblib.dump(bundle, out)
    print(f"saved {out} (backed up old → .pre_align_bak)")

    # rebuild reference npz with SAME extractor (SHAP background + drift baseline now in-distribution)
    npz = ROOT / "data" / "epilepsy" / "sample" / "epilepsy_sample_100.npz"
    if npz.exists():
        npz.rename(npz.with_suffix(".npz.pre_align_bak"))
    sids = np.array([abs(hash(g)) % 100000 for g in groups])
    np.savez(npz, X=X, y=y, subject_ids=sids,
             feature_names=np.array(feat_names), class_names=np.array(["Control", "Epilepsy"]))
    print(f"rebuilt {npz} with serving extractor ({X.shape[0]} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
