"""Model Comparison Dashboard — backend analytics for model_comparison table."""
import sqlite3, os

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

def _conn():
    return sqlite3.connect(DB)

def overview():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    total_models = c.execute("SELECT COUNT(*) FROM model_comparison").fetchone()[0]
    distinct_model_types = c.execute("SELECT COUNT(DISTINCT model_type) FROM model_comparison").fetchone()[0]
    distinct_tasks = c.execute("SELECT COUNT(DISTINCT task) FROM model_comparison").fetchone()[0]
    distinct_datasets = c.execute("SELECT COUNT(DISTINCT dataset) FROM model_comparison").fetchone()[0]
    avg_accuracy = c.execute("SELECT ROUND(AVG(accuracy),4) FROM model_comparison WHERE status='completed'").fetchone()[0]
    avg_f1 = c.execute("SELECT ROUND(AVG(f1_score),4) FROM model_comparison WHERE status='completed'").fetchone()[0]
    avg_auc_roc = c.execute("SELECT ROUND(AVG(auc_roc),4) FROM model_comparison WHERE status='completed'").fetchone()[0]

    best_row = c.execute("SELECT accuracy, model_name FROM model_comparison WHERE status='completed' ORDER BY accuracy DESC LIMIT 1").fetchone()
    best_accuracy = best_row[0] if best_row else None
    best_accuracy_model = best_row[1] if best_row else None

    completed_count = c.execute("SELECT COUNT(*) FROM model_comparison WHERE status='completed'").fetchone()[0]
    failed_count = c.execute("SELECT COUNT(*) FROM model_comparison WHERE status='failed'").fetchone()[0]

    model_type_dist = [dict(r) for r in c.execute(
        "SELECT model_type, COUNT(*) AS count FROM model_comparison GROUP BY model_type ORDER BY count DESC")]

    task_dist = [dict(r) for r in c.execute(
        "SELECT task, COUNT(*) AS count FROM model_comparison GROUP BY task ORDER BY count DESC")]

    dataset_dist = [dict(r) for r in c.execute(
        "SELECT dataset, COUNT(*) AS count FROM model_comparison GROUP BY dataset ORDER BY count DESC")]

    status_dist = [dict(r) for r in c.execute(
        "SELECT status, COUNT(*) AS count FROM model_comparison GROUP BY status ORDER BY count DESC")]

    accuracy_by_model_type = [dict(r) for r in c.execute("""
        SELECT model_type,
               ROUND(AVG(accuracy),4) AS avg_accuracy,
               ROUND(AVG(f1_score),4) AS avg_f1,
               ROUND(AVG(auc_roc),4) AS avg_auc_roc
        FROM model_comparison WHERE status='completed'
        GROUP BY model_type ORDER BY avg_accuracy DESC
    """)]

    accuracy_by_task = [dict(r) for r in c.execute("""
        SELECT task,
               ROUND(AVG(accuracy),4) AS avg_accuracy,
               ROUND(AVG(f1_score),4) AS avg_f1,
               ROUND(AVG(auc_roc),4) AS avg_auc_roc
        FROM model_comparison WHERE status='completed'
        GROUP BY task ORDER BY avg_accuracy DESC
    """)]

    monthly_trend = [dict(r) for r in c.execute("""
        SELECT SUBSTR(created_at,1,7) AS month,
               COUNT(*) AS count,
               ROUND(AVG(accuracy),4) AS avg_accuracy
        FROM model_comparison GROUP BY month ORDER BY month
    """)]

    training_method_dist = [dict(r) for r in c.execute(
        "SELECT trained_by, COUNT(*) AS count FROM model_comparison GROUP BY trained_by ORDER BY count DESC")]

    conn.close()
    return {
        "kpis": {
            "total_models": total_models,
            "distinct_model_types": distinct_model_types,
            "distinct_tasks": distinct_tasks,
            "distinct_datasets": distinct_datasets,
            "avg_accuracy": avg_accuracy,
            "avg_f1": avg_f1,
            "avg_auc_roc": avg_auc_roc,
            "best_accuracy": best_accuracy,
            "best_accuracy_model": best_accuracy_model,
            "completed_count": completed_count,
            "failed_count": failed_count,
        },
        "model_type_dist": model_type_dist,
        "task_dist": task_dist,
        "dataset_dist": dataset_dist,
        "status_dist": status_dist,
        "accuracy_by_model_type": accuracy_by_model_type,
        "accuracy_by_task": accuracy_by_task,
        "monthly_trend": monthly_trend,
        "training_method_dist": training_method_dist,
    }

def breakdown():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    models = [dict(r) for r in c.execute(
        "SELECT * FROM model_comparison ORDER BY accuracy DESC LIMIT 300")]

    by_model_type = [dict(r) for r in c.execute("""
        SELECT model_type,
               COUNT(*) AS count,
               ROUND(AVG(accuracy),4) AS avg_accuracy,
               ROUND(AVG(f1_score),4) AS avg_f1,
               ROUND(AVG(auc_roc),4) AS avg_auc_roc,
               ROUND(AVG(training_time_sec),2) AS avg_training_time,
               ROUND(AVG(inference_time_ms),2) AS avg_inference_time,
               MAX(accuracy) AS best_accuracy
        FROM model_comparison WHERE status='completed'
        GROUP BY model_type ORDER BY avg_accuracy DESC
    """)]

    by_dataset = [dict(r) for r in c.execute("""
        SELECT dataset,
               COUNT(*) AS count,
               ROUND(AVG(accuracy),4) AS avg_accuracy,
               ROUND(AVG(f1_score),4) AS avg_f1,
               ROUND(AVG(auc_roc),4) AS avg_auc_roc,
               SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS models_completed,
               SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS models_failed
        FROM model_comparison GROUP BY dataset ORDER BY avg_accuracy DESC
    """)]

    conn.close()
    return {
        "models": models,
        "by_model_type": by_model_type,
        "by_dataset": by_dataset,
    }

def definitions():
    return {
        "title": "Model Comparison Dashboard — Definitions",
        "concepts": [
            {"name": "Model Type", "description": "The machine learning algorithm family used: RandomForest, XGBoost, LightGBM, SVM, MLP, LogisticRegression. Determines training characteristics, interpretability, and computational cost."},
            {"name": "Task", "description": "The clinical prediction task the model was trained for: seizure_detection, eeg_classification, anomaly_detection, seizure_prediction. Each task has distinct evaluation criteria and clinical relevance."},
            {"name": "Dataset", "description": "The EEG dataset used for training and evaluation: bonn_eeg, chb_mit, tuh_eeg, internal_clinical. Dataset choice affects generalisability and benchmark comparability."},
            {"name": "Accuracy", "description": "Proportion of correctly classified samples out of total samples. Primary performance metric, but can be misleading on imbalanced datasets."},
            {"name": "Precision", "description": "Proportion of true positives among all predicted positives. High precision means fewer false alarms, critical for clinical alerting systems."},
            {"name": "Recall", "description": "Proportion of true positives among all actual positives (sensitivity). High recall means fewer missed events, essential for seizure detection safety."},
            {"name": "F1 Score", "description": "Harmonic mean of precision and recall, balancing both metrics. Preferred over accuracy when class distribution is imbalanced."},
            {"name": "AUC-ROC", "description": "Area Under the Receiver Operating Characteristic curve. Measures discrimination ability across all classification thresholds; 1.0 = perfect, 0.5 = random."},
            {"name": "Training Time", "description": "Wall-clock time in seconds to train the model. Includes hyperparameter search when applicable. Affects iteration speed and computational cost."},
            {"name": "Inference Time", "description": "Time in milliseconds to generate a single prediction. Critical for real-time seizure detection where latency constraints apply."},
            {"name": "Hyperparams", "description": "JSON-encoded hyperparameter configuration used for the training run. Enables reproducibility and comparison across tuning strategies."},
            {"name": "Status", "description": "Training run outcome: completed (successful training and evaluation) or failed (error during training, data issue, or timeout)."},
            {"name": "Trained By", "description": "The tuning method used: grid_search (exhaustive), hyperopt (Bayesian optimisation), auto_pipeline (automated ML), manual_tuning (expert-configured). Affects result quality and compute cost."},
        ],
        "data_sources": [
            "model_comparison table — 224 model runs, 6 model types, 4 tasks, 4 datasets",
        ],
    }

if __name__ == "__main__":
    import json
    print(json.dumps(overview(), indent=2))
