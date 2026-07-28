"""System Configuration Dashboard — backend analytics for system_config table."""
import sqlite3, os

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

def _conn():
    return sqlite3.connect(DB)

def overview():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    total = c.execute("SELECT COUNT(*) FROM system_config").fetchone()[0]
    categories = c.execute("SELECT COUNT(DISTINCT category) FROM system_config").fetchone()[0]
    updaters = c.execute("SELECT COUNT(DISTINCT updated_by) FROM system_config").fetchone()[0]
    latest_update = c.execute("SELECT MAX(updated_at) FROM system_config").fetchone()[0]
    oldest_update = c.execute("SELECT MIN(updated_at) FROM system_config").fetchone()[0]

    # Boolean configs (true/false values)
    bool_true = c.execute("SELECT COUNT(*) FROM system_config WHERE LOWER(value) = 'true'").fetchone()[0]
    bool_false = c.execute("SELECT COUNT(*) FROM system_config WHERE LOWER(value) = 'false'").fetchone()[0]
    numeric = total - bool_true - bool_false

    # Category distribution
    category_distribution = [dict(r) for r in c.execute(
        "SELECT category, COUNT(*) AS count FROM system_config GROUP BY category ORDER BY count DESC")]

    # Updater distribution
    updater_distribution = [dict(r) for r in c.execute(
        "SELECT updated_by AS updater, COUNT(*) AS count FROM system_config GROUP BY updated_by ORDER BY count DESC")]

    # Recent changes (last 5)
    recent_changes = [dict(r) for r in c.execute(
        "SELECT config_id, key, value, category, description, updated_at, updated_by FROM system_config ORDER BY updated_at DESC LIMIT 5")]

    # Config freshness: how many updated in each month
    monthly_updates = [dict(r) for r in c.execute("""
        SELECT SUBSTR(updated_at,1,7) AS month, COUNT(*) AS count
        FROM system_config GROUP BY month ORDER BY month
    """)]

    # Security configs summary
    security_configs = [dict(r) for r in c.execute(
        "SELECT key, value, description FROM system_config WHERE category = 'Security' ORDER BY key")]

    # AI configs summary
    ai_configs = [dict(r) for r in c.execute(
        "SELECT key, value, description FROM system_config WHERE category = 'AI' ORDER BY key")]

    # Performance configs summary
    perf_configs = [dict(r) for r in c.execute(
        "SELECT key, value, description FROM system_config WHERE category = 'Performance' ORDER BY key")]

    conn.close()
    return {
        "total_configs": total,
        "total_categories": categories,
        "total_updaters": updaters,
        "latest_update": latest_update,
        "oldest_update": oldest_update,
        "boolean_true": bool_true,
        "boolean_false": bool_false,
        "numeric_configs": numeric,
        "category_distribution": category_distribution,
        "updater_distribution": updater_distribution,
        "recent_changes": recent_changes,
        "monthly_updates": monthly_updates,
        "security_configs": security_configs,
        "ai_configs": ai_configs,
        "performance_configs": perf_configs,
    }

def breakdown():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    configs = [dict(r) for r in c.execute(
        "SELECT * FROM system_config ORDER BY category, key")]

    # By category with details
    by_category = []
    for row in c.execute("SELECT DISTINCT category FROM system_config ORDER BY category"):
        cat = row["category"]
        items = [dict(r) for r in c.execute(
            "SELECT config_id, key, value, description, updated_at, updated_by FROM system_config WHERE category=? ORDER BY key",
            (cat,))]
        by_category.append({"category": cat, "count": len(items), "items": items})

    # By updater
    by_updater = [dict(r) for r in c.execute("""
        SELECT updated_by AS updater, COUNT(*) AS configs_managed,
               MAX(updated_at) AS last_update
        FROM system_config GROUP BY updated_by ORDER BY configs_managed DESC
    """)]

    conn.close()
    return {
        "configs": configs,
        "by_category": by_category,
        "by_updater": by_updater,
    }

def definitions():
    return {
        "title": "System Configuration Dashboard — Definitions",
        "concepts": [
            {"name": "System Configuration", "description": "Key-value settings that control the behavior of the clinical platform, including security policies, AI model parameters, data retention rules, notification preferences, and performance tuning. Each configuration has a category, description, and audit trail (who changed it and when)."},
            {"name": "Category", "description": "Logical grouping of configuration settings. Categories include Security (auth, rate limits), AI (model thresholds, retraining), Data (retention, uploads, sampling), Notification (email, SMS, webhooks), Performance (caching, connection pools, timeouts), and General (session, logging, maintenance)."},
            {"name": "Config Freshness", "description": "How recently a configuration was last updated. Stale configs may indicate settings that haven't been reviewed in accordance with best-practice review cycles. Monthly update trends show configuration management activity over time."},
            {"name": "Boolean vs Numeric", "description": "Configs are classified as boolean (true/false toggles like maintenance_mode, email_notification_enabled) or numeric (threshold values, timeouts, limits). Boolean configs control feature enablement; numeric configs tune operational parameters."},
            {"name": "Updater", "description": "The admin or engineer who last modified a configuration setting. Tracking updaters supports accountability and audit compliance (e.g., who changed the rate limit, when, and why)."},
        ],
        "categories": [
            {"category": "Security", "description": "Authentication, authorization, rate limiting, password policies, MFA, JWT expiry."},
            {"category": "AI", "description": "Model confidence thresholds, inference timeouts, drift detection, auto-retraining triggers."},
            {"category": "Data", "description": "EEG sampling rates, upload size limits, data retention periods, backup frequency."},
            {"category": "Notification", "description": "Email/SMS alerting toggles, webhook retries, alert cooldown periods."},
            {"category": "Performance", "description": "Cache TTL, connection pool sizing, concurrency limits, query timeouts."},
            {"category": "General", "description": "Session management, logging levels, maintenance mode."},
        ],
        "data_sources": [
            "system_config table — 25 configuration entries across 6 categories",
            "Managed by platform administrators with full audit trail",
        ],
    }

if __name__ == "__main__":
    import json
    print(json.dumps(overview(), indent=2))
