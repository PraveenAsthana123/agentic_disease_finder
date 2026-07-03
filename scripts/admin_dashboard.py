"""Admin Dashboard — user management, feature flags, system health monitoring,
and configuration management for the NeuroLab clinical EEG AI platform.

Populates and reads from:
  - admin_users        (system users: roles, MFA, login tracking)
  - feature_flags      (feature toggles: rollout, ownership, categories)
  - system_health_log  (health checks: response time, CPU, memory, disk)
  - system_config      (platform configuration: keys, values, categories)

Uses deterministic seed(99) for reproducible sample data generation.
"""

import os
import random
import sqlite3

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

random.seed(99)

ROLES = ['Admin', 'Neurologist', 'EEG Tech', 'Researcher', 'Nurse', 'Data Scientist', 'IT Support']
USER_STATUSES = ['active', 'inactive', 'suspended']
FLAG_CATEGORIES = ['AI Model', 'UI', 'Data Pipeline', 'Security', 'Integration', 'Reporting']
HEALTH_COMPONENTS = ['API', 'Database', 'ML Pipeline', 'Frontend', 'Cache', 'Queue', 'Storage']
HEALTH_STATUSES = ['healthy', 'degraded', 'down']
CONFIG_CATEGORIES = ['General', 'Security', 'AI', 'Data', 'Notification', 'Performance']

DEPARTMENTS = [
    'Neurology', 'IT', 'Research', 'Nursing', 'Data Science',
    'Clinical Engineering', 'Administration', 'Radiology',
]

FIRST_NAMES = [
    'Aisha', 'Brian', 'Carmen', 'David', 'Elena', 'Farhan', 'Grace',
    'Hiroshi', 'Ingrid', 'Jamal', 'Katrina', 'Liam', 'Mei', 'Naveen', 'Olga',
]

LAST_NAMES = [
    'Patel', 'Chen', 'Rodriguez', 'Nakamura', 'Schmidt', 'Okafor', 'Williams',
    'Johansson', 'Hassan', 'Kim', 'Fernandez', 'Singh', 'Ivanova', 'Brown', 'Tanaka',
]

FLAG_NAMES = [
    'seizure_detection_v2', 'dark_mode', 'real_time_eeg_stream', 'mfa_enforcement',
    'auto_report_generation', 'federated_learning', 'patient_portal_v3',
    'advanced_spectral_analysis', 'slack_integration', 'audit_log_export',
    'gpu_inference_offload', 'hipaa_enhanced_logging', 'batch_processing',
    'anomaly_alerting', 'onnx_model_serving', 'data_lake_sync',
    'role_based_dashboards', 'eeg_artifact_rejection', 'consent_management',
    'predictive_seizure_risk',
]

FLAG_DESCRIPTIONS = [
    'Next-gen seizure detection model with transformer architecture',
    'Dark mode theme for clinical workstations',
    'Real-time EEG streaming with sub-100ms latency',
    'Enforce MFA for all user accounts',
    'Automated clinical report generation from EEG analysis',
    'Multi-site federated learning aggregation',
    'Redesigned patient-facing portal with self-service features',
    'Advanced spectral analysis with wavelet decomposition',
    'Slack integration for clinical alerts and notifications',
    'Export audit logs to external SIEM systems',
    'Offload ML inference to GPU cluster',
    'Enhanced HIPAA-compliant logging with PHI masking',
    'Batch processing pipeline for overnight EEG studies',
    'Automated anomaly detection and alerting system',
    'ONNX model serving for cross-platform inference',
    'Synchronize processed data to cloud data lake',
    'Role-specific dashboard views and permissions',
    'Automated EEG artifact rejection preprocessing',
    'Patient consent management and tracking workflow',
    'Predictive seizure risk scoring model',
]

CONFIG_KEYS = [
    ('session_timeout_minutes', '30', 'General', 'User session timeout in minutes'),
    ('max_login_attempts', '5', 'Security', 'Max failed login attempts before lockout'),
    ('password_min_length', '12', 'Security', 'Minimum password length requirement'),
    ('mfa_grace_period_days', '7', 'Security', 'Days before MFA enforcement after account creation'),
    ('model_inference_timeout_ms', '5000', 'AI', 'Maximum time for model inference response'),
    ('model_confidence_threshold', '0.85', 'AI', 'Minimum confidence score for seizure detection'),
    ('eeg_sample_rate_hz', '256', 'Data', 'Default EEG sampling rate in Hz'),
    ('max_upload_size_mb', '500', 'Data', 'Maximum file upload size in megabytes'),
    ('data_retention_days', '2555', 'Data', 'Data retention period in days (7 years)'),
    ('email_notification_enabled', 'true', 'Notification', 'Enable email notifications'),
    ('sms_alerting_enabled', 'false', 'Notification', 'Enable SMS alerts for critical events'),
    ('alert_cooldown_minutes', '15', 'Notification', 'Minimum time between repeated alerts'),
    ('cache_ttl_seconds', '300', 'Performance', 'Default cache TTL in seconds'),
    ('max_concurrent_requests', '100', 'Performance', 'Maximum concurrent API requests'),
    ('db_connection_pool_size', '20', 'Performance', 'Database connection pool size'),
    ('log_level', 'INFO', 'General', 'Application logging level'),
    ('maintenance_mode', 'false', 'General', 'Enable maintenance mode (disables user access)'),
    ('api_rate_limit_per_minute', '60', 'Security', 'API rate limit per user per minute'),
    ('jwt_expiry_hours', '24', 'Security', 'JWT token expiry in hours'),
    ('model_auto_retrain_enabled', 'true', 'AI', 'Enable automatic model retraining on drift'),
    ('drift_threshold', '0.15', 'AI', 'Drift score threshold triggering retraining'),
    ('backup_frequency_hours', '6', 'Data', 'Automated backup frequency in hours'),
    ('webhook_retry_count', '3', 'Notification', 'Number of webhook delivery retries'),
    ('query_timeout_seconds', '30', 'Performance', 'Database query timeout in seconds'),
    ('feature_flag_cache_seconds', '60', 'Performance', 'Feature flag evaluation cache duration'),
]


def _db_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = _db_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _avg(values):
    return round(sum(values) / len(values), 2) if values else 0


def _ensure_tables():
    """Create and populate admin tables if they don't exist."""
    if not os.path.exists(DB):
        os.makedirs(os.path.dirname(DB), exist_ok=True)

    conn = _db_conn()
    try:
        # Create tables
        conn.execute('''CREATE TABLE IF NOT EXISTS admin_users (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            full_name TEXT,
            email TEXT,
            role TEXT,
            status TEXT,
            last_login TEXT,
            created_at TEXT,
            login_count INTEGER,
            mfa_enabled INTEGER,
            department TEXT
        )''')

        conn.execute('''CREATE TABLE IF NOT EXISTS feature_flags (
            flag_id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            enabled INTEGER,
            rollout_percentage INTEGER,
            created_at TEXT,
            updated_at TEXT,
            owner TEXT,
            category TEXT
        )''')

        conn.execute('''CREATE TABLE IF NOT EXISTS system_health_log (
            check_id INTEGER PRIMARY KEY,
            timestamp TEXT,
            component TEXT,
            status TEXT,
            response_time_ms INTEGER,
            cpu_pct REAL,
            memory_pct REAL,
            disk_pct REAL,
            error_count INTEGER
        )''')

        conn.execute('''CREATE TABLE IF NOT EXISTS system_config (
            config_id TEXT PRIMARY KEY,
            key TEXT,
            value TEXT,
            category TEXT,
            description TEXT,
            updated_at TEXT,
            updated_by TEXT
        )''')

        conn.commit()

        # Check if already populated
        count = conn.execute('SELECT COUNT(*) FROM admin_users').fetchone()[0]
        if count > 0:
            conn.close()
            return

        rng = random.Random(99)

        # --- Populate admin_users (15 users) ---
        usernames = []
        for i in range(15):
            first = FIRST_NAMES[i]
            last = LAST_NAMES[i]
            username = f'{first.lower()}.{last.lower()}'
            usernames.append(username)
            user_id = f'USR-{i + 1:04d}'
            full_name = f'{first} {last}'
            email = f'{first.lower()}.{last.lower()}@neurolab.health'
            role = rng.choice(ROLES)
            status = rng.choices(USER_STATUSES, weights=[0.70, 0.20, 0.10])[0]
            created_month = rng.randint(1, 12)
            created_day = rng.randint(1, 28)
            created_at = f'2024-{created_month:02d}-{created_day:02d}T{rng.randint(8, 17):02d}:{rng.randint(0, 59):02d}:00'
            login_month = rng.randint(1, 6)
            login_day = rng.randint(1, 28)
            last_login = f'2025-{login_month:02d}-{login_day:02d}T{rng.randint(6, 22):02d}:{rng.randint(0, 59):02d}:00'
            login_count = rng.randint(5, 350)
            mfa_enabled = rng.choices([1, 0], weights=[0.7, 0.3])[0]
            department = rng.choice(DEPARTMENTS)

            conn.execute(
                'INSERT INTO admin_users VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (user_id, username, full_name, email, role, status, last_login,
                 created_at, login_count, mfa_enabled, department)
            )

        # --- Populate feature_flags (20 flags) ---
        for i in range(20):
            flag_id = f'FLAG-{i + 1:04d}'
            name = FLAG_NAMES[i]
            description = FLAG_DESCRIPTIONS[i]
            enabled = rng.choices([1, 0], weights=[0.6, 0.4])[0]
            rollout_percentage = rng.choice([0, 10, 25, 50, 75, 100]) if enabled else 0
            created_month = rng.randint(1, 12)
            created_at = f'2024-{created_month:02d}-{rng.randint(1, 28):02d}T{rng.randint(8, 17):02d}:{rng.randint(0, 59):02d}:00'
            updated_month = rng.randint(1, 6)
            updated_at = f'2025-{updated_month:02d}-{rng.randint(1, 28):02d}T{rng.randint(8, 17):02d}:{rng.randint(0, 59):02d}:00'
            owner = rng.choice(usernames)
            category = rng.choice(FLAG_CATEGORIES)

            conn.execute(
                'INSERT INTO feature_flags VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (flag_id, name, description, enabled, rollout_percentage,
                 created_at, updated_at, owner, category)
            )

        # --- Populate system_health_log (30 entries) ---
        for i in range(30):
            check_id = i + 1
            day = max(1, min(28, i + 1))
            hour = rng.randint(0, 23)
            minute = rng.randint(0, 59)
            timestamp = f'2025-06-{day:02d}T{hour:02d}:{minute:02d}:00'
            component = rng.choice(HEALTH_COMPONENTS)
            status = rng.choices(HEALTH_STATUSES, weights=[0.75, 0.20, 0.05])[0]
            response_time_ms = rng.randint(12, 850) if status != 'down' else rng.randint(5000, 30000)
            cpu_pct = round(rng.uniform(15.0, 92.0), 1)
            memory_pct = round(rng.uniform(30.0, 88.0), 1)
            disk_pct = round(rng.uniform(40.0, 85.0), 1)
            error_count = 0 if status == 'healthy' else rng.randint(1, 50)

            conn.execute(
                'INSERT INTO system_health_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (check_id, timestamp, component, status, response_time_ms,
                 cpu_pct, memory_pct, disk_pct, error_count)
            )

        # --- Populate system_config (25 entries) ---
        for i, (key, value, category, desc) in enumerate(CONFIG_KEYS):
            config_id = f'CFG-{i + 1:04d}'
            updated_month = rng.randint(1, 6)
            updated_at = f'2025-{updated_month:02d}-{rng.randint(1, 28):02d}T{rng.randint(8, 17):02d}:{rng.randint(0, 59):02d}:00'
            updated_by = rng.choice(usernames)

            conn.execute(
                'INSERT INTO system_config VALUES (?, ?, ?, ?, ?, ?, ?)',
                (config_id, key, value, category, desc, updated_at, updated_by)
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def overview():
    """Return KPI cards + chart data for the Admin Dashboard overview tab."""
    _ensure_tables()

    users = _db_query('SELECT * FROM admin_users')
    flags = _db_query('SELECT * FROM feature_flags')
    health = _db_query('SELECT * FROM system_health_log')
    configs = _db_query('SELECT * FROM system_config')

    # KPIs
    total_users = len(users)
    active_users = sum(1 for u in users if u.get('status') == 'active')
    mfa_count = sum(1 for u in users if u.get('mfa_enabled'))
    mfa_adoption_pct = round((mfa_count / total_users) * 100, 1) if total_users else 0
    flags_enabled = sum(1 for f in flags if f.get('enabled'))
    flags_disabled = sum(1 for f in flags if not f.get('enabled'))
    response_times = [h.get('response_time_ms', 0) for h in health]
    avg_response_time_ms = _avg(response_times)
    healthy_checks = sum(1 for h in health if h.get('status') == 'healthy')
    system_uptime_pct = round((healthy_checks / len(health)) * 100, 1) if health else 0
    config_entries = len(configs)

    # Users by role
    from collections import Counter
    role_counts = Counter(u.get('role', '') for u in users)
    users_by_role = [{'role': r, 'count': role_counts.get(r, 0)} for r in ROLES if role_counts.get(r, 0) > 0]

    # Users by status
    status_counts = Counter(u.get('status', '') for u in users)
    users_by_status = [{'status': s, 'count': status_counts.get(s, 0)} for s in USER_STATUSES if status_counts.get(s, 0) > 0]

    # Flags by category
    cat_counts = Counter(f.get('category', '') for f in flags)
    flags_by_category = [{'category': c, 'count': cat_counts.get(c, 0)} for c in FLAG_CATEGORIES if cat_counts.get(c, 0) > 0]

    # Health trend (last 30 checks ordered by timestamp)
    health_sorted = sorted(health, key=lambda h: h.get('timestamp', ''))
    health_trend = [
        {
            'timestamp': h.get('timestamp', ''),
            'response_time_ms': h.get('response_time_ms', 0),
            'cpu_pct': h.get('cpu_pct', 0),
            'memory_pct': h.get('memory_pct', 0),
        }
        for h in health_sorted
    ]

    # Component health summary
    comp_data = {}
    for h in health:
        comp = h.get('component', '')
        if comp not in comp_data:
            comp_data[comp] = {'response_times': [], 'errors': 0, 'total': 0}
        comp_data[comp]['response_times'].append(h.get('response_time_ms', 0))
        comp_data[comp]['errors'] += h.get('error_count', 0)
        comp_data[comp]['total'] += 1

    component_health = [
        {
            'component': comp,
            'avg_response_time': _avg(d['response_times']),
            'error_rate': round(d['errors'] / max(d['total'], 1), 2),
        }
        for comp, d in sorted(comp_data.items())
    ]

    return {
        'kpis': {
            'total_users': total_users,
            'active_users': active_users,
            'mfa_adoption_pct': mfa_adoption_pct,
            'flags_enabled': flags_enabled,
            'flags_disabled': flags_disabled,
            'avg_response_time_ms': avg_response_time_ms,
            'system_uptime_pct': system_uptime_pct,
            'config_entries': config_entries,
        },
        'users_by_role': users_by_role,
        'users_by_status': users_by_status,
        'flags_by_category': flags_by_category,
        'health_trend': health_trend,
        'component_health': component_health,
    }


def breakdown():
    """Return full records for users, flags, health checks, and configs."""
    _ensure_tables()

    users = _db_query('SELECT * FROM admin_users')
    flags = _db_query('SELECT * FROM feature_flags')
    health_checks = _db_query('SELECT * FROM system_health_log ORDER BY timestamp')
    configs = _db_query('SELECT * FROM system_config')

    return {
        'users': users,
        'flags': flags,
        'health_checks': health_checks,
        'configs': configs,
    }


def definitions():
    """Return admin terminology definitions."""
    return [
        {
            'title': 'User Management',
            'description': 'The process of creating, modifying, and deactivating user accounts within the NeuroLab platform. Includes role assignment (Admin, Neurologist, EEG Tech, Researcher, Nurse, Data Scientist, IT Support), access control, login monitoring, and account lifecycle management. All user operations are logged for HIPAA audit trail compliance.',
        },
        {
            'title': 'Feature Flag',
            'description': 'A configuration toggle that enables or disables specific platform functionality without requiring code deployment. Feature flags allow gradual rollouts, A/B testing, and instant kill-switch capability for new features. Each flag has an owner, category, and rollout percentage controlling what fraction of users see the feature.',
        },
        {
            'title': 'Rollout Percentage',
            'description': 'The percentage of eligible users (0-100%) who will see a feature when its flag is enabled. Allows progressive delivery: start at 10% to validate stability, increase to 50% for broader testing, then 100% for general availability. Users are assigned deterministically based on user ID hash to ensure consistent experience.',
        },
        {
            'title': 'MFA (Multi-Factor Authentication)',
            'description': 'A security mechanism requiring two or more verification factors for user authentication. In the NeuroLab platform, MFA combines password (knowledge factor) with TOTP authenticator app or hardware security key (possession factor). Required for Admin and elevated-privilege roles. HIPAA Security Rule recommends MFA for all systems containing PHI.',
        },
        {
            'title': 'System Health',
            'description': 'A real-time assessment of platform component availability and performance. Monitors API, Database, ML Pipeline, Frontend, Cache, Queue, and Storage subsystems. Each component reports status (healthy/degraded/down), response time, and error counts. Health checks run every 60 seconds with automated alerting on degradation.',
        },
        {
            'title': 'Response Time',
            'description': 'The elapsed time in milliseconds between a request being received and its response being sent. Measured at the application layer for each component. Target SLAs: API < 200ms (p95), Database < 50ms (p95), ML Pipeline < 5000ms (p95). Response times exceeding thresholds trigger performance alerts and auto-scaling events.',
        },
        {
            'title': 'CPU Utilization',
            'description': 'The percentage of available CPU compute capacity currently in use by a platform component. Sustained utilization above 80% indicates potential performance degradation and triggers horizontal scaling. ML Pipeline components may spike to 90%+ during model inference batches, which is expected and time-bounded.',
        },
        {
            'title': 'Memory Utilization',
            'description': 'The percentage of allocated RAM currently consumed by a platform component. Includes heap memory, cached data, and buffer pools. Memory leaks are detected when utilization trends upward without corresponding load increase. Database and Cache components typically operate at 60-80% by design for optimal performance.',
        },
        {
            'title': 'Configuration Management',
            'description': 'The systematic process of maintaining platform settings, thresholds, and operational parameters. All configuration changes are versioned, attributed to a user, and timestamped for audit purposes. Categories include General, Security, AI, Data, Notification, and Performance. Changes to Security and AI configs require dual approval.',
        },
        {
            'title': 'Audit Trail',
            'description': 'A chronological record of all system activities, user actions, and configuration changes. Required by HIPAA for accountability and non-repudiation. Captures who did what, when, from where (IP), and the before/after state of any modification. Audit logs are immutable, encrypted at rest, and retained for a minimum of 6 years per HIPAA requirements.',
        },
    ]


# Ensure tables exist on module import
_ensure_tables()


if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    pprint.pprint(overview())
    print('\n=== BREAKDOWN (users) ===')
    bd = breakdown()
    for u in bd['users'][:3]:
        pprint.pprint(u)
    print(f'\nTotal users: {len(bd["users"])}')
    print(f'Total flags: {len(bd["flags"])}')
    print(f'Total health checks: {len(bd["health_checks"])}')
    print(f'Total configs: {len(bd["configs"])}')
    print('\n=== DEFINITIONS ===')
    defs = definitions()
    for d in defs:
        print(f'  {d["title"]}')
