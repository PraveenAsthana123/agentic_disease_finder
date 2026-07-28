"""Comorbidities Dashboard — backend analytics for comorbidities table.

Real data: comorbidities (27 rows, 27 patients).
Each row has fields_json with: comorbidities list, comorbidity_count,
screening_instruments, screening_date, behavioral_risk_score, risk_severity,
functional_impact, treatment_status, notes.
"""
import sqlite3, os, json
from collections import Counter

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _parse_rows():
    """Return list of dicts: one per patient, with fields_json parsed."""
    conn = _conn()
    rows = conn.execute("SELECT id, patient_id, fields_json, created_at FROM comorbidities ORDER BY id").fetchall()
    result = []
    for r in rows:
        try:
            fields = json.loads(r[2]) if r[2] else {}
        except Exception:
            fields = {}
        fields['id'] = r[0]
        fields['patient_id'] = r[1]
        fields['created_at'] = r[3]
        result.append(fields)
    conn.close()
    return result


def overview():
    rows = _parse_rows()
    total_patients = len(rows)
    total_with_comorbidities = sum(1 for r in rows if r.get('comorbidity_count', 0) > 0)
    total_without = total_patients - total_with_comorbidities
    comorbidity_rate = round(total_with_comorbidities / total_patients * 100, 1) if total_patients else 0
    avg_count = round(sum(r.get('comorbidity_count', 0) for r in rows) / total_patients, 1) if total_patients else 0
    max_count = max((r.get('comorbidity_count', 0) for r in rows), default=0)
    avg_risk = round(sum(r.get('behavioral_risk_score', 0) for r in rows) / total_patients, 1) if total_patients else 0
    screened_count = sum(1 for r in rows if r.get('screened'))
    screening_rate = round(screened_count / total_patients * 100, 1) if total_patients else 0

    # Risk severity distribution
    severity_counter = Counter(r.get('risk_severity', 'unknown') for r in rows)
    severity_dist = [{'severity': k, 'count': v} for k, v in severity_counter.most_common()]

    # Functional impact distribution
    impact_counter = Counter(r.get('functional_impact', 'unknown') for r in rows)
    impact_dist = [{'impact': k, 'count': v} for k, v in impact_counter.most_common()]

    # Treatment status distribution
    treatment_counter = Counter(r.get('treatment_status', 'unknown') for r in rows)
    treatment_dist = [{'status': k, 'count': v} for k, v in treatment_counter.most_common()]

    # Individual comorbidity frequency
    all_conditions = []
    for r in rows:
        all_conditions.extend(r.get('comorbidities', []))
    condition_counter = Counter(all_conditions)
    condition_dist = [{'condition': k, 'count': v} for k, v in condition_counter.most_common()]

    # Comorbidity count distribution (0, 1, 2, 3, 4, 5+)
    count_counter = Counter()
    for r in rows:
        c = r.get('comorbidity_count', 0)
        bucket = str(c) if c < 5 else '5+'
        count_counter[bucket] += 1
    count_dist = [{'bucket': k, 'count': v} for k, v in sorted(count_counter.items(), key=lambda x: (x[0] == '5+', x[0]))]

    # Screening instruments frequency
    all_instruments = []
    for r in rows:
        all_instruments.extend(r.get('screening_instruments', []))
    instrument_counter = Counter(all_instruments)
    instrument_dist = [{'instrument': k, 'count': v} for k, v in instrument_counter.most_common()]

    # Risk score by severity
    from collections import defaultdict
    severity_scores = defaultdict(list)
    for r in rows:
        sev = r.get('risk_severity', 'unknown')
        severity_scores[sev].append(r.get('behavioral_risk_score', 0))
    risk_by_severity = [
        {'severity': k, 'avg_score': round(sum(v) / len(v), 1), 'count': len(v)}
        for k, v in severity_scores.items()
    ]
    risk_by_severity.sort(key=lambda x: x['avg_score'])

    # Monthly screening trend
    from collections import defaultdict as dd
    monthly = dd(lambda: {'screened': 0, 'with_comorbidity': 0})
    for r in rows:
        dt = r.get('screening_date') or r.get('created_at', '')
        month = dt[:7] if dt else 'unknown'
        monthly[month]['screened'] += 1
        if r.get('comorbidity_count', 0) > 0:
            monthly[month]['with_comorbidity'] += 1
    monthly_trend = [{'month': k, **v} for k, v in sorted(monthly.items()) if k != 'unknown']

    return {
        'total_patients': total_patients,
        'total_with_comorbidities': total_with_comorbidities,
        'total_without': total_without,
        'comorbidity_rate': comorbidity_rate,
        'avg_comorbidity_count': avg_count,
        'max_comorbidity_count': max_count,
        'avg_behavioral_risk_score': avg_risk,
        'screening_rate': screening_rate,
        'screened_count': screened_count,
        'severity_distribution': severity_dist,
        'impact_distribution': impact_dist,
        'treatment_distribution': treatment_dist,
        'condition_distribution': condition_dist,
        'count_distribution': count_dist,
        'instrument_distribution': instrument_dist,
        'risk_by_severity': risk_by_severity,
        'monthly_trend': monthly_trend,
    }


def breakdown():
    rows = _parse_rows()

    # All patients table
    patients = []
    for r in rows:
        patients.append({
            'patient_id': r.get('patient_id'),
            'comorbidity_count': r.get('comorbidity_count', 0),
            'comorbidities': ', '.join(r.get('comorbidities', [])) or 'None',
            'risk_severity': r.get('risk_severity', 'unknown'),
            'behavioral_risk_score': r.get('behavioral_risk_score', 0),
            'functional_impact': r.get('functional_impact', 'unknown'),
            'treatment_status': r.get('treatment_status', 'unknown'),
            'screening_date': r.get('screening_date', ''),
            'instruments': ', '.join(r.get('screening_instruments', [])),
            'notes': r.get('notes', ''),
        })

    # By condition: for each unique condition, list patients and stats
    from collections import defaultdict
    cond_patients = defaultdict(list)
    for r in rows:
        for cond in r.get('comorbidities', []):
            cond_patients[cond].append({
                'patient_id': r.get('patient_id'),
                'risk_severity': r.get('risk_severity'),
                'behavioral_risk_score': r.get('behavioral_risk_score', 0),
                'treatment_status': r.get('treatment_status'),
            })
    by_condition = []
    for cond, pts in sorted(cond_patients.items(), key=lambda x: -len(x[1])):
        avg_risk = round(sum(p['behavioral_risk_score'] for p in pts) / len(pts), 1)
        by_condition.append({
            'condition': cond,
            'patient_count': len(pts),
            'avg_risk_score': avg_risk,
            'patients': pts,
        })

    # By severity
    sev_groups = defaultdict(list)
    for r in rows:
        sev_groups[r.get('risk_severity', 'unknown')].append(r.get('patient_id'))
    by_severity = [
        {'severity': k, 'patient_count': len(v), 'patients': v}
        for k, v in sorted(sev_groups.items())
    ]

    return {
        'patients': patients,
        'by_condition': by_condition,
        'by_severity': by_severity,
    }


def definitions():
    return {
        'fields': [
            {'field': 'comorbidities', 'description': 'List of diagnosed psychiatric or neurological comorbid conditions for epilepsy patients.'},
            {'field': 'comorbidity_count', 'description': 'Number of comorbid conditions identified per patient.'},
            {'field': 'screening_instruments', 'description': 'Validated psychiatric screening tools used (e.g., PHQ-9, GAD-7, C-SSRS, NDDI-E, PCL-5, MDQ).'},
            {'field': 'screening_date', 'description': 'Date when comorbidity screening was performed.'},
            {'field': 'behavioral_risk_score', 'description': 'Composite behavioral risk score (0-100). Higher values indicate greater psychiatric risk.'},
            {'field': 'risk_severity', 'description': 'Categorized risk level: minimal, mild, moderate, or severe.'},
            {'field': 'functional_impact', 'description': 'Degree to which comorbidities affect daily functioning: none, mild, moderate, or severe.'},
            {'field': 'treatment_status', 'description': 'Current treatment status: none, untreated, stable, partial_response, treatment_resistant.'},
        ],
        'conditions': [
            {'name': 'Major Depressive Disorder', 'description': 'Clinical depression; common in epilepsy (20-30% prevalence).'},
            {'name': 'Generalized Anxiety Disorder', 'description': 'Chronic anxiety; highly prevalent in epilepsy populations.'},
            {'name': 'PTSD', 'description': 'Post-traumatic stress disorder; can co-occur with epilepsy, especially after traumatic brain injury.'},
            {'name': 'Social Anxiety Disorder', 'description': 'Fear of social situations; linked to seizure-related stigma.'},
            {'name': 'Insomnia Disorder', 'description': 'Chronic sleep difficulty; seizures and AEDs can disrupt sleep.'},
            {'name': 'Substance Use Disorder', 'description': 'Alcohol or substance misuse; lowers seizure threshold.'},
            {'name': 'Conversion Disorder', 'description': 'Functional neurological symptom disorder; psychogenic non-epileptic seizures (PNES).'},
            {'name': 'Bipolar Disorder', 'description': 'Mood disorder with manic/depressive episodes; some AEDs are mood stabilizers.'},
            {'name': 'OCD', 'description': 'Obsessive-compulsive disorder; occasionally co-occurs with temporal lobe epilepsy.'},
            {'name': 'Panic Disorder', 'description': 'Recurrent panic attacks; can mimic or co-occur with seizures.'},
        ],
        'instruments': [
            {'name': 'PHQ-9', 'description': 'Patient Health Questionnaire-9: screens for depression severity (0-27 scale).'},
            {'name': 'GAD-7', 'description': 'Generalized Anxiety Disorder 7-item: screens for anxiety severity (0-21 scale).'},
            {'name': 'C-SSRS', 'description': 'Columbia Suicide Severity Rating Scale: assesses suicidal ideation and behavior.'},
            {'name': 'NDDI-E', 'description': 'Neurological Disorders Depression Inventory for Epilepsy: epilepsy-specific depression screen.'},
            {'name': 'PCL-5', 'description': 'PTSD Checklist for DSM-5: screens for post-traumatic stress disorder.'},
            {'name': 'MDQ', 'description': 'Mood Disorder Questionnaire: screens for bipolar spectrum disorders.'},
        ],
        'data_source': 'comorbidities table (27 rows, 27 patients) — psychiatric comorbidity screening records with validated instruments.',
    }
