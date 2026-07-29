"""Seed component_findings table with realistic doctor-AI agreement data."""
import sqlite3, random, os
from datetime import datetime, timedelta

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

def seed():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    patients = [r[0] for r in cur.execute('SELECT DISTINCT patient_id FROM patients').fetchall()]
    components = ['acquisition', 'artifacts', 'background', 'epileptiform', 'explainability', 'video']
    doctors = ['Dr. Patel', 'Dr. Chen', 'Dr. Rodriguez', 'Dr. Kim', 'Dr. Thompson', 'Dr. Nguyen', 'Dr. Williams', 'Dr. Garcia']
    findings_by_component = {
        'acquisition': ['Good signal quality, all channels', 'Moderate impedance on T3/T4', 'Poor signal quality, re-record recommended', 'Acceptable quality with minor artifacts', 'Excellent acquisition, all impedances <5kOhm'],
        'artifacts': ['Minimal muscle artifact', 'Significant eye-blink artifact, ICA cleaned', 'Movement artifact during HV', 'EMG contamination in temporal leads', 'Clean recording, no significant artifacts'],
        'background': ['Normal posterior dominant rhythm 9-10 Hz', 'Diffuse slowing theta range', 'Asymmetric alpha, lower on left', 'Age-appropriate background', 'Generalized slowing with intermittent delta'],
        'epileptiform': ['No epileptiform discharges', 'Right temporal sharp waves', 'Bilateral independent temporal spikes', 'Generalized spike-and-wave 3Hz', 'Left frontal epileptiform discharges'],
        'explainability': ['SHAP features consistent with clinical impression', 'AI top features: delta power, sharp wave count', 'Feature importance aligns with EEG findings', 'AI explanation partially consistent', 'SHAP analysis supports clinical conclusion'],
        'video': ['No clinical events captured', 'Behavioral arrest correlated with EEG discharge', 'Subclinical seizure on EEG only', 'Video-EEG concordant for focal onset', 'No video available for this study'],
    }
    agree_options = ['agree', 'disagree', 'partial']
    agree_weights = [0.65, 0.15, 0.20]

    cur.execute('DELETE FROM component_findings')
    base_date = datetime(2026, 5, 1)
    inserted = 0
    for p in patients:
        n_comps = random.randint(3, 6)
        reviewed_comps = random.sample(components, n_comps)
        doc = random.choice(doctors)
        for comp in reviewed_comps:
            finding = random.choice(findings_by_component[comp])
            agree = random.choices(agree_options, weights=agree_weights, k=1)[0]
            ts = base_date + timedelta(days=random.randint(0, 80), hours=random.randint(8, 17), minutes=random.randint(0, 59))
            created = ts.strftime('%Y-%m-%dT%H:%M:%S-06:00')
            updated = (ts + timedelta(minutes=random.randint(5, 120))).strftime('%Y-%m-%dT%H:%M:%S-06:00')
            try:
                cur.execute(
                    'INSERT INTO component_findings (patient_id, component, doctor_finding, doctor, agree_with_ai, created_at, updated_at) VALUES (?,?,?,?,?,?,?)',
                    (p, comp, finding, doc, agree, created, updated))
                inserted += 1
            except sqlite3.IntegrityError:
                pass
    conn.commit()
    total = cur.execute('SELECT COUNT(*) FROM component_findings').fetchone()[0]
    print(f'Inserted {inserted} rows, total now: {total}')
    by_comp = cur.execute('SELECT component, COUNT(*) FROM component_findings GROUP BY component').fetchall()
    for c, n in by_comp:
        print(f'  {c}: {n}')
    by_agree = cur.execute('SELECT agree_with_ai, COUNT(*) FROM component_findings GROUP BY agree_with_ai').fetchall()
    for a, n in by_agree:
        print(f'  {a}: {n}')
    conn.close()

if __name__ == '__main__':
    seed()
