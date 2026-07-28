"""Patient Demographics Dashboard — backend analytics for patient_demographics table."""
import sqlite3, os

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

def _conn():
    return sqlite3.connect(DB)

def overview():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    total_patients = c.execute("SELECT COUNT(*) FROM patient_demographics").fetchone()[0]
    avg_age = c.execute("SELECT ROUND(AVG(age),1) FROM patient_demographics").fetchone()[0]
    avg_bmi = c.execute("SELECT ROUND(AVG(bmi),1) FROM patient_demographics").fetchone()[0]

    interp_needed = c.execute("SELECT COUNT(*) FROM patient_demographics WHERE interpreter_needed=1").fetchone()[0]
    interpreter_needed_pct = round(interp_needed / total_patients * 100, 1) if total_patients else 0

    avg_years_with_epilepsy = c.execute("SELECT ROUND(AVG(years_with_epilepsy),1) FROM patient_demographics").fetchone()[0]
    avg_onset_age = c.execute("SELECT ROUND(AVG(epilepsy_onset_age),1) FROM patient_demographics").fetchone()[0]

    sex_dist = [dict(r) for r in c.execute(
        "SELECT sex, COUNT(*) AS count FROM patient_demographics GROUP BY sex ORDER BY count DESC")]

    epilepsy_type_dist = [dict(r) for r in c.execute(
        "SELECT epilepsy_type, COUNT(*) AS count FROM patient_demographics GROUP BY epilepsy_type ORDER BY count DESC")]

    insurance_type_dist = [dict(r) for r in c.execute(
        "SELECT insurance_type, COUNT(*) AS count FROM patient_demographics GROUP BY insurance_type ORDER BY count DESC")]

    blood_type_dist = [dict(r) for r in c.execute(
        "SELECT blood_type, COUNT(*) AS count FROM patient_demographics GROUP BY blood_type ORDER BY count DESC")]

    referral_source_dist = [dict(r) for r in c.execute(
        "SELECT referral_source, COUNT(*) AS count FROM patient_demographics GROUP BY referral_source ORDER BY count DESC")]

    education_level_dist = [dict(r) for r in c.execute(
        "SELECT education_level, COUNT(*) AS count FROM patient_demographics GROUP BY education_level ORDER BY count DESC")]

    employment_status_dist = [dict(r) for r in c.execute(
        "SELECT employment_status, COUNT(*) AS count FROM patient_demographics GROUP BY employment_status ORDER BY count DESC")]

    age_histogram = [dict(r) for r in c.execute("""
        SELECT
            CASE
                WHEN age < 20 THEN '0-19'
                WHEN age < 30 THEN '20-29'
                WHEN age < 40 THEN '30-39'
                WHEN age < 50 THEN '40-49'
                WHEN age < 60 THEN '50-59'
                WHEN age < 70 THEN '60-69'
                ELSE '70+'
            END AS age_bin,
            COUNT(*) AS count
        FROM patient_demographics
        GROUP BY age_bin
        ORDER BY age_bin
    """)]

    enrollment_trend = [dict(r) for r in c.execute("""
        SELECT SUBSTR(enrollment_date,1,7) AS month,
               COUNT(*) AS patients
        FROM patient_demographics
        GROUP BY month ORDER BY month
    """)]

    state_distribution = [dict(r) for r in c.execute(
        "SELECT address_state AS state, COUNT(*) AS count FROM patient_demographics GROUP BY address_state ORDER BY count DESC")]

    bmi_category_dist = [dict(r) for r in c.execute("""
        SELECT
            CASE
                WHEN bmi < 18.5 THEN 'Underweight'
                WHEN bmi < 25.0 THEN 'Normal'
                WHEN bmi < 30.0 THEN 'Overweight'
                ELSE 'Obese'
            END AS category,
            COUNT(*) AS count
        FROM patient_demographics
        GROUP BY category
        ORDER BY category
    """)]

    conn.close()
    return {
        "kpis": {
            "total_patients": total_patients,
            "avg_age": avg_age,
            "avg_bmi": avg_bmi,
            "interpreter_needed_pct": interpreter_needed_pct,
            "avg_years_with_epilepsy": avg_years_with_epilepsy,
            "avg_onset_age": avg_onset_age,
        },
        "sex_dist": sex_dist,
        "epilepsy_type_dist": epilepsy_type_dist,
        "insurance_type_dist": insurance_type_dist,
        "blood_type_dist": blood_type_dist,
        "referral_source_dist": referral_source_dist,
        "education_level_dist": education_level_dist,
        "employment_status_dist": employment_status_dist,
        "age_histogram": age_histogram,
        "enrollment_trend": enrollment_trend,
        "state_distribution": state_distribution,
        "bmi_category_dist": bmi_category_dist,
    }

def breakdown():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    patients = [dict(r) for r in c.execute(
        "SELECT * FROM patient_demographics ORDER BY patient_id LIMIT 200")]

    by_epilepsy_type = [dict(r) for r in c.execute("""
        SELECT epilepsy_type,
               COUNT(*) AS patient_count,
               ROUND(AVG(age),1) AS avg_age,
               ROUND(AVG(years_with_epilepsy),1) AS avg_years_with_epilepsy,
               ROUND(AVG(bmi),1) AS avg_bmi
        FROM patient_demographics
        GROUP BY epilepsy_type ORDER BY patient_count DESC
    """)]

    by_neurologist = [dict(r) for r in c.execute("""
        SELECT primary_neurologist,
               COUNT(*) AS patient_count,
               ROUND(AVG(age),1) AS avg_age
        FROM patient_demographics
        GROUP BY primary_neurologist ORDER BY patient_count DESC
    """)]

    conn.close()
    return {
        "patients": patients,
        "by_epilepsy_type": by_epilepsy_type,
        "by_neurologist": by_neurologist,
    }

def definitions():
    return {
        "title": "Patient Demographics Dashboard — Definitions",
        "concepts": [
            {"name": "Total Patients", "description": "Total number of patients enrolled in the epilepsy clinical portal with demographic records on file."},
            {"name": "Average Age", "description": "Mean age in years across all enrolled patients. Computed from the age column at time of data capture."},
            {"name": "Average BMI", "description": "Mean Body Mass Index (kg/m^2) across all patients. BMI categories: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (30+)."},
            {"name": "Interpreter Needed %", "description": "Percentage of patients who require a language interpreter for clinical encounters. Flags potential communication barriers and compliance requirements."},
            {"name": "Average Years with Epilepsy", "description": "Mean duration in years since each patient's epilepsy diagnosis. Longer durations may correlate with medication complexity and comorbidities."},
            {"name": "Average Onset Age", "description": "Mean age at which epilepsy was first diagnosed across the cohort. Early onset vs. late onset may influence treatment strategy and prognosis."},
            {"name": "Sex", "description": "Biological sex recorded in the patient's medical record (Male, Female, Other). Used for clinical risk stratification and medication dosing considerations."},
            {"name": "Epilepsy Type", "description": "Clinical classification of the patient's epilepsy (e.g., Focal, Generalized, Combined, Unknown). Determines treatment protocols, medication choices, and surgical candidacy."},
            {"name": "Insurance Type", "description": "Primary insurance coverage (Private, Medicare, Medicaid, Uninsured, etc.). Affects access to medications, devices, and specialist referrals."},
            {"name": "Blood Type", "description": "ABO blood group and Rh factor. Relevant for surgical planning and transfusion compatibility in pre-surgical epilepsy evaluations."},
            {"name": "Referral Source", "description": "Origin of the patient referral to the epilepsy center (e.g., Primary Care, Neurologist, Emergency, Self-referral). Tracks referral patterns and outreach effectiveness."},
            {"name": "Education Level", "description": "Highest level of education attained. Influences health literacy, medication adherence education strategies, and self-management capabilities."},
            {"name": "Employment Status", "description": "Current employment situation (Employed, Unemployed, Retired, Student, Disabled). Epilepsy significantly impacts employment; tracked for social determinants of health."},
            {"name": "BMI Category", "description": "Body Mass Index classification: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (30+). Anti-epileptic drugs can affect weight; BMI monitoring is part of medication management."},
            {"name": "Primary Neurologist", "description": "The attending neurologist responsible for the patient's epilepsy care. Used for workload analysis and panel distribution."},
            {"name": "Enrollment Date", "description": "Date the patient was enrolled in the epilepsy clinical portal. Used to track enrollment trends and cohort growth over time."},
        ],
        "data_sources": [
            "patient_demographics table — 30 patients, demographics + epilepsy clinical data",
            "Electronic Health Record (EHR) demographic intake forms",
            "Epilepsy clinical portal enrollment system",
        ],
    }

if __name__ == "__main__":
    import json
    print(json.dumps(overview(), indent=2))
