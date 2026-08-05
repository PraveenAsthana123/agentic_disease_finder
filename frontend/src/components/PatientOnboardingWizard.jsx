import React, { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

// ── Palette ────────────────────────────────────────────────────────────────
const C = {
  blue: '#3b82f6', green: '#22c55e', amber: '#f59e0b',
  red: '#ef4444', purple: '#8b5cf6', slate: '#64748b',
  bg: '#f8fafc', card: '#ffffff', border: '#e2e8f0',
  text: '#1e293b', muted: '#64748b',
}

// ── Layout helpers ─────────────────────────────────────────────────────────
function Card({ children, style }) {
  return (
    <div style={{
      background: C.card, borderRadius: 10, padding: 20, marginBottom: 16,
      boxShadow: '0 1px 4px rgba(0,0,0,0.08)', border: `1px solid ${C.border}`,
      ...style,
    }}>{children}</div>
  )
}

function Field({ label, required, children }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: C.muted, marginBottom: 4 }}>
        {label}{required && <span style={{ color: C.red }}> *</span>}
      </label>
      {children}
    </div>
  )
}

const inputStyle = {
  width: '100%', padding: '8px 10px', borderRadius: 6, fontSize: 13,
  border: `1px solid ${C.border}`, color: C.text, background: '#fff',
  outline: 'none', boxSizing: 'border-box',
}

function Input({ value, onChange, placeholder, type = 'text' }) {
  return (
    <input
      type={type} value={value} onChange={e => onChange(e.target.value)}
      placeholder={placeholder} style={inputStyle}
    />
  )
}

function Select({ value, onChange, options }) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)} style={inputStyle}>
      <option value="">— select —</option>
      {options.map(o => <option key={o} value={o}>{o}</option>)}
    </select>
  )
}

function Btn({ label, onClick, variant = 'primary', disabled }) {
  const bg = variant === 'primary' ? C.blue : variant === 'success' ? C.green : C.border
  const color = variant === 'ghost' ? C.text : '#fff'
  return (
    <button
      onClick={onClick} disabled={disabled}
      style={{
        padding: '9px 20px', borderRadius: 7, border: 'none', cursor: disabled ? 'not-allowed' : 'pointer',
        background: disabled ? '#cbd5e1' : bg, color: disabled ? '#94a3b8' : color,
        fontWeight: 600, fontSize: 13, transition: 'opacity 0.15s',
      }}
    >{label}</button>
  )
}

// ── Progress bar ───────────────────────────────────────────────────────────
function ProgressBar({ step, total }) {
  const steps = ['Demographics & Seizure History', 'Medications & Emergency Contact', 'Document Upload']
  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ display: 'flex', gap: 4, marginBottom: 10 }}>
        {steps.map((s, i) => (
          <div key={i} style={{ flex: 1, textAlign: 'center' }}>
            <div style={{
              width: 28, height: 28, borderRadius: '50%', margin: '0 auto 4px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 13, fontWeight: 700,
              background: i + 1 < step ? C.green : i + 1 === step ? C.blue : C.border,
              color: i + 1 <= step ? '#fff' : C.muted,
            }}>
              {i + 1 < step ? '✓' : i + 1}
            </div>
            <div style={{ fontSize: 11, color: i + 1 === step ? C.blue : C.muted, fontWeight: i + 1 === step ? 600 : 400 }}>
              {s}
            </div>
          </div>
        ))}
      </div>
      <div style={{ height: 4, background: C.border, borderRadius: 2 }}>
        <div style={{
          height: 4, borderRadius: 2, background: C.blue,
          width: `${((step - 1) / (total - 1)) * 100}%`,
          transition: 'width 0.4s',
        }} />
      </div>
    </div>
  )
}

// ── Step 1: Demographics + Clinical Core ──────────────────────────────────
function Step1({ data, setData }) {
  const set = (k, v) => setData(d => ({ ...d, [k]: v }))
  return (
    <div>
      <h3 style={{ fontSize: 15, fontWeight: 700, color: C.text, margin: '0 0 16px' }}>
        Step 1 — Demographics & Seizure History
        <span style={{ fontSize: 11, fontWeight: 400, color: C.muted, marginLeft: 8 }}>~5 min · 56 fields (auto-counted)</span>
      </h3>

      {/* Demographics */}
      <Card>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.blue, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Demographics</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 16px' }}>
          <Field label="Full Name" required>
            <Input value={data.full_name || ''} onChange={v => set('full_name', v)} placeholder="Patient full name" />
          </Field>
          <Field label="Date of Birth" required>
            <Input type="date" value={data.date_of_birth || ''} onChange={v => set('date_of_birth', v)} />
          </Field>
          <Field label="Sex" required>
            <Select value={data.sex || ''} onChange={v => set('sex', v)} options={['Male', 'Female', 'Other', 'Prefer not to say']} />
          </Field>
          <Field label="Primary Language">
            <Select value={data.primary_language || ''} onChange={v => set('primary_language', v)}
              options={['English', 'French', 'Hindi', 'Punjabi', 'Mandarin', 'Spanish', 'Arabic', 'Other']} />
          </Field>
          <Field label="Occupation">
            <Input value={data.occupation || ''} onChange={v => set('occupation', v)} placeholder="e.g. Teacher, Engineer" />
          </Field>
          <Field label="Referring Provider">
            <Input value={data.referral_source || ''} onChange={v => set('referral_source', v)} placeholder="Referring physician/clinic" />
          </Field>
        </div>
      </Card>

      {/* Chief Complaint */}
      <Card>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.purple, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Chief Complaint</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 16px' }}>
          <Field label="Reason for Visit" required>
            <Input value={data.reason_for_visit || ''} onChange={v => set('reason_for_visit', v)} placeholder="e.g. Seizure evaluation" />
          </Field>
          <Field label="Suspected Diagnosis">
            <Select value={data.suspected_diagnosis || ''} onChange={v => set('suspected_diagnosis', v)}
              options={['Focal epilepsy', 'Generalized epilepsy', 'Unknown onset', 'PNES', 'Febrile seizure', 'Other']} />
          </Field>
        </div>
      </Card>

      {/* Seizure History */}
      <Card>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.amber, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Seizure History</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 16px' }}>
          <Field label="Epilepsy Type">
            <Select value={data.epilepsy_type || ''} onChange={v => set('epilepsy_type', v)}
              options={['Focal', 'Generalized', 'Combined', 'Unknown', 'Non-epileptic']} />
          </Field>
          <Field label="Age of Onset (years)">
            <Input type="number" value={data.epilepsy_onset_age || ''} onChange={v => set('epilepsy_onset_age', v)} placeholder="e.g. 12" />
          </Field>
          <Field label="Current Seizure Frequency">
            <Select value={data.seizure_frequency || ''} onChange={v => set('seizure_frequency', v)}
              options={['Seizure-free', '<1/year', '1-11/year', '1-3/month', '1-6/week', 'Daily', 'Multiple/day']} />
          </Field>
          <Field label="Last Seizure Date">
            <Input type="date" value={data.last_seizure_date || ''} onChange={v => set('last_seizure_date', v)} />
          </Field>
          <Field label="Drug-Resistant Epilepsy (DRE)?">
            <Select value={data.dre_status || ''} onChange={v => set('dre_status', v)}
              options={['Yes — DRE confirmed', 'No — controlled', 'Unknown', 'Suspected DRE']} />
          </Field>
        </div>
      </Card>

      {/* Key Risks */}
      <Card>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.red, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Key Risk Flags</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0 16px' }}>
          <Field label="Status Epilepticus History">
            <Select value={data.status_epilepticus_history || ''} onChange={v => set('status_epilepticus_history', v)}
              options={['Yes', 'No', 'Unknown']} />
          </Field>
          <Field label="Driving Status">
            <Select value={data.driving_status || ''} onChange={v => set('driving_status', v)}
              options={['Not driving', 'Driving — restricted', 'Driving — unrestricted', 'Unknown']} />
          </Field>
          <Field label="Lives Alone">
            <Select value={data.lives_alone || ''} onChange={v => set('lives_alone', v)}
              options={['Yes', 'No', 'Part-time alone']} />
          </Field>
        </div>
      </Card>
    </div>
  )
}

// ── Step 2: Medications + Emergency Contact ────────────────────────────────
const EMPTY_MED = { drug: '', dose: '', frequency: '', start_date: '' }

function Step2({ data, setData }) {
  const setEC = (k, v) => setData(d => ({ ...d, emergency_contact: { ...d.emergency_contact, [k]: v } }))
  const ec = data.emergency_contact || {}
  const meds = data.medications || [{ ...EMPTY_MED }]

  const updateMed = (i, k, v) => {
    const next = meds.map((m, idx) => idx === i ? { ...m, [k]: v } : m)
    setData(d => ({ ...d, medications: next }))
  }
  const addMed = () => setData(d => ({ ...d, medications: [...(d.medications || [EMPTY_MED]), { ...EMPTY_MED }] }))
  const removeMed = i => setData(d => ({ ...d, medications: (d.medications || []).filter((_, idx) => idx !== i) }))

  return (
    <div>
      <h3 style={{ fontSize: 15, fontWeight: 700, color: C.text, margin: '0 0 16px' }}>
        Step 2 — Medications & Emergency Contact
        <span style={{ fontSize: 11, fontWeight: 400, color: C.muted, marginLeft: 8 }}>Current AEDs + first responder</span>
      </h3>

      <Card>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.green, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
          Emergency Contact <span style={{ fontWeight: 400, color: C.muted }}>(primary)</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0 16px' }}>
          <Field label="Contact Name" required>
            <Input value={ec.contact_name || ''} onChange={v => setEC('contact_name', v)} placeholder="Full name" />
          </Field>
          <Field label="Phone" required>
            <Input value={ec.phone || ''} onChange={v => setEC('phone', v)} placeholder="+1 (555) 000-0000" />
          </Field>
          <Field label="Relationship">
            <Select value={ec.relationship || ''} onChange={v => setEC('relationship', v)}
              options={['Spouse/Partner', 'Parent', 'Child', 'Sibling', 'Friend', 'Caregiver', 'Other']} />
          </Field>
        </div>
      </Card>

      <Card>
        <div style={{
          fontSize: 12, fontWeight: 700, color: C.blue, marginBottom: 12,
          textTransform: 'uppercase', letterSpacing: 1,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <span>Current Medications (AEDs)</span>
          <Btn label="+ Add Medication" onClick={addMed} variant="ghost" />
        </div>

        {meds.map((med, i) => (
          <div key={i} style={{
            display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr auto',
            gap: '0 10px', alignItems: 'end', marginBottom: 10,
            background: '#f8fafc', padding: '10px 12px', borderRadius: 6,
          }}>
            <Field label={i === 0 ? 'Drug Name' : ''}>
              <Input value={med.drug || ''} onChange={v => updateMed(i, 'drug', v)} placeholder="e.g. Levetiracetam" />
            </Field>
            <Field label={i === 0 ? 'Dose' : ''}>
              <Input value={med.dose || ''} onChange={v => updateMed(i, 'dose', v)} placeholder="e.g. 500 mg" />
            </Field>
            <Field label={i === 0 ? 'Frequency' : ''}>
              <Select value={med.frequency || ''} onChange={v => updateMed(i, 'frequency', v)}
                options={['Once daily', 'Twice daily', 'Three times daily', 'As needed', 'Weekly']} />
            </Field>
            <Field label={i === 0 ? 'Start Date' : ''}>
              <Input type="date" value={med.start_date || ''} onChange={v => updateMed(i, 'start_date', v)} />
            </Field>
            <div style={{ paddingBottom: 14 }}>
              {meds.length > 1 && (
                <button onClick={() => removeMed(i)} style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  color: C.red, fontSize: 16, padding: '6px',
                }}>✕</button>
              )}
            </div>
          </div>
        ))}
      </Card>
    </div>
  )
}

// ── Step 3: Document Upload ────────────────────────────────────────────────
const DOC_TYPES = [
  { label: 'EEG Report (PDF)', key: 'eeg_report', fills: 'Acquisition params · Background rhythm · Spike findings · Localization' },
  { label: 'MRI Report (PDF)', key: 'mri_report', fills: 'Lesion findings · MTS · Concordance' },
  { label: 'EMR Export', key: 'emr', fills: 'Demographics · Full med history · Comorbidities' },
  { label: 'Prior Neurology Notes', key: 'neuro_notes', fills: 'Etiology · Treatment history' },
]

function Step3({ data, setData }) {
  const noted = data.docs_noted || []
  const toggle = key => {
    setData(d => ({
      ...d,
      docs_noted: noted.includes(key) ? noted.filter(k => k !== key) : [...noted, key],
    }))
  }

  return (
    <div>
      <h3 style={{ fontSize: 15, fontWeight: 700, color: C.text, margin: '0 0 16px' }}>
        Step 3 — Document Upload
        <span style={{ fontSize: 11, fontWeight: 400, color: C.muted, marginLeft: 8 }}>~2 min · auto-extract saves ~40% manual effort</span>
      </h3>

      <Card>
        <div style={{
          background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 8,
          padding: 14, marginBottom: 16, fontSize: 13, color: '#1d4ed8',
        }}>
          <strong>How it works:</strong> Upload documents below using the <em>Analyze Upload</em> tab.
          The AI auto-extracts fields into patient_demographics, eeg_interpretation, and mri_findings.
          Check off documents you plan to provide so the system knows what's pending.
        </div>

        {DOC_TYPES.map(doc => (
          <div key={doc.key} style={{
            display: 'flex', alignItems: 'flex-start', gap: 12,
            padding: '12px 14px', borderRadius: 7, marginBottom: 8,
            background: noted.includes(doc.key) ? '#f0fdf4' : '#f8fafc',
            border: `1px solid ${noted.includes(doc.key) ? '#86efac' : C.border}`,
            cursor: 'pointer',
          }} onClick={() => toggle(doc.key)}>
            <div style={{
              width: 20, height: 20, borderRadius: 4, flexShrink: 0, marginTop: 1,
              background: noted.includes(doc.key) ? C.green : '#fff',
              border: `2px solid ${noted.includes(doc.key) ? C.green : C.border}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: '#fff', fontSize: 12, fontWeight: 700,
            }}>
              {noted.includes(doc.key) ? '✓' : ''}
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: C.text }}>{doc.label}</div>
              <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>Auto-fills: {doc.fills}</div>
            </div>
          </div>
        ))}

        <div style={{ marginTop: 16, padding: '10px 14px', background: '#fefce8', borderRadius: 7, border: '1px solid #fde68a', fontSize: 12, color: '#92400e' }}>
          <strong>Deferred fields (~1,170):</strong> Seizure diary, trigger tracking, medication adherence, sleep/mood/QoL,
          wearables data — these fill automatically through ongoing portal use. Not required at intake.
        </div>
      </Card>
    </div>
  )
}

// ── Completion screen ──────────────────────────────────────────────────────
function Completion({ patientId, onReset }) {
  const [status, setStatus] = useState(null)

  useEffect(() => {
    axios.get(`${API_URL}/api/patient-wizard/status/${patientId}`)
      .then(r => setStatus(r.data))
      .catch(() => {})
  }, [patientId])

  return (
    <Card style={{ textAlign: 'center', padding: 40 }}>
      <div style={{ fontSize: 48, marginBottom: 12 }}>🎉</div>
      <div style={{ fontSize: 20, fontWeight: 700, color: C.green, marginBottom: 8 }}>Intake Complete!</div>
      <div style={{ fontSize: 13, color: C.muted, marginBottom: 24 }}>
        Patient <strong>{patientId}</strong> has been onboarded. Data saved to clinical DB.
      </div>
      {status && (
        <div style={{ display: 'flex', justifyContent: 'center', gap: 16, marginBottom: 24 }}>
          {status.steps.map(s => (
            <div key={s.step} style={{
              padding: '8px 16px', borderRadius: 8,
              background: s.complete ? '#f0fdf4' : '#fef2f2',
              border: `1px solid ${s.complete ? '#86efac' : '#fca5a5'}`,
              fontSize: 12, color: s.complete ? '#166534' : '#991b1b',
            }}>
              {s.complete ? '✓' : '⚠'} Step {s.step}: {s.title}
            </div>
          ))}
        </div>
      )}
      <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
        <Btn label="Onboard Another Patient" onClick={onReset} variant="primary" />
      </div>
      <div style={{ marginTop: 16, fontSize: 12, color: C.muted }}>
        Next: Upload EEG/MRI reports via <strong>Analyze Upload</strong> tab for auto-extraction.
      </div>
    </Card>
  )
}

// ── Main Wizard ────────────────────────────────────────────────────────────
export default function PatientOnboardingWizard() {
  const [step, setStep] = useState(1)
  const [patientId, setPatientId] = useState('')
  const [step1Data, setStep1Data] = useState({})
  const [step2Data, setStep2Data] = useState({ emergency_contact: {}, medications: [{ ...EMPTY_MED }] })
  const [step3Data, setStep3Data] = useState({ docs_noted: [] })
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)
  const [done, setDone] = useState(false)
  const [wizardMeta, setWizardMeta] = useState(null)

  useEffect(() => {
    axios.get(`${API_URL}/api/patient-wizard/steps`).then(r => setWizardMeta(r.data)).catch(() => {})
  }, [])

  const submit = async (stepNum, data) => {
    setSaving(true)
    setError(null)
    try {
      await axios.post(`${API_URL}/api/patient-wizard/submit`, {
        patient_id: patientId,
        step: stepNum,
        data,
      })
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Save failed')
      setSaving(false)
      return false
    }
    setSaving(false)
    return true
  }

  const next = async () => {
    if (!patientId.trim()) { setError('Patient ID is required'); return }
    if (step === 1) {
      if (!step1Data.full_name || !step1Data.date_of_birth || !step1Data.sex) {
        setError('Full Name, Date of Birth, and Sex are required'); return
      }
      const ok = await submit(1, step1Data)
      if (ok) setStep(2)
    } else if (step === 2) {
      const ok = await submit(2, step2Data)
      if (ok) setStep(3)
    } else if (step === 3) {
      const ok = await submit(3, step3Data)
      if (ok) setDone(true)
    }
  }

  const reset = () => {
    setStep(1); setPatientId(''); setStep1Data({}); setDone(false); setError(null)
    setStep2Data({ emergency_contact: {}, medications: [{ ...EMPTY_MED }] })
    setStep3Data({ docs_noted: [] })
  }

  return (
    <div style={{ background: C.bg, minHeight: '100vh', padding: '24px 20px' }}>
      <div style={{ maxWidth: 860, margin: '0 auto' }}>

        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 22, fontWeight: 800, color: C.text }}>Patient Onboarding Wizard</div>
          <div style={{ fontSize: 13, color: C.muted, marginTop: 4 }}>
            {wizardMeta?.goal || '2-3 hour intake → 8-10 min active capture (80 required fields, ~1170 deferred)'}
          </div>
        </div>

        {done ? (
          <Completion patientId={patientId} onReset={reset} />
        ) : (
          <>
            {/* Patient ID input */}
            <Card>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: C.text, flexShrink: 0 }}>Patient ID</div>
                <input
                  value={patientId}
                  onChange={e => setPatientId(e.target.value)}
                  placeholder="e.g. CHB01 or NEW-2026-001"
                  style={{ ...inputStyle, maxWidth: 280 }}
                />
                <div style={{ fontSize: 11, color: C.muted }}>
                  Enter existing patient ID to update, or a new ID to register
                </div>
              </div>
            </Card>

            <ProgressBar step={step} total={3} />

            {step === 1 && <Step1 data={step1Data} setData={setStep1Data} />}
            {step === 2 && <Step2 data={step2Data} setData={setStep2Data} />}
            {step === 3 && <Step3 data={step3Data} setData={setStep3Data} />}

            {error && (
              <div style={{
                padding: '10px 14px', borderRadius: 7, marginBottom: 12,
                background: '#fef2f2', border: '1px solid #fca5a5', color: C.red, fontSize: 13,
              }}>
                ⚠ {error}
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
              <Btn label="← Back" onClick={() => setStep(s => s - 1)} variant="ghost" disabled={step === 1} />
              <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <span style={{ fontSize: 12, color: C.muted }}>Step {step} of 3</span>
                <Btn
                  label={saving ? 'Saving…' : step === 3 ? 'Complete Intake ✓' : 'Save & Continue →'}
                  onClick={next}
                  variant={step === 3 ? 'success' : 'primary'}
                  disabled={saving}
                />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
