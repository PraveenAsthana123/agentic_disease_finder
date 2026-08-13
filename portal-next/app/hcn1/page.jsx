'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — HCN1 / Ih channel
const ACCENT2 = '#b71c1c';   // dark red — CI / danger / fever
const ACCENT3 = '#1b5e20';   // dark green — KD / success / precision
const ACCENT4 = '#e65100';   // deep orange — GOF / temperature / fever alert

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function PctBar({ label, pct, color = ACCENT }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header fw-bold" style={{ backgroundColor: '#eef2fb', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function TabBtn({ label, active, onClick }) {
  return (
    <button
      className={`btn btn-sm me-1 mb-1 ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
      style={active ? { backgroundColor: ACCENT, borderColor: ACCENT } : {}}
      onClick={onClick}
    >{label}</button>
  );
}

// ── Tab: Overview ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const k = data.kpis || {};

  return (
    <>
      <div className="alert alert-primary mb-4" style={{ borderLeft: `5px solid ${ACCENT}`, fontSize: 14 }}>
        <strong>⚡ {data.syndrome}</strong><br />
        Gene: <strong>{data.gene}</strong> · {data.inheritance}<br />
        <strong>EEG hallmark:</strong> {data.eeg_hallmark}<br />
        <strong>Key biomarker:</strong> {data.key_biomarker}
      </div>

      {/* Dual-mechanism banner */}
      <div className="alert mb-3" style={{ backgroundColor: '#fff3e0', borderLeft: `5px solid ${ACCENT4}`, fontSize: 13 }}>
        <strong>🔀 DUAL-MECHANISM CHANNELOPATHY (like KCNA2):</strong><br />
        <span className="text-danger fw-semibold">GOF ({k.gof_pct}%)</span> — Constitutive Ih → chronic depolarisation → Dravet-like fever-sensitive DEE<br />
        <span className="text-primary fw-semibold">LOF ({k.lof_pct}%)</span> — Reduced Ih → thalamo-cortical burst rebound + excess dendritic integration → DEE<br />
        <strong>→ GOF/LOF functional electrophysiology assay MANDATORY before LTG, IVM, or NaV-blocker decisions.</strong>
      </div>

      {/* Critical Alerts */}
      {(data.critical_alerts || []).map((a, i) => (
        <div key={i} className={`alert alert-${a.color} py-2 mb-2`} style={{ fontSize: 13 }}>
          <strong>⚠️ {a.severity}:</strong> {a.alert}<br />
          <span className="text-muted small">→ {a.action}</span>
        </div>
      ))}

      {/* KPI tiles */}
      <div className="row g-2 mb-4 mt-2">
        {[
          ['Patients', k.total_patients, ACCENT],
          ['GOF', `${k.gof_pct ?? '—'}%`, ACCENT4],
          ['LOF', `${k.lof_pct ?? '—'}%`, '#1565c0'],
          ['Fever-Sensitive', `${k.fever_pct ?? '—'}%`, ACCENT2],
          ['On KD', `${k.kd_pct ?? '—'}%`, ACCENT3],
          ['Drug-Resistant', `${k.drug_resistant_pct ?? '—'}%`, ACCENT2],
          ['FFA Used', k.ffa_used, '#7b1fa2'],
          ['STP Used', k.stp_used, '#006064'],
          ['LTG Worsened', k.ltg_worsened, ACCENT2],
          ['Etiology Classes', k.etiology_classes, ACCENT],
          ['Seizure Types', k.seizure_types, ACCENT],
          ['Treatments', k.treatments, '#004d40'],
        ].map(([label, val, color]) => (
          <KPI key={label} label={label} value={val ?? '—'} color={color} />
        ))}
      </div>

      {/* Pathway Summary */}
      <SectionCard title="🗺️ Clinical Management Pathway" borderColor={ACCENT3}>
        <p className="small mb-0" style={{ lineHeight: 1.7 }}>{data.pathway_summary}</p>
      </SectionCard>

      {/* EEG Hallmarks */}
      {data.eeg_hallmarks && (
        <SectionCard title="🧠 EEG Hallmarks — HCN1-DEE24" borderColor={ACCENT4}>
          <ul className="mb-0 small">
            {data.eeg_hallmarks.map((h, i) => <li key={i}>{h}</li>)}
          </ul>
        </SectionCard>
      )}

      {/* Temperature-Ih mechanism */}
      <SectionCard title="🌡️ Fever Mechanism — Ih Q10 Amplification (GOF)" borderColor={ACCENT2}>
        <div className="small mb-2">
          <strong>Fever trigger in {k.fever_pct}% of patients</strong> — HCN1-GOF Ih has Q10 ≈ 1.4-1.7.
          Each +1°C above 37°C raises Ih by ~15-20%. At 38°C: +15-20% Ih; at 39°C: +30-40% Ih (on top of constitutive GOF Ih).
        </div>
        <div className="d-flex gap-3 flex-wrap">
          {[['37°C', 'Normal Ih', 'bg-success'], ['38°C', 'Ih +15-20% ⚠️', 'bg-warning text-dark'], ['39°C', 'Ih +30-40% 🚨', 'bg-danger']].map(([t, label, cls]) => (
            <div key={t} className="text-center">
              <div className={`badge ${cls} px-3 py-2 fs-6`}>{t}</div>
              <div className="small text-muted mt-1">{label}</div>
            </div>
          ))}
        </div>
        <div className="mt-3 p-2 rounded" style={{ backgroundColor: '#fff3e0', fontSize: 12 }}>
          <strong>Action threshold:</strong> Paracetamol at <strong>37.5°C</strong> (not 38.5°C standard).
          No hot baths. Buccal midazolam <strong>0.3 mg/kg</strong> (not 0.2) if seizure &gt;3 min.
          Written fever plan at EVERY outpatient visit.
        </div>
      </SectionCard>

      {/* Standards */}
      <SectionCard title="📋 Evidence Standards" borderColor="#546e7a">
        <ul className="mb-0 small">
          {(data.standards || []).map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Etiology ──────────────────────────────────────────────────
function EtiologyTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const [sel, setSel] = useState(null);

  return (
    <>
      <SectionCard title="📊 Etiology Distribution — 5 Classes (N=41)" borderColor={ACCENT}>
        {(data.etiology_distribution || []).map((e, i) => (
          <div key={i} className="mb-3">
            <div
              className="d-flex justify-content-between align-items-center mb-1"
              onClick={() => setSel(sel === i ? null : i)}
              style={{ cursor: 'pointer' }}
            >
              <span className="small fw-semibold">{e.etiology}</span>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{e.n} / {e.pct}%</span>
            </div>
            <div className="progress mb-1" style={{ height: 12 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT }} />
            </div>
            {sel === i && (
              <div className="border rounded p-2 mt-1 bg-light small">
                <strong>Mechanism:</strong> {e.mechanism_short}<br />
                <strong>EEG:</strong> {e.eeg_signature_short}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      {/* Patient table */}
      <SectionCard title="👥 Patient Summary Table (N=41)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-hover" style={{ fontSize: 11 }}>
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Age(mo)</th><th>Onset(mo)</th><th>Etiology</th>
                <th>GOF/LOF</th><th>Fever</th><th>VPA</th><th>POLG</th><th>KD</th>
                <th>β-OHB</th><th>STP</th><th>FFA</th><th>LTG</th><th>LTG↓</th><th>Control</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients_sample || []).map(p => (
                <tr key={p.id}>
                  <td><span className="badge" style={{ backgroundColor: ACCENT }}>{p.id}</span></td>
                  <td>{p.age_months}</td>
                  <td>{p.onset_months}</td>
                  <td><span className="text-truncate d-inline-block" style={{ maxWidth: 75 }} title={p.etiology_category}>{p.etiology_category.split('-').slice(0,3).join('-')}</span></td>
                  <td>
                    {p.is_gof && <span className="badge" style={{ backgroundColor: ACCENT4 }}>GOF</span>}
                    {p.is_lof && <span className="badge bg-primary">LOF</span>}
                    {!p.is_gof && !p.is_lof && <span className="badge bg-secondary">?</span>}
                  </td>
                  <td><span className={`badge ${p.fever_sensitive ? 'bg-danger' : 'bg-secondary'}`}>{p.fever_sensitive ? '🌡️Y' : 'N'}</span></td>
                  <td><span className={`badge ${p.vpa_used ? 'bg-success' : 'bg-secondary'}`}>{p.vpa_used ? 'Y' : 'N'}</span></td>
                  <td><span className={`badge ${p.polg_tested === 'Y' ? 'bg-success' : p.polg_tested === 'N' ? 'bg-danger' : 'bg-secondary'}`}>{p.polg_tested}</span></td>
                  <td><span className={`badge ${p.on_kd ? 'bg-success' : 'bg-secondary'}`}>{p.on_kd ? 'Y' : 'N'}</span></td>
                  <td><span className="text-muted small">{p.kd_ketosis_mmol ? `${p.kd_ketosis_mmol}` : '—'}</span></td>
                  <td><span className={`badge ${p.stiripentol ? 'bg-info text-dark' : 'bg-secondary'}`}>{p.stiripentol ? 'Y' : 'N'}</span></td>
                  <td><span className={`badge ${p.ffa_used ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.ffa_used ? 'Y' : 'N'}</span></td>
                  <td><span className={`badge ${p.ltg_used ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.ltg_used ? 'Y' : 'N'}</span></td>
                  <td><span className={`badge ${p.ltg_worsened ? 'bg-danger' : 'bg-secondary'}`}>{p.ltg_worsened ? '⚠️Y' : 'N'}</span></td>
                  <td><span className={`badge ${p.seizure_control === 'drug-resistant' ? 'bg-danger' : p.seizure_control === 'partially-controlled' ? 'bg-warning text-dark' : 'bg-success'}`}>{p.seizure_control}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* GOF/LOF summary */}
      <SectionCard title="📈 GOF vs LOF Summary" borderColor={ACCENT3}>
        <div className="row text-center">
          {[
            ['GOF (Constitutive Ih)', data.summary?.gof_pct, ACCENT4],
            ['LOF (Reduced Ih)', data.summary?.lof_pct, '#1565c0'],
            ['Drug-Resistant', data.summary?.drug_resistant_pct, ACCENT2],
          ].map(([label, val, color]) => (
            <div key={label} className="col-4">
              <div className="fw-bold fs-5" style={{ color }}>{val ?? '—'}%</div>
              <div className="small text-muted">{label}</div>
            </div>
          ))}
        </div>
        {data.summary?.ltg_worsened > 0 && (
          <Alert text={`⚠️ ${data.summary.ltg_worsened} patient(s) experienced seizure WORSENING on LTG (LOF HCN1) — LTG contraindicated in LOF.`} variant="danger" />
        )}
        {data.summary?.vpa_without_polg > 0 && (
          <Alert text={`⚠️ ${data.summary.vpa_without_polg} patient(s) received VPA WITHOUT documented POLG testing — immediate action required.`} variant="danger" />
        )}
      </SectionCard>
    </>
  );
}

// ── Tab: Seizure Types & Triggers ─────────────────────────────────────────────
function SeizureTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const [selSeizure, setSelSeizure] = useState(null);
  const [selTrigger, setSelTrigger] = useState(null);

  return (
    <>
      <SectionCard title="⚡ Seizure Type Prevalence" borderColor={ACCENT}>
        {(data.seizure_types || []).map((s, i) => (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span>{s.type}</span>
              <span className="text-muted">{s.pct}%</span>
            </div>
            <div className="progress" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${s.pct}%`, backgroundColor: ACCENT }} />
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔬 Seizure Type Detail (click to expand)" borderColor={ACCENT4}>
        {(data.seizure_detail || []).map((s, i) => (
          <div key={i} className="mb-3 border rounded p-2">
            <div
              className="d-flex justify-content-between align-items-center"
              style={{ cursor: 'pointer' }}
              onClick={() => setSelSeizure(selSeizure === i ? null : i)}
            >
              <span className="fw-semibold small">{s.type} — {s.prevalence_pct}%</span>
              <span className="text-muted small">{selSeizure === i ? '▲' : '▼'}</span>
            </div>
            {selSeizure === i && (
              <div className="mt-2 small">
                <div className="mb-2"><strong>Semiology:</strong> {s.semiology}</div>
                <div className="mb-2"><strong>EEG Pattern:</strong> {s.eeg_pattern}</div>
                <div className="p-2 rounded" style={{ backgroundColor: '#fff8e1' }}>
                  <strong>⚡ Clinical Tip:</strong> {s.clinical_tip}
                </div>
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🌡️ Trigger Prevalence (Fever #1 — 92%)" borderColor={ACCENT2}>
        {(data.triggers || []).map((t, i) => (
          <PctBar key={i} label={t.trigger} pct={t.pct} color={
            t.pct >= 90 ? ACCENT2 : t.pct >= 70 ? '#e65100' : t.pct >= 50 ? '#f57c00' : '#558b2f'
          } />
        ))}
      </SectionCard>

      <SectionCard title="⚠️ Trigger Detail — Management Protocols (click to expand)" borderColor="#6a1b9a">
        {(data.trigger_detail || []).map((t, i) => (
          <div key={i} className="mb-2 border rounded p-2">
            <div
              className="d-flex justify-content-between"
              style={{ cursor: 'pointer' }}
              onClick={() => setSelTrigger(selTrigger === i ? null : i)}
            >
              <span className="fw-semibold small">{t.trigger} — {t.prevalence_pct}%</span>
              <span className="text-muted small">{selTrigger === i ? '▲' : '▼'}</span>
            </div>
            {selTrigger === i && (
              <div className="mt-2 small">
                <div className="mb-1"><strong>Mechanism:</strong> {t.mechanism}</div>
                <div className="p-2 rounded" style={{ backgroundColor: '#f1f8e9' }}>
                  <strong>Management:</strong> {t.management}
                </div>
              </div>
            )}
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const [selTx, setSelTx] = useState(null);
  const [selCI, setSelCI] = useState(null);

  return (
    <>
      <Alert
        text="⚠️ HCN1-DEE24 SAFETY: LTG CI in LOF (Ih blocker worsens LOF) · IVM CI in GOF (activates HCN1 worsens constitutive Ih) · CBZ/OXC/PHT caution (Dravet-like aggravation) · Hot bath PROHIBITED · POLG before VPA · Fever threshold 37.5°C"
        variant="danger"
      />

      {/* GOF vs LOF treatment split */}
      <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
        <div className="card-header fw-bold" style={{ backgroundColor: '#fff3e0', color: ACCENT4 }}>
          🔀 GOF vs LOF Treatment Divergence
        </div>
        <div className="card-body">
          <div className="row">
            <div className="col-md-6 mb-3">
              <div className="fw-bold mb-2" style={{ color: ACCENT4 }}>GOF HCN1 (Constitutive Ih)</div>
              <ul className="small mb-0">
                <li>VPA + CLB first-line (both)</li>
                <li>Add Stiripentol (Dravet-like protocol)</li>
                <li>Add Fenfluramine (σ-1R + 5-HT2C)</li>
                <li>LTG: may help but insufficient evidence — avoid</li>
                <li><strong className="text-danger">IVM: ABSOLUTE CI</strong></li>
                <li>Hot bath: ABSOLUTE prohibition</li>
              </ul>
            </div>
            <div className="col-md-6 mb-3">
              <div className="fw-bold mb-2" style={{ color: '#1565c0' }}>LOF HCN1 (Reduced Ih)</div>
              <ul className="small mb-0">
                <li>VPA + CLB first-line (both)</li>
                <li>Add ETX for absence-like seizures (T-type Ca2+ block)</li>
                <li>IVM: investigational (Ih activator) — LOF only</li>
                <li><strong className="text-danger">LTG: ABSOLUTE CI (Ih blocker worsens LOF)</strong></li>
                <li>KD: mechanistically ideal (Ih-independent)</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <SectionCard title="💊 Treatment Ladder (click for full detail)" borderColor={ACCENT3}>
        {(data.treatment_detail || []).map((t, i) => (
          <div key={i} className="mb-3 border rounded p-3"
            style={{ borderLeft: `4px solid ${t.evidence.startsWith('Level A') ? ACCENT3 : t.evidence.startsWith('Level B') ? '#1565c0' : '#6d4c41'}` }}
          >
            <div
              className="d-flex justify-content-between align-items-start"
              style={{ cursor: 'pointer' }}
              onClick={() => setSelTx(selTx === i ? null : i)}
            >
              <div>
                <span className="fw-bold">{t.name}</span>
                <span className="badge ms-2" style={{ backgroundColor: t.evidence.startsWith('Level A') ? ACCENT3 : t.evidence.startsWith('Level B') ? '#1565c0' : '#6d4c41' }}>
                  {t.evidence.split('—')[0].trim()}
                </span>
              </div>
              <span className="text-muted small">{selTx === i ? '▲' : '▼'}</span>
            </div>
            <div className="small text-muted mt-1">{t.status}</div>
            {selTx === i && (
              <div className="mt-3 small">
                <div className="mb-2 p-2 rounded" style={{ backgroundColor: '#e8f5e9' }}>
                  <strong>Dose:</strong> {t.dose}
                </div>
                <div className="mb-2"><strong>Mechanism:</strong> {t.moa_short}</div>
                <div className="mb-2 p-2 rounded" style={{ backgroundColor: '#e3f2fd' }}>
                  <strong>Efficacy:</strong> {t.efficacy}
                </div>
                <div className="mb-2 p-2 rounded" style={{ backgroundColor: '#fff3e0' }}>
                  <strong>Safety:</strong> {t.safety_short}
                </div>
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications — HCN1-DEE24" borderColor={ACCENT2}>
        {(data.contraindications || []).map((c, i) => (
          <div key={i} className="mb-2 border border-danger rounded p-2"
            style={{ cursor: 'pointer' }}
            onClick={() => setSelCI(selCI === i ? null : i)}
          >
            <div className="d-flex justify-content-between">
              <span className="fw-bold text-danger small">{c.drug}</span>
              <span className={`badge ${c.severity.startsWith('ABSOLUTE') ? 'bg-danger' : c.severity.startsWith('HIGH') ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                {c.severity.split('—')[0].trim()}
              </span>
            </div>
            {selCI === i && (
              <div className="mt-2 small">
                <div className="mb-1"><strong>Mechanism:</strong> {c.mechanism_short}</div>
                <div className="p-2 rounded" style={{ backgroundColor: '#ffebee' }}>
                  <strong>Action:</strong> {c.action}
                </div>
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔍 Monitoring Protocol" borderColor="#0277bd">
        <div className="table-responsive">
          <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
            <thead className="table-primary">
              <tr><th>Item</th><th>Frequency</th><th>Threshold / Action</th></tr>
            </thead>
            <tbody>
              {(data.monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td><strong>{m.item}</strong><br /><span className="text-muted">{m.rationale}</span></td>
                  <td><span className="badge bg-info text-dark">{m.frequency}</span></td>
                  <td className="text-danger small">{m.threshold}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🗂️ Lifecycle Management Windows" borderColor={ACCENT}>
        {(data.lifecycle || []).map((lc, i) => (
          <div key={i} className="mb-3 border-start border-3 ps-3" style={{ borderColor: ACCENT }}>
            <div className="fw-bold small">{lc.window}</div>
            <div className="text-muted small">{lc.focus}</div>
            <div className="small mt-1"><strong>Interventions:</strong> {lc.interventions}</div>
            <div className="small"><strong>Goals:</strong> {lc.goals}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const [selDef, setSelDef] = useState(null);

  return (
    <>
      <SectionCard title={`📖 Key Concepts — ${data.total_definitions} Definitions`} borderColor={ACCENT}>
        {(data.definitions || []).map((d, i) => (
          <div key={i} className="mb-2 border rounded p-2"
            style={{ cursor: 'pointer', borderLeft: `3px solid ${ACCENT}` }}
            onClick={() => setSelDef(selDef === i ? null : i)}
          >
            <div className="fw-semibold small" style={{ color: ACCENT }}>{d.term}</div>
            {selDef === i && (
              <div className="mt-1 small text-muted">{d.definition}</div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="⚡ Thresholds — Action Triggers" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
            <thead className="table-danger">
              <tr><th>Threshold / Rule</th><th>Value</th></tr>
            </thead>
            <tbody>
              {(data.thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td>{t.threshold}</td>
                  <td><span className="badge bg-danger">{t.value}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📚 Evidence Standards" borderColor="#546e7a">
        <ul className="mb-0 small">
          {(data.standards || []).map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="📄 Key References" borderColor={ACCENT3}>
        <ol className="mb-0 small">
          {(data.references || []).map((r, i) => <li key={i}>{r}</li>)}
        </ol>
      </SectionCard>
    </>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function HCN1Page() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('Overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hcn1/overview`).then(r => r.json()),
      fetch(`${API}/api/hcn1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hcn1/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOverview(o); setBreakdown(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!overview) return <div className="text-muted p-3">Loading HCN1 / DEE24 dashboard…</div>;

  return (
    <div className="container-fluid py-3">
      <h3 style={{ color: ACCENT }}>⚡ HCN1 Epilepsy — DEE24 / Ih Channelopathy (Dual GOF-LOF / Fever-Sensitive)</h3>
      <p className="text-muted small">
        {overview.gene} · {overview.inheritance} · {overview.total} patients · Dashboard #{overview.dashboard_number}
      </p>

      {/* Tab buttons */}
      <div className="mb-3">
        {TABS.map(t => (
          <TabBtn key={t} label={t} active={tab === t} onClick={() => setTab(t)} />
        ))}
      </div>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Patients & Etiology' && <EtiologyTab data={breakdown} />}
      {tab === 'Seizure Types & Triggers' && <SeizureTab data={breakdown} />}
      {tab === 'Treatments' && <TreatmentsTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={defs} />}

      <div className="text-muted small mt-4 border-top pt-2">
        HCN1 (5p12) · Ih funny current · DEE24 GOF/LOF ·
        LTG CI in LOF · IVM CI in GOF · Fever 37.5°C threshold · VPA+CLB+STP/FFA (GOF) · VPA+CLB+ETX+KD (LOF) · N={overview.total}
      </div>
    </div>
  );
}
