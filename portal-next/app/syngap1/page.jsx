'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a5276';   // deep navy — SYNGAP1 / neuroscience
const ACCENT2 = '#922b21';   // deep red — CBZ/OXC AVOID / drop attack danger
const ACCENT3 = '#1e8449';   // dark green — treatment success / KD
const ACCENT4 = '#6c3483';   // purple — eyelid myoclonia / photosensitivity

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#eaf4fb', color: borderColor }}>
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
      onClick={onClick}
    >{label}</button>
  );
}

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading overview…</div>;
  const { kpis = [], top_clinical_alerts = [], key_concept, etiology_summary = [],
    seizure_prevalence = [], trigger_prevalence = [], gene, locus, protein, condition, omim } = data;

  const kpiColors = [ACCENT, ACCENT4, ACCENT4, '#d35400', ACCENT2, ACCENT, ACCENT3, '#2471a3'];

  return (
    <div>
      {/* Gene badge */}
      <div className="alert alert-primary py-2 mb-3" style={{ fontSize: 13 }}>
        <strong>🧬 {gene}</strong> · Locus: {locus} · Protein: {protein}<br />
        <span className="text-muted">{condition}</span><br />
        <span className="text-muted small">OMIM: {omim}</span>
      </div>

      {/* Clinical alerts */}
      <SectionCard title="⚠️ Critical Clinical Alerts" borderColor={ACCENT2}>
        {top_clinical_alerts.map((a, i) => (
          <Alert key={i} text={a} variant={i < 2 ? 'danger' : i < 4 ? 'warning' : 'info'} />
        ))}
      </SectionCard>

      {/* Key concept */}
      {key_concept && (
        <SectionCard title="🔑 Key Concept — SYNGAPathy" borderColor={ACCENT4}>
          <p className="mb-0" style={{ fontSize: 14 }}>{key_concept}</p>
        </SectionCard>
      )}

      {/* KPIs */}
      <SectionCard title="📊 Cohort KPIs">
        <div className="row">
          {kpis.map((k, i) => (
            <KPI key={i} label={k.label} value={k.value} color={kpiColors[i % kpiColors.length]} />
          ))}
        </div>
      </SectionCard>

      {/* Charts row */}
      <div className="row">
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ color: ACCENT }}>Etiology Classes</div>
            <div className="card-body">
              {etiology_summary.map((e, i) => (
                <PctBar key={i} label={e.label.replace(/-/g, ' ')} pct={e.pct}
                  color={[ACCENT, ACCENT2, ACCENT3, ACCENT4, '#7f8c8d'][i % 5]} />
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ color: ACCENT2 }}>Seizure Types (%)</div>
            <div className="card-body">
              {seizure_prevalence.map((s, i) => (
                <PctBar key={i} label={s.type} pct={s.pct}
                  color={[ACCENT2, ACCENT4, '#e67e22', '#2874a6'][i % 4]} />
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ color: ACCENT4 }}>Top Triggers (%)</div>
            <div className="card-body">
              {trigger_prevalence.map((t, i) => (
                <PctBar key={i} label={t.trigger} pct={t.pct}
                  color={[ACCENT4, '#e67e22', ACCENT2, '#2980b9', '#27ae60', '#8e44ad', '#e74c3c', '#f39c12'][i % 8]} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Tab: Patients & Etiology ──────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading breakdown…</div>;
  const { etiology_catalog = [], patients = [] } = data;

  return (
    <div>
      <SectionCard title="🧬 Etiology Catalog — 5 Classes (N=41)" borderColor={ACCENT}>
        {etiology_catalog.map((e, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ background: '#f8f9fa', borderLeft: `4px solid ${[ACCENT, ACCENT2, ACCENT3, ACCENT4, '#7f8c8d'][i]}` }}>
            <div className="fw-bold mb-1">{e.etiology} <span className="badge bg-primary ms-2">{e.pct}% (n={e.n})</span></div>
            <div className="mb-2"><strong>Mechanism:</strong> <span style={{ fontSize: 13 }}>{e.mechanism}</span></div>
            <div className="mb-1"><strong>EEG Signature:</strong> <span style={{ fontSize: 13 }}>{e.eeg_signature}</span></div>
            <div className="mb-1"><strong>MRI:</strong> <span style={{ fontSize: 13 }}>{e.mri}</span></div>
            <div className="mt-2 p-2 rounded" style={{ background: '#fff3cd', fontSize: 13 }}>
              <strong>📋 Clinical Note:</strong> {e.clinical_note}
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Patient table */}
      <SectionCard title={`👥 Patient Roster (N=${patients.length})`}>
        <div className="table-responsive">
          <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Onset (Y)</th><th>Etiology</th>
                <th>Drops/Day</th><th>Eyelid Myo</th><th>Photo</th><th>ASD</th><th>Treatment</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i}>
                  <td className="fw-bold">{p.patient_id}</td>
                  <td>{p.sex}</td>
                  <td>{p.age_onset_years}</td>
                  <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                    title={p.etiology_class}>{p.etiology_class.replace(/-/g, ' ')}</td>
                  <td>
                    <span className={`badge ${p.drops_per_day_baseline > 10 ? 'bg-danger' : p.drops_per_day_baseline > 3 ? 'bg-warning text-dark' : 'bg-success'}`}>
                      {p.drops_per_day_baseline}
                    </span>
                  </td>
                  <td><span className={`badge ${p.eyelid_myoclonia === 'Yes' ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.eyelid_myoclonia}</span></td>
                  <td><span className={`badge ${p.photosensitive === 'Yes' ? 'bg-info text-dark' : 'bg-secondary'}`}>{p.photosensitive}</span></td>
                  <td><span className={`badge ${p.asd_diagnosis === 'Yes' ? 'bg-primary' : 'bg-secondary'}`}>{p.asd_diagnosis}</span></td>
                  <td style={{ fontSize: 11 }}>{p.current_tx}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Seizure Types & Triggers ─────────────────────────────────────────────
function SeizureTriggersTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading breakdown…</div>;
  const { seizure_types = [], triggers = [] } = data;

  return (
    <div>
      <SectionCard title="⚡ Seizure Types (4)" borderColor={ACCENT2}>
        {seizure_types.map((s, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ background: '#fef9f0', borderLeft: `4px solid ${[ACCENT2, ACCENT4, '#e67e22', '#2874a6'][i]}` }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <div className="fw-bold">{s.type}</div>
              <span className="badge ms-2" style={{ background: ACCENT2 }}>{s.prevalence_pct}%</span>
            </div>
            <div className="small text-muted mb-2">Onset: {s.onset_age}</div>
            <PctBar label="Prevalence" pct={s.prevalence_pct} color={[ACCENT2, ACCENT4, '#e67e22', '#2874a6'][i]} />
            <div className="mb-2"><strong>EEG Correlate:</strong> <span style={{ fontSize: 13 }}>{s.eeg_correlate}</span></div>
            <div className="mt-2 p-2 rounded" style={{ background: '#e8f5e9', fontSize: 13 }}>
              <strong>💡 Clinical Tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔥 Triggers (8)" borderColor={ACCENT4}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: '#f5eef8', borderLeft: `4px solid ${ACCENT4}` }}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <div className="fw-bold">{t.trigger}</div>
              <span className="badge" style={{ background: ACCENT4 }}>{t.rate_pct}%</span>
            </div>
            <PctBar label="Frequency" pct={t.rate_pct} color={ACCENT4} />
            <div className="small mb-1"><strong>Mechanism:</strong> {t.mechanism}</div>
            <div className="small p-2 rounded" style={{ background: '#d5f5e3', fontSize: 12 }}>
              <strong>Management:</strong> {t.management}
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Treatments ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading breakdown…</div>;
  const { treatments = [], contraindications = [], monitoring = [] } = data;

  const levelColor = { 'Level B': ACCENT3, 'Level C': '#2471a3', 'Level C (investigational / off-label)': '#8e44ad', 'Level C (with CAUTION — may exacerbate myoclonic-atonic)': ACCENT2 };

  return (
    <div>
      <SectionCard title="💊 Treatment Ladder (8 drugs)" borderColor={ACCENT3}>
        {treatments.map((t, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ background: '#f0faf4', borderLeft: `4px solid ${ACCENT3}` }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <div className="fw-bold">{t.drug}</div>
              <span className="badge ms-2" style={{ background: levelColor[t.level] || ACCENT3 }}>{t.level}</span>
            </div>
            <div className="small mb-2 text-muted"><em>{t.indication}</em></div>
            <div className="row small">
              <div className="col-md-6 mb-2">
                <strong>Dose:</strong> {t.dose}
              </div>
              <div className="col-md-6 mb-2">
                <strong>Efficacy:</strong> {t.efficacy}
              </div>
              <div className="col-12 mb-2">
                <strong>MOA:</strong> {t.moa}
              </div>
              <div className="col-12 mb-2">
                <div className="p-2 rounded" style={{ background: '#fef9e7', fontSize: 12 }}>
                  <strong>⚠️ Safety:</strong> {t.safety}
                </div>
              </div>
              <div className="col-12">
                <div className="p-2 rounded" style={{ background: '#eaf4fb', fontSize: 12 }}>
                  <strong>🔬 Monitoring:</strong> {t.monitoring}
                </div>
              </div>
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Contraindications */}
      <SectionCard title="🚫 Contraindications (4)" borderColor={ACCENT2}>
        {contraindications.map((c, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: '#fdf2f8', borderLeft: `4px solid ${ACCENT2}` }}>
            <div className="fw-bold mb-1">{c.drug}</div>
            <span className={`badge mb-2 ${c.severity.includes('AVOID') ? 'bg-danger' : 'bg-warning text-dark'}`}>{c.severity}</span>
            <div style={{ fontSize: 13 }}>{c.reason}</div>
          </div>
        ))}
      </SectionCard>

      {/* Monitoring */}
      <SectionCard title="🔬 Monitoring Protocol (8 items)" borderColor="#2471a3">
        {monitoring.map((m, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: '#eaf4fb', borderLeft: '4px solid #2471a3' }}>
            <div className="fw-bold mb-1">{m.item}</div>
            <div className="small mb-1 text-muted">Frequency: <strong>{m.frequency}</strong></div>
            <div style={{ fontSize: 13 }}>{m.rationale}</div>
          </div>
        ))}
      </SectionCard>

      {/* Lifecycle */}
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading definitions…</div>;
  const { concepts = [], standards = [], thresholds = [], references = [] } = data;

  return (
    <div>
      <SectionCard title="📖 Key Concepts (14)" borderColor={ACCENT4}>
        <div className="row">
          {concepts.map((c, i) => (
            <div key={i} className="col-md-6 mb-3">
              <div className="p-2 rounded h-100" style={{ background: '#f5eef8', borderLeft: `3px solid ${ACCENT4}` }}>
                <div className="fw-bold small mb-1" style={{ color: ACCENT4 }}>{c.term.replace(/-/g, ' ')}</div>
                <div style={{ fontSize: 12 }}>{c.definition}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="📏 Clinical Thresholds (10)" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
            <thead className="table-dark">
              <tr><th>Threshold</th><th>Value</th><th>Action</th></tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.threshold}</td>
                  <td><span className="badge bg-warning text-dark">{t.value}</span></td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📋 Clinical Standards (8)" borderColor={ACCENT}>
        {standards.map((s, i) => (
          <div key={i} className="mb-2 p-2 rounded" style={{ background: '#eaf4fb', fontSize: 13 }}>
            <span className="badge me-2" style={{ background: ACCENT }}>{s.code}</span>
            <strong>{s.title}</strong> — {s.scope}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📚 Key References (6)" borderColor={ACCENT3}>
        {references.map((r, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f0faf4', fontSize: 13, borderLeft: `3px solid ${ACCENT3}` }}>
            <div className="fw-bold small mb-1">{r.ref}</div>
            <div className="fst-italic mb-1">{r.citation}</div>
            <div className="text-muted small">{r.impact}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function SYNGAP1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/syngap1/overview`).then(r => r.json()),
      fetch(`${API}/api/syngap1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/syngap1/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
    }).catch(e => setError(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h2 className="fw-bold" style={{ color: ACCENT }}>
          🧬 SYNGAP1 Encephalopathy — SYNGAPathy / SYNGAP1-DEE / MRD5
        </h2>
        <p className="text-muted mb-1" style={{ fontSize: 14 }}>
          41-patient cohort · SYNGAP1 (6p21.32) · SynGAP1 RasGAP · De novo haploinsufficiency ·
          Myoclonic-atonic + Eyelid myoclonia + Eye-closure sensitivity + ASD/ID
        </p>
        <div className="d-flex flex-wrap gap-2 mb-2">
          <span className="badge" style={{ background: ACCENT2 }}>⚠️ AVOID CBZ/OXC/PHT</span>
          <span className="badge" style={{ background: '#d35400' }}>🪖 DROP ATTACK HELMET MANDATORY</span>
          <span className="badge" style={{ background: ACCENT4 }}>👁️ EC-Sensitivity Pathognomonic</span>
          <span className="badge" style={{ background: ACCENT3 }}>🥗 KD for Drug-Resistant Drops</span>
          <span className="badge bg-secondary">💊 VPA+ETH First-Line</span>
        </div>
      </div>

      {error && <div className="alert alert-danger">Error: {error}</div>}

      {/* Tab bar */}
      <div className="mb-3">
        {TABS.map((t, i) => (
          <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />
        ))}
      </div>

      {/* Tab content */}
      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizureTriggersTab data={breakdown} />}
      {tab === 3 && (
        <div>
          <TreatmentsTab data={breakdown} />
          {/* Lifecycle */}
          {breakdown?.lifecycle && (
            <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT}` }}>
              <div className="card-header fw-bold" style={{ backgroundColor: '#eaf4fb', color: ACCENT }}>
                🗓️ Lifecycle Windows (6)
              </div>
              <div className="card-body">
                {breakdown.lifecycle.map((lc, i) => (
                  <div key={i} className="mb-3 p-3 rounded" style={{ background: '#f0f3f4', borderLeft: `3px solid ${ACCENT}` }}>
                    <div className="fw-bold mb-1">{lc.window}</div>
                    <div className="small mb-1"><strong>Key Events:</strong> {lc.key_events}</div>
                    <div className="small text-success"><strong>Management Focus:</strong> {lc.management_focus}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
