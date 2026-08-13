'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — X-linked / ARX
const ACCENT2 = '#b71c1c';   // dark red — CI / danger
const ACCENT3 = '#e65100';   // deep orange — fever / XLAG alert
const ACCENT4 = '#1b5e20';   // deep green — West / ACTH
const ACCENT5 = '#4a148c';   // purple — interneuron / MGE

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
      <div className="card-header py-2" style={{ backgroundColor: '#f8f9fa', borderBottom: `1px solid ${borderColor}20` }}>
        <strong style={{ color: borderColor }}>{title}</strong>
      </div>
      <div className="card-body py-3">{children}</div>
    </div>
  );
}

function TabBtn({ label, active, onClick }) {
  return (
    <button
      className={`btn btn-sm me-2 mb-2 ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
      style={active ? { backgroundColor: ACCENT, borderColor: ACCENT } : {}}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

// ── Tab 1: Overview ───────────────────────────────────────────────────────────
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <div className="alert alert-danger fw-bold py-2 mb-3" style={{ fontSize: 13 }}>
        🧬 <strong>KEY AHA:</strong> {ov.key_aha}
      </div>

      <div className="row g-2 mb-3">
        <div className="col-12 col-md-6">
          <div className="alert py-2 mb-0" style={{ fontSize: 12, backgroundColor: '#fce4ec', borderColor: ACCENT2, color: '#880e4f' }}>
            🚨 <strong>TIAGABINE:</strong> {ov.tiagabine_alert}
          </div>
        </div>
        <div className="col-12 col-md-6">
          <div className="alert py-2 mb-0" style={{ fontSize: 12, backgroundColor: '#fff3e0', borderColor: ACCENT3, color: '#bf360c' }}>
            ⚠️ <strong>CBZ/OXC/PHT:</strong> {ov.cbz_alert}
          </div>
        </div>
      </div>

      <div className="alert py-2 mb-3" style={{ fontSize: 12, backgroundColor: '#f3e5f5', borderColor: ACCENT5, color: '#4a148c' }}>
        🏥 <strong>POLG ALERT:</strong> {ov.polg_alert}
      </div>

      <div className="alert py-2 mb-3" style={{ fontSize: 12, backgroundColor: '#fff8e1', borderColor: ACCENT3, color: '#e65100' }}>
        🧠 <strong>XLAG ALERT:</strong> {ov.xlag_alert}
      </div>

      <SectionCard title="🧬 ARX Gene & Syndrome" borderColor={ACCENT}>
        <div className="row g-2">
          {[
            ['Gene', ov.gene],
            ['Locus', ov.locus],
            ['Inheritance', ov.inheritance],
            ['Protein', ov.protein],
          ].map(([k, v]) => (
            <div className="col-12 col-md-6" key={k}>
              <strong>{k}:</strong> <span className="text-secondary">{v}</span>
            </div>
          ))}
          <div className="col-12 mt-1">
            <strong>Mechanism:</strong> <span className="text-secondary">{ov.mechanism}</span>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="📊 Cohort KPIs (n={ov.n_patients})" borderColor={ACCENT4}>
        <div className="row g-2">
          <KPI label="Ohtahara" value={`${ov.ohtahara_pct}%`} color={ACCENT2} />
          <KPI label="West Syndrome" value={`${ov.west_pct}%`} color={ACCENT4} />
          <KPI label="Partington" value={`${ov.partington_pct}%`} color={ACCENT5} />
          <KPI label="XLAG" value={`${ov.xlag_pct}%`} color={ACCENT3} />
          <KPI label="DRE" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
          <KPI label="On ACTH" value={`${ov.on_acth_pct}%`} color={ACCENT4} />
          <KPI label="VGB (West)" value={`${ov.on_vgb_pct}%`} color={ACCENT4} />
          <KPI label="On KD" value={`${ov.on_kd_pct}%`} color={ACCENT5} />
          <KPI label="POLG Done" value={`${ov.polg_done_pct}%`} color={ov.polg_done_pct >= 80 ? ACCENT4 : ACCENT2} />
          <KPI label="Counselled" value={`${ov.genetic_counselled_pct}%`} color={ACCENT4} />
          <KPI label="Avg Onset" value={`${ov.avg_onset_months}m`} color={ACCENT} />
          <KPI label="VPA w/o POLG" value={ov.vpa_without_polg} color={ov.vpa_without_polg > 0 ? ACCENT2 : ACCENT4} />
        </div>
      </SectionCard>

      <SectionCard title="⚠️ Contraindications Summary" borderColor={ACCENT2}>
        {(ov.contraindications_summary || []).map((c, i) => (
          <div key={i} className="alert alert-danger py-1 mb-1" style={{ fontSize: 12 }}>{c}</div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Key Thresholds" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm mb-0" style={{ fontSize: 12 }}>
            <thead><tr><th>Parameter</th><th>Value</th><th>Action</th></tr></thead>
            <tbody>
              {(ov.thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td><strong>{t.param}</strong></td>
                  <td style={{ color: ACCENT3 }}>{t.value}</td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Etiology ────────────────────────────────────────────────
function PatientsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const { summary, etiology_distribution, patients_sample } = bk;
  return (
    <div>
      <SectionCard title="📊 Cohort Summary" borderColor={ACCENT}>
        <div className="row g-2">
          {[
            ['Total Patients', summary.n, ACCENT],
            ['Ohtahara', `${summary.ohtahara_pct}%`, ACCENT2],
            ['West Syndrome', `${summary.west_pct}%`, ACCENT4],
            ['Partington', `${summary.partington_pct}%`, ACCENT5],
            ['XLAG', `${summary.xlag_pct}%`, ACCENT3],
            ['DRE', `${summary.drug_resistant_pct}%`, ACCENT2],
            ['ACTH', `${summary.on_acth_pct}%`, ACCENT4],
            ['VGB', `${summary.on_vgb_pct}%`, ACCENT4],
            ['KD', `${summary.on_kd_pct}%`, ACCENT5],
            ['POLG Done', `${summary.polg_done_pct}%`, summary.polg_done_pct >= 80 ? ACCENT4 : ACCENT2],
            ['VPA w/o POLG', summary.vpa_without_polg, summary.vpa_without_polg > 0 ? ACCENT2 : ACCENT4],
          ].map(([k, v, c]) => (
            <div className="col-6 col-md-3" key={k}>
              <div className="card text-center py-2 shadow-sm">
                <div className="fw-bold" style={{ color: c }}>{v}</div>
                <div className="small text-muted">{k}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🧬 Etiology Distribution (5 Classes)" borderColor={ACCENT5}>
        {(etiology_distribution || []).map((e, i) => (
          <div key={i} className="mb-3 p-2 border rounded" style={{ borderLeft: `4px solid ${ACCENT5} !important` }}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong style={{ fontSize: 13 }}>{e.etiology}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT5 }}>{e.pct}% (n≈{e.n})</span>
            </div>
            <div style={{ fontSize: 12, color: '#555' }}>{e.mechanism_short}…</div>
            <div className="mt-1" style={{ fontSize: 12, color: '#777' }}>EEG: {e.eeg_signature_short}…</div>
            <div className="progress mt-1" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT5 }} />
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="👥 Patient Sample (first 15)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-hover" style={{ fontSize: 11 }}>
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Name</th><th>Sex</th><th>Age(y)</th><th>Onset(m)</th>
                <th>Phenotype</th><th>DRE</th><th>ACTH</th><th>VGB</th><th>KD</th><th>POLG</th><th>Counselled</th>
              </tr>
            </thead>
            <tbody>
              {(patients_sample || []).map(p => (
                <tr key={p.id}>
                  <td><code>{p.id}</code></td>
                  <td>{p.name}</td>
                  <td>{p.sex}</td>
                  <td>{p.age_y}</td>
                  <td>{p.onset_months}</td>
                  <td style={{ fontSize: 10 }}>
                    {p.ohtahara ? '🔴 Ohtahara' : p.west_syndrome ? '🟡 West' : p.partington ? '🟢 Partington' : p.xlag ? '🟠 XLAG' : '⚫ Pheno'}
                  </td>
                  <td>{p.drug_resistant ? <span style={{ color: ACCENT2 }}>Yes</span> : 'No'}</td>
                  <td>{p.on_acth ? '✓' : '—'}</td>
                  <td>{p.on_vgb ? '✓' : '—'}</td>
                  <td>{p.on_kd ? '✓' : '—'}</td>
                  <td style={{ color: p.polg_tested === 'Y' ? ACCENT4 : ACCENT2 }}>{p.polg_tested}</td>
                  <td>{p.genetic_counselling ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Seizures & Triggers ────────────────────────────────────────────────
function SeizuresTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const { seizure_detail, trigger_detail } = bk;
  return (
    <div>
      <SectionCard title="⚡ Seizure Types (ARX)" borderColor={ACCENT}>
        {(seizure_detail || []).map((s, i) => (
          <div key={i} className="mb-3 p-2 border rounded">
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ color: ACCENT, fontSize: 13 }}>{s.type}</strong>
              <span className="badge bg-primary">{s.prevalence_pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${s.prevalence_pct}%`, backgroundColor: ACCENT }} />
            </div>
            <div style={{ fontSize: 12 }}><strong>Semiology:</strong> {s.semiology}</div>
            <div style={{ fontSize: 12 }} className="mt-1"><strong>EEG:</strong> {s.eeg_pattern}</div>
            <div className="alert alert-info py-1 mt-1 mb-0" style={{ fontSize: 12 }}>
              💡 <strong>Tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🌡️ Seizure Triggers (ARX)" borderColor={ACCENT3}>
        {(trigger_detail || []).map((t, i) => (
          <div key={i} className="mb-2 p-2 border rounded">
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ fontSize: 13 }}>{t.trigger}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.prevalence_pct}%</span>
            </div>
            <div className="progress mb-1" style={{ height: 5 }}>
              <div className="progress-bar" style={{ width: `${t.prevalence_pct}%`, backgroundColor: ACCENT3 }} />
            </div>
            <div style={{ fontSize: 12, color: '#555' }}>{t.detail}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Treatments ─────────────────────────────────────────────────────────
function TreatmentsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const { treatment_detail, contraindication_detail, monitoring, lifecycle } = bk;
  return (
    <div>
      <SectionCard title="💊 Treatments (ARX — Evidence-Based)" borderColor={ACCENT4}>
        {(treatment_detail || []).map((t, i) => (
          <div key={i} className="mb-3 p-2 border rounded" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ color: ACCENT4, fontSize: 13 }}>{t.drug}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.level}</span>
            </div>
            <div style={{ fontSize: 12 }}><strong>Indication:</strong> {t.indication}</div>
            <div style={{ fontSize: 12 }}><strong>Dose:</strong> {t.dose}</div>
            <div style={{ fontSize: 12 }}><strong>MOA:</strong> {t.moa}</div>
            <div style={{ fontSize: 12 }}><strong>Efficacy:</strong> {t.efficacy}</div>
            <div style={{ fontSize: 12 }}><strong>Safety:</strong> {t.safety}</div>
            <div style={{ fontSize: 12 }}><strong>Monitoring:</strong> {t.monitoring}</div>
            {t.arx_specific && (
              <div className="alert py-1 mt-1 mb-0" style={{ fontSize: 12, backgroundColor: '#e8f5e9', borderColor: ACCENT4, color: '#1b5e20' }}>
                🧬 <strong>ARX-specific:</strong> {t.arx_specific}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications (ARX)" borderColor={ACCENT2}>
        {(contraindication_detail || []).map((c, i) => (
          <div key={i} className="mb-3 alert alert-danger py-2" style={{ fontSize: 12 }}>
            <div className="fw-bold mb-1">{c.drug}</div>
            <div><strong>Risk:</strong> {c.risk}</div>
            <div className="mt-1"><strong>Reason:</strong> {c.reason}</div>
            <div className="mt-1"><strong>Mechanism:</strong> {c.mechanism}</div>
            {c.alternative && <div className="mt-1"><strong>Alternative:</strong> {c.alternative}</div>}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📋 Monitoring Protocol" borderColor={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm mb-0" style={{ fontSize: 12 }}>
            <thead className="table-light">
              <tr><th>Monitoring Item</th><th>Frequency</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td><strong>{m.item}</strong></td>
                  <td style={{ color: ACCENT5 }}>{m.frequency}</td>
                  <td>{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📅 Disease Lifecycle (ARX — 6 Windows)" borderColor={ACCENT}>
        {(lifecycle || []).map((l, i) => (
          <div key={i} className="mb-2 p-2 border rounded">
            <div className="fw-bold" style={{ color: ACCENT, fontSize: 13 }}>{l.window}</div>
            <div style={{ fontSize: 12 }}><strong>Events:</strong> {l.key_events}</div>
            <div style={{ fontSize: 12 }}><strong>Actions:</strong> {l.actions}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 5: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ def }) {
  if (!def) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="📖 Key Concepts (ARX)" borderColor={ACCENT}>
        {(def.concepts || []).map((c, i) => (
          <div key={i} className="mb-2 p-2 border-bottom">
            <div className="fw-bold" style={{ color: ACCENT, fontSize: 13 }}>{c.term}</div>
            <div style={{ fontSize: 12, color: '#444' }}>{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Thresholds" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm mb-0" style={{ fontSize: 12 }}>
            <thead className="table-light">
              <tr><th>Parameter</th><th>Value</th><th>Action</th></tr>
            </thead>
            <tbody>
              {(def.thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td><strong>{t.param}</strong></td>
                  <td style={{ color: ACCENT3 }}>{t.value}</td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📚 Standards & References" borderColor={ACCENT5}>
        <div className="mb-2">
          <strong>Standards:</strong>
          <ul className="mb-0" style={{ fontSize: 12 }}>
            {(def.standards || []).map((s, i) => <li key={i}>{s}</li>)}
          </ul>
        </div>
        <div>
          <strong>References:</strong>
          <ol className="mb-0" style={{ fontSize: 12 }}>
            {(def.references || []).map((r, i) => <li key={i}>{r}</li>)}
          </ol>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function ARXPage() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [def, setDef] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/arx/overview`).then(r => r.json()),
      fetch(`${API}/api/arx/breakdown`).then(r => r.json()),
      fetch(`${API}/api/arx/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDef(d); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          🧬 ARX Epilepsy (X-linked DEE / Ohtahara / West / XLAG / Partington — Xp21.3)
        </h4>
        <div className="text-muted small">
          ARX — Aristaless-Related Homeobox · Paired-type Homeodomain Transcription Factor ·
          X-linked (Xp21.3) · MGE Interneuron Migration Master Regulator ·
          Most Common X-linked Epileptic Encephalopathy Gene
        </div>
      </div>

      {err && <div className="alert alert-danger">Error: {err}</div>}

      <div className="mb-3 border-bottom pb-2">
        {TABS.map((t, i) => (
          <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />
        ))}
      </div>

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <PatientsTab bk={bk} />}
      {tab === 2 && <SeizuresTab bk={bk} />}
      {tab === 3 && <TreatmentsTab bk={bk} />}
      {tab === 4 && <DefinitionsTab def={def} />}
    </div>
  );
}
