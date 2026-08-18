'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#00695c';   // deep teal — GluN2D / subcortical / extrasynaptic
const ACCENT2 = '#b71c1c';   // dark red — CI / danger
const ACCENT3 = '#1b5e20';   // dark green — memantine GOF / success
const ACCENT4 = '#e65100';   // deep orange — fever / GOF alert / movement disorder
const ACCENT5 = '#4a148c';   // purple — LOF / investigational

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#e0f2f1', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
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

      <SectionCard title="🧬 GRIN2D Gene & Syndrome" borderColor={ACCENT}>
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

      <div className="row g-3 mb-4">
        <KPI label="Patients" value={ov.n_patients} color={ACCENT} />
        <KPI label="GOF" value={`${ov.gof_pct}%`} color={ACCENT4} />
        <KPI label="LOF" value={`${ov.lof_pct}%`} color={ACCENT5} />
        <KPI label="West Syn." value={`${ov.west_syndrome_pct}%`} color={ACCENT2} />
        <KPI label="Dystonia" value={`${ov.dystonia_choreoathetosis_pct}%`} color="#f57c00" />
        <KPI label="DRE" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="On KD" value={`${ov.on_kd_pct}%`} color={ACCENT3} />
        <KPI label="Memantine" value={`${ov.memantine_rx_pct}%`} color={ACCENT3} />
        <KPI label="POLG Done" value={`${ov.polg_done_pct}%`} color={ACCENT} />
        <KPI label="Focal Sz" value={`${ov.focal_seizure_pct}%`} color="#5c6bc0" />
        <KPI label="Avg Onset" value={`${ov.avg_onset_months}m`} color={ACCENT} />
      </div>

      <SectionCard title="⚠️ Contraindications Summary" borderColor={ACCENT2}>
        {(ov.contraindications_summary || []).map((c, i) => (
          <Alert key={i} text={c} variant={c.includes('ABSOLUTE') ? 'danger' : 'warning'} />
        ))}
      </SectionCard>

      <SectionCard title="📏 Key Thresholds" borderColor={ACCENT}>
        <ul className="mb-0 ps-3">
          {(ov.thresholds || []).map((t, i) => <li key={i} className="mb-1 small">{t}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="📚 References" borderColor={ACCENT}>
        <ul className="mb-0 ps-3">
          {(ov.references || []).map((r, i) => <li key={i} className="small text-secondary">{r}</li>)}
        </ul>
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
        <div className="row g-3">
          <div className="col-md-6">
            <PctBar label="GOF variants" pct={summary.gof_pct} color={ACCENT4} />
            <PctBar label="LOF variants" pct={summary.lof_pct} color={ACCENT5} />
            <PctBar label="West syndrome" pct={summary.west_pct} color={ACCENT2} />
            <PctBar label="Dystonia/Choreoathetosis" pct={summary.dystonia_pct} color="#f57c00" />
          </div>
          <div className="col-md-6">
            <PctBar label="Drug-resistant (DRE)" pct={summary.drug_resistant_pct} color={ACCENT2} />
            <PctBar label="On ketogenic diet" pct={summary.kd_pct} color={ACCENT3} />
            <PctBar label="Memantine prescribed" pct={summary.memantine_rx_pct} color={ACCENT3} />
            {summary.polg_not_tested_count > 0 && (
              <Alert text={`⚠️ ${summary.polg_not_tested_count} patient(s) without POLG1 testing — VPA must NOT be prescribed until cleared`} variant="danger" />
            )}
          </div>
        </div>
      </SectionCard>

      <SectionCard title="🧬 Etiology Distribution" borderColor={ACCENT}>
        {(etiology_distribution || []).map((e, i) => (
          <div key={i} className="border rounded p-3 mb-3">
            <div className="d-flex justify-content-between mb-1">
              <strong className="text-dark" style={{ fontSize: 14 }}>{e.class}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT, fontSize: 12 }}>{e.pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT }} />
            </div>
            <div className="small text-secondary mb-1"><strong>Examples:</strong> {e.examples}</div>
            <div className="small text-secondary mb-1"><strong>Mechanism:</strong> {e.mechanism}</div>
            <div className="small"><strong>Precision:</strong> <span style={{ color: ACCENT3 }}>{e.precision}</span></div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="👥 Patient Cohort (40 patients)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
            <thead className="table-light">
              <tr>
                <th>PID</th><th>Age (y)</th><th>Onset (m)</th><th>Sex</th>
                <th>Variant</th><th>GOF/LOF</th><th>West</th><th>Dystonia</th>
                <th>DRE</th><th>KD</th><th>Memantine</th><th>POLG</th>
              </tr>
            </thead>
            <tbody>
              {(patients_sample || []).map(p => (
                <tr key={p.pid}>
                  <td><code>{p.pid}</code></td>
                  <td>{p.age_years}</td>
                  <td>{p.onset_months}</td>
                  <td>{p.sex}</td>
                  <td className="small">{p.variant}</td>
                  <td>
                    <span className="badge" style={{ backgroundColor: p.gof_lof === 'GOF' ? ACCENT4 : ACCENT5 }}>
                      {p.gof_lof}
                    </span>
                  </td>
                  <td>{p.west_syndrome ? '✅' : '—'}</td>
                  <td>{p.dystonia_choreoathetosis ? '🔄' : '—'}</td>
                  <td>{p.drug_resistant ? '⚠️' : '—'}</td>
                  <td>{p.on_kd ? '🥗' : '—'}</td>
                  <td>{p.memantine_rx ? '💊' : '—'}</td>
                  <td>
                    <span style={{ color: p.polg_tested === 'Y' ? ACCENT3 : ACCENT2 }}>
                      {p.polg_tested === 'Y' ? '✅' : '❌'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Seizure Types & Triggers ──────────────────────────────────────────
function SeizuresTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const { seizure_detail, trigger_detail } = bk;
  return (
    <div>
      <SectionCard title="⚡ Seizure Types (GRIN2D-specific)" borderColor={ACCENT}>
        {(seizure_detail || []).map((s, i) => (
          <div key={i} className="border rounded p-3 mb-3">
            <div className="d-flex justify-content-between mb-1">
              <strong>{s.type}</strong>
              <span className="badge" style={{ backgroundColor: s.type.includes('Non-ictal') ? '#f57c00' : ACCENT }}>
                {s.prevalence_pct}%
              </span>
            </div>
            <div className="progress mb-2" style={{ height: 7 }}>
              <div className="progress-bar" style={{ width: `${s.prevalence_pct}%`, backgroundColor: s.type.includes('Non-ictal') ? '#f57c00' : ACCENT }} />
            </div>
            <div className="small text-secondary mb-1"><strong>EEG:</strong> {s.eeg}</div>
            <div className="small text-secondary mb-1"><strong>Semiology:</strong> {s.semiology}</div>
            <div className="small text-info"><strong>Clinical tip:</strong> {s.clinical_tip}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🌡️ Seizure Triggers" borderColor={ACCENT4}>
        {(trigger_detail || []).map((t, i) => (
          <div key={i} className="border rounded p-3 mb-3">
            <div className="d-flex justify-content-between mb-1">
              <strong>{t.trigger}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.prevalence_pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 7 }}>
              <div className="progress-bar" style={{ width: `${t.prevalence_pct}%`, backgroundColor: ACCENT4 }} />
            </div>
            <div className="small text-secondary mb-1">{t.note}</div>
            <div className="small text-success"><strong>Management:</strong> {t.management}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Treatments ────────────────────────────────────────────────────────
function TreatmentsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const { treatment_detail, contraindication_detail, monitoring, lifecycle } = bk;
  return (
    <div>
      <div className="alert alert-warning fw-bold py-2 mb-3" style={{ fontSize: 13 }}>
        ⚠️ <strong>MANDATORY BEFORE ANY TREATMENT:</strong> (1) GOF/LOF functional assay — determines memantine eligibility. (2) POLG1 full gene panel — before VPA (fatal hepatotoxicity risk). (3) Video-EEG — distinguish non-ictal dystonia from focal seizures.
      </div>

      <SectionCard title="💊 Treatments" borderColor={ACCENT3}>
        {(treatment_detail || []).map((t, i) => (
          <div key={i} className="border rounded p-3 mb-3">
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ color: ACCENT }}>{t.drug}</strong>
              <span className="badge bg-secondary">{t.evidence}</span>
            </div>
            <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>
            <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
            <div className="small p-2 rounded" style={{ backgroundColor: '#e0f2f1', color: '#00695c' }}>
              <strong>GRIN2D-specific:</strong> {t.grin2d_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications" borderColor={ACCENT2}>
        {(contraindication_detail || []).map((c, i) => (
          <div key={i} className={`border rounded p-3 mb-3 border-${c.severity === 'ABSOLUTE CI' ? 'danger' : 'warning'}`}>
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ color: ACCENT2 }}>{c.drug}</strong>
              <span className={`badge bg-${c.severity === 'ABSOLUTE CI' ? 'danger' : 'warning'} text-dark`}>
                {c.severity}
              </span>
            </div>
            <div className="small mb-1"><strong>Mechanism:</strong> {c.mechanism}</div>
            <div className="small mb-1 text-danger"><strong>Consequence:</strong> {c.clinical_consequence}</div>
            <div className="small mb-1 text-success"><strong>Alternative:</strong> {c.alternative}</div>
            <div className="small p-2 rounded" style={{ backgroundColor: '#fce4ec', color: '#b71c1c' }}>
              <strong>GRIN2D note:</strong> {c.grin2d_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📋 Monitoring" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead className="table-light">
              <tr><th>Item</th><th>Timing</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td><strong>{m.item}</strong></td>
                  <td>{m.timing}</td>
                  <td className="text-secondary">{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🔄 Lifecycle Stages" borderColor={ACCENT}>
        {(lifecycle || []).map((l, i) => (
          <div key={i} className="border-start border-3 ps-3 mb-3" style={{ borderColor: ACCENT }}>
            <strong>{l.stage}</strong>
            <div className="small text-secondary mt-1">{l.features}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 5: Definitions ───────────────────────────────────────────────────────
function DefinitionsTab({ df }) {
  if (!df) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="📖 Key Concepts (GRIN2D)" borderColor={ACCENT}>
        {(df.concepts || []).map((c, i) => (
          <div key={i} className="border-bottom pb-2 mb-2">
            <strong style={{ color: ACCENT }}>{c.term}</strong>
            <div className="small text-secondary mt-1">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Full Contraindication List" borderColor={ACCENT2}>
        {(df.contraindications_full || []).map((c, i) => (
          <div key={i} className="mb-2">
            <span className={`badge me-2 bg-${c.severity === 'ABSOLUTE CI' ? 'danger' : 'warning'} text-dark`}>
              {c.severity}
            </span>
            <strong>{c.drug}</strong>
            <div className="small text-secondary">{c.mechanism}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Clinical Thresholds" borderColor={ACCENT}>
        <ul className="mb-0 ps-3">
          {(df.thresholds || []).map((t, i) => <li key={i} className="mb-1 small">{t}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="📜 Standards & Guidelines" borderColor={ACCENT}>
        <ul className="mb-0 ps-3">
          {(df.standards || []).map((s, i) => <li key={i} className="mb-1 small">{s}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="📚 References" borderColor={ACCENT}>
        <ol className="mb-0 ps-3">
          {(df.references || []).map((r, i) => <li key={i} className="mb-2 small">{r}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function GRIN2DPage() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/grin2d/overview`).then(r => r.json()).then(setOv).catch(console.error);
    fetch(`${API}/api/grin2d/breakdown`).then(r => r.json()).then(setBk).catch(console.error);
    fetch(`${API}/api/grin2d/definitions`).then(r => r.json()).then(setDf).catch(console.error);
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="mb-4">
        <h1 className="h3 fw-bold" style={{ color: ACCENT }}>
          🧠 GRIN2D Epilepsy
          <span className="fs-6 fw-normal text-muted ms-2">
            DEE / GluN2D / NMDA Subunit 2D / Extrasynaptic-Subcortical / Movement Disorder / 19q13.33
          </span>
        </h1>
        <div className="d-flex flex-wrap gap-2 mb-2">
          <span className="badge" style={{ backgroundColor: ACCENT }}>GRIN2D 19q13.33</span>
          <span className="badge" style={{ backgroundColor: ACCENT4 }}>GOF 70% — DEE + Movement Disorder</span>
          <span className="badge" style={{ backgroundColor: ACCENT5 }}>LOF 20% — Focal DEE</span>
          <span className="badge bg-danger">Memantine CI in LOF</span>
          <span className="badge bg-secondary">Ifenprodil Useless (no NTD site)</span>
          <span className="badge bg-danger">TGB Absolute CI</span>
          <span className="badge" style={{ backgroundColor: ACCENT3 }}>Baclofen for Dystonia</span>
        </div>
        <p className="text-muted small mb-0">
          GluN2D: slowest NMDA subunit (tau ~4s) · Extrasynaptic/subcortical (STN/SN/VTA/LC) ·
          GOF → constitutive tonic NMDA → basal ganglia excitotoxicity → DEE + dystonia/choreoathetosis ·
          LOF → STN disinhibition → focal DEE · Memantine (GOF ONLY) · 40-patient cohort
        </p>
      </div>

      <div className="mb-3">
        {TABS.map((t, i) => (
          <TabBtn key={t} label={t} active={tab === i} onClick={() => setTab(i)} />
        ))}
      </div>

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <PatientsTab bk={bk} />}
      {tab === 2 && <SeizuresTab bk={bk} />}
      {tab === 3 && <TreatmentsTab bk={bk} />}
      {tab === 4 && <DefinitionsTab df={df} />}
    </div>
  );
}
