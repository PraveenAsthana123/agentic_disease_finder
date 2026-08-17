'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#2d5016';   // deep forest green — mTOR / TSC / everolimus
const ACCENT2 = '#7b1c1c';   // deep crimson — contraindications / danger
const ACCENT3 = '#1a3a5c';   // deep navy — precision therapy / diagnostics
const ACCENT4 = '#5c3a1a';   // warm brown — cortical tubers / neuropathology

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#eef2f7', color: borderColor }}>
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
      onClick={onClick}
    >
      {label}
    </button>
  );
}

function Badge({ text, color = ACCENT }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: 11 }}>
      {text}
    </span>
  );
}

// ── Tab 1: Overview ───────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const {
    gene, locus, protein, syndrome, incidence, inheritance, omim, summary,
    n_patients, on_everolimus_pct, drug_resistant_pct, seizure_free_pct,
    infantile_spasms_pct, sega_pct, aml_pct, tand_asd_pct,
    polg_done_pct, erg_done_pct, vpa_without_polg,
    tiagabine_alert, everolimus_alert, vgb_alert, polg_alert, sega_alert,
    contraindications_summary = [], thresholds = [], references = [],
  } = data;

  return (
    <div>
      <div className="alert alert-success fw-bold mb-3" style={{ borderLeft: `5px solid ${ACCENT}`, fontSize: 14 }}>
        🧬 {gene} ({locus}) — {syndrome} · {incidence} · {inheritance}
        <div className="mt-1 small text-muted">{protein}</div>
      </div>

      <Alert text={`⛔ ${tiagabine_alert}`} variant="danger" />
      <Alert text={`🔬 ${everolimus_alert}`} variant="info" />
      <Alert text={`💊 ${vgb_alert}`} variant="warning" />
      <Alert text={`🧬 ${polg_alert}`} variant="warning" />
      {sega_alert && <Alert text={`🧠 ${sega_alert}`} variant="info" />}
      {vpa_without_polg > 0 && (
        <Alert text={`⚠ ${vpa_without_polg} patient(s) on VPA without POLG screening — immediate action required`} variant="danger" />
      )}

      <SectionCard title={`Cohort KPIs — ${n_patients} Patients`} borderColor={ACCENT}>
        <div className="row g-2">
          <KPI label="On Everolimus" value={`${on_everolimus_pct}%`} color={ACCENT} />
          <KPI label="Drug-Resistant" value={`${drug_resistant_pct}%`} color={ACCENT2} />
          <KPI label="Seizure-Free" value={`${seizure_free_pct}%`} color="#1a6b2f" />
          <KPI label="Infantile Spasms" value={`${infantile_spasms_pct}%`} color={ACCENT4} />
          <KPI label="SEGA Present" value={`${sega_pct}%`} color={ACCENT3} />
          <KPI label="Renal AML" value={`${aml_pct}%`} color="#6b5a1a" />
          <KPI label="TAND (ASD)" value={`${tand_asd_pct}%`} color={ACCENT4} />
          <KPI label="POLG Tested" value={`${polg_done_pct}%`} color={ACCENT} />
          <KPI label="ERG Done (VGB)" value={`${erg_done_pct}%`} color={ACCENT3} />
        </div>
      </SectionCard>

      <SectionCard title="Gene & Clinical Summary" borderColor={ACCENT3}>
        <p style={{ fontSize: 13 }}>{summary}</p>
        <div className="mt-2">
          <Badge text={`OMIM: ${omim}`} color={ACCENT3} />
          <Badge text={incidence} color={ACCENT} />
          <Badge text={inheritance} color={ACCENT4} />
        </div>
      </SectionCard>

      <SectionCard title="Contraindications Summary" borderColor={ACCENT2}>
        {contraindications_summary.map((ci, i) => (
          <div key={i} className="mb-1 small" style={{ color: ACCENT2 }}>
            ⛔ {ci}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
            <thead className="table-dark">
              <tr><th>Parameter</th><th>Target</th><th>Action</th></tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td>{t.param}</td>
                  <td><span className="badge bg-success">{t.target}</span></td>
                  <td className="text-danger small">{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {references.length > 0 && (
        <SectionCard title="Key References" borderColor={ACCENT3}>
          {references.map((r, i) => <div key={i} className="small text-muted">📄 {r}</div>)}
        </SectionCard>
      )}
    </div>
  );
}

// ── Tab 2: Patients & Etiology ────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { summary, etiology_distribution = [], patients_sample = [] } = data;

  return (
    <div>
      <SectionCard title="Cohort Summary" borderColor={ACCENT}>
        <div className="row g-2">
          <KPI label="Everolimus" value={`${summary.on_everolimus_pct}%`} color={ACCENT} />
          <KPI label="Drug-Resistant" value={`${summary.drug_resistant_pct}%`} color={ACCENT2} />
          <KPI label="Seizure-Free" value={`${summary.seizure_free_pct}%`} color="#1a6b2f" />
          <KPI label="Has IS" value={`${summary.infantile_spasms_pct}%`} color={ACCENT4} />
          <KPI label="SEGA" value={`${summary.sega_pct}%`} color={ACCENT3} />
          <KPI label="AML" value={`${summary.aml_pct}%`} color="#6b5a1a" />
        </div>
      </SectionCard>

      <SectionCard title="Etiology Distribution — 5 Classes" borderColor={ACCENT4}>
        {etiology_distribution.map((e, i) => (
          <div key={i} className="mb-3 p-2 border rounded" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
            <div className="fw-bold small">{e.etiology}</div>
            <div className="d-flex align-items-center gap-2 mt-1">
              <div className="progress flex-grow-1" style={{ height: 12 }}>
                <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT4 }} />
              </div>
              <span className="small text-muted">{e.pct}% (n={e.n})</span>
            </div>
            <div className="text-muted small mt-1">{e.mechanism_short}…</div>
            <div className="text-muted small fst-italic">{e.eeg_signature_short}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Sample (15 of 40)" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-bordered mb-0" style={{ fontSize: 11 }}>
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Age</th><th>Etiology</th>
                <th>IS</th><th>SEGA</th><th>Tubers</th>
                <th>Everolimus</th><th>Trough</th>
                <th>DR</th><th>Sz-Free</th>
                <th>AML(cm)</th><th>ASD</th><th>POLG</th>
              </tr>
            </thead>
            <tbody>
              {patients_sample.map((p) => (
                <tr key={p.patient_id}>
                  <td>{p.patient_id}</td>
                  <td>{p.age}y</td>
                  <td style={{ maxWidth: 160, fontSize: 10 }}>{p.etiology}</td>
                  <td>{p.has_infantile_spasms ? '✓' : ''}</td>
                  <td>{p.has_sega ? '⚠' : ''}</td>
                  <td>{p.tuber_count}</td>
                  <td>{p.on_everolimus ? '✓' : ''}</td>
                  <td>{p.everolimus_trough != null ? `${p.everolimus_trough}` : '—'}</td>
                  <td>{p.drug_resistant ? <span className="text-danger">DR</span> : ''}</td>
                  <td>{p.seizure_free ? <span className="text-success">✓</span> : ''}</td>
                  <td style={{ color: p.aml_size_cm >= 4 ? '#c00' : 'inherit' }}>
                    {p.aml_present ? p.aml_size_cm : '—'}
                  </td>
                  <td>{p.tand_asd ? 'ASD' : ''}</td>
                  <td style={{ color: p.polg_tested === 'N' && p.on_vpa ? '#c00' : 'inherit' }}>
                    {p.polg_tested}
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

// ── Tab 3: Seizures & Triggers ─────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_detail = [], trigger_detail = [] } = data;

  return (
    <div>
      <SectionCard title="Seizure Types (6) — TSC1 Spectrum" borderColor={ACCENT4}>
        {seizure_detail.map((s, i) => (
          <div key={i} className="mb-3 p-2 border rounded" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
            <div className="d-flex justify-content-between">
              <span className="fw-bold small">{s.type}</span>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{s.prevalence_pct}%</span>
            </div>
            <div className="progress my-1" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${s.prevalence_pct}%`, backgroundColor: ACCENT4 }} />
            </div>
            <div className="small text-muted"><strong>Onset:</strong> {s.onset_age}</div>
            <div className="small text-muted"><strong>EEG:</strong> {s.eeg}</div>
            <div className="small text-muted"><strong>Semiology:</strong> {s.semiology}</div>
            <div className="small mt-1" style={{ color: ACCENT3 }}><strong>Tip:</strong> {s.clinical_tip}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers (8)" borderColor={ACCENT2}>
        {trigger_detail.map((t, i) => (
          <div key={i} className="mb-2">
            <PctBar label={t.trigger} pct={t.prevalence_pct} color={ACCENT2} />
            <div className="small text-muted ms-1">{t.detail}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Treatments ────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatment_detail = [], contraindication_detail = [], monitoring = [] } = data;

  return (
    <div>
      <SectionCard title="Treatments — 7 Options" borderColor={ACCENT}>
        {treatment_detail.map((t, i) => (
          <div key={i} className="mb-4 p-3 border rounded" style={{ borderLeft: `4px solid ${ACCENT}` }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold">{t.drug}</span>
              <Badge text={t.level.split(' — ')[0]} color={t.level.includes('Level A') ? ACCENT : ACCENT3} />
            </div>
            <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>
            <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small mb-1 text-danger"><strong>Safety:</strong> {t.safety}</div>
            <div className="small mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
            <div className="small p-2 rounded mt-1" style={{ backgroundColor: '#f0f8ff', color: ACCENT3 }}>
              <strong>TSC1 note:</strong> {t.tsc_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications" borderColor={ACCENT2}>
        {contraindication_detail.map((c, i) => (
          <div key={i} className="mb-3 p-2 rounded border border-danger" style={{ backgroundColor: '#fff5f5' }}>
            <div className="d-flex justify-content-between">
              <span className="fw-bold text-danger">{c.drug}</span>
              <span className="badge bg-danger">{c.severity}</span>
            </div>
            <div className="small mt-1">{c.reason}</div>
            <div className="small text-success mt-1"><strong>Alternative:</strong> {c.alternative}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Panel (12 Items)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
            <thead className="table-dark">
              <tr><th>Monitoring Item</th><th>Frequency</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {monitoring.map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold">{m.item}</td>
                  <td>{m.frequency}</td>
                  <td className="text-muted">{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 5: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { concepts = [], thresholds = [], standards = [], references = [] } = data;

  return (
    <div>
      <SectionCard title="Key Concepts (15)" borderColor={ACCENT3}>
        {concepts.map((c, i) => (
          <div key={i} className="mb-2 p-2 border rounded" style={{ borderLeft: `3px solid ${ACCENT3}` }}>
            <div className="fw-bold small" style={{ color: ACCENT3 }}>{c.term}</div>
            <div className="small text-muted">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
            <thead className="table-dark">
              <tr><th>Parameter</th><th>Target</th><th>Action Threshold</th></tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td>{t.param}</td>
                  <td><span className="badge bg-success">{t.target}</span></td>
                  <td className="text-danger small">{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Standards & Guidelines" borderColor={ACCENT}>
        {standards.map((s, i) => (
          <div key={i} className="small text-muted mb-1">📋 {s}</div>
        ))}
      </SectionCard>

      <SectionCard title="References" borderColor={ACCENT3}>
        {references.map((r, i) => (
          <div key={i} className="small text-muted mb-1">📄 {r}</div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Lifecycle strip ──────────────────────────────────────────────────────────
function LifecycleStrip({ lifecycle = [] }) {
  if (!lifecycle.length) return null;
  const colors = [ACCENT3, ACCENT4, ACCENT2, ACCENT, ACCENT3, ACCENT4];
  return (
    <div className="card mb-4 shadow-sm">
      <div className="card-header fw-bold" style={{ backgroundColor: '#eef2f7', color: ACCENT }}>
        Disease Lifecycle — 6 Windows
      </div>
      <div className="card-body p-2">
        <div className="d-flex flex-wrap gap-2">
          {lifecycle.map((lc, i) => (
            <div key={i} className="p-2 rounded text-white small" style={{ backgroundColor: colors[i % colors.length], minWidth: 150, flex: '1 1 160px' }}>
              <div className="fw-bold mb-1">{lc.window}</div>
              <div style={{ fontSize: 11, opacity: 0.9 }}>{lc.key_events?.substring(0, 120)}…</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function TSC1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/tsc1/overview`).then(r => r.json()),
      fetch(`${API}/api/tsc1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/tsc1/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  return (
    <div className="container-fluid py-3 px-4">
      <div className="mb-3" style={{ borderBottom: `3px solid ${ACCENT}`, paddingBottom: 8 }}>
        <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
          🌿 TSC1 Epilepsy — Tuberous Sclerosis Complex (mTOR / Hamartin / 9q34.13)
        </h4>
        <div className="text-muted small mt-1">
          Cortical tubers · mTORC1 hyperactivation · Everolimus precision therapy (EXIST-3) ·
          Vigabatrin (IS, Level A) · SEGA · TAND · 40-patient cohort
        </div>
      </div>

      <div className="mb-3">
        {TABS.map((t, i) => <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />)}
      </div>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && (
        <>
          <PatientsTab data={breakdown} />
          {breakdown && <LifecycleStrip lifecycle={breakdown.lifecycle || []} />}
        </>
      )}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
