'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a4a0a';   // deeper forest green — TSC2/mTOR (darker than TSC1 to distinguish)
const ACCENT2 = '#7b1c1c';   // deep crimson — contraindications / danger
const ACCENT3 = '#1a3a5c';   // deep navy — precision therapy / diagnostics
const ACCENT4 = '#5c1a3a';   // deep plum — TSC2 severe/LAM/PKD1

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
    polg_done_pct, erg_done_pct, vpa_without_polg, pkd1_deletion_n,
    tiagabine_alert, everolimus_alert, vgb_alert, polg_alert,
    sega_alert, lam_alert, pkd1_alert,
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
      {lam_alert && <Alert text={`🫁 ${lam_alert}`} variant="warning" />}
      {pkd1_alert && <Alert text={`🫘 ${pkd1_alert}`} variant="info" />}

      <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT}` }}>
        <div className="card-header fw-bold" style={{ backgroundColor: '#eef7ee', color: ACCENT }}>
          🧬 TSC2 — Clinical Summary
        </div>
        <div className="card-body small text-secondary">{summary}</div>
      </div>

      <div className="card mb-4 shadow-sm">
        <div className="card-header fw-bold" style={{ backgroundColor: '#eef2f7', color: ACCENT3 }}>
          OMIM / Locus
        </div>
        <div className="card-body small">{omim}</div>
      </div>

      <h6 className="fw-bold mb-3 mt-4" style={{ color: ACCENT }}>
        📊 Cohort KPIs — {n_patients} TSC2 Patients
      </h6>
      <div className="row g-2 mb-4">
        <KPI label="On Everolimus" value={`${on_everolimus_pct}%`} color={ACCENT3} />
        <KPI label="Drug-Resistant" value={`${drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Seizure-Free" value={`${seizure_free_pct}%`} color={ACCENT} />
        <KPI label="Infantile Spasms" value={`${infantile_spasms_pct}%`} color={ACCENT4} />
        <KPI label="SEGA" value={`${sega_pct}%`} color={ACCENT3} />
        <KPI label="Renal AML" value={`${aml_pct}%`} color={ACCENT4} />
        <KPI label="TAND/ASD" value={`${tand_asd_pct}%`} color={ACCENT3} />
        <KPI label="POLG Tested" value={`${polg_done_pct}%`} color={ACCENT} />
        <KPI label="ERG Done (VGB)" value={`${erg_done_pct}%`} color={ACCENT} />
        <KPI label="VPA w/o POLG" value={vpa_without_polg} color={ACCENT2} />
        <KPI label="TSC2+PKD1 Del." value={pkd1_deletion_n} color={ACCENT4} />
      </div>

      <SectionCard title="⛔ Contraindications Summary" borderColor={ACCENT2}>
        {contraindications_summary.map((c, i) => (
          <div key={i} className="mb-1 small">
            <Badge text={i === 0 ? 'ABSOLUTE CI' : 'HIGH RISK'} color={i === 0 ? '#6b0000' : ACCENT2} />
            {c}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📐 Key Thresholds" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0" style={{ fontSize: 12 }}>
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}><td>{t.parameter}</td><td className="fw-bold">{t.value}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📚 Key References" borderColor={ACCENT4}>
        {references.map((r, i) => (
          <div key={i} className="small mb-1">• {r}</div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Etiology ─────────────────────────────────────────────────
function EtiologyTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { summary, etiology_distribution = [], etiology_catalog = [], patients_sample = [] } = data;

  return (
    <div>
      <div className="row g-2 mb-4">
        <KPI label="Patients" value={summary?.n} color={ACCENT} />
        <KPI label="On Everolimus" value={`${summary?.on_everolimus_pct}%`} color={ACCENT3} />
        <KPI label="Drug-Resistant" value={`${summary?.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Seizure-Free" value={`${summary?.seizure_free_pct}%`} color={ACCENT} />
        <KPI label="IS History" value={`${summary?.infantile_spasms_pct}%`} color={ACCENT4} />
        <KPI label="TAND/ASD" value={`${summary?.tand_asd_pct}%`} color={ACCENT3} />
      </div>

      <SectionCard title="🧬 Etiology Distribution — 5 TSC2 Classes" borderColor={ACCENT}>
        {etiology_distribution.map((e, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between small mb-1">
              <span className="fw-bold">{e.etiology}</span>
              <span className="text-muted">{e.pct}% (n={e.n})</span>
            </div>
            <div className="progress mb-1" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT }} />
            </div>
            <div className="text-muted small">{e.mechanism_short}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📋 Etiology Catalog — Full Detail" borderColor={ACCENT3}>
        {etiology_catalog.map((ec, i) => (
          <div key={i} className="mb-4 pb-3 border-bottom">
            <div className="fw-bold mb-1" style={{ color: ACCENT }}>
              {ec.etiology} — {ec.pct}% (n={ec.n})
            </div>
            <div className="mb-1 small"><strong>Mechanism:</strong> {ec.mechanism}</div>
            <div className="mb-1 small"><strong>EEG Correlate:</strong> {ec.eeg_correlate}</div>
            <div className="mb-1 small">
              <strong>Key Treatments:</strong>{' '}
              {ec.key_treatments?.map((t, j) => <Badge key={j} text={t} color={ACCENT3} />)}
            </div>
            <div className="small text-muted">
              <strong>Clinical Tip:</strong> {ec.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="👤 Patient Sample (first 15)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0" style={{ fontSize: 11 }}>
            <thead>
              <tr>
                <th>ID</th><th>Etiology</th><th>Age (y)</th><th>Onset (m)</th>
                <th>Tubers</th><th>Drug-Res.</th><th>IS</th>
                <th>SEGA</th><th>AML</th><th>TAND</th>
                <th>Everolimus</th><th>PKD1-Del</th>
              </tr>
            </thead>
            <tbody>
              {patients_sample.map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td className="small">{p.etiology?.replace('TSC2-', '')}</td>
                  <td>{p.age_y}</td>
                  <td>{p.onset_months}</td>
                  <td>{p.tuber_count}</td>
                  <td>{p.drug_resistant ? '🔴 Yes' : '🟢 No'}</td>
                  <td>{p.has_infantile_spasms ? '⚡ Yes' : '—'}</td>
                  <td>{p.has_sega ? '🧠 Yes' : '—'}</td>
                  <td>{p.aml_present ? '🫘 Yes' : '—'}</td>
                  <td>{p.tand_asd ? '🧠 Yes' : '—'}</td>
                  <td>{p.on_everolimus ? '✅' : '—'}</td>
                  <td>{p.pkd1_deletion ? '⚠️ Yes' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Seizure Types & Triggers ────────────────────────────────────────────
function SeizureTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_detail = [], trigger_detail = [] } = data;

  return (
    <div>
      <SectionCard title="⚡ Seizure Types in TSC2 (6 types)" borderColor={ACCENT}>
        {seizure_detail.map((s, i) => (
          <div key={i} className="mb-4 pb-3 border-bottom">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold" style={{ color: ACCENT }}>{s.type}</span>
              <Badge text={`${s.prevalence_pct}%`} color={ACCENT} />
            </div>
            <PctBar label="" pct={s.prevalence_pct} color={ACCENT} />
            <div className="small mb-1"><strong>Semiology:</strong> {s.semiology}</div>
            <div className="small mb-1"><strong>EEG Correlate:</strong> {s.eeg}</div>
            <div className="small text-muted">
              <strong>Clinical Tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔥 Seizure Triggers (8 types)" borderColor={ACCENT2}>
        {trigger_detail.map((t, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold" style={{ color: ACCENT2 }}>{t.trigger}</span>
              <Badge text={`${t.prevalence_pct}%`} color={ACCENT2} />
            </div>
            <PctBar label="" pct={t.prevalence_pct} color={ACCENT2} />
            <div className="small mb-1">{t.mechanism}</div>
            <div className="small text-muted">
              <strong>Management:</strong> {t.management}
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Treatments ─────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatment_detail = [], contraindication_detail = [], monitoring = [] } = data;

  return (
    <div>
      <SectionCard title="💊 Treatments (7 evidence levels)" borderColor={ACCENT}>
        {treatment_detail.map((t, i) => (
          <div key={i} className="mb-4 pb-3 border-bottom">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold" style={{ color: ACCENT }}>{t.drug}</span>
              <Badge text={t.level} color={t.level === 'Level A' ? ACCENT : ACCENT3} />
            </div>
            <div className="small mb-1"><strong>Indication:</strong> {t.indication}</div>
            <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>Mechanism:</strong> {t.moa}</div>
            <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small mb-1"><strong>Safety:</strong> {t.safety}</div>
            <div className="small mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
            {t.tsc2_note && (
              <div className="small p-2 rounded mt-1" style={{ backgroundColor: '#eef7ee', borderLeft: `3px solid ${ACCENT}` }}>
                <strong>TSC2 Note:</strong> {t.tsc2_note}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="⛔ Contraindications in TSC2" borderColor={ACCENT2}>
        {contraindication_detail.map((c, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold" style={{ color: ACCENT2 }}>{c.drug}</span>
              <Badge text={c.level} color={c.level === 'ABSOLUTE CI' ? '#6b0000' : ACCENT2} />
            </div>
            <div className="small mb-1">{c.reason}</div>
            <div className="small text-muted">
              <strong>Alternative:</strong> {c.alternative}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🩺 Monitoring Panel (14 items)" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0" style={{ fontSize: 12 }}>
            <thead><tr><th>Monitoring Item</th><th>Frequency</th></tr></thead>
            <tbody>
              {monitoring.map((m, i) => (
                <tr key={i}><td>{m.item}</td><td className="text-muted">{m.freq}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 5: Definitions ─────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { concepts = [], thresholds = [], standards = [], references = [] } = data;

  return (
    <div>
      <SectionCard title="📖 Key Concepts (15)" borderColor={ACCENT}>
        {concepts.map((c, i) => (
          <div key={i} className="mb-2 pb-2 border-bottom">
            <div className="fw-bold small" style={{ color: ACCENT }}>{c.term}</div>
            <div className="small text-secondary">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📐 Thresholds" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0" style={{ fontSize: 12 }}>
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}><td>{t.parameter}</td><td className="fw-bold">{t.value}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📋 Standards & Guidelines" borderColor={ACCENT4}>
        {standards.map((s, i) => (
          <div key={i} className="small mb-1">
            <Badge text={s.id} color={ACCENT4} /> {s.ref}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📚 References" borderColor={ACCENT3}>
        {references.map((r, i) => (
          <div key={i} className="small mb-2">• {r}</div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function TSC2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/tsc2/overview`).then(r => r.json()),
      fetch(`${API}/api/tsc2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/tsc2/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov);
        setBreakdown(bd);
        setDefinitions(df);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
            🌿 TSC2 Epilepsy Dashboard
          </h4>
          <div className="text-muted small">
            Tuberous Sclerosis Complex (TSC2) — Tuberin / mTOR / 16p13.3 ·{' '}
            <span className="fw-bold" style={{ color: ACCENT4 }}>Most Common &amp; Severe TSC (70%)</span>
          </div>
        </div>
        <div className="ms-auto">
          <span className="badge fs-6" style={{ backgroundColor: ACCENT }}>TSC2 · 16p13.3</span>
          <span className="badge fs-6 ms-2" style={{ backgroundColor: ACCENT4 }}>70% TSC · SEVERE</span>
        </div>
      </div>

      {error && (
        <div className="alert alert-danger">Error loading data: {error}</div>
      )}
      {loading && (
        <div className="text-center py-4">
          <div className="spinner-border text-success" role="status" />
          <div className="mt-2 text-muted small">Loading TSC2 data…</div>
        </div>
      )}

      <div className="mb-3">
        {TABS.map((t, i) => (
          <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />
        ))}
      </div>

      {!loading && !error && (
        <>
          {tab === 0 && <OverviewTab data={overview} />}
          {tab === 1 && <EtiologyTab data={breakdown} />}
          {tab === 2 && <SeizureTab data={breakdown} />}
          {tab === 3 && <TreatmentsTab data={breakdown} />}
          {tab === 4 && <DefinitionsTab data={definitions} />}
        </>
      )}
    </div>
  );
}
