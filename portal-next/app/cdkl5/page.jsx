'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT = '#1a3a6b'; // deep blue — X-linked/kinase theme

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#f0f4ff', color: borderColor }}>
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

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const k = data.kpis || {};
  const colors = { ok: '#1a3a6b', warn: '#e67e22', danger: '#e74c3c', info: '#27ae60' };

  return (
    <>
      <div className="alert alert-primary mb-4" style={{ borderLeft: `5px solid ${ACCENT}`, fontSize: 14 }}>
        <strong>🧬 {data.syndrome}</strong> — Gene: <strong>{data.gene}</strong> ({data.inheritance})<br />
        <strong>EEG hallmark:</strong> {data.eeg_hallmark}<br />
        <strong>Key biomarker:</strong> {data.key_biomarker}
      </div>

      <div className="alert alert-danger mb-4" style={{ fontSize: 13 }}>
        <strong>⚠️ KEY ALERTS:</strong> {data.key_aha}
      </div>

      <div className="row g-2 mb-4">
        <KPI label="N Patients" value={data.n_patients} color={colors.ok} />
        <KPI label="Drug-Resistant" value={`${k.drug_resistant_pct}%`} color={colors.danger} />
        <KPI label="Seizure-Free" value={`${k.seizure_free_pct}%`} color={colors.info} />
        <KPI label="On KD" value={`${k.kd_responder_pct}%`} color={colors.warn} />
        <KPI label="VGB Ever" value={`${k.vgb_ever_pct}%`} color={colors.ok} />
        <KPI label="ACTH Hx" value={`${k.acth_history_pct}%`} color={colors.warn} />
        <KPI label="VF Loss" value={`${k.vf_loss_pct}%`} color={colors.danger} />
        <KPI label="Gastrostomy" value={`${k.gastrostomy_pct}%`} color={colors.warn} />
        <KPI label="Avg Onset" value={`${k.avg_onset_age_months}mo`} color={colors.ok} />
      </div>

      <SectionCard title="Etiology Distribution (N=41)">
        {(data.etiologies || []).map(e => (
          <PctBar key={e.category} label={e.etiology} pct={e.pct} color={ACCENT} />
        ))}
      </SectionCard>

      <SectionCard title="Seizure Type Prevalence">
        {Object.entries(data.seizure_type_prevalence || {}).map(([k, v]) => (
          <PctBar key={k} label={k} pct={v} color={v >= 80 ? colors.danger : v >= 60 ? colors.warn : ACCENT} />
        ))}
      </SectionCard>

      <SectionCard title="Trigger → Seizure Rate">
        {Object.entries(data.trigger_seizure_rates || {}).map(([k, v]) => (
          <PctBar key={k} label={k} pct={v} color={v >= 80 ? colors.danger : v >= 60 ? colors.warn : ACCENT} />
        ))}
      </SectionCard>

      <SectionCard title="Clinical Alerts" borderColor="#c0392b">
        {(data.clinical_alerts || []).map((a, i) => (
          <Alert key={i} text={a} variant={
            a.includes('MANDATORY') || a.includes('ABSOLUTE') ? 'danger' :
            a.includes('RELATIVE') || a.includes('relative') ? 'warning' : 'info'
          } />
        ))}
      </SectionCard>

      <SectionCard title="Lifecycle Windows">
        {(data.lifecycle_windows || []).map((w, i) => (
          <div key={i} className="mb-3">
            <div className="fw-semibold" style={{ color: ACCENT }}>
              {i + 1}. {w.window}
            </div>
            <ul className="mb-0 small">
              {(w.key_features || []).map((f, j) => <li key={j}>{f}</li>)}
            </ul>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Etiology ───────────────────────────────────────────────
function PatientsTab({ data }) {
  const [sort, setSort] = useState('id');
  const [search, setSearch] = useState('');
  if (!data) return <div className="text-muted">Loading…</div>;

  const pts = (data.patients || [])
    .filter(p => !search || JSON.stringify(p).toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      if (sort === 'id') return a.id.localeCompare(b.id);
      if (sort === 'age') return a.age_years - b.age_years;
      if (sort === 'onset') return a.onset_age_months - b.onset_age_months;
      if (sort === 'ctrl') return a.seizure_control.localeCompare(b.seizure_control);
      return 0;
    });

  const eth = data.etiology_catalog || [];

  return (
    <>
      <SectionCard title="Etiology Catalog (5 classes)">
        {eth.map((e, i) => (
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: i < eth.length - 1 ? '1px solid #eee' : 'none' }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold" style={{ color: ACCENT }}>{e.etiology}</span>
              <span className="badge rounded-pill ms-2" style={{ backgroundColor: ACCENT }}>{e.pct}%</span>
            </div>
            <div className="small text-muted mb-1"><strong>Mechanism:</strong> {(e.mechanism || '').substring(0, 400)}…</div>
            <div className="small text-muted mb-1"><strong>EEG:</strong> {(e.eeg_correlate || '').substring(0, 300)}…</div>
            <div className="small text-muted"><strong>MRI:</strong> {(e.mri_finding || '').substring(0, 200)}…</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title={`Patient List (N=${pts.length})`}>
        <div className="d-flex gap-2 mb-3 flex-wrap">
          <input className="form-control form-control-sm w-auto" placeholder="Search…"
            value={search} onChange={e => setSearch(e.target.value)} style={{ minWidth: 160 }} />
          {['id', 'age', 'onset', 'ctrl'].map(s => (
            <button key={s} className={`btn btn-sm ${sort === s ? 'btn-primary' : 'btn-outline-secondary'}`}
              style={sort === s ? { backgroundColor: ACCENT, borderColor: ACCENT } : {}}
              onClick={() => setSort(s)}>
              Sort {s === 'id' ? 'ID' : s === 'age' ? 'Age' : s === 'onset' ? 'Onset' : 'Control'}
            </button>
          ))}
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Age</th><th>Sex</th><th>Etiology</th><th>Onset(mo)</th>
                <th>Control</th><th>AEDs</th><th>KD</th><th>VGB</th><th>VF Loss</th><th>ACTH</th><th>Gastrostomy</th>
              </tr>
            </thead>
            <tbody>
              {pts.map(p => (
                <tr key={p.id}>
                  <td className="fw-bold">{p.id}</td>
                  <td>{p.age_years}y</td>
                  <td>{p.sex}</td>
                  <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                    title={p.etiology}>{p.etiology}</td>
                  <td>{p.onset_age_months}</td>
                  <td>
                    <span className={`badge ${p.seizure_control === 'seizure-free' ? 'bg-success' :
                      p.seizure_control === 'drug-resistant' ? 'bg-danger' : 'bg-warning text-dark'}`}>
                      {p.seizure_control}
                    </span>
                  </td>
                  <td style={{ maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                    title={p.current_aeds}>{p.current_aeds}</td>
                  <td><span className={`badge ${p.kd_on === 'Y' ? 'bg-success' : 'bg-secondary'}`}>{p.kd_on}</span></td>
                  <td><span className={`badge ${p.vgb_ever === 'Y' ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.vgb_ever}</span></td>
                  <td><span className={`badge ${p.vf_loss === 'Y' ? 'bg-danger' : 'bg-secondary'}`}>{p.vf_loss}</span></td>
                  <td><span className={`badge ${p.acth_history === 'Y' ? 'bg-info text-dark' : 'bg-secondary'}`}>{p.acth_history}</span></td>
                  <td><span className={`badge ${p.gastrostomy === 'Y' ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.gastrostomy}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Seizure Types & Triggers ───��──────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const seizureTypes = data.seizure_types || [];
  const triggers = data.triggers || [];

  return (
    <>
      <SectionCard title="Seizure Types (4) — EEG Correlates & Clinical Tips">
        {seizureTypes.map((s, i) => (
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: i < seizureTypes.length - 1 ? '1px solid #eee' : 'none' }}>
            <div className="d-flex justify-content-between align-items-center mb-2">
              <span className="fw-bold fs-6" style={{ color: ACCENT }}>{s.type}</span>
              <span className="badge rounded-pill" style={{
                backgroundColor: s.prevalence_pct >= 80 ? '#e74c3c' : s.prevalence_pct >= 60 ? '#e67e22' : ACCENT
              }}>{s.prevalence_pct}%</span>
            </div>
            <div className="mb-2">
              <span className="badge bg-secondary me-1">EEG</span>
              <span className="small text-muted">{s.eeg_correlate}</span>
            </div>
            <div>
              <span className="badge me-1" style={{ backgroundColor: ACCENT }}>Clinical</span>
              <span className="small">{s.clinical_tip}</span>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers (8) — Rates & Management">
        {triggers.map((t, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="fw-semibold small">{t.trigger}</span>
              <span className="badge rounded-pill" style={{
                backgroundColor: t.rate_pct >= 80 ? '#e74c3c' : t.rate_pct >= 60 ? '#e67e22' : ACCENT
              }}>{t.rate_pct}%</span>
            </div>
            <div className="progress mb-1" style={{ height: 8 }}>
              <div className="progress-bar" style={{
                width: `${t.rate_pct}%`,
                backgroundColor: t.rate_pct >= 80 ? '#e74c3c' : t.rate_pct >= 60 ? '#e67e22' : ACCENT
              }} />
            </div>
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments ─────��──────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  const [expanded, setExpanded] = useState(null);
  if (!data) return <div className="text-muted">Loading…</div>;
  const treatments = data.treatments || [];
  const contraindications = data.absolute_contraindications || [];
  const monitoring = data.aed_monitoring || [];

  return (
    <>
      <SectionCard title="Treatments (8) — Dose / MOA / Efficacy / Safety">
        {treatments.map((t, i) => (
          <div key={i} className="mb-3">
            <div
              className="d-flex justify-content-between align-items-center p-2 rounded"
              style={{ backgroundColor: '#f0f4ff', cursor: 'pointer' }}
              onClick={() => setExpanded(expanded === i ? null : i)}
            >
              <div>
                <span className="fw-bold me-2" style={{ color: ACCENT }}>{t.drug}</span>
                <span className="badge bg-secondary">{t.evidence}</span>
              </div>
              <span>{expanded === i ? '▲' : '▼'}</span>
            </div>
            {expanded === i && (
              <div className="p-3 border rounded-bottom border-top-0" style={{ borderColor: ACCENT }}>
                <div className="row">
                  <div className="col-md-6 mb-2">
                    <strong style={{ color: ACCENT }}>Dose:</strong>
                    <div className="small">{t.dose}</div>
                  </div>
                  <div className="col-md-6 mb-2">
                    <strong style={{ color: ACCENT }}>Efficacy:</strong>
                    <div className="small">{t.efficacy}</div>
                  </div>
                  <div className="col-12 mb-2">
                    <strong style={{ color: ACCENT }}>MOA:</strong>
                    <div className="small">{t.moa}</div>
                  </div>
                  <div className="col-md-6 mb-2">
                    <strong style={{ color: ACCENT }}>Safety:</strong>
                    <div className="small">{t.safety}</div>
                  </div>
                  <div className="col-md-6 mb-2">
                    <strong style={{ color: ACCENT }}>Monitoring:</strong>
                    <div className="small">{t.monitoring}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications (4)" borderColor="#c0392b">
        {contraindications.map((c, i) => (
          <div key={i} className="mb-3 p-2 rounded"
            style={{ backgroundColor: c.severity.includes('ABSOLUTE') ? '#fff0f0' : '#fffbf0' }}>
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-bold" style={{ color: '#c0392b' }}>{c.drug}</span>
              <span className={`badge ${c.severity.includes('ABSOLUTE') ? 'bg-danger' : 'bg-warning text-dark'}`}>
                {c.severity.includes('ABSOLUTE') ? 'ABSOLUTE CI' : 'Relative CI / REMS'}
              </span>
            </div>
            <div className="small">{c.reason}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="AED Monitoring Protocol (5 items)" borderColor="#27ae60">
        {monitoring.map((m, i) => (
          <div key={i} className="mb-3">
            <div className="fw-semibold" style={{ color: '#27ae60' }}>{m.item}</div>
            <div className="small"><strong>Schedule:</strong> {m.schedule}</div>
            <div className="small"><strong>Target / Action:</strong> {m.target}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ───��───────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const concepts = data.concepts || [];
  const thresholds = data.thresholds || [];
  const references = data.references || [];
  const standards = data.standards || [];

  return (
    <>
      <SectionCard title="Key Concepts (14)">
        {concepts.map((c, i) => (
          <div key={i} className="mb-3">
            <div className="fw-bold" style={{ color: ACCENT }}>{c.term}</div>
            <div className="small text-muted">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds (8)">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead className="table-light">
              <tr><th>Threshold</th><th>Value</th><th>Unit</th></tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td>{t.threshold}</td>
                  <td className="fw-bold" style={{ color: ACCENT }}>{t.value}</td>
                  <td className="text-muted">{t.unit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Standards (6)">
        {standards.map((s, i) => (
          <div key={i} className="mb-2">
            <span className="badge me-2" style={{ backgroundColor: ACCENT }}>{s.code}</span>
            <span className="small fw-semibold">{s.title}</span>
            <div className="small text-muted ms-4">{s.relevance}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="References (6)">
        {references.map((r, i) => (
          <div key={i} className="mb-3">
            <div className="small fw-semibold">{i + 1}. {r.citation}</div>
            <div className="small text-muted">→ {r.key_finding}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function CDKL5Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAll = async () => {
      setLoading(true);
      setError(null);
      try {
        const [ov, bk, df] = await Promise.all([
          fetch(`${API}/api/cdkl5/overview`).then(r => r.json()),
          fetch(`${API}/api/cdkl5/breakdown`).then(r => r.json()),
          fetch(`${API}/api/cdkl5/definitions`).then(r => r.json()),
        ]);
        setOverview(ov);
        setBreakdown(bk);
        setDefinitions(df);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    fetchAll();
  }, []);

  const ovDefns = definitions ? { ...definitions, lifecycle_windows: overview?.lifecycle_windows } : null;
  const treatData = breakdown ? {
    treatments: breakdown.treatments,
    absolute_contraindications: definitions?.absolute_contraindications,
    aed_monitoring: breakdown.aed_monitoring,
  } : null;

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4">
        <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🧬 CDKL5 Deficiency Disorder (CDD)
        </h2>
        <p className="text-muted mb-0" style={{ fontSize: 14 }}>
          <strong>CDKL5 (Xp22.13)</strong> · X-linked serine/threonine kinase ·
          Early infantile epileptic encephalopathy · 41-patient cohort ·
          IS onset &lt;5 months · ~75% drug-resistant · KD Level B · VGB SHARE REMS mandatory
        </p>
      </div>

      <div className="mb-3 d-flex flex-wrap gap-1">
        {TABS.map((t, i) => (
          <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />
        ))}
      </div>

      {loading && <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>}
      {error && <div className="alert alert-danger">Error: {error}</div>}

      {!loading && !error && (
        <>
          {tab === 0 && <OverviewTab data={overview} />}
          {tab === 1 && <PatientsTab data={breakdown} />}
          {tab === 2 && <SeizuresTab data={breakdown} />}
          {tab === 3 && <TreatmentsTab data={treatData} />}
          {tab === 4 && <DefinitionsTab data={ovDefns} />}
        </>
      )}
    </div>
  );
}
