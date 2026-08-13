'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#6c3483';   // deep purple — PCDH19 / cadherin / female-predominant
const ACCENT2 = '#922b21';   // deep red — contraindications / danger
const ACCENT3 = '#1e8449';   // dark green — fenfluramine / effective treatments

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#f5eef8', color: borderColor }}>
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
        <strong>🧬 {data.syndrome}</strong> — Gene: <strong>{data.gene}</strong> ({data.chromosome})<br />
        <strong>Inheritance:</strong> {data.inheritance}<br />
        <strong>EEG hallmark:</strong> {data.eeg_hallmark}<br />
        <strong>Key biomarker:</strong> {data.key_biomarker}
      </div>

      {/* Cellular Interference Paradox Banner */}
      <div className="alert alert-info mb-4" style={{ borderLeft: `5px solid #2e86c1`, fontSize: 13 }}>
        <strong>⚡ X-linked Cellular Interference Paradox:</strong> Heterozygous <strong>FEMALES are AFFECTED</strong>;
        hemizygous males are <strong>UNAFFECTED carriers</strong>.<br />
        Mosaic mixture of PCDH19-expressing and PCDH19-null cells disrupts cortical GABAergic circuit assembly.
        Uniformly null males lack the mismatch → no cellular interference → no epilepsy.<br />
        <span className="text-danger fw-bold">▶ TEST FATHER for PCDH19 carrier status (X-linked inheritance; 50% daughter risk).</span>
      </div>

      {/* Fenfluramine FDA-approved 2022 Banner */}
      <div className="alert alert-success mb-4" style={{ borderLeft: `5px solid ${ACCENT3}`, fontSize: 13 }}>
        <strong>✅ FENFLURAMINE (FINTEPLA) FDA/EMA APPROVED 2022 for PCDH19-CE</strong> —
        First disease-specific approved agent. Phase 3 (Lagae 2022): 74% of patients achieved ≥50% reduction
        in monthly cluster days. Enrol in <strong>FINTEPLA REMS</strong> (USA) before prescribing;
        echocardiogram q6M mandatory.
      </div>

      <div className="alert alert-info mb-4" style={{ fontSize: 13 }}>
        <strong>⚡ Clinical AHA — PCDH19-CE:</strong> {data.key_aha}
      </div>

      <div className="row g-2 mb-4">
        <KPI label="N Patients"     value={data.n_patients ?? '—'}           color={ACCENT} />
        <KPI label="Female %"       value={`${k.female_pct ?? '—'}%`}        color={ACCENT} />
        <KPI label="Truncating"     value={`${k.truncating_pct ?? '—'}%`}    color={ACCENT2} />
        <KPI label="Missense"       value={`${k.missense_pct ?? '—'}%`}      color='#7d6608' />
        <KPI label="DRE Rate"       value={`${k.dre_pct ?? '—'}%`}           color={ACCENT2} />
        <KPI label="Cluster-Free"   value={`${k.cluster_free_pct ?? '—'}%`}  color={ACCENT3} />
        <KPI label="FFA Active"     value={`${k.ffa_on_pct ?? '—'}%`}        color={ACCENT3} />
        <KPI label="KD Active"      value={`${k.kd_on_pct ?? '—'}%`}         color='#1a5276' />
        <KPI label="ASD Features"   value={`${k.asd_pct ?? '—'}%`}           color='#8e44ad' />
        <KPI label="Catamenial"     value={`${k.catamenial_pct ?? '—'}%`}    color='#c0392b' />
      </div>

      <SectionCard title="⚠️ Clinical Alerts" borderColor={ACCENT2}>
        {(data.clinical_alerts || []).map((a, i) => (
          <Alert key={i} text={a}
            variant={
              a.includes('🚨') || a.includes('MANDATORY') || a.includes('ABSOLUTE') ? 'danger'
              : a.includes('✅') || a.includes('APPROVED') ? 'success'
              : a.includes('⚡') || a.includes('TEST FATHER') ? 'info'
              : 'warning'
            } />
        ))}
      </SectionCard>

      <SectionCard title="🧬 Etiology Distribution (N=41)" borderColor={ACCENT}>
        {(data.etiologies || []).map((e, i) => (
          <PctBar key={i} label={`${e.etiology} (N=${e.n})`} pct={e.pct}
            color={i === 0 ? ACCENT2 : i === 1 ? '#2874a6' : i === 2 ? '#7d6608' : i === 3 ? '#117a65' : '#7f8c8d'} />
        ))}
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="⚡ Seizure Type Prevalence">
            {Object.entries(data.seizure_type_prevalence || {}).map(([k2, v], i) => (
              <PctBar key={i} label={k2} pct={v} color={ACCENT} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="🌡️ Seizure Trigger Rates" borderColor={ACCENT2}>
            {Object.entries(data.trigger_seizure_rates || {}).map(([k2, v], i) => (
              <PctBar key={i} label={k2} pct={v} color={ACCENT2} />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🕐 Lifecycle Windows" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#f5eef8' }}>
              <tr><th>Window</th><th>Age</th><th>Focus</th><th>Key Action</th></tr>
            </thead>
            <tbody>
              {(data.lifecycle_windows || []).map((w, i) => (
                <tr key={i}>
                  <td className="fw-bold">{w.window}</td>
                  <td className="text-nowrap">{w.age_range}</td>
                  <td>{w.focus}</td>
                  <td><strong>{w.key_action}</strong></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Etiology ──────────────────────────────────────────────────
function PatientsTab({ data }) {
  const [search, setSearch] = useState('');
  const [catFilter, setCatFilter] = useState('All');
  const [classFilter, setClassFilter] = useState('All');
  if (!data) return <div className="text-muted">Loading…</div>;

  const cats    = ['All', ...new Set((data.patients || []).map(p => p.category))];
  const classes = ['All', ...new Set((data.patients || []).map(p => p.functional_class))];
  const filtered = (data.patients || []).filter(p => {
    const matchCat    = catFilter === 'All'    || p.category === catFilter;
    const matchClass  = classFilter === 'All'  || p.functional_class === classFilter;
    const matchSearch = !search || JSON.stringify(p).toLowerCase().includes(search.toLowerCase());
    return matchCat && matchClass && matchSearch;
  });

  return (
    <>
      <div className="row mb-3 g-2">
        <div className="col-md-4">
          <input className="form-control form-control-sm" placeholder="Search patients…"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <div className="col-md-4">
          <select className="form-select form-select-sm" value={catFilter} onChange={e => setCatFilter(e.target.value)}>
            {cats.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="col-md-4">
          <select className="form-select form-select-sm" value={classFilter} onChange={e => setClassFilter(e.target.value)}>
            {classes.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </div>

      <div className="table-responsive mb-4">
        <table className="table table-sm table-hover table-bordered small">
          <thead className="table-dark">
            <tr>
              <th>ID</th><th>Age(M)</th><th>Sex</th><th>Onset(M)</th>
              <th>Class</th><th>Phase</th><th>Tx</th><th>Control</th>
              <th>CLB NCZ(ng/mL)</th><th>VPA(mg/L)</th><th>KBr(mmol/L)</th>
              <th>FFA</th><th>KD</th><th>ASD</th><th>Catamenial</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(p => (
              <tr key={p.id}
                style={{ backgroundColor: p.seizure_control === 'drug-resistant' ? '#fdf2f2' : undefined }}>
                <td className="fw-bold">{p.id}</td>
                <td>{p.age_months}</td>
                <td>
                  <span className={`badge ${p.sex === 'F' ? 'bg-danger' : 'bg-secondary'}`}>{p.sex}</span>
                </td>
                <td>{p.onset_age_months}</td>
                <td>
                  <span className={`badge ${
                    p.functional_class === 'truncating'      ? 'bg-danger'
                    : p.functional_class === 'missense'      ? 'bg-warning text-dark'
                    : p.functional_class === 'splice'        ? 'bg-primary'
                    : p.functional_class === 'CNV-deletion'  ? 'bg-info text-dark'
                    : 'bg-secondary'
                  }`}>{p.functional_class}</span>
                </td>
                <td className="text-nowrap">{p.disease_phase}</td>
                <td className="text-nowrap">{p.current_treatment}</td>
                <td>
                  <span className={`badge ${
                    p.seizure_control === 'cluster-free'    ? 'bg-success'
                    : p.seizure_control === 'drug-resistant' ? 'bg-danger'
                    : 'bg-warning text-dark'
                  }`}>{p.seizure_control}</span>
                </td>
                <td>{p.norclobazam_level_ngml ?? '—'}</td>
                <td className={p.vpa_level_mgL && p.vpa_level_mgL < 50 ? 'text-danger fw-bold' : ''}>{p.vpa_level_mgL ?? '—'}</td>
                <td className={p.bromide_level_mmolL && p.bromide_level_mmolL > 2.5 ? 'text-danger fw-bold' : ''}>{p.bromide_level_mmolL ?? '—'}</td>
                <td>{p.ffa_on ? <span className="badge bg-success">✓</span> : '—'}</td>
                <td>{p.kd_on ? '✓' : '—'}</td>
                <td>{p.asd_features ? <span className="badge bg-warning text-dark">ASD</span> : '—'}</td>
                <td>{p.catamenial ? <span className="badge bg-info text-dark">Cat</span> : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-muted small">Showing {filtered.length} of {(data.patients || []).length} patients</div>
      </div>

      <div className="row">
        {(data.etiology_catalog || []).map((e, i) => (
          <div className="col-md-6 mb-3" key={i}>
            <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${
              i===0 ? ACCENT2 : i===1 ? '#2874a6' : i===2 ? '#7d6608' : i===3 ? '#117a65' : '#7f8c8d'
            }` }}>
              <div className="card-header small fw-bold">{e.etiology} — N={e.n} ({e.pct}%)</div>
              <div className="card-body small">
                <p><strong>Mechanism:</strong> {e.mechanism}</p>
                <p className="mb-1"><strong>EEG:</strong> {e.eeg_signature}</p>
                <p className="mb-1"><strong>MRI:</strong> {e.mri}</p>
                <p className="mb-0"><strong>Note:</strong> {e.clinical_note}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </>
  );
}

// ── Tab: Seizure Types & Triggers ──────────────────────────────────────────────
function SeizureTriggersTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <>
      <SectionCard title="⚡ Seizure Types (N=41 cohort)" borderColor={ACCENT}>
        {(data.seizure_types || []).map((s, i) => (
          <div className="mb-4 pb-3 border-bottom" key={i}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="fw-bold">{s.type}</span>
              <span className="badge bg-primary">{s.prevalence_pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${s.prevalence_pct}%`, backgroundColor: ACCENT }} />
            </div>
            <div className="small text-muted mb-1"><strong>Onset:</strong> {s.onset_age}</div>
            <div className="small mb-1"><strong>EEG correlate:</strong> {s.eeg_correlate}</div>
            <div className="small"><strong>Clinical tip:</strong> <em>{s.clinical_tip}</em></div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🌡️ Seizure Triggers (N=41 cohort)" borderColor={ACCENT2}>
        {(data.triggers || []).map((t, i) => (
          <div className="mb-4 pb-3 border-bottom" key={i}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="fw-bold">{t.trigger}</span>
              <span className="badge bg-danger">{t.rate_pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar bg-danger" style={{ width: `${t.rate_pct}%` }} />
            </div>
            <div className="small mb-1">{t.mechanism}</div>
            <div className="small text-success"><strong>Management:</strong> {t.management}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments ──────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const levelColor = (lvl) =>
    lvl?.includes('Level A') || lvl?.includes('Level B') ? ACCENT3
    : lvl?.includes('Level C') ? '#7d6608' : ACCENT2;

  return (
    <>
      {/* Absolute Contraindications box */}
      {(data.aed_monitoring || []).length > 0 && (
        <div className="alert alert-warning mb-4" style={{ fontSize: 13 }}>
          <strong>⚠️ Monitoring Requirements:</strong> See AED Monitoring section below.
          Fenfluramine requires echocardiogram q6M (FINTEPLA REMS); VPA requires POLG exclusion;
          bromide requires TDM 1.0–2.5 mmol/L.
        </div>
      )}

      <div className="row">
        {(data.treatments || []).map((t, i) => (
          <div className="col-md-6 mb-3" key={i}>
            <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${levelColor(t.evidence_level)}` }}>
              <div className="card-header small fw-bold d-flex justify-content-between">
                <span>{t.drug}</span>
                <span className="badge" style={{ backgroundColor: levelColor(t.evidence_level), fontSize: 10 }}>
                  {t.evidence_level?.split('(')[0]?.trim()}
                </span>
              </div>
              <div className="card-body small">
                <p className="mb-1"><strong>Role:</strong> {t.role}</p>
                <p className="mb-1"><strong>Dose:</strong> {t.dose}</p>
                <p className="mb-1"><strong>MOA:</strong> {t.moa}</p>
                <p className="mb-1"><strong>Efficacy:</strong> {t.efficacy}</p>
                <p className="mb-1"><strong>Monitoring:</strong> <span className={t.monitoring?.includes('⚠️') ? 'text-danger' : ''}>{t.monitoring}</span></p>
                <p className="mb-0"><strong>Safety:</strong> <span className={t.safety?.includes('⚠️') ? 'text-danger fw-bold' : ''}>{t.safety}</span></p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <SectionCard title="🩺 AED Monitoring Checklist" borderColor="#2e86c1">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#eaf3fb' }}>
              <tr><th>Monitor Item</th><th>Scope</th><th>Action</th><th>Evidence</th></tr>
            </thead>
            <tbody>
              {(data.aed_monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold">{m.item}</td>
                  <td>{m.scope}</td>
                  <td>{m.action}</td>
                  <td className="text-muted">{m.evidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <>
      <SectionCard title="🚫 Absolute Contraindications" borderColor={ACCENT2}>
        {(data.absolute_contraindications || []).map((ci, i) => (
          <div className="mb-3 pb-3 border-bottom" key={i}>
            <div className="fw-bold text-danger mb-1">⛔ {ci.drug}</div>
            <div className="small text-muted mb-1"><strong>Scope:</strong> {ci.scope}</div>
            <div className="small mb-1">{ci.mechanism}</div>
            <div className="small text-danger"><strong>Action:</strong> {ci.action}</div>
            <div className="small text-muted"><strong>Evidence:</strong> {ci.evidence}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📊 Clinical Thresholds" borderColor='#1a5276'>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#eaf4fb' }}>
              <tr><th>Threshold</th><th>Action</th></tr>
            </thead>
            <tbody>
              {(data.thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.threshold}</td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📚 Key Concepts" borderColor={ACCENT}>
        {(data.concepts || []).map((c, i) => (
          <div className="mb-3" key={i}>
            <span className="fw-bold" style={{ color: ACCENT }}>{c.term}</span>
            <span className="text-muted small"> — </span>
            <span className="small">{c.definition}</span>
          </div>
        ))}
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="📋 Evidence Standards" borderColor={ACCENT3}>
            {(data.standards || []).map((s, i) => (
              <div className="mb-2" key={i}>
                <span className="badge me-2" style={{ backgroundColor: ACCENT3 }}>{s.standard}</span>
                <span className="small fw-bold">{s.title}</span>
                <div className="small text-muted">{s.relevance}</div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="🔬 Key References" borderColor='#7d6608'>
            {(data.references || []).map((r, i) => (
              <div className="mb-2" key={i}>
                <span className="badge me-2 bg-warning text-dark">{r.ref}</span>
                <span className="small fw-bold">{r.title}</span>
                <div className="small text-muted">{r.relevance}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>
    </>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function PCDH19Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/pcdh19/overview`).then(r => r.json()),
      fetch(`${API}/api/pcdh19/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pcdh19/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="d-flex align-items-center mb-3">
        <div>
          <h3 className="mb-0" style={{ color: ACCENT }}>
            🧬 PCDH19 Clustering Epilepsy (PCDH19-CE)
          </h3>
          <div className="text-muted small">
            PCDH19 · Xq22.1 · Protocadherin-19 · OMIM 300088 ·
            <span className="ms-1 badge" style={{ backgroundColor: ACCENT, fontSize: 11 }}>ILAE 2022</span>
            <span className="ms-1 badge bg-success" style={{ fontSize: 11 }}>Fenfluramine FDA 2022</span>
            <span className="ms-1 badge bg-danger" style={{ fontSize: 11 }}>Female-predominant</span>
            <span className="ms-1 badge bg-info text-dark" style={{ fontSize: 11 }}>41 patients</span>
          </div>
        </div>
      </div>

      {/* OMIM / gene badge strip */}
      <div className="mb-3">
        {['PCDH19', 'Xq22.1', 'Protocadherin-19', 'Cellular-Interference-Paradox',
          'Fever-Triggered-Clusters', 'EFMR', 'Fintepla-REMS', 'OMIM-300088'].map(tag => (
          <span key={tag} className="badge me-1 mb-1" style={{ backgroundColor: '#f5eef8', color: ACCENT, border: `1px solid ${ACCENT}` }}>{tag}</span>
        ))}
      </div>

      {/* Nav tabs */}
      <div className="mb-4">
        {TABS.map((t, i) => (
          <TabBtn key={t} label={t} active={tab === i} onClick={() => setTab(i)} />
        ))}
      </div>

      {loading && <div className="text-muted">Loading PCDH19-CE dashboard…</div>}
      {error   && <div className="alert alert-danger">Error: {error}</div>}

      {!loading && !error && (
        <>
          {tab === 0 && <OverviewTab        data={overview} />}
          {tab === 1 && <PatientsTab        data={breakdown} />}
          {tab === 2 && <SeizureTriggersTab data={breakdown} />}
          {tab === 3 && <TreatmentsTab      data={breakdown} />}
          {tab === 4 && <DefinitionsTab     data={definitions} />}
        </>
      )}
    </div>
  );
}
