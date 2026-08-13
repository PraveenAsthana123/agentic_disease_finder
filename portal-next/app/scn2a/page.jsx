'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT = '#1a5276';  // deep navy — Nav1.2 / sodium channel theme
const ACCENT2 = '#922b21'; // deep red — GOF danger / treatment pivot

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#eaf2ff', color: borderColor }}>
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

  return (
    <>
      <div className="alert alert-primary mb-4" style={{ borderLeft: `5px solid ${ACCENT}`, fontSize: 14 }}>
        <strong>⚡ {data.syndrome}</strong> — Gene: <strong>{data.gene}</strong> ({data.inheritance})<br />
        <strong>EEG hallmark:</strong> {data.eeg_hallmark}<br />
        <strong>Key biomarker:</strong> {data.key_biomarker}
      </div>

      {/* GOF vs LOF Treatment Pivot Banner */}
      <div className="alert alert-danger mb-4" style={{ borderLeft: `5px solid ${ACCENT2}`, fontSize: 13 }}>
        <strong>🔄 TREATMENT PIVOT — GOF vs LOF (most critical axis in genetic epilepsy):</strong><br />
        <span className="text-success fw-bold">GOF → CBZ / OXC FIRST-LINE</span> (seizure freedom ~40–50%)<br />
        <span className="text-danger fw-bold">LOF → Na-channel blockers ABSOLUTELY CONTRAINDICATED</span> (acute worsening 24–72h)
      </div>

      <div className="alert alert-warning mb-4" style={{ fontSize: 13 }}>
        <strong>⚡ Clinical AHA — SCN2A-DEE:</strong> {data.key_aha}
      </div>

      <div className="row g-2 mb-4">
        <KPI label="N Patients" value={data.n_patients ?? '—'} color={ACCENT} />
        <KPI label="GOF" value={`${k.gof_pct ?? '—'}%`} color={ACCENT} />
        <KPI label="LOF" value={`${k.lof_pct ?? '—'}%`} color={ACCENT2} />
        <KPI label="GOF Sz-Free" value={`${k.gof_seizure_free_pct ?? '—'}%`} color='#27ae60' />
        <KPI label="LOF DRE" value={`${k.lof_drug_resistant_pct ?? '—'}%`} color={ACCENT2} />
        <KPI label="ASD Dx (LOF)" value={`${k.asd_dx_pct ?? '—'}%`} color='#8e44ad' />
        <KPI label="KD On" value={`${k.kd_on_pct ?? '—'}%`} color='#2471a3' />
        <KPI label="POLG Tested" value={`${k.polg_tested_pct ?? '—'}%`} color='#117a65' />
      </div>

      <SectionCard title="⚠️ Clinical Alerts" borderColor={ACCENT2}>
        {(data.clinical_alerts || []).map((a, i) => (
          <Alert key={i} text={a}
            variant={a.includes('🚨') || a.includes('CONTRAINDICATED') || a.includes('ABSOLUTELY') ? 'danger'
              : a.includes('⚡') || a.includes('LVFA') || a.includes('EEG') ? 'primary'
              : a.includes('🧪') || a.includes('HLA') ? 'warning' : 'warning'} />
        ))}
      </SectionCard>

      <SectionCard title="🧬 Etiology Distribution (N=41)" borderColor={ACCENT2}>
        {(data.etiologies || []).map((e, i) => (
          <PctBar key={i} label={e.etiology} pct={e.pct}
            color={i === 0 ? ACCENT : i === 1 ? ACCENT2 : i === 2 ? '#2874a6' : '#7d6608'} />
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
          <SectionCard title="⚡ Seizure Trigger Rates" borderColor={ACCENT2}>
            {Object.entries(data.trigger_seizure_rates || {}).map(([k2, v], i) => (
              <PctBar key={i} label={k2} pct={v} color={ACCENT2} />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🕐 Lifecycle Windows" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#eaf2ff' }}>
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

  const cats = ['All', ...new Set((data.patients || []).map(p => p.category))];
  const filtered = (data.patients || []).filter(p => {
    const matchCat = catFilter === 'All' || p.category === catFilter;
    const matchClass = classFilter === 'All' || p.functional_class === classFilter;
    const matchSearch = !search || JSON.stringify(p).toLowerCase().includes(search.toLowerCase());
    return matchCat && matchClass && matchSearch;
  });

  return (
    <>
      <div className="row mb-3">
        <div className="col-md-4">
          <input className="form-control form-control-sm" placeholder="Search patients…"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <div className="col-md-4">
          <select className="form-select form-select-sm" value={classFilter} onChange={e => setClassFilter(e.target.value)}>
            {['All', 'GOF', 'LOF'].map(c => <option key={c}>{c}</option>)}
          </select>
        </div>
        <div className="col-md-4">
          <select className="form-select form-select-sm" value={catFilter} onChange={e => setCatFilter(e.target.value)}>
            {cats.map(c => <option key={c}>{c}</option>)}
          </select>
        </div>
      </div>

      <div className="table-responsive mb-4">
        <table className="table table-sm table-striped table-bordered small">
          <thead style={{ backgroundColor: '#eaf2ff', position: 'sticky', top: 0 }}>
            <tr>
              <th>ID</th><th>Age(M)</th><th>Sex</th><th>Onset(d)</th>
              <th>Class</th><th>Category</th><th>Phase</th><th>Treatment</th>
              <th>Control</th><th>CBZ(µg/mL)</th><th>MHD(µg/mL)</th>
              <th>Na+(mEq/L)</th><th>KD</th><th>BHB</th>
              <th>HLA-B*1502</th><th>POLG</th><th>ASD</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(p => (
              <tr key={p.id}
                style={{
                  backgroundColor: p.seizure_control === 'drug-resistant' ? '#fdf2f2'
                    : p.seizure_control === 'seizure-free' ? '#eafaf1' : 'inherit'
                }}>
                <td className="fw-bold text-nowrap">{p.id}</td>
                <td>{p.age_months}</td>
                <td>{p.sex}</td>
                <td>{p.onset_age_days}</td>
                <td>
                  <span className={`badge ${p.functional_class === 'GOF' ? 'bg-primary' : 'bg-danger'}`}>
                    {p.functional_class}
                  </span>
                </td>
                <td style={{ fontSize: 10 }}>{p.category}</td>
                <td style={{ fontSize: 10 }}>{p.disease_phase}</td>
                <td style={{ fontSize: 10 }}>{p.current_treatment}</td>
                <td>
                  <span className={`badge ${p.seizure_control === 'drug-resistant' ? 'bg-danger'
                    : p.seizure_control === 'seizure-free' ? 'bg-success' : 'bg-warning text-dark'}`}>
                    {p.seizure_control}
                  </span>
                </td>
                <td>{p.cbz_level_ugml ?? '—'}</td>
                <td>{p.mhd_level_ugml ?? '—'}</td>
                <td style={{ color: p.na_level_meql < 133 ? '#e74c3c' : 'inherit' }}>
                  {p.na_level_meql}
                </td>
                <td>{p.kd_on}</td>
                <td>{p.bhb_mmoll ?? '—'}</td>
                <td>
                  <span className={`badge ${p.hla_b1502_result === 'Positive' ? 'bg-danger'
                    : p.hla_b1502_tested === 'Y' ? 'bg-success' : 'bg-secondary'}`}>
                    {p.hla_b1502_result}
                  </span>
                </td>
                <td>
                  <span className={`badge ${p.polg_tested === 'Y' ? 'bg-success' : 'bg-warning text-dark'}`}>
                    {p.polg_tested}
                  </span>
                </td>
                <td>
                  <span className={`badge ${p.asd_diagnosis === 'Y' ? 'bg-info text-dark' : 'bg-secondary'}`}>
                    {p.asd_diagnosis}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <SectionCard title="🧬 Etiology Catalog — Detailed">
        {(data.etiology_catalog || []).map((e, i) => (
          <div key={i} className="mb-4 p-3 border rounded" style={{ borderLeft: `4px solid ${ACCENT}` }}>
            <div className="fw-bold mb-1" style={{ color: ACCENT }}>{e.etiology} ({e.pct}%, N={e.n})</div>
            <div className="small mb-2"><strong>Mechanism:</strong> {e.mechanism}</div>
            <div className="small mb-2"><strong>EEG Correlate:</strong> {e.eeg_correlate}</div>
            <div className="small mb-2"><strong>MRI Finding:</strong> {e.mri_finding}</div>
            <div className="small alert alert-light py-1 mb-0"><strong>Clinical Note:</strong> {e.clinical_note}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Seizure Types & Triggers ─────────────────────────────────────────────
function SeizureTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;

  return (
    <>
      <SectionCard title="⚡ Seizure Types — EEG Correlates & Clinical Tips">
        {(data.seizure_types || []).map((s, i) => (
          <div key={i} className="mb-4 p-3 border rounded" style={{ borderLeft: `4px solid ${ACCENT}` }}>
            <div className="d-flex align-items-center mb-2">
              <span className="fw-bold me-2" style={{ color: ACCENT }}>{s.type}</span>
              <span className="badge bg-primary ms-auto">{s.prevalence_pct}%</span>
            </div>
            <div className="small mb-2"><strong>EEG Correlate:</strong> {s.eeg_correlate}</div>
            <div className="small alert alert-info py-1 mb-0"><strong>Clinical Tip:</strong> {s.clinical_tip}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🌡️ Seizure Triggers" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: '#eaf2ff' }}>
              <tr><th>Trigger</th><th>Prevalence</th><th>Management Note</th></tr>
            </thead>
            <tbody>
              {(data.triggers || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.trigger}</td>
                  <td><div className="d-flex align-items-center gap-2">
                    <div className="progress flex-grow-1" style={{ height: 8 }}>
                      <div className="progress-bar" style={{ width: `${t.prevalence_pct}%`, backgroundColor: ACCENT2 }} />
                    </div>
                    <span className="text-nowrap">{t.prevalence_pct}%</span>
                  </div></td>
                  <td>{t.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments ───────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;

  return (
    <>
      <div className="alert alert-danger mb-4" style={{ fontSize: 13 }}>
        <strong>🔄 TREATMENT PIVOT:</strong> Determine GOF vs LOF BEFORE prescribing any AED.<br />
        <strong>GOF</strong> → CBZ / OXC (Na-channel blockers). <strong>LOF</strong> → ACTH/VGB/LEV/VPA/KD (Na-channel blockers CONTRAINDICATED).
      </div>

      {(data.treatments || []).map((t, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT}` }}>
          <div className="card-header fw-bold d-flex justify-content-between"
            style={{ backgroundColor: '#eaf2ff', color: ACCENT }}>
            <span>{t.drug}</span>
            <span className="badge" style={{
              backgroundColor: t.evidence.includes('Level B') ? '#2471a3'
                : t.evidence.includes('Level A') ? '#27ae60'
                : t.evidence.includes('Phase') ? '#8e44ad' : '#7f8c8d'
            }}>{t.evidence}</span>
          </div>
          <div className="card-body small">
            <div className="mb-2"><strong>Dose:</strong> {t.dose}</div>
            <div className="mb-2"><strong>MOA:</strong> {t.moa}</div>
            <div className="mb-2"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="mb-2"><strong>Safety:</strong> {t.safety}</div>
            <div className="alert alert-light py-1 mb-0"><strong>Monitoring:</strong> {t.monitoring}</div>
          </div>
        </div>
      ))}

      <SectionCard title="📊 AED Monitoring Requirements" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: '#eaf2ff' }}>
              <tr><th>Monitoring Item</th><th>Target</th><th>Frequency</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(data.aed_monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold">{m.item}</td>
                  <td>{m.target}</td>
                  <td>{m.frequency}</td>
                  <td>{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📋 Clinical Standards" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: '#eaf2ff' }}>
              <tr><th>Standard</th><th>Title</th><th>Note</th></tr>
            </thead>
            <tbody>
              {(data.standards || []).map((s, i) => (
                <tr key={i}>
                  <td className="fw-bold text-nowrap">{s.std}</td>
                  <td><em>{s.title}</em></td>
                  <td>{s.note}</td>
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
      <SectionCard title="📚 Key Concepts & Definitions">
        {(data.concepts || []).map((c, i) => (
          <div key={i} className="mb-3 p-3 border rounded" style={{ borderLeft: `4px solid ${ACCENT}` }}>
            <div className="fw-bold mb-1" style={{ color: ACCENT }}>{c.term}</div>
            <div className="small">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Absolute Contraindications" borderColor={ACCENT2}>
        {(data.absolute_contraindications || []).map((c, i) => (
          <div key={i} className="alert alert-danger mb-2">
            <div className="fw-bold">{c.drug}</div>
            <div className="small"><strong>Severity:</strong> {c.severity}</div>
            <div className="small">{c.reason}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Clinical Thresholds">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: '#eaf2ff' }}>
              <tr><th>Threshold</th><th>Action Required</th></tr>
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

      <SectionCard title="📖 Key References" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: '#eaf2ff' }}>
              <tr><th>Reference</th><th>Title</th><th>Clinical Note</th></tr>
            </thead>
            <tbody>
              {(data.references || []).map((r, i) => (
                <tr key={i}>
                  <td className="fw-bold text-nowrap">{r.ref}</td>
                  <td><em>{r.title}</em></td>
                  <td>{r.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function SCN2APage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/scn2a/overview`).then(r => r.json()),
      fetch(`${API}/api/scn2a/breakdown`).then(r => r.json()),
      fetch(`${API}/api/scn2a/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <div className="alert alert-danger m-3">Error: {error}</div>;

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 style={{ color: ACCENT }}>
          ⚡ SCN2A Encephalopathy (SCN2A-DEE / EIEE11)
        </h4>
        <p className="text-muted small mb-2">
          SCN2A (2q24.3) · Nav1.2 Voltage-Gated Sodium Channel · 41-patient DEE cohort ·
          GOF: Hemisynchronous BS + LVFA → CBZ/OXC first-line (seizure freedom ~40–50%) ·
          LOF: West/IS + ASD → Na-channel blockers CONTRAINDICATED · HLA-B*1502 CPIC Level A ·
          POLG exclusion mandatory · SIADH Na+ q4 weeks · ASO gene therapy trials
        </p>
        <div className="d-flex flex-wrap">
          {TABS.map((t, i) => <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />)}
        </div>
      </div>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizureTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
