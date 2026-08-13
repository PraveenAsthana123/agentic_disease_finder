'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1b4f72';   // deep blue — KNa1.1 / KCNT1 potassium channel
const ACCENT2 = '#6e2c00';   // burnt orange — contraindications / danger
const ACCENT3 = '#1e6823';   // dark green — KD / effective treatments

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#eaf3fb', color: borderColor }}>
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
        <strong>⚡ {data.syndrome}</strong> — Gene: <strong>{data.gene}</strong> ({data.inheritance})<br />
        <strong>EEG hallmark:</strong> {data.eeg_hallmark}<br />
        <strong>Key biomarker:</strong> {data.key_biomarker}
      </div>

      {/* Quinidine RCT Negative Banner */}
      <div className="alert alert-warning mb-4" style={{ borderLeft: `5px solid ${ACCENT2}`, fontSize: 13 }}>
        <strong>⚠️ QUINIDINE NOT RECOMMENDED IN EIMFS</strong> — Phase 2 RCT (Numis 2020, <em>Epilepsia</em>) showed NO significant seizure reduction vs placebo (p=0.38) in KCNT1-EIMFS.<br />
        <span className="text-success fw-bold">PREFERRED: Ketogenic Diet 4:1 → 50–80% seizure reduction in EIMFS case series. Initiate at 2nd AED failure.</span><br />
        <span className="text-muted">Quinidine may be trialled in NFLE-KCNT1 (familial, milder) with QTc monitoring — NOT for EIMFS infants.</span>
      </div>

      <div className="alert alert-info mb-4" style={{ fontSize: 13 }}>
        <strong>⚡ Clinical AHA — KCNT1/EIMFS:</strong> {data.key_aha}
      </div>

      <div className="row g-2 mb-4">
        <KPI label="N Patients"       value={data.n_patients ?? '—'}                    color={ACCENT} />
        <KPI label="GOF Severe"        value={`${k.gof_severe_pct ?? '—'}%`}             color={ACCENT2} />
        <KPI label="GOF Moderate"      value={`${k.gof_moderate_pct ?? '—'}%`}           color='#7d6608' />
        <KPI label="DRE Rate"          value={`${k.dre_pct ?? '—'}%`}                    color={ACCENT2} />
        <KPI label="Sz-Free"           value={`${k.seizure_free_pct ?? '—'}%`}           color={ACCENT3} />
        <KPI label="KD Active"         value={`${k.kd_on_pct ?? '—'}%`}                  color={ACCENT3} />
        <KPI label="VNS Implanted"     value={`${k.vns_implanted_pct ?? '—'}%`}          color='#2471a3' />
        <KPI label="Migration EEG"     value={`${k.migration_positive_pct ?? '—'}%`}     color='#8e44ad' />
      </div>

      <SectionCard title="⚠️ Clinical Alerts" borderColor={ACCENT2}>
        {(data.clinical_alerts || []).map((a, i) => (
          <Alert key={i} text={a}
            variant={
              a.includes('🚨') || a.includes('NOT RECOMMENDED') || a.includes('MANDATORY') ? 'danger'
              : a.includes('⚡') || a.includes('EEG') || a.includes('KD') ? 'success'
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
            <thead style={{ backgroundColor: '#eaf3fb' }}>
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
    const matchCat   = catFilter === 'All'   || p.category === catFilter;
    const matchClass = classFilter === 'All' || p.functional_class === classFilter;
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
              <th>ID</th><th>Age(M)</th><th>Sex</th><th>Onset(d)</th>
              <th>Class</th><th>Phase</th><th>Tx</th><th>Control</th>
              <th>PB(µg/mL)</th><th>Na+</th><th>Migration</th><th>KD</th><th>VNS</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(p => (
              <tr key={p.id}
                style={{ backgroundColor: p.seizure_control === 'drug-resistant' ? '#fdf2f2' : undefined }}>
                <td className="fw-bold">{p.id}</td>
                <td>{p.age_months}</td>
                <td>{p.sex}</td>
                <td>{p.onset_age_days}</td>
                <td>
                  <span className={`badge ${
                    p.functional_class === 'GOF-severe'   ? 'bg-danger'
                    : p.functional_class === 'GOF-moderate' ? 'bg-warning text-dark'
                    : p.functional_class === 'GOF-familial' ? 'bg-primary'
                    : 'bg-secondary'
                  }`}>{p.functional_class}</span>
                </td>
                <td className="text-nowrap">{p.disease_phase}</td>
                <td className="text-nowrap">{p.current_treatment}</td>
                <td>
                  <span className={`badge ${
                    p.seizure_control === 'seizure-free'   ? 'bg-success'
                    : p.seizure_control === 'drug-resistant' ? 'bg-danger'
                    : 'bg-warning text-dark'
                  }`}>{p.seizure_control}</span>
                </td>
                <td>{p.pb_level_ugml ?? '—'}</td>
                <td className={p.na_mmoll < 130 ? 'text-danger fw-bold' : ''}>{p.na_mmoll}</td>
                <td><span className={`badge ${p.migration_pattern === 'Yes' ? 'bg-danger' : 'bg-secondary'}`}>{p.migration_pattern}</span></td>
                <td>{p.kd_active ? '✓' : '—'}</td>
                <td>{p.vns_implanted ? '✓' : '—'}</td>
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

      <SectionCard title="🌡️ Seizure Triggers & Rates" borderColor={ACCENT2}>
        {(data.triggers || []).map((t, i) => (
          <div className="mb-3 pb-2 border-bottom" key={i}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="fw-bold">{t.trigger}</span>
              <span className="badge bg-danger">{t.rate_pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar bg-danger" style={{ width: `${t.rate_pct}%` }} />
            </div>
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  const [expanded, setExpanded] = useState(null);
  if (!data) return <div className="text-muted">Loading…</div>;

  const evidenceColor = (ev) => {
    if (!ev) return '#6c757d';
    if (ev.includes('NOT RECOMMENDED') || ev.includes('CONTRAINDICATED')) return '#922b21';
    if (ev.includes('A')) return '#1a5276';
    if (ev.includes('B')) return ACCENT;
    if (ev.includes('C') || ev.includes('Phase')) return '#117a65';
    return '#6c757d';
  };

  return (
    <>
      <div className="alert alert-warning mb-3" style={{ fontSize: 13 }}>
        ⚠️ <strong>QUINIDINE NOT RECOMMENDED IN EIMFS</strong> — negative RCT 2020 (Numis et al. <em>Epilepsia</em>). No significant seizure reduction vs placebo. See entry below.
      </div>
      <div className="alert alert-success mb-3" style={{ fontSize: 13 }}>
        ✅ <strong>KETOGENIC DIET 4:1</strong> — most effective intervention in EIMFS. 50–80% seizure reduction (case series). Initiate at 2nd AED failure — do NOT defer.
      </div>

      {(data.treatments || []).map((t, i) => (
        <div className="card mb-3 shadow-sm" key={i}
          style={{ borderLeft: `4px solid ${t.evidence?.includes('NOT RECOMMENDED') ? ACCENT2 : evidenceColor(t.evidence)}` }}>
          <div className="card-header d-flex justify-content-between align-items-center"
            style={{ cursor: 'pointer', backgroundColor: t.evidence?.includes('NOT RECOMMENDED') ? '#fef9e7' : '#eaf3fb' }}
            onClick={() => setExpanded(expanded === i ? null : i)}>
            <span className="fw-bold small">{t.drug}</span>
            <div className="d-flex align-items-center gap-2">
              <span className="badge" style={{ backgroundColor: evidenceColor(t.evidence), fontSize: 10 }}>{t.evidence}</span>
              <span>{expanded === i ? '▲' : '▼'}</span>
            </div>
          </div>
          {expanded === i && (
            <div className="card-body small">
              <p><strong>Indication:</strong> {t.indication}</p>
              <p><strong>Dose:</strong> {t.dose}</p>
              <p><strong>MOA:</strong> {t.moa}</p>
              <p><strong>Efficacy:</strong> {t.efficacy}</p>
              <p><strong>Safety:</strong> {t.safety}</p>
              <p><strong>Monitoring:</strong> {t.monitoring}</p>
              <p className="mb-0"><strong>Contraindications:</strong> {t.contraindications}</p>
            </div>
          )}
        </div>
      ))}

      <SectionCard title="📊 AED Monitoring Panel" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#eaf3fb' }}>
              <tr><th>Item</th><th>Target</th><th>Frequency</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(data.aed_monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold">{m.item}</td>
                  <td className="text-nowrap">{m.target}</td>
                  <td>{m.frequency}</td>
                  <td className="small">{m.rationale}</td>
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
      <SectionCard title="🚫 Absolute Contraindications / Not Recommended" borderColor={ACCENT2}>
        {(data.absolute_contraindications || []).map((c, i) => (
          <div className="mb-3 pb-2 border-bottom" key={i}>
            <div className="fw-bold text-danger">{c.drug}</div>
            <div className="small mb-1"><strong>Scope:</strong> {c.scope}</div>
            <div className="small mb-1"><strong>Mechanism:</strong> {c.mechanism}</div>
            <div className="small mb-1"><strong>Action:</strong> <em>{c.action}</em></div>
            <div className="small text-muted">{c.evidence}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Monitoring Thresholds (10)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#eaf3fb' }}>
              <tr><th>Threshold</th><th>Required Action</th></tr>
            </thead>
            <tbody>
              {(data.thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold text-nowrap">{t.threshold}</td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🧠 Key Concepts (14)" borderColor={ACCENT}>
        {(data.concepts || []).map((c, i) => (
          <div className="mb-3 pb-2 border-bottom" key={i}>
            <div className="fw-bold text-primary">{c.term}</div>
            <div className="small">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📚 Evidence Standards" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#eaf3fb' }}>
              <tr><th>Standard</th><th>Title</th><th>Relevance</th></tr>
            </thead>
            <tbody>
              {(data.standards || []).map((s, i) => (
                <tr key={i}>
                  <td className="fw-bold text-nowrap">{s.standard}</td>
                  <td>{s.title}</td>
                  <td className="small">{s.relevance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🔬 Key References (6)" borderColor={ACCENT}>
        {(data.references || []).map((r, i) => (
          <div className="mb-2 pb-1 border-bottom" key={i}>
            <div className="fw-bold small">{r.ref}</div>
            <div className="small text-muted">{r.title}</div>
            <div className="small"><em>{r.relevance}</em></div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function KCNT1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/kcnt1/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2 || tab === 3) {
      if (!breakdown) {
        fetch(`${API}/api/kcnt1/breakdown`)
          .then(r => r.json()).then(setBreakdown)
          .catch(e => setError(e.message));
      }
    }
    if (tab === 4) {
      if (!definitions) {
        fetch(`${API}/api/kcnt1/definitions`)
          .then(r => r.json()).then(setDefinitions)
          .catch(e => setError(e.message));
      }
    }
  }, [tab, breakdown, definitions]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <div>
          <h4 className="mb-0" style={{ color: ACCENT }}>
            ⚡ KCNT1 Encephalopathy (EIMFS / NFLE-KCNT1-DEE)
          </h4>
          <div className="text-muted small">
            KNa1.1 / Slack / Slo2.2 · Migrating Focal Seizures · KD 4:1 First-Line · Quinidine RCT Negative · POLG-VPA · 41 patients
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger small">API error: {error}</div>}

      <div className="mb-3">
        {TABS.map((t, i) => (
          <TabBtn key={t} label={t} active={tab === i} onClick={() => setTab(i)} />
        ))}
      </div>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizureTriggersTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
