'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a5276';   // deep navy — POLG / mitochondrial
const ACCENT2 = '#7b241c';   // dark red — VPA CI / danger
const ACCENT3 = '#1e6823';   // dark green — LEV safe / mitochondrial cofactors
const ACCENT4 = '#6c3483';   // purple — EPC / Alpers

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
        <strong>🧬 {data.syndrome}</strong><br />
        Gene: <strong>{data.gene}</strong> · {data.inheritance}<br />
        <strong>EEG hallmark:</strong> {data.eeg_hallmark}<br />
        <strong>Key biomarker:</strong> {data.key_biomarker}
      </div>

      {/* VPA DANGER Banner — most prominent element */}
      <div className="alert alert-danger mb-4" style={{ borderLeft: `5px solid ${ACCENT2}`, fontSize: 14 }}>
        <strong>🚨 VPA ABSOLUTE CONTRAINDICATION — ALL POLG PATIENTS</strong><br />
        Valproate in POLG = acute liver failure (Alpers hepatopathy) in <strong>32–45%</strong> · mortality <strong>&gt;80%</strong>.<br />
        Document as <strong>ALLERGY</strong> in EMR immediately. Alert GP, school nurse, A&amp;E team.<br />
        Use <strong>LEV 60 mg/kg IV</strong> as first-line — hepatically safe. <strong>Never VPA even in SE.</strong>
      </div>

      <div className="alert alert-success mb-4" style={{ borderLeft: `5px solid ${ACCENT3}`, fontSize: 13 }}>
        <strong>✅ FIRST-LINE: {data.precision_therapy}</strong><br />
        <span className="text-muted">Mitochondrial cofactors (Riboflavin B2 + CoQ10 + L-carnitine) for ALL POLG patients. Sick day plan mandatory (CLB rescue + glucose). POLG sequencing BEFORE VPA in at-risk children.</span>
      </div>

      <div className="alert alert-warning mb-4" style={{ borderLeft: `5px solid ${ACCENT4}`, fontSize: 13 }}>
        <strong>⚡ EPC = POLG UNTIL PROVEN OTHERWISE:</strong> Epilepsia partialis continua + liver disease + regression = Alpers-Huttenlocher syndrome.<br />
        Start POLG workup immediately · LEV + CLB + ketamine infusion · NEVER phenobarbitone first-line
      </div>

      <div className="alert alert-info mb-4" style={{ fontSize: 13 }}>
        <strong>💡 Clinical AHA:</strong> {data.key_aha}
      </div>

      <div className="row g-2 mb-4">
        <KPI label="N Patients"       value={data.n_patients ?? '—'}               color={ACCENT} />
        <KPI label="EPC Present"      value={`${k.epc_pct ?? '—'}%`}               color={ACCENT4} />
        <KPI label="Drug Resistant"   value={`${k.dre_pct ?? '—'}%`}               color={ACCENT2} />
        <KPI label="VPA Exposed"      value={`${k.vpa_exposed_pct ?? '—'}%`}       color='#922b21' />
        <KPI label="VPA Liver Injury" value={`${k.vpa_liver_injury_pct ?? '—'}%`}  color={ACCENT2} />
        <KPI label="KD Trialed"       value={`${k.kd_trialed_pct ?? '—'}%`}        color='#7d6608' />
        <KPI label="On Cofactors"     value={`${k.cofactors_pct ?? '—'}%`}         color={ACCENT3} />
      </div>

      <SectionCard title="🚨 Clinical Alerts" borderColor={ACCENT2}>
        {(data.clinical_alerts || []).map((a, i) => (
          <Alert key={i} text={a}
            variant={
              a.includes('🚨') || a.includes('ABSOLUTE') || a.includes('NEVER') ? 'danger'
              : a.includes('⚠️') || a.includes('MANDATORY') ? 'warning'
              : a.includes('✅') ? 'success'
              : 'info'
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
              <PctBar key={i} label={k2} pct={v} color={ACCENT4} />
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
  const [epcFilter, setEpcFilter] = useState('All');
  if (!data) return <div className="text-muted">Loading…</div>;

  const cats = ['All', ...new Set((data.patients || []).map(p => p.category))];
  const filtered = (data.patients || []).filter(p => {
    const matchCat   = catFilter === 'All' || p.category === catFilter;
    const matchEpc   = epcFilter === 'All' || (epcFilter === 'EPC' ? p.epc_present : !p.epc_present);
    const matchSearch = !search || JSON.stringify(p).toLowerCase().includes(search.toLowerCase());
    return matchCat && matchEpc && matchSearch;
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
          <select className="form-select form-select-sm" value={epcFilter} onChange={e => setEpcFilter(e.target.value)}>
            {['All','EPC','No EPC'].map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </div>

      <div className="table-responsive mb-4">
        <table className="table table-sm table-hover table-bordered small">
          <thead className="table-dark">
            <tr>
              <th>ID</th><th>Age(M)</th><th>Sex</th><th>Onset(Y)</th>
              <th>Class</th><th>Phase</th><th>Treatment</th><th>Control</th>
              <th>EPC</th><th>VPA Exp</th><th>Liver Inj</th>
              <th>CSF Lac</th><th>mtDNA%</th><th>Cofactors</th><th>KD</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(p => (
              <tr key={p.id}
                style={{
                  backgroundColor: p.vpa_exposed && p.liver_injury_vpa ? '#fdf2f2'
                    : p.epc_present ? '#fdf5ff' : undefined
                }}>
                <td className="fw-bold">{p.id}</td>
                <td>{p.age_months}</td>
                <td>{p.sex}</td>
                <td>{p.onset_years}</td>
                <td>
                  <span className={`badge ${
                    p.functional_class === 'AR-mtDNA-depletion-Alpers' ? 'bg-danger'
                    : p.functional_class === 'AR-mtDNA-depletion-infantile-severe' ? 'bg-dark'
                    : p.functional_class === 'AR-mtDNA-depletion-juvenile' ? 'bg-warning text-dark'
                    : p.functional_class === 'AD-mtDNA-deletions-PEO' ? 'bg-info text-dark'
                    : 'bg-secondary'
                  } small`} style={{ fontSize: 10 }}>{p.functional_class}</span>
                </td>
                <td className="text-nowrap small">{p.disease_phase}</td>
                <td className="text-nowrap small">{p.current_treatment}</td>
                <td>
                  <span className={`badge ${
                    p.seizure_control === 'drug-resistant' ? 'bg-danger'
                    : 'bg-warning text-dark'
                  }`}>{p.seizure_control}</span>
                </td>
                <td>
                  {p.epc_present
                    ? <span className="badge" style={{ backgroundColor: ACCENT4 }}>EPC ⚡</span>
                    : <span className="text-muted small">—</span>}
                </td>
                <td>
                  {p.vpa_exposed
                    ? <span className="badge bg-danger">VPA⚠</span>
                    : <span className="text-muted small">—</span>}
                </td>
                <td>
                  {p.liver_injury_vpa
                    ? <span className="badge bg-danger">ALF⚠</span>
                    : <span className="text-muted small">—</span>}
                </td>
                <td className={p.csf_lactate_mmol > 3.0 ? 'text-danger fw-bold' : ''}>{p.csf_lactate_mmol}</td>
                <td className={p.mtdna_depletion_pct > 70 ? 'text-danger fw-bold' : ''}>{p.mtdna_depletion_pct ?? '—'}%</td>
                <td>{p.mito_cofactors ? '✓' : '—'}</td>
                <td>{p.kd_trialed ? '✓' : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-muted small">
          Showing {filtered.length} of {(data.patients || []).length} patients ·
          Red row = VPA-exposed + liver injury · Purple row = EPC present ·
          CSF Lactate &gt;3.0 mmol/L = red · mtDNA depletion &gt;70% = red
        </div>
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
                <p className="mb-0"><strong>Clinical Note:</strong> {e.clinical_note}</p>
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
      <SectionCard title="⚡ Seizure Types (N=41 cohort)" borderColor={ACCENT4}>
        {(data.seizure_types || []).map((s, i) => (
          <div className="mb-4 pb-3 border-bottom" key={i}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="fw-bold">{s.type}</span>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{s.prevalence_pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${s.prevalence_pct}%`, backgroundColor: ACCENT4 }} />
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
    if (ev.includes('ABSOLUTE CI') || ev.includes('ABSOLUTE')) return '#922b21';
    if (ev.includes('Level A')) return '#1a5276';
    if (ev.includes('Level B')) return ACCENT;
    if (ev.includes('Level C')) return ACCENT3;
    return '#6c757d';
  };

  return (
    <>
      <div className="alert alert-danger mb-3" style={{ fontSize: 13, borderLeft: `5px solid ${ACCENT2}` }}>
        🚨 <strong>VPA / VALPROATE — ABSOLUTE CONTRAINDICATION IN ALL POLG</strong> — any dose, any route, any indication including SE. ALF in 32–45%; mortality &gt;80%. Document as allergy. Use LEV IV 60 mg/kg for SE.
      </div>
      <div className="alert alert-success mb-3" style={{ fontSize: 13 }}>
        ✅ <strong>LEV 60 mg/kg IV loading</strong> — FIRST-LINE in SE (hepatically safe, no mitochondrial toxicity). Maintenance 30-60 mg/kg/day. No drug interactions. Preferred in all POLG disease stages.
      </div>
      <div className="alert alert-primary mb-3" style={{ fontSize: 13 }}>
        🧬 <strong>MITOCHONDRIAL COFACTORS</strong> — ALL POLG patients: Riboflavin (B2) + CoQ10 10-30 mg/kg/day + L-carnitine 50-100 mg/kg/day. Rational metabolic support — no RCT but mechanistically justified, minimal risk.
      </div>

      {(data.treatments || []).map((t, i) => (
        <div className="card mb-3 shadow-sm" key={i}
          style={{ borderLeft: `4px solid ${evidenceColor(t.evidence)}` }}>
          <div className="card-header d-flex justify-content-between align-items-center"
            style={{ cursor: 'pointer', backgroundColor: t.evidence?.includes('ABSOLUTE') ? '#fdf2f2' : '#eaf3fb' }}
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

      <SectionCard title="🔬 Metabolic Monitoring Panel" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#eaf3fb' }}>
              <tr><th>Item</th><th>Target</th><th>Frequency</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(data.aed_monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold">{m.item}</td>
                  <td className="text-nowrap small">{m.target}</td>
                  <td className="small">{m.frequency}</td>
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
      <SectionCard title="🚫 Absolute Contraindications / Critical Safety Rules" borderColor={ACCENT2}>
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
        {(data.definitions || []).map((c, i) => (
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
export default function POLGPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/polg/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2 || tab === 3) {
      if (!breakdown) {
        fetch(`${API}/api/polg/breakdown`)
          .then(r => r.json()).then(setBreakdown)
          .catch(e => setError(e.message));
      }
    }
    if (tab === 4) {
      if (!definitions) {
        fetch(`${API}/api/polg/definitions`)
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
            🧬 POLG Epilepsy — Alpers-Huttenlocher Syndrome (POLG-DEE / mtDNA Depletion)
          </h4>
          <div className="text-muted small">
            POLG 15q26.1 · Mitochondrial DNA Polymerase Gamma · AR (AHS) + AD (PEO/SANDO) ·
            EPC hallmark · VPA ABSOLUTE CI · LEV first-line · Mitochondrial cofactors · 41 patients
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
