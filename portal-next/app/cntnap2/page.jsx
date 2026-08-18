'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR   = '#006064'; // dark cyan-teal — axon-glia adhesion / myelin / neurexin superfamily
const DANGER  = '#b71c1c';
const SUCCESS = '#1b5e20';
const WARN    = '#f57f17';
const AUTO_COLOR = '#6a1b9a'; // purple for autoimmune distinction
const INTERNEURON_COLOR = '#00695c'; // teal for interneuron/GABA

function KPI({ label, value, color = COLOR }) {
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

function Bar({ label, value, max = 100, color = COLOR }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card shadow-sm mb-3">
      <div className="card-header fw-semibold text-white py-2" style={{ background: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function Alert({ msg, severity = 'danger' }) {
  const icon = severity === 'danger' ? '🚫' : severity === 'warning' ? '⚠️' : 'ℹ️';
  return (
    <div className={`alert alert-${severity} py-2 mb-2 small`}>
      {icon} {msg}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const cohort = data.cohort || {};
  const etiologies = data.etiologies || [];
  const alerts = data.key_alerts || [];

  return (
    <div>
      {/* Critical alerts */}
      <SectionCard title="⚠️ Critical Clinical Alerts — CNTNAP2 / CASPR2" borderColor={DANGER}>
        {alerts.map((a, i) => (
          <Alert key={i} msg={a}
            severity={a.includes('ABSOLUTE') || a.includes('MANDATORY') ? 'danger' : a.includes('Surgery') || a.includes('Bumetanide') || a.includes('Largest') ? 'warning' : 'info'} />
        ))}
      </SectionCard>

      {/* Gene identity */}
      <SectionCard title="🧬 CNTNAP2 — Gene Identity & Mechanism">
        <div className="row g-2 mb-3">
          <div className="col-md-6">
            <table className="table table-sm table-bordered mb-0">
              <tbody>
                <tr><td className="fw-semibold">Gene</td><td>CNTNAP2 — Contactin-Associated Protein-Like 2</td></tr>
                <tr><td className="fw-semibold">Protein</td><td>{data.protein}</td></tr>
                <tr><td className="fw-semibold">Chromosome</td><td>{data.chromosome}</td></tr>
                <tr><td className="fw-semibold">Gene Size</td><td>2.3 Mb — <span className="text-danger fw-bold">LARGEST GENE IN HUMAN GENOME</span></td></tr>
                <tr><td className="fw-semibold">Protein Family</td><td>{data.protein_family}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="col-md-6">
            <table className="table table-sm table-bordered mb-0">
              <tbody>
                <tr><td className="fw-semibold">Inheritance</td><td>{data.inheritance}</td></tr>
                <tr><td className="fw-semibold">OMIM Gene</td><td>{data.omim?.gene}</td></tr>
                <tr><td className="fw-semibold">CDFE Syndrome</td><td>{data.omim?.cdfe}</td></tr>
                <tr><td className="fw-semibold">Pitt-Hopkins-like-1</td><td>{data.omim?.pthsl1}</td></tr>
                <tr><td className="fw-semibold">Founder Mutation</td><td className="small">{data.founder_mutation}</td></tr>
              </tbody>
            </table>
          </div>
        </div>
        <div className="alert alert-info py-2 mb-2 small">
          <strong>Mechanism:</strong> {data.mechanism}
        </div>
        <div className="alert alert-warning py-2 mb-0 small">
          <strong>CASPR2 AUTOIMMUNE vs GENETIC — CRITICAL DISTINCTION:</strong> Genetic CNTNAP2 = AR/AD LOF (childhood onset, developmental regression, FCD). CASPR2-IgG autoimmune = adult limbic encephalitis + Morvan syndrome (neuromyotonia + insomnia + autonomic). <strong>CASPR2-IgG negative in genetic patients.</strong> Test CASPR2 antibodies at diagnosis to exclude autoimmune mimicry — treatment fundamentally different (immunotherapy vs AED).
        </div>
      </SectionCard>

      {/* KPIs */}
      <SectionCard title="📊 Cohort Snapshot (n=40)">
        <div className="row g-2">
          <KPI label="Total Patients" value={cohort.total} />
          <KPI label="Surgery Engel I" value={cohort.seizure_free_surgery_pct} color={SUCCESS} />
          <KPI label="AED Response" value={cohort.seizure_free_aed_pct} color={INTERNEURON_COLOR} />
          <KPI label="Autism Comorbidity" value={`${cohort.autism_comorbidity_pct}%`} color={AUTO_COLOR} />
          <KPI label="Regression Phase" value={`${cohort.regression_phase_pct}%`} color={WARN} />
          <KPI label="Onset (median, y)" value={cohort.median_onset_years} color={COLOR} />
        </div>
        <div className="row g-2 mt-1">
          <KPI label="Surgical Candidates" value={`${cohort.surgical_candidates_pct}%`} color={COLOR} />
        </div>
      </SectionCard>

      {/* Etiology pie */}
      <SectionCard title="🧫 Etiology Distribution">
        {etiologies.map((e, i) => (
          <div key={i} className="mb-2">
            <Bar label={e.class.replace(/-/g, ' ')} value={e.pct} color={i === 4 ? AUTO_COLOR : COLOR} />
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function EtiologiesTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const etiologies = data.etiologies || [];
  return (
    <div>
      {etiologies.map((e, i) => (
        <div className="card shadow-sm mb-3" key={i}>
          <div className="card-header fw-semibold text-white py-2"
            style={{ background: i === 4 ? AUTO_COLOR : COLOR }}>
            {e.class.replace(/-/g, ' ')} — {e.pct}%
          </div>
          <div className="card-body">
            <div className="row g-2">
              <div className="col-md-6">
                <table className="table table-sm table-bordered mb-0">
                  <tbody>
                    <tr><td className="fw-semibold">Variants</td><td>{(e.variants || []).join('; ')}</td></tr>
                    <tr><td className="fw-semibold">Phenotype</td><td>{e.phenotype}</td></tr>
                    <tr><td className="fw-semibold">Inheritance</td><td>{e.inheritance_detail}</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="col-md-6">
                <table className="table table-sm table-bordered mb-0">
                  <tbody>
                    <tr><td className="fw-semibold">MRI</td><td>{e.mri}</td></tr>
                    <tr><td className="fw-semibold">EEG</td><td>{e.eeg}</td></tr>
                    <tr><td className="fw-semibold">Prognosis</td><td>{e.prognosis}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function SeizuresTriggersTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const seizures = data.seizure_types || [];
  const triggers = data.triggers || [];
  return (
    <div>
      <SectionCard title="⚡ Seizure Types (CNTNAP2-CDFE)">
        {seizures.map((s, i) => (
          <div className="card border-0 bg-light mb-2" key={i}>
            <div className="card-body py-2">
              <div className="d-flex justify-content-between mb-1">
                <span className="fw-semibold small">{s.type}</span>
                <span className="badge" style={{ background: COLOR }}>{s.pct}%</span>
              </div>
              <Bar label="" value={s.pct} color={COLOR} />
              <div className="row g-1 mt-1">
                <div className="col-md-4"><span className="text-muted small">EEG: </span><span className="small">{s.eeg_pattern}</span></div>
                <div className="col-md-4"><span className="text-muted small">Semiology: </span><span className="small">{s.semiology}</span></div>
                <div className="col-md-4"><span className="text-muted small">Clinical tip: </span><span className="small text-primary">{s.tips}</span></div>
              </div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🌡️ Seizure Triggers">
        {triggers.map((t, i) => (
          <div className="card border-0 bg-light mb-2" key={i}>
            <div className="card-body py-2">
              <div className="d-flex justify-content-between mb-1">
                <span className="fw-semibold small">{t.trigger}</span>
                <span className="badge" style={{ background: WARN, color: '#000' }}>{t.pct}%</span>
              </div>
              <Bar label="" value={t.pct} color={WARN} />
              <div className="row g-1 mt-1">
                <div className="col-md-6"><span className="text-muted small">Mechanism: </span><span className="small">{t.mechanism}</span></div>
                <div className="col-md-6"><span className="text-muted small">Management: </span><span className="small text-success">{t.management}</span></div>
              </div>
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const treatments = data.treatments || [];
  const cis = data.contraindications || [];
  const monitoring = data.monitoring || [];
  return (
    <div>
      <SectionCard title="💊 Treatments — CNTNAP2 / CDFE">
        {treatments.map((t, i) => (
          <div className="card border-0 bg-light mb-2" key={i}>
            <div className="card-body py-2">
              <div className="d-flex justify-content-between mb-1">
                <span className="fw-semibold">{t.drug}</span>
                <span className="badge bg-secondary">{t.level}</span>
              </div>
              <div className="row g-1">
                <div className="col-md-3"><span className="text-muted small">Mechanism: </span><span className="small">{t.mechanism}</span></div>
                <div className="col-md-2"><span className="text-muted small">Dose: </span><span className="small">{t.dose}</span></div>
                <div className="col-md-2"><span className="text-muted small">Efficacy: </span><span className="small">{t.efficacy}</span></div>
                <div className="col-md-2"><span className="text-muted small">Monitoring: </span><span className="small">{t.monitoring}</span></div>
                <div className="col-md-3"><span className="text-muted small">CNTNAP2-specific: </span><span className="small text-primary fw-semibold">{t.cntnap2_specific}</span></div>
              </div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications" borderColor={DANGER}>
        {cis.map((c, i) => (
          <div className={`alert alert-${c.severity === 'ABSOLUTE' ? 'danger' : c.severity === 'HIGH' ? 'warning' : 'info'} py-2 mb-2`} key={i}>
            <div className="d-flex justify-content-between">
              <strong>{c.drug}</strong>
              <span className={`badge bg-${c.severity === 'ABSOLUTE' ? 'danger' : c.severity === 'HIGH' ? 'warning text-dark' : 'secondary'}`}>{c.severity}</span>
            </div>
            <div className="small mt-1">{c.reason}</div>
            <div className="small text-success mt-1"><strong>Alternative:</strong> {c.alternative}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🩺 Monitoring Checklist">
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr><th>Item</th><th>Frequency</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {monitoring.map((m, i) => (
                <tr key={i}>
                  <td className="fw-semibold small">{m.item}</td>
                  <td className="small">{m.frequency}</td>
                  <td className="small">{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const concepts = data.key_concepts || [];
  const standards = data.standards || [];
  const refs = data.references || [];
  const thresholds = (data.thresholds || []);
  const summary = data.gene_summary || {};
  return (
    <div>
      <SectionCard title="📋 Gene Summary">
        <div className="row g-2">
          <div className="col-md-6">
            <table className="table table-sm table-bordered mb-0">
              <tbody>
                <tr><td className="fw-semibold">Gene</td><td>{summary.gene}</td></tr>
                <tr><td className="fw-semibold">Chromosome</td><td>{summary.chromosome}</td></tr>
                <tr><td className="fw-semibold">Gene Size</td><td>{summary.size_mb} Mb (largest human gene)</td></tr>
                <tr><td className="fw-semibold">Protein</td><td>{summary.protein}</td></tr>
                <tr><td className="fw-semibold">OMIM Gene</td><td>{summary.omim_gene}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="col-md-6">
            <table className="table table-sm table-bordered mb-0">
              <tbody>
                <tr><td className="fw-semibold">Syndromes</td><td>{(summary.syndromes || []).join(', ')}</td></tr>
                <tr><td className="fw-semibold">Key Treatments</td><td>{(summary.key_treatments || []).join(', ')}</td></tr>
                <tr><td className="fw-semibold">Absolute CIs</td><td className="text-danger fw-semibold">{(summary.absolute_CIs || []).join(', ')}</td></tr>
                <tr><td className="fw-semibold">Dashboard Color</td><td><span className="badge px-3" style={{ background: summary.dashboard_colour }}>{summary.dashboard_colour}</span></td></tr>
                <tr><td className="fw-semibold">Color Rationale</td><td className="small">{summary.colour_rationale}</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="📚 Key Concepts (15)">
        <div className="accordion" id="conceptsAcc">
          {concepts.map((c, i) => (
            <div className="accordion-item" key={i}>
              <h2 className="accordion-header">
                <button className="accordion-button collapsed py-2 small fw-semibold"
                  type="button" data-bs-toggle="collapse" data-bs-target={`#concept-${i}`}>
                  {c.term}
                </button>
              </h2>
              <div id={`concept-${i}`} className="accordion-collapse collapse" data-bs-parent="#conceptsAcc">
                <div className="accordion-body py-2 small">{c.definition}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="⚖️ Thresholds (12)">
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold small">{t.name}</td>
                  <td className="small text-danger fw-semibold">{t.value}</td>
                  <td className="small">{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📐 Standards (12)">
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr><th>ID</th><th>Standard</th><th>Applies To</th></tr>
            </thead>
            <tbody>
              {standards.map((s, i) => (
                <tr key={i}>
                  <td className="fw-semibold small">{s.id}</td>
                  <td className="small">{s.name}</td>
                  <td className="small">{s.applies_to}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📖 References (6)">
        {refs.map((r, i) => (
          <div className="mb-2 pb-2 border-bottom small" key={i}>
            <div className="fw-semibold text-primary">{r.id}</div>
            <div className="fst-italic">{r.citation}</div>
            <div className="text-muted mt-1"><strong>Relevance:</strong> {r.relevance}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function LifecycleTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const stages = data.lifecycle_stages || [];
  return (
    <SectionCard title="🔄 Patient Lifecycle Stages">
      <div className="timeline">
        {stages.map((s, i) => (
          <div className="card border-start border-4 mb-3 ps-0" key={i}
            style={{ borderColor: COLOR + ' !important' }}>
            <div className="card-body py-2 ps-3" style={{ borderLeft: `4px solid ${COLOR}` }}>
              <div className="fw-semibold small mb-1" style={{ color: COLOR }}>{i + 1}. {s.stage}</div>
              <div className="small mb-1"><strong>Actions:</strong> {s.key_actions}</div>
              <div className="small text-warning"><strong>Watch for:</strong> {s.watchfor}</div>
            </div>
          </div>
        ))}
      </div>
    </SectionCard>
  );
}

export default function CNTNAP2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/cntnap2/overview`).then(r => r.json()),
      fetch(`${API}/api/cntnap2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/cntnap2/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const renderTab = () => {
    if (loading) return <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /></div>;
    if (error) return <div className="alert alert-danger">Error: {error}</div>;
    switch (tab) {
      case 0: return <OverviewTab data={overview} />;
      case 1: return <EtiologiesTab data={overview} />;
      case 2: return <SeizuresTriggersTab data={breakdown} />;
      case 3: return <TreatmentsTab data={breakdown} />;
      case 4: return (
        <div>
          <DefinitionsTab data={definitions} />
          <LifecycleTab data={breakdown} />
        </div>
      );
      default: return null;
    }
  };

  return (
    <div>
      {/* Header */}
      <div className="py-3 px-3 mb-3 text-white" style={{ background: COLOR }}>
        <h4 className="mb-0 fw-bold">
          🧬 CNTNAP2 Epilepsy — CASPR2 / Cortical Dysplasia-Focal Epilepsy (CDFE) / Pitt-Hopkins-like-1
        </h4>
        <div className="small opacity-75 mt-1">
          Neurexin Superfamily · Juxtaparanodal Kv1.1 Clustering · PV+ Interneuron Deficit · AR-LOF CDFE (#610042) · AD-het PTHSL1 (#614161) · Bumetanide-NKCC1 Investigational · Surgery 50-60% Engel I · 7q35-36.1 · Largest Human Gene (2.3 Mb)
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs px-3 mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button
              className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      <div className="px-3 pb-4">{renderTab()}</div>
    </div>
  );
}
