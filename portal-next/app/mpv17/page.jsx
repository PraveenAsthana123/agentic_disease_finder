'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Variants', 'Metabolic & Seizures', 'Treatments', 'Definitions'];
const COLOR = '#1b5e20';   // deep forest green — MPV17/MDDS6 (hepatocerebral + peripheral neuropathy; Navajo NNH)
const LIGHT = '#e8f5e9';

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

function Bar({ label, value, color = COLOR }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ variant, text }) {
  const bg = variant === 'danger' ? '#ffebee' : variant === 'warning' ? '#fff8e1' : variant === 'success' ? '#e8f5e9' : '#e3f2fd';
  const border = variant === 'danger' ? '#c62828' : variant === 'warning' ? '#f57f17' : variant === 'success' ? '#2e7d32' : '#1565c0';
  return (
    <div className="mb-2 p-2 rounded small" style={{ background: bg, borderLeft: `4px solid ${border}` }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview...</div>;
  const kpis = data.kpis || {};
  const highlights = data.clinical_highlights || [];
  const cis = data.contraindications || [];
  const thresholds = data.thresholds || [];
  const ddx = data.ddx_table || [];

  return (
    <div>
      {/* Critical VPA+KD Warning Banner */}
      <div className="mb-3 p-3 rounded fw-bold text-center" style={{ background: '#b71c1c', color: 'white', fontSize: '1.05rem' }}>
        ⛔ VPA = ABSOLUTE CONTRAINDICATION IN MPV17 — LETHAL HEPATOTOXICITY IN mtDNA DEPLETION
      </div>
      <div className="mb-4 p-2 rounded fw-semibold text-center" style={{ background: '#e65100', color: 'white', fontSize: '0.95rem' }}>
        🚫 KETOGENIC DIET = CONTRAINDICATED — Forces OXPHOS-Dependent Fat Oxidation That Fails in mtDNA Depletion
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Hepatic Failure" value={`${kpis.hepatic_failure_pct}%`} color="#c62828" />
        <KPI label="Lactic Acidosis" value={`${kpis.lactic_acidosis_pct}%`} color="#e65100" />
        <KPI label="Peripheral Neuropathy (CARDINAL)" value={`${kpis.peripheral_neuropathy_pct}%`} color={COLOR} />
        <KPI label="Hypoglycemia" value={`${kpis.hypoglycemia_pct}%`} color="#f57f17" />
        <KPI label="Epilepsy" value={`${kpis.epilepsy_pct}%`} color="#6a1b9a" />
        <KPI label="Hepatic-Only Form" value={`${kpis.hepatic_only_form_pct}%`} color="#2e7d32" />
      </div>

      {/* Identity */}
      <SectionCard title="🧬 Disease Identity">
        <div className="row g-2 small">
          <div className="col-md-6"><strong>Disease:</strong> {data.disease}</div>
          <div className="col-md-6"><strong>Gene:</strong> {data.gene?.split(';')[0]}</div>
          <div className="col-md-4"><strong>Chromosome:</strong> {data.chromosome}</div>
          <div className="col-md-4"><strong>OMIM Gene:</strong> {data.omim_gene} &nbsp; <strong>Disease:</strong> {data.omim_disease}</div>
          <div className="col-md-4"><strong>Inheritance:</strong> {data.inheritance?.split(';')[0]}</div>
          <div className="col-md-6"><strong>Prevalence:</strong> {data.prevalence}</div>
          <div className="col-md-6"><strong>First described:</strong> {data.first_described}</div>
          <div className="col-12"><strong>Category:</strong> {data.category}</div>
          <div className="col-12"><strong>Protein:</strong> <span className="text-muted">{data.gene}</span></div>
        </div>
      </SectionCard>

      {/* Two Clinical Forms */}
      <SectionCard title="⚠️ Two Clinical Forms — Critical Management Distinction" borderColor="#e65100">
        <div className="row g-3 small">
          <div className="col-md-6">
            <div className="p-3 rounded" style={{ background: '#ffebee', border: '2px solid #c62828' }}>
              <div className="fw-bold mb-1" style={{ color: '#c62828' }}>🔴 HEPATOCEREBRAL FORM (90%) — Progressive, Fatal</div>
              <div>Null/severe genotype → hepatic failure + peripheral neuropathy + leukoencephalopathy + regression</div>
              <div className="mt-1 fw-semibold">Liver transplant: corrects hepatic disease but does NOT prevent neurological progression</div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-3 rounded" style={{ background: '#e8f5e9', border: '2px solid #2e7d32' }}>
              <div className="fw-bold mb-1" style={{ color: '#2e7d32' }}>🟢 HEPATIC-ONLY FORM (10%) — Rare, OLT May Cure</div>
              <div>Mild genotype → liver disease predominant; CNS relatively preserved</div>
              <div className="mt-1 fw-semibold">Liver transplant: may be curative if performed before neurological compromise</div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Navajo Founder Banner */}
      <div className="mb-4 p-3 rounded" style={{ background: '#e8eaf6', border: '2px solid #3949ab' }}>
        <div className="fw-bold mb-1" style={{ color: '#3949ab' }}>🧬 Navajo Neurohepatopathy (NNH) — Extreme Founder Effect</div>
        <div className="small">p.Arg50Gln (c.149G>A) founder allele: ~75% of Navajo MPV17 alleles · Carrier freq ~1:26 Navajo · Disease prevalence ~1:1,600 Navajo live births</div>
        <div className="small mt-1">Navajo children with cirrhosis + peripheral neuropathy → MPV17 sequencing SAME DAY</div>
      </div>

      {/* Clinical Highlights */}
      <SectionCard title="🏥 Clinical Highlights">
        {highlights.map((h, i) => (
          <Alert key={i}
            variant={i === 0 ? 'danger' : i === 5 ? 'success' : i === 6 ? 'warning' : 'info'}
            text={h}
          />
        ))}
      </SectionCard>

      {/* Contraindications */}
      <SectionCard title="⛔ Contraindications" borderColor="#c62828">
        {cis.map((ci, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: i < 2 ? '#ffebee' : '#fff8e1', border: `1px solid ${i < 2 ? '#c62828' : '#f57f17'}` }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold small">{ci.drug}</span>
              <span className="badge" style={{ background: i < 2 ? '#c62828' : '#e65100', fontSize: '0.65rem' }}>{ci.level.split('—')[0].trim()}</span>
            </div>
            <div className="text-muted small">{ci.reason}</div>
          </div>
        ))}
      </SectionCard>

      {/* Thresholds */}
      <SectionCard title="📏 Clinical Thresholds & Monitoring">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr style={{ background: LIGHT }}>
              <th>Marker</th><th>Cutoff</th><th>Interpretation</th>
            </tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td><strong>{t.marker}</strong></td>
                  <td><code>{t.cutoff}</code></td>
                  <td className="text-muted">{t.interpretation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* DDx Table */}
      <SectionCard title="🔀 Differential Diagnosis">
        {ddx.map((d, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff', border: '1px solid #ddd' }}>
            <div className="fw-bold small mb-1" style={{ color: COLOR }}>{d.disease}</div>
            <div className="small"><strong>Shared:</strong> <span className="text-muted">{d.shared}</span></div>
            <div className="small mt-1"><strong>Distinguishing:</strong> {d.distinguishing}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const groups = data.phenotype_groups || [];
  const variants = data.variant_distribution || [];
  const bm = data.biomarker_summary || {};

  return (
    <div>
      <SectionCard title="👥 Cohort Clinical Forms (n=40, seed-551)">
        <div className="row g-3 mb-3">
          {groups.map((g, i) => (
            <div key={i} className="col-md-6 mb-3">
              <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${i === 0 ? '#c62828' : '#2e7d32'}` }}>
                <div className="card-body text-center">
                  <div className="fw-bold fs-3" style={{ color: i === 0 ? '#c62828' : '#2e7d32' }}>{g.n}</div>
                  <div className="fw-semibold small">{g.group}</div>
                  <div className="text-muted small">{g.pct}% of cohort</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🧬 Variant Distribution">
        {variants.map((v, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff', border: '1px solid #c8e6c9' }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold small" style={{ color: COLOR }}>{v.variant}</span>
              <span className="badge" style={{ background: COLOR }}>{v.n} ({v.pct}%)</span>
            </div>
            <div className="small text-muted">{v.mechanism}</div>
            <div className="small mt-1"><strong>Form:</strong> {v.form}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📊 Biomarker Summary">
        <div className="row g-2 mb-3">
          {[
            { label: "Hepatic Failure", val: bm.hepatic_failure_pct },
            { label: "Lactic Acidosis", val: bm.lactic_acidosis_pct },
            { label: "Peripheral Neuropathy", val: bm.peripheral_neuropathy_pct },
            { label: "Hypoglycemia", val: bm.hypoglycemia_pct },
            { label: "Coagulopathy", val: bm.coagulopathy_pct },
            { label: "Leukoencephalopathy", val: bm.leukoencephalopathy_pct },
            { label: "Hypotonia", val: bm.hypotonia_pct },
            { label: "Regression", val: bm.regression_pct },
            { label: "Epilepsy", val: bm.epilepsy_pct },
            { label: "Nystagmus (uncommon in MPV17)", val: bm.nystagmus_pct },
          ].filter(x => x.val != null).map((x, i) => <div key={i} className="col-12"><Bar label={x.label} value={x.val} /></div>)}
        </div>
        <div className="small">
          <strong>3-MGA-uria:</strong> <span className="text-success">{bm.three_mga_uria}</span><br />
          <strong>Navajo founder (p.Arg50Gln):</strong> {bm.navajo_founder_pct}% of cohort<br />
          <strong>Hepatocerebral form:</strong> {bm.hepatocerebral_form_pct}% &nbsp;|&nbsp; <strong>Hepatic-only:</strong> {bm.hepatic_only_form_pct}%<br />
          <strong>OLT performed:</strong> {bm.olt_performed_pct}% &nbsp;|&nbsp;
          <strong>Median onset:</strong> {bm.median_onset_months} months &nbsp;|&nbsp;
          <strong>Median diagnosis delay:</strong> {bm.median_diagnosis_delay_weeks} weeks
        </div>
      </SectionCard>
    </div>
  );
}

function MetabolicTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const metabolic = data.metabolic_outcomes || [];
  const seizures = data.seizure_profile || [];
  const outcomes = data.outcomes || {};

  return (
    <div>
      <SectionCard title="🧪 Metabolic Outcomes">
        {metabolic.map((m, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff', border: '1px solid #c8e6c9' }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold small">{m.outcome}</span>
              <span className="badge" style={{ background: COLOR }}>{m.n} / {data.cohort} ({m.pct}%)</span>
            </div>
            <div className="small text-muted">{m.notes}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="⚡ Seizure Profile">
        {seizures.map((s, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: '#fff3e0', border: '1px solid #ff9800' }}>
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-bold small">{s.type}</span>
              <span className="badge bg-warning text-dark">{s.n} ({s.pct}%)</span>
            </div>
            <div className="small text-muted">{s.desc}</div>
          </div>
        ))}
        <Alert variant="danger" text="⛔ NEVER use VPA for seizures in MPV17 — absolute contraindication regardless of seizure type. LEV is the preferred AED." />
      </SectionCard>

      <SectionCard title="📈 Outcome Summary">
        <div className="row g-2 small">
          {Object.entries(outcomes).map(([k, v]) => (
            <div key={k} className="col-md-6">
              <strong>{k.replace(/_/g, ' ')}:</strong> <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const treatments = data.treatment_distribution || [];

  return (
    <div>
      <Alert variant="danger" text="⛔ VPA = ABSOLUTE CONTRAINDICATION in MPV17 regardless of seizure type or indication. No safe dose exists in mtDNA depletion. Document in allergy alerts." />
      <Alert variant="warning" text="🚫 Ketogenic Diet = CONTRAINDICATED — forces OXPHOS-dependent fat oxidation that fails in mtDNA depletion." />
      <Alert variant="warning" text="🚫 Propofol AVOID — PRIS risk in mitochondrial disease. Alternative: ketamine + sevoflurane." />
      <Alert variant="info" text="✅ LEV = Preferred AED — renal excretion, no mito toxicity, safe in hepatic failure. IV 20-40 mg/kg loading for acute seizures." />
      <Alert variant="success" text="🦵 Peripheral Neuropathy = CARDINAL MPV17 feature — physiotherapy + AFOs from diagnosis; NCS/EMG 6-monthly monitoring." />

      {treatments.map((t, i) => (
        <div key={i} className="card mb-3 shadow-sm">
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold small">{t.treatment}</span>
              <span className="badge" style={{ background: COLOR }}>{t.n} ({t.pct}%)</span>
            </div>
            <div className="small text-muted">{t.indication}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const defs = data.definitions || [];
  return (
    <div>
      <div className="mb-3 small text-muted">
        Gene: <strong>{data.gene}</strong> · OMIM Gene: {data.omim_gene} · Disease: {data.omim_disease}
      </div>
      {defs.map((d, i) => (
        <div key={i} className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${COLOR}` }}>
          <div className="card-body">
            <h6 className="fw-bold mb-2" style={{ color: COLOR }}>{d.term}</h6>
            <p className="small text-muted mb-2">{d.definition}</p>
            <div className="small p-2 rounded" style={{ background: LIGHT }}>
              <strong>Clinical Relevance:</strong> {d.relevance}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function MPV17Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);

  const fetchData = async (endpoint, setter, key) => {
    if (loading[key]) return;
    setLoading(l => ({ ...l, [key]: true }));
    try {
      const res = await fetch(`${API}${endpoint}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setter(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(l => ({ ...l, [key]: false }));
    }
  };

  useEffect(() => {
    fetchData('/api/mpv17/overview', setOverview, 'overview');
  }, []);

  useEffect(() => {
    if (activeTab === 1 || activeTab === 2) {
      if (!breakdown) fetchData('/api/mpv17/breakdown', setBreakdown, 'breakdown');
    }
    if (activeTab === 3) {
      if (!breakdown) fetchData('/api/mpv17/breakdown', setBreakdown, 'breakdown');
    }
    if (activeTab === 4) {
      if (!definitions) fetchData('/api/mpv17/definitions', setDefinitions, 'definitions');
    }
  }, [activeTab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 MPV17 Hepatocerebral mtDNA Depletion Syndrome
        </h4>
        <div className="text-muted small">
          Mitochondrial DNA Depletion Syndrome 6 (MDDS6) / Navajo Neurohepatopathy (NNH) ·
          MPV17 Inner Mitochondrial Membrane Channel · 176 aa · 2p23.3 ·
          OMIM Gene 137960 · Disease 256810 · AR
        </div>
        <div className="mt-1 small fw-semibold" style={{ color: '#c62828' }}>
          ⛔ VPA ABSOLUTE CI · 🚫 KD CONTRAINDICATED · 🦵 Peripheral Neuropathy 80% CARDINAL ·
          🚫 No Nystagmus (DDx from DGUOK) · 🌿 Navajo Founder p.Arg50Gln
        </div>
      </div>

      {error && <div className="alert alert-danger small">Error: {error}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${activeTab === i ? ' active fw-semibold' : ''}`}
              onClick={() => setActiveTab(i)}
              style={activeTab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {activeTab === 0 && <OverviewTab data={overview} />}
      {activeTab === 1 && <PatientsTab data={breakdown} />}
      {activeTab === 2 && <MetabolicTab data={breakdown} />}
      {activeTab === 3 && <TreatmentsTab data={breakdown} />}
      {activeTab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
