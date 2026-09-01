'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Variants', 'Metabolic & Seizures', 'Treatments', 'Definitions'];
const COLOR = '#1a237e';   // deep navy — DGUOK/MDDS3 (hepatocerebral mtDNA depletion; VPA+KD absolute CI)
const LIGHT = '#e8eaf6';

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
        ⛔ VPA = ABSOLUTE CONTRAINDICATION IN DGUOK — LETHAL HEPATOTOXICITY
      </div>
      <div className="mb-4 p-2 rounded fw-semibold text-center" style={{ background: '#e65100', color: 'white', fontSize: '0.95rem' }}>
        🚫 KETOGENIC DIET = CONTRAINDICATED — Forces OXPHOS-Dependent Pathway That Fails in mtDNA Depletion
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Nystagmus (PATHOGNOMONIC)" value={`${kpis.nystagmus_pct}%`} color={COLOR} />
        <KPI label="Lactic Acidosis" value={`${kpis.lactic_acidosis_pct}%`} color="#c62828" />
        <KPI label="Hepatic Failure" value={`${kpis.hepatic_failure_pct}%`} color="#e65100" />
        <KPI label="Hypoglycemia" value={`${kpis.hypoglycemia_pct}%`} color="#f57f17" />
        <KPI label="Hypotonia" value={`${kpis.hypotonia_pct}%`} color="#6a1b9a" />
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

      {/* Key Alerts */}
      <SectionCard title="⚠️ Two Clinical Forms — Critical Management Distinction" borderColor="#e65100">
        <div className="row g-3 small">
          <div className="col-md-6">
            <div className="p-3 rounded" style={{ background: '#ffebee', border: '2px solid #c62828' }}>
              <div className="fw-bold mb-1" style={{ color: '#c62828' }}>🔴 HEPATOCEREBRAL FORM (75%) — Fatal</div>
              <div>Null/severe genotype → &lt;10% DGUOK activity → neonatal liver failure + nystagmus + encephalopathy + epilepsy + regression</div>
              <div className="mt-1 fw-semibold">Liver transplant: corrects hepatic disease but does NOT prevent neurological progression</div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-3 rounded" style={{ background: '#e8f5e9', border: '2px solid #2e7d32' }}>
              <div className="fw-bold mb-1" style={{ color: '#2e7d32' }}>🟢 HEPATIC-ONLY FORM (25%) — OLT Curative</div>
              <div>Mild genotype → ≥20% DGUOK activity → liver + nystagmus; preserved CNS development</div>
              <div className="mt-1 fw-semibold">Liver transplant: CURATIVE if performed before neurological compromise</div>
            </div>
          </div>
        </div>
      </SectionCard>

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
      <SectionCard title="👥 Cohort Clinical Forms (n=40, seed-549)">
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

      <SectionCard title="🧬 Variant Distribution (biallelic AR)">
        {variants.map((v, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: i < 2 ? LIGHT : '#fff', border: '1px solid #ddd' }}>
            <div className="d-flex justify-content-between mb-1">
              <span className="small fw-bold">{v.variant}</span>
              <span className="small text-muted">{v.pct}% ({v.n_alleles} alleles)</span>
            </div>
            <div className="progress mb-1" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${v.pct}%`, backgroundColor: i === 0 ? COLOR : i === 1 ? '#283593' : '#888' }} />
            </div>
            <div className="text-muted" style={{ fontSize: '0.75rem' }}>{v.effect}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔬 Key Biomarkers">
        <div className="row g-3 small">
          {[
            { label: 'Nystagmus (PATHOGNOMONIC)', pct: bm.nystagmus_pct, color: COLOR, desc: 'Rotary/pendular; first sign in neonatal period' },
            { label: 'Lactic Acidosis (100%)', pct: bm.lactic_acidosis_pct, color: '#c62828', desc: 'pH <7.1; lactate >10 mmol/L neonatal; L:P >20:1' },
            { label: 'Hepatic Failure', pct: bm.hepatic_failure_pct, color: '#e65100', desc: 'Hepatomegaly + coagulopathy + jaundice' },
            { label: 'Hypotonia', pct: bm.hypotonia_pct, color: '#6a1b9a', desc: 'Generalised; early; central origin' },
            { label: 'Psychomotor Regression', pct: bm.regression_pct, color: '#4a148c', desc: 'Hepatocerebral form; after initial development' },
            { label: 'Hypoglycemia', pct: bm.hypoglycemia_pct, color: '#f57f17', desc: 'Hepatic gluconeogenesis failure; glucose monitoring mandatory' },
            { label: 'Epilepsy', pct: bm.epilepsy_pct, color: '#880e4f', desc: 'Focal motor, GTC, myoclonic; secondary to metabolic encephalopathy' },
            { label: 'Hepatic-Only Form', pct: bm.hepatic_only_form_pct, color: '#2e7d32', desc: 'Preserved neurological development; OLT curative' },
          ].map((b, i) => (
            <div key={i} className="col-md-3 col-6">
              <div className="card shadow-sm text-center">
                <div className="card-body py-2">
                  <div className="fw-bold fs-5" style={{ color: b.color }}>
                    {typeof b.pct === 'number' ? `${b.pct}%` : b.pct}
                  </div>
                  <div className="fw-semibold small">{b.label}</div>
                  <div className="text-muted" style={{ fontSize: '0.7rem' }}>{b.desc}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-3 p-2 rounded fw-bold small" style={{ background: '#e8f5e9', borderLeft: '4px solid #2e7d32' }}>
          🟢 NO 3-MGA-uria (0%) — CRITICAL DDx: excludes all 3-methylglutaconic aciduria syndromes (SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB). Urine organics: lactic acid elevated only.
        </div>
      </SectionCard>
    </div>
  );
}

function MetabolicSeizuresTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const bm = data.biomarker_summary || {};
  const seizures = data.seizure_profile || [];
  const metabolic = data.metabolic_outcomes || [];

  return (
    <div>
      <SectionCard title="🧪 Metabolic Crisis Profile" borderColor="#e65100">
        <Alert variant="danger" text="VPA IS ABSOLUTELY CONTRAINDICATED — identical mechanism to POLG: mtDNA depletion + CoA sequestration + epoxide hepatotoxicity. ANY illness + NPO: IV 10% dextrose GIR 8-10 mg/kg/min IMMEDIATELY. Glucose 2-hourly. Never allow fasting >2h in DGUOK." />
        <Alert variant="warning" text="KETOGENIC DIET CONTRAINDICATED — high-fat substrate forces β-oxidation + ketone oxidation (OXPHOS-dependent pathways that fail in mtDNA depletion) → worsens lactic acidosis → life-threatening decompensation. Never prescribe KD in any mtDNA depletion syndrome." />
        <div className="table-responsive mt-2">
          <table className="table table-sm small">
            <thead><tr style={{ background: '#fff3e0' }}>
              <th>Metabolic Outcome</th><th>N</th><th>%</th><th>Clinical Notes</th>
            </tr></thead>
            <tbody>
              {metabolic.map((m, i) => (
                <tr key={i} style={{ background: i === 0 ? '#fff3e0' : 'transparent' }}>
                  <td><strong>{m.outcome}</strong></td>
                  <td>{m.n}</td>
                  <td><span className={`badge ${m.pct >= 80 ? 'bg-danger' : m.pct >= 50 ? 'bg-warning text-dark' : 'bg-info text-dark'}`}>{m.pct}%</span></td>
                  <td className="text-muted">{m.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Alert variant="info" text="METABOLIC CRISIS PROTOCOL: (1) Glucose FIRST — IV 10% dextrose 2 mL/kg bolus + GIR 8-10 mg/kg/min; (2) Lactic acidosis — NaHCO₃ 0.5 mmol/kg only if pH <7.1; IV dextrose is primary; (3) Seizures — IV LEV 20-40 mg/kg + IV midazolam; NEVER VPA/propofol; (4) Liver protection — LFTs + INR + ammonia + glucose 4-hourly; (5) Call metabolic genetics on-call." />
      </SectionCard>

      <SectionCard title="⚡ Epilepsy Profile — Secondary to Metabolic Encephalopathy" borderColor="#880e4f">
        <Alert variant="danger" text={`Epilepsy in ${bm.epilepsy_pct}% — secondary to neuronal OXPHOS failure from cerebellar/cortical mtDNA depletion; NOT the primary driver unlike POLG/Alpers. NEVER use VPA, NEVER propofol. IV LEV 20-40 mg/kg is first-line for SE.`} />
        <Alert variant="warning" text="Unlike POLG, DGUOK does NOT produce EPC (epilepsia partialis continua) — EPC is a POLG/Alpers hallmark. DGUOK seizures are focal motor, GTC, or myoclonic triggered by metabolic decompensation. Correcting hypoglycemia often terminates seizures without additional AEDs." />
        <div className="table-responsive mt-2">
          <table className="table table-sm small">
            <thead><tr style={{ background: '#fce4ec' }}>
              <th>Seizure Type</th><th>N</th><th>%</th><th>Clinical Notes</th>
            </tr></thead>
            <tbody>
              {seizures.map((s, i) => (
                <tr key={i}>
                  <td><strong>{s.type}</strong></td>
                  <td>{s.n}</td>
                  <td><span className={`badge ${s.pct >= 40 ? 'bg-danger' : s.pct >= 20 ? 'bg-warning text-dark' : 'bg-secondary'}`}>{s.pct}%</span></td>
                  <td className="text-muted">{s.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📊 Disease Outcomes">
        <div className="row g-3 small">
          {Object.entries(data.outcomes || {}).map(([k, v], i) => (
            <div key={i} className="col-md-4 col-6">
              <div className="card text-center shadow-sm">
                <div className="card-body py-2">
                  <div className="fw-bold" style={{ color: k.includes('vpa') || k.includes('survival') || k.includes('neurological') ? '#c62828' : k.includes('curative') || k.includes('language') ? '#2e7d32' : COLOR }}>
                    {typeof v === 'number' && k.includes('pct') ? `${v}%` :
                     typeof v === 'number' && k.includes('months') ? `${v} mo` :
                     typeof v === 'number' && k.includes('weeks') ? `${v} wk` :
                     v}
                  </div>
                  <div className="text-muted" style={{ fontSize: '0.72rem' }}>{k.replace(/_/g, ' ')}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading treatments...</div>;
  const treatments = data.treatment_distribution || [];

  return (
    <div>
      <Alert variant="danger" text="⛔ ABSOLUTE RULE: VPA (VALPROATE) IS PERMANENTLY CONTRAINDICATED IN DGUOK/MDDS3. Document in every medical record, allergy system, and emergency letter. No exceptions." />
      <Alert variant="warning" text="🚫 KETOGENIC DIET: CONTRAINDICATED in DGUOK and all mtDNA depletion syndromes. High-fat metabolism requires intact OXPHOS — forces pathway that fails → worsens lactic acidosis." />

      <SectionCard title="💊 Treatment Distribution (n=40)">
        {treatments.map((t, i) => {
          const isLevelA = t.indication?.includes('Level A');
          const isLevelB = t.indication?.includes('Level B');
          return (
            <div key={i} className="mb-3 p-2 rounded" style={{ background: isLevelA ? LIGHT : '#f9f9f9', border: isLevelA ? `1px solid ${COLOR}` : '1px solid #eee' }}>
              <div className="d-flex justify-content-between align-items-start mb-1">
                <div>
                  <span className="fw-bold small">{t.treatment}</span>
                  {isLevelA && <span className="badge ms-2" style={{ background: COLOR, fontSize: '0.65rem' }}>LEVEL A</span>}
                  {isLevelB && !isLevelA && <span className="badge ms-2 bg-info" style={{ fontSize: '0.65rem' }}>LEVEL B</span>}
                </div>
                <span className="small text-muted">{t.n}/{data.cohort} ({t.pct}%)</span>
              </div>
              <div className="progress mb-1" style={{ height: 8 }}>
                <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: isLevelA ? COLOR : isLevelB ? '#1565c0' : '#888' }} />
              </div>
              <div className="text-muted" style={{ fontSize: '0.75rem' }}>{t.indication}</div>
            </div>
          );
        })}
      </SectionCard>

      <SectionCard title="🚦 Prescribing Decision Guide">
        <Alert variant="danger" text="VPA (Valproate): ABSOLUTE CONTRAINDICATION — identical mechanism to POLG: (1) VPA inhibits residual DGUOK dNTP supply → complete mtDNA depletion in already-depleted hepatocytes → necrosis; (2) CoA sequestration → FAO collapse; (3) VPA epoxide → direct hepatotoxicity. NEVER USE in any mtDNA depletion syndrome." />
        <Alert variant="danger" text="Ketogenic Diet: CONTRAINDICATED — forces β-oxidation + ketone body oxidation (OXPHOS-dependent) → fails in mtDNA-depleted mitochondria → worsens lactic acidosis → life-threatening decompensation. Unlike channelopathies (SCN1A, KCNQ2) where KD is first-line, KD is unsafe in DGUOK." />
        <Alert variant="success" text="IV 10% Dextrose: LEVEL A MANDATORY — any NPO >2h or intercurrent illness; GIR 8-10 mg/kg/min; glucose target 5-10 mmol/L; families carry Dextrogel; written sick-day protocol." />
        <Alert variant="success" text="LEV (levetiracetam): PREFERRED AED — no hepatic metabolism; no mito toxicity; no ammonia effect; no P450 induction; IV loading 20-40 mg/kg for SE; safe in liver failure." />
        <Alert variant="success" text="Liver transplant (hepatic-only form): LEVEL B — curative if performed before neurological compromise; specialist metabolic + hepatology + neurology + ethics consensus required; DOES NOT help hepatocerebral form." />
        <Alert variant="warning" text="Propofol: AVOID — PRIS risk in any mitochondrial disease. Use ketamine + sevoflurane/desflurane. Alert anaesthesia team to DGUOK diagnosis before every procedure." />
        <Alert variant="info" text="Nucleotide supplementation (dG + dA): investigational Level D — oral deoxyguanosine + deoxyadenosine to replenish dNTP pool; animal model benefit; no RCT data; compassionate use only; does not reverse established depletion." />
        <Alert variant="info" text="CoQ10 / Riboflavin: empirical Level D — no controlled evidence in DGUOK; generally safe; marginal support for residual OXPHOS; does not alter mtDNA depletion course." />
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const defs = data.definitions || [];

  return (
    <div>
      <SectionCard title="📖 DGUOK MDDS3 — Definitions">
        <div className="small text-muted mb-3">
          Disease: {data.disease} · Gene: {data.gene} · OMIM Gene: {data.omim_gene} · Disease: {data.omim_disease}
        </div>
        {defs.map((d, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ background: i % 2 === 0 ? '#fff' : '#fafafa', border: '1px solid #eee' }}>
            <div className="fw-bold mb-1" style={{ color: COLOR }}>{d.term}</div>
            <div className="small mb-2">{d.definition}</div>
            <div className="small text-muted p-2 rounded" style={{ background: LIGHT, borderLeft: `3px solid ${COLOR}` }}>
              <strong>Clinical relevance:</strong> {d.relevance}
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

export default function DGUOKPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/dguok/overview`).then(r => r.json()),
      fetch(`${API}/api/dguok/breakdown`).then(r => r.json()),
      fetch(`${API}/api/dguok/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#f8f9fa' }}>
      {/* Header */}
      <div style={{ background: COLOR, color: 'white', padding: '16px 24px' }}>
        <h4 className="mb-1 fw-bold">⛔ DGUOK — Hepatocerebral mtDNA Depletion Syndrome (MDDS3)</h4>
        <div className="small opacity-75">
          mtDNA Depletion Syndrome 3 · 2p13.1 · OMIM 251880 · VPA ABSOLUTE CI · KD CONTRAINDICATED · Nystagmus 90% PATHOGNOMONIC
        </div>
        <div className="small opacity-75 mt-1">
          DGUOK (277aa) · Deoxyguanosine Kinase · Purine Salvage dGTP/dATP · Hepatocerebral 75% / Hepatic-Only 25% (OLT Curative) · NO 3-MGA · Mandel 2001 NatGenet
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: 'white', borderBottom: '1px solid #dee2e6' }}>
        <div className="container-fluid px-3">
          <ul className="nav nav-tabs border-0">
            {TABS.map((t, i) => (
              <li key={i} className="nav-item">
                <button
                  className={`nav-link ${tab === i ? 'active' : ''}`}
                  style={tab === i ? { color: COLOR, borderBottomColor: COLOR, fontWeight: 600 } : {}}
                  onClick={() => setTab(i)}
                >{t}</button>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Content */}
      <div className="container-fluid px-3 py-3">
        {loading && <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /></div>}
        {error && <div className="alert alert-danger">Error: {error}</div>}
        {!loading && !error && (
          <>
            {tab === 0 && <OverviewTab data={overview} />}
            {tab === 1 && <PatientsTab data={breakdown} />}
            {tab === 2 && <MetabolicSeizuresTab data={breakdown} />}
            {tab === 3 && <TreatmentsTab data={breakdown} />}
            {tab === 4 && <DefinitionsTab data={definitions} />}
          </>
        )}
      </div>
    </div>
  );
}
