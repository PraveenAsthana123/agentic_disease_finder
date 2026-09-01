'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Variants', 'Epilepsy & Hepatopathy', 'Treatments', 'Definitions'];
const COLOR = '#b71c1c';   // deep red — POLG/Alpers (VPA absolute CI, lethal hepatotoxicity)
const LIGHT = '#ffebee';

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
      {/* Critical VPA Warning Banner */}
      <div className="mb-4 p-3 rounded fw-bold text-center" style={{ background: '#b71c1c', color: 'white', fontSize: '1.1rem' }}>
        ⛔ VPA (VALPROATE) = ABSOLUTE CONTRAINDICATION IN POLG — LETHAL HEPATOTOXICITY — DO NOT USE
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
          <div className="col-md-6"><strong>Protein:</strong> {data.protein?.split(';')[0]}</div>
          <div className="col-md-6"><strong>Category:</strong> {data.category}</div>
        </div>
      </SectionCard>

      {/* KPIs */}
      <SectionCard title="📊 Key Clinical Metrics">
        <div className="row">
          <KPI label="Epilepsy" value={`${kpis.epilepsy_pct}%`} color={COLOR} />
          <KPI label="EPC (hallmark)" value={`${kpis.epc_pct}%`} color="#c62828" />
          <KPI label="Hepatopathy" value={`${kpis.hepatopathy_pct}%`} color="#e65100" />
          <KPI label="Regression" value={`${kpis.regression_pct}%`} color="#4a148c" />
          <KPI label="Visual involvement" value={`${kpis.visual_pct}%`} color="#1565c0" />
          <KPI label="Acute liver failure" value={`${kpis.acute_liver_failure_pct}%`} color="#b71c1c" />
        </div>
        <div className="row mt-2 g-2">
          <div className="col-md-4">
            <div className="p-2 rounded text-center small fw-bold" style={{ background: '#b71c1c', color: 'white' }}>
              <div>VPA Status</div>
              <div className="fs-6">ABSOLUTE CONTRAINDICATION</div>
              <div style={{ fontSize: '0.7rem', fontWeight: 400 }}>lethal hepatotoxicity — DO NOT USE</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-2 rounded text-center small" style={{ background: '#fff8e1', border: '1px solid #f57f17' }}>
              <div className="fw-bold" style={{ color: '#e65100' }}>mtDNA Depletion</div>
              <div className="fs-6">{kpis.mtdna_depletion}</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-2 rounded text-center small" style={{ background: '#e3f2fd', border: '1px solid #1565c0' }}>
              <div className="fw-bold" style={{ color: '#1565c0' }}>Founder Variants</div>
              <div className="fs-6" style={{ fontSize: '0.8rem' }}>{kpis.founder_variant}</div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Clinical Highlights */}
      <SectionCard title="⚡ Clinical Highlights">
        {highlights.map((h, i) => (
          <Alert key={i} variant={
            h.includes('ABSOLUTE') || h.includes('CONTRAINDICATION') || h.includes('lethal') ? 'danger' :
            h.includes('CARDINAL') || h.includes('INTRACTABLE') || h.includes('EPC') || h.includes('MANDATORY') ? 'warning' :
            h.includes('preferred') || h.includes('Level A') ? 'success' : 'info'
          } text={h} />
        ))}
      </SectionCard>

      {/* Contraindications */}
      <SectionCard title="🚫 Contraindications & Cautions">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr>
              <th>Drug / Intervention</th><th>Level</th><th>Reason</th>
            </tr></thead>
            <tbody>
              {cis.map((c, i) => (
                <tr key={i} style={{ background: c.level.includes('ABSOLUTE') ? '#ffcdd2' : c.level.includes('AVOID') || c.level.includes('CAUTION') ? '#fff8e1' : 'transparent' }}>
                  <td><strong>{c.drug}</strong></td>
                  <td><span className={`badge ${c.level.includes('ABSOLUTE') ? 'bg-danger' : c.level.includes('AVOID') || c.level.includes('CAUTION') ? 'bg-warning text-dark' : 'bg-info text-dark'}`}>{c.level.slice(0, 40)}</span></td>
                  <td className="small">{c.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Thresholds */}
      <SectionCard title="🎯 Clinical Decision Thresholds">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Marker</th><th>Threshold</th><th>Interpretation</th></tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i} style={{ background: i < 2 ? LIGHT : 'transparent' }}>
                  <td><strong>{t.marker}</strong></td>
                  <td><code>{t.cutoff}</code></td>
                  <td>{t.interpretation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* DDx Table */}
      <SectionCard title="🔍 Differential Diagnosis">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Disease</th><th>Shared with POLG/Alpers</th><th>Key Distinguishing Features</th></tr></thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i}>
                  <td><strong>{d.disease}</strong></td>
                  <td>{d.shared}</td>
                  <td>{d.distinguishing}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const groups = data.phenotype_groups || [];
  const variants = data.variant_distribution || [];
  const bm = data.biomarker_summary || {};

  return (
    <div>
      <SectionCard title="👥 Phenotype Groups (n=40)">
        <div className="row">
          {groups.map((g, i) => (
            <div key={i} className="col-md-4 mb-3">
              <div className="card h-100 shadow-sm">
                <div className="card-body text-center">
                  <div className="fw-bold fs-4" style={{ color: COLOR }}>{g.n}</div>
                  <div className="text-muted small">{g.group}</div>
                  <div className="text-muted small">{g.pct}% of cohort</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🧬 Variant Distribution (biallelic AR)">
        {variants.map((v, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: i < 2 ? LIGHT : '#fff' }}>
            <div className="d-flex justify-content-between mb-1">
              <span className="small fw-bold">{v.variant}</span>
              <span className="small text-muted">{v.pct}% ({v.n_alleles} alleles)</span>
            </div>
            <div className="progress mb-1" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${v.pct}%`, backgroundColor: i === 0 ? COLOR : i === 1 ? '#c62828' : '#888' }} />
            </div>
            <div className="text-muted" style={{ fontSize: '0.75rem' }}>{v.effect}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔬 Key Biomarkers">
        <div className="row g-3 small">
          {[
            { label: 'Epilepsy', pct: bm.epilepsy_pct, color: COLOR, desc: 'CARDINAL, INTRACTABLE; occipital onset' },
            { label: 'EPC (hallmark)', pct: bm.epc_pct, color: '#c62828', desc: 'Epilepsia partialis continua; focal motor; continuous >1h' },
            { label: 'Status Epilepticus', pct: bm.status_epilepticus_pct, color: '#d32f2f', desc: 'Refractory SE; precipitates rapid regression' },
            { label: 'Hepatopathy', pct: bm.hepatopathy_pct, color: '#e65100', desc: `Transaminase elevation; liver failure ${bm.acute_liver_failure_pct}%` },
            { label: 'Psychomotor regression', pct: bm.regression_pct, color: '#4a148c', desc: `Language loss ${bm.language_loss_pct}%; ambulation loss ${bm.ambulation_loss_pct}%` },
            { label: 'Visual involvement', pct: bm.visual_pct, color: '#1565c0', desc: `Occipital cortical; cortical blindness ${bm.cortical_blindness_pct}%` },
          ].map((b, i) => (
            <div key={i} className="col-md-4 col-6">
              <div className="card shadow-sm text-center">
                <div className="card-body py-2">
                  <div className="fw-bold fs-5" style={{ color: b.color }}>{b.pct}%</div>
                  <div className="fw-semibold small">{b.label}</div>
                  <div className="text-muted" style={{ fontSize: '0.7rem' }}>{b.desc}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function EpilepsyHepatopathyTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const bm = data.biomarker_summary || {};
  const seizures = data.seizure_profile || [];
  const hepatic = data.hepatic_outcomes || [];

  return (
    <div>
      <SectionCard title="⚡ Epilepsy Profile — EPC + Status + Occipital Seizures" borderColor="#c62828">
        <Alert variant="danger" text="VPA IS ABSOLUTELY CONTRAINDICATED — never use valproate in POLG/Alpers for ANY seizure type, duration, or severity. Use LEV (IV loading 20-40 mg/kg) + midazolam + lacosamide + phenobarbitone as SE protocol." />
        <Alert variant="warning" text={`EPC (Epilepsia Partialis Continua) = ${bm.epc_pct}% — HALLMARK seizure of Alpers; continuous focal motor jerk >1h; highly resistant; occurs while child is conscious; correlates with occipital cortical neuronal loss.`} />
        <div className="table-responsive mt-2">
          <table className="table table-sm small">
            <thead><tr style={{ background: LIGHT }}>
              <th>Seizure Type</th><th>N</th><th>%</th><th>Clinical Notes</th>
            </tr></thead>
            <tbody>
              {seizures.map((s, i) => (
                <tr key={i} style={{ background: i === 0 ? '#ffcdd2' : 'transparent' }}>
                  <td><strong>{s.type}</strong></td>
                  <td>{s.n}</td>
                  <td><span className={`badge ${s.pct >= 60 ? 'bg-danger' : s.pct >= 40 ? 'bg-warning text-dark' : 'bg-info text-dark'}`}>{s.pct}%</span></td>
                  <td className="text-muted">{s.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Alert variant="info" text="SE PROTOCOL WITHOUT VPA: Stage 1: buccal midazolam 0.3 mg/kg; Stage 2: IV LEV 20-40 mg/kg + IV midazolam infusion; Stage 3: IV lacosamide 6-8 mg/kg; Stage 4: IV phenobarbitone 15-20 mg/kg; Stage 5: GA thiopentone (NOT propofol — PRIS). NEVER fosphenytoin high-dose." />
      </SectionCard>

      <SectionCard title="🫀 Hepatopathy — VPA Hepatotoxicity and Liver Outcomes" borderColor="#e65100">
        <Alert variant="danger" text={`ACUTE LIVER FAILURE in ${bm.acute_liver_failure_pct}% — VPA is the precipitant in ${bm.vpa_exposure_in_hepatic_failure_pct}% of hepatic failure cases. Mortality 80% once liver failure established without transplant. Liver transplant does NOT cure neurological disease.`} />
        <div className="table-responsive mt-2">
          <table className="table table-sm small">
            <thead><tr style={{ background: '#fff3e0' }}>
              <th>Hepatic Outcome</th><th>N</th><th>%</th><th>Clinical Notes</th>
            </tr></thead>
            <tbody>
              {hepatic.map((h, i) => (
                <tr key={i} style={{ background: i === 1 ? '#ffcdd2' : 'transparent' }}>
                  <td><strong>{h.outcome}</strong></td>
                  <td>{h.n}</td>
                  <td><span className={`badge ${h.pct >= 70 ? 'bg-danger' : h.pct >= 30 ? 'bg-warning text-dark' : 'bg-info text-dark'}`}>{h.pct}%</span></td>
                  <td className="text-muted">{h.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Alert variant="warning" text="LFT MONITORING: Quarterly minimum in all POLG patients. Stop ALL hepatotoxic drugs at ALT/AST >3× ULN. Weekly LFTs at 3× ULN. Liver team at 10× ULN or rising bilirubin. Coagulation + ammonia in acute decompensation." />
        <Alert variant="info" text="LIVER TRANSPLANT: May prevent liver-failure death but does NOT cure neurological disease — brain mtDNA depletion continues. Only consider in patients with severe but not terminal liver failure + neurological function still present + family/team consensus. Most centres do NOT offer OLT in late-stage AHS." />
      </SectionCard>

      <SectionCard title="📊 Disease Outcomes">
        <div className="row g-3 small">
          {Object.entries(data.outcomes || {}).map(([k, v], i) => (
            <div key={i} className="col-md-4 col-6">
              <div className="card text-center shadow-sm">
                <div className="card-body py-2">
                  <div className="fw-bold" style={{ color: k.includes('vpa') || k.includes('failure') ? '#c62828' : COLOR }}>
                    {typeof v === 'number' && k.includes('pct') ? `${v}%` :
                     typeof v === 'number' && k.includes('months') ? `${v} mo` :
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
      <Alert variant="danger" text="⛔ ABSOLUTE RULE: VPA (VALPROATE) IS PERMANENTLY CONTRAINDICATED IN POLG/ALPERS. Document this in every medical record, allergy system, and emergency letter. No exceptions. No safe dose." />

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
        <Alert variant="danger" text="VPA (Valproate): ABSOLUTE CONTRAINDICATION — lethal hepatotoxicity through 3 synergistic mechanisms: (1) direct POLG inhibition → complete mtDNA depletion; (2) CoA sequestration → fatty acid oxidation collapse; (3) VPA epoxide → direct hepatocyte necrosis. Published evidence: 65-70% of POLG liver failure had VPA exposure. Latency 3wk–9mo. Irreversible. NEVER USE." />
        <Alert variant="success" text="LEV (levetiracetam): PREFERRED first-line AED — no hepatic metabolism; no mito toxicity; no ammonia effect; no cytochrome P450 induction; IV loading 20-40 mg/kg for SE; oral 30-50 mg/kg/day divided; renal excretion." />
        <Alert variant="success" text="Buccal midazolam: LEVEL A — home rescue for all families; 0.3 mg/kg (max 10 mg); seizures >5min; families trained; all AHS families must have home supply with written protocol." />
        <Alert variant="success" text="IV Dextrose: LEVEL A — mandatory for any NPO >4h; GIR 6-8 mg/kg/min of 10% dextrose; prevents catabolism → lactic acidosis; all families have sick-day action card." />
        <Alert variant="warning" text="Propofol: AVOID — PRIS risk in any mitochondrial disease. Use ketamine + sevoflurane. Alert anaesthesia team at every procedural encounter. Document POLG diagnosis in pre-anaesthetic assessment." />
        <Alert variant="warning" text="IV Fosphenytoin (high-dose): CAUTION — supratherapeutic doses inhibit Complex I; avoid as SE rescue. Use midazolam → IV LEV → IV lacosamide → IV phenobarbitone → thiopentone GA instead." />
        <Alert variant="info" text="CoQ10 / Riboflavin: empirical (Level D) — no controlled evidence in POLG; generally safe; does not alter disease course. Focus on evidence-based: LEV, midazolam, dextrose protocol, NG/PEG feeds, palliative care." />
        <Alert variant="info" text="Deoxynucleoside supplementation: investigational — compassionate use in European centres; replenishes dNTP pools; no RCT data; families must understand experimental nature." />
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const defs = data.definitions || [];

  return (
    <div>
      <SectionCard title="📖 POLG Alpers-Huttenlocher — Definitions">
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

export default function POLGPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/polg/overview`).then(r => r.json()),
      fetch(`${API}/api/polg/breakdown`).then(r => r.json()),
      fetch(`${API}/api/polg/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#f8f9fa' }}>
      {/* Header */}
      <div style={{ background: COLOR, color: 'white', padding: '16px 24px' }}>
        <h4 className="mb-1 fw-bold">⛔ POLG — Alpers-Huttenlocher Syndrome</h4>
        <div className="small opacity-75">
          mtDNA Depletion Syndrome · 15q25.1 · OMIM 203700 · VPA ABSOLUTE CI — LETHAL HEPATOTOXICITY
        </div>
        <div className="small opacity-75 mt-1">
          POLG (1240aa) · DNA Polymerase Gamma · Exonuclease-Spacer-Polymerase · EPC 60% · Hepatopathy 80% · Naviaux 2004 AJHG
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
            {tab === 2 && <EpilepsyHepatopathyTab data={breakdown} />}
            {tab === 3 && <TreatmentsTab data={breakdown} />}
            {tab === 4 && <DefinitionsTab data={definitions} />}
          </>
        )}
      </div>
    </div>
  );
}
