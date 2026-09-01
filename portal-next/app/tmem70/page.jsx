'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Metabolic', 'Neonatal Crisis', 'Treatments', 'Definitions'];
const COLOR = '#1a237e';   // deep indigo — TMEM70/Complex V (ATP, mitochondrial energy failure)
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

function Bar({ label, value, max, color = COLOR }) {
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

function Alert({ variant, text }) {
  const bg = variant === 'danger' ? '#ffebee' : variant === 'warning' ? '#fff8e1' : variant === 'success' ? '#e8f5e9' : LIGHT;
  const border = variant === 'danger' ? '#c62828' : variant === 'warning' ? '#f57f17' : variant === 'success' ? '#2e7d32' : COLOR;
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
          <KPI label="Lactic Acidosis" value={`${kpis.neonatal_lactic_acidosis_pct}%`} color={COLOR} />
          <KPI label="Hyperammonemia" value={`${kpis.hyperammonemia_pct}%`} color="#6a1b9a" />
          <KPI label="DCM" value={`${kpis.dcm_pct}%`} color="#c62828" />
          <KPI label="Hypotonia" value={`${kpis.hypotonia_pct}%`} color="#1565c0" />
          <KPI label="PAH" value={`${kpis.pah_pct}%`} color="#00838f" />
          <KPI label="Neonatal Mortality (untreated)" value={`${kpis.neonatal_mortality_untreated_pct}%`} color="#b71c1c" />
        </div>
        <div className="row mt-2">
          <div className="col-md-4">
            <div className="p-2 rounded text-center small" style={{ background: '#ffebee', border: '1px solid #c62828' }}>
              <div className="fw-bold text-danger">VPA Status</div>
              <div className="fs-6 text-danger">{kpis.vpa_ci}</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-2 rounded text-center small" style={{ background: '#ffebee', border: '1px solid #c62828' }}>
              <div className="fw-bold text-danger">Ketogenic Diet</div>
              <div className="fs-6 text-danger">CONTRAINDICATED</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-2 rounded text-center small" style={{ background: '#e8f5e9', border: '1px solid #2e7d32' }}>
              <div className="fw-bold" style={{ color: '#2e7d32' }}>C4-DC Acylcarnitine</div>
              <div className="fs-6">{kpis.c4dc_elevated} </div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Clinical Highlights */}
      <SectionCard title="⚡ Clinical Highlights">
        {highlights.map((h, i) => (
          <Alert key={i} variant={
            h.includes('ABSOLUTE') || h.includes('CONTRAINDICATED') || h.includes('MANDATORY') ? 'danger' :
            h.includes('PATHOGNOMONIC') ? 'warning' :
            h.includes('normal') || h.includes('Normal') ? 'success' : 'info'
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
                <tr key={i} style={{ background: c.level.includes('ABSOLUTE') || c.level.includes('CONTRAINDICATED') ? '#ffebee' : c.level.includes('AVOID') ? '#fff8e1' : 'transparent' }}>
                  <td><strong>{c.drug}</strong></td>
                  <td><span className={`badge ${c.level.includes('ABSOLUTE') || c.level.includes('CONTRAINDICATED') ? 'bg-danger' : c.level.includes('AVOID') ? 'bg-warning text-dark' : 'bg-secondary'}`}>{c.level}</span></td>
                  <td>{c.reason}</td>
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
                <tr key={i} style={{ background: i < 3 ? '#fff3e0' : 'transparent' }}>
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
      <SectionCard title="🔍 Differential Diagnosis — 3-MGA-uria & Neonatal Hyperammonemia">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Disease</th><th>Shared with TMEM70</th><th>Key Distinguishing Features</th></tr></thead>
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
  const metabolic = data.metabolic_profile_by_age || [];

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
          <div key={i} className="mb-3 p-2 rounded" style={{ background: i === 0 ? '#e8eaf6' : '#fff' }}>
            <div className="d-flex justify-content-between mb-1">
              <span className="small fw-bold">{v.variant}</span>
              <span className="small text-muted">{v.pct}% ({v.n_alleles} alleles)</span>
            </div>
            <div className="progress mb-1" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${v.pct}%`, backgroundColor: i === 0 ? '#6a1b9a' : COLOR }} />
            </div>
            <div className="text-muted" style={{ fontSize: '0.75rem' }}>{v.effect}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔬 Key Biomarkers">
        <div className="row g-3 small">
          {[
            { label: 'Lactic acidosis', pct: bm.lactic_acidosis_pct, color: '#c62828', desc: `Peak lactate ${bm.neonatal_lactate_peak_mmol} mmol/L (range ${bm.neonatal_lactate_range_mmol})` },
            { label: 'Hyperammonemia', pct: bm.hyperammonemia_pct, color: '#6a1b9a', desc: `NH3 mean ${bm.nh3_mean_umol} µmol/L (range ${bm.nh3_range_umol})` },
            { label: 'DCM', pct: bm.dcm_pct, color: '#b71c1c', desc: 'Onset first week to 3 months; OXPHOS failure → cardiomyocyte ATP deficit' },
            { label: 'PAH', pct: bm.pah_pct, color: '#00838f', desc: 'Neonatal PAH; responds to iNO; 75% resolve in year 1' },
            { label: 'Normal acylcarnitine (no C4-DC)', pct: bm.normal_acylcarnitine_pct, color: '#2e7d32', desc: 'KEY DDx from TAZ/Barth — C4-DC absent here' },
            { label: 'Intellectual disability', pct: bm.id_pct, color: '#5d4037', desc: 'Mild-moderate; neonatal insult + chronic ATP deficit' },
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

      <SectionCard title="📈 Metabolic Profile Over Time">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr>
              <th>Age Group</th><th>Lactate (mmol/L)</th><th>NH3 (µmol/L)</th><th>pH</th><th>Notes</th>
            </tr></thead>
            <tbody>
              {metabolic.map((m, i) => (
                <tr key={i} style={{ background: i === 0 ? '#ffebee' : i === 1 ? '#fff8e1' : 'transparent' }}>
                  <td><strong>{m.age_group}</strong></td>
                  <td>
                    <span className={`badge ${m.lactate_mmol > 10 ? 'bg-danger' : m.lactate_mmol > 5 ? 'bg-warning text-dark' : 'bg-success'}`}>{m.lactate_mmol}</span>
                  </td>
                  <td>
                    <span className={`badge ${m.nh3_umol > 150 ? 'bg-danger' : m.nh3_umol > 80 ? 'bg-warning text-dark' : 'bg-success'}`}>{m.nh3_umol}</span>
                  </td>
                  <td>
                    <span className={`badge ${m.ph_mean < 7.1 ? 'bg-danger' : m.ph_mean < 7.3 ? 'bg-warning text-dark' : 'bg-success'}`}>{m.ph_mean}</span>
                  </td>
                  <td className="small text-muted">{m.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function NeonatalCrisisTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading crisis data...</div>;
  const bm = data.biomarker_summary || {};
  const outcomes = data.outcomes || {};

  return (
    <div>
      <SectionCard title="🚨 Neonatal Crisis Protocol — Step-by-Step" borderColor="#c62828">
        <Alert variant="danger" text="STEP 1 — STOP CATABOLISM: IV 10% dextrose; GIR 8-10 mg/kg/min; NPO; glucose monitoring hourly. Any catabolism → lactic acidosis crisis in Complex V deficiency." />
        <Alert variant="danger" text="STEP 2 — CORRECT ACIDOSIS: Sodium bicarbonate IV bolus (1-2 mEq/kg); repeat if pH <7.2; continuous bicarbonate infusion if pH <7.1 recurrent." />
        <Alert variant="danger" text="STEP 3 — AMMONIA: if NH3 >200 µmol/L → sodium benzoate 250 mg/kg IV load + sodium phenylbutyrate 250 mg/kg IV load; urgent metabolic specialist contact. If NH3 >500 µmol/L → dialysis (CVVHD)." />
        <Alert variant="warning" text="STEP 4 — DCM SUPPORT: dopamine/dobutamine if cardiogenic shock; echocardiogram urgently; ACE-I + BB once haemodynamically stable (Level A for DCM)." />
        <Alert variant="warning" text="STEP 5 — RESPIRATORY: intubation if pH <7.1 or respiratory failure; iNO 20 ppm if PAH on echo; ECMO if refractory PAH + cardiogenic shock (bridge to stabilisation)." />
        <Alert variant="info" text="NEVER FAST: maintain GIR ≥6 mg/kg/min during ALL procedures, illness, NPO periods. Fasting = lactic acidosis crisis. Pre-procedure glucose mandatory." />
      </SectionCard>

      <SectionCard title="⚡ The Pathognomonic Triad" borderColor="#6a1b9a">
        <div className="row g-3">
          {[
            { label: '3-MGA-uria (Type VI)', pct: 100, color: '#1a237e', desc: '50-300 mmol/mol Cr; secondary overflow from OXPHOS failure; urine organic acids always ordered' },
            { label: 'Lactic acidosis', pct: 100, color: '#c62828', desc: 'pH <7.1 + lactate >10 mmol/L; L/P ratio >20 confirms mitochondrial block (not cytoplasmic)' },
            { label: 'Hyperammonemia', pct: 90, color: '#6a1b9a', desc: 'NH3 50-500 µmol/L; secondary UCD failure (CPS1 ATP-dependent); differentiates TMEM70 from all other 3-MGA diseases' },
          ].map((b, i) => (
            <div key={i} className="col-md-4">
              <div className="card shadow-sm text-center h-100">
                <div className="card-body">
                  <div className="fw-bold fs-2" style={{ color: b.color }}>{b.pct}%</div>
                  <div className="fw-bold">{b.label}</div>
                  <div className="text-muted small mt-2">{b.desc}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-3">
          <Alert variant="warning" text="CRITICAL: 3-MGA + lactic acidosis + hyperammonemia together = TMEM70 until proven otherwise. Do NOT treat as urea cycle disorder alone (no protein restriction without glucose cover). Order lactate + ammonia + organic acids SIMULTANEOUSLY." />
        </div>
      </SectionCard>

      <SectionCard title="📊 Clinical Outcomes">
        <div className="row g-3 small">
          {Object.entries(outcomes).map(([k, v], i) => (
            <div key={i} className="col-md-4 col-6">
              <div className="card text-center shadow-sm">
                <div className="card-body py-2">
                  <div className="fw-bold" style={{ color: COLOR }}>
                    {typeof v === 'number' && k.includes('pct') ? `${v}%` :
                     typeof v === 'number' && k.includes('days') ? `${v} days` :
                     typeof v === 'number' && k.includes('age_diagnosis') ? `day ${v}` : v}
                  </div>
                  <div className="text-muted" style={{ fontSize: '0.72rem' }}>{k.replace(/_/g, ' ')}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🩺 Distinguishing TMEM70 from Urea Cycle Disorders" borderColor="#f57f17">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead>
              <tr style={{ background: '#fff8e1' }}>
                <th>Feature</th><th>TMEM70</th><th>OTC / CPS1 Deficiency</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['Lactic acidosis', 'YES (severe, L/P >20)', 'NO (mild or absent)'],
                ['3-methylglutaconic acid (urine)', 'ELEVATED (50-300 mmol/mol)', 'NORMAL'],
                ['Lactate:pyruvate ratio', '>20 (mitochondrial)', 'Normal (<15)'],
                ['Urine orotic acid', 'Normal', 'HIGH (OTC) / Normal (CPS1)'],
                ['Plasma amino acids', 'Glutamine ↑; Citrulline borderline low', 'Citrulline very low; Glutamine ↑↑'],
                ['DCM on echo', 'YES (80-90%)', 'NO'],
                ['Sex', 'Both sexes (AR)', 'Males primarily (OTC X-linked)'],
                ['Acylcarnitine', 'Normal', 'Normal'],
              ].map(([f, t, u], i) => (
                <tr key={i}>
                  <td><strong>{f}</strong></td>
                  <td style={{ color: t.includes('ELEVATED') || t.includes('YES') ? '#c62828' : t.includes('NORMAL') ? '#2e7d32' : 'inherit' }}>{t}</td>
                  <td>{u}</td>
                </tr>
              ))}
            </tbody>
          </table>
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
      <SectionCard title="💊 Treatment Distribution (n=40)">
        {treatments.map((t, i) => {
          const isMandatory = t.indication?.includes('MANDATORY') || t.indication?.includes('Level A');
          return (
            <div key={i} className="mb-3 p-2 rounded" style={{ background: isMandatory ? '#e8eaf6' : '#f9f9f9', border: isMandatory ? `1px solid ${COLOR}` : '1px solid #eee' }}>
              <div className="d-flex justify-content-between align-items-start mb-1">
                <div>
                  <span className="fw-bold small">{t.treatment}</span>
                  {isMandatory && <span className="badge ms-2" style={{ background: COLOR, fontSize: '0.65rem' }}>MANDATORY</span>}
                </div>
                <span className="small text-muted">{t.n}/{data.cohort} ({t.pct}%)</span>
              </div>
              <div className="progress mb-1" style={{ height: 8 }}>
                <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: isMandatory ? COLOR : '#888' }} />
              </div>
              <div className="text-muted" style={{ fontSize: '0.75rem' }}>{t.indication}</div>
            </div>
          );
        })}
      </SectionCard>

      <SectionCard title="🚦 Prescribing Decision Guide">
        <Alert variant="danger" text="VPA (Valproate): ABSOLUTE CONTRAINDICATION. Dual lethal mechanism: (1) Complex I inhibition + CoA sequestration in already-ATP-depleted state; (2) VPA independently raises ammonia → fatal hyperammonemic crisis on top of baseline elevated NH3 in TMEM70. NEVER use." />
        <Alert variant="danger" text="Ketogenic Diet: CONTRAINDICATED. Fat-based ATP via Complex V is impossible when Complex V is absent. KD worsens the energy deficit and can accelerate lactic acidosis. Completely opposite to PDH deficiency (where KD helps)." />
        <Alert variant="danger" text="Fasting / Prolonged NPO: HIGH RISK. Any catabolism triggers crisis. IV dextrose GIR ≥6 mg/kg/min mandatory during illness/procedures. Sick-day protocol card mandatory for families." />
        <Alert variant="warning" text="Ammonia scavengers: sodium benzoate + sodium phenylbutyrate are first-line for NH3 >200 µmol/L. Give IV load in neonatal crisis. Oral maintenance in chronic management. Dialysis if NH3 >500 µmol/L." />
        <Alert variant="warning" text="iNO (inhaled nitric oxide): 20 ppm for neonatal PAH. Wean over 1-2 weeks. Rebound PAH on abrupt withdrawal — taper sildenafil as bridge. Oral sildenafil maintenance if PAH persists beyond acute phase." />
        <Alert variant="success" text="LEV (levetiracetam): Preferred AED for seizures. Renal excretion; no mito toxicity; no ammonia effect. Monitor renal function (may be impaired in neonatal acidosis). Safe in TMEM70." />
        <Alert variant="info" text="Riboflavin / CoQ10: empirical only (Level D). No controlled evidence in Complex V deficiency. Generally safe; some centres use routinely. Do not oversell benefit to families." />
        <Alert variant="info" text="Propofol: AVOID (PRIS risk in mito disease). Use ketamine or inhalational agents. Alert anaesthesia team at every procedural encounter — include TMEM70 diagnosis on anaesthetic record." />
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const defs = data.definitions || [];

  return (
    <div>
      <SectionCard title="📖 TMEM70 Complex V Deficiency — Definitions">
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

export default function TMEM70Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/tmem70/overview`).then(r => r.json()),
      fetch(`${API}/api/tmem70/breakdown`).then(r => r.json()),
      fetch(`${API}/api/tmem70/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#f8f9fa' }}>
      {/* Header */}
      <div style={{ background: COLOR, color: 'white', padding: '16px 24px' }}>
        <h4 className="mb-1 fw-bold">⚡ TMEM70 — Complex V Deficiency / 3-MGA-uria Type VI</h4>
        <div className="small opacity-75">
          Mitochondrial Complex V Deficiency, Nuclear Type 2 · 3-MGA-uria Type VI · 8q11.23 · OMIM 614052
        </div>
        <div className="small opacity-75 mt-1">
          TMEM70 (260aa) · ATP Synthase c-ring Assembly Factor · IMM · Neonatal Lactic Acidosis + Hyperammonemia + DCM · Czech/Slovak Roma c.317-2A>G · Cizkova 2008
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
            {tab === 2 && <NeonatalCrisisTab data={breakdown} />}
            {tab === 3 && <TreatmentsTab data={breakdown} />}
            {tab === 4 && <DefinitionsTab data={definitions} />}
          </>
        )}
      </div>
    </div>
  );
}
