'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Variants', 'Cataracts & Neutropenia', 'Treatments', 'Definitions'];
const COLOR = '#4527a0';   // deep violet — CLPB/MGCA7 (protein disaggregase, unique cataracts)
const LIGHT = '#ede7f6';

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
          <KPI label="Cataracts" value={`${kpis.cataracts_pct}%`} color={COLOR} />
          <KPI label="3-MGA-uria" value={`${kpis.three_mga_pct}%`} color="#00695c" />
          <KPI label="Neutropenia" value={`${kpis.neutropenia_pct}%`} color="#1565c0" />
          <KPI label="Neurological" value={`${kpis.neurological_pct}%`} color="#6a1b9a" />
          <KPI label="DCM" value={`${kpis.dcm_pct}%`} color="#2e7d32" />
          <KPI label="Hyperammonemia" value={`${kpis.hyperammonemia_pct}%`} color="#2e7d32" />
        </div>
        <div className="row mt-2">
          <div className="col-md-4">
            <div className="p-2 rounded text-center small" style={{ background: '#fff8e1', border: '1px solid #f57f17' }}>
              <div className="fw-bold" style={{ color: '#e65100' }}>VPA Status</div>
              <div className="fs-6" style={{ color: '#e65100' }}>{kpis.vpa_risk}</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-2 rounded text-center small" style={{ background: '#e8f5e9', border: '1px solid #2e7d32' }}>
              <div className="fw-bold" style={{ color: '#2e7d32' }}>C4-DC Acylcarnitine</div>
              <div className="fs-6">{kpis.c4dc_elevated}</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-2 rounded text-center small" style={{ background: '#e8f5e9', border: '1px solid #2e7d32' }}>
              <div className="fw-bold" style={{ color: '#2e7d32' }}>DCM Present?</div>
              <div className="fs-6">NO DCM — KEY DDx from TAZ/TMEM70</div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Clinical Highlights */}
      <SectionCard title="⚡ Clinical Highlights">
        {highlights.map((h, i) => (
          <Alert key={i} variant={
            h.includes('ABSOLUTE') || h.includes('CONTRAINDICATED') || h.includes('MANDATORY') ? 'danger' :
            h.includes('PATHOGNOMONIC') || h.includes('NO DCM') || h.includes('NO SNHL') || h.includes('NO hyper') ? 'warning' :
            h.includes('normal') || h.includes('Normal') || h.includes('Level A') ? 'success' : 'info'
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
                <tr key={i} style={{ background: c.level.includes('ABSOLUTE') || c.level.includes('CONTRAINDICATED') ? '#ffebee' : c.level.includes('AVOID') || c.level.includes('CAUTION') ? '#fff8e1' : 'transparent' }}>
                  <td><strong>{c.drug}</strong></td>
                  <td><span className={`badge ${c.level.includes('ABSOLUTE') || c.level.includes('CONTRAINDICATED') ? 'bg-danger' : c.level.includes('AVOID') || c.level.includes('CAUTION') ? 'bg-warning text-dark' : 'bg-info text-dark'}`}>{c.level}</span></td>
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
      <SectionCard title="🔍 Differential Diagnosis — 3-MGA-uria & Cataracts">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Disease</th><th>Shared with CLPB</th><th>Key Distinguishing Features</th></tr></thead>
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
              <div className="progress-bar" style={{ width: `${v.pct}%`, backgroundColor: i === 0 ? COLOR : i === 1 ? '#6a1b9a' : '#888' }} />
            </div>
            <div className="text-muted" style={{ fontSize: '0.75rem' }}>{v.effect}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔬 Key Biomarkers">
        <div className="row g-3 small">
          {[
            { label: '3-MGA-uria', pct: bm.three_mga_pct, color: COLOR, desc: `Mean ${bm.three_mga_mean_mmol_cr} mmol/mol Cr (range ${bm.three_mga_range_mmol_cr})` },
            { label: 'Cataracts', pct: bm.cataracts_pct, color: '#0d47a1', desc: 'Bilateral infantile; posterior cortical or nuclear; PATHOGNOMONIC for 3-MGA VII' },
            { label: 'Neutropenia', pct: bm.neutropenia_pct, color: '#1565c0', desc: `Severe (<0.5) ${bm.severe_neutropenia_pct}%; G-CSF responsive` },
            { label: 'Movement disorder', pct: bm.movement_disorder_pct, color: '#6a1b9a', desc: 'Ataxia + dystonia; correlates with NBD1/NBD2 genotype' },
            { label: 'Intellectual disability (any)', pct: bm.id_mild_moderate_pct + bm.id_severe_pct, color: '#4e342e', desc: `Mild-moderate ${bm.id_mild_moderate_pct}%; severe ${bm.id_severe_pct}%; normal cognition ${bm.normal_cognition_pct}%` },
            { label: 'Seizures', pct: bm.seizures_pct, color: '#c62828', desc: 'Focal or generalised; LEV preferred; not the dominant phenotype' },
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

      <SectionCard title="🧠 Neurological Features Detail">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Feature</th><th>N</th><th>%</th><th>Notes</th></tr></thead>
            <tbody>
              {(data.neurological_features || []).map((nf, i) => (
                <tr key={i}>
                  <td><strong>{nf.feature}</strong></td>
                  <td>{nf.n}</td>
                  <td><span className={`badge ${nf.pct >= 50 ? 'bg-warning text-dark' : nf.pct >= 25 ? 'bg-info text-dark' : 'bg-secondary'}`}>{nf.pct}%</span></td>
                  <td className="text-muted">{nf.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function CataractsNeutropeniaTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const bm = data.biomarker_summary || {};
  const neut = data.neutropenia_profile || [];
  const outcomes = data.outcomes || {};

  return (
    <div>
      <SectionCard title="👁️ Cataracts — PATHOGNOMONIC for 3-MGA Type VII" borderColor="#0d47a1">
        <Alert variant="warning" text="CATARACTS in any infant with elevated 3-MGA on urine organic acids = CLPB until proven otherwise. No other 3-MGA disease (TAZ, TMEM70, SERAC1, DNAJC19, OPA3, AUH) has cataracts as a feature." />
        <div className="row g-3 mt-1">
          {[
            { label: 'Cataracts prevalence', value: `${bm.cataracts_pct}%`, color: '#0d47a1', desc: 'Bilateral; posterior cortical or nuclear; infantile onset (birth to 12 months)' },
            { label: 'Surgery before 12 weeks', value: 'Level A', color: '#2e7d32', desc: 'Visually significant opacity → operate early; prevents irreversible deprivation amblyopia' },
            { label: 'Normal vision post-surgery', value: `${outcomes.normal_vision_post_surgery_pct}%`, color: '#00695c', desc: 'With early surgery + aggressive visual rehab (aphakic glasses / CL + patching)' },
          ].map((b, i) => (
            <div key={i} className="col-md-4">
              <div className="card shadow-sm text-center h-100">
                <div className="card-body">
                  <div className="fw-bold fs-4" style={{ color: b.color }}>{b.value}</div>
                  <div className="fw-bold small">{b.label}</div>
                  <div className="text-muted" style={{ fontSize: '0.72rem' }}>{b.desc}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-3">
          <Alert variant="info" text="MECHANISM: Lens crystallins (αA, αB, βγ) are constitutively aggregation-prone. CLPB disaggregase activity maintains crystallin solubility in lens epithelial cells. CLPB LOF → crystallin aggregates → posterior cortical or nuclear opacification → infantile cataracts. UNIQUE mechanism — not seen in other 3-MGA diseases." />
          <Alert variant="success" text="OPHTHALMOLOGY PROTOCOL: slit-lamp examination at diagnosis + every 6 months in first 2 years. Bilateral cataract surgery if visually significant opacity (visual axis involvement or nystagmus). Aphakic contact lenses / glasses post-surgery. Occlusion therapy if asymmetric amblyopia." />
        </div>
      </SectionCard>

      <SectionCard title="🩸 Neutropenia — Mechanism, Types, Management" borderColor="#1565c0">
        <Alert variant="info" text="MECHANISM: CLPB LOF → protein aggregate accumulation during promyelocyte mitosis → maturation arrest at promyelocyte/myelocyte transition → cyclic or chronic neutropenia. G-CSF (filgrastim) reverses arrest in most cases. NOT cardiolipin-based (different from Barth/TAZ neutropenia)." />
        <div className="table-responsive mt-2">
          <table className="table table-sm small">
            <thead>
              <tr style={{ background: LIGHT }}>
                <th>Neutropenia Type</th><th>N</th><th>%</th><th>G-CSF Response</th>
              </tr>
            </thead>
            <tbody>
              {neut.map((np, i) => (
                <tr key={i} style={{ background: i === 0 ? '#ffebee' : 'transparent' }}>
                  <td><strong>{np.type}</strong></td>
                  <td>{np.n}</td>
                  <td><span className={`badge ${np.pct >= 30 ? 'bg-warning text-dark' : 'bg-info text-dark'}`}>{np.pct}%</span></td>
                  <td className="text-muted small">{np.g_csf_response}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Alert variant="warning" text="G-CSF PROTOCOL: filgrastim 5-10 µg/kg/day SC (start 5, titrate to ANC >1.0 × 10⁹/L). Cyclic neutropenia: 3 consecutive daily doses at nadir onset. Chronic: daily or alternate days. Monitor CBC weekly initially, then monthly once stable. Stop if ANC >5 × 10⁹/L to avoid splenomegaly." />
        <Alert variant="danger" text="INFECTION MANAGEMENT: ANC <0.5 = fever is emergency. IV pip-tazo or cefepime empirically; blood cultures × 2 before antibiotics; add antifungal if ANC <0.5 >5 days. Live vaccines contraindicated when ANC <0.5 — time immunisations with G-CSF nadir prevention." />
        <Alert variant="info" text="CLPB vs BARTH NEUTROPENIA: Both show promyelocyte arrest on bone marrow biopsy — bone marrow alone cannot distinguish. CLPB: normal C4-DC + cataracts + no DCM; Barth: C4-DC elevated + DCM + X-linked males. Molecular confirmation (WES) is essential." />
      </SectionCard>

      <SectionCard title="📊 Clinical Outcomes">
        <div className="row g-3 small">
          {Object.entries(outcomes).map(([k, v], i) => (
            <div key={i} className="col-md-4 col-6">
              <div className="card text-center shadow-sm">
                <div className="card-body py-2">
                  <div className="fw-bold" style={{ color: COLOR }}>
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
        <Alert variant="success" text="Cataract surgery: Level A — early bilateral surgery prevents deprivation amblyopia; operate before 8-12 weeks if visually significant; ophthalmology team must be co-managing from diagnosis." />
        <Alert variant="success" text="G-CSF (filgrastim): Level B — effective in ~80% of neutropenic CLPB patients; 5-10 µg/kg/day SC; target ANC >1.0 × 10⁹/L; prophylactic antibiotics during nadir." />
        <Alert variant="success" text="LEV (levetiracetam): Preferred AED — no hepatic metabolism, no mito toxicity, no ammonia effect, no cytochrome P450 induction; 20-40 mg/kg/day divided doses; renal excretion." />
        <Alert variant="warning" text="VPA (Valproate): MODERATE CAUTION (not absolute CI unlike TMEM70/POLG). Rule out POLG mutations first (absolute CI in POLG). If VPA used: monthly LFTs + ammonia + drug levels; stop immediately if ALT/AST >2× ULN." />
        <Alert variant="danger" text="Propofol: AVOID — PRIS risk in any mitochondrial disease. Use ketamine + sevoflurane for anaesthesia. Document CLPB diagnosis on every anaesthetic record." />
        <Alert variant="danger" text="Leucine restriction diet: NOT INDICATED — do not apply 3-MGA Type I (AUH) management to CLPB. CLPB 3-MGA is secondary overflow, not primary leucine catabolism enzyme defect; restriction is nutritionally harmful and mechanistically irrelevant." />
        <Alert variant="info" text="CoQ10 / Riboflavin: empirical (Level D) — no controlled evidence in CLPB; generally safe; some centres use for mitochondrial support. Do not oversell to families; focus on evidence-based interventions (cataract surgery, G-CSF, LEV, PT/OT)." />
        <Alert variant="info" text="Sick-day protocol: maintain oral glucose during illness; hospital threshold: vomiting >2 episodes + fever >38°C; IV dextrose if unable to feed; CBC on admission (illness exacerbates neutropenia); severity less than TMEM70 but neutropenic sepsis is the dominant sick-day risk." />
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const defs = data.definitions || [];

  return (
    <div>
      <SectionCard title="📖 CLPB MGCA7 — Definitions">
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

export default function CLPBPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/clpb/overview`).then(r => r.json()),
      fetch(`${API}/api/clpb/breakdown`).then(r => r.json()),
      fetch(`${API}/api/clpb/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#f8f9fa' }}>
      {/* Header */}
      <div style={{ background: COLOR, color: 'white', padding: '16px 24px' }}>
        <h4 className="mb-1 fw-bold">👁️ CLPB — 3-MGA-uria Type VII / MGCA7</h4>
        <div className="small opacity-75">
          3-Methylglutaconic Aciduria with Cataracts, Neurologic Involvement &amp; Neutropenia · 11q13.1 · OMIM 616228
        </div>
        <div className="small opacity-75 mt-1">
          CLPB (707aa) · Mitochondrial AAA+ Disaggregase · MTS-BAP-MD-NBD1-NBD2 · Cataracts PATHOGNOMONIC · No DCM · No Hyperammonemia · Wortmann 2015 AJHG
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
            {tab === 2 && <CataractsNeutropeniaTab data={breakdown} />}
            {tab === 3 && <TreatmentsTab data={breakdown} />}
            {tab === 4 && <DefinitionsTab data={definitions} />}
          </>
        )}
      </div>
    </div>
  );
}
