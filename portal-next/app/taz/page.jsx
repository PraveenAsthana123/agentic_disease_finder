'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Cardiac & Neutropenia', 'Treatments', 'Definitions'];
const COLOR = '#b71c1c';   // deep red — TAZ/Barth (heart disease, cardiolipin, DCM)
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
          <KPI label="DCM" value={`${kpis.dcm_pct}%`} color={COLOR} />
          <KPI label="Neutropenia" value={`${kpis.neutropenia_pct}%`} color="#e53935" />
          <KPI label="Skeletal Myopathy" value={`${kpis.myopathy_pct}%`} color="#6a1a1a" />
          <KPI label="Normal Cognition" value={`${kpis.normal_cognition_pct}%`} color="#2e7d32" />
          <KPI label="Heart Transplant" value={`${kpis.heart_transplant_pct}%`} color="#c62828" />
          <KPI label="C4-DC Elevated" value={`${kpis.c4dc_elevated_pct}%`} color="#880e4f" />
        </div>
        <div className="row mt-2">
          <div className="col-md-4">
            <div className="p-2 rounded text-center small" style={{ background: '#ffebee', border: `1px solid ${COLOR}` }}>
              <div className="fw-bold" style={{ color: COLOR }}>MLCL:CL Ratio</div>
              <div className="fs-6">{kpis.mlcl_cl_ratio_cutoff} = diagnostic</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-2 rounded text-center small" style={{ background: '#ffebee', border: '1px solid #c62828' }}>
              <div className="fw-bold text-danger">VPA Status</div>
              <div className="fs-6 text-danger">{kpis.vpa_ci}</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-2 rounded text-center small" style={{ background: '#e8f5e9', border: '1px solid #2e7d32' }}>
              <div className="fw-bold" style={{ color: '#2e7d32' }}>Inheritance</div>
              <div className="fs-6">X-linked recessive (males)</div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Clinical Highlights */}
      <SectionCard title="⚡ Clinical Highlights">
        {highlights.map((h, i) => (
          <Alert key={i} variant={
            h.includes('ABSOLUTE') || h.includes('MANDATORY') ? 'danger' :
            h.includes('NORMAL') || h.includes('normal') ? 'success' : 'info'
          } text={h} />
        ))}
      </SectionCard>

      {/* Contraindications */}
      <SectionCard title="🚫 Contraindications & Cautions">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr>
              <th>Drug</th><th>Level</th><th>Reason</th>
            </tr></thead>
            <tbody>
              {cis.map((c, i) => (
                <tr key={i} style={{ background: c.level.includes('ABSOLUTE') ? '#ffebee' : c.level.includes('AVOID') ? '#fff8e1' : 'transparent' }}>
                  <td><strong>{c.drug}</strong></td>
                  <td><span className={`badge ${c.level.includes('ABSOLUTE') ? 'bg-danger' : c.level.includes('AVOID') ? 'bg-warning text-dark' : 'bg-secondary'}`}>{c.level}</span></td>
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
                <tr key={i}>
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
      <SectionCard title="🔍 Differential Diagnosis vs Other 3-MGA-uria Diseases">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Disease</th><th>Shared with TAZ</th><th>Key Distinguishing Features</th></tr></thead>
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
  const outcomes = data.outcomes || {};

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

      <SectionCard title="🧬 Variant Distribution (hemizygous males, X-linked)">
        {variants.map((v, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#fff' }}>
            <div className="d-flex justify-content-between mb-1">
              <span className="small fw-bold">{v.variant}</span>
              <span className="small text-muted">{v.pct}% ({v.n_alleles} alleles)</span>
            </div>
            <div className="progress mb-1" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${v.pct}%`, backgroundColor: COLOR }} />
            </div>
            <div className="text-muted" style={{ fontSize: '0.75rem' }}>{v.effect}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔬 Biomarker Summary">
        <div className="row g-3 small">
          <div className="col-md-6">
            <div className="p-2 rounded" style={{ background: LIGHT }}>
              <strong>MLCL:CL ratio:</strong> mean {bm.mlcl_cl_ratio_mean} (range {bm.mlcl_cl_ratio_range})<br />
              {bm.mlcl_cl_above_05_pct}% above 0.5 diagnostic threshold
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-2 rounded" style={{ background: '#fff3e0' }}>
              <strong>C4-DC elevated:</strong> {bm.c4dc_elevated_pct}% — pathognomonic<br />
              <strong>C0 carnitine low:</strong> {bm.c0_carnitine_low_pct}%
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-2 rounded" style={{ background: '#e8f5e9' }}>
              <strong>SNHL:</strong> {bm.snhl_pct}% (ABSENT — DDx SERAC1)<br />
              <strong>Optic atrophy:</strong> {bm.optic_atrophy_pct}% (ABSENT — DDx OPA3)<br />
              <strong>Movement disorder:</strong> {bm.movement_disorder_pct}% (ABSENT)
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-2 rounded" style={{ background: '#fce4ec' }}>
              <strong>LVNC variant:</strong> {bm.lvnc_pct}% of patients<br />
              <strong>Neutropenia cyclic:</strong> {bm.neutropenia_cyclic_pct}%<br />
              <strong>Neutropenia chronic:</strong> {bm.neutropenia_chronic_pct}%
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="📈 Clinical Outcomes">
        <div className="row g-3 small">
          {Object.entries(outcomes).map(([k, v], i) => (
            <div key={i} className="col-md-4 col-6">
              <div className="card text-center shadow-sm">
                <div className="card-body py-2">
                  <div className="fw-bold" style={{ color: COLOR }}>{typeof v === 'number' && v > 1 ? `${v}${k.includes('pct') ? '%' : k.includes('months') ? ' mo' : k.includes('years') ? ' yr' : ''}` : v}</div>
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

function CardiacNeutropeniaTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading cardiac data...</div>;
  const cardiac = data.cardiac_outcomes_by_age || [];
  const bm = data.biomarker_summary || {};

  return (
    <div>
      <SectionCard title="❤️ Cardiac Progression by Age Group">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr>
              <th>Age Group</th><th>DCM Present %</th><th>Mean EF (%)</th><th>Notes</th>
            </tr></thead>
            <tbody>
              {cardiac.map((c, i) => (
                <tr key={i} style={{ background: c.mean_ef < 40 ? '#ffebee' : c.mean_ef < 50 ? '#fff8e1' : 'transparent' }}>
                  <td><strong>{c.age_group}</strong></td>
                  <td>
                    <div className="progress" style={{ height: 12, minWidth: 80 }}>
                      <div className="progress-bar" style={{ width: `${c.dcm_present_pct}%`, backgroundColor: COLOR }} />
                    </div>
                    <span className="small">{c.dcm_present_pct}%</span>
                  </td>
                  <td>
                    <span className={`badge ${c.mean_ef < 40 ? 'bg-danger' : c.mean_ef < 50 ? 'bg-warning text-dark' : 'bg-success'}`}>{c.mean_ef}%</span>
                  </td>
                  <td className="small text-muted">{c.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-2 small text-muted">
          EF target: &ge;55% on ACE-I + BB; transplant threshold EF &lt;25% refractory. LVNC variant: {bm.lvnc_pct}% of cohort.
        </div>
      </SectionCard>

      <SectionCard title="🩸 Neutropenia Profile">
        <div className="row g-3">
          {[
            { label: 'Cyclic neutropenia', pct: bm.neutropenia_cyclic_pct, color: '#c62828', desc: '21-day cycles; ANC nadir <0.5×10⁹/L; recurrent infections' },
            { label: 'Chronic neutropenia', pct: bm.neutropenia_chronic_pct, color: '#e53935', desc: 'Sustained ANC <1.0×10⁹/L; constant infection risk' },
            { label: 'Neutropenia absent', pct: bm.neutropenia_absent_pct, color: '#43a047', desc: 'Normal ANC throughout; mild phenotype' },
          ].map((n, i) => (
            <div key={i} className="col-md-4">
              <div className="card shadow-sm text-center">
                <div className="card-body">
                  <div className="fw-bold fs-3" style={{ color: n.color }}>{n.pct}%</div>
                  <div className="fw-bold small">{n.label}</div>
                  <div className="text-muted" style={{ fontSize: '0.72rem' }}>{n.desc}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-3">
          <Alert variant="danger" text="G-CSF (filgrastim) first-line for ANC <0.5×10⁹/L or recurrent severe infections — Level B. Monitor CBC 2× weekly during dose titration." />
          <Alert variant="warning" text="Cyclic neutropenia: 21-day cycle tracking allows prediction of nadir. Educate family: fever + neutropenia = emergency; immediate antibiotics." />
          <Alert variant="info" text="Antibiotic prophylaxis (TMP-SMX or azithromycin) consider in recurrent infections — Level C. IVIG investigational if G-CSF fails — Level D." />
        </div>
      </SectionCard>

      <SectionCard title="🔬 Key Biomarkers for Diagnosis">
        <div className="row g-3 small">
          {[
            { label: 'MLCL:CL ratio >0.5', pct: 100, color: '#880e4f', desc: 'Most specific marker; DBS/fibroblasts; normal <0.1' },
            { label: 'C4-DC elevated on NBS', pct: bm.c4dc_elevated_pct, color: '#e53935', desc: 'Pathognomonic; triggers reflex MLCL:CL + TAZ sequencing' },
            { label: 'C0 carnitine low', pct: bm.c0_carnitine_low_pct, color: '#f57f17', desc: 'Secondary depletion; supplement to C0 30-60 µmol/L' },
            { label: 'Normal cognition', pct: bm.normal_cognition_pct, color: '#2e7d32', desc: 'Unique among 3-MGA diseases; intelligence preserved' },
            { label: 'SNHL absent', pct: 100 - bm.snhl_pct, color: '#1565c0', desc: 'No hearing loss; KEY DDx from SERAC1 (SNHL 100%)' },
            { label: 'Optic atrophy absent', pct: 100 - bm.optic_atrophy_pct, color: '#6a1a1a', desc: 'No optic atrophy; KEY DDx from OPA3 (100%) + MECR (80%)' },
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

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading treatments...</div>;
  const treatments = data.treatment_distribution || [];

  return (
    <div>
      <SectionCard title="💊 Treatment Distribution (n=40)">
        {treatments.map((t, i) => {
          const isMandatory = t.indication?.includes('MANDATORY') || t.indication?.includes('Level A');
          const isCI = t.indication?.includes('ABSOLUTE CI');
          return (
            <div key={i} className="mb-3 p-2 rounded" style={{ background: isMandatory ? '#fce4ec' : '#f9f9f9', border: isMandatory ? `1px solid ${COLOR}` : '1px solid #eee' }}>
              <div className="d-flex justify-content-between align-items-start mb-1">
                <div>
                  <span className="fw-bold small">{t.treatment}</span>
                  {isMandatory && <span className="badge bg-danger ms-2" style={{ fontSize: '0.65rem' }}>MANDATORY</span>}
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
        <Alert variant="danger" text="VPA (Valproate): ABSOLUTE CONTRAINDICATION in Barth syndrome. Complex I inhibition + CoA sequestration → fatal lactic acidosis + hepatotoxicity in cardiolipin deficiency. Never use." />
        <Alert variant="danger" text="ACE Inhibitor + Beta-Blocker: MANDATORY from diagnosis. Level A evidence. Do NOT withhold; cardiac function deteriorates rapidly without RAAS/adrenergic blockade." />
        <Alert variant="warning" text="Heart Transplant threshold: EF <25% refractory to optimised ACE-I + BB. List early; post-Tx outcomes excellent (10yr survival ~80%). TAZ defect does not recur in donor heart." />
        <Alert variant="warning" text="G-CSF (filgrastim): Start when ANC <0.5×10⁹/L or recurrent severe bacterial infections. Monitor CBC 2× weekly. Cycle tracking for cyclic neutropenia predicts nadir timing." />
        <Alert variant="success" text="LEV (levetiracetam): Preferred AED if seizures occur (uncommon in Barth). Renal excretion; no mito toxicity. Always exclude DCM-related cerebral embolism before diagnosing primary epilepsy." />
        <Alert variant="info" text="Elamipretide (SS-31): Investigational — TAZPOWER Phase II trial improved 6MWT and fatigue. Available compassionate use via Barth Syndrome Foundation. Not yet FDA approved." />
        <Alert variant="info" text="PHT / CBZ / OXC: Avoid in DCM — Na-channel blockade worsens cardiac conduction. If needed, joint cardiology-neurology decision required with continuous cardiac monitoring." />
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const defs = data.definitions || [];

  return (
    <div>
      <SectionCard title="📖 TAZ Barth Syndrome — Definitions">
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

export default function TAZPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/taz/overview`).then(r => r.json()),
      fetch(`${API}/api/taz/breakdown`).then(r => r.json()),
      fetch(`${API}/api/taz/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#f8f9fa' }}>
      {/* Header */}
      <div style={{ background: COLOR, color: 'white', padding: '16px 24px' }}>
        <h4 className="mb-1 fw-bold">❤️ TAZ — Barth Syndrome</h4>
        <div className="small opacity-75">
          3-MGA-uria Type II · X-linked Cardiomyopathy + Neutropenia + Myopathy · Xq28 · OMIM 302060
        </div>
        <div className="small opacity-75 mt-1">
          Tafazzin (292aa) · Cardiolipin Remodeling Enzyme · IMM · MLCL→Mature CL (TLCL) · ~200-300 patients worldwide · Barth 1983
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
            {tab === 2 && <CardiacNeutropeniaTab data={breakdown} />}
            {tab === 3 && <TreatmentsTab data={breakdown} />}
            {tab === 4 && <DefinitionsTab data={definitions} />}
          </>
        )}
      </div>
    </div>
  );
}
