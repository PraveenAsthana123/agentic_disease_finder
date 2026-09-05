'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  GP1BA:   '#1a237e',  // deep navy — BSS, giant platelets, AR
  ITGA2B:  '#880e4f',  // deep crimson — GT-A, zero aggregation, AR
  ITGB3:   '#b71c1c',  // deep red — GT-B, HPA-1a NAIT, AR
  MYH9:    '#1b5e20',  // deep green — AD, neutrophil inclusions, SNHL/nephritis
  NBEAL2:  '#37474f',  // dark slate — GPS, alpha-granule absent, myelofibrosis
  ANKRD26: '#827717',  // dark amber — THC2, 5UTR, AML risk, AD
  GFI1B:   '#006064',  // dark teal — macrocytosis pathognomonic, delta-granule, AD
  RUNX1:   '#4a148c',  // deep purple — FPD/AML, 35-44% AML, NO aspirin, AD
};

const GENE_DISEASE = {
  GP1BA:   'Bernard-Soulier Syndrome (BSS) — Giant Platelets, GPIb Absent, DDAVP Ineffective (AR)',
  ITGA2B:  'Glanzmann Thrombasthenia Type A — ALL Aggregation Absent, Normal Count, rFVIIa (AR)',
  ITGB3:   'Glanzmann Thrombasthenia Type B — Identical to GT-A; HPA-1a NAIT Risk (AR)',
  MYH9:    'MYH9-Related Disease — Giant Platelets + Neutrophil Inclusions; SNHL; Nephritis (AD)',
  NBEAL2:  'Gray Platelet Syndrome — Alpha-Granule Absent; Myelofibrosis 3rd-4th Decade (AR)',
  ANKRD26: 'Thrombocytopenia-2 (THC2) — 5\'UTR Mutations (WES Misses!); AML Risk 5-8% (AD)',
  GFI1B:   'GFI1B-Related Thrombocytopenia — Red Cell Macrocytosis PATHOGNOMONIC; Delta-Granule (AD)',
  RUNX1:   'Familial Platelet Disorder/AML (FPD/AML) — AML Risk 35-44%; NO Aspirin; Exclude Family Donors (AD)',
};

const AD_GENES = ['MYH9', 'ANKRD26', 'GFI1B', 'RUNX1'];
const AR_GENES  = ['GP1BA', 'ITGA2B', 'ITGB3', 'NBEAL2'];
const GT_GENES  = ['ITGA2B', 'ITGB3'];
const HIGH_RISK = ['RUNX1', 'ANKRD26'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Platelet Disorders atlas…</p>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger m-4"><strong>Error:</strong> {msg}</div>;
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-3 mb-3">
      <div className="card h-100 shadow-sm border-0">
        <div className="card-body text-center p-3">
          <div className="fw-bold fs-3" style={{ color: color || '#1a237e' }}>{value}</div>
          <div className="small text-muted">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 px-3 mb-2`} style={{ fontSize: '0.85rem' }}>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const kpis = ov.kpis || [];
  return (
    <div>
      <h4 className="fw-bold mb-1" style={{ color: '#1a237e' }}>{ov.atlas_name}</h4>
      <p className="text-muted mb-3">{ov.subtitle} · {ov.n_patients} patients · {ov.gene_count} genes · Seeds {ov.seeds}</p>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={ov.n_patients} color="#1a237e" />
        <KPI label="Genes Covered" value={ov.gene_count} color="#880e4f" />
        {kpis.slice(0, 6).map((k, i) => (
          <KPI key={i} label={k.label} value={k.value} color={k.color || '#1a237e'} />
        ))}
      </div>

      {(ov.drug_alerts || []).length > 0 && (
        <div className="mb-4">
          <h6 className="fw-bold text-danger mb-2">⚠ CRITICAL DRUG / PROCEDURE ALERTS</h6>
          {ov.drug_alerts.map((p, i) => (
            <Alert key={i} text={p} variant="danger" />
          ))}
        </div>
      )}

      {(ov.critical_rules || []).length > 0 && (
        <div className="mb-4">
          <h6 className="fw-bold text-secondary mb-2">KEY CLINICAL RULES</h6>
          {ov.critical_rules.map((p, i) => (
            <Alert key={i} text={p} variant="warning" />
          ))}
        </div>
      )}

      <h6 className="fw-bold text-secondary mb-2">GENE SUMMARY</h6>
      <div className="table-responsive">
        <table className="table table-bordered table-sm align-middle mb-0" style={{ fontSize: '0.82rem' }}>
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Protein</th><th>Size</th><th>Locus</th>
              <th>Inheritance</th><th>Phenotype</th><th>Hallmark</th>
            </tr>
          </thead>
          <tbody>
            {(ov.gene_summary || []).map(g => (
              <tr key={g.gene}>
                <td><span className="badge" style={{ background: GENE_COLORS[g.gene] || '#333', color: '#fff' }}>{g.gene}</span></td>
                <td className="fw-semibold">{g.protein}</td>
                <td>{g.aa}</td>
                <td><code>{g.locus}</code></td>
                <td>{g.inheritance}</td>
                <td>{g.phenotype_short}</td>
                <td>{g.hallmark_short}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h5 className="fw-bold mb-3" style={{ color: '#1a237e' }}>Per-Gene Cohort Statistics</h5>
      <div className="table-responsive">
        <table className="table table-bordered table-sm align-middle" style={{ fontSize: '0.82rem' }}>
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Disease</th><th>Locus</th><th>aa</th>
              <th>Inheritance</th><th>N</th><th>Seed</th>
              <th>Age Mean</th><th>Mean PLT (k)</th><th>Female %</th>
            </tr>
          </thead>
          <tbody>
            {data.map(g => (
              <tr key={g.gene}>
                <td>
                  <span className="badge" style={{ background: GENE_COLORS[g.gene] || '#333', color: '#fff' }}>{g.gene}</span>
                </td>
                <td style={{ maxWidth: 200, fontSize: '0.78rem' }}>{GENE_DISEASE[g.gene] || g.gene}</td>
                <td><code>{g.locus}</code></td>
                <td>{g.aa}</td>
                <td>{g.inheritance?.split(';')[0]}</td>
                <td className="text-center fw-bold">{g.cohort_stats?.n}</td>
                <td className="text-center">{g.cohort_stats?.seed}</td>
                <td className="text-center">{g.cohort_stats?.age_mean}</td>
                <td className="text-center fw-bold">{g.cohort_stats?.mean_platelet_count_k}</td>
                <td className="text-center">{g.cohort_stats?.female_pct}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data.map(g => (
        <div key={g.gene} className="mb-4">
          <h6 className="fw-bold mt-4 mb-2" style={{ color: GENE_COLORS[g.gene] || '#333' }}>
            {g.gene} — {GENE_DISEASE[g.gene]} ({g.cohort_stats?.n} patients · mean PLT {g.cohort_stats?.mean_platelet_count_k}k)
          </h6>
          <div className="table-responsive">
            <table className="table table-sm table-bordered" style={{ fontSize: '0.78rem' }}>
              <thead style={{ background: GENE_COLORS[g.gene], color: '#fff' }}>
                <tr>
                  <th>ID</th><th>Age</th><th>Sex</th><th>PLT (k)</th><th>Platelet Size</th>
                  {g.gene === 'GP1BA' && <>
                    <th>Giant Platelets</th><th>Ristocetin Absent</th><th>DDAVP Given</th><th>DDAVP Effective</th><th>Alloimmunized</th>
                  </>}
                  {GT_GENES.includes(g.gene) && <>
                    <th>Clot Retraction</th><th>Ristocetin</th><th>rFVIIa Given</th><th>Alloimmunized</th>
                    {g.gene === 'ITGB3' && <th>NAIT Risk</th>}
                  </>}
                  {g.gene === 'MYH9' && <>
                    <th>Neutrophil Inclusions</th><th>SNHL</th><th>Nephritis</th><th>Cataracts</th><th>ACE-I Prescribed</th><th>Aspirin Given</th>
                  </>}
                  {g.gene === 'NBEAL2' && <>
                    <th>Gray Platelets</th><th>Splenomegaly</th><th>Myelofibrosis</th><th>BM Biopsy</th>
                  </>}
                  {g.gene === 'ANKRD26' && <>
                    <th>5UTR Variant</th><th>WES Detected</th><th>Targeted Seq</th><th>AML/MDS</th><th>TPO-Agonist</th><th>Annual CBC</th>
                  </>}
                  {g.gene === 'GFI1B' && <>
                    <th>Macrocytosis</th><th>MCV (fL)</th><th>CD34+ Platelets</th><th>Delta-Granule Absent</th>
                  </>}
                  {g.gene === 'RUNX1' && <>
                    <th>Collagen Agg. Reduced</th><th>AML/MDS</th><th>Annual CBC</th><th>HSCT Done</th><th>Aspirin Given</th>
                  </>}
                </tr>
              </thead>
              <tbody>
                {(g.patients || []).slice(0, 10).map(p => (
                  <tr key={p.patient_id}>
                    <td>{p.patient_id}</td>
                    <td>{p.age_at_dx}</td>
                    <td>{p.sex}</td>
                    <td className="fw-bold">{p.platelet_count_k}</td>
                    <td><span className="badge bg-secondary">{p.platelet_size}</span></td>
                    {g.gene === 'GP1BA' && <>
                      <td>{p.giant_platelets ? '✓' : '—'}</td>
                      <td>{p.ristocetin_agglutination === false ? '✓ absent' : '—'}</td>
                      <td>{p.desmopressin_given ? '✓' : '—'}</td>
                      <td>{p.desmopressin_effective ? '✓' : <span className="text-danger">✗</span>}</td>
                      <td>{p.alloimmunized ? <span className="text-danger">✓</span> : '—'}</td>
                    </>}
                    {GT_GENES.includes(g.gene) && <>
                      <td>{p.clot_retraction === false ? <span className="text-danger">✗ absent</span> : '✓'}</td>
                      <td>{p.ristocetin_agglutination ? '✓ normal' : '—'}</td>
                      <td>{p.fvii_given ? '✓' : '—'}</td>
                      <td>{p.alloimmunized ? <span className="text-danger">✓</span> : '—'}</td>
                      {g.gene === 'ITGB3' && <td>{p.nait_risk ? <span className="text-danger">✓</span> : '—'}</td>}
                    </>}
                    {g.gene === 'MYH9' && <>
                      <td>{p.doehle_bodies ? '✓' : '—'}</td>
                      <td>{p.sensorineural_hearing_loss ? <span className="text-warning">✓</span> : '—'}</td>
                      <td>{p.nephritis ? <span className="text-danger">✓</span> : '—'}</td>
                      <td>{p.cataracts ? <span className="text-warning">✓</span> : '—'}</td>
                      <td>{p.ace_inhibitor_prescribed ? '✓' : '—'}</td>
                      <td>{p.aspirin_given ? <span className="text-danger">✓ AVOID</span> : '—'}</td>
                    </>}
                    {g.gene === 'NBEAL2' && <>
                      <td>{p.gray_platelets ? '✓' : '—'}</td>
                      <td>{p.splenomegaly ? <span className="text-warning">✓</span> : '—'}</td>
                      <td>{p.myelofibrosis ? <span className="text-danger">✓</span> : '—'}</td>
                      <td>{p.bm_biopsy_done ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'ANKRD26' && <>
                      <td>{p['5utr_variant'] ? '✓' : '—'}</td>
                      <td>{p.wes_detected ? '✓' : <span className="text-danger">✗ WES fails</span>}</td>
                      <td>{p.targeted_sequencing_done ? '✓' : '—'}</td>
                      <td>{p.aml_mds_developed ? <span className="text-danger">✓</span> : '—'}</td>
                      <td>{p.tpo_agonist_given ? <span className="text-danger">✓ AVOID</span> : '—'}</td>
                      <td>{p.annual_cbc_done ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'GFI1B' && <>
                      <td>{p.red_cell_macrocytosis ? '✓' : '—'}</td>
                      <td>{p.mcv}</td>
                      <td>{p.cd34_on_platelets ? '✓' : '—'}</td>
                      <td>{p.delta_granules_absent ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'RUNX1' && <>
                      <td>{p.collagen_aggregation_reduced ? '✓' : '—'}</td>
                      <td>{p.aml_mds_developed ? <span className="text-danger">✓</span> : '—'}</td>
                      <td>{p.annual_cbc_done ? '✓' : '—'}</td>
                      <td>{p.hsct_done ? '✓' : '—'}</td>
                      <td>{p.aspirin_given ? <span className="text-danger">✓ AVOID</span> : '—'}</td>
                    </>}
                  </tr>
                ))}
              </tbody>
            </table>
            {(g.patients || []).length > 10 && (
              <p className="text-muted small ms-1">Showing 10 of {g.patients.length} patients</p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      {data.map(g => (
        <div key={g.gene} className="card mb-4 border-0 shadow-sm">
          <div className="card-header d-flex align-items-center" style={{ background: GENE_COLORS[g.gene] || '#333', color: '#fff' }}>
            <span className="fw-bold fs-5 me-3">{g.gene}</span>
            <span>{g.protein} · {g.aa} · {g.locus} · {g.inheritance?.split(';')[0]}</span>
          </div>
          <div className="card-body p-3">
            <div className="row g-3">
              <div className="col-md-6">
                <h6 className="fw-bold text-secondary">GENE CLASS</h6>
                <p style={{ fontSize: '0.82rem' }}>{g.gene_class}</p>
              </div>
              <div className="col-md-6">
                <h6 className="fw-bold text-secondary">PHENOTYPE</h6>
                <p style={{ fontSize: '0.82rem' }}>{g.phenotype}</p>
              </div>
              <div className="col-md-6">
                <h6 className="fw-bold text-secondary">KEY HALLMARKS</h6>
                <ul style={{ fontSize: '0.82rem' }} className="mb-0">
                  {(g.key_hallmarks || []).map((h, i) => <li key={i}>{h}</li>)}
                </ul>
              </div>
              <div className="col-md-6">
                <h6 className="fw-bold text-secondary">TREATMENT ALERTS</h6>
                <ul style={{ fontSize: '0.82rem' }} className="mb-0">
                  {(g.treatment_alerts || []).map((t, i) => <li key={i}>{t}</li>)}
                </ul>
              </div>
              <div className="col-12">
                <h6 className="fw-bold text-secondary">KEY DDx</h6>
                <ul style={{ fontSize: '0.82rem' }} className="mb-0">
                  {(g.ddx || []).map((d, i) => <li key={i}>{d}</li>)}
                </ul>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.definitions || [];
  return (
    <div>
      <h5 className="fw-bold mb-3" style={{ color: '#1a237e' }}>Clinical Definitions — Platelet Disorders Atlas</h5>
      {defs.map((d, i) => (
        <div key={i} className="card mb-3 border-0 shadow-sm">
          <div className="card-body p-3">
            <h6 className="fw-bold mb-1" style={{ color: '#1a237e' }}>{d.term}</h6>
            <p className="text-muted mb-2" style={{ fontSize: '0.85rem' }}><em>{d.short}</em></p>
            <p style={{ fontSize: '0.82rem' }} className="mb-2">{d.detail}</p>
            {d.clinical_rule && (
              <div className="alert alert-warning py-1 px-2 mb-0" style={{ fontSize: '0.8rem' }}>
                <strong>Clinical Rule:</strong> {d.clinical_rule}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function PlateletDisordersAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/platelet-disorders-atlas`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  const renderTab = () => {
    switch (tab) {
      case 'Overview':       return <OverviewTab data={overview} />;
      case 'Gene Table':     return <GeneTableTab data={breakdown} />;
      case 'Clinical Atlas': return <ClinicalAtlasTab data={breakdown} />;
      case 'Definitions':    return <DefinitionsTab data={definitions} />;
      default:               return null;
    }
  };

  return (
    <div className="container-fluid py-3 px-4" style={{ maxWidth: 1400 }}>
      <div className="d-flex align-items-center mb-3">
        <span style={{ fontSize: '2rem', marginRight: 12 }}>🩸</span>
        <div>
          <h3 className="mb-0 fw-bold" style={{ color: '#1a237e' }}>Platelet-Disorders-Atlas</h3>
          <small className="text-muted">
            Complete 8-Gene Hereditary Platelet Disorders Atlas ·{' '}
            {Object.entries(GENE_COLORS).map(([g, c]) => (
              <span key={g} className="badge me-1" style={{ background: c, color: '#fff' }}>{g}</span>
            ))}
            · 320 patients (8×40, seeds 1334–1341)
          </small>
        </div>
      </div>

      <div className="d-flex gap-2 mb-3 flex-wrap">
        <span className="badge" style={{ background: '#1a237e', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          🔵 AR Adhesion Defect: GP1BA (BSS)
        </span>
        <span className="badge" style={{ background: '#880e4f', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          ⚠ AR Aggregation Defect: ITGA2B · ITGB3 (GT)
        </span>
        <span className="badge" style={{ background: '#1b5e20', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          🟢 AD Morphological: MYH9 (neutrophil inclusions)
        </span>
        <span className="badge" style={{ background: '#37474f', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          ⬛ AR Granule Defect: NBEAL2 (GPS)
        </span>
        <span className="badge" style={{ background: '#827717', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          🟡 AD AML Risk: ANKRD26 · RUNX1
        </span>
        <span className="badge" style={{ background: '#006064', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          🟦 AD Delta-Granule: GFI1B (macrocytosis)
        </span>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active fw-semibold' : ''}`}
              onClick={() => setTab(t)}
              style={tab === t ? { color: '#1a237e', borderBottomColor: '#1a237e', borderBottomWidth: 2 } : {}}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {renderTab()}
    </div>
  );
}
