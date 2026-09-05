'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  CASR:   '#1a237e',  // deep navy — FHH1/ADH1/NSHPT, Ca-sensing receptor
  GNA11:  '#880e4f',  // deep crimson — FHH2/ADH2, Gα11 transducer
  AP2S1:  '#4e342e',  // deep brown — FHH3, Arg15 hotspot, more symptomatic
  MEN1:   '#1b5e20',  // deep green — MEN1, 4-gland hyperplasia
  CDC73:  '#b71c1c',  // deep red — HPT-JT, parathyroid carcinoma
  CDKN1B: '#37474f', // dark slate — MEN4, p27/Kip1
  GCM2:   '#006064',  // dark teal — FIH/FIHP, GCM2 master regulator
  PTH:    '#4a148c',  // deep purple — IHP, preproPTH, undetectable PTH
};

const GENE_DISEASE = {
  CASR:   'FHH1 (AD LOF) / ADH1 (AD GOF) / NSHPT (AR) — CCCR<0.01; Thiazide CI; No Surgery in FHH',
  GNA11:  'FHH2 (AD LOF) / ADH2 (AD GOF) — CASR-Negative FHH; Gα11 Downstream of CASR',
  AP2S1:  'FHH3 (AD LOF) — Arg15 Hotspot (His/Cys/Leu); More Symptomatic; AP2 CASR Endocytosis',
  MEN1:   'MEN1 (AD) — 4-Gland Hyperplasia 90%; ZES/Insulinoma; Pituitary; Screen from Age 8',
  CDC73:  'HPT-JT (AD) — Parathyroid Carcinoma 15-20%; Parafibromin IHC Absent; En-Bloc Resection',
  CDKN1B: 'MEN4 (AD) — p27/Kip1; Same Triad as MEN1; Order After MEN1-Negative WES',
  GCM2:   'Familial Isolated Hypoparathyroidism (AD/AR LOF) / FIHP (GOF) — Mg First; Ca 2.0-2.1',
  PTH:    'Isolated Familial Hypoparathyroidism (AD/AR) — Undetectable PTH; ER Stress (AD); Mg First',
};

const HYPERPARATHYROID_GENES = ['CASR', 'GNA11', 'AP2S1', 'MEN1', 'CDC73', 'CDKN1B'];
const HYPOPARATHYROID_GENES  = ['GCM2', 'PTH'];
const FHH_GENES = ['CASR', 'GNA11', 'AP2S1'];
const MEN_GENES = ['MEN1', 'CDKN1B'];
const HIGH_CANCER_RISK = ['CDC73', 'MEN1'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Parathyroid Disorders atlas…</p>
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
      <p className="text-muted mb-3" style={{ fontSize: '0.82rem' }}>{ov.subtitle} · {ov.n_patients} patients · {ov.gene_count} genes · Seeds {ov.seeds}</p>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={ov.n_patients} color="#1a237e" />
        <KPI label="Genes Covered" value={ov.gene_count} color="#880e4f" />
        {kpis.slice(2).map((k, i) => (
          <KPI key={i} label={k.label} value={k.value} color={Object.values(GENE_COLORS)[i % 8]} />
        ))}
      </div>

      {ov.drug_alerts && (
        <div className="mb-4">
          <h6 className="fw-bold text-danger mb-2">⚠ Drug &amp; Treatment Alerts</h6>
          {ov.drug_alerts.map((a, i) => <Alert key={i} text={a} variant="danger" />)}
        </div>
      )}

      {ov.critical_rules && (
        <div className="mb-4">
          <h6 className="fw-bold mb-2" style={{ color: '#1a237e' }}>Critical Clinical Rules</h6>
          {ov.critical_rules.map((r, i) => (
            <div key={i} className="alert alert-primary py-2 px-3 mb-2" style={{ fontSize: '0.83rem' }}>
              {r}
            </div>
          ))}
        </div>
      )}

      <div className="mb-3">
        <h6 className="fw-bold mb-2">Gene Colour Key</h6>
        <div className="d-flex flex-wrap gap-2">
          {Object.entries(GENE_COLORS).map(([gene, color]) => (
            <span key={gene} className="badge" style={{ backgroundColor: color, fontSize: '0.78rem' }}>
              {gene}
            </span>
          ))}
        </div>
      </div>

      <div className="row g-3 mt-2">
        <div className="col-md-4">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: '#1a237e' }}>FHH Cascade</h6>
              {FHH_GENES.map(g => (
                <div key={g} className="mb-1">
                  <span className="badge me-1" style={{ backgroundColor: GENE_COLORS[g] }}>{g}</span>
                  <span style={{ fontSize: '0.80rem' }}>{GENE_DISEASE[g].split('—')[0]}</span>
                </div>
              ))}
              <div className="mt-2 small text-muted">CCCR &lt;0.01 → all benign; no surgery; no thiazides</div>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: '#1b5e20' }}>MEN Syndromes</h6>
              {MEN_GENES.map(g => (
                <div key={g} className="mb-1">
                  <span className="badge me-1" style={{ backgroundColor: GENE_COLORS[g] }}>{g}</span>
                  <span style={{ fontSize: '0.80rem' }}>{GENE_DISEASE[g].split('—')[0]}</span>
                </div>
              ))}
              <div className="mt-2 small text-muted">Annual screen from age 8; cascade all first-degree</div>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: '#006064' }}>Hypoparathyroidism</h6>
              {HYPOPARATHYROID_GENES.map(g => (
                <div key={g} className="mb-1">
                  <span className="badge me-1" style={{ backgroundColor: GENE_COLORS[g] }}>{g}</span>
                  <span style={{ fontSize: '0.80rem' }}>{GENE_DISEASE[g].split('—')[0]}</span>
                </div>
              ))}
              <div className="mt-2 small text-muted">Mg first; Ca target 2.0-2.1; annual renal USS</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const [selected, setSelected] = useState(null);

  return (
    <div>
      <h5 className="fw-bold mb-3">Per-Gene Breakdown — 8 Genes (320 Patients)</h5>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-hover align-middle" style={{ fontSize: '0.80rem' }}>
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Protein</th><th>Locus</th><th>aa</th>
              <th>Inheritance</th><th>Syndrome</th><th>N</th><th>Details</th>
            </tr>
          </thead>
          <tbody>
            {data.map((g) => (
              <tr key={g.gene}>
                <td>
                  <span className="badge fw-bold" style={{ backgroundColor: GENE_COLORS[g.gene] }}>{g.gene}</span>
                </td>
                <td style={{ maxWidth: 180 }}><small>{g.protein}</small></td>
                <td><code>{g.locus}</code></td>
                <td>{g.aa}</td>
                <td><small>{(g.inheritance || '').split(';')[0]}</small></td>
                <td style={{ maxWidth: 200 }}>
                  <small>{GENE_DISEASE[g.gene]}</small>
                </td>
                <td>{g.cohort_stats?.n}</td>
                <td>
                  <button className="btn btn-outline-primary btn-sm py-0" onClick={() => setSelected(selected === g.gene ? null : g.gene)}>
                    {selected === g.gene ? 'Hide' : 'Show'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && (() => {
        const g = data.find(x => x.gene === selected);
        if (!g) return null;
        const sf = g.cohort_stats?.special_features || {};
        return (
          <div className="card border-0 shadow mb-4" style={{ borderLeft: `4px solid ${GENE_COLORS[g.gene]}` }}>
            <div className="card-body">
              <h5 className="fw-bold mb-1" style={{ color: GENE_COLORS[g.gene] }}>{g.gene} — {g.protein}</h5>
              <p className="small text-muted mb-2">OMIM gene: {g.omim_gene} · Disease: {g.omim_disease} · {g.kDa}</p>

              <div className="mb-3">
                <strong>Gene Class &amp; Pathophysiology:</strong>
                <p className="small mb-1 mt-1">{g.gene_class}</p>
              </div>
              <div className="mb-3">
                <strong>Phenotype:</strong>
                <p className="small mb-1 mt-1">{g.phenotype}</p>
              </div>
              <div className="mb-3">
                <strong>Inheritance:</strong>
                <p className="small mb-1 mt-1">{g.inheritance}</p>
              </div>

              <div className="row g-3">
                <div className="col-md-6">
                  <strong className="text-danger">Treatment Alerts:</strong>
                  <ul className="small mb-0 mt-1">
                    {(g.treatment_alerts || []).map((a, i) => <li key={i}>{a}</li>)}
                  </ul>
                </div>
                <div className="col-md-6">
                  <strong>Key Hallmarks:</strong>
                  <ul className="small mb-0 mt-1">
                    {(g.key_hallmarks || []).map((h, i) => <li key={i}>{h}</li>)}
                  </ul>
                </div>
              </div>

              {Object.keys(sf).length > 0 && (
                <div className="mt-3">
                  <strong>Cohort Special Features ({g.cohort_stats?.n} patients, seed {g.cohort_stats?.seed}):</strong>
                  <div className="d-flex flex-wrap gap-2 mt-2">
                    {Object.entries(sf).filter(([, v]) => typeof v !== 'object').map(([k, v]) => (
                      <span key={k} className="badge bg-secondary">{k.replace(/_/g, ' ')}: {String(v)}</span>
                    ))}
                  </div>
                  {sf.arg15_distribution && (
                    <div className="mt-2 small">
                      <strong>Arg15 variant distribution: </strong>
                      {Object.entries(sf.arg15_distribution).map(([v, c]) => `${v}=${c}`).join(', ')}
                    </div>
                  )}
                </div>
              )}

              <div className="mt-3">
                <strong>Key DDx:</strong>
                <ul className="small mb-0 mt-1">
                  {(g.ddx || []).map((d, i) => <li key={i}>{d}</li>)}
                </ul>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const [gene, setGene] = useState('CASR');
  const gd = data.find(x => x.gene === gene);

  return (
    <div>
      <h5 className="fw-bold mb-3">Clinical Atlas — Patient Cohort Browser</h5>
      <div className="d-flex flex-wrap gap-2 mb-4">
        {data.map(g => (
          <button key={g.gene}
            className={`btn btn-sm ${gene === g.gene ? 'text-white' : 'btn-outline-secondary'}`}
            style={gene === g.gene ? { backgroundColor: GENE_COLORS[g.gene], border: 'none' } : {}}
            onClick={() => setGene(g.gene)}>
            {g.gene}
          </button>
        ))}
      </div>

      {gd && (
        <>
          <div className="card border-0 shadow-sm mb-4" style={{ borderLeft: `4px solid ${GENE_COLORS[gene]}` }}>
            <div className="card-body">
              <h6 className="fw-bold mb-0" style={{ color: GENE_COLORS[gene] }}>{gene} — {gd.protein}</h6>
              <div className="small text-muted mb-2">{GENE_DISEASE[gene]}</div>
              <div className="row g-2">
                <div className="col-6 col-md-3">
                  <div className="fw-bold">{gd.cohort_stats?.n}</div><div className="small text-muted">Patients</div>
                </div>
                <div className="col-6 col-md-3">
                  <div className="fw-bold">{gd.cohort_stats?.female_pct}%</div><div className="small text-muted">Female</div>
                </div>
                <div className="col-6 col-md-3">
                  <div className="fw-bold">{gd.cohort_stats?.age_mean} yrs</div><div className="small text-muted">Mean Age Dx</div>
                </div>
                <div className="col-6 col-md-3">
                  <div className="fw-bold">{gd.cohort_stats?.mean_diagnosis_delay_yrs} yrs</div><div className="small text-muted">Mean Dx Delay</div>
                </div>
              </div>
            </div>
          </div>

          <div className="table-responsive">
            <table className="table table-sm table-striped" style={{ fontSize: '0.78rem' }}>
              <thead className="table-dark">
                <tr>
                  <th>Patient ID</th><th>Sex</th><th>Age Dx</th><th>Delay (yrs)</th>
                  {gene === 'CASR' && <><th>Phenotype</th><th>Ca (mmol/L)</th><th>CCCR</th></>}
                  {gene === 'GNA11' && <><th>Phenotype</th><th>Ca (mmol/L)</th><th>CASR Normal</th></>}
                  {gene === 'AP2S1' && <><th>Arg15 Variant</th><th>Ca (mmol/L)</th><th>Symptomatic</th><th>Neuropsych</th></>}
                  {gene === 'MEN1' && <><th>Ca (mmol/L)</th><th>Pituitary</th><th>Pancreatic NET</th><th>Nephrolithiasis</th></>}
                  {gene === 'CDC73' && <><th>Ca (mmol/L)</th><th>Carcinoma</th><th>Jaw Fibroma</th><th>Capsule Violated</th></>}
                  {gene === 'CDKN1B' && <><th>Ca (mmol/L)</th><th>Pituitary</th><th>MEN1 Normal</th><th>Annual Screen</th></>}
                  {gene === 'GCM2' && <><th>Phenotype</th><th>Ca (mmol/L)</th><th>Seizures</th><th>Mg Checked</th></>}
                  {gene === 'PTH' && <><th>Inheritance</th><th>Ca (mmol/L)</th><th>Seizures</th><th>Mg Checked</th></>}
                </tr>
              </thead>
              <tbody>
                {(gd.patients || []).slice(0, 40).map(p => (
                  <tr key={p.patient_id}>
                    <td><code style={{ fontSize: '0.72rem' }}>{p.patient_id}</code></td>
                    <td>{p.sex}</td>
                    <td>{p.age_at_dx}</td>
                    <td>{p.dx_delay_years}</td>
                    {gene === 'CASR' && (
                      <>
                        <td><span className="badge" style={{ backgroundColor: p.phenotype === 'FHH1' ? '#1a237e' : p.phenotype === 'ADH1' ? '#880e4f' : '#b71c1c', fontSize: '0.72rem' }}>{p.phenotype}</span></td>
                        <td>{p.serum_ca_mmol}</td>
                        <td>{p.cccr ? p.cccr.toFixed(4) : p.phenotype === 'NSHPT' ? 'N/A' : '—'}</td>
                      </>
                    )}
                    {gene === 'GNA11' && (
                      <>
                        <td><span className="badge" style={{ backgroundColor: p.phenotype === 'FHH2' ? '#880e4f' : '#4a148c', fontSize: '0.72rem' }}>{p.phenotype}</span></td>
                        <td>{p.serum_ca_mmol}</td>
                        <td>{p.casr_gene_normal ? '✓' : '—'}</td>
                      </>
                    )}
                    {gene === 'AP2S1' && (
                      <>
                        <td><code style={{ fontSize: '0.72rem' }}>{p.arg15_variant}</code></td>
                        <td>{p.serum_ca_mmol}</td>
                        <td>{p.symptomatic ? <span className="badge bg-warning text-dark">Yes</span> : 'No'}</td>
                        <td>{p.neuropsychiatric_features ? <span className="badge bg-danger">Yes</span> : 'No'}</td>
                      </>
                    )}
                    {gene === 'MEN1' && (
                      <>
                        <td>{p.serum_ca_mmol}</td>
                        <td>{p.pituitary_adenoma ? <span className="badge bg-warning text-dark">{p.pituitary_type}</span> : 'No'}</td>
                        <td>{p.pancreatic_net ? <span className="badge bg-danger">{p.gastrinoma_zes ? 'ZES' : p.insulinoma ? 'Insulinoma' : 'pNET'}</span> : 'No'}</td>
                        <td>{p.nephrolithiasis ? '✓' : '—'}</td>
                      </>
                    )}
                    {gene === 'CDC73' && (
                      <>
                        <td>{p.serum_ca_mmol}</td>
                        <td>{p.parathyroid_carcinoma ? <span className="badge bg-danger">⚠ Carcinoma</span> : 'Adenoma'}</td>
                        <td>{p.jaw_ossifying_fibroma ? <span className="badge bg-warning text-dark">Yes</span> : 'No'}</td>
                        <td>{p.capsule_violated_intraop ? <span className="badge bg-danger">⚠ Yes</span> : '—'}</td>
                      </>
                    )}
                    {gene === 'CDKN1B' && (
                      <>
                        <td>{p.serum_ca_mmol}</td>
                        <td>{p.pituitary_adenoma ? <span className="badge bg-warning text-dark">{p.pituitary_type}</span> : 'No'}</td>
                        <td>{p.men1_gene_normal ? '✓ Required' : '—'}</td>
                        <td>{p.annual_screen_compliant ? <span className="badge bg-success">✓</span> : <span className="badge bg-danger">✗</span>}</td>
                      </>
                    )}
                    {gene === 'GCM2' && (
                      <>
                        <td><span className="badge" style={{ backgroundColor: p.phenotype === 'FIH' ? '#006064' : '#1b5e20', fontSize: '0.72rem' }}>{p.phenotype}</span></td>
                        <td>{p.serum_ca_mmol}</td>
                        <td>{p.seizures ? <span className="badge bg-danger">⚠ Yes</span> : 'No'}</td>
                        <td>{p.mg_checked_corrected ? <span className="badge bg-success">✓</span> : <span className="badge bg-danger">✗ Not Done</span>}</td>
                      </>
                    )}
                    {gene === 'PTH' && (
                      <>
                        <td><span className="badge" style={{ backgroundColor: p.inheritance_type === 'AD' ? '#4a148c' : '#006064', fontSize: '0.72rem' }}>{p.inheritance_type}</span></td>
                        <td>{p.serum_ca_mmol}</td>
                        <td>{p.seizures ? <span className="badge bg-danger">⚠ Yes</span> : 'No'}</td>
                        <td>{p.mg_checked ? <span className="badge bg-success">✓</span> : <span className="badge bg-danger">✗ Not Done</span>}</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.definitions || [];
  const [open, setOpen] = useState(null);

  return (
    <div>
      <h5 className="fw-bold mb-3">Clinical Definitions &amp; Decision Rules</h5>
      {defs.map((d, i) => (
        <div key={i} className="card border-0 shadow-sm mb-3">
          <div className="card-header bg-light d-flex justify-content-between align-items-center"
            style={{ cursor: 'pointer' }} onClick={() => setOpen(open === i ? null : i)}>
            <span className="fw-bold" style={{ fontSize: '0.9rem' }}>{d.term}</span>
            <span className="text-muted">{open === i ? '▲' : '▼'}</span>
          </div>
          <div className="card-body py-2">
            <p className="small text-muted mb-2"><strong>Summary:</strong> {d.short}</p>
            {open === i && (
              <>
                <p className="small mb-2">{d.detail}</p>
                {d.clinical_rule && (
                  <div className="alert alert-primary py-2 px-3 mb-0" style={{ fontSize: '0.83rem' }}>
                    <strong>Clinical Rule:</strong> {d.clinical_rule}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function ParathyroidDisordersAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/parathyroid-disorders-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/parathyroid-disorders-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/parathyroid-disorders-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4">
      <div className="mb-3">
        <h3 className="fw-bold mb-0" style={{ color: '#1a237e' }}>
          &#x1f9ec; Parathyroid-Disorders-Atlas
        </h3>
        <p className="text-muted mb-0" style={{ fontSize: '0.85rem' }}>
          Complete 8-Gene Hereditary Parathyroid &amp; Calcium Metabolism Disorders Reference ·
          CASR · GNA11 · AP2S1 · MEN1 · CDC73 · CDKN1B · GCM2 · PTH ·
          320 patients (8×40, seeds 1342-1349)
        </p>
      </div>

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active fw-bold' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'      && <OverviewTab     data={overview}     />}
      {tab === 'Gene Table'    && <GeneTableTab    data={breakdown}    />}
      {tab === 'Clinical Atlas'&& <ClinicalAtlasTab data={breakdown}   />}
      {tab === 'Definitions'   && <DefinitionsTab  data={definitions}  />}
    </div>
  );
}
