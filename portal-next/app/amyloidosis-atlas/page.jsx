'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  TTR:   '#1a237e',  // deep navy — ATTR-FAP/CM, most common, patisiran/tafamidis
  APOA1: '#880e4f',  // deep crimson — AApoAI, low HDL, renal/hepatic/cardiac
  APOA2: '#4e342e',  // deep brown — AApoAII, renal predominant, frameshift C-terminal
  LYZ:   '#b71c1c',  // deep red — ALys, renal + hepatic + GI bleed, Ile56Thr
  GSN:   '#1b5e20',  // deep green — AGel Meretoja, cranial neuropathy + lattice corneal PATHOGNOMONIC
  FGA:   '#37474f',  // dark slate — AFib Ostertag, renal, liver Tx CURATIVE
  CST3:  '#006064',  // dark teal — ACys Icelandic, cerebrovascular young stroke fatal
  B2M:   '#4a148c',  // deep purple — AB2M dialysis, carpal tunnel PATHOGNOMONIC, HDF prevention
};

const GENE_DISEASE = {
  TTR:   'ATTR Amyloidosis (AD) — Transthyretin; FAP (neuropathy) + CM (cardiomyopathy); Val30Met Founder; Patisiran/Inotersen/Tafamidis',
  APOA1: 'AApoAI Amyloidosis (AD) — Apolipoprotein A-I; LOW HDL PATHOGNOMONIC; Renal + Hepatic + Cardiac; No Approved Amyloid Rx',
  APOA2: 'AApoAII Amyloidosis (AD) — Apolipoprotein A-II; RENAL PREDOMINANT; C-terminal Frameshift Extension; Ostertag Type',
  LYZ:   'ALys Amyloidosis (AD) — Lysozyme; Ile56Thr/Asp67His; Renal + Hepatic + GI Bleeding; No Neuropathy',
  GSN:   'AGel Amyloidosis / Meretoja Syndrome (AD) — Gelsolin; CRANIAL NEUROPATHY + LATTICE CORNEAL DYSTROPHY PATHOGNOMONIC; Asp187Asn Finnish',
  FGA:   'AFib Amyloidosis / Ostertag (AD) — Fibrinogen Alpha Chain; RENAL PREDOMINANT; Glu526Val; LIVER Tx CURATIVE (Renal Tx Alone FAILS)',
  CST3:  'ACys Amyloidosis / HCHWA-I (AD) — Cystatin C; CEREBROVASCULAR; Young Stroke <40y; Glu68Gln Icelandic Founder; Fatal <30y',
  B2M:   'AB2M Amyloidosis (dialysis) — Beta-2 Microglobulin; CARPAL TUNNEL PATHOGNOMONIC in Dialysis; Destructive Spondyloarthropathy; HDF Prevents',
};

const NEUROPATHY_GENES    = ['TTR'];
const CARDIAC_GENES       = ['TTR', 'APOA1'];
const RENAL_GENES         = ['APOA1', 'APOA2', 'LYZ', 'FGA'];
const CEREBROVASCULAR_GENES = ['CST3'];
const CRANIAL_NEURO_GENES = ['GSN'];
const DIALYSIS_GENES      = ['B2M'];
const LIVER_TX_CURATIVE   = ['FGA'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Hereditary Amyloidosis atlas…</p>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger m-4"><strong>Error:</strong> {msg}</div>;
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-sm-4 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-body text-center p-2" style={{ borderTop: `4px solid ${color}` }}>
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function AlertBadge({ text, color = '#b71c1c' }) {
  return (
    <span className="badge me-1 mb-1" style={{ background: color, fontSize: '0.7rem' }}>
      {text}
    </span>
  );
}

/* ── OVERVIEW TAB ── */
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats;

  return (
    <div>
      <div className="alert border-0 mb-4" style={{ background: '#e8eaf6' }}>
        <h5 className="mb-1">🧬 {data.atlas}</h5>
        <div className="text-muted small">{data.subtitle} · {data.total_patients} patients (8×40, seeds {data.seed_range})</div>
      </div>

      {/* Aggregate KPIs */}
      <h6 className="text-uppercase text-muted mb-3 small">Aggregate Cohort Statistics</h6>
      <div className="row g-2 mb-4">
        <KPI label="Peripheral Neuropathy" value={`${s.neuropathy_pct}%`} color="#1a237e" />
        <KPI label="Cardiomyopathy" value={`${s.cardiomyopathy_pct}%`} color="#880e4f" />
        <KPI label="Renal Involvement" value={`${s.renal_involvement_pct}%`} color="#b71c1c" />
        <KPI label="Cerebrovascular Event" value={`${s.cerebrovascular_pct}%`} color="#006064" />
        <KPI label="Carpal Tunnel" value={`${s.carpal_tunnel_pct}%`} color="#4a148c" />
        <KPI label="Cranial Neuropathy" value={`${s.cranial_neuropathy_pct}%`} color="#1b5e20" />
        <KPI label="Corneal Dystrophy" value={`${s.corneal_dystrophy_pct}%`} color="#37474f" />
        <KPI label="Total Genes" value="8" color="#455a64" />
      </div>

      {/* Key DDx Anchors */}
      <h6 className="text-uppercase text-muted mb-2 small">Key Clinical DDx Anchors</h6>
      <div className="card border-0 shadow-sm mb-4">
        <div className="card-body p-3">
          {data.key_ddx_anchor.map((k, i) => (
            <div key={i} className="d-flex align-items-start mb-2">
              <span className="me-2 mt-1" style={{ color: '#b71c1c', fontWeight: 'bold' }}>▶</span>
              <small>{k}</small>
            </div>
          ))}
        </div>
      </div>

      {/* Gene cards */}
      <h6 className="text-uppercase text-muted mb-3 small">Gene Summary</h6>
      <div className="row g-3">
        {data.genes_summary.map((g) => (
          <div key={g.gene} className="col-12 col-md-6">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-header text-white py-2 px-3" style={{ background: GENE_COLORS[g.gene] }}>
                <div className="d-flex justify-content-between align-items-center">
                  <span className="fw-bold">{g.gene}</span>
                  <span className="small opacity-75">{g.locus} · {g.aa}</span>
                </div>
                <div className="small opacity-90">{GENE_DISEASE[g.gene]?.split('—')[0].trim()}</div>
              </div>
              <div className="card-body p-3">
                <div className="row g-1 mb-2">
                  {g.neuropathy_pct  > 0 && <div className="col-6"><small className="text-muted">Neuropathy:</small> <strong>{g.neuropathy_pct}%</strong></div>}
                  {g.cardiomyopathy_pct > 0 && <div className="col-6"><small className="text-muted">Cardiomyopathy:</small> <strong>{g.cardiomyopathy_pct}%</strong></div>}
                  {g.renal_pct       > 0 && <div className="col-6"><small className="text-muted">Renal:</small> <strong>{g.renal_pct}%</strong></div>}
                  {g.hepatic_pct     > 0 && <div className="col-6"><small className="text-muted">Hepatic:</small> <strong>{g.hepatic_pct}%</strong></div>}
                  {g.carpal_tunnel_pct > 0 && <div className="col-6"><small className="text-muted">Carpal Tunnel:</small> <strong>{g.carpal_tunnel_pct}%</strong></div>}
                  {g.cranial_neuropathy_pct > 0 && <div className="col-6"><small className="text-muted">Cranial Neuro:</small> <strong>{g.cranial_neuropathy_pct}%</strong></div>}
                  {g.corneal_dystrophy_pct > 0 && <div className="col-6"><small className="text-muted">Corneal Dystrophy:</small> <strong>{g.corneal_dystrophy_pct}%</strong></div>}
                  {g.cerebrovascular_pct > 0 && <div className="col-6"><small className="text-muted">Cerebrovascular:</small> <strong>{g.cerebrovascular_pct}%</strong></div>}
                </div>
                <div className="mt-2">
                  {LIVER_TX_CURATIVE.includes(g.gene) && (
                    <AlertBadge text="LIVER Tx CURATIVE" color="#b71c1c" />
                  )}
                  {g.gene === 'TTR' && <AlertBadge text="LIVER Tx WORSENS CARDIAC" color="#880e4f" />}
                  {g.gene === 'CST3' && <AlertBadge text="FATAL <30y" color="#006064" />}
                  {g.gene === 'GSN' && <AlertBadge text="CRANIAL NERVE + CORNEA" color="#1b5e20" />}
                  {g.gene === 'B2M' && <AlertBadge text="HDF PREVENTS" color="#4a148c" />}
                  {g.gene === 'APOA1' && <AlertBadge text="LOW HDL" color="#37474f" />}
                </div>
                <div className="mt-2">
                  {g.hallmarks.slice(0, 2).map((h, i) => (
                    <div key={i} className="small text-muted">• {h}</div>
                  ))}
                </div>
                <div className="mt-2 p-2 rounded" style={{ background: '#fff3e0', fontSize: '0.72rem' }}>
                  <strong>⚠ Alert:</strong> {g.top_treatment_alert}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── GENE TABLE TAB ── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h6 className="text-uppercase text-muted mb-3 small">8-Gene Hereditary Amyloidosis Reference Table</h6>
      <div className="table-responsive">
        <table className="table table-sm table-bordered align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Protein (aa)</th><th>Locus</th><th>Disease</th>
              <th>Inheritance</th><th>Primary Organ</th><th>Key Feature</th><th>OMIM</th>
            </tr>
          </thead>
          <tbody>
            {data.map((g) => (
              <tr key={g.gene}>
                <td>
                  <span className="badge" style={{ background: GENE_COLORS[g.gene] }}>{g.gene}</span>
                </td>
                <td><small>{g.protein.split('(')[0].trim()} ({g.aa})</small></td>
                <td><small>{g.locus}</small></td>
                <td><small>{GENE_DISEASE[g.gene]?.split('—')[0].trim()}</small></td>
                <td><small>{g.inheritance.split(';')[0]}</small></td>
                <td>
                  <span className="badge" style={{ background: GENE_COLORS[g.gene], fontSize: '0.7rem' }}>
                    {NEUROPATHY_GENES.includes(g.gene) ? 'Neurologic' :
                     CEREBROVASCULAR_GENES.includes(g.gene) ? 'Cerebrovascular' :
                     CRANIAL_NEURO_GENES.includes(g.gene) ? 'Cranial/Corneal' :
                     DIALYSIS_GENES.includes(g.gene) ? 'MSK/Dialysis' : 'Renal'}
                  </span>
                </td>
                <td><small>{g.hallmarks[0]}</small></td>
                <td><small><a href={`https://omim.org/entry/${g.omim_disease}`} target="_blank" rel="noreferrer">#{g.omim_disease}</a></small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-gene deep detail */}
      <h6 className="text-uppercase text-muted mt-4 mb-3 small">Per-Gene Clinical Detail</h6>
      {data.map((g) => (
        <div key={g.gene} className="card border-0 shadow-sm mb-4">
          <div className="card-header text-white" style={{ background: GENE_COLORS[g.gene] }}>
            <div className="d-flex justify-content-between align-items-center">
              <span className="fw-bold fs-6">{g.gene} — {g.protein}</span>
              <span className="small opacity-75">{g.locus} · {g.aa} · OMIM #{g.omim_disease}</span>
            </div>
            <div className="small opacity-90 mt-1">{GENE_DISEASE[g.gene]}</div>
          </div>
          <div className="card-body p-3">
            <div className="row g-3">
              <div className="col-md-6">
                <h6 className="small text-uppercase text-muted">Hallmarks</h6>
                <ul className="small mb-0">
                  {g.hallmarks.map((h, i) => <li key={i}>{h}</li>)}
                </ul>
              </div>
              <div className="col-md-6">
                <h6 className="small text-uppercase text-muted">Treatment Alerts</h6>
                <ul className="small mb-0">
                  {g.treatment_alerts.map((t, i) => <li key={i}>{t}</li>)}
                </ul>
              </div>
              <div className="col-md-6">
                <h6 className="small text-uppercase text-muted">Key DDx</h6>
                <ul className="small mb-0">
                  {g.key_ddx.map((d, i) => <li key={i}>{d}</li>)}
                </ul>
              </div>
              <div className="col-md-6">
                <h6 className="small text-uppercase text-muted">Clinical Pearls</h6>
                <ul className="small mb-0">
                  {g.clinical_pearls.map((p, i) => <li key={i}>{p}</li>)}
                </ul>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h6 className="text-uppercase text-muted mb-3 small">Cohort Statistics per Gene</h6>
      <div className="row g-3">
        {data.map((g) => (
          <div key={g.gene} className="col-12 col-md-6">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-header text-white py-2 px-3" style={{ background: GENE_COLORS[g.gene] }}>
                <span className="fw-bold">{g.gene}</span>
                <span className="ms-2 small opacity-75">n={g.n_patients} · {g.males_pct}% male</span>
              </div>
              <div className="card-body p-3">
                {/* Feature rates */}
                <h6 className="small text-uppercase text-muted">Feature Rates</h6>
                <div className="row g-1 mb-2">
                  {Object.entries(g.feature_rates || {}).filter(([, v]) => v > 0).map(([k, v]) => (
                    <div key={k} className="col-6">
                      <small className="text-muted">{k.replace(/_pct$/, '').replace(/_/g, ' ')}:</small>{' '}
                      <strong>{v}%</strong>
                    </div>
                  ))}
                </div>

                {/* Diagnosis delay */}
                {g.avg_diagnosis_delay_years != null && (
                  <>
                    <h6 className="small text-uppercase text-muted mt-2">Diagnosis Metrics</h6>
                    <div className="row g-1 mb-2">
                      <div className="col-6"><small className="text-muted">Avg Onset:</small> <strong>{g.avg_age_at_onset} y</strong></div>
                      <div className="col-6"><small className="text-muted">Dx Delay:</small> <strong>{g.avg_diagnosis_delay_years} y</strong></div>
                    </div>
                  </>
                )}

                {/* Organ system distribution */}
                {g.organ_system_distribution && (
                  <>
                    <h6 className="small text-uppercase text-muted mt-2">Primary Organ System</h6>
                    <div>
                      {Object.entries(g.organ_system_distribution).map(([sys, d]) => (
                        <div key={sys} className="d-flex justify-content-between align-items-center mb-1">
                          <small className="text-muted text-truncate me-2" style={{ maxWidth: '65%' }}>{sys}</small>
                          <div className="d-flex align-items-center gap-1">
                            <div style={{ width: 60, height: 8, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden' }}>
                              <div style={{ width: `${d.pct}%`, height: '100%', background: GENE_COLORS[g.gene] }} />
                            </div>
                            <small className="text-muted">{d.pct}%</small>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}

                {/* Treatment distribution */}
                {g.treatment_distribution && (
                  <>
                    <h6 className="small text-uppercase text-muted mt-2">Treatment Distribution</h6>
                    <div>
                      {Object.entries(g.treatment_distribution).map(([tx, d]) => (
                        <div key={tx} className="d-flex justify-content-between align-items-center mb-1">
                          <small className="text-muted text-truncate me-2" style={{ maxWidth: '65%' }}>{tx}</small>
                          <div className="d-flex align-items-center gap-1">
                            <div style={{ width: 60, height: 8, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden' }}>
                              <div style={{ width: `${d.pct}%`, height: '100%', background: GENE_COLORS[g.gene] }} />
                            </div>
                            <small className="text-muted">{d.pct}%</small>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}

                {/* Etiology distribution */}
                {g.etiology_distribution && g.etiology_distribution.length > 0 && (
                  <>
                    <h6 className="small text-uppercase text-muted mt-2">Etiology / Variant</h6>
                    <div>
                      {g.etiology_distribution.map((e) => (
                        <div key={e.etiology} className="d-flex justify-content-between align-items-center mb-1">
                          <small className="text-muted text-truncate me-2" style={{ maxWidth: '75%' }}>{e.etiology}</small>
                          <div className="d-flex align-items-center gap-1">
                            <div style={{ width: 60, height: 8, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden' }}>
                              <div style={{ width: `${e.pct}%`, height: '100%', background: GENE_COLORS[g.gene] }} />
                            </div>
                            <small className="text-muted">{e.pct}%</small>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── DEFINITIONS TAB ── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h6 className="text-uppercase text-muted mb-3 small">Pharmacological Distinctions</h6>
      <div className="card border-0 shadow-sm mb-4">
        <div className="card-body p-3">
          {data.pharmacological_distinctions.map((d, i) => (
            <div key={i} className="d-flex align-items-start mb-2">
              <span className="me-2 mt-1 text-danger fw-bold">⚠</span>
              <small>{d}</small>
            </div>
          ))}
        </div>
      </div>

      <h6 className="text-uppercase text-muted mb-3 small">Clinical Definitions</h6>
      {data.definitions.map((def, i) => (
        <div key={i} className="card border-0 shadow-sm mb-3">
          <div className="card-body p-3">
            <h6 className="fw-bold mb-1" style={{ color: '#1a237e' }}>{def.term}</h6>
            <p className="small text-muted mb-0">{def.definition}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── MAIN PAGE ── */
export default function AmyloidosisAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/amyloidosis-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/amyloidosis-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/amyloidosis-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, br, def]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(def);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4 px-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: '2rem' }}>🧬</span>
        <div>
          <h4 className="mb-0 fw-bold">Hereditary Amyloidosis Atlas</h4>
          <div className="text-muted small">
            Complete 8-Gene Hereditary Amyloidosis Atlas ·
            TTR · APOA1 · APOA2 · LYZ · GSN · FGA · CST3 · B2M ·
            320 patients (8×40, seeds 1358–1365)
          </div>
        </div>
      </div>

      {/* Key DDx banner */}
      <div className="alert border-0 mb-3 py-2 px-3" style={{ background: '#e8eaf6', fontSize: '0.8rem' }}>
        <strong>🔑 Key DDx:</strong>{' '}
        <span className="text-primary fw-bold">TTR Liver Tx → cardiac WORSENS</span> (wild-type replaces mutant deposits) ·{' '}
        <span className="text-danger fw-bold">FGA Liver Tx CURATIVE</span> (renal Tx alone fails) ·{' '}
        <span className="fw-bold" style={{ color: '#1b5e20' }}>GSN</span> = cranial neuropathy + lattice corneal dystrophy (Finnish) ·{' '}
        <span className="text-info fw-bold">CST3</span> = stroke &lt;40y, fatal ·{' '}
        <span style={{ color: '#4a148c', fontWeight: 'bold' }}>B2M</span> = carpal tunnel in dialysis, HDF prevents
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active fw-bold' : ''}`}
              onClick={() => setTab(t)}
            >
              {t === 'Overview' && '📊 '}
              {t === 'Gene Table' && '🧬 '}
              {t === 'Clinical Atlas' && '🏥 '}
              {t === 'Definitions' && '📖 '}
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'       && <OverviewTab data={overview} />}
      {tab === 'Gene Table'     && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'    && <DefinitionsTab data={definitions} />}
    </div>
  );
}
