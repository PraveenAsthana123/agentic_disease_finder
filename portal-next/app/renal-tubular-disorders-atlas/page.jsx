'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  SLC12A1: '#1a237e',  // deep navy — Bartter 1, NKCC2, loop-diuretic target
  KCNJ1:   '#880e4f',  // deep crimson — Bartter 2, ROMK, neonatal hyperK paradox
  CLCNKB:  '#4e342e',  // deep brown — Bartter 3, most common, MLPA mandatory
  BSND:    '#b71c1c',  // deep red — Bartter 4a, deafness PATHOGNOMONIC, barttin
  SLC12A3: '#1b5e20',  // deep green — Gitelman, HYPOcalciuria, Mg first
  SCNN1B:  '#37474f',  // dark slate — Liddle/PHA1B, ENaC-β, amiloride not spiro
  CLCN5:   '#006064',  // dark teal — Dent 1, XLR, LMW proteinuria mandatory
  SLC34A1: '#4a148c',  // deep purple — HHRH, NaPi-IIa, FGF23 LOW vs XLH HIGH
};

const GENE_DISEASE = {
  SLC12A1: 'Bartter Syndrome Type 1 (AR) — NKCC2; Antenatal Polyhydramnios; HYPERcalciuria; Loop-Diuretic Phenocopy',
  KCNJ1:   'Bartter Syndrome Type 2 (AR) — ROMK; Neonatal HYPERkalemia Resolves Spontaneously; HYPERcalciuria',
  CLCNKB:  'Bartter Syndrome Type 3 (AR) — ClC-Kb; Most Common Bartter; Later Onset; MLPA Mandatory',
  BSND:    'Bartter Type 4a (AR) — Barttin; SENSORINEURAL DEAFNESS PATHOGNOMONIC; Most Severe; GJB2 First',
  SLC12A3: 'Gitelman Syndrome (AR) — NCC; HYPOcalciuria + HYPOmag; Mg First; Avoid Thiazides Absolutely',
  SCNN1B:  'Liddle Syndrome (AD GOF) / PHA1B (AR LOF) — ENaC-β; LOW Renin LOW Aldosterone; Amiloride NOT Spiro',
  CLCN5:   'Dent Disease Type 1 (XLR) — ClC-5; LMW Proteinuria Mandatory; HYPERcalciuria; MLPA Mandatory',
  SLC34A1: 'HHRH (AR) — NaPi-IIa; FGF23 LOW (vs XLH HIGH); Rickets; Burosumab CONTRAINDICATED',
};

const BARTTER_GENES = ['SLC12A1', 'KCNJ1', 'CLCNKB', 'BSND'];
const HYPO_CA_GENES  = ['SLC12A3'];
const HYPER_CA_GENES = ['SLC12A1', 'KCNJ1', 'BSND', 'CLCN5', 'SLC34A1'];
const MLPA_MANDATORY = ['CLCNKB', 'CLCN5'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Renal Tubular Disorders atlas…</p>
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
      <div className="alert border-0 mb-4" style={{ background: '#e8f5e9' }}>
        <h5 className="mb-1">🧬 {data.atlas}</h5>
        <div className="text-muted small">{data.subtitle} · {data.total_patients} patients (8×40, seeds {data.seed_range})</div>
      </div>

      {/* Aggregate KPIs */}
      <h6 className="text-uppercase text-muted mb-3 small">Aggregate Cohort Statistics</h6>
      <div className="row g-2 mb-4">
        <KPI label="Antenatal Polyhydramnios" value={`${s.antenatal_polyhydramnios_pct}%`} color="#1a237e" />
        <KPI label="SNHL (Bartter 4)" value={`${s.snhl_pct}%`} color="#b71c1c" />
        <KPI label="Nephrocalcinosis/Stone" value={`${s.nephrocalcinosis_or_stone_pct}%`} color="#4e342e" />
        <KPI label="Hypertension (Liddle)" value={`${s.hypertension_pct}%`} color="#37474f" />
        <KPI label="LMW Proteinuria (Dent)" value={`${s.lmw_proteinuria_pct}%`} color="#006064" />
        <KPI label="Rickets (HHRH)" value={`${s.rickets_pct}%`} color="#4a148c" />
        <KPI label="Hypomagnesaemia" value={`${s.hypomagnesemia_pct}%`} color="#1b5e20" />
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
                  {g.snhl_pct > 0 && <div className="col-6"><small className="text-muted">SNHL:</small> <strong>{g.snhl_pct}%</strong></div>}
                  {g.antenatal_pct > 0 && <div className="col-6"><small className="text-muted">Antenatal:</small> <strong>{g.antenatal_pct}%</strong></div>}
                  {g.nephrocalcinosis_pct > 0 && <div className="col-6"><small className="text-muted">Nephrocalcinosis:</small> <strong>{g.nephrocalcinosis_pct}%</strong></div>}
                  {g.hypomagnesemia_pct > 0 && <div className="col-6"><small className="text-muted">HypoMg:</small> <strong>{g.hypomagnesemia_pct}%</strong></div>}
                  {g.hypertension_pct > 0 && <div className="col-6"><small className="text-muted">HTN:</small> <strong>{g.hypertension_pct}%</strong></div>}
                  {g.lmw_proteinuria_pct > 0 && <div className="col-6"><small className="text-muted">LMW Prot:</small> <strong>{g.lmw_proteinuria_pct}%</strong></div>}
                  {g.rickets_pct > 0 && <div className="col-6"><small className="text-muted">Rickets:</small> <strong>{g.rickets_pct}%</strong></div>}
                  <div className="col-6"><small className="text-muted">Calciuria:</small> <strong>
                    {HYPO_CA_GENES.includes(g.gene) ? 'HYPO' : HYPER_CA_GENES.includes(g.gene) ? 'HYPER' : 'Variable'}
                  </strong></div>
                </div>
                <div className="mt-2">
                  {MLPA_MANDATORY.includes(g.gene) && (
                    <AlertBadge text="MLPA MANDATORY" color="#b71c1c" />
                  )}
                  {g.gene === 'BSND' && <AlertBadge text="GJB2 FIRST" color="#880e4f" />}
                  {g.gene === 'SCNN1B' && <AlertBadge text="AMILORIDE NOT SPIRO" color="#37474f" />}
                  {g.gene === 'SLC34A1' && <AlertBadge text="FGF23 LOW" color="#4a148c" />}
                  {g.gene === 'SLC12A3' && <AlertBadge text="AVOID THIAZIDES" color="#b71c1c" />}
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
      <h6 className="text-uppercase text-muted mb-3 small">8-Gene Renal Tubular Disorders Reference Table</h6>
      <div className="table-responsive">
        <table className="table table-sm table-bordered align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Protein (aa)</th><th>Locus</th><th>Disease</th>
              <th>Inheritance</th><th>Calciuria</th><th>Key Feature</th><th>OMIM</th>
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
                  <span className={`badge ${
                    HYPO_CA_GENES.includes(g.gene) ? 'bg-success' :
                    HYPER_CA_GENES.includes(g.gene) ? 'bg-danger' : 'bg-secondary'
                  }`}>
                    {HYPO_CA_GENES.includes(g.gene) ? 'HYPO' :
                     HYPER_CA_GENES.includes(g.gene) ? 'HYPER' : 'Variable'}
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
                  {Object.entries(g.feature_rates).filter(([, v]) => v > 0).map(([k, v]) => (
                    <div key={k} className="col-6">
                      <small className="text-muted">{k.replace(/_pct$/, '').replace(/_/g, ' ')}:</small>{' '}
                      <strong>{v}%</strong>
                    </div>
                  ))}
                </div>

                {/* Lab averages */}
                <h6 className="small text-uppercase text-muted mt-2">Lab Averages</h6>
                <div className="row g-1 mb-2">
                  <div className="col-6"><small className="text-muted">K⁺:</small> <strong>{g.avg_k_serum} mmol/L</strong></div>
                  <div className="col-6"><small className="text-muted">HCO₃:</small> <strong>{g.avg_bicarb} mmol/L</strong></div>
                  <div className="col-6"><small className="text-muted">Mg²⁺:</small> <strong>{g.avg_mg_serum} mmol/L</strong></div>
                  <div className="col-6"><small className="text-muted">QTc:</small> <strong>{g.avg_qtc_ms} ms</strong></div>
                </div>

                {/* Calciuria distribution */}
                <h6 className="small text-uppercase text-muted mt-2">Calciuria</h6>
                <div className="d-flex flex-wrap gap-1 mb-2">
                  {Object.entries(g.calciuria_distribution).map(([cat, d]) => (
                    <span key={cat} className="badge" style={{
                      background: cat === 'HYPERcalciuria' ? '#b71c1c' :
                                  cat === 'HYPOcalciuria' ? '#1b5e20' : '#546e7a',
                      fontSize: '0.7rem'
                    }}>
                      {cat}: {d.pct}%
                    </span>
                  ))}
                </div>

                {/* Treatment rates */}
                <h6 className="small text-uppercase text-muted mt-2">Treatment Rates</h6>
                <div className="row g-1">
                  {Object.entries(g.treatment_rates).filter(([, v]) => v > 0).map(([k, v]) => (
                    <div key={k} className="col-6">
                      <small className="text-muted">{k.replace(/_pct$/, '').replace(/_/g, ' ')}:</small>{' '}
                      <strong>{v}%</strong>
                    </div>
                  ))}
                </div>

                {/* Etiology distribution */}
                <h6 className="small text-uppercase text-muted mt-2">Etiology</h6>
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
export default function RenalTubularDisordersAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/renal-tubular-disorders-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/renal-tubular-disorders-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/renal-tubular-disorders-atlas/definitions`).then(r => r.json()),
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
          <h4 className="mb-0 fw-bold">Renal Tubular Disorders Atlas</h4>
          <div className="text-muted small">
            Complete 8-Gene Hereditary Renal Tubular Transport Disorders ·
            SLC12A1 · KCNJ1 · CLCNKB · BSND · SLC12A3 · SCNN1B · CLCN5 · SLC34A1 ·
            320 patients (8×40, seeds 1350–1357)
          </div>
        </div>
      </div>

      {/* Calciuria DDx banner */}
      <div className="alert border-0 mb-3 py-2 px-3" style={{ background: '#e3f2fd', fontSize: '0.8rem' }}>
        <strong>🔑 Calciuria DDx:</strong>{' '}
        <span className="text-danger fw-bold">HYPERcalciuria</span> (Bartter 1/2/4, Dent, HHRH) vs{' '}
        <span className="text-success fw-bold">HYPOcalciuria</span> (Gitelman) ·{' '}
        <strong>MLPA mandatory:</strong> CLCNKB + CLCN5 ·{' '}
        <strong>Liddle:</strong> Amiloride NOT spironolactone ·{' '}
        <strong>HHRH:</strong> FGF23 LOW (vs XLH HIGH — burosumab CI)
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
