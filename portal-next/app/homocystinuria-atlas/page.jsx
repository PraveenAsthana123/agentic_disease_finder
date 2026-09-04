'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// Homocystinuria color palette
const COLOR  = '#4a148c';  // deep purple — methionine cycle
const LIGHT  = '#f3e5f5';  // lavender tint
const COLOR2 = '#b71c1c';  // thromboembolism / danger
const COLOR3 = '#1565c0';  // B12/cobalamin — blue
const COLOR4 = '#e65100';  // betaine / warning
const COLOR5 = '#1b5e20';  // B6-responsive / treatable
const COLOR6 = '#880e4f';  // retinal disease / cblC

const GENE_COLORS = {
  CBS:    '#b71c1c',   // most common; thromboembolism risk
  MTHFR:  '#1565c0',  // remethylation; methylfolate
  MTR:    '#4a148c',  // cblG; cobalamin
  MTRR:   '#6a1b9a',  // cblE; cobalamin reductase
  MMACHC: '#880e4f',  // cblC; retinal disease; most common combined
  MMADHC: '#c62828',  // cblD; variable
  AHCY:   '#e65100',  // SAH hydrolase; myopathy
  MAT1A:  '#1b5e20',  // hepatic; usually benign
};

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function BarRow({ label, pct, color = COLOR }) {
  return (
    <div className="mb-1">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ text, color = COLOR2 }) {
  return (
    <div className="alert mb-2 py-2 px-3 small" style={{ backgroundColor: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 4 }}>
      {text}
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: '0.7rem' }}>{text}</span>
  );
}

export default function HomocystinuriaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState({});
  const [error, setError] = useState({});

  function load(endpoint, setter, key) {
    setLoading(l => ({ ...l, [key]: true }));
    fetch(`${API}${endpoint}`)
      .then(r => r.json())
      .then(d => { setter(d); setLoading(l => ({ ...l, [key]: false })); })
      .catch(e => { setError(ev => ({ ...ev, [key]: e.message })); setLoading(l => ({ ...l, [key]: false })); });
  }

  useEffect(() => {
    load('/api/homocystinuria-atlas/overview', setOverview, 'ov');
    load('/api/homocystinuria-atlas/breakdown', setBreakdown, 'bd');
    load('/api/homocystinuria-atlas/definitions', setDefinitions, 'def');
  }, []);

  const cs = overview?.cohort_stats || {};
  const kt = overview?.key_teaching || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="p-3 mb-3 rounded" style={{ background: `linear-gradient(135deg, ${COLOR} 0%, #7b1fa2 100%)`, color: '#fff' }}>
        <h4 className="mb-0 fw-bold">&#x1f9ea; Homocystinuria-Atlas — Complete 8-Gene Homocystinuria &amp; Remethylation Disorders Atlas</h4>
        <div className="small mt-1 opacity-75">
          CBS (Classical HCU) · MTHFR · MTR (cblG) · MTRR (cblE) · MMACHC (cblC) · MMADHC (cblD) · AHCY · MAT1A
          &nbsp;|&nbsp; 320-patient aggregate (8×40, seeds 950–957)
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ── */}
      {tab === 'Overview' && (
        <div>
          {loading.ov && <div className="text-muted">Loading…</div>}
          {error.ov && <div className="alert alert-danger">{error.ov}</div>}
          {overview && (
            <>
              {/* KPI row */}
              <div className="row mb-3">
                <KPI label="Genes" value={overview.n_genes} color={COLOR} />
                <KPI label="Patients" value={overview.n_patients} color={COLOR} />
                <KPI label="Lens Dislocation (CBS)" value={cs.n_lens_dislocation} color={COLOR2} />
                <KPI label="Thromboembolism" value={cs.n_thromboembolism} color={COLOR2} />
                <KPI label="Retinal Disease (cblC)" value={cs.n_retinal_disease} color={COLOR6} />
                <KPI label="B6-Responsive (CBS)" value={cs.n_b6_responsive} color={COLOR5} />
                <KPI label="NBS Detected" value={cs.n_nbs_detected} color={COLOR3} />
                <KPI label="Megaloblastic Anemia" value={cs.n_megaloblastic_anemia} color={COLOR4} />
                <KPI label="Elevated MMA" value={cs.n_elevated_mma} color={COLOR4} />
                <KPI label="Leukoencephalopathy" value={cs.n_mri_leukoencephalopathy} color={COLOR} />
                <KPI label="Encephalopathy at Dx" value={cs.n_encephalopathy_at_dx} color={COLOR2} />
                <KPI label="Myopathy (AHCY)" value={cs.n_myopathy} color={COLOR4} />
              </div>

              {/* Prevalence bars */}
              <div className="row mb-3">
                <div className="col-md-6">
                  <div className="card shadow-sm">
                    <div className="card-header fw-bold" style={{ background: LIGHT }}>Cohort Prevalence</div>
                    <div className="card-body">
                      <BarRow label="NBS Detected" pct={cs.pct_nbs_detected} color={COLOR3} />
                      <BarRow label="Lens Dislocation" pct={cs.pct_lens_dislocation} color={COLOR2} />
                      <BarRow label="Thromboembolism" pct={cs.pct_thromboembolism} color={COLOR2} />
                      <BarRow label="Retinal Disease (cblC)" pct={cs.pct_retinal_disease} color={COLOR6} />
                      <BarRow label="Megaloblastic Anemia" pct={cs.pct_megaloblastic_anemia} color={COLOR4} />
                      <BarRow label="MRI Leukoencephalopathy" pct={cs.pct_mri_leuko} color={COLOR} />
                      <BarRow label="Elevated MMA" pct={cs.pct_elevated_mma} color={COLOR4} />
                      <BarRow label="B6-Responsive" pct={cs.pct_b6_responsive} color={COLOR5} />
                    </div>
                  </div>
                </div>
                <div className="col-md-6">
                  {/* Gene color legend */}
                  <div className="card shadow-sm mb-3">
                    <div className="card-header fw-bold" style={{ background: LIGHT }}>Gene Legend</div>
                    <div className="card-body">
                      {Object.entries(GENE_COLORS).map(([g, c]) => (
                        <div key={g} className="d-flex align-items-center mb-1">
                          <span style={{ width: 12, height: 12, borderRadius: 2, background: c, display: 'inline-block', marginRight: 8, flexShrink: 0 }} />
                          <span className="fw-bold me-2 small" style={{ color: c }}>{g}</span>
                          <span className="text-muted small">
                            {g === 'CBS' ? 'Classical HCU type I — transsulfuration block' :
                             g === 'MTHFR' ? 'MTHFR deficiency — 5-MTHF synthesis block' :
                             g === 'MTR' ? 'cblG — methionine synthase; MeCbl-dependent' :
                             g === 'MTRR' ? 'cblE — methionine synthase reductase' :
                             g === 'MMACHC' ? 'cblC — most common combined MMA+HC; retinal disease' :
                             g === 'MMADHC' ? 'cblD — variable: MMA-only / HC-only / combined' :
                             g === 'AHCY' ? 'SAH hydrolase — global methylation inhibition; myopathy' :
                             'MAT I/III — hepatic; usually benign hypermethioninemia'}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Key Teaching Points */}
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold" style={{ background: LIGHT }}>&#x1f4a1; Key Teaching Points</div>
                <div className="card-body">
                  {Object.entries(kt).map(([k, v]) => (
                    <Alert key={k} text={v} color={
                      k.includes('lens') ? COLOR2 :
                      k.includes('thromboembolism') ? COLOR2 :
                      k.includes('retinal') ? COLOR6 :
                      k.includes('B6') || k.includes('b6') ? COLOR5 :
                      k.includes('betaine') ? COLOR4 :
                      k.includes('hydroxo') ? COLOR3 :
                      k.includes('nbs') || k.includes('NBS') ? COLOR3 :
                      k.includes('methionine') ? COLOR :
                      COLOR
                    } />
                  ))}
                </div>
              </div>

              {/* Gene summary cards */}
              <div className="row">
                {(overview.genes || []).map(g => (
                  <div key={g.gene} className="col-md-6 col-lg-3 mb-3">
                    <div className="card h-100 shadow-sm" style={{ borderTop: `4px solid ${GENE_COLORS[g.gene] || COLOR}` }}>
                      <div className="card-body p-2">
                        <div className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</div>
                        <div className="text-muted small mb-1">{g.protein}</div>
                        <div className="small mb-1"><span className="badge bg-secondary me-1">{g.locus}</span><span className="badge bg-light text-dark me-1">{g.n_patients} pts</span></div>
                        <div className="small mb-1 text-muted">{g.hcu_subgroup}</div>
                        <div className="small">
                          {g.methionine_high
                            ? <Badge text="Met HIGH" color={COLOR2} />
                            : <Badge text="Met LOW" color={COLOR3} />}
                          {g.lens_dislocation && <Badge text="Lens Disloc." color={COLOR2} />}
                          {g.retinal_disease && <Badge text="Retinal Dz" color={COLOR6} />}
                          {g.combined_mma && <Badge text="MMA+HC" color={COLOR4} />}
                          {g.b6_responsive_pct > 0 && <Badge text={`B6-resp ${g.b6_responsive_pct}%`} color={COLOR5} />}
                          <Badge text={`tHcy~${g.mean_thcy_umolL} µmol/L`} color={GENE_COLORS[g.gene] || COLOR} />
                        </div>
                        <div className="small text-muted mt-1">{g.nbs_marker}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* ── Gene Table Tab ── */}
      {tab === 'Gene Table' && (
        <div>
          {loading.bd && <div className="text-muted">Loading…</div>}
          {error.bd && <div className="alert alert-danger">{error.bd}</div>}
          {breakdown && (
            <div className="table-responsive">
              <table className="table table-bordered table-sm table-hover align-middle small">
                <thead style={{ backgroundColor: COLOR, color: '#fff' }}>
                  <tr>
                    <th>Gene</th>
                    <th>Subgroup</th>
                    <th>Locus</th>
                    <th>Met</th>
                    <th>Mean tHcy (µmol/L)</th>
                    <th>Mean Met (µmol/L)</th>
                    <th>Lens Disloc.</th>
                    <th>Thrombosis</th>
                    <th>Retinal Dz</th>
                    <th>Megaloblastic</th>
                    <th>B6-Resp.</th>
                    <th>MMA Elev.</th>
                    <th>NBS Detected</th>
                    <th>Myopathy</th>
                    <th>Leuko-MRI</th>
                    <th>Thromb. Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.genes || []).map(g => (
                    <tr key={g.gene}>
                      <td><span className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</span></td>
                      <td>{g.alias?.split('—')[1]?.trim().split(';')[0] || ''}</td>
                      <td><span className="badge bg-secondary">{g.locus}</span></td>
                      <td>
                        {g.methionine_high
                          ? <span className="badge" style={{ backgroundColor: COLOR2 }}>HIGH</span>
                          : <span className="badge" style={{ backgroundColor: COLOR3 }}>LOW</span>}
                      </td>
                      <td className="fw-bold">{g.mean_thcy_dx}</td>
                      <td>{g.mean_methionine_dx}</td>
                      <td>{g.n_lens_dislocation > 0 ? <span className="badge" style={{ backgroundColor: COLOR2 }}>{g.n_lens_dislocation}</span> : <span className="text-muted">—</span>}</td>
                      <td>{g.n_thromboembolism > 0 ? <span className="badge" style={{ backgroundColor: COLOR2 }}>{g.n_thromboembolism}</span> : <span className="text-muted">—</span>}</td>
                      <td>{g.n_retinal_disease > 0 ? <span className="badge" style={{ backgroundColor: COLOR6 }}>{g.n_retinal_disease}</span> : <span className="text-muted">—</span>}</td>
                      <td>{g.n_megaloblastic_anemia > 0 ? <span className="badge" style={{ backgroundColor: COLOR4 }}>{g.n_megaloblastic_anemia}</span> : <span className="text-muted">—</span>}</td>
                      <td>{g.n_b6_responsive > 0 ? <span className="badge" style={{ backgroundColor: COLOR5 }}>{g.n_b6_responsive}</span> : <span className="text-muted">—</span>}</td>
                      <td>{g.n_mma_elevated > 0 ? <span className="badge" style={{ backgroundColor: COLOR4 }}>{g.n_mma_elevated}</span> : <span className="text-muted">—</span>}</td>
                      <td>{g.n_nbs_detected}/{g.n_patients}</td>
                      <td>{g.n_myopathy > 0 ? <span className="badge" style={{ backgroundColor: COLOR4 }}>{g.n_myopathy}</span> : <span className="text-muted">—</span>}</td>
                      <td>{g.n_mri_leuko}</td>
                      <td><span className="badge" style={{ backgroundColor: g.thromboembolism_risk === 'HIGH' ? COLOR2 : g.thromboembolism_risk === 'MODERATE' ? COLOR4 : '#607d8b' }}>{g.thromboembolism_risk}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ── Clinical Atlas Tab ── */}
      {tab === 'Clinical Atlas' && (
        <div>
          {loading.bd && <div className="text-muted">Loading…</div>}
          {error.bd && <div className="alert alert-danger">{error.bd}</div>}
          {breakdown && (breakdown.genes || []).map(g => (
            <div key={g.gene} className="card shadow-sm mb-4">
              <div className="card-header" style={{ background: GENE_COLORS[g.gene] || COLOR, color: '#fff' }}>
                <span className="fw-bold fs-6">{g.gene} — {g.protein}</span>
                <span className="ms-2 opacity-75 small">{g.locus} · {g.n_patients} patients · mean tHcy {g.mean_thcy_dx} µmol/L</span>
              </div>
              <div className="card-body">
                <div className="row">
                  <div className="col-md-6">
                    <div className="fw-bold small text-muted mb-1">Hallmarks</div>
                    <div className="small mb-2" style={{ whiteSpace: 'pre-wrap' }}>{g.hallmark}</div>
                  </div>
                  <div className="col-md-6">
                    <div className="fw-bold small text-muted mb-1">Clinical Detail</div>
                    <div className="small" style={{ maxHeight: 280, overflowY: 'auto', whiteSpace: 'pre-wrap' }}>{g.disease}</div>
                  </div>
                </div>
                <hr className="my-2" />
                <div className="row small">
                  <div className="col-md-4">
                    <span className="fw-bold">NBS: </span>{g.nbs_marker}<br />
                    <span className="fw-bold">Key Biomarker: </span>{g.key_biomarker}
                  </div>
                  <div className="col-md-4">
                    <span className="fw-bold">Founder Variant: </span>{g.founder_variant}<br />
                    <span className="fw-bold">Severity: </span>{g.severity_spectrum}
                  </div>
                  <div className="col-md-4">
                    <div className="mb-1">
                      {g.methionine_high ? <Badge text="Met HIGH" color={COLOR2} /> : <Badge text="Met LOW" color={COLOR3} />}
                      {g.lens_dislocation && <Badge text="Lens Disloc." color={COLOR2} />}
                      {g.retinal_disease && <Badge text="Retinal Dz PATHOGNOMONIC" color={COLOR6} />}
                      {g.combined_mma && <Badge text="Combined MMA+HC" color={COLOR4} />}
                      {g.b6_responsive_pct > 0 && <Badge text={`B6-Responsive ~${g.b6_responsive_pct}%`} color={COLOR5} />}
                    </div>
                    <div className="text-muted small">{g.alias}</div>
                  </div>
                </div>
                {/* Patient preview table */}
                {(g.patients || []).length > 0 && (
                  <div className="mt-2">
                    <div className="fw-bold small text-muted mb-1">First 10 Patients</div>
                    <div className="table-responsive">
                      <table className="table table-sm table-bordered mb-0" style={{ fontSize: '0.72rem' }}>
                        <thead style={{ backgroundColor: LIGHT }}>
                          <tr>
                            <th>ID</th><th>Age Dx (y)</th><th>Sex</th>
                            <th>tHcy (µmol/L)</th><th>Met (µmol/L)</th><th>MMA (µmol/L)</th>
                            <th>B6-Resp</th><th>Lens Disloc</th><th>Thrombosis</th>
                            <th>Retinal</th><th>Megaloblastic</th><th>Enceph.</th><th>NBS</th>
                          </tr>
                        </thead>
                        <tbody>
                          {g.patients.map(p => (
                            <tr key={p.id}>
                              <td className="fw-bold">{p.id}</td>
                              <td>{p.age_dx_y}</td>
                              <td>{p.sex}</td>
                              <td className="fw-bold" style={{ color: p.thcy_umolL > 100 ? COLOR2 : p.thcy_umolL > 30 ? COLOR4 : 'inherit' }}>{p.thcy_umolL}</td>
                              <td style={{ color: p.methionine_high ? COLOR2 : COLOR3 }}>{p.methionine_umolL}</td>
                              <td style={{ color: p.mma_umolL > 50 ? COLOR4 : 'inherit' }}>{p.mma_umolL}</td>
                              <td>{p.b6_responsive ? '✓' : '—'}</td>
                              <td>{p.lens_dislocation ? '✓' : '—'}</td>
                              <td>{p.thromboembolism ? <span style={{ color: COLOR2 }}>✓</span> : '—'}</td>
                              <td>{p.retinal_disease ? <span style={{ color: COLOR6 }}>✓</span> : '—'}</td>
                              <td>{p.megaloblastic_anemia ? '✓' : '—'}</td>
                              <td>{p.encephalopathy_at_dx ? '✓' : '—'}</td>
                              <td>{p.nbs_detected ? <span style={{ color: COLOR5 }}>✓</span> : <span style={{ color: COLOR2 }}>✗</span>}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'Definitions' && (
        <div>
          {loading.def && <div className="text-muted">Loading…</div>}
          {error.def && <div className="alert alert-danger">{error.def}</div>}
          {definitions && (
            <>
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold" style={{ background: LIGHT }}>Atlas Overview</div>
                <div className="card-body small">
                  <p>{definitions.hcu_overview?.full_name}</p>
                  <p><span className="fw-bold">Genes in Atlas: </span>{definitions.hcu_overview?.genes_in_atlas}</p>
                  <p><span className="fw-bold">Unifying Biomarker: </span>{definitions.hcu_overview?.unifying_biomarker}</p>
                  <p><span className="fw-bold">Key DDx Point: </span>{definitions.hcu_overview?.key_ddx_point}</p>
                  <p><span className="fw-bold">Central Treatment: </span>{definitions.hcu_overview?.central_treatment}</p>
                </div>
              </div>
              {(definitions.definitions || []).map((d, i) => (
                <div key={i} className="card shadow-sm mb-3">
                  <div className="card-header fw-bold" style={{ background: LIGHT }}>{d.term}</div>
                  <div className="card-body small" style={{ whiteSpace: 'pre-wrap' }}>{d.definition}</div>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
