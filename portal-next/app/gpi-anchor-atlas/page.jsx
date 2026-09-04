'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// GPI-Anchor Atlas color palette
const COLOR  = '#1a237e';  // deep indigo — GPI anchor / membrane proteins
const LIGHT  = '#e8eaf6';  // indigo tint
const COLOR2 = '#b71c1c';  // danger / severity
const COLOR3 = '#00695c';  // teal — HPMRS/ALP group
const COLOR4 = '#e65100';  // orange — MCAHS/congenital group
const COLOR5 = '#4a148c';  // purple — transamidase
const COLOR6 = '#827717';  // olive — mild (PIGG)

const GENE_COLORS = {
  PIGA:  '#c62828',  // X-linked — red (unique, most severe)
  PIGV:  '#1565c0',  // HPMRS1 — blue
  PIGL:  '#2e7d32',  // CHIME — green (ichthyosis)
  PGAP2: '#00838f',  // HPMRS3 — cyan
  PGAP3: '#4527a0',  // HPMRS4 — deep purple (most severe HPMRS)
  PIGN:  '#e65100',  // HPMRS2/MCAHS overlap — orange
  PIGT:  '#880e4f',  // MCAHS3 — pink/maroon
  PIGG:  '#558b2f',  // mild — green (sidebranch)
};

const GROUP_COLOR = {
  "GPI Synthesis (early steps)": '#1565c0',
  "Post-GPI Processing (lipid remodelling)": '#00838f',
  "GPI Transamidase (protein attachment)": '#4527a0',
  "GPI Synthesis (sidebranch, mild phenotype)": '#558b2f',
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

export default function GpiAnchorAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState({});

  useEffect(() => {
    setLoading(l => ({ ...l, overview: true }));
    fetch(`${API}/api/gpi-anchor-atlas/overview`).then(r => r.json()).then(d => { setOverview(d); setLoading(l => ({ ...l, overview: false })); }).catch(() => setLoading(l => ({ ...l, overview: false })));
  }, []);

  useEffect(() => {
    if (tab === 'Gene Table' || tab === 'Clinical Atlas') {
      if (!breakdown) {
        setLoading(l => ({ ...l, breakdown: true }));
        fetch(`${API}/api/gpi-anchor-atlas/breakdown`).then(r => r.json()).then(d => { setBreakdown(d); setLoading(l => ({ ...l, breakdown: false })); }).catch(() => setLoading(l => ({ ...l, breakdown: false })));
      }
    }
    if (tab === 'Definitions' && !definitions) {
      setLoading(l => ({ ...l, definitions: true }));
      fetch(`${API}/api/gpi-anchor-atlas/definitions`).then(r => r.json()).then(d => { setDefinitions(d); setLoading(l => ({ ...l, definitions: false })); }).catch(() => setLoading(l => ({ ...l, definitions: false })));
    }
  }, [tab]);

  const ov = overview;
  const bd = breakdown?.breakdown || [];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${COLOR} 0%, #283593 100%)`, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">🔗 GPI-Anchor Biosynthesis Disorders Atlas</h4>
        <div className="small opacity-75">Complete 8-Gene GPI-Anchor Biosynthesis Disorders Reference · PIGA/MCAHS1 · PIGV/HPMRS1 · PIGL/CHIME · PGAP2/HPMRS3 · PGAP3/HPMRS4 · PIGN · PIGT/MCAHS3 · PIGG/HPMRS6 · 320 patients (8×40, seeds 974–981)</div>
      </div>

      {/* Key alerts */}
      <Alert text="🔑 HIGH SERUM ALP IN CHILD WITH ID + SEIZURES = GPI ANCHOR DISORDER until proven otherwise — TNAP is a GPI-anchored enzyme; defects shed TNAP into serum → hyperphosphatasia" color={COLOR} />
      <Alert text="🔬 DIAGNOSTIC TEST: Flow cytometry (FLAER + CD16 + CD24) on GRANULOCYTES — lymphocytes cannot be used (they shed GPI proteins normally → false-low results)" color={COLOR3} />
      <Alert text="⚡ PIGA IS X-LINKED: the only X-linked GPI gene; germline PIGA (MCAHS1) is NOT PNH — completely different disease; somatic PIGA mutations cause PNH (adult clonal disorder)" color={COLOR2} />

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t}>
            <button className={`nav-link${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && (
        <div>
          {loading.overview && <div className="text-muted">Loading overview…</div>}
          {ov && (
            <>
              {/* KPIs */}
              <div className="row g-2 mb-3">
                <KPI label="Genes" value={ov.n_genes} color={COLOR} />
                <KPI label="Patients" value={ov.total_patients} color={COLOR} />
                <KPI label="Avg Onset" value={`${ov.avg_onset_y}y`} color={COLOR3} />
                <KPI label="Avg Dx Delay" value={`${ov.avg_dx_delay_y}y`} color={COLOR2} />
                <KPI label="Avg ALP" value={`${ov.avg_alp_fold_uln}× ULN`} color={COLOR4} />
                <KPI label="Avg FLAER" value={`${ov.avg_flaer_pct_normal}%`} color={COLOR5} />
              </div>

              <div className="row g-3 mb-3">
                {/* Severity */}
                <div className="col-md-4">
                  <div className="card shadow-sm h-100">
                    <div className="card-body">
                      <h6 className="fw-bold mb-2" style={{ color: COLOR }}>Severity Distribution</h6>
                      {Object.entries(ov.severity_distribution || {}).map(([k, v]) => (
                        <BarRow key={k} label={k} pct={Math.round((v / ov.total_patients) * 100)}
                          color={k === 'Severe' ? COLOR2 : k === 'Moderate' ? COLOR4 : COLOR3} />
                      ))}
                    </div>
                  </div>
                </div>
                {/* Gene Counts */}
                <div className="col-md-4">
                  <div className="card shadow-sm h-100">
                    <div className="card-body">
                      <h6 className="fw-bold mb-2" style={{ color: COLOR }}>Patients per Gene (40 each)</h6>
                      {Object.entries(ov.gene_counts || {}).map(([gene, cnt]) => (
                        <BarRow key={gene} label={gene} pct={Math.round((cnt / ov.total_patients) * 100 * 8)}
                          color={GENE_COLORS[gene] || COLOR} />
                      ))}
                      <div className="text-muted small mt-1">40 patients per gene (balanced cohort)</div>
                    </div>
                  </div>
                </div>
                {/* Outcome */}
                <div className="col-md-4">
                  <div className="card shadow-sm h-100">
                    <div className="card-body">
                      <h6 className="fw-bold mb-2" style={{ color: COLOR }}>Seizure Outcomes</h6>
                      {Object.entries(ov.outcome_distribution || {}).map(([k, v]) => (
                        <BarRow key={k} label={k} pct={Math.round((v / ov.total_patients) * 100)}
                          color={k.includes('free') ? COLOR3 : k.includes('resistant') || k.includes('death') ? COLOR2 : COLOR4} />
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* GPI Groups */}
              <div className="card shadow-sm mb-3">
                <div className="card-body">
                  <h6 className="fw-bold mb-2" style={{ color: COLOR }}>GPI Pathway Subgroups</h6>
                  <div className="row g-2">
                    {Object.entries(ov.gpi_groups || {}).map(([grp, genes]) => (
                      <div key={grp} className="col-md-6 col-lg-3">
                        <div className="p-2 rounded" style={{ background: (GROUP_COLOR[grp] || COLOR) + '15', borderLeft: `4px solid ${GROUP_COLOR[grp] || COLOR}` }}>
                          <div className="fw-bold small mb-1" style={{ color: GROUP_COLOR[grp] || COLOR }}>{grp}</div>
                          <div>{genes.map(g => <Badge key={g} text={g} color={GENE_COLORS[g] || COLOR} />)}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Phenotype Groups */}
              <div className="card shadow-sm mb-3">
                <div className="card-body">
                  <h6 className="fw-bold mb-2" style={{ color: COLOR }}>Clinical Phenotype Groups</h6>
                  <div className="row g-2">
                    {Object.entries(ov.phenotype_groups || {}).map(([grp, genes]) => (
                      <div key={grp} className="col-md-6">
                        <div className="p-2 rounded" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
                          <div className="fw-bold small mb-1" style={{ color: COLOR }}>{grp}</div>
                          <div>{genes.map(g => <Badge key={g} text={g} color={GENE_COLORS[g] || COLOR} />)}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Key Teaching */}
              <div className="card shadow-sm mb-3">
                <div className="card-body">
                  <h6 className="fw-bold mb-2" style={{ color: COLOR }}>Key Teaching Points</h6>
                  {(ov.key_teaching || []).map((pt, i) => (
                    <Alert key={i} text={pt} color={i % 3 === 0 ? COLOR : i % 3 === 1 ? COLOR3 : COLOR4} />
                  ))}
                </div>
              </div>

              {/* Emergency Summary */}
              <div className="card shadow-sm mb-3">
                <div className="card-body">
                  <h6 className="fw-bold mb-2" style={{ color: COLOR2 }}>Emergency Management by Gene</h6>
                  <div className="row g-2">
                    {Object.entries(ov.emergency_summary || {}).map(([gene, action]) => (
                      <div key={gene} className="col-md-6">
                        <div className="p-2 rounded" style={{ background: (GENE_COLORS[gene] || COLOR) + '15', borderLeft: `3px solid ${GENE_COLORS[gene] || COLOR}` }}>
                          <span className="badge me-2" style={{ backgroundColor: GENE_COLORS[gene] || COLOR }}>{gene}</span>
                          <span className="small">{action}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && (
        <div>
          {loading.breakdown && <div className="text-muted">Loading gene table…</div>}
          {bd.length > 0 && (
            <div className="table-responsive">
              <table className="table table-sm table-bordered table-hover small">
                <thead style={{ backgroundColor: COLOR, color: '#fff' }}>
                  <tr>
                    <th>Gene</th><th>Protein</th><th>Locus</th><th>OMIM Gene</th><th>OMIM Disease</th>
                    <th>Inheritance</th><th>GPI Subgroup</th><th>Pathway Step</th>
                    <th>N Pts</th><th>Avg Onset (y)</th><th>Avg Dx Delay (y)</th>
                    <th>Avg ALP (× ULN)</th><th>Avg FLAER (%)</th>
                    <th>Severe</th><th>Moderate</th><th>Mild</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.map(g => (
                    <tr key={g.gene}>
                      <td><strong style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</strong></td>
                      <td className="text-muted">{g.protein}</td>
                      <td>{g.locus}</td>
                      <td>{g.omim_gene}</td>
                      <td>{g.omim_disease}</td>
                      <td>{g.inheritance?.split('.')[0]}</td>
                      <td><span className="badge" style={{ backgroundColor: COLOR, fontSize: '0.65rem' }}>{g.gpi_subgroup?.split(' — ')[0]}</span></td>
                      <td className="text-muted small">{g.pathway_step?.slice(0, 50)}…</td>
                      <td className="fw-bold text-center">{g.n_patients}</td>
                      <td className="text-center">{g.avg_onset_y}</td>
                      <td className="text-center" style={{ color: g.avg_dx_delay_y > 4 ? '#b71c1c' : '#2e7d32' }}>{g.avg_dx_delay_y}y</td>
                      <td className="text-center fw-bold" style={{ color: g.avg_alp_fold_uln > 5 ? COLOR2 : g.avg_alp_fold_uln < 2.5 ? COLOR6 : COLOR4 }}>{g.avg_alp_fold_uln}×</td>
                      <td className="text-center" style={{ color: g.avg_flaer_pct_normal < 35 ? COLOR2 : COLOR3 }}>{g.avg_flaer_pct_normal}%</td>
                      <td className="text-center">{g.severity_distribution?.Severe}</td>
                      <td className="text-center">{g.severity_distribution?.Moderate}</td>
                      <td className="text-center">{g.severity_distribution?.Mild}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && (
        <div>
          {loading.breakdown && <div className="text-muted">Loading atlas…</div>}
          {bd.map(g => (
            <div key={g.gene} className="card shadow-sm mb-4">
              <div className="card-header fw-bold" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR, color: '#fff' }}>
                {g.gene} — {g.protein} &nbsp;|&nbsp; {g.locus} &nbsp;|&nbsp; OMIM {g.omim_disease} &nbsp;|&nbsp; {g.inheritance?.split('.')[0]}
                &nbsp;|&nbsp; <span className="badge bg-light text-dark">{g.gpi_subgroup?.split(' — ')[0]}</span>
              </div>
              <div className="card-body">
                <div className="row g-3">
                  <div className="col-md-6">
                    <div className="mb-2">
                      <span className="fw-bold small" style={{ color: COLOR }}>Pathway Step: </span>
                      <span className="small">{g.pathway_step}</span>
                    </div>
                    <div className="mb-2">
                      <span className="fw-bold small" style={{ color: COLOR }}>Gene Class: </span>
                      <span className="small text-muted">{g.gene_class}</span>
                    </div>
                    <div className="mb-2">
                      <span className="fw-bold small" style={{ color: COLOR }}>Phenotype: </span>
                      <span className="small">{g.phenotype_summary}</span>
                    </div>
                    <div className="mb-2">
                      <span className="fw-bold small" style={{ color: COLOR }}>Disease: </span>
                      <span className="small text-muted">{g.disease_summary}</span>
                    </div>
                  </div>
                  <div className="col-md-6">
                    <div className="mb-2 p-2 rounded" style={{ background: LIGHT }}>
                      <div className="fw-bold small mb-1" style={{ color: COLOR }}>Hallmarks</div>
                      <pre className="small mb-0" style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>{g.hallmark}</pre>
                    </div>
                    <div className="mb-2">
                      <span className="fw-bold small" style={{ color: COLOR3 }}>Key Biomarker: </span>
                      <span className="small">{g.key_biomarker}</span>
                    </div>
                    <div className="mb-2">
                      <span className="fw-bold small" style={{ color: COLOR4 }}>Severity Spectrum: </span>
                      <span className="small">{g.severity_spectrum}</span>
                    </div>
                    <div className="mb-2">
                      <span className="fw-bold small" style={{ color: COLOR2 }}>Emergency: </span>
                      <span className="small">{g.emergency}</span>
                    </div>
                    <div className="mb-2">
                      <span className="fw-bold small" style={{ color: COLOR }}>NBS: </span>
                      <span className="small text-muted">{g.nbs_marker}</span>
                    </div>
                    {/* Stats */}
                    <div className="row g-2 mt-1">
                      <div className="col-4 text-center">
                        <div className="fw-bold" style={{ color: COLOR }}>{g.avg_onset_y}y</div>
                        <div className="text-muted" style={{ fontSize: '0.7rem' }}>Avg Onset</div>
                      </div>
                      <div className="col-4 text-center">
                        <div className="fw-bold" style={{ color: g.avg_alp_fold_uln > 5 ? COLOR2 : COLOR4 }}>{g.avg_alp_fold_uln}×</div>
                        <div className="text-muted" style={{ fontSize: '0.7rem' }}>ALP fold ULN</div>
                      </div>
                      <div className="col-4 text-center">
                        <div className="fw-bold" style={{ color: g.avg_flaer_pct_normal < 35 ? COLOR2 : COLOR3 }}>{g.avg_flaer_pct_normal}%</div>
                        <div className="text-muted" style={{ fontSize: '0.7rem' }}>FLAER %</div>
                      </div>
                    </div>
                    {/* Outcome */}
                    <div className="mt-2">
                      <div className="fw-bold small mb-1" style={{ color: COLOR }}>Outcomes</div>
                      {Object.entries(g.outcome_distribution || {}).map(([k, v]) => (
                        <BarRow key={k} label={k} pct={Math.round((v / g.n_patients) * 100)}
                          color={k.includes('free') ? COLOR3 : k.includes('resistant') || k.includes('death') ? COLOR2 : COLOR4} />
                      ))}
                    </div>
                    {/* Top Treatments */}
                    <div className="mt-2">
                      <div className="fw-bold small mb-1" style={{ color: COLOR }}>Top Treatments</div>
                      {(g.top_treatments || []).map((t, i) => (
                        <div key={i} className="small text-muted">#{i + 1} {t.treatment} ({t.n} pts)</div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && (
        <div>
          {loading.definitions && <div className="text-muted">Loading definitions…</div>}
          {definitions?.definitions?.map((d, i) => (
            <div key={i} className="card shadow-sm mb-2">
              <div className="card-body py-2">
                <div className="fw-bold small mb-1" style={{ color: COLOR }}>{d.term}</div>
                <div className="small text-muted">{d.definition}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
