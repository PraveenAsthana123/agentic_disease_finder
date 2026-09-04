'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// Vitamin-Atlas color palette
const COLOR  = '#1b5e20';  // deep green — vitamins / treatable
const LIGHT  = '#e8f5e9';  // green tint
const COLOR2 = '#b71c1c';  // danger / emergency
const COLOR3 = '#1565c0';  // B6 group
const COLOR4 = '#e65100';  // B1 group
const COLOR5 = '#4a148c';  // B7 group
const COLOR6 = '#0277bd';  // B2 group

const GENE_COLORS = {
  ALDH7A1: '#1565c0',  // PDE — B6/pyridoxine — blue
  PNPO:    '#283593',  // PLP-neonatal — B6/PLP — deep blue
  PLPBP:   '#0d47a1',  // PLP homeostasis — B6 — navy
  TPK1:    '#e65100',  // thiamine — B1 — orange
  SLC19A3: '#bf360c',  // BTBGD — B1+B7 — deep orange
  BTD:     '#4a148c',  // biotinidase — B7 — purple
  HLCS:    '#6a1b9a',  // MCD — B7 — violet
  SLC52A2: '#0277bd',  // BVVL — B2 — teal-blue
};

const VIT_LABEL = {
  ALDH7A1: 'B6', PNPO: 'B6', PLPBP: 'B6',
  TPK1: 'B1', SLC19A3: 'B1+B7',
  BTD: 'B7', HLCS: 'B7',
  SLC52A2: 'B2',
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

export default function VitaminAtlasPage() {
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
      .catch(e => { setError(er => ({ ...er, [key]: e.message })); setLoading(l => ({ ...l, [key]: false })); });
  }

  useEffect(() => { load('/api/vitamin-atlas/overview', setOverview, 'ov'); }, []);
  useEffect(() => {
    if (tab === 'Gene Table' && !breakdown) load('/api/vitamin-atlas/breakdown', setBreakdown, 'bd');
    if (tab === 'Clinical Atlas' && !breakdown) load('/api/vitamin-atlas/breakdown', setBreakdown, 'bd');
    if (tab === 'Definitions' && !definitions) load('/api/vitamin-atlas/definitions', setDefinitions, 'df');
  }, [tab]);

  const spin = <div className="d-flex justify-content-center py-5"><div className="spinner-border" style={{ color: COLOR }} /></div>;

  // ── Overview Tab ────────────────────────────────────────────────────────────
  function TabOverview() {
    if (loading.ov) return spin;
    if (!overview) return null;
    const ov = overview;
    const sevTotal = Object.values(ov.severity_distribution).reduce((a, b) => a + b, 0);

    return (
      <div>
        {/* Title */}
        <div className="p-3 mb-4 rounded-3" style={{ background: `linear-gradient(135deg, ${COLOR} 0%, #2e7d32 100%)`, color: '#fff' }}>
          <h4 className="fw-bold mb-1">🌿 {ov.atlas}</h4>
          <p className="mb-0 small opacity-75">{ov.subtitle}</p>
          <p className="mb-0 mt-1 small">{ov.description}</p>
        </div>

        {/* KPI row */}
        <div className="row g-2 mb-4">
          <KPI label="Genes Covered" value={ov.n_genes} color={COLOR} />
          <KPI label="Total Patients" value={ov.total_patients} color={COLOR} />
          <KPI label="Avg Onset (y)" value={ov.avg_onset_y} color={COLOR3} />
          <KPI label="Avg Dx Delay (y)" value={ov.avg_dx_delay_y} color={COLOR4} />
          <KPI label="Severe Cases" value={`${Math.round(100 * ov.severity_distribution.Severe / sevTotal)}%`} color={COLOR2} />
          <KPI label="Seeds" value={ov.seeds_used} color="#555" />
        </div>

        {/* Emergency summary */}
        <div className="card mb-4 border-danger">
          <div className="card-header fw-bold" style={{ background: COLOR2, color: '#fff' }}>
            ⚡ EMERGENCY VITAMIN PROTOCOLS — TREAT BEFORE DIAGNOSIS
          </div>
          <div className="card-body py-2">
            {Object.entries(ov.emergency_summary).map(([gene, protocol]) => (
              <div key={gene} className="mb-2 p-2 rounded small" style={{ background: GENE_COLORS[gene] + '15', borderLeft: `3px solid ${GENE_COLORS[gene]}` }}>
                <span className="badge me-2 fw-bold" style={{ backgroundColor: GENE_COLORS[gene] }}>{gene}</span>
                <span className="badge me-2" style={{ backgroundColor: '#777' }}>{VIT_LABEL[gene]}</span>
                <span>{protocol}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="row g-3 mb-4">
          {/* Vitamin groups */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold" style={{ color: COLOR }}>💊 Vitamin Groups</div>
              <div className="card-body">
                {Object.entries(ov.vitamin_groups).map(([vit, genes]) => (
                  <div key={vit} className="mb-2">
                    <div className="small fw-bold text-muted mb-1">{vit}</div>
                    <div>{genes.map(g => (
                      <Badge key={g} text={g} color={GENE_COLORS[g]} />
                    ))}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Severity */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold" style={{ color: COLOR }}>📊 Severity Distribution</div>
              <div className="card-body">
                {Object.entries(ov.severity_distribution).map(([sev, n]) => {
                  const pct = Math.round(100 * n / sevTotal);
                  const c = sev === 'Severe' ? COLOR2 : sev === 'Moderate' ? COLOR4 : COLOR;
                  return <BarRow key={sev} label={`${sev} (n=${n})`} pct={pct} color={c} />;
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Key teaching points */}
        <div className="card mb-4">
          <div className="card-header fw-bold" style={{ color: COLOR }}>🎓 Key Teaching Points</div>
          <div className="card-body py-2">
            {(ov.key_teaching || []).map((t, i) => (
              <Alert key={i} text={t} color={i < 3 ? COLOR2 : i < 6 ? COLOR4 : COLOR} />
            ))}
          </div>
        </div>

        {/* Gene patient counts */}
        <div className="card mb-4">
          <div className="card-header fw-bold" style={{ color: COLOR }}>🧬 Patients per Gene</div>
          <div className="card-body">
            {Object.entries(ov.gene_counts).map(([gene, n]) => (
              <BarRow key={gene} label={`${gene} (n=${n})`} pct={Math.round(100 * n / ov.total_patients)} color={GENE_COLORS[gene]} />
            ))}
          </div>
        </div>
      </div>
    );
  }

  // ── Gene Table Tab ──────────────────────────────────────────────────────────
  function TabGeneTable() {
    if (loading.bd) return spin;
    if (!breakdown) return null;
    const genes = breakdown.breakdown || [];
    return (
      <div>
        <h5 className="fw-bold mb-3" style={{ color: COLOR }}>Per-Gene Summary Table</h5>
        <div className="table-responsive">
          <table className="table table-sm table-hover small align-middle">
            <thead style={{ backgroundColor: COLOR, color: '#fff' }}>
              <tr>
                <th>Gene</th>
                <th>Vitamin</th>
                <th>Protein</th>
                <th>Locus</th>
                <th>OMIM Disease</th>
                <th>Inheritance</th>
                <th>Pts</th>
                <th>Avg Onset (y)</th>
                <th>Avg Delay (y)</th>
                <th>Severe %</th>
                <th>NBS</th>
                <th>Key Biomarker</th>
              </tr>
            </thead>
            <tbody>
              {genes.map(g => {
                const sevPct = Math.round(100 * (g.severity_distribution?.Severe || 0) / g.n_patients);
                return (
                  <tr key={g.gene}>
                    <td><Badge text={g.gene} color={GENE_COLORS[g.gene]} /></td>
                    <td><span className="badge" style={{ backgroundColor: '#555' }}>{VIT_LABEL[g.gene]}</span></td>
                    <td className="text-muted">{g.protein}</td>
                    <td><code>{g.locus}</code></td>
                    <td><a href={`https://omim.org/entry/${g.omim_disease}`} target="_blank" rel="noopener noreferrer" className="text-decoration-none">#{g.omim_disease}</a></td>
                    <td>{g.inheritance?.split('.')[0]}</td>
                    <td>{g.n_patients}</td>
                    <td>{g.avg_onset_y}</td>
                    <td>{g.avg_dx_delay_y}</td>
                    <td><span className={`badge bg-${sevPct > 40 ? 'danger' : sevPct > 20 ? 'warning text-dark' : 'success'}`}>{sevPct}%</span></td>
                    <td className="text-muted" style={{ maxWidth: 120 }}>{g.nbs_marker?.substring(0, 40)}…</td>
                    <td className="text-muted" style={{ maxWidth: 160 }}>{g.key_biomarker?.substring(0, 60)}…</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // ── Clinical Atlas Tab ──────────────────────────────────────────────────────
  function TabClinicalAtlas() {
    if (loading.bd) return spin;
    if (!breakdown) return null;
    const genes = breakdown.breakdown || [];
    return (
      <div>
        <h5 className="fw-bold mb-3" style={{ color: COLOR }}>Clinical Atlas — Per-Gene Deep Dive</h5>
        <div className="row g-3">
          {genes.map(g => (
            <div key={g.gene} className="col-12">
              <div className="card shadow-sm" style={{ borderLeft: `5px solid ${GENE_COLORS[g.gene]}` }}>
                <div className="card-header d-flex align-items-center gap-2 flex-wrap" style={{ backgroundColor: GENE_COLORS[g.gene] + '15' }}>
                  <Badge text={g.gene} color={GENE_COLORS[g.gene]} />
                  <Badge text={VIT_LABEL[g.gene]} color="#555" />
                  <span className="fw-bold">{g.protein}</span>
                  <span className="ms-auto text-muted small"><code>{g.locus}</code> · {g.aa} · OMIM #{g.omim_disease}</span>
                </div>
                <div className="card-body">
                  <div className="row g-2 mb-3">
                    <div className="col-md-6 col-lg-3">
                      <div className="small fw-bold text-muted mb-1">Subgroup</div>
                      <div className="small">{g.vit_subgroup}</div>
                    </div>
                    <div className="col-md-6 col-lg-3">
                      <div className="small fw-bold text-muted mb-1">Vitamin Treatment</div>
                      <div className="small" style={{ color: GENE_COLORS[g.gene] }}>{g.vitamin}</div>
                    </div>
                    <div className="col-md-6 col-lg-3">
                      <div className="small fw-bold text-muted mb-1">Avg Onset / Dx Delay</div>
                      <div className="small">{g.avg_onset_y}y onset · {g.avg_dx_delay_y}y delay</div>
                    </div>
                    <div className="col-md-6 col-lg-3">
                      <div className="small fw-bold text-muted mb-1">Inheritance</div>
                      <div className="small">{g.inheritance?.split('.')[0]}</div>
                    </div>
                  </div>

                  {/* Hallmark */}
                  <div className="mb-2">
                    <div className="small fw-bold mb-1" style={{ color: GENE_COLORS[g.gene] }}>⚡ Hallmarks</div>
                    {(g.hallmark || '').split('\n').filter(Boolean).map((h, i) => (
                      <Alert key={i} text={h} color={GENE_COLORS[g.gene]} />
                    ))}
                  </div>

                  {/* Emergency */}
                  <div className="mb-2 p-2 rounded small fw-bold" style={{ background: COLOR2 + '18', borderLeft: `3px solid ${COLOR2}`, color: COLOR2 }}>
                    ⚡ Emergency: {g.emergency}
                  </div>

                  {/* Biomarker + NBS */}
                  <div className="row g-2 mb-2">
                    <div className="col-md-6">
                      <div className="small fw-bold text-muted mb-1">Key Biomarker</div>
                      <div className="small">{g.key_biomarker}</div>
                    </div>
                    <div className="col-md-6">
                      <div className="small fw-bold text-muted mb-1">NBS</div>
                      <div className="small">{g.nbs_marker}</div>
                    </div>
                  </div>

                  {/* Treatments */}
                  <div className="mb-2">
                    <div className="small fw-bold text-muted mb-1">Top Treatments</div>
                    <div>{(g.top_treatments || []).map((t, i) => (
                      <Badge key={i} text={`${t.treatment} (n=${t.n})`} color={GENE_COLORS[g.gene]} />
                    ))}</div>
                  </div>

                  {/* CI drugs */}
                  {g.ci_drugs?.length > 0 && (
                    <div className="mb-2">
                      <div className="small fw-bold text-muted mb-1">⛔ Contraindicated / Dangerous</div>
                      <div>{g.ci_drugs.map((d, i) => <Badge key={i} text={d} color={COLOR2} />)}</div>
                    </div>
                  )}

                  {/* Disease summary */}
                  <details className="mt-2">
                    <summary className="small text-muted" style={{ cursor: 'pointer' }}>Disease / Phenotype summary…</summary>
                    <div className="mt-2 small text-muted" style={{ whiteSpace: 'pre-wrap' }}>{g.disease_summary}</div>
                    <div className="mt-1 small text-muted">{g.phenotype_summary}</div>
                  </details>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ── Definitions Tab ─────────────────────────────────────────────────────────
  function TabDefinitions() {
    if (loading.df) return spin;
    if (!definitions) return null;
    const defs = definitions.definitions || [];
    return (
      <div>
        <h5 className="fw-bold mb-3" style={{ color: COLOR }}>Clinical Definitions ({defs.length} terms)</h5>
        <div className="accordion" id="defAccordion">
          {defs.map((d, i) => (
            <div key={i} className="accordion-item">
              <h2 className="accordion-header">
                <button className="accordion-button collapsed fw-bold small py-2" type="button"
                  data-bs-toggle="collapse" data-bs-target={`#def${i}`}>
                  {d.term}
                </button>
              </h2>
              <div id={`def${i}`} className="accordion-collapse collapse" data-bs-parent="#defAccordion">
                <div className="accordion-body small">{d.definition}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="container-fluid py-3">
      {/* Page header */}
      <div className="mb-3">
        <h3 className="fw-bold" style={{ color: COLOR }}>
          🌿 Vitamin-Responsive Epileptic Encephalopathies Atlas
        </h3>
        <p className="text-muted small mb-0">
          8-gene vitamin-responsive atlas · B6 (ALDH7A1/PNPO/PLPBP) · B1 (TPK1/SLC19A3) · B7 (BTD/HLCS) · B2 (SLC52A2/BVVL) · 320 patients (8×40, seeds 966–973)
        </p>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active fw-bold' : ''}`}
              style={tab === t ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(t)}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* Error display */}
      {Object.entries(error).map(([k, v]) => (
        <div key={k} className="alert alert-danger small py-2">{v}</div>
      ))}

      {/* Tab content */}
      {tab === 'Overview' && <TabOverview />}
      {tab === 'Gene Table' && <TabGeneTable />}
      {tab === 'Clinical Atlas' && <TabClinicalAtlas />}
      {tab === 'Definitions' && <TabDefinitions />}
    </div>
  );
}
