'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// MD-Atlas color palette — muscle fibre / connective tissue
const COLOR  = '#1b5e20';  // deep green — muscle fibres
const LIGHT  = '#e8f5e9';  // light green tint
const COLOR2 = '#b71c1c';  // red — cardiac / lethal
const COLOR3 = '#1565c0';  // blue — dystrophin
const COLOR4 = '#e65100';  // orange — warning
const COLOR5 = '#4a148c';  // dark purple — repeat expansion
const COLOR6 = '#37474f';  // blue-grey — structural
const COLOR7 = '#880e4f';  // dark magenta — severe/fatal

const GENE_COLORS = {
  DMD:    '#1565c0',  // blue — dystrophin
  DMPK:   '#4a148c',  // dark purple — repeat expansion DM1
  CNBP:   '#7b1fa2',  // purple — repeat expansion DM2
  EMD:    '#b71c1c',  // red — lethal arrhythmia EDMD1
  SMCHD1: '#2e7d32',  // dark green — FSHD2
  COL6A1: '#e65100',  // orange — collagen VI spectrum
  LAMA2:  '#00695c',  // teal — merosin CMD
  POMT1:  '#880e4f',  // dark magenta — Walker-Warburg/fatal
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

function BarRow({ label, pct, color = COLOR, note }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between mb-1">
        <span className="small fw-semibold">{label}</span>
        <span className="small text-muted">{typeof pct === 'number' ? `${pct}%` : pct}{note ? ` — ${note}` : ''}</span>
      </div>
      {typeof pct === 'number' && (
        <div className="progress" style={{ height: 8 }}>
          <div className="progress-bar" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color }} />
        </div>
      )}
    </div>
  );
}

function AlertBox({ type = 'info', title, children }) {
  const icons = { danger: '🚨', warning: '⚠️', info: 'ℹ️', success: '✅' };
  return (
    <div className={`alert alert-${type} py-2 px-3 mb-3`}>
      <strong>{icons[type]} {title}</strong>
      <div className="small mt-1">{children}</div>
    </div>
  );
}

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border" style={{ color: COLOR }} />
      <div className="mt-2 text-muted small">Loading MD-Atlas…</div>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger">{msg}</div>;
}

// ── Tab: Overview ──────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const cf = ov.clinical_features_prevalence || {};
  const sev = ov.severity || {};
  const cat = ov.disease_category_breakdown || {};
  const kpis = ov.kpis || [];

  return (
    <div>
      <h5 className="fw-bold mb-1" style={{ color: COLOR }}>{ov.full_name}</h5>
      <p className="text-muted small mb-3">{ov.subtitle}</p>
      <p className="mb-3">{ov.description}</p>

      {/* Drug alerts */}
      {(ov.drug_alerts || []).map((a, i) => (
        <AlertBox key={i}
          type={
            a.includes('ABSOLUTELY') || a.includes('MANDATORY') || a.includes('PALLIATIVE') || a.includes('LETHAL') || a.includes('CONTRAINDICATED')
              ? 'danger'
              : a.includes('AVOID') || a.includes('WARNING') ? 'warning' : 'info'
          }
          title="Clinical Rule / Treatment Alert"
        >{a}</AlertBox>
      ))}

      {/* Diagnostic pearls */}
      {(ov.diagnostic_pearls || []).length > 0 && (
        <div className="card mb-3" style={{ borderLeft: `4px solid ${COLOR}` }}>
          <div className="card-body py-2 px-3">
            <h6 className="fw-bold mb-2" style={{ color: COLOR }}>🔬 Diagnostic Pearls</h6>
            {ov.diagnostic_pearls.map((p, i) => (
              <div key={i} className="small mb-1">▸ {p}</div>
            ))}
          </div>
        </div>
      )}

      {/* KPIs */}
      <div className="row mb-3">
        {kpis.map((k, i) => <KPI key={i} label={k.label} value={k.value} color={k.color} />)}
      </div>

      <div className="row">
        {/* Disease categories */}
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold small py-2" style={{ backgroundColor: LIGHT }}>
              🧬 Disease Categories
            </div>
            <div className="card-body py-2 px-3">
              {Object.entries(cat).map(([label, genes], i) => (
                <div key={i} className="mb-2">
                  <div className="small fw-semibold mb-1">{label}</div>
                  <div className="d-flex gap-1 flex-wrap">
                    {genes.map(g => (
                      <span key={g} className="badge rounded-pill px-2 py-1 small"
                        style={{ backgroundColor: GENE_COLORS[g] || COLOR, color: '#fff' }}>
                        {g}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Clinical features */}
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold small py-2" style={{ backgroundColor: LIGHT }}>
              📊 Clinical Features (% cohort)
            </div>
            <div className="card-body py-2 px-3">
              {Object.entries(cf).map(([k, v], i) => (
                <BarRow key={i} label={k} pct={v} color={
                  k.includes('NIV') || k.includes('Respiratory') ? COLOR2
                  : k.includes('Cardiac') || k.includes('DCM') || k.includes('ICD') ? COLOR2
                  : k.includes('Steroid') || k.includes('Exon') ? COLOR3
                  : k.includes('Contracture') ? COLOR4
                  : k.includes('Scoliosis') ? COLOR5
                  : COLOR
                } />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Severity */}
      <div className="card mb-3">
        <div className="card-body py-2 px-3">
          <h6 className="fw-semibold mb-2">Severity Distribution (all 320 patients)</h6>
          <BarRow label="Mild" pct={sev.mild_pct} color="#2e7d32" />
          <BarRow label="Moderate" pct={sev.moderate_pct} color="#e65100" />
          <BarRow label="Severe" pct={sev.severe_pct} color="#b71c1c" />
          <div className="text-muted small mt-2">
            Mean onset: <strong>{ov.mean_onset_age_y} y</strong> ·
            Mean CK: <strong>{ov.mean_ck_iu_l?.toLocaleString()} IU/L</strong>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Tab: Gene Table ────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.gene_order || [];
  const breakdown = data.genes || {};

  return (
    <div className="table-responsive">
      <table className="table table-sm table-hover align-middle small">
        <thead className="table-dark">
          <tr>
            <th>Gene</th>
            <th>Disease</th>
            <th>Locus</th>
            <th>OMIM Disease</th>
            <th>Mean Onset</th>
            <th>Mean CK</th>
            <th>Mild%</th>
            <th>Severe%</th>
            <th>Cardiac%</th>
            <th>NIV%</th>
            <th>Ambulant%</th>
            <th>1st-Line</th>
          </tr>
        </thead>
        <tbody>
          {genes.map(g => {
            const d = breakdown[g] || {};
            return (
              <tr key={g}>
                <td>
                  <span className="badge px-2" style={{ backgroundColor: GENE_COLORS[g] || COLOR, color: '#fff' }}>
                    {g}
                  </span>
                </td>
                <td style={{ maxWidth: 180 }}>{d.disease_type}</td>
                <td><code className="small">{d.locus}</code></td>
                <td>
                  <a href={`https://omim.org/entry/${d.omim_disease}`} target="_blank" rel="noreferrer"
                    className="text-decoration-none small">#{d.omim_disease}</a>
                </td>
                <td>{d.mean_onset_age_y} y</td>
                <td>{d.mean_ck_iu_l?.toLocaleString()}</td>
                <td><span className="text-success fw-semibold">{d.severity_distribution?.mild_pct}%</span></td>
                <td><span className="text-danger fw-semibold">{d.severity_distribution?.severe_pct}%</span></td>
                <td>{d.cardiac_pct}%</td>
                <td>{d.niv_pct}%</td>
                <td>{d.ambulant_pct}%</td>
                <td style={{ maxWidth: 140 }} className="text-truncate">{d.first_line_drug}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Tab: Clinical Atlas ────────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  const [selected, setSelected] = useState(null);
  if (!data) return <Loading />;
  const genes = data.gene_order || [];
  const breakdown = data.genes || {};
  const active = selected || genes[0];
  const d = breakdown[active] || {};

  return (
    <div className="row">
      {/* Gene selector */}
      <div className="col-md-3 mb-3">
        <div className="list-group list-group-flush">
          {genes.map(g => (
            <button key={g}
              className={`list-group-item list-group-item-action py-2 small ${active === g ? 'active' : ''}`}
              style={active === g ? { backgroundColor: GENE_COLORS[g] || COLOR, borderColor: GENE_COLORS[g] || COLOR } : {}}
              onClick={() => setSelected(g)}
            >
              <span className="fw-bold">{g}</span>
              <div className="text-truncate opacity-75" style={{ fontSize: '0.72rem' }}>{d.disease_type}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Gene detail */}
      <div className="col-md-9">
        <div className="card">
          <div className="card-header py-2" style={{ backgroundColor: GENE_COLORS[active] || COLOR, color: '#fff' }}>
            <div className="fw-bold">{active} — {d.protein}</div>
            <div className="small opacity-90">{d.alias}</div>
          </div>
          <div className="card-body py-3 px-3">

            {/* Key metrics */}
            <div className="row g-2 mb-3">
              {[
                { label: 'Locus', value: d.locus },
                { label: 'OMIM Gene', value: `#${d.omim_gene}` },
                { label: 'OMIM Disease', value: `#${d.omim_disease}` },
                { label: 'Mean Onset', value: `${d.mean_onset_age_y} y` },
                { label: 'Mean CK', value: `${d.mean_ck_iu_l?.toLocaleString()} IU/L` },
                { label: 'Ambulant', value: `${d.ambulant_pct}%` },
              ].map((m, i) => (
                <div key={i} className="col-6 col-md-4">
                  <div className="border rounded p-2 text-center h-100">
                    <div className="fw-bold small" style={{ color: GENE_COLORS[active] || COLOR }}>{m.value}</div>
                    <div className="text-muted" style={{ fontSize: '0.72rem' }}>{m.label}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Drug alert */}
            {d.critical_avoid && (
              <AlertBox type="danger" title="Critical — Avoid / Monitor">
                {d.critical_avoid}
              </AlertBox>
            )}

            {/* Severity bars */}
            <div className="mb-3">
              <div className="fw-semibold small mb-2">Severity Distribution ({d.n_patients} patients)</div>
              <BarRow label="Mild" pct={d.severity_distribution?.mild_pct} color="#2e7d32" />
              <BarRow label="Moderate" pct={d.severity_distribution?.moderate_pct} color="#e65100" />
              <BarRow label="Severe" pct={d.severity_distribution?.severe_pct} color="#b71c1c" />
            </div>

            {/* Clinical bars */}
            <div className="mb-3">
              <div className="fw-semibold small mb-2">Clinical Features</div>
              {[
                { label: 'Cardiac Involvement', pct: d.cardiac_pct, color: COLOR2 },
                { label: 'NIV Required', pct: d.niv_pct, color: COLOR4 },
                { label: 'Contractures', pct: d.contracture_pct, color: COLOR5 },
                { label: 'Ambulant', pct: d.ambulant_pct, color: '#2e7d32' },
              ].map((r, i) => <BarRow key={i} {...r} />)}
            </div>

            {/* Inheritance */}
            <div className="mb-3">
              <div className="fw-semibold small mb-1">Inheritance</div>
              <p className="small text-muted mb-0">{d.inheritance}</p>
            </div>

            {/* Phenotype */}
            <div className="mb-3">
              <div className="fw-semibold small mb-1">Phenotype</div>
              <p className="small text-muted mb-0">{d.phenotype}</p>
            </div>

            {/* Mechanism */}
            {d.mechanism && (
              <div className="mb-3">
                <div className="fw-semibold small mb-1">Molecular Mechanism</div>
                <p className="small text-muted mb-0">{d.mechanism}</p>
              </div>
            )}

            {/* Treatment */}
            <div className="mb-3">
              <div className="fw-semibold small mb-2">Treatment Options</div>
              <ul className="small mb-0 ps-3">
                {(d.treatment_options || []).map((t, i) => (
                  <li key={i} className="mb-1 text-muted">{t}</li>
                ))}
              </ul>
            </div>

            {/* DDx */}
            <div>
              <div className="fw-semibold small mb-2">Key Differential Diagnoses</div>
              <ul className="small mb-0 ps-3">
                {(d.key_ddx || []).map((t, i) => (
                  <li key={i} className="mb-1 text-muted">{t}</li>
                ))}
              </ul>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}

// ── Tab: Definitions ───────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  const [search, setSearch] = useState('');
  if (!data) return <Loading />;
  const defs = data.definitions || {};
  const filtered = Object.entries(defs).filter(([k, v]) =>
    !search || k.toLowerCase().includes(search.toLowerCase()) || v.toLowerCase().includes(search.toLowerCase())
  );
  return (
    <div>
      <input className="form-control form-control-sm mb-3" placeholder="Search definitions…"
        value={search} onChange={e => setSearch(e.target.value)} />
      <p className="text-muted small mb-3">{data.subtitle}</p>
      {filtered.map(([term, def], i) => (
        <div key={i} className="mb-3 pb-2 border-bottom">
          <div className="fw-bold small mb-1" style={{ color: COLOR }}>{term}</div>
          <div className="small text-muted">{def}</div>
        </div>
      ))}
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────
export default function MDAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    const base = `${API}/api/md-atlas`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setErr(e.message));
  }, []);

  if (err) return (
    <div className="container-fluid py-3">
      <ErrorMsg msg={err} />
    </div>
  );

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3">
        <div>
          <h4 className="fw-bold mb-0" style={{ color: COLOR }}>💪 MD-Atlas</h4>
          <p className="text-muted small mb-0">
            Complete 8-Gene Muscular Dystrophy Atlas — DMD · DMPK · CNBP · EMD · SMCHD1 · COL6A1 · LAMA2 · POMT1
          </p>
          <p className="text-muted small mb-0">
            320 patients (8×40, seeds 1062–1069) · Duchenne/Becker · DM1 · DM2 · EDMD1 · FSHD2 · Bethlem/Ullrich · MDC1A · Walker-Warburg
          </p>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active fw-semibold' : ''}`}
              style={tab === t ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(t)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* Tab content */}
      {tab === 'Overview'      && <OverviewTab data={overview} />}
      {tab === 'Gene Table'    && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'   && <DefinitionsTab data={definitions} />}
    </div>
  );
}
