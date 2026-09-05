'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// DEE-Atlas color palette — EEG / epilepsy / neural
const COLOR  = '#1a237e';  // deep indigo — neural/brain
const LIGHT  = '#e8eaf6';  // light indigo tint
const COLOR2 = '#b71c1c';  // red — SUDEP / lethal
const COLOR3 = '#1565c0';  // blue — targeted therapy
const COLOR4 = '#e65100';  // orange — warning / avoid
const COLOR5 = '#4a148c';  // dark purple — X-linked
const COLOR6 = '#37474f';  // blue-grey — structural
const COLOR7 = '#880e4f';  // dark magenta — severe/fatal

const GENE_COLORS = {
  SCN1A:  '#b71c1c',  // red — Dravet/LOF Nav1.1 (AVOID Na blockers)
  KCNQ2:  '#1565c0',  // blue — KCNQ2-DEE/BFNS neonatal
  CDKL5:  '#7b1fa2',  // purple — X-linked CDD spasms
  ARX:    '#4a148c',  // dark purple — X-linked males EIEE
  STXBP1: '#e65100',  // orange — Ohtahara→West→LGS
  PCDH19: '#880e4f',  // dark magenta — PCDH19 females only
  SCN8A:  '#c62828',  // dark red — GOF Nav1.6 high SUDEP
  GRIN2A: '#2e7d32',  // green — epilepsy-aphasia spectrum (milder)
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
      <div className="mt-2 text-muted small">Loading DEE-Atlas…</div>
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
            a.includes('ABSOLUTELY') || a.includes('MANDATORY') || a.includes('AVOID') || a.includes('CONTRAINDICATED') || a.includes('SUDEP')
              ? 'danger'
              : a.includes('WARNING') || a.includes('CAUTION') ? 'warning' : 'info'
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
                  k.includes('SUDEP') ? COLOR2
                  : k.includes('Status') ? COLOR4
                  : k.includes('Seizure-Free') ? '#2e7d32'
                  : k.includes('Drug') ? COLOR7
                  : k.includes('Targeted') ? COLOR3
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
            Mean age at diagnosis: <strong>{ov.mean_diagnosis_age_y} y</strong>
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
            <th>Mild%</th>
            <th>Severe%</th>
            <th>Seizure-Free%</th>
            <th>Status-Hx%</th>
            <th>SUDEP-High%</th>
            <th>Drug-Err%</th>
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
                <td><span className="text-success fw-semibold">{d.severity_distribution?.mild_pct}%</span></td>
                <td><span className="text-danger fw-semibold">{d.severity_distribution?.severe_pct}%</span></td>
                <td><span className="text-success">{d.seizure_freedom_pct}%</span></td>
                <td><span className="text-warning fw-semibold">{d.status_hx_pct}%</span></td>
                <td><span className="text-danger fw-semibold">{d.sudep_risk_pct}%</span></td>
                <td><span className={d.drug_error_pct > 0 ? 'text-danger' : 'text-muted'}>{d.drug_error_pct}%</span></td>
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
              <div className="text-truncate opacity-75" style={{ fontSize: '0.72rem' }}>{breakdown[g]?.disease_type}</div>
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
                { label: 'Seizure-Free', value: `${d.seizure_freedom_pct}%` },
                { label: 'SUDEP Risk High', value: `${d.sudep_risk_pct}%` },
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
              <AlertBox type="danger" title="Critical — Avoid / Caution">
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

            {/* DEE-specific clinical bars */}
            <div className="mb-3">
              <div className="fw-semibold small mb-2">Clinical Features</div>
              {[
                { label: 'Seizure-Free (current)', pct: d.seizure_freedom_pct, color: '#2e7d32' },
                { label: 'Status Epilepticus Hx', pct: d.status_hx_pct, color: COLOR4 },
                { label: 'SUDEP Risk — High', pct: d.sudep_risk_pct, color: COLOR2 },
                { label: 'Drug-Avoid Error Detected', pct: d.drug_error_pct, color: COLOR7 },
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
export default function DEEAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    const base = `${API}/api/dee-atlas`;
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
          <h4 className="fw-bold mb-0" style={{ color: COLOR }}>⚡ DEE-Atlas</h4>
          <p className="text-muted small mb-0">
            Complete 8-Gene Developmental &amp; Epileptic Encephalopathy Atlas — SCN1A · KCNQ2 · CDKL5 · ARX · STXBP1 · PCDH19 · SCN8A · GRIN2A
          </p>
          <p className="text-muted small mb-0">
            320 patients (8×40, seeds 1070–1077) · Dravet · KCNQ2-DEE/BFNS · CDD · EIEE1/ARX · Ohtahara/STXBP1 · PCDH19-Clustering · SCN8A-DEE · Epilepsy-Aphasia/GRIN2A
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
      {tab === 'Overview'       && <OverviewTab data={overview} />}
      {tab === 'Gene Table'     && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'    && <DefinitionsTab data={definitions} />}
    </div>
  );
}
