'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// Cardiomyopathy Atlas color palette — cardiac / heart
const COLOR  = '#b71c1c';  // cardiac red — SCD / HCM / ICD
const LIGHT  = '#ffebee';  // light red tint
const COLOR2 = '#1a237e';  // deep blue — HCM / structural
const COLOR3 = '#1b5e20';  // dark green — treatable / targeted therapy
const COLOR4 = '#e65100';  // orange — ARVC / warning
const COLOR5 = '#4a148c';  // dark purple — ARVC8/DSP / biventricular
const COLOR6 = '#37474f';  // blue-grey — TTN / DCM
const COLOR7 = '#880e4f';  // dark magenta — LMNA / critical rule

const GENE_COLORS = {
  MYH7:   '#1565c0',  // blue — thick filament HCM1
  MYBPC3: '#283593',  // dark blue — most common HCM4
  TNNT2:  '#c62828',  // red — malignant SCD risk
  PKP2:   '#e65100',  // orange — ARVC sports restriction
  DSP:    '#6a1b9a',  // purple — ARVC8 biventricular
  LMNA:   '#880e4f',  // dark magenta — DCM ICD mandatory
  TTN:    '#00695c',  // teal — DCM largest protein
  RBM20:  '#37474f',  // blue-grey — most aggressive DCM
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
      <div className="mt-2 text-muted small">Loading Cardiomyopathy Atlas…</div>
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
  const sp = ov.severity_prevalence || {};
  const cf = ov.clinical_features_prevalence || {};
  const sev = ov.severity || {};
  const cat = ov.disease_category_breakdown || {};
  const kpis = ov.kpis || [];

  return (
    <div>
      <h5 className="fw-bold mb-1" style={{ color: COLOR }}>{ov.full_name}</h5>
      <p className="text-muted small mb-3">{ov.subtitle}</p>
      <p className="mb-3">{ov.description}</p>

      {/* Drug / clinical alerts */}
      {(ov.drug_alerts || []).map((a, i) => (
        <AlertBox key={i}
          type={
            a.includes('CONTRAINDICATED') || a.includes('MANDATORY') || a.includes('ABSOLUTELY') ||
            a.includes('PATHOGNOMONIC') || a.includes('INSUFFICIENT') || a.includes('MALIGNANT')
              ? 'danger'
              : a.includes('WARNING') || a.includes('CAUTION') || a.includes('SCREEN') ? 'warning' : 'info'
          }
          title="Clinical Rule / Treatment Alert"
        >{a}</AlertBox>
      ))}

      {/* Diagnostic pearls */}
      {(ov.diagnostic_pearls || []).length > 0 && (
        <div className="card mb-3" style={{ borderLeft: `4px solid ${COLOR}` }}>
          <div className="card-body py-2 px-3">
            <h6 className="fw-bold mb-2" style={{ color: COLOR }}>❤️ Diagnostic Pearls</h6>
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

        {/* Cohort-wide clinical prevalence */}
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold small py-2" style={{ backgroundColor: LIGHT }}>
              📊 Cohort Clinical Features (% of 320 patients)
            </div>
            <div className="card-body py-2 px-3">
              {Object.entries(sp).map(([k, v], i) => (
                <BarRow key={i} label={k} pct={v} color={
                  k.includes('SCD') || k.includes('Transplant') ? COLOR
                  : k.includes('ICD') ? COLOR7
                  : k.includes('Arrhythmia') ? COLOR4
                  : k.includes('LVOTO') ? COLOR2
                  : k.includes('Error') ? COLOR
                  : k.includes('Hospitalisation') ? COLOR5
                  : COLOR6
                } />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Severity + Arrhythmia per gene */}
      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card">
            <div className="card-body py-2 px-3">
              <h6 className="fw-semibold mb-2">Severity Distribution (320 patients)</h6>
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
        <div className="col-md-6 mb-3">
          <div className="card">
            <div className="card-header fw-semibold small py-2" style={{ backgroundColor: LIGHT }}>
              💓 Arrhythmia Rate by Gene
            </div>
            <div className="card-body py-2 px-3">
              {Object.entries(cf).map(([gene, pct], i) => (
                <BarRow key={i} label={gene} pct={pct} color={GENE_COLORS[gene] || COLOR} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Tab: Gene Table ────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Array.isArray(data) ? data : [];

  return (
    <div className="table-responsive">
      <table className="table table-sm table-hover align-middle small">
        <thead className="table-dark">
          <tr>
            <th>Gene</th>
            <th>Disease Type</th>
            <th>Locus</th>
            <th>Mean Onset</th>
            <th>Mild%</th>
            <th>Severe%</th>
            <th>ICD Elig%</th>
            <th>SCD Risk%</th>
            <th>Arrh%</th>
            <th>LVOTO%</th>
            <th>Transplant%</th>
            <th>Drug-Err%</th>
            <th>1st-Line</th>
          </tr>
        </thead>
        <tbody>
          {genes.map(d => (
            <tr key={d.gene}>
              <td>
                <span className="badge px-2" style={{ backgroundColor: GENE_COLORS[d.gene] || COLOR, color: '#fff' }}>
                  {d.gene}
                </span>
              </td>
              <td style={{ maxWidth: 160 }} className="text-truncate">{d.disease_type}</td>
              <td><code className="small">{d.locus}</code></td>
              <td>{d.mean_onset_age_y} y</td>
              <td><span className="text-success fw-semibold">{d.severity_distribution?.mild_pct}%</span></td>
              <td><span className="text-danger fw-semibold">{d.severity_distribution?.severe_pct}%</span></td>
              <td><span className={d.icd_eligible_pct > 30 ? 'text-danger fw-semibold' : 'text-warning'}>{d.icd_eligible_pct}%</span></td>
              <td><span className={d.scd_risk_high_pct > 30 ? 'text-danger fw-bold' : 'text-muted'}>{d.scd_risk_high_pct}%</span></td>
              <td><span className={d.arrhythmia_pct > 40 ? 'text-warning fw-semibold' : 'text-muted'}>{d.arrhythmia_pct}%</span></td>
              <td><span className={d.lvoto_pct > 20 ? 'text-primary fw-semibold' : 'text-muted'}>{d.lvoto_pct}%</span></td>
              <td><span className={d.cardiac_transplant_pct > 10 ? 'text-danger' : 'text-muted'}>{d.cardiac_transplant_pct}%</span></td>
              <td><span className={d.drug_error_pct > 0 ? 'text-danger' : 'text-muted'}>{d.drug_error_pct}%</span></td>
              <td style={{ maxWidth: 180 }} className="text-truncate small">{d.treatment_first_line}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Tab: Clinical Atlas ────────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  const [selected, setSelected] = useState(null);
  if (!data) return <Loading />;
  const genes = Array.isArray(data) ? data : [];
  const activeGene = selected || (genes[0]?.gene);
  const d = genes.find(g => g.gene === activeGene) || {};

  return (
    <div className="row">
      {/* Gene selector */}
      <div className="col-md-3 mb-3">
        <div className="list-group list-group-flush">
          {genes.map(g => (
            <button key={g.gene}
              className={`list-group-item list-group-item-action py-2 small ${activeGene === g.gene ? 'active' : ''}`}
              style={activeGene === g.gene ? { backgroundColor: GENE_COLORS[g.gene] || COLOR, borderColor: GENE_COLORS[g.gene] || COLOR } : {}}
              onClick={() => setSelected(g.gene)}
            >
              <span className="fw-bold">{g.gene}</span>
              <div className="text-truncate opacity-75" style={{ fontSize: '0.72rem' }}>{g.disease_type}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Gene detail */}
      <div className="col-md-9">
        <div className="card">
          <div className="card-header py-2" style={{ backgroundColor: GENE_COLORS[activeGene] || COLOR, color: '#fff' }}>
            <div className="fw-bold">{d.gene} — {d.protein}</div>
            <div className="small opacity-90">{d.locus} · {d.aa}</div>
          </div>
          <div className="card-body py-3 px-3">

            {/* Key metrics */}
            <div className="row g-2 mb-3">
              {[
                { label: 'Locus', value: d.locus },
                { label: 'Mean Onset', value: `${d.mean_onset_age_y} y` },
                { label: 'ICD Eligible', value: `${d.icd_eligible_pct}%` },
                { label: 'SCD Risk High', value: `${d.scd_risk_high_pct}%` },
                { label: 'Arrhythmia', value: `${d.arrhythmia_pct}%` },
                { label: 'Drug Error', value: `${d.drug_error_pct}%` },
              ].map((m, i) => (
                <div key={i} className="col-6 col-md-4">
                  <div className="border rounded p-2 text-center h-100">
                    <div className="fw-bold small" style={{ color: GENE_COLORS[activeGene] || COLOR }}>{m.value}</div>
                    <div className="text-muted" style={{ fontSize: '0.72rem' }}>{m.label}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Critical avoid */}
            {d.critical_avoid && (
              <AlertBox type="danger" title="Critical — Avoid / Key Rule">
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

            {/* Cardiac feature bars */}
            <div className="mb-3">
              <div className="fw-semibold small mb-2">Cardiac Features</div>
              {[
                { label: 'ICD Eligible', pct: d.icd_eligible_pct, color: COLOR7 },
                { label: 'High SCD Risk', pct: d.scd_risk_high_pct, color: COLOR },
                { label: 'Arrhythmia (VT/VF/AF)', pct: d.arrhythmia_pct, color: COLOR4 },
                { label: 'LVOTO (HCM)', pct: d.lvoto_pct, color: COLOR2 },
                { label: 'HF Hospitalisation', pct: d.hf_hospitalised_pct, color: COLOR5 },
                { label: 'Cardiac Transplant', pct: d.cardiac_transplant_pct, color: COLOR },
                { label: 'Disease Progression', pct: d.progression_pct, color: COLOR4 },
                { label: 'Drug-Prescribing Error', pct: d.drug_error_pct, color: COLOR },
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

            {/* First-line treatment */}
            {d.treatment_first_line && (
              <div className="mb-3">
                <div className="fw-semibold small mb-1">First-Line Treatment</div>
                <p className="small text-muted mb-0">{d.treatment_first_line}</p>
              </div>
            )}

            {/* Key features / DDx */}
            {(d.key_features || []).length > 0 && (
              <div>
                <div className="fw-semibold small mb-2">Key Features / Differential Diagnoses</div>
                <ul className="small mb-0 ps-3">
                  {d.key_features.map((t, i) => (
                    <li key={i} className="mb-1 text-muted">{t}</li>
                  ))}
                </ul>
              </div>
            )}

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
  const defs = Array.isArray(data) ? data : (data.definitions ? Object.entries(data.definitions).map(([term, definition]) => ({ term, definition })) : []);
  const filtered = defs.filter(d =>
    !search ||
    (d.term || '').toLowerCase().includes(search.toLowerCase()) ||
    (d.definition || '').toLowerCase().includes(search.toLowerCase())
  );
  return (
    <div>
      <input className="form-control form-control-sm mb-3" placeholder="Search definitions…"
        value={search} onChange={e => setSearch(e.target.value)} />
      {filtered.map((d, i) => (
        <div key={i} className="mb-3 pb-2 border-bottom">
          <div className="fw-bold small mb-1" style={{ color: COLOR }}>{d.term}</div>
          <div className="small text-muted">{d.definition}</div>
          {d.importance && (
            <div className="small mt-1" style={{ color: COLOR4 }}>
              <span className="fw-semibold">Clinical Importance:</span> {d.importance}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────
export default function CardiomyopathyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    const base = `${API}/api/cardiomyopathy-atlas`;
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
          <h4 className="fw-bold mb-0" style={{ color: COLOR }}>❤️ Hereditary Cardiomyopathy Atlas</h4>
          <p className="text-muted small mb-0">
            Complete 8-Gene Cardiomyopathy Atlas — MYH7 · MYBPC3 · TNNT2 · PKP2 · DSP · LMNA · TTN · RBM20
          </p>
          <p className="text-muted small mb-0">
            320 patients (8×40, seeds 1102–1109) · HCM1 · HCM4 · ARVC9 · ARVC8 · DCM1A · DCM1G · DCM1HH · Mavacamten FDA 2022
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
