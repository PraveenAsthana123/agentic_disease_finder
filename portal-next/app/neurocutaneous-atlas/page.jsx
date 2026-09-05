'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  NF1:     '#1b5e20',
  NF2:     '#4a148c',
  TSC1:    '#e65100',
  TSC2:    '#bf360c',
  VHL:     '#0d47a1',
  PTEN:    '#880e4f',
  PTCH1:   '#c62828',
  SMARCB1: '#37474f',
};

/* ── small helpers ─────────────────────────────────────────────────────── */
function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-success" role="status" />
      <p className="mt-3 text-muted">Loading neurocutaneous atlas…</p>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return (
    <div className="alert alert-danger m-4">
      <strong>Error:</strong> {msg}
    </div>
  );
}

function KPI({ label, value, sub, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm border-0">
        <div className="card-body text-center p-3">
          <div className="fw-bold fs-3" style={{ color: color || '#1b5e20' }}>
            {value}
          </div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function BarRow({ label, pct, color }) {
  const p = Math.min(100, Math.max(0, Math.round(pct || 0)));
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-bold">{p}%</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div
          className="progress-bar"
          style={{ width: `${p}%`, backgroundColor: color || '#1b5e20' }}
        />
      </div>
    </div>
  );
}

function AlertBox({ variant, title, children }) {
  return (
    <div className={`alert alert-${variant} mb-3`}>
      <strong>{title}</strong> {children}
    </div>
  );
}

/* ── Overview tab ─────────────────────────────────────────────────────── */
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const kpis = ov.kpis || {};
  const agg  = ov.aggregate_clinical || {};
  const alerts = ov.drug_alerts || [];
  const rules  = ov.critical_rules || [];
  const pathways = ov.pathway_targets || {};

  return (
    <div>
      {/* Hero */}
      <div className="rounded-3 p-4 mb-4 text-white" style={{ background: 'linear-gradient(135deg,#1b5e20,#4a148c)' }}>
        <h3 className="mb-1">{ov.atlas_name || 'Neurocutaneous Syndromes Atlas'}</h3>
        <p className="mb-1 opacity-75">{ov.atlas_subtitle}</p>
        <div className="d-flex flex-wrap gap-3 mt-3">
          <span className="badge bg-white text-dark fs-6">{ov.n_genes} Genes</span>
          <span className="badge bg-white text-dark fs-6">{ov.n_patients} Patients</span>
          <span className="badge bg-white text-dark fs-6">Seeds {ov.seeds?.[0]}–{ov.seeds?.[(ov.seeds?.length||1)-1]}</span>
        </div>
      </div>

      {/* Critical Alerts */}
      <h5 className="mb-3">Critical Clinical Alerts</h5>
      {alerts.map((a, i) => (
        <AlertBox key={i} variant={a.severity === 'CRITICAL' ? 'danger' : a.severity === 'HIGH' ? 'warning' : 'info'} title={`[${a.gene}] ${a.type}:`}>
          {a.message}
        </AlertBox>
      ))}

      {/* KPIs */}
      <h5 className="mt-4 mb-3">Aggregate Cohort KPIs (n={ov.n_patients})</h5>
      <div className="row">
        <KPI label="Epilepsy"             value={`${kpis.epilepsy_pct ?? '—'}%`}              color="#e65100" />
        <KPI label="CNS Tumor"            value={`${kpis.cns_tumor_pct ?? '—'}%`}             color="#4a148c" />
        <KPI label="Malignancy"           value={`${kpis.malignancy_pct ?? '—'}%`}            color="#c62828" />
        <KPI label="Skin Lesion"          value={`${kpis.skin_lesion_pct ?? '—'}%`}           color="#1b5e20" />
        <KPI label="Learning Disability"  value={`${kpis.learning_disability_pct ?? '—'}%`}  color="#0d47a1" />
        <KPI label="De Novo Variant"      value={`${kpis.de_novo_pct ?? '—'}%`}              color="#880e4f" />
      </div>

      {/* Aggregate bars */}
      <div className="row mt-2">
        <div className="col-md-6">
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: '#e8f5e9' }}>Clinical Feature Rates</div>
            <div className="card-body">
              <BarRow label="Epilepsy"             pct={agg.epilepsy_pct}            color="#e65100" />
              <BarRow label="CNS Tumor"            pct={agg.cns_tumor_pct}           color="#4a148c" />
              <BarRow label="Skin Lesion"          pct={agg.skin_lesion_pct}         color="#1b5e20" />
              <BarRow label="Learning Disability"  pct={agg.learning_disability_pct} color="#0d47a1" />
              <BarRow label="Malignancy"           pct={agg.malignancy_pct}          color="#c62828" />
              <BarRow label="De Novo Variant"      pct={agg.de_novo_pct}            color="#880e4f" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: '#e8f5e9' }}>Pathway Targets</div>
            <div className="card-body">
              {Object.entries(pathways).map(([gene, pathway]) => (
                <div key={gene} className="d-flex align-items-center mb-2">
                  <span
                    className="badge me-2"
                    style={{ background: GENE_COLORS[gene] || '#6c757d', minWidth: 70, fontSize: '0.75rem' }}
                  >
                    {gene}
                  </span>
                  <small className="text-muted">{pathway}</small>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Critical Rules */}
      {rules.length > 0 && (
        <div className="card border-danger border-2 mb-3">
          <div className="card-header bg-danger text-white fw-bold">Critical Rules — Must Know</div>
          <div className="card-body">
            <ul className="mb-0">
              {rules.map((r, i) => <li key={i} className="mb-1 small">{r}</li>)}
            </ul>
          </div>
        </div>
      )}

      {/* Description */}
      {ov.description && (
        <div className="alert alert-light border mt-2">
          <small className="text-muted">{ov.description}</small>
        </div>
      )}
    </div>
  );
}

/* ── Gene Table tab ───────────────────────────────────────────────────── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || [];

  return (
    <div>
      <h5 className="mb-3">Per-Gene Cohort Breakdown (40 patients each)</h5>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th>
              <th>Disease</th>
              <th>Inheritance</th>
              <th>n</th>
              <th>Epilepsy %</th>
              <th>CNS Tumor %</th>
              <th>Malignancy %</th>
              <th>Skin Lesion %</th>
              <th>De Novo %</th>
              <th>Drug Errors</th>
            </tr>
          </thead>
          <tbody>
            {genes.map(g => (
              <tr key={g.gene}>
                <td>
                  <span
                    className="badge"
                    style={{ background: GENE_COLORS[g.gene] || '#6c757d', fontSize: '0.85rem' }}
                  >
                    {g.gene}
                  </span>
                </td>
                <td className="small">{g.disease}</td>
                <td><span className="badge bg-secondary">{g.inheritance}</span></td>
                <td>{g.n_patients}</td>
                <td>{g.epilepsy_pct ?? '—'}%</td>
                <td>{g.cns_tumor_pct ?? '—'}%</td>
                <td>{g.malignancy_pct ?? '—'}%</td>
                <td>{g.skin_lesion_pct ?? '—'}%</td>
                <td>{g.de_novo_pct ?? '—'}%</td>
                <td>
                  {g.drug_errors > 0
                    ? <span className="badge bg-danger">{g.drug_errors}</span>
                    : <span className="badge bg-success">0</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Clinical Atlas tab ───────────────────────────────────────────────── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || [];

  return (
    <div>
      <h5 className="mb-3">Clinical Atlas — Gene Cards</h5>
      <div className="row">
        {genes.map(g => {
          const color = GENE_COLORS[g.gene] || '#6c757d';
          return (
            <div key={g.gene} className="col-md-6 col-xl-4 mb-4">
              <div className="card h-100 shadow-sm border-0" style={{ borderTop: `4px solid ${color}` }}>
                <div className="card-header text-white fw-bold" style={{ background: color }}>
                  {g.gene} — {g.disease}
                </div>
                <div className="card-body p-3">
                  <div className="mb-2">
                    <span className="badge bg-secondary me-1">{g.inheritance}</span>
                    <span className="badge bg-light text-dark">{g.locus}</span>
                  </div>
                  <p className="small text-muted mb-2">{g.protein_function}</p>

                  <div className="mb-2">
                    <span className="fw-semibold small">Key Features:</span>
                    <ul className="mb-1 ps-3">
                      {(g.key_features || []).map((f, i) => (
                        <li key={i} className="small">{f}</li>
                      ))}
                    </ul>
                  </div>

                  {g.pathognomonic && (
                    <div className="alert alert-info py-1 px-2 mb-2">
                      <span className="small fw-bold">Pathognomonic: </span>
                      <span className="small">{g.pathognomonic}</span>
                    </div>
                  )}

                  {g.targeted_therapy && (
                    <div className="alert alert-success py-1 px-2 mb-2">
                      <span className="small fw-bold">Targeted Rx: </span>
                      <span className="small">{g.targeted_therapy}</span>
                    </div>
                  )}

                  {g.critical_avoid && (
                    <div className="alert alert-danger py-1 px-2 mb-2">
                      <span className="small fw-bold">AVOID: </span>
                      <span className="small">{g.critical_avoid}</span>
                    </div>
                  )}

                  <div className="row text-center border-top pt-2 mt-2">
                    <div className="col-4">
                      <div className="fw-bold" style={{ color }}>{g.epilepsy_pct}%</div>
                      <div style={{ fontSize: '0.65rem' }} className="text-muted">Epilepsy</div>
                    </div>
                    <div className="col-4">
                      <div className="fw-bold" style={{ color }}>{g.cns_tumor_pct}%</div>
                      <div style={{ fontSize: '0.65rem' }} className="text-muted">CNS Tumor</div>
                    </div>
                    <div className="col-4">
                      <div className="fw-bold" style={{ color }}>{g.malignancy_pct}%</div>
                      <div style={{ fontSize: '0.65rem' }} className="text-muted">Malignancy</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Definitions tab ──────────────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.definitions || [];

  return (
    <div>
      <h5 className="mb-3">Definitions &amp; Key Concepts ({defs.length} terms)</h5>
      <div className="row">
        {defs.map((d, i) => (
          <div key={i} className="col-md-6 mb-3">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body">
                <h6 className="card-title fw-bold" style={{ color: '#1b5e20' }}>{d.term}</h6>
                <p className="card-text small text-muted mb-1">{d.definition}</p>
                {d.clinical_relevance && (
                  <p className="small text-secondary mb-0">
                    <em>Clinical relevance: {d.clinical_relevance}</em>
                  </p>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Main page ────────────────────────────────────────────────────────── */
export default function NeurocutaneousAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview,    setOverview]    = useState(null);
  const [breakdown,   setBreakdown]   = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [ov, br, df] = await Promise.all([
          fetch(`${API}/api/neurocutaneous-atlas/overview`).then(r => r.json()),
          fetch(`${API}/api/neurocutaneous-atlas/breakdown`).then(r => r.json()),
          fetch(`${API}/api/neurocutaneous-atlas/definitions`).then(r => r.json()),
        ]);
        setOverview(ov);
        setBreakdown(br);
        setDefinitions(df);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <Loading />;
  if (error)   return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1400 }}>
      {/* Page header */}
      <div className="mb-3">
        <h2 className="mb-0" style={{ color: '#1b5e20' }}>
          &#x1f9e0; Neurocutaneous Syndromes Atlas — 8 Genes
        </h2>
        <p className="text-muted mb-0">
          NF1 · NF2 · TSC1 · TSC2 · VHL · PTEN · PTCH1 · SMARCB1 — 320 patients, seeds 1134–1141
        </p>
      </div>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${activeTab === t ? 'active fw-semibold' : ''}`}
              onClick={() => setActiveTab(t)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* Tab content */}
      {activeTab === 'Overview'       && <OverviewTab      data={overview} />}
      {activeTab === 'Gene Table'     && <GeneTableTab     data={breakdown} />}
      {activeTab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {activeTab === 'Definitions'    && <DefinitionsTab   data={definitions} />}
    </div>
  );
}
