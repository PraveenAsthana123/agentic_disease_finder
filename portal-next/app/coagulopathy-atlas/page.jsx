'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  F8:     '#b71c1c',
  F9:     '#880e4f',
  VWF:    '#1a237e',
  F11:    '#1b5e20',
  F7:     '#e65100',
  F13A1:  '#4a148c',
  ITGA2B: '#006064',
  GP1BA:  '#37474f',
};

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-danger" role="status" />
      <p className="mt-3 text-muted">Loading coagulopathy atlas…</p>
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

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm border-0">
        <div className="card-body text-center p-3">
          <div className="fw-bold fs-3" style={{ color: color || '#b71c1c' }}>{value}</div>
          <div className="small text-muted">{label}</div>
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
        <div className="progress-bar" style={{ width: `${p}%`, backgroundColor: color || '#b71c1c' }} />
      </div>
    </div>
  );
}

/* ── Overview tab ─────────────────────────────────────────────────────── */
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const agg = ov.aggregate_clinical || {};
  const alerts = ov.drug_alerts || [];
  const rules = ov.critical_rules || [];
  const pathways = ov.pathway_targets || {};
  const kpis = ov.kpis || [];

  return (
    <div>
      {/* Hero */}
      <div className="rounded-3 p-4 mb-4 text-white" style={{ background: 'linear-gradient(135deg,#b71c1c,#880e4f)' }}>
        <h3 className="mb-1">{ov.atlas_name || 'Coagulopathy Atlas'}</h3>
        <p className="mb-1 opacity-75">{ov.atlas_subtitle}</p>
        <div className="d-flex flex-wrap gap-3 mt-3">
          <span className="badge bg-white text-dark fs-6">{ov.n_genes} Genes</span>
          <span className="badge bg-white text-dark fs-6">{ov.n_patients} Patients</span>
          <span className="badge bg-white text-dark fs-6">Seeds {ov.seeds}</span>
        </div>
      </div>

      {/* Drug Alerts */}
      <h5 className="mb-3">Critical Drug &amp; Clinical Alerts</h5>
      {alerts.map((a, i) => (
        <div key={i} className={`alert alert-${a.type === 'danger' ? 'danger' : 'warning'} mb-3`}>
          <strong>{a.title}</strong>
          <p className="mb-0 mt-1 small">{a.body}</p>
        </div>
      ))}

      {/* KPIs */}
      <h5 className="mt-4 mb-3">Aggregate Cohort KPIs (n={ov.n_patients})</h5>
      <div className="row">
        {kpis.map((k, i) => (
          <KPI key={i} label={k.label} value={k.value} color={Object.values(GENE_COLORS)[i % 8]} />
        ))}
      </div>

      {/* Aggregate bars */}
      <div className="row mt-2">
        <div className="col-md-6">
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: '#ffebee' }}>Bleeding &amp; Complication Rates</div>
            <div className="card-body">
              <BarRow label="Severe Bleed"        pct={agg.severe_bleed_pct}       color="#b71c1c" />
              <BarRow label="Inhibitors (HA/HB)"  pct={agg.inhibitor_pct}          color="#880e4f" />
              <BarRow label="ICH (Intracranial)"  pct={agg.icb_pct}                color="#4a148c" />
              <BarRow label="On Prophylaxis"      pct={agg.on_prophylaxis_pct}     color="#1b5e20" />
              <BarRow label="Drug Error"          pct={agg.drug_error_pct}         color="#e65100" />
              <BarRow label="Alloimmunised"       pct={agg.alloimmunised_pct}      color="#006064" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: '#ffebee' }}>Pathway Targets</div>
            <div className="card-body">
              {Object.entries(pathways).map(([key, pathway]) => (
                <div key={key} className="d-flex align-items-start mb-2">
                  <span className="badge me-2 flex-shrink-0"
                    style={{ background: GENE_COLORS[key.split('_')[0]] || '#6c757d', fontSize: '0.72rem', minWidth: 70 }}>
                    {key}
                  </span>
                  <small className="text-muted">{pathway}</small>
                </div>
              ))}
            </div>
          </div>
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: '#ffebee' }}>Special Rates</div>
            <div className="card-body">
              <BarRow label="DDAVP Error (Type2B)" pct={agg.type2b_ddavp_error_pct} color="#b71c1c" />
              <BarRow label="Umbilical Bleed (FXIII)" pct={agg.umbilical_bleed_pct} color="#4a148c" />
              <BarRow label="Giant Platelets (BSS)" pct={agg.giant_platelet_pct}  color="#37474f" />
              <BarRow label="Miscarriage (FXIII)"  pct={agg.miscarriage_pct}      color="#880e4f" />
              <BarRow label="Ashkenazi (FXI)"      pct={agg.ashkenazi_pct}        color="#1b5e20" />
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
              <th>Protein</th>
              <th>Locus</th>
              <th>Inheritance</th>
              <th>n</th>
              <th>Severe Bleed %</th>
              <th>Inhibitor %</th>
              <th>ICH %</th>
              <th>Prophylaxis %</th>
              <th>Drug Error %</th>
              <th>PT/aPTT</th>
            </tr>
          </thead>
          <tbody>
            {genes.map(g => (
              <tr key={g.gene}>
                <td>
                  <span className="badge" style={{ background: GENE_COLORS[g.gene] || '#6c757d', fontSize: '0.85rem' }}>
                    {g.gene}
                  </span>
                </td>
                <td className="small fw-semibold">{g.protein}</td>
                <td><code className="small">{g.locus}</code></td>
                <td><span className="badge bg-secondary small">{g.inheritance?.split(';')[0]}</span></td>
                <td>{g.cohort_n}</td>
                <td>
                  <span className={`fw-bold ${g.severe_bleed_pct >= 70 ? 'text-danger' : g.severe_bleed_pct >= 50 ? 'text-warning' : 'text-success'}`}>
                    {g.severe_bleed_pct}%
                  </span>
                </td>
                <td>{g.inhibitor_pct > 0 ? <span className="badge bg-danger">{g.inhibitor_pct}%</span> : '—'}</td>
                <td>{g.icb_pct > 0 ? <span className="badge bg-warning text-dark">{g.icb_pct}%</span> : '—'}</td>
                <td>{g.prophylaxis_pct}%</td>
                <td>
                  {g.drug_error_pct > 0
                    ? <span className="badge bg-danger">{g.drug_error_pct}%</span>
                    : <span className="badge bg-success">0%</span>}
                </td>
                <td><small className="text-muted">{g.pt_ptt}</small></td>
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
      <h5 className="mb-3">Clinical Atlas — Gene Cards (8 Hereditary Bleeding Disorders)</h5>
      <div className="row">
        {genes.map(g => {
          const color = GENE_COLORS[g.gene] || '#6c757d';
          return (
            <div key={g.gene} className="col-md-6 col-xl-6 mb-4">
              <div className="card h-100 shadow-sm border-0" style={{ borderTop: `4px solid ${color}` }}>
                <div className="card-header text-white fw-bold small" style={{ background: color }}>
                  {g.gene} — {g.protein} — {g.locus}
                </div>
                <div className="card-body p-3">
                  <div className="mb-2">
                    <span className="badge bg-secondary me-1 small">{g.inheritance?.split(';')[0]?.trim()}</span>
                    <span className="badge bg-light text-dark small">{g.aa}</span>
                  </div>

                  <div className="small mb-2">
                    <span className="fw-semibold">Phenotype: </span>
                    <span className="text-muted">{g.phenotype}</span>
                  </div>

                  <div className="small mb-2">
                    <span className="fw-semibold text-danger">Hallmark: </span>
                    <span className="text-muted">{g.hallmark}</span>
                  </div>

                  <div className="small mb-2">
                    <span className="fw-semibold">PT/aPTT: </span>
                    <code className="small">{g.pt_ptt}</code>
                  </div>

                  {g.treatment_alert && (
                    <div className="alert alert-warning py-1 px-2 mb-2">
                      <small><strong>Rx: </strong>{g.treatment_alert}</small>
                    </div>
                  )}

                  {g.key_ddx && (
                    <div className="alert alert-info py-1 px-2 mb-2">
                      <small><strong>DDx: </strong>{g.key_ddx}</small>
                    </div>
                  )}

                  <div className="row text-center border-top pt-2 mt-2">
                    <div className="col-3">
                      <div className="fw-bold" style={{ color }}>{g.severe_bleed_pct}%</div>
                      <div style={{ fontSize: '0.65rem' }} className="text-muted">Severe Bleed</div>
                    </div>
                    <div className="col-3">
                      <div className="fw-bold" style={{ color }}>{g.inhibitor_pct}%</div>
                      <div style={{ fontSize: '0.65rem' }} className="text-muted">Inhibitor</div>
                    </div>
                    <div className="col-3">
                      <div className="fw-bold" style={{ color }}>{g.icb_pct}%</div>
                      <div style={{ fontSize: '0.65rem' }} className="text-muted">ICH</div>
                    </div>
                    <div className="col-3">
                      <div className="fw-bold" style={{ color }}>{g.drug_error_pct}%</div>
                      <div style={{ fontSize: '0.65rem' }} className="text-muted">Drug Error</div>
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
  const defs = Array.isArray(data) ? data : (data.definitions || data);

  return (
    <div>
      <h5 className="mb-3">Definitions &amp; Key Concepts ({defs.length} terms)</h5>
      <div className="row">
        {defs.map((d, i) => (
          <div key={i} className="col-md-6 mb-3">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body">
                <h6 className="card-title fw-bold" style={{ color: '#b71c1c' }}>{d.term}</h6>
                <p className="card-text small text-muted mb-0">{d.definition}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Main page ────────────────────────────────────────────────────────── */
export default function CoagulopathyAtlasPage() {
  const [activeTab,    setActiveTab]    = useState('Overview');
  const [overview,     setOverview]     = useState(null);
  const [breakdown,    setBreakdown]    = useState(null);
  const [definitions,  setDefinitions]  = useState(null);
  const [loading,      setLoading]      = useState(true);
  const [error,        setError]        = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [ov, br, df] = await Promise.all([
          fetch(`${API}/api/coagulopathy-atlas/overview`).then(r => r.json()),
          fetch(`${API}/api/coagulopathy-atlas/breakdown`).then(r => r.json()),
          fetch(`${API}/api/coagulopathy-atlas/definitions`).then(r => r.json()),
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
      <div className="mb-3">
        <h2 className="mb-0" style={{ color: '#b71c1c' }}>
          🩸 Coagulopathy-Atlas — 8 Genes
        </h2>
        <p className="text-muted mb-0">
          F8 · F9 · VWF · F11 · F7 · F13A1 · ITGA2B · GP1BA — 320 patients, seeds 1142–1149
        </p>
      </div>

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

      {activeTab === 'Overview'       && <OverviewTab      data={overview} />}
      {activeTab === 'Gene Table'     && <GeneTableTab     data={breakdown} />}
      {activeTab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {activeTab === 'Definitions'    && <DefinitionsTab   data={definitions} />}
    </div>
  );
}
