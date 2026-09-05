'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  CFTR:     '#01579b',   // deep blue — CF airway disease
  SERPINA1: '#4a148c',   // deep purple — protease inhibitor
  BMPR2:    '#880e4f',   // deep pink/red — PAH vascular
  SFTPB:    '#b71c1c',   // deep red — lethal surfactant
  SFTPC:    '#bf360c',   // deep orange-red — childhood ILD
  ABCA3:    '#e65100',   // deep orange — lipid transporter
  'NKX2-1': '#1a237e',   // deep navy — transcription factor BLT
  FLCN:     '#1b5e20',   // deep green — tumour suppressor BHD
};

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading pulmonary atlas…</p>
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
          <div className="fw-bold fs-3" style={{ color: color || '#01579b' }}>{value}</div>
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
        <div className="progress-bar" style={{ width: `${p}%`, backgroundColor: color || '#01579b' }} />
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
      <div className="rounded-3 p-4 mb-4 text-white" style={{ background: 'linear-gradient(135deg,#01579b,#1a237e)' }}>
        <h3 className="mb-1">{ov.atlas_name || 'Pulmonary Atlas'}</h3>
        <p className="mb-1 opacity-75">{ov.atlas_subtitle}</p>
        <div className="d-flex flex-wrap gap-3 mt-3">
          <span className="badge bg-white text-dark fs-6">{ov.n_genes} Genes</span>
          <span className="badge bg-white text-dark fs-6">{ov.n_patients} Patients</span>
          <span className="badge bg-white text-dark fs-6">Seeds {ov.seeds}</span>
        </div>
      </div>

      {/* Drug / Clinical Alerts */}
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
            <div className="card-header fw-semibold" style={{ background: '#e3f2fd' }}>Clinical Rates (Aggregate)</div>
            <div className="card-body">
              <BarRow label="ESRD Rate"          pct={agg.esrd_pct}               color="#b71c1c" />
              <BarRow label="Hypertension"        pct={agg.hypertension_pct}       color="#880e4f" />
              <BarRow label="Transplant/HSCT"     pct={agg.transplant_rate_pct}    color="#1b5e20" />
              <BarRow label="Drug Error"          pct={agg.drug_error_pct}         color="#e65100" />
              <BarRow label="Delayed Diagnosis"   pct={agg.diagnosis_delayed_pct}  color="#f57f17" />
              <BarRow label="Surveillance OK"     pct={agg.surveillance_adherent_pct} color="#01579b" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: '#e3f2fd' }}>Pathway Targets</div>
            <div className="card-body">
              {Object.entries(pathways).map(([key, pathway]) => (
                <div key={key} className="d-flex align-items-start mb-2">
                  <span className="badge me-2 flex-shrink-0"
                    style={{ background: GENE_COLORS[key] || '#6c757d', fontSize: '0.72rem', minWidth: 80 }}>
                    {key}
                  </span>
                  <small className="text-muted">{pathway}</small>
                </div>
              ))}
            </div>
          </div>
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: '#e3f2fd' }}>Severity Distribution</div>
            <div className="card-body">
              <BarRow label="Mild"     pct={agg.severity_mild_pct}     color="#1b5e20" />
              <BarRow label="Moderate" pct={agg.severity_moderate_pct} color="#f57f17" />
              <BarRow label="Severe"   pct={agg.severity_severe_pct}   color="#b71c1c" />
            </div>
          </div>
        </div>
      </div>

      {/* Critical Rules */}
      {rules.length > 0 && (
        <div className="card border-danger border-2 mb-3">
          <div className="card-header bg-danger text-white fw-bold">Critical Rules — Must Know</div>
          <div className="card-body">
            <ul className="mb-0 ps-3">
              {rules.map((r, i) => (
                <li key={i} className="mb-1 small">{r}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Disease Category Breakdown */}
      {ov.disease_category_breakdown && (
        <div className="card border-0 shadow-sm mb-3">
          <div className="card-header fw-semibold" style={{ background: '#e3f2fd' }}>Disease Pathway Category Distribution</div>
          <div className="card-body">
            {Object.entries(ov.disease_category_breakdown).map(([cat, pct], i) => (
              <BarRow key={i} label={cat} pct={pct} color={Object.values(GENE_COLORS)[i % 8]} />
            ))}
          </div>
        </div>
      )}

      {/* Description */}
      <div className="card border-0 bg-light mb-3">
        <div className="card-body small text-muted">
          <strong>Atlas Description:</strong> {ov.description}
        </div>
      </div>
    </div>
  );
}

/* ── Gene Table tab ────────────────────────────────────────────────────── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || [];
  return (
    <div>
      <h5 className="mb-3">8-Gene Hereditary Pulmonary Disease Reference Table</h5>
      <div className="table-responsive">
        <table className="table table-bordered table-hover small align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th>
              <th>Protein</th>
              <th>Locus</th>
              <th>Size</th>
              <th>Inheritance</th>
              <th>Disease</th>
              <th>Drug Err %</th>
              <th>Dx Delay %</th>
              <th>Severe %</th>
            </tr>
          </thead>
          <tbody>
            {genes.map((g) => (
              <tr key={g.gene}>
                <td>
                  <span className="badge" style={{ background: GENE_COLORS[g.gene] || '#6c757d' }}>
                    {g.gene}
                  </span>
                </td>
                <td className="fw-semibold">{g.protein}</td>
                <td><code>{g.locus}</code></td>
                <td>{g.aa}</td>
                <td>{g.inheritance?.split(';')[0]}</td>
                <td>{g.phenotype?.substring(0, 80)}…</td>
                <td className="text-center" style={{ color: g.cohort_stats?.drug_error_pct > 20 ? '#b71c1c' : '#1b5e20' }}>
                  {g.cohort_stats?.drug_error_pct}%
                </td>
                <td className="text-center" style={{ color: g.cohort_stats?.dx_delayed_pct > 35 ? '#b71c1c' : '#1b5e20' }}>
                  {g.cohort_stats?.dx_delayed_pct}%
                </td>
                <td className="text-center fw-bold" style={{ color: '#b71c1c' }}>
                  {g.cohort_stats?.severity?.Severe}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-gene severity bars */}
      <div className="row mt-3">
        {genes.map((g) => (
          <div key={g.gene} className="col-md-6 mb-3">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-header fw-semibold small"
                style={{ background: (GENE_COLORS[g.gene] || '#6c757d') + '22', borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6c757d'}` }}>
                <span className="badge me-2" style={{ background: GENE_COLORS[g.gene] || '#6c757d' }}>{g.gene}</span>
                {g.protein}
              </div>
              <div className="card-body small">
                <div className="mb-1"><strong>Hallmark:</strong> {g.hallmark?.substring(0, 150)}…</div>
                <div className="mb-1"><strong>Treatment Alert:</strong> <span className="text-danger">{g.treatment_alert?.substring(0, 120)}…</span></div>
                <div className="mb-2"><strong>Primary Complication:</strong> {g.primary_complication}</div>
                <BarRow label="Drug Error" pct={g.cohort_stats?.drug_error_pct}    color="#e65100" />
                <BarRow label="Dx Delay"   pct={g.cohort_stats?.dx_delayed_pct}   color="#f57f17" />
                <BarRow label="Severe"     pct={g.cohort_stats?.severity?.Severe}  color="#b71c1c" />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Clinical Atlas tab ────────────────────────────────────────────────── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || [];
  const [selected, setSelected] = useState(genes[0]?.gene || '');
  const gene = genes.find(g => g.gene === selected);

  return (
    <div>
      <h5 className="mb-3">Clinical Atlas — Full Gene Details</h5>
      <div className="d-flex flex-wrap gap-2 mb-4">
        {genes.map(g => (
          <button
            key={g.gene}
            className={`btn btn-sm ${selected === g.gene ? 'text-white' : 'btn-outline-secondary'}`}
            style={selected === g.gene ? { background: GENE_COLORS[g.gene] || '#01579b', borderColor: GENE_COLORS[g.gene] || '#01579b' } : {}}
            onClick={() => setSelected(g.gene)}
          >
            {g.gene}
          </button>
        ))}
      </div>

      {gene && (
        <div>
          <div className="card border-0 shadow mb-3">
            <div className="card-header text-white fw-bold"
              style={{ background: GENE_COLORS[gene.gene] || '#01579b' }}>
              {gene.gene} — {gene.protein}
            </div>
            <div className="card-body">
              <div className="row mb-3">
                <div className="col-md-3">
                  <small className="text-muted">Locus</small>
                  <div className="fw-bold"><code>{gene.locus}</code></div>
                </div>
                <div className="col-md-3">
                  <small className="text-muted">Size</small>
                  <div className="fw-bold">{gene.aa} · {gene.kDa}</div>
                </div>
                <div className="col-md-3">
                  <small className="text-muted">OMIM Gene</small>
                  <div className="fw-bold">{gene.omim_gene}</div>
                </div>
                <div className="col-md-3">
                  <small className="text-muted">OMIM Disease</small>
                  <div className="fw-bold">{gene.omim_disease}</div>
                </div>
              </div>
              <div className="mb-2">
                <strong>Gene Class / Function:</strong>
                <p className="small text-muted mb-1">{gene.gene_class}</p>
              </div>
              <div className="mb-2">
                <strong>Inheritance:</strong>
                <p className="small mb-1">{gene.inheritance}</p>
              </div>
              <div className="mb-2">
                <strong>Phenotype:</strong>
                <p className="small mb-1">{gene.phenotype}</p>
              </div>
              <div className="mb-2">
                <strong>Hallmark / Diagnostic Key:</strong>
                <p className="small text-primary mb-1">{gene.hallmark}</p>
              </div>
            </div>
          </div>

          <div className="card border-danger border-2 mb-3">
            <div className="card-header bg-danger text-white fw-bold">⚠ Treatment Alert</div>
            <div className="card-body small">{gene.treatment_alert}</div>
          </div>

          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header fw-semibold">Key Differential Diagnosis</div>
            <div className="card-body small">{gene.key_ddx}</div>
          </div>

          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header fw-semibold">Clinical Pattern</div>
            <div className="card-body small">
              <div><strong>Renal / GFR:</strong> {gene.gfr_pattern}</div>
              <div className="mt-1"><strong>Proteinuria:</strong> {gene.proteinuria_pattern}</div>
              <div className="mt-1"><strong>Primary Complication:</strong> {gene.primary_complication}</div>
            </div>
          </div>

          {gene.drug_ci && gene.drug_ci.length > 0 && (
            <div className="card border-warning border-2 mb-3">
              <div className="card-header bg-warning fw-bold">Contraindications / Critical Avoid</div>
              <div className="card-body">
                <ul className="mb-0 ps-3">
                  {gene.drug_ci.map((ci, i) => (
                    <li key={i} className="small text-danger mb-1">{ci}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          <div className="card border-0 bg-light mb-3">
            <div className="card-header fw-semibold">Disease Mechanism (Full Detail)</div>
            <div className="card-body small text-muted" style={{ whiteSpace: 'pre-line', maxHeight: 400, overflowY: 'auto' }}>
              {gene.disease_detail}
            </div>
          </div>

          {/* Variant table */}
          {gene.variants && gene.variants.length > 0 && (
            <div className="card border-0 shadow-sm mb-3">
              <div className="card-header fw-semibold">Key Variants</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0 small">
                  <thead className="table-light">
                    <tr><th>Variant</th><th>Effect</th><th>Frequency</th></tr>
                  </thead>
                  <tbody>
                    {gene.variants.map((v, i) => (
                      <tr key={i}>
                        <td><code>{v.variant}</code></td>
                        <td>{v.effect}</td>
                        <td>{v.frequency}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Cohort stats */}
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header fw-semibold">Cohort Statistics (n={gene.cohort_n})</div>
            <div className="card-body">
              <div className="row">
                <div className="col-md-6">
                  <BarRow label="Hypertension"    pct={gene.cohort_stats?.htn_pct}           color="#880e4f" />
                  <BarRow label="Transplant"       pct={gene.cohort_stats?.transplant_pct}    color="#1b5e20" />
                  <BarRow label="Surveillance OK"  pct={gene.cohort_stats?.surveillance_adherent_pct} color="#01579b" />
                </div>
                <div className="col-md-6">
                  <BarRow label="Drug Error"       pct={gene.cohort_stats?.drug_error_pct}    color="#e65100" />
                  <BarRow label="Delayed Dx"       pct={gene.cohort_stats?.dx_delayed_pct}    color="#f57f17" />
                  <BarRow label="Severe"           pct={gene.cohort_stats?.severity?.Severe}  color="#b71c1c" />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Definitions tab ───────────────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = Array.isArray(data) ? data : [];
  return (
    <div>
      <h5 className="mb-3">Clinical Definitions — Hereditary Pulmonary Disease Terms</h5>
      {defs.map((d, i) => (
        <div key={i} className="card border-0 shadow-sm mb-3">
          <div className="card-header fw-bold" style={{ background: '#e3f2fd' }}>
            {d.term}
            {d.full && d.full !== d.term && (
              <span className="text-muted fw-normal ms-2 small"> — {d.full}</span>
            )}
          </div>
          <div className="card-body small text-muted">{d.explanation}</div>
        </div>
      ))}
    </div>
  );
}

/* ── Main page ─────────────────────────────────────────────────────────── */
export default function PulmonaryAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/pulmonary-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/pulmonary-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pulmonary-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov);
        setBreakdown(bd);
        setDefinitions(df);
      })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4">
      <div className="mb-3">
        <h2 className="mb-0">
          <span style={{ color: '#01579b' }}>🫁</span> Pulmonary Atlas
        </h2>
        <p className="text-muted small mb-0">
          Complete 8-Gene Hereditary Pulmonary Disease Atlas —
          CFTR · SERPINA1 · BMPR2 · SFTPB · SFTPC · ABCA3 · NKX2-1 · FLCN ·
          320 patients (8×40, seeds 1190–1197)
        </p>
      </div>

      {/* Tab nav */}
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
