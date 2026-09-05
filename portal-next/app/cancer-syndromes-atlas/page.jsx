'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  BRCA1:  '#880e4f',   // deep pink-red — HBOC1 breast/ovarian
  BRCA2:  '#b71c1c',   // deep red — HBOC2 breast/ovarian/pancreatic
  MLH1:   '#1a237e',   // deep navy — Lynch MMR
  MSH2:   '#01579b',   // deep blue — Lynch urothelial
  TP53:   '#212121',   // near-black — guardian of genome LFS
  RB1:    '#4a148c',   // deep purple — retinoblastoma pRb
  CDKN2A: '#bf360c',   // deep orange — FAMMM/pancreatic
  APC:    '#1b5e20',   // deep green — FAP polyposis
};

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading cancer syndromes atlas…</p>
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
          <div className="fw-bold fs-3" style={{ color: color || '#880e4f' }}>{value}</div>
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
        <div className="progress-bar" style={{ width: `${p}%`, backgroundColor: color || '#880e4f' }} />
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
      <div className="rounded-3 p-4 mb-4 text-white" style={{ background: 'linear-gradient(135deg,#880e4f,#1a237e)' }}>
        <h3 className="mb-1">{ov.atlas_name || 'Cancer Syndromes Atlas'}</h3>
        <p className="mb-1 opacity-75">{ov.atlas_subtitle}</p>
        <div className="d-flex flex-wrap gap-3 mt-3">
          <span className="badge bg-white text-dark">{ov.n_genes} Genes</span>
          <span className="badge bg-white text-dark">{ov.n_patients} Patients</span>
          <span className="badge bg-white text-dark">Seeds {ov.seeds}</span>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-4">
        {kpis.map((k, i) => (
          <KPI key={i} label={k.label} value={k.value} color="#880e4f" />
        ))}
      </div>

      {/* Description */}
      <div className="card border-0 shadow-sm mb-4">
        <div className="card-body">
          <h6 className="fw-semibold mb-2">Atlas Description</h6>
          <p className="small mb-0">{ov.description}</p>
        </div>
      </div>

      {/* Aggregate clinical bars */}
      <div className="card border-0 shadow-sm mb-4">
        <div className="card-body">
          <h6 className="fw-semibold mb-3">Aggregate Clinical Metrics (320 patients)</h6>
          <div className="row">
            <div className="col-md-6">
              <BarRow label="Drug / Clinical Errors" pct={agg.drug_error_pct} color="#c62828" />
              <BarRow label="Diagnosis Delayed" pct={agg.diagnosis_delayed_pct} color="#e65100" />
              <BarRow label="Surveillance Adherent" pct={agg.surveillance_adherent_pct} color="#2e7d32" />
            </div>
            <div className="col-md-6">
              <BarRow label="Severity: Mild" pct={agg.severity_mild_pct} color="#43a047" />
              <BarRow label="Severity: Moderate" pct={agg.severity_moderate_pct} color="#f57c00" />
              <BarRow label="Severity: Severe" pct={agg.severity_severe_pct} color="#c62828" />
            </div>
          </div>
        </div>
      </div>

      {/* Drug Alerts */}
      <div className="mb-4">
        <h6 className="fw-semibold mb-3">Critical Drug & Clinical Alerts</h6>
        {alerts.map((a, i) => (
          <div key={i} className={`alert alert-${a.type === 'danger' ? 'danger' : 'warning'} mb-3`}>
            <strong>{a.title}</strong>
            <p className="mb-0 mt-1 small">{a.body}</p>
          </div>
        ))}
      </div>

      {/* Pathway targets */}
      <div className="card border-0 shadow-sm mb-4">
        <div className="card-body">
          <h6 className="fw-semibold mb-3">Molecular Pathway Targets</h6>
          <div className="row row-cols-1 row-cols-md-2 g-2">
            {Object.entries(pathways).map(([gene, target]) => (
              <div key={gene} className="col">
                <div className="d-flex align-items-start gap-2 p-2 rounded" style={{ background: '#fafafa' }}>
                  <span className="badge rounded-pill text-white fw-bold" style={{ backgroundColor: GENE_COLORS[gene] || '#880e4f', minWidth: 72 }}>{gene}</span>
                  <span className="small">{target}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Critical rules */}
      <div className="card border-0 shadow-sm mb-4">
        <div className="card-body">
          <h6 className="fw-semibold mb-3">Critical Clinical Rules (10 Must-Know)</h6>
          <ol className="mb-0 ps-3">
            {rules.map((r, i) => <li key={i} className="small mb-1">{r}</li>)}
          </ol>
        </div>
      </div>

      {/* Disease category breakdown */}
      {ov.disease_category_breakdown && (
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-body">
            <h6 className="fw-semibold mb-3">Disease Category Breakdown (% of cohort)</h6>
            {Object.entries(ov.disease_category_breakdown).map(([cat, pct]) => (
              <BarRow key={cat} label={cat} pct={pct} color="#880e4f" />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Gene Table tab ───────────────────────────────────────────────────── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = (data.genes || []);
  return (
    <div>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th>
              <th>Locus</th>
              <th>Size</th>
              <th>Inheritance</th>
              <th>OMIM (Disease)</th>
              <th>Drug Errors %</th>
              <th>Dx Delayed %</th>
              <th>Surveillance %</th>
            </tr>
          </thead>
          <tbody>
            {genes.map((g, i) => {
              const cs = g.cohort_stats || {};
              return (
                <tr key={i}>
                  <td>
                    <span className="badge text-white fw-bold" style={{ backgroundColor: GENE_COLORS[g.gene] || '#880e4f' }}>{g.gene}</span>
                  </td>
                  <td><code className="small">{g.locus}</code></td>
                  <td className="small">{g.aa}</td>
                  <td className="small">{g.inheritance}</td>
                  <td className="small">
                    <a href={`https://omim.org/entry/${g.omim_disease}`} target="_blank" rel="noreferrer">#{g.omim_disease}</a>
                  </td>
                  <td><span className={`badge ${cs.drug_error_pct > 25 ? 'bg-danger' : 'bg-warning text-dark'}`}>{cs.drug_error_pct}%</span></td>
                  <td><span className={`badge ${cs.dx_delayed_pct > 35 ? 'bg-danger' : 'bg-secondary'}`}>{cs.dx_delayed_pct}%</span></td>
                  <td><span className={`badge ${cs.surveillance_adherent_pct >= 60 ? 'bg-success' : 'bg-warning text-dark'}`}>{cs.surveillance_adherent_pct}%</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Clinical Atlas tab ───────────────────────────────────────────────── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = (data.genes || []);
  const [selected, setSelected] = useState(null);

  const sel = selected !== null ? genes[selected] : null;

  return (
    <div className="row">
      {/* Gene list */}
      <div className="col-md-3 mb-3">
        <div className="list-group">
          {genes.map((g, i) => (
            <button
              key={i}
              className={`list-group-item list-group-item-action d-flex align-items-center gap-2 ${selected === i ? 'active' : ''}`}
              style={selected === i ? { backgroundColor: GENE_COLORS[g.gene] || '#880e4f', borderColor: GENE_COLORS[g.gene] || '#880e4f' } : {}}
              onClick={() => setSelected(i)}
            >
              <span className="badge text-white fw-bold" style={{ backgroundColor: selected === i ? 'rgba(255,255,255,0.3)' : (GENE_COLORS[g.gene] || '#880e4f') }}>{g.gene}</span>
              <span className="small">{g.aa}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Gene detail */}
      <div className="col-md-9">
        {!sel ? (
          <div className="alert alert-info">Select a gene to view full clinical profile.</div>
        ) : (
          <div>
            <div className="p-3 rounded text-white mb-3" style={{ backgroundColor: GENE_COLORS[sel.gene] || '#880e4f' }}>
              <h5 className="mb-1">{sel.gene} — {sel.protein}</h5>
              <div className="small opacity-75">{sel.alias}</div>
            </div>

            <div className="row g-3 mb-3">
              {[
                ['Locus', sel.locus],
                ['Size', sel.aa],
                ['kDa', sel.kDa],
                ['OMIM Gene', sel.omim_gene],
                ['OMIM Disease', sel.omim_disease],
                ['Inheritance', sel.inheritance],
              ].map(([label, val]) => (
                <div key={label} className="col-6 col-md-4">
                  <div className="card h-100 border-0 shadow-sm">
                    <div className="card-body p-2 text-center">
                      <div className="small text-muted">{label}</div>
                      <div className="fw-semibold small">{val}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <Section title="Gene Class / Molecular Mechanism" body={sel.gene_class} />
            <Section title="Clinical Phenotype" body={sel.phenotype} />
            <Section title="Hallmark / Pathognomonic Features" body={sel.hallmark} color="#fff3e0" />
            <Section title="Treatment Alerts" body={sel.treatment_alert} color="#fce4ec" />
            <Section title="Key DDx" body={sel.key_ddx} />
            <Section title="Disease Detail" body={sel.disease_detail} />

            {sel.variants && sel.variants.length > 0 && (
              <div className="card border-0 shadow-sm mb-3">
                <div className="card-body">
                  <h6 className="fw-semibold mb-2">Key Variants</h6>
                  {sel.variants.map((v, i) => (
                    <div key={i} className="d-flex gap-2 mb-1 small">
                      <span className="badge bg-secondary text-white">{v.name}</span>
                      <span className="text-muted">{v.frequency}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {sel.drug_ci && sel.drug_ci.length > 0 && (
              <div className="card border-danger border-2 shadow-sm mb-3">
                <div className="card-body">
                  <h6 className="fw-semibold text-danger mb-2">Drug Contraindications / Alerts</h6>
                  <ul className="mb-0 ps-3">
                    {sel.drug_ci.map((ci, i) => <li key={i} className="small mb-1">{ci}</li>)}
                  </ul>
                </div>
              </div>
            )}

            {/* Cohort stats */}
            <div className="card border-0 shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-semibold mb-2">Cohort Stats (n={sel.cohort_n})</h6>
                <div className="row">
                  {[
                    ['Drug Errors', sel.cohort_stats?.drug_error_pct, '#c62828'],
                    ['Dx Delayed', sel.cohort_stats?.dx_delayed_pct, '#e65100'],
                    ['Surveillance', sel.cohort_stats?.surveillance_adherent_pct, '#2e7d32'],
                  ].map(([lbl, pct, col]) => (
                    <div key={lbl} className="col-md-4">
                      <BarRow label={lbl} pct={pct} color={col} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Section({ title, body, color }) {
  return (
    <div className="card border-0 shadow-sm mb-3" style={color ? { backgroundColor: color } : {}}>
      <div className="card-body">
        <h6 className="fw-semibold mb-1">{title}</h6>
        <p className="small mb-0">{body}</p>
      </div>
    </div>
  );
}

/* ── Definitions tab ──────────────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = Array.isArray(data) ? data : [];
  return (
    <div>
      {defs.map((d, i) => (
        <div key={i} className="card border-0 shadow-sm mb-3">
          <div className="card-body">
            <h6 className="fw-semibold mb-1">{d.term}</h6>
            <p className="text-muted small mb-1"><em>{d.full}</em></p>
            <p className="small mb-0">{d.explanation}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Main page ────────────────────────────────────────────────────────── */
export default function CancerSyndromesAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/cancer-syndromes-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/cancer-syndromes-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/cancer-syndromes-atlas/definitions`).then(r => r.json()),
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
          <span style={{ color: '#880e4f' }}>🧬</span> Cancer Syndromes Atlas
        </h2>
        <p className="text-muted small mb-0">
          Complete 8-Gene Hereditary Cancer Predisposition Atlas —
          BRCA1 · BRCA2 · MLH1 · MSH2 · TP53 · RB1 · CDKN2A · APC ·
          320 patients (8×40, seeds 1198–1205)
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
