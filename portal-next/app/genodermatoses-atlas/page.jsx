'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ABCA12:  '#bf360c',  // deep burnt orange — harlequin, most severe ichthyosis
  KRT1:    '#4a148c',  // deep purple — epidermolytic ichthyosis AD
  STS:     '#1a237e',  // deep navy — X-linked ichthyosis
  COL7A1:  '#b71c1c',  // deep red — dystrophic EB SCC risk
  LAMA3:   '#880e4f',  // deep magenta — JEB lethal
  KRT5:    '#1b5e20',  // deep green — EBS no scarring
  ATP2C1:  '#e65100',  // deep amber — Hailey-Hailey botox
  EDA:     '#37474f',  // dark slate — XLHED anhidrosis
};

const GENE_DISEASE = {
  ABCA12:  'Harlequin Ichthyosis (AR)',
  KRT1:    'Epidermolytic Ichthyosis (AD)',
  STS:     'X-linked Recessive Ichthyosis (XLR)',
  COL7A1:  'Dystrophic EB (AD/AR)',
  LAMA3:   'Junctional EB Herlitz (AR lethal)',
  KRT5:    'EB Simplex Dowling-Meara (AD)',
  ATP2C1:  'Hailey-Hailey Disease (AD)',
  EDA:     'X-linked Hypohidrotic Ectodermal Dysplasia (XLR)',
};

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Genodermatoses atlas…</p>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger m-4"><strong>Error:</strong> {msg}</div>;
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm border-0">
        <div className="card-body text-center p-3">
          <div className="fw-bold fs-3" style={{ color: color || '#bf360c' }}>{value}</div>
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
        <div className="progress-bar" style={{ width: `${p}%`, backgroundColor: color || '#bf360c' }} />
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const agg = ov.aggregate_clinical || {};
  const alerts = ov.drug_alerts || [];
  const pearls = ov.clinical_pearls || [];

  return (
    <div>
      {/* Hero */}
      <div className="rounded-3 p-4 mb-4 text-white" style={{ background: 'linear-gradient(135deg,#bf360c,#880e4f)' }}>
        <h2 className="fw-bold">{ov.atlas_name}</h2>
        <p className="mb-1 opacity-90">{ov.atlas_subtitle}</p>
        <div className="d-flex gap-3 flex-wrap mt-2">
          <span className="badge bg-light text-dark">{ov.n_genes} Genes</span>
          <span className="badge bg-light text-dark">{ov.n_patients} Patients</span>
          <span className="badge bg-light text-dark">Seeds {ov.seeds}</span>
          <span className="badge bg-light text-dark">8 Genes: {(ov.genes || []).join(' · ')}</span>
        </div>
      </div>

      {/* KPIs */}
      <div className="row g-3 mb-4">
        <KPI label="Collodion Membrane %" value={`${agg.collodion_membrane_pct || 0}%`} color="#bf360c" />
        <KPI label="Blistering %" value={`${agg.blistering_pct || 0}%`} color="#b71c1c" />
        <KPI label="Granulation Tissue %" value={`${agg.granulation_tissue_pct || 0}%`} color="#880e4f" />
        <KPI label="SCC %" value={`${agg.scc_pct || 0}%`} color="#b71c1c" />
        <KPI label="Anhidrosis %" value={`${agg.anhidrosis_pct || 0}%`} color="#37474f" />
        <KPI label="Heat Stroke/Hyperthermia %" value={`${agg.heat_stroke_pct || 0}%`} color="#e65100" />
      </div>

      {/* Description */}
      <div className="card border-0 shadow-sm mb-4">
        <div className="card-body">
          <h5 className="fw-bold mb-3" style={{ color: '#bf360c' }}>Atlas Overview</h5>
          <p className="mb-0 text-muted">{ov.description}</p>
        </div>
      </div>

      {/* Aggregate bars — two columns */}
      <div className="row g-4 mb-4">
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Skin / EB Features</h6>
              <BarRow label="Collodion Membrane (ABCA12)" pct={agg.collodion_membrane_pct} color="#bf360c" />
              <BarRow label="Ectropion (ABCA12 HI)" pct={agg.ectropion_pct} color="#bf360c" />
              <BarRow label="Hyperkeratosis (KRT1)" pct={agg.hyperkeratosis_pct} color="#4a148c" />
              <BarRow label="Pseudosyndactyly (COL7A1 RDEB)" pct={agg.pseudosyndactyly_pct} color="#b71c1c" />
              <BarRow label="Esophageal Stricture (COL7A1)" pct={agg.esophageal_stricture_pct} color="#b71c1c" />
              <BarRow label="SCC (COL7A1 RDEB)" pct={agg.scc_pct} color="#b71c1c" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Systemic / Treatment Features</h6>
              <BarRow label="Granulation Tissue (LAMA3 JEB)" pct={agg.granulation_tissue_pct} color="#880e4f" />
              <BarRow label="Corneal Opacity (STS XLRI)" pct={agg.corneal_opacity_pct} color="#1a237e" />
              <BarRow label="Anhidrosis (EDA XLHED)" pct={agg.anhidrosis_pct} color="#37474f" />
              <BarRow label="Candida Superinfection (ATP2C1)" pct={agg.candida_infection_pct} color="#e65100" />
              <BarRow label="Retinoid Use (ABCA12+KRT1)" pct={agg.retinoid_use_pct} color="#bf360c" />
              <BarRow label="Botulinum Toxin (ATP2C1)" pct={agg.botox_pct} color="#e65100" />
            </div>
          </div>
        </div>
      </div>

      {/* Drug / Treatment Alerts */}
      <h5 className="fw-bold mb-3" style={{ color: '#b71c1c' }}>Critical Treatment Alerts</h5>
      {alerts.map((a, i) => (
        <div key={i} className="alert border-0 shadow-sm mb-3" style={{ borderLeft: '4px solid #b71c1c', backgroundColor: '#fff5f5' }}>
          <strong>{a.title}</strong>
          <p className="mb-0 small mt-1 text-muted">{a.body}</p>
        </div>
      ))}

      {/* Clinical Pearls */}
      <h5 className="fw-bold mb-3 mt-4" style={{ color: '#bf360c' }}>Clinical Pearls</h5>
      <div className="card border-0 shadow-sm">
        <div className="card-body">
          <ol className="mb-0">
            {pearls.map((p, i) => <li key={i} className="mb-2 small">{p}</li>)}
          </ol>
        </div>
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const bd = data.breakdown || {};
  const genes = Object.keys(bd);

  return (
    <div>
      <h5 className="fw-bold mb-3" style={{ color: '#bf360c' }}>8-Gene Reference Table</h5>
      <div className="table-responsive">
        <table className="table table-hover table-bordered small align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th>
              <th>Disease</th>
              <th>Protein / kDa</th>
              <th>Locus</th>
              <th>Inheritance</th>
              <th>OMIM Gene</th>
              <th>N Patients</th>
            </tr>
          </thead>
          <tbody>
            {genes.map(g => {
              const info = bd[g];
              const cs = info.cohort_stats || {};
              return (
                <tr key={g}>
                  <td>
                    <span className="badge" style={{ backgroundColor: GENE_COLORS[g] || '#555' }}>{g}</span>
                  </td>
                  <td className="fw-bold small">{GENE_DISEASE[g] || g}</td>
                  <td className="text-muted small">{info.protein?.split(' ').slice(0, 4).join(' ')} · {info.kDa}</td>
                  <td><code>{info.locus}</code></td>
                  <td className="small">{info.inheritance?.split(';')[0]}</td>
                  <td>
                    <a href={`https://omim.org/entry/${info.omim_gene}`} target="_blank" rel="noreferrer" className="text-decoration-none">
                      {info.omim_gene}
                    </a>
                  </td>
                  <td className="text-center fw-bold">{cs.n || 40}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Per-gene detail cards */}
      {genes.map(g => {
        const info = bd[g];
        const cs = info.cohort_stats || {};
        const color = GENE_COLORS[g] || '#555';
        return (
          <div key={g} className="card border-0 shadow-sm mb-4">
            <div className="card-header text-white" style={{ backgroundColor: color }}>
              <h6 className="mb-0 fw-bold">{g} — {GENE_DISEASE[g]}</h6>
              <small className="opacity-90">{info.aa} · {info.locus} · {info.inheritance?.split(';')[0]}</small>
            </div>
            <div className="card-body">
              <div className="row g-3">
                <div className="col-md-7">
                  <p className="small mb-2"><strong>Hallmark:</strong> {info.hallmark}</p>
                  <div>
                    <strong className="small">Treatment Alerts:</strong>
                    <ul className="small mt-1 mb-0">
                      {(info.treatment_alerts || []).slice(0, 3).map((a, i) => (
                        <li key={i} className="text-danger">{a}</li>
                      ))}
                    </ul>
                  </div>
                </div>
                <div className="col-md-5">
                  <strong className="small">Cohort Stats ({cs.n} pts):</strong>
                  <div className="mt-2">
                    {Object.entries(cs)
                      .filter(([k]) => k.endsWith('_pct'))
                      .slice(0, 5)
                      .map(([k, v]) => (
                        <BarRow
                          key={k}
                          label={k.replace(/_pct$/, '').replace(/_/g, ' ')}
                          pct={v}
                          color={color}
                        />
                      ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const bd = data.breakdown || {};
  const genes = Object.keys(bd);

  return (
    <div>
      <h5 className="fw-bold mb-3" style={{ color: '#bf360c' }}>Clinical Atlas — Phenotype & DDx</h5>
      {genes.map(g => {
        const info = bd[g];
        const color = GENE_COLORS[g] || '#555';
        return (
          <div key={g} className="card border-0 shadow-sm mb-4">
            <div className="card-header d-flex align-items-center" style={{ borderLeft: `6px solid ${color}`, backgroundColor: '#f8f9fa' }}>
              <span className="badge me-2" style={{ backgroundColor: color }}>{g}</span>
              <span className="fw-bold">{GENE_DISEASE[g]}</span>
              <code className="ms-auto small text-muted">{info.locus}</code>
            </div>
            <div className="card-body">
              <div className="row g-3">
                <div className="col-md-6">
                  <h6 className="fw-bold small mb-2" style={{ color }}>Gene Class</h6>
                  <p className="small text-muted mb-3" style={{ maxHeight: 150, overflowY: 'auto' }}>{info.gene_class}</p>
                  <h6 className="fw-bold small mb-2" style={{ color }}>Phenotype</h6>
                  <p className="small text-muted mb-0" style={{ maxHeight: 150, overflowY: 'auto' }}>{info.phenotype}</p>
                </div>
                <div className="col-md-6">
                  <h6 className="fw-bold small mb-2" style={{ color }}>Key DDx</h6>
                  <ul className="small mb-3">
                    {(info.key_ddx || []).map((d, i) => <li key={i}>{d}</li>)}
                  </ul>
                  <h6 className="fw-bold small mb-2" style={{ color }}>All Treatment Alerts</h6>
                  <ul className="small mb-0">
                    {(info.treatment_alerts || []).map((a, i) => (
                      <li key={i} className="text-danger mb-1">{a}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.definitions || {};

  return (
    <div>
      <h5 className="fw-bold mb-3" style={{ color: '#bf360c' }}>Clinical Definitions</h5>
      {Object.entries(defs).map(([term, body]) => (
        <div key={term} className="card border-0 shadow-sm mb-3">
          <div className="card-header" style={{ borderLeft: '4px solid #bf360c', backgroundColor: '#fff8f6' }}>
            <strong>{term}</strong>
          </div>
          <div className="card-body">
            <p className="small text-muted mb-0">{body}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function GenodermatosesAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/genodermatoses-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/genodermatoses-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/genodermatoses-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1400 }}>
      {/* Page header */}
      <div className="mb-4">
        <h1 className="fw-bold" style={{ color: '#bf360c' }}>Genodermatoses Atlas</h1>
        <p className="text-muted mb-0">
          Complete 8-Gene Hereditary Skin Disorder Reference ·{' '}
          <span className="badge bg-secondary">ABCA12 · KRT1 · STS · COL7A1 · LAMA3 · KRT5 · ATP2C1 · EDA</span>{' '}
          · 320 patients (8×40, seeds 1278–1285)
        </p>
      </div>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active fw-bold' : ''}`}
              style={tab === t ? { color: '#bf360c', borderBottomColor: '#bf360c' } : {}}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* Tab content */}
      {loading && <Loading />}
      {!loading && tab === 'Overview'      && <OverviewTab data={overview} />}
      {!loading && tab === 'Gene Table'    && <GeneTableTab data={breakdown} />}
      {!loading && tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {!loading && tab === 'Definitions'   && <DefinitionsTab data={definitions} />}
    </div>
  );
}
