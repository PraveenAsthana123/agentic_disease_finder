'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  AIP:     '#1a237e',  // deep navy — FIPA, GH-secreting adenoma, SSA resistance
  PRKAR1A: '#880e4f',  // deep magenta — Carney Complex, cardiac myxoma, PPNAD
  PROP1:   '#4a148c',  // deep purple — CPHD2, pituitary hyperplasia, AR
  POU1F1:  '#0d47a1',  // deep blue — CPHD1, Pit-1, thyroxine-first rule
  LHX3:    '#1b5e20',  // deep green — rigid cervical spine, CPHD3
  HESX1:   '#e65100',  // deep amber — SOD, optic hypoplasia, De Morsier
  GLI2:    '#37474f',  // dark slate — HPE, single central incisor, SHH
  CABLES1: '#b71c1c',  // deep red — corticotropinoma, Nelson syndrome
};

const GENE_DISEASE = {
  AIP:     'FIPA — Familial Isolated Pituitary Adenoma (AD)',
  PRKAR1A: 'Carney Complex — Myxoma + PPNAD + GH Adenoma (AD)',
  PROP1:   'CPHD2 — Combined Pituitary Hormone Deficiency (AR)',
  POU1F1:  'CPHD1 — GH + TSH + PRL Deficiency (AD/AR)',
  LHX3:    'CPHD3 — Rigid Cervical Spine + CPHD (AR)',
  HESX1:   'SOD — Septo-Optic Dysplasia (AD/AR)',
  GLI2:    'HPE9/CPHD — Holoprosencephaly + SCMI (AD)',
  CABLES1: 'Familial Corticotropinoma — Cushing\'s Disease (AD)',
};

const TUMOR_GENES = ['AIP', 'PRKAR1A', 'CABLES1'];
const CPHD_GENES  = ['PROP1', 'POU1F1', 'LHX3', 'HESX1', 'GLI2'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Pituitary Disorders atlas…</p>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger m-4"><strong>Error:</strong> {msg}</div>;
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-3 mb-3">
      <div className="card h-100 shadow-sm border-0">
        <div className="card-body text-center p-3">
          <div className="fw-bold fs-3" style={{ color: color || '#1a237e' }}>{value}</div>
          <div className="small text-muted">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 px-3 mb-2`} style={{ fontSize: '0.85rem' }}>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const agg = ov.aggregate_clinical || {};
  return (
    <div>
      <h4 className="fw-bold mb-1" style={{ color: '#1a237e' }}>{ov.atlas_name}</h4>
      <p className="text-muted mb-3">{ov.subtitle} · {ov.n_patients} patients · {ov.gene_count} genes · Seeds {ov.seeds}</p>

      {/* KPI row */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={ov.n_patients} color="#1a237e" />
        <KPI label="Genes" value={ov.gene_count} color="#880e4f" />
        <KPI label="CPHD Deficiency %" value={`${agg.combined_hormone_deficiency_pct ?? '—'}%`} color="#4a148c" />
        <KPI label="GH Deficiency %" value={`${agg.gh_deficiency_pct ?? '—'}%`} color="#0d47a1" />
        <KPI label="ACTH Crisis Risk %" value={`${agg.acth_crisis_risk_pct ?? '—'}%`} color="#b71c1c" />
        <KPI label="Tumour Predisposition %" value={`${agg.tumour_predisposition_pct ?? '—'}%`} color="#1b5e20" />
        <KPI label="Rigid Spine %" value={`${agg.rigid_spine_pct ?? '—'}%`} color="#1b5e20" />
        <KPI label="Cardiac Surveillance %" value={`${agg.cardiac_surveillance_required_pct ?? '—'}%`} color="#880e4f" />
      </div>

      {/* Clinical rules */}
      <h5 className="fw-bold mb-2" style={{ color: '#37474f' }}>Critical Clinical Rules</h5>
      {(ov.key_clinical_rules || []).map((r, i) => (
        <Alert key={i} text={r} variant={i % 3 === 0 ? 'danger' : i % 3 === 1 ? 'warning' : 'info'} />
      ))}

      {/* Gene summary mini-cards */}
      <h5 className="fw-bold mt-4 mb-3">Gene Summary</h5>
      <div className="row g-3">
        {(ov.gene_summary || []).map(g => (
          <div key={g.gene} className="col-md-6">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-header py-2" style={{ background: GENE_COLORS[g.gene] || '#333', color: '#fff' }}>
                <strong>{g.gene}</strong> — {g.protein} · {g.aa} · {g.locus}
                <span className="badge bg-light text-dark ms-2" style={{ fontSize: '0.7rem' }}>{g.inheritance}</span>
              </div>
              <div className="card-body py-2 px-3">
                <p className="mb-1" style={{ fontSize: '0.82rem' }}><strong>Phenotype:</strong> {g.phenotype_short}</p>
                <p className="mb-0 text-danger" style={{ fontSize: '0.8rem' }}><strong>Hallmark:</strong> {g.hallmark_short}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div className="table-responsive">
      <table className="table table-sm table-bordered align-middle" style={{ fontSize: '0.8rem' }}>
        <thead className="table-dark">
          <tr>
            <th>Gene</th><th>Protein</th><th>aa</th><th>Locus</th>
            <th>OMIM Gene</th><th>OMIM Disease</th><th>Inheritance</th>
            <th>Disease / Phenotype</th><th>Hallmark</th>
          </tr>
        </thead>
        <tbody>
          {data.map(g => (
            <tr key={g.gene}>
              <td><strong style={{ color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</strong></td>
              <td>{g.protein}</td>
              <td>{g.aa}</td>
              <td>{g.locus}</td>
              <td>{g.omim_gene}</td>
              <td>{g.omim_disease}</td>
              <td><span className="badge" style={{ background: '#37474f', color: '#fff', fontSize: '0.7rem' }}>{g.inheritance?.split(';')[0]}</span></td>
              <td style={{ maxWidth: 220 }}>{(g.phenotype || '').slice(0, 120)}{g.phenotype?.length > 120 ? '…' : ''}</td>
              <td style={{ maxWidth: 180 }}>{g.key_hallmarks?.[0] || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const [selected, setSelected] = useState(data[0]?.gene || '');
  const gene = data.find(g => g.gene === selected);

  return (
    <div className="row">
      {/* Gene selector */}
      <div className="col-md-2 mb-3">
        <div className="list-group">
          {data.map(g => (
            <button
              key={g.gene}
              onClick={() => setSelected(g.gene)}
              className={`list-group-item list-group-item-action py-1 px-2 ${selected === g.gene ? 'active' : ''}`}
              style={{
                background: selected === g.gene ? GENE_COLORS[g.gene] : '',
                borderColor: GENE_COLORS[g.gene] || '#ccc',
                fontSize: '0.82rem',
                color: selected === g.gene ? '#fff' : GENE_COLORS[g.gene] || '#333',
              }}
            >
              {g.gene}
            </button>
          ))}
        </div>
      </div>

      {/* Gene detail */}
      <div className="col-md-10">
        {gene && (
          <>
            <div className="card border-0 shadow-sm mb-3">
              <div className="card-header py-2" style={{ background: GENE_COLORS[gene.gene] || '#333', color: '#fff' }}>
                <strong>{gene.gene}</strong> — {gene.protein} · {gene.aa} · {gene.locus} · {gene.inheritance?.split(';')[0]}
              </div>
              <div className="card-body py-2">
                <p style={{ fontSize: '0.82rem' }}><strong>Mechanism:</strong> {gene.gene_class}</p>
                <p style={{ fontSize: '0.82rem' }}><strong>Phenotype:</strong> {gene.phenotype}</p>
              </div>
            </div>

            {/* Hallmarks */}
            <h6 className="fw-bold mt-2">Key Hallmarks</h6>
            {(gene.key_hallmarks || []).map((h, i) => (
              <Alert key={i} text={h} variant="danger" />
            ))}

            {/* Treatment alerts */}
            <h6 className="fw-bold mt-3">Treatment Alerts</h6>
            {(gene.treatment_alerts || []).map((t, i) => (
              <Alert key={i} text={t} variant="warning" />
            ))}

            {/* DDx */}
            <h6 className="fw-bold mt-3">Key DDx</h6>
            {(gene.ddx || []).map((d, i) => (
              <Alert key={i} text={d} variant="info" />
            ))}

            {/* Cohort stats */}
            {gene.cohort_stats && Object.keys(gene.cohort_stats).length > 0 && (
              <>
                <h6 className="fw-bold mt-3">Cohort Statistics (n={gene.n_patients})</h6>
                <div className="row g-2">
                  {Object.entries(gene.cohort_stats).filter(([, v]) => typeof v === 'number').map(([k, v]) => (
                    <div key={k} className="col-6 col-md-4">
                      <div className="p-2 rounded" style={{ background: '#f5f5f5', fontSize: '0.8rem' }}>
                        <span className="fw-bold">{v}%</span> {k.replace(/_/g, ' ').replace(/pct$/, '').trim()}
                      </div>
                    </div>
                  ))}
                  {gene.cohort_stats.hpe_severity_breakdown && (
                    <div className="col-12">
                      <div className="p-2 rounded" style={{ background: '#e3f2fd', fontSize: '0.8rem' }}>
                        <strong>HPE severity:</strong>{' '}
                        {Object.entries(gene.cohort_stats.hpe_severity_breakdown).map(([k, v]) =>
                          `${k}: ${v}%`
                        ).join(' · ')}
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.definitions || [];
  return (
    <div>
      <p className="text-muted mb-3">{data.total_definitions} clinical definitions · {data.atlas}</p>
      {defs.map((d, i) => (
        <div key={i} className="card border-0 shadow-sm mb-3">
          <div className="card-header py-2" style={{ background: '#1a237e', color: '#fff', fontSize: '0.9rem' }}>
            <strong>{d.term}</strong>
          </div>
          <div className="card-body py-2 px-3">
            <p className="mb-1 fw-semibold" style={{ fontSize: '0.85rem' }}>{d.short}</p>
            <p className="mb-2" style={{ fontSize: '0.82rem', color: '#555' }}>{d.detail}</p>
            {d.clinical_rule && (
              <div className="alert alert-danger py-1 px-2 mb-0" style={{ fontSize: '0.8rem' }}>
                <strong>Clinical Rule:</strong> {d.clinical_rule}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function PituitaryDisordersAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/pituitary-disorders-atlas`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  const renderTab = () => {
    switch (tab) {
      case 'Overview':      return <OverviewTab data={overview} />;
      case 'Gene Table':    return <GeneTableTab data={breakdown} />;
      case 'Clinical Atlas':return <ClinicalAtlasTab data={breakdown} />;
      case 'Definitions':   return <DefinitionsTab data={definitions} />;
      default:              return null;
    }
  };

  return (
    <div className="container-fluid py-3 px-4" style={{ maxWidth: 1400 }}>
      {/* Page header */}
      <div className="d-flex align-items-center mb-3">
        <span style={{ fontSize: '2rem', marginRight: 12 }}>🧠</span>
        <div>
          <h3 className="mb-0 fw-bold" style={{ color: '#1a237e' }}>Pituitary-Disorders-Atlas</h3>
          <small className="text-muted">
            Complete 8-Gene Hereditary Pituitary Disorders Atlas ·{' '}
            <span className="badge" style={{ background: '#1a237e', color: '#fff' }}>AIP</span>{' '}
            <span className="badge" style={{ background: '#880e4f', color: '#fff' }}>PRKAR1A</span>{' '}
            <span className="badge" style={{ background: '#4a148c', color: '#fff' }}>PROP1</span>{' '}
            <span className="badge" style={{ background: '#0d47a1', color: '#fff' }}>POU1F1</span>{' '}
            <span className="badge" style={{ background: '#1b5e20', color: '#fff' }}>LHX3</span>{' '}
            <span className="badge" style={{ background: '#e65100', color: '#fff' }}>HESX1</span>{' '}
            <span className="badge" style={{ background: '#37474f', color: '#fff' }}>GLI2</span>{' '}
            <span className="badge" style={{ background: '#b71c1c', color: '#fff' }}>CABLES1</span>
            {' '}· 320 patients (8×40, seeds 1310–1317)
          </small>
        </div>
      </div>

      {/* Tumour vs CPHD category bar */}
      <div className="d-flex gap-2 mb-3 flex-wrap">
        <span className="badge" style={{ background: '#1a237e', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          🔬 Tumour Predisposition: AIP · PRKAR1A · CABLES1
        </span>
        <span className="badge" style={{ background: '#4a148c', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          🧬 Developmental CPHD: PROP1 · POU1F1 · LHX3 · HESX1 · GLI2
        </span>
      </div>

      {/* Tab navigation */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active fw-semibold' : ''}`}
              onClick={() => setTab(t)}
              style={tab === t ? { color: '#1a237e', borderBottomColor: '#1a237e', borderBottomWidth: 2 } : {}}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {renderTab()}
    </div>
  );
}
