'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  HFE:     '#b71c1c',  // deep red — most common HH, C282Y
  HJV:     '#880e4f',  // deep magenta — juvenile HH, cardiomyopathy
  HAMP:    '#4a148c',  // deep purple — hepcidin structural gene
  TFR2:    '#1a237e',  // deep navy — adult HH, HFE normal
  SLC40A1: '#e65100',  // deep amber — ferroportin, AD, two subtypes
  TMPRSS6: '#1b5e20',  // deep green — IRIDA, oral iron fails
  CP:      '#37474f',  // dark slate — aceruloplasminemia, neurodegeneration
  FTL:     '#f57f17',  // amber-gold — HHCS, do NOT venesect
};

const GENE_DISEASE = {
  HFE:     'HH Type 1 — C282Y — AR',
  HJV:     'HH Type 2A — Juvenile — AR',
  HAMP:    'HH Type 2B — Hepcidin — AR',
  TFR2:    'HH Type 3 — Adult HFE-normal — AR',
  SLC40A1: 'HH Type 4 — Ferroportin — AD',
  TMPRSS6: 'IRIDA — Oral Iron Fails — AR',
  CP:      'Aceruloplasminemia — Neurodegeneration+DM+Retina — AR',
  FTL:     'HHCS — Cataracts — AD — DO NOT VENESECT',
};

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Iron Disorders atlas…</p>
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
          <div className="fw-bold fs-3" style={{ color: color || '#b71c1c' }}>{value}</div>
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
  const ac = data.aggregate_clinical || {};
  const rules = data.key_treatment_rules || [];
  return (
    <div>
      <div className="row mb-4">
        <div className="col-12">
          <h4 className="fw-bold text-danger">{data.subtitle}</h4>
          <p className="text-muted mb-1">
            {data.n_patients} patients · {data.n_genes} genes · seeds {data.seeds}
          </p>
          <span className="badge bg-danger me-2">8 Genes</span>
          <span className="badge bg-secondary me-2">AR × 6</span>
          <span className="badge bg-warning text-dark me-2">AD × 2</span>
        </div>
      </div>

      <div className="row mb-4">
        <KPI label="HFE Arthropathy MCP" value={`${ac.hfe_arthropathy_mcp_pct}%`} color="#b71c1c" />
        <KPI label="HFE Cirrhosis" value={`${ac.hfe_hh1_cirrhosis_pct}%`} color="#c62828" />
        <KPI label="HJV Cardiomyopathy" value={`${ac.hjv_cardiomyopathy_pct}%`} color="#880e4f" />
        <KPI label="HJV Hypogonadism" value={`${ac.hjv_hypogonadism_pct}%`} color="#6a1b4d" />
        <KPI label="TMPRSS6 Oral Iron Fails" value={`${ac.tmprss6_oral_iron_fails_pct}%`} color="#1b5e20" />
        <KPI label="TMPRSS6 IV Iron Response" value={`${ac.tmprss6_iv_iron_response_pct}%`} color="#2e7d32" />
        <KPI label="CP Neurodegeneration" value={`${ac.cp_neurodegeneration_pct}%`} color="#37474f" />
        <KPI label="CP Triple Triad All" value={`${ac.cp_triple_triad_all_pct}%`} color="#455a64" />
        <KPI label="FTL Erroneous Venesection" value={`${ac.ftl_erroneous_venesection_pct}%`} color="#e65100" />
        <KPI label="FTL Iatrogenic Anaemia" value={`${ac.ftl_anemia_from_venesection_pct}%`} color="#f57f17" />
      </div>

      <div className="card border-danger mb-4">
        <div className="card-header bg-danger text-white fw-bold">⚠️ Critical Treatment Rules — Iron Disorders</div>
        <div className="card-body p-3">
          {rules.map((r, i) => <Alert key={i} text={r} variant={r.includes('FAILS') || r.includes('VENESECT') ? 'danger' : 'warning'} />)}
        </div>
      </div>

      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header bg-dark text-white fw-bold">Diseases — AR Inheritance (hepcidin-axis)</div>
            <ul className="list-group list-group-flush">
              {(data.inheritance_pattern?.AR || []).map(g => (
                <li key={g} className="list-group-item d-flex align-items-center">
                  <span className="badge me-2" style={{ background: GENE_COLORS[g] }}>{g}</span>
                  <span className="small">{GENE_DISEASE[g]}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header bg-warning text-dark fw-bold">Diseases — AD Inheritance</div>
            <ul className="list-group list-group-flush">
              {(data.inheritance_pattern?.AD || []).map(g => (
                <li key={g} className="list-group-item d-flex align-items-center">
                  <span className="badge me-2" style={{ background: GENE_COLORS[g] }}>{g}</span>
                  <span className="small">{GENE_DISEASE[g]}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const bd = data.breakdown || {};
  return (
    <div className="table-responsive">
      <table className="table table-bordered table-hover align-middle small">
        <thead className="table-dark">
          <tr>
            <th>Gene</th>
            <th>Disease</th>
            <th>Protein / aa</th>
            <th>Locus</th>
            <th>Inheritance</th>
            <th>OMIM Gene</th>
            <th>OMIM Disease</th>
            <th>N</th>
          </tr>
        </thead>
        <tbody>
          {Object.values(bd).map(g => (
            <tr key={g.gene}>
              <td><span className="badge fw-bold" style={{ background: GENE_COLORS[g.gene] }}>{g.gene}</span></td>
              <td><span className="small">{GENE_DISEASE[g.gene]}</span></td>
              <td><code className="small">{g.protein?.split(' ')[0]} · {g.aa}</code></td>
              <td><code>{g.locus}</code></td>
              <td><span className="badge bg-secondary">{g.inheritance?.split(';')[0]}</span></td>
              <td><a href={`https://omim.org/entry/${g.omim_gene}`} target="_blank" rel="noreferrer">{g.omim_gene}</a></td>
              <td><a href={`https://omim.org/entry/${g.omim_disease}`} target="_blank" rel="noreferrer">{g.omim_disease}</a></td>
              <td>{g.cohort_stats?.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  const [sel, setSel] = useState(null);
  if (!data) return <Loading />;
  const bd = data.breakdown || {};
  const genes = Object.keys(bd);
  const active = sel || genes[0];
  const g = bd[active] || {};
  const stats = g.cohort_stats || {};

  return (
    <div className="row">
      <div className="col-md-3 mb-3">
        <div className="list-group">
          {genes.map(gn => (
            <button
              key={gn}
              className={`list-group-item list-group-item-action d-flex align-items-center ${active === gn ? 'active' : ''}`}
              style={active === gn ? { background: GENE_COLORS[gn], borderColor: GENE_COLORS[gn] } : {}}
              onClick={() => setSel(gn)}
            >
              <span className="fw-bold me-2">{gn}</span>
              <span className="small">{g.gene === gn ? g.aa : bd[gn]?.aa}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="col-md-9">
        <div className="card border-0 shadow-sm">
          <div className="card-header text-white fw-bold" style={{ background: GENE_COLORS[active] }}>
            {active} — {GENE_DISEASE[active]}
          </div>
          <div className="card-body">
            <p className="small mb-2"><strong>Protein:</strong> {g.protein}</p>
            <p className="small mb-2"><strong>Locus:</strong> {g.locus} · <strong>Size:</strong> {g.aa} · <strong>kDa:</strong> {g.kDa}</p>
            <p className="small mb-3"><strong>Inheritance:</strong> {g.inheritance}</p>

            <h6 className="fw-bold text-danger">⚠️ Hallmark</h6>
            <p className="small mb-3 text-danger">{g.hallmark}</p>

            <h6 className="fw-bold">Phenotype</h6>
            <p className="small mb-3">{g.phenotype}</p>

            <h6 className="fw-bold text-warning">Treatment Alerts</h6>
            <ul className="small">
              {(g.treatment_alerts || []).map((a, i) => <li key={i} className="mb-1">{a}</li>)}
            </ul>

            <h6 className="fw-bold text-info">Key DDx</h6>
            <p className="small mb-3">{g.key_ddx}</p>

            <h6 className="fw-bold text-secondary">Cohort Stats (n={stats.n})</h6>
            <div className="row">
              {Object.entries(stats).filter(([k]) => k !== 'n' && k !== 'sex_m_pct').map(([k, v]) => (
                <div key={k} className="col-6 col-lg-4 mb-2">
                  <div className="card border-0 bg-light p-2">
                    <div className="fw-bold small" style={{ color: GENE_COLORS[active] }}>
                      {typeof v === 'number' ? (k.endsWith('_pct') ? `${v}%` : v) : String(v)}
                    </div>
                    <div className="text-muted" style={{ fontSize: '0.7rem' }}>{k.replace(/_/g, ' ')}</div>
                  </div>
                </div>
              ))}
            </div>

            <h6 className="fw-bold mt-3">Gene Function</h6>
            <p className="small">{g.gene_class}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.definitions || {};
  return (
    <div>
      {Object.entries(defs).map(([k, v]) => (
        <div key={k} className="card border-0 shadow-sm mb-3">
          <div className="card-header bg-dark text-white fw-bold" style={{ fontSize: '0.9rem' }}>
            {k.replace(/_/g, ' ')}
          </div>
          <div className="card-body">
            <p className="small mb-0" style={{ whiteSpace: 'pre-wrap' }}>{v}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function IronDisordersAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/iron-disorders-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/iron-disorders-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/iron-disorders-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4">
      <div className="mb-4">
        <h2 className="fw-bold text-danger mb-1">
          🧬 Iron-Disorders-Atlas
        </h2>
        <p className="text-muted small">
          Complete 8-Gene Hereditary Iron Metabolism Disorders Atlas ·
          HFE · HJV · HAMP · TFR2 · SLC40A1 · TMPRSS6 · CP · FTL ·
          320 patients (8×40, seeds 1286–1293)
        </p>
      </div>

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active' : ''}`}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'      && <OverviewTab      data={overview} />}
      {tab === 'Gene Table'    && <GeneTableTab     data={breakdown} />}
      {tab === 'Clinical Atlas'&& <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'   && <DefinitionsTab   data={definitions} />}
    </div>
  );
}
