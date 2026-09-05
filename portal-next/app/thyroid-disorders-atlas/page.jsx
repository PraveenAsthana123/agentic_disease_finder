'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  TSHR:    '#004d40',  // deep teal — TSH receptor, congenital hypothyroidism
  PAX8:    '#1565c0',  // deep blue — thyroid dysgenesis, ectopy
  TPO:     '#6a1b9a',  // deep purple — organification defect, perchlorate positive
  TG:      '#37474f',  // dark slate — thyroglobulin, low Tg paradox
  SLC5A5:  '#e65100',  // deep amber — NIS, iodide transport defect
  DUOX2:   '#558b2f',  // deep green — dual oxidase, transient hypothyroidism
  SLC26A4: '#880e4f',  // deep magenta — Pendred syndrome, EVA, deafness
  FOXE1:   '#b71c1c',  // deep red — Bamforth-Lazarus, choanal atresia emergency
};

const GENE_DISEASE = {
  TSHR:    'RTSH — Resistance to TSH — AR/AD (GOF)',
  PAX8:    'Thyroid Dysgenesis — AD — Ectopy / Agenesis',
  TPO:     'Dyshormonogenesis DH1 — AR — Organification Defect',
  TG:      'Dyshormonogenesis DH3 — AR — Goiter + Low Tg',
  SLC5A5:  'Iodide Transport Defect — AR — NIS LOF',
  DUOX2:   'Dyshormonogenesis DH6 — AR/monoallelic — Transient CHT',
  SLC26A4: 'Pendred Syndrome — AR — Deafness + EVA + Goiter',
  FOXE1:   'Bamforth-Lazarus Syndrome — AR — Athyreosis + Choanal Atresia',
};

const AR_GENES = ['TSHR', 'TPO', 'TG', 'SLC5A5', 'DUOX2', 'SLC26A4', 'FOXE1'];
const AD_GENES = ['PAX8'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-success" role="status" />
      <p className="mt-3 text-muted">Loading Thyroid Disorders atlas…</p>
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
          <div className="fw-bold fs-3" style={{ color: color || '#004d40' }}>{value}</div>
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

const TREATMENT_RULES = [
  'PAX8 ECTOPIC THYROID: DO NOT surgically remove ectopic sublingual thyroid without nuclear medicine scan confirming it is non-functional — it may be the ONLY thyroid tissue.',
  'FOXE1 CHOANAL ATRESIA: Bilateral choanal atresia in a neonate = AIRWAY EMERGENCY — McGovern nipple or immediate surgical repair; start levothyroxine day 1 (athyreosis).',
  'DUOX2 MONOALLELIC: Transient neonatal hypothyroidism from monoallelic DUOX2 → attempt levothyroxine cessation trial at age 3 under endocrinology supervision (not lifelong).',
  'PERCHLORATE TEST: A positive perchlorate discharge test (>10% iodide released) indicates an organification defect — TPO or DUOX2; NEGATIVE test with absent radioiodine uptake = NIS (SLC5A5).',
  'SLC26A4 (PENDRED): EVA (enlarged vestibular aqueduct) on CT/MRI temporal bone is PATHOGNOMONIC for SLC26A4 — any child with sensorineural deafness + goiter needs temporal bone CT.',
  'TG PARADOX: Very LOW serum thyroglobulin despite a LARGE goiter = TG LOF dyshormonogenesis — do NOT interpret low Tg as absence of thyroid tissue.',
  'TSHR RTSH: If neonatal hypothyroidism persists beyond age 3 years and maternal anti-TPO antibodies are initially blamed — sequence TSHR (maternal antibodies clear by 6 months).',
  'NIS (SLC5A5): IV iodide supplementation can partially bypass the NIS defect at high concentrations — adjunct to levothyroxine in some patients.',
];

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ac = data.aggregate_clinical || {};
  return (
    <div>
      <div className="row mb-4">
        <div className="col-12">
          <h4 className="fw-bold text-success">{data.subtitle}</h4>
          <p className="text-muted mb-1">
            {data.n_patients} patients · {data.gene_count} genes · seeds {data.seeds}
          </p>
          <span className="badge bg-success me-2">8 Genes</span>
          <span className="badge bg-secondary me-2">AR × 7</span>
          <span className="badge bg-primary me-2">AD × 1 (PAX8)</span>
        </div>
      </div>

      <div className="row mb-4">
        <KPI label="NBS Detected" value={`${ac.neonatal_screening_detected_pct}%`} color="#004d40" />
        <KPI label="Levothyroxine Rx" value={`${ac.levothyroxine_prescribed_pct}%`} color="#00695c" />
        <KPI label="Goiter (Any)" value={`${ac.goiter_pct}%`} color="#558b2f" />
        <KPI label="Sensorineural Deafness" value={`${ac.deafness_pct}%`} color="#880e4f" />
        <KPI label="Choanal Atresia" value={`${ac.choanal_atresia_pct}%`} color="#b71c1c" />
        <KPI label="Perchlorate Positive" value={`${ac.perchlorate_positive_pct}%`} color="#6a1b9a" />
        <KPI label="Radioiodine Absent" value={`${ac.radioiodine_absent_pct}%`} color="#e65100" />
        <KPI label="Median TSH (mU/L)" value={`${ac.median_tsh_at_diagnosis}`} color="#1565c0" />
      </div>

      <div className="card border-success mb-4">
        <div className="card-header bg-success text-white fw-bold">⚠️ Critical Treatment Rules — Hereditary Thyroid Disorders</div>
        <div className="card-body p-3">
          {TREATMENT_RULES.map((r, i) => (
            <Alert
              key={i}
              text={r}
              variant={
                r.includes('EMERGENCY') || r.includes('CHOANAL') || r.includes('AIRWAY')
                  ? 'danger'
                  : r.includes('DO NOT') || r.includes('PARADOX')
                  ? 'warning'
                  : 'info'
              }
            />
          ))}
        </div>
      </div>

      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header bg-dark text-white fw-bold">AR Genes (7) — Dyshormonogenesis + Dysgenesis</div>
            <ul className="list-group list-group-flush">
              {AR_GENES.map(g => (
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
            <div className="card-header bg-primary text-white fw-bold">AD Gene (1) — Thyroid Dysgenesis</div>
            <ul className="list-group list-group-flush">
              {AD_GENES.map(g => (
                <li key={g} className="list-group-item d-flex align-items-center">
                  <span className="badge me-2" style={{ background: GENE_COLORS[g] }}>{g}</span>
                  <span className="small">{GENE_DISEASE[g]}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="card border-0 shadow-sm mt-3 p-3">
            <div className="fw-bold small text-success mb-2">Clinical Pearl</div>
            <p className="small mb-0 text-muted">{data.clinical_pearl}</p>
          </div>
          <div className="card border-0 shadow-sm mt-3 p-3">
            <div className="fw-bold small text-secondary mb-2">Cascade Testing</div>
            <p className="small mb-0 text-muted">{data.cascade_testing_note}</p>
          </div>
        </div>
      </div>

      <div className="row">
        {(data.gene_summary || []).map(gs => (
          <div key={gs.gene} className="col-md-6 col-lg-3 mb-3">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-header text-white fw-bold small" style={{ background: GENE_COLORS[gs.gene] }}>
                {gs.gene} — {gs.aa} — {gs.locus}
              </div>
              <div className="card-body p-2">
                <p className="small mb-1"><strong>Inheritance:</strong> {gs.inheritance}</p>
                <p className="small mb-1">{gs.phenotype_short}</p>
                <p className="small text-danger mb-0"><strong>Hallmark:</strong> {gs.hallmark_short}</p>
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
  const bd = data.breakdown || {};
  return (
    <div className="table-responsive">
      <table className="table table-bordered table-hover align-middle small">
        <thead className="table-dark">
          <tr>
            <th>Gene</th>
            <th>Disease / Syndrome</th>
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
              <td><span className="badge bg-secondary">{g.inheritance?.split(';')[0]?.split('(')[0]?.trim()}</span></td>
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
              <span className="small">{bd[gn]?.aa}</span>
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

export default function ThyroidDisordersAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/thyroid-disorders-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/thyroid-disorders-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/thyroid-disorders-atlas/definitions`).then(r => r.json()),
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
        <h2 className="fw-bold text-success mb-1">
          🦋 Thyroid-Disorders-Atlas
        </h2>
        <p className="text-muted small">
          Complete 8-Gene Hereditary Thyroid Disorders Atlas ·
          TSHR · PAX8 · TPO · TG · SLC5A5 · DUOX2 · SLC26A4 · FOXE1 ·
          320 patients (8×40, seeds 1302–1309)
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

      {tab === 'Overview'       && <OverviewTab      data={overview} />}
      {tab === 'Gene Table'     && <GeneTableTab     data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'    && <DefinitionsTab   data={definitions} />}
    </div>
  );
}
