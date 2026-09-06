'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PRSS1: '#b71c1c',  // deep red — HP, trypsin GOF, cancer risk
  SPINK1:'#e65100',  // deep orange — modifier, N34S, NOT causal alone
  CTRC:  '#4a148c',  // deep purple — LOF, trypsin cleaving failure
  CPA1:  '#1b5e20',  // deep green — ER stress, misfolding, NOT trypsin
  CFTR:  '#006064',  // dark teal — CFTR-P, ductal bicarbonate, compound het
  CLDN2: '#37474f',  // dark slate — X-linked, male only, alcoholic pancreatitis
  CEL:   '#880e4f',  // deep crimson — MODY8, CEL-HYB, VNTR, dual failure
  CASR:  '#1a237e',  // deep navy — hypercalcemia, cinacalcet, treat Ca first
};

const GENE_DISEASE = {
  PRSS1: 'Hereditary Pancreatitis (AD) — PRSS1; R122H/N29I GOF; Trypsin Autolysis Site Eliminated; 40% PDAC Risk by 70; Annual MRI',
  SPINK1:'Chronic Pancreatitis modifier (AR/mod) — SPINK1; N34S Risk Allele 3x; NOT Causal Alone; 2nd Hit Required',
  CTRC:  'Chronic Pancreatitis (AR) — CTRC; Chymotrypsin C LOF; Fails to Cleave Trypsin Arg122; A73T/G61del',
  CPA1:  'Hereditary Pancreatitis ER Stress (AD) — CPA1; Misfolding; DDIT3/CHOP; NOT Trypsin Pathway',
  CFTR:  'CFTR-related Pancreatitis (AR) — CFTR; Compound Heterozygote; DeltaF508+R117H-5T; Borderline Sweat Test',
  CLDN2: 'Alcoholic Pancreatitis risk (XL) — CLDN2; Xq22.3; Hemizygous Males 5x Risk; Females Protected',
  CEL:   'MODY8 + EPI (AD) — CEL; CEL-HYB Hybrid Allele; Standard Sequencing MISSES; VNTR Required; Dual Failure',
  CASR:  'Hypercalcemia-Induced Pancreatitis (AD) — CASR; CaSR LOF; Treat Hypercalcemia First; Cinacalcet',
};

const TRYPSIN_GENES = ['PRSS1', 'SPINK1', 'CTRC'];
const ER_STRESS_GENES = ['CPA1'];
const DUCTAL_GENES = ['CFTR', 'CLDN2'];
const EPI_MODY_GENES = ['CEL'];
const METABOLIC_GENES = ['CASR'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Hereditary Pancreatitis Atlas…</p>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger m-4"><strong>Error:</strong> {msg}</div>;
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-sm-4 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-body text-center p-2" style={{ borderTop: `4px solid ${color}` }}>
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function AlertBadge({ text, color = '#b71c1c' }) {
  return (
    <span className="badge me-1 mb-1" style={{ background: color, fontSize: '0.7rem' }}>
      {text}
    </span>
  );
}

/* ── OVERVIEW TAB ── */
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats;

  const statItems = [
    { key: 'recurrent_acute_pancreatitis',         label: 'Recurrent Acute Pancreatitis',    color: '#b71c1c' },
    { key: 'chronic_pancreatitis',                 label: 'Chronic Pancreatitis',             color: '#b71c1c' },
    { key: 'exocrine_pancreatic_insufficiency',    label: 'Exocrine Pancreatic Insufficiency',color: '#4a148c' },
    { key: 'diabetes_mellitus_type3c',             label: 'Type 3c Diabetes',                 color: '#4a148c' },
    { key: 'pancreatic_ductal_stones',             label: 'Pancreatic Ductal Stones',         color: '#e65100' },
    { key: 'pancreatic_cancer_by_70',              label: 'PDAC Risk by Age 70 (PRSS1)',      color: '#b71c1c' },
    { key: 'steatorrhoea',                         label: 'Steatorrhoea',                     color: '#880e4f' },
    { key: 'male_sex',                             label: 'Male Sex (CLDN2 cohort)',           color: '#37474f' },
    { key: 'hypercalcemia_ionised_ca_3mmol',       label: 'Hypercalcaemia >3 mmol/L (CASR)', color: '#1a237e' },
    { key: 'vntr_analysis_required_for_dx',        label: 'VNTR Analysis Required (CEL)',     color: '#880e4f' },
    { key: 'alcohol_as_2nd_hit',                   label: 'Alcohol as 2nd Hit',               color: '#37474f' },
    { key: 'pain_requiring_opioids',               label: 'Pain Requiring Opioids',           color: '#006064' },
  ];

  return (
    <div>
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={data.total_patients} color="#37474f" />
        <KPI label="Genes" value={data.genes?.length} color="#37474f" />
        <KPI label="Trypsin Pathway" value="3" color="#b71c1c" />
        <KPI label="ER Stress / Ductal / EPI" value="4" color="#4a148c" />
        <KPI label="Metabolic" value="1" color="#1a237e" />
        <KPI label="Seeds" value={data.seed_range} color="#37474f" />
      </div>

      <div className="alert alert-danger mb-3">
        <strong>🚨 PRSS1 R122H/N29I:</strong> Test HOT SPOTS first — if negative, full PRSS1 gene sequencing. 40% cumulative pancreatic cancer risk by age 70. Annual MRI from age 40 mandatory.
      </div>
      <div className="alert alert-warning mb-3">
        <strong>⚠️ SPINK1 N34S:</strong> NOT causal alone — modifier allele; risk 3x in homozygotes only when 2nd hit present (PRSS1 GOF / CTRC LOF / alcohol / tropical diet). Report as modifier, not pathogenic.
      </div>
      <div className="alert alert-info mb-3">
        <strong>ℹ️ CEL-HYB:</strong> Standard sequencing (Sanger/NGS/exome) MISSES CEL-HYB — VNTR analysis required. MODY8 hallmark: Diabetes PLUS Exocrine Insufficiency (dual pancreatic failure).
      </div>
      <div className="alert alert-secondary mb-4">
        <strong>🔬 CLDN2:</strong> Male only (hemizygous Xq22.3) — 5x alcoholic pancreatitis risk. Female heterozygotes have NO significant risk (X-inactivation). CASR: treat hypercalcaemia FIRST — pancreatitis resolves.
      </div>

      <h6 className="fw-bold mb-3">Aggregate Clinical Features (320 patients, 8 genes)</h6>
      <div className="row g-2 mb-4">
        {statItems.map(({ key, label, color }) => s?.[key] != null && (
          <div key={key} className="col-6 col-md-4 col-lg-3">
            <div className="card border-0 shadow-sm">
              <div className="card-body p-2" style={{ borderLeft: `4px solid ${color}` }}>
                <div className="fw-bold" style={{ color }}>{s[key]}%</div>
                <div className="text-muted small">{label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mb-2">Gene Classification</h6>
      <div className="row g-3 mb-4">
        <div className="col-md-4">
          <div className="card border-0 shadow-sm">
            <div className="card-header" style={{ background: '#b71c1c', color: 'white' }}>
              <strong>Trypsin Pathway (3 genes)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {TRYPSIN_GENES.map(g => (
                <li key={g} className="list-group-item py-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span>{' — '}
                  <span className="text-muted">{GENE_DISEASE[g].split('—')[1]?.split(';')[0]}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-3">
          <div className="card border-0 shadow-sm">
            <div className="card-header" style={{ background: '#1b5e20', color: 'white' }}>
              <strong>ER Stress (1 gene)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {ER_STRESS_GENES.map(g => (
                <li key={g} className="list-group-item py-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span>{' — '}
                  <span className="text-muted">{GENE_DISEASE[g].split('—')[1]?.split(';')[0]}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-3">
          <div className="card border-0 shadow-sm">
            <div className="card-header" style={{ background: '#006064', color: 'white' }}>
              <strong>Ductal / Risk Modifier (2 genes)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {DUCTAL_GENES.map(g => (
                <li key={g} className="list-group-item py-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span>{' — '}
                  <span className="text-muted">{GENE_DISEASE[g].split('—')[1]?.split(';')[0]}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-2">
          <div className="card border-0 shadow-sm">
            <div className="card-header" style={{ background: '#880e4f', color: 'white' }}>
              <strong>EPI / Metabolic (2 genes)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {[...EPI_MODY_GENES, ...METABOLIC_GENES].map(g => (
                <li key={g} className="list-group-item py-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span>{' — '}
                  <span className="text-muted">{GENE_DISEASE[g].split('—')[1]?.split(';')[0]}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <h6 className="fw-bold mb-2">Top Clinical Alerts</h6>
      <div className="mb-3">
        {(data.top_alerts || []).map((a, i) => (
          <AlertBadge key={i} text={a}
            color={a.includes('PRSS1') ? '#b71c1c' : a.includes('SPINK1') ? '#e65100' :
                   a.includes('CPA1') ? '#1b5e20' : a.includes('CEL') ? '#880e4f' :
                   a.includes('CFTR') ? '#006064' : a.includes('CLDN2') ? '#37474f' :
                   a.includes('CASR') ? '#1a237e' : a.includes('CTRC') ? '#4a148c' : '#546e7a'} />
        ))}
      </div>

      <div className="row g-3">
        {Object.entries(data.diseases || {}).map(([gene, desc]) => (
          <div key={gene} className="col-12 col-md-6">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body p-3" style={{ borderLeft: `5px solid ${GENE_COLORS[gene] || '#546e7a'}` }}>
                <div className="fw-bold small mb-1" style={{ color: GENE_COLORS[gene] }}>{gene}</div>
                <div className="text-muted" style={{ fontSize: '0.78rem' }}>{desc}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── GENE TABLE TAB ── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);

  return (
    <div className="table-responsive">
      <table className="table table-sm table-hover align-middle">
        <thead className="table-dark">
          <tr>
            <th>Gene</th><th>Protein</th><th>aa</th><th>Locus</th>
            <th>Inheritance</th><th>OMIM Gene</th><th>OMIM Disease</th>
            <th>Organ System</th><th>N Patients</th>
          </tr>
        </thead>
        <tbody>
          {genes.map(g => (
            <tr key={g.gene}>
              <td><span className="fw-bold" style={{ color: GENE_COLORS[g.gene] }}>{g.gene}</span></td>
              <td style={{ fontSize: '0.8rem' }}>{g.protein}</td>
              <td>{g.aa}</td>
              <td><code style={{ fontSize: '0.75rem' }}>{g.locus}</code></td>
              <td>
                <span className={`badge ${g.inheritance?.startsWith('AR') ? 'bg-success' : g.inheritance?.startsWith('XL') || g.inheritance?.startsWith('X-linked') ? 'bg-warning text-dark' : 'bg-primary'}`}
                  style={{ fontSize: '0.65rem' }}>
                  {g.inheritance?.split(' ')[0]}
                </span>
              </td>
              <td><a href={`https://omim.org/entry/${g.omim_gene}`} target="_blank" rel="noreferrer"
                style={{ fontSize: '0.8rem' }}>{g.omim_gene}</a></td>
              <td><a href={`https://omim.org/entry/${g.omim_disease}`} target="_blank" rel="noreferrer"
                style={{ fontSize: '0.8rem' }}>{g.omim_disease}</a></td>
              <td style={{ fontSize: '0.75rem', maxWidth: 200 }}>{g.organ_system}</td>
              <td className="text-center">{g.n_patients}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const [selected, setSelected] = useState(Object.keys(data)[0]);
  const g = data[selected];
  if (!g) return null;

  return (
    <div className="row g-3">
      <div className="col-md-2">
        <div className="list-group list-group-flush">
          {Object.keys(data).map(gene => (
            <button key={gene}
              className={`list-group-item list-group-item-action py-1 px-2 ${selected === gene ? 'active' : ''}`}
              style={selected === gene ? { background: GENE_COLORS[gene], borderColor: GENE_COLORS[gene] } : {}}
              onClick={() => setSelected(gene)}>
              <span className="fw-bold small">{gene}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="col-md-10">
        <div className="card border-0 shadow-sm">
          <div className="card-header" style={{ background: GENE_COLORS[selected], color: 'white' }}>
            <strong>{g.gene}</strong> — {g.protein} | {g.aa} | {g.locus} | {g.inheritance?.split(' ')[0]}
          </div>
          <div className="card-body">
            <div className="row g-3 mb-3">
              <div className="col-md-6">
                <h6 className="fw-bold">Clinical Hallmarks</h6>
                <ul className="small mb-0">
                  {(g.hallmarks || []).map((h, i) => <li key={i} className="mb-1">{h}</li>)}
                </ul>
              </div>
              <div className="col-md-6">
                <h6 className="fw-bold">Treatment Alerts</h6>
                <ul className="small mb-0">
                  {(g.treatment_alerts || []).map((t, i) => <li key={i} className="mb-1">{t}</li>)}
                </ul>
              </div>
            </div>

            <div className="mb-3">
              <h6 className="fw-bold">Feature Frequencies ({g.n_patients} patients)</h6>
              <div className="row g-1">
                {Object.entries(g.stats || {}).map(([k, v]) => (
                  <div key={k} className="col-6 col-md-4">
                    <div className="d-flex align-items-center gap-2 small">
                      <div style={{ width: 40, height: 8, borderRadius: 4, background: '#e0e0e0', position: 'relative', flexShrink: 0 }}>
                        <div style={{ width: `${v}%`, height: '100%', borderRadius: 4, background: GENE_COLORS[selected] }} />
                      </div>
                      <span className="text-muted" style={{ fontSize: '0.7rem' }}>{k.replace(/_/g, ' ')} <strong>{v}%</strong></span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="mb-3">
              <h6 className="fw-bold">Etiology Distribution</h6>
              <ul className="small mb-0">
                {(g.etiology_distribution || []).map((e, i) => (
                  <li key={i}><strong>{Math.round(e.fraction * 100)}%</strong> — {e.etiology}</li>
                ))}
              </ul>
            </div>

            <div>
              <h6 className="fw-bold">Primary Treatment</h6>
              <p className="small mb-0 text-muted">{g.primary_treatment}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── DEFINITIONS TAB ── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;

  return (
    <div>
      <h6 className="fw-bold mb-3">Disease Classification</h6>
      {Object.entries(data.classification || {}).map(([cat, genes]) => (
        <div key={cat} className="mb-3">
          <h6 className="text-muted small fw-bold border-bottom pb-1">{cat.replace(/_/g, ' ')}</h6>
          <ul className="small">
            {Object.entries(genes).map(([k, v]) => (
              <li key={k}><strong>{k.replace(/_/g, ' ')}</strong>: {v}</li>
            ))}
          </ul>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4">Key Diagnostic Rules</h6>
      {Object.entries(data.key_diagnostic_rules || {}).map(([rule, text]) => (
        <div key={rule} className="mb-3 p-3 rounded" style={{ background: '#f8f9fa', borderLeft: '4px solid #37474f' }}>
          <div className="fw-bold small mb-1" style={{ color: '#37474f' }}>{rule.replace(/_/g, ' ')}</div>
          <div className="small text-muted">{text}</div>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4">Treatment Hierarchies</h6>
      {Object.entries(data.treatment_hierarchy || {}).map(([gene, steps]) => (
        <div key={gene} className="mb-3">
          <h6 className="small fw-bold" style={{ color: GENE_COLORS[gene.split('_')[0]] || '#37474f' }}>{gene.replace(/_/g, ' ')}</h6>
          <ol className="small mb-0">
            {steps.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
          </ol>
        </div>
      ))}
    </div>
  );
}

/* ── MAIN PAGE ── */
export default function HereditaryPancreatitisAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-pancreatitis-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-pancreatitis-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-pancreatitis-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, def]) => { setOverview(ov); setBreakdown(bd); setDefinitions(def); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4">
      <div className="mb-4">
        <h4 className="fw-bold mb-1">🧬 Hereditary Pancreatitis Atlas</h4>
        <p className="text-muted small mb-0">
          Complete 8-Gene Hereditary Pancreatitis and Exocrine Pancreatic Insufficiency Reference —
          PRSS1 (HP) · SPINK1 (modifier) · CTRC (CP) · CPA1 (ER stress) ·
          CFTR (CFTR-P) · CLDN2 (alcoholic-P) · CEL (MODY8/EPI) · CASR (hypercalcemia-P) |
          320 patients · 8×40 · seeds 1406–1413
        </p>
      </div>

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'       && <OverviewTab data={overview} />}
      {tab === 'Gene Table'     && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'    && <DefinitionsTab data={definitions} />}
    </div>
  );
}
