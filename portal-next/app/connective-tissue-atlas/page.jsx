'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  COL5A1:  '#1a237e',  // deep navy — cEDS, most common EDS, atrophic scars
  COL3A1:  '#b71c1c',  // deep red — vEDS, LIFE-THREATENING, arterial rupture
  TNXB:    '#1b5e20',  // deep green — clEDS, no scars, tenascin-X
  PLOD1:   '#006064',  // dark teal — kEDS, kyphoscoliosis, ocular fragility
  ADAMTS2: '#4a148c',  // deep purple — dEDS, extreme skin fragility, EM fibrils
  FBN1:    '#e65100',  // deep orange — Marfan, aortic root, losartan
  TGFBR2:  '#37474f',  // dark slate — LDS2, bifid uvula, aggressive threshold
  ACTA2:   '#880e4f',  // deep crimson — MSMD, iris flocculi, Moya Moya
};

const GENE_DISEASE = {
  COL5A1:  'Classical EDS (AD) — COL5A1; Haploinsufficiency/DN; Wide Atrophic Scars; Beighton ≥5; Skin Hyperextensibility; Most Common EDS',
  COL3A1:  'Vascular EDS (AD) — COL3A1; Collagen III; Arterial/Bowel/Uterine Rupture; AVOID Colonoscopy/Angiography; Celiprolol Level B',
  TNXB:    'Classical-like EDS (AR) — TNXB; Tenascin-X; Low Serum TNX <4 mcg/mL Diagnostic; NO Atrophic Scars; DDx cEDS',
  PLOD1:   'Kyphoscoliotic EDS type 1 (AR) — PLOD1; Lysyl Hydroxylase 1; Neonatal Kyphoscoliosis; Ocular Fragility Globe Rupture; Pyridoxine B6 Trial',
  ADAMTS2: 'Dermatosparaxis EDS (AR) — ADAMTS2; Procollagen N-proteinase; Extreme Skin Fragility; Loose Drooping Folds; Hieroglyphic EM Fibrils',
  FBN1:    'Marfan Syndrome (AD) — FBN1; Fibrillin-1; Aortic Root Dilatation; Ectopia Lentis Upward; Losartan Level A; Surgery 4.5 cm',
  TGFBR2:  'Loeys-Dietz Syndrome 2 (AD) — TGFBR2; TGF-β Receptor 2; Bifid Uvula; Arterial Tortuosity; Surgery 4.0 cm Aggressive',
  ACTA2:   'MSMD/FTAAD6 (AD) — ACTA2; Alpha-SMA; Iris Flocculi Pathognomonic; Moya Moya Stroke; Arg179His Most Severe; Surgery 4.0 cm',
};

const EDS_GENES = ['COL5A1', 'COL3A1', 'TNXB', 'PLOD1', 'ADAMTS2'];
const MARFAN_SPECTRUM_GENES = ['FBN1', 'TGFBR2'];
const SMOOTH_MUSCLE_GENES = ['ACTA2'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Connective Tissue Atlas…</p>
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
    { key: 'skin_hyperextensibility',          label: 'Skin Hyperextensibility',      color: '#1a237e' },
    { key: 'wide_atrophic_scarring',           label: 'Wide Atrophic Scars (cEDS)',    color: '#1a237e' },
    { key: 'joint_hypermobility_beighton_5',   label: 'JHM Beighton ≥5',              color: '#1a237e' },
    { key: 'arterial_dissection_rupture',      label: 'Arterial Rupture (vEDS)',       color: '#b71c1c' },
    { key: 'easy_bruising',                    label: 'Easy Bruising',                 color: '#b71c1c' },
    { key: 'neonatal_kyphoscoliosis',          label: 'Neonatal Kyphoscoliosis (kEDS)',color: '#006064' },
    { key: 'ocular_fragility_globe_rupture_risk', label: 'Ocular Fragility (kEDS)',   color: '#006064' },
    { key: 'aortic_root_dilatation',           label: 'Aortic Root Dilatation',       color: '#e65100' },
    { key: 'ectopia_lentis',                   label: 'Ectopia Lentis (Marfan)',       color: '#e65100' },
    { key: 'bifid_uvula_cleft_palate',         label: 'Bifid Uvula (LDS2)',            color: '#37474f' },
    { key: 'iris_flocculi',                    label: 'Iris Flocculi (ACTA2)',         color: '#880e4f' },
    { key: 'moya_moya_intracranial_occlusion', label: 'Moya Moya (ACTA2)',            color: '#880e4f' },
  ];

  return (
    <div>
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={data.total_patients} color="#37474f" />
        <KPI label="Genes" value={data.genes?.length} color="#37474f" />
        <KPI label="EDS Subtypes" value="5" color="#1a237e" />
        <KPI label="Marfan Spectrum" value="2" color="#e65100" />
        <KPI label="Smooth Muscle" value="1" color="#880e4f" />
        <KPI label="Seeds" value={data.seed_range} color="#37474f" />
      </div>

      <div className="alert alert-danger mb-3">
        <strong>🚨 COL3A1 vEDS:</strong> AVOID colonoscopy, angiography, elective surgery — bowel/arterial perforation risk. Celiprolol 400 mg/day from diagnosis. Median first arterial event age 29.
      </div>
      <div className="alert alert-warning mb-3">
        <strong>⚠️ PLOD1 kEDS:</strong> Globe rupture risk from minor eye trauma — protective glasses MANDATORY. Pyridoxine B6 5 mg/kg/day trial mandatory (~30% respond). Urine LP ratio &gt;0.09 diagnostic.
      </div>
      <div className="alert alert-info mb-3">
        <strong>ℹ️ TGFBR2 LDS2:</strong> Surgery threshold 4.0–4.2 cm (NOT 4.5 cm like Marfan) — dissection at smaller diameters. Bifid uvula/cleft palate pathognomonic. Full-body MRA mandatory.
      </div>
      <div className="alert alert-secondary mb-4">
        <strong>🔬 ACTA2 MSMD:</strong> Iris flocculi PATHOGNOMONIC — specifically request slit-lamp for pupillary margin fibres. MRA for Moya Moya intracranial occlusion.
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
        <div className="col-md-5">
          <div className="card border-0 shadow-sm">
            <div className="card-header" style={{ background: '#1a237e', color: 'white' }}>
              <strong>EDS Subtypes (5 genes)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {EDS_GENES.map(g => (
                <li key={g} className="list-group-item py-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span>{' — '}
                  <span className="text-muted">{GENE_DISEASE[g].split('—')[1]?.split(';')[0]}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card border-0 shadow-sm">
            <div className="card-header" style={{ background: '#e65100', color: 'white' }}>
              <strong>Marfan Spectrum (2 genes)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {MARFAN_SPECTRUM_GENES.map(g => (
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
            <div className="card-header" style={{ background: '#880e4f', color: 'white' }}>
              <strong>Smooth Muscle (1 gene)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {SMOOTH_MUSCLE_GENES.map(g => (
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
            color={a.includes('COL3A1') ? '#b71c1c' : a.includes('PLOD1') ? '#006064' :
                   a.includes('FBN1') ? '#e65100' : a.includes('TGFBR2') ? '#37474f' :
                   a.includes('ACTA2') ? '#880e4f' : a.includes('TNXB') ? '#1b5e20' : '#546e7a'} />
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
                <span className={`badge ${g.inheritance?.startsWith('AR') ? 'bg-success' : g.inheritance?.startsWith('XL') ? 'bg-warning text-dark' : 'bg-primary'}`}
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
export default function ConnectiveTissueAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/connective-tissue-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/connective-tissue-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/connective-tissue-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, def]) => { setOverview(ov); setBreakdown(bd); setDefinitions(def); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4">
      <div className="mb-4">
        <h4 className="fw-bold mb-1">🧬 Connective Tissue Atlas</h4>
        <p className="text-muted small mb-0">
          Complete 8-Gene Hereditary Connective Tissue Disorders Reference —
          COL5A1 (cEDS) · COL3A1 (vEDS) · TNXB (clEDS) · PLOD1 (kEDS) · ADAMTS2 (dEDS) ·
          FBN1 (Marfan) · TGFBR2 (LDS2) · ACTA2 (MSMD/FTAAD6) |
          320 patients · 8×40 · seeds 1398–1405
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
