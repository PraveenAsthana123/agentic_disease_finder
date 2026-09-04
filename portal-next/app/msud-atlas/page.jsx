'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// MSUD color palette — amber/gold for maple syrup; deep red for leucine danger
const COLOR  = '#4e342e';  // dark brown — maple syrup
const LIGHT  = '#fff8e1';  // amber tint
const COLOR2 = '#b71c1c';  // leucine danger / crisis
const COLOR3 = '#1565c0';  // liver transplant / curative
const COLOR4 = '#e65100';  // KD contraindicated / warning
const COLOR5 = '#1b5e20';  // thiamine responsive / treatable
const COLOR6 = '#4a148c';  // BCKDK inverse / unique

const GENE_COLORS = {
  BCKDHA: '#b71c1c',   // most common, classic, crisis
  BCKDHB: '#c62828',   // second most common
  DBT:    '#d84315',   // intermediate possible
  DLD:    '#4a148c',   // triple complex, unique
  BCKDK:  '#1b5e20',   // inverse phenotype — green (treatable with Leu supplement)
  PPM1K:  '#37474f',   // phosphatase, mild
  BCAT2:  '#1565c0',   // pre-BCKDH, no allo-Ile
  SLC7A5: '#880e4f',   // BBB transport, ASD + epilepsy
};

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function BarRow({ label, pct, color = COLOR }) {
  return (
    <div className="mb-1">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ text, color = COLOR2 }) {
  return (
    <div className="alert mb-2 py-2 px-3 small" style={{ backgroundColor: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 4 }}>
      {text}
    </div>
  );
}

// ── Overview tab ──────────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-center p-4"><div className="spinner-border" /></div>;
  const cs = data.cohort_stats || {};
  const kt = data.key_teaching || {};
  return (
    <div>
      <div className="alert py-2 px-3 mb-3" style={{ backgroundColor: '#fff8e1', borderLeft: `5px solid ${COLOR}` }}>
        <strong>MSUD-Atlas</strong> — {data.title} · {data.n_genes} genes · {data.n_patients} patients (seeds {data.seeds})
      </div>

      {/* Critical teaching alerts */}
      <Alert color={COLOR2} text={<><strong>ALLOISOLEUCINE PATHOGNOMONIC:</strong> Allo-isoleucine present in ALL classic MSUD types (BCKDHA/B/DBT/DLD/PPM1K). Not found normally in humans. Confirms BCKDH complex deficiency instantly. ABSENT in BCKDK deficiency (low BCAA) and BCAT2 deficiency (pre-BCKDH).</>} />
      <Alert color={COLOR4} text={<><strong>KD CONTRAINDICATED</strong> in BCKDHA/B/DBT/DLD/PPM1K: fat catabolism mobilises BCAA from muscle → worsens leucine crisis. Prefer LEV for seizures. VPA HIGH RISK in all classic MSUD.</>} />
      <Alert color={COLOR3} text={<><strong>LIVER TRANSPLANT CURATIVE</strong> for BCKDHA/BCKDHB/DBT/PPM1K — liver provides &gt;95% systemic BCKDH activity. Post-transplant: normal diet, leucine normalises, allo-Ile disappears. NOT curative for DLD (ubiquitous E3).</>} />
      <Alert color={COLOR6} text={<><strong>BCKDK INVERSE PHENOTYPE:</strong> LOF kinase → hyperactive BCKDH → BCAAs pathologically LOW → autism + epilepsy + ID. Treat with LEUCINE SUPPLEMENTATION (opposite of classic MSUD). NBS misses BCKDK.</>} />

      {/* KPI row */}
      <div className="row mb-4">
        <KPI label="Genes" value={data.n_genes} color={COLOR} />
        <KPI label="Patients" value={data.n_patients} color={COLOR} />
        <KPI label="Allo-Ile +" value={cs.n_allo_isoleucine_positive} color={COLOR2} />
        <KPI label="Metabolic Crisis" value={cs.n_metabolic_crisis_at_dx} color={COLOR2} />
        <KPI label="Liver Tx" value={cs.n_liver_transplant} color={COLOR3} />
        <KPI label="ASD diagnoses" value={cs.n_asd} color={COLOR6} />
        <KPI label="EEG Abnormal" value={cs.n_eeg_abnormal} color={COLOR4} />
        <KPI label="Hepatopathy" value={cs.n_hepatopathy} color={COLOR4} />
        <KPI label="Thiamine Resp" value={cs.n_thiamine_responsive} color={COLOR5} />
        <KPI label="Drug-Resist Ep" value={cs.n_drug_resistant_epilepsy} color={COLOR2} />
      </div>

      {/* Frequency bars */}
      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm p-3 h-100">
            <h6 className="fw-bold mb-2" style={{ color: COLOR }}>Cohort Frequency</h6>
            <BarRow label="Alloisoleucine positive" pct={cs.pct_allo_positive} color={COLOR2} />
            <BarRow label="EEG abnormal" pct={cs.pct_eeg_abnormal} color={COLOR4} />
            <BarRow label="MRI abnormal" pct={cs.pct_mri_abnormal} color={COLOR} />
            <BarRow label="Metabolic crisis at dx" pct={cs.pct_crisis} color={COLOR2} />
            <BarRow label="Hepatopathy" pct={cs.pct_hepatopathy} color={COLOR4} />
            <BarRow label="ASD diagnosis" pct={cs.pct_asd} color={COLOR6} />
            <BarRow label="Liver transplant" pct={cs.pct_liver_transplant} color={COLOR3} />
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm p-3 h-100">
            <h6 className="fw-bold mb-2" style={{ color: COLOR }}>Key Teaching Points</h6>
            {Object.entries(kt).map(([k, v]) => (
              <div key={k} className="mb-2 small">
                <span className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}: </span>
                <span>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Gene cards */}
      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>8 Genes at a Glance</h6>
      <div className="row">
        {(data.genes || []).map(g => (
          <div key={g.gene} className="col-md-6 col-lg-3 mb-3">
            <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${GENE_COLORS[g.gene] || COLOR}` }}>
              <div className="card-body p-2">
                <div className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</div>
                <div className="small text-muted mb-1">{g.protein}</div>
                <div className="small mb-1"><strong>{g.locus}</strong></div>
                <div className="small mb-1 text-truncate">{g.msud_subgroup}</div>
                <div className="small">
                  {g.alloisoleucine_positive
                    ? <span className="badge me-1" style={{ backgroundColor: COLOR2 }}>Allo-Ile +</span>
                    : <span className="badge me-1 bg-secondary">Allo-Ile −</span>}
                  {g.liver_transplant_curative
                    ? <span className="badge me-1" style={{ backgroundColor: COLOR3 }}>LTx curative</span>
                    : <span className="badge me-1 bg-secondary">LTx NOT curative</span>}
                  {g.kd_contraindicated && <span className="badge" style={{ backgroundColor: COLOR4 }}>KD CI</span>}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Gene Table tab ────────────────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <div className="text-center p-4"><div className="spinner-border" /></div>;
  const genes = data.genes || [];
  return (
    <div className="table-responsive">
      <table className="table table-sm table-bordered small align-middle">
        <thead style={{ backgroundColor: COLOR, color: '#fff' }}>
          <tr>
            <th>Gene / Protein</th>
            <th>MSUD Subtype</th>
            <th>Locus</th>
            <th>Allo-Ile</th>
            <th>LTx Curative</th>
            <th>KD CI</th>
            <th>Thiamine Resp</th>
            <th>Founder Variant</th>
            <th>NBS Marker</th>
            <th>Key Biomarker</th>
            <th>Severity Spectrum</th>
          </tr>
        </thead>
        <tbody>
          {genes.map(g => (
            <tr key={g.gene}>
              <td>
                <div className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</div>
                <div className="text-muted">{g.protein}</div>
              </td>
              <td style={{ maxWidth: 160 }}>{g.msud_subgroup}</td>
              <td className="font-monospace">{g.locus}</td>
              <td className="text-center">
                {g.alloisoleucine_positive
                  ? <span className="badge" style={{ backgroundColor: COLOR2 }}>YES ✓</span>
                  : <span className="badge bg-secondary">NO</span>}
              </td>
              <td className="text-center">
                {g.liver_transplant_curative
                  ? <span className="badge" style={{ backgroundColor: COLOR3 }}>YES ✓</span>
                  : <span className="badge bg-secondary">NO</span>}
              </td>
              <td className="text-center">
                {g.kd_contraindicated
                  ? <span className="badge" style={{ backgroundColor: COLOR4 }}>CI ⚠</span>
                  : <span className="badge bg-secondary">–</span>}
              </td>
              <td className="text-center">
                {g.thiamine_responsive
                  ? <span className="badge" style={{ backgroundColor: COLOR5 }}>~10%</span>
                  : <span className="badge bg-secondary">No</span>}
              </td>
              <td style={{ maxWidth: 130 }}><small>{g.founder_variant}</small></td>
              <td style={{ maxWidth: 130 }}><small>{g.nbs_marker}</small></td>
              <td style={{ maxWidth: 140 }}><small>{g.key_biomarker}</small></td>
              <td style={{ maxWidth: 150 }}><small>{g.severity_spectrum}</small></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Clinical Atlas tab ────────────────────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  if (!data) return <div className="text-center p-4"><div className="spinner-border" /></div>;
  const genes = data.genes || [];
  return (
    <div>
      {genes.map(g => (
        <div key={g.gene} className="card shadow-sm mb-4" style={{ borderLeft: `5px solid ${GENE_COLORS[g.gene] || COLOR}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-2 flex-wrap gap-2">
              <div>
                <h5 className="mb-0 fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene} — {g.protein}</h5>
                <div className="small text-muted">{g.msud_subgroup} · {g.locus} · {g.n_patients} patients</div>
              </div>
              <div className="d-flex gap-1 flex-wrap">
                {g.alloisoleucine_positive && <span className="badge" style={{ backgroundColor: COLOR2 }}>Allo-Ile POSITIVE</span>}
                {!g.alloisoleucine_positive && <span className="badge bg-secondary">Allo-Ile ABSENT</span>}
                {g.liver_transplant_curative && <span className="badge" style={{ backgroundColor: COLOR3 }}>LTx CURATIVE</span>}
                {!g.liver_transplant_curative && <span className="badge bg-secondary">LTx NOT curative</span>}
                {g.kd_contraindicated && <span className="badge" style={{ backgroundColor: COLOR4 }}>KD CONTRAINDICATED</span>}
                {g.thiamine_responsive && <span className="badge" style={{ backgroundColor: COLOR5 }}>Thiamine responsive ~10%</span>}
              </div>
            </div>

            {/* Stats row */}
            <div className="row mb-3">
              <div className="col-6 col-md-2 text-center">
                <div className="fw-bold" style={{ color: COLOR2 }}>{g.n_crisis}</div>
                <div className="small text-muted">Crisis at Dx</div>
              </div>
              <div className="col-6 col-md-2 text-center">
                <div className="fw-bold" style={{ color: COLOR3 }}>{g.n_liver_tx}</div>
                <div className="small text-muted">Liver Tx</div>
              </div>
              <div className="col-6 col-md-2 text-center">
                <div className="fw-bold" style={{ color: COLOR4 }}>{g.n_eeg_abnormal}</div>
                <div className="small text-muted">EEG Abnormal</div>
              </div>
              <div className="col-6 col-md-2 text-center">
                <div className="fw-bold" style={{ color: COLOR }}>{g.n_mri_abnormal}</div>
                <div className="small text-muted">MRI Abnormal</div>
              </div>
              <div className="col-6 col-md-2 text-center">
                <div className="fw-bold" style={{ color: COLOR4 }}>{g.n_hepatopathy}</div>
                <div className="small text-muted">Hepatopathy</div>
              </div>
              <div className="col-6 col-md-2 text-center">
                <div className="fw-bold" style={{ color: COLOR6 }}>{g.n_asd}</div>
                <div className="small text-muted">ASD</div>
              </div>
            </div>

            {/* Hallmarks */}
            <div className="mb-2 small" style={{ backgroundColor: LIGHT, borderRadius: 4, padding: '8px 12px', borderLeft: `3px solid ${GENE_COLORS[g.gene] || COLOR}` }}>
              <strong>Hallmarks:</strong> {g.hallmark}
            </div>

            {/* Disease */}
            <details className="small">
              <summary className="fw-semibold mb-1" style={{ cursor: 'pointer', color: GENE_COLORS[g.gene] || COLOR }}>
                Full disease description (click to expand)
              </summary>
              <div className="mt-1" style={{ whiteSpace: 'pre-line' }}>{g.disease}</div>
            </details>

            {/* Sample patients */}
            {g.patients && g.patients.length > 0 && (
              <details className="mt-2 small">
                <summary className="fw-semibold" style={{ cursor: 'pointer', color: COLOR }}>
                  Sample patients (first 10)
                </summary>
                <div className="table-responsive mt-2">
                  <table className="table table-sm table-bordered small">
                    <thead style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR, color: '#fff' }}>
                      <tr>
                        <th>ID</th><th>Age Dx</th><th>Sex</th>
                        <th>Leu µmol/L</th><th>Allo-Ile</th><th>Crisis</th>
                        <th>MRI</th><th>EEG</th><th>LTx</th><th>AED</th>
                      </tr>
                    </thead>
                    <tbody>
                      {g.patients.map(p => (
                        <tr key={p.id}>
                          <td className="font-monospace">{p.id}</td>
                          <td>{p.age_dx_y < 0.1 ? `${Math.round(p.age_dx_y * 365)}d` : `${p.age_dx_y}y`}</td>
                          <td>{p.sex}</td>
                          <td className={p.leucine_dx_umolL > 1000 ? 'fw-bold text-danger' : ''}>{p.leucine_dx_umolL}</td>
                          <td>{p.allo_isoleucine_positive ? '✓' : '–'}</td>
                          <td>{p.metabolic_crisis_at_dx ? '⚠' : '–'}</td>
                          <td>{p.mri_abnormal ? '⚠' : '–'}</td>
                          <td>{p.eeg_abnormal ? '⚠' : '–'}</td>
                          <td>{p.liver_transplant ? '✓' : '–'}</td>
                          <td className={p.vpa_used ? 'text-danger' : ''}>{p.aed}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Definitions tab ───────────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center p-4"><div className="spinner-border" /></div>;
  const ov = data.msud_overview || {};
  const defs = data.definitions || [];
  return (
    <div>
      <div className="card shadow-sm mb-4 p-3" style={{ borderLeft: `5px solid ${COLOR}` }}>
        <h6 className="fw-bold mb-2" style={{ color: COLOR }}>MSUD-Atlas Overview</h6>
        <dl className="row small mb-0">
          <dt className="col-sm-3">Full name</dt><dd className="col-sm-9">{ov.full_name}</dd>
          <dt className="col-sm-3">Genes in atlas</dt><dd className="col-sm-9">{ov.genes_in_atlas}</dd>
          <dt className="col-sm-3">Pathognomonic biomarker</dt><dd className="col-sm-9">{ov.pathognomonic_biomarker}</dd>
          <dt className="col-sm-3">Curative treatment</dt><dd className="col-sm-9">{ov.curative_treatment}</dd>
          <dt className="col-sm-3">KD rule</dt><dd className="col-sm-9 fw-bold text-danger">{ov.kd_rule}</dd>
        </dl>
      </div>

      {defs.map((d, i) => (
        <div key={i} className="card shadow-sm mb-3">
          <div className="card-header py-2 px-3 fw-semibold small" style={{ backgroundColor: COLOR + '18', color: COLOR }}>
            {d.term}
          </div>
          <div className="card-body p-3 small" style={{ whiteSpace: 'pre-line' }}>
            {d.definition}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────────
export default function MSUDAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const load = async (path, setter) => {
      try {
        const r = await fetch(`${API}${path}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        setter(await r.json());
      } catch (e) { setError(e.message); }
    };
    load('/api/msud-atlas/overview', setOverview);
    load('/api/msud-atlas/breakdown', setBreakdown);
    load('/api/msud-atlas/definitions', setDefinitions);
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <span style={{ fontSize: 28 }}>🍁</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            MSUD-Atlas — Complete 8-Gene Maple Syrup Urine Disease & BCAA Disorders Atlas
          </h4>
          <div className="small text-muted">
            BCKDHA · BCKDHB · DBT · DLD · BCKDK · PPM1K · BCAT2 · SLC7A5/LAT1 — 320 patients (8×40, seeds 942–949)
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <GeneTableTab data={breakdown} />}
      {tab === 2 && <ClinicalAtlasTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
