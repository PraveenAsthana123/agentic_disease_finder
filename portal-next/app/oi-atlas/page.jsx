'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  COL1A1:   '#1a237e',  // deep navy — most common OI, glycine substitution C-terminal severity
  COL1A2:   '#880e4f',  // deep crimson — OI I-IV, recessive EDS overlap homotrimers
  IFITM5:   '#b71c1c',  // deep red — OI type V, hyperplastic callus, osteosarcoma mimic
  SERPINF1: '#1b5e20',  // deep green — OI type VI, PEDF undetectable, fish-scale bone
  CRTAP:    '#4e342e',  // deep brown — OI type VII, rhizomelic shortening, white sclerae
  P3H1:     '#006064',  // dark teal — OI type VIII, West African founder, Pro986 underhydroxylation
  FKBP10:   '#37474f',  // dark slate — OI XI/Bruck, contractures, absent LP crosslinks
  WNT1:     '#4a148c',  // deep purple — OI XV, trabecular collapse, brain, EOOP heterozygous
};

const GENE_DISEASE = {
  COL1A1:   'OI type I/II/III/IV (AD) — Collagen Alpha-1(I); Glycine Substitution C-terminal Severity; Haploinsufficiency OI-I',
  COL1A2:   'OI type I–IV (AD) / Recessive EDS-overlap (AR) — Collagen Alpha-2(I); Homotrimers; Moderate-Severe OI',
  IFITM5:   'OI type V (AD GOF) — BRIL; c.-14C>T 5\'UTR; HYPERPLASTIC CALLUS Osteosarcoma Mimic; Interosseous Calcification',
  SERPINF1: 'OI type VI (AR) — PEDF; UNDETECTABLE SERUM PEDF Diagnostic; Fish-Scale Bone Biopsy; Bisphosphonates Less Effective',
  CRTAP:    'OI type VII (AR) — Cartilage-Assoc Protein; RHIZOMELIC SHORTENING; WHITE SCLERAE; Pro986 Underhydroxylation',
  P3H1:     'OI type VIII (AR) — Prolyl-3-Hydroxylase; WEST AFRICAN FOUNDER Arg989Cys; 15-20% Severe OI in African Americans',
  FKBP10:   'OI XI/Bruck Syndrome (AR) — FKBP65; OI + CONGENITAL CONTRACTURES; ABSENT URINE LP CROSSLINKS Diagnostic',
  WNT1:     'OI type XV (AR biallelic) / EOOP (AD het) — Wnt1; TRABECULAR COLLAPSE; BRAIN MRI Mandatory; Romosozumab',
};

const AD_GENES  = ['COL1A1', 'COL1A2', 'IFITM5'];
const AR_GENES  = ['SERPINF1', 'CRTAP', 'P3H1', 'FKBP10', 'WNT1'];
const COLLAGEN_GENES = ['COL1A1', 'COL1A2'];
const P3H_COMPLEX = ['CRTAP', 'P3H1'];
const CROSSLINK_GENES = ['FKBP10'];
const WNT_GENES = ['WNT1'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Osteogenesis Imperfecta atlas…</p>
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

  return (
    <div>
      <div className="alert border-0 mb-4" style={{ background: '#e8eaf6' }}>
        <h5 className="mb-1">🧬 {data.atlas}</h5>
        <div className="text-muted small">{data.subtitle} · {data.total_patients} patients (8×40, seeds {data.seed_range})</div>
      </div>

      {/* Aggregate KPIs */}
      <h6 className="text-uppercase text-muted mb-3 small">Aggregate Cohort Statistics</h6>
      <div className="row g-2 mb-4">
        <KPI label="Recurrent Fractures" value={`${s.fractures_pct}%`} color="#1a237e" />
        <KPI label="Blue Sclerae" value={`${s.blue_sclerae_pct}%`} color="#880e4f" />
        <KPI label="Dentinogenesis Imperff" value={`${s.dentinogenesis_imperfecta_pct}%`} color="#b71c1c" />
        <KPI label="Hearing Loss" value={`${s.hearing_loss_pct}%`} color="#1b5e20" />
        <KPI label="Short Stature" value={`${s.short_stature_pct}%`} color="#4e342e" />
        <KPI label="Scoliosis" value={`${s.scoliosis_pct}%`} color="#006064" />
        <KPI label="Joint Contractures" value={`${s.joint_contractures_pct}%`} color="#37474f" />
        <KPI label="Hyperplastic Callus" value={`${s.hyperplastic_callus_pct}%`} color="#b71c1c" />
        <KPI label="Basilar Invagination" value={`${s.basilar_invagination_pct}%`} color="#4a148c" />
        <KPI label="Total Genes" value="8" color="#455a64" />
        <KPI label="Total Patients" value={data.total_patients} color="#546e7a" />
      </div>

      {/* Gene badge strip */}
      <h6 className="text-uppercase text-muted mb-3 small">8 OI Genes Covered</h6>
      <div className="mb-4">
        {data.genes.map(g => (
          <span key={g} className="badge me-2 mb-2 px-3 py-2"
            style={{ background: GENE_COLORS[g] || '#607d8b', fontSize: '0.78rem' }}>
            {g}
          </span>
        ))}
      </div>

      {/* Key DDx anchors */}
      <h6 className="text-uppercase text-muted mb-3 small">Key Clinical Alerts &amp; DDx Anchors</h6>
      <div className="mb-4">
        {data.key_ddx_anchor.map((anchor, i) => (
          <div key={i} className="alert border-start border-4 py-2 px-3 mb-2"
            style={{ borderColor: Object.values(GENE_COLORS)[i % 8], background: '#fafafa', fontSize: '0.82rem' }}>
            {anchor}
          </div>
        ))}
      </div>

      {/* Gene summary cards */}
      <h6 className="text-uppercase text-muted mb-3 small">Per-Gene Summary</h6>
      <div className="row g-3">
        {data.genes_summary.map(gs => (
          <div key={gs.gene} className="col-12 col-md-6 col-xl-4">
            <div className="card h-100 shadow-sm border-0">
              <div className="card-header text-white py-2 px-3"
                style={{ background: GENE_COLORS[gs.gene] || '#607d8b' }}>
                <div className="fw-bold">{gs.gene} — {gs.aa} · {gs.locus}</div>
                <div className="small opacity-75">{gs.primary_organ_system}</div>
              </div>
              <div className="card-body p-3">
                <div className="small text-muted mb-2">
                  OMIM gene <strong>{gs.omim_gene}</strong> · disease <strong>{gs.omim_disease}</strong> · {gs.n_patients} patients
                </div>
                <div className="row g-1 mb-2">
                  {[
                    ['Fractures', gs.fractures_pct],
                    ['Blue Sclerae', gs.blue_sclerae_pct],
                    ['DI', gs.dentinogenesis_imperfecta_pct],
                    ['Hearing Loss', gs.hearing_loss_pct],
                    ['Short Stature', gs.short_stature_pct],
                    ['Scoliosis', gs.scoliosis_pct],
                    ['Contractures', gs.joint_contractures_pct],
                    ['H. Callus', gs.hyperplastic_callus_pct],
                  ].map(([label, val]) => (
                    <div key={label} className="col-6">
                      <div className="small"><span className="fw-bold">{val}%</span> {label}</div>
                    </div>
                  ))}
                </div>
                <div className="small text-muted mb-1">Avg onset: {gs.avg_age_at_onset} yr · Diag delay: {gs.avg_diagnosis_delay_years} yr</div>
                <div className="mb-2">
                  {gs.hallmarks.map((h, i) => (
                    <div key={i} className="small text-muted" style={{ fontSize: '0.76rem' }}>• {h}</div>
                  ))}
                </div>
                <div className="alert py-1 px-2 mb-0 border-0"
                  style={{ background: '#fff3e0', fontSize: '0.74rem', borderLeft: `3px solid ${GENE_COLORS[gs.gene]}` }}>
                  ⚠ {gs.top_treatment_alert}
                </div>
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
  return (
    <div>
      <h6 className="text-uppercase text-muted mb-3 small">8-Gene OI Reference Table</h6>
      <div className="mb-3">
        <span className="badge me-2" style={{ background: '#1a237e' }}>AD (dominant)</span>
        <span className="badge me-2" style={{ background: '#880e4f' }}>AR (recessive)</span>
        <span className="badge me-2" style={{ background: '#b71c1c' }}>GOF (gain-of-function)</span>
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle" style={{ fontSize: '0.8rem' }}>
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Protein</th><th>aa</th><th>Locus</th>
              <th>Inh.</th><th>OI Type</th><th>OMIM Disease</th>
              <th>Fractures%</th><th>Blue Scl%</th><th>DI%</th>
              <th>Contractures%</th><th>H.Callus%</th><th>Basilar%</th>
              <th>Key Hallmark</th>
            </tr>
          </thead>
          <tbody>
            {data.genes_summary.map(gs => {
              const inh = AD_GENES.includes(gs.gene) ? 'AD' : 'AR';
              const oi_type = {
                COL1A1: 'I / II / III / IV',
                COL1A2: 'I / II / III / IV (AR: overlap)',
                IFITM5: 'V (GOF)',
                SERPINF1: 'VI',
                CRTAP: 'VII',
                P3H1: 'VIII',
                FKBP10: 'XI / Bruck 1',
                WNT1: 'XV (AR) / EOOP (AD)',
              }[gs.gene] || '-';
              return (
                <tr key={gs.gene}>
                  <td>
                    <span className="fw-bold" style={{ color: GENE_COLORS[gs.gene] }}>{gs.gene}</span>
                  </td>
                  <td className="text-muted" style={{ maxWidth: '140px' }}>{gs.protein}</td>
                  <td>{gs.aa}</td>
                  <td><code>{gs.locus}</code></td>
                  <td>
                    <span className="badge"
                      style={{ background: inh === 'AD' ? '#1565c0' : '#880e4f', fontSize: '0.7rem' }}>
                      {inh}
                    </span>
                  </td>
                  <td style={{ maxWidth: '100px' }}>{oi_type}</td>
                  <td><code>{gs.omim_disease}</code></td>
                  <td className="text-center fw-bold" style={{ color: '#1a237e' }}>{gs.fractures_pct}%</td>
                  <td className="text-center">{gs.blue_sclerae_pct}%</td>
                  <td className="text-center">{gs.dentinogenesis_imperfecta_pct}%</td>
                  <td className="text-center fw-bold"
                    style={{ color: gs.joint_contractures_pct > 50 ? '#b71c1c' : 'inherit' }}>
                    {gs.joint_contractures_pct}%
                  </td>
                  <td className="text-center fw-bold"
                    style={{ color: gs.hyperplastic_callus_pct > 50 ? '#b71c1c' : 'inherit' }}>
                    {gs.hyperplastic_callus_pct}%
                  </td>
                  <td className="text-center">{gs.basilar_invagination_pct}%</td>
                  <td style={{ maxWidth: '200px', fontSize: '0.72rem' }}>{gs.hallmarks[0]}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Collagen vs Non-collagen distinction */}
      <div className="row g-3 mt-2">
        <div className="col-12 col-md-4">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header py-2" style={{ background: '#1a237e', color: '#fff' }}>
              Collagen-Defect OI
            </div>
            <div className="card-body small p-3">
              {COLLAGEN_GENES.map(g => (
                <div key={g} className="mb-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span> — {GENE_DISEASE[g]}
                </div>
              ))}
              <hr className="my-2" />
              <div className="text-muted" style={{ fontSize: '0.74rem' }}>
                Dominant-negative glycine substitutions or haploinsufficiency → structural collagen defects
              </div>
            </div>
          </div>
        </div>
        <div className="col-12 col-md-4">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header py-2" style={{ background: '#006064', color: '#fff' }}>
              P3H Complex / Post-Translational OI
            </div>
            <div className="card-body small p-3">
              {P3H_COMPLEX.map(g => (
                <div key={g} className="mb-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span> — {GENE_DISEASE[g]}
                </div>
              ))}
              <div className="mt-2 mb-1">
                <span className="fw-bold" style={{ color: GENE_COLORS['FKBP10'] }}>FKBP10</span> — {GENE_DISEASE['FKBP10']}
              </div>
              <hr className="my-2" />
              <div className="text-muted" style={{ fontSize: '0.74rem' }}>
                Collagen post-translational modification defects → Pro986 underhydroxylation or crosslink loss
              </div>
            </div>
          </div>
        </div>
        <div className="col-12 col-md-4">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header py-2" style={{ background: '#4a148c', color: '#fff' }}>
              Non-Collagen OI (Mineralisation / Signalling)
            </div>
            <div className="card-body small p-3">
              <div className="mb-1">
                <span className="fw-bold" style={{ color: GENE_COLORS['IFITM5'] }}>IFITM5</span> — {GENE_DISEASE['IFITM5']}
              </div>
              <div className="mb-1">
                <span className="fw-bold" style={{ color: GENE_COLORS['SERPINF1'] }}>SERPINF1</span> — {GENE_DISEASE['SERPINF1']}
              </div>
              <div className="mb-1">
                <span className="fw-bold" style={{ color: GENE_COLORS['WNT1'] }}>WNT1</span> — {GENE_DISEASE['WNT1']}
              </div>
              <hr className="my-2" />
              <div className="text-muted" style={{ fontSize: '0.74rem' }}>
                Primary mineralisation or signalling defects — collagen biochemistry NORMAL in IFITM5/WNT1
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h6 className="text-uppercase text-muted mb-3 small">Per-Gene Clinical Atlas</h6>
      {data.map(gd => (
        <div key={gd.gene} className="card mb-4 shadow-sm border-0">
          <div className="card-header text-white py-2 px-3"
            style={{ background: GENE_COLORS[gd.gene] || '#607d8b' }}>
            <div className="d-flex justify-content-between align-items-center flex-wrap">
              <span className="fw-bold fs-6">{gd.gene} — {gd.protein}</span>
              <span className="small opacity-75">{gd.aa} · {gd.locus} · {gd.inheritance}</span>
            </div>
          </div>
          <div className="card-body p-3">
            <div className="row g-3">
              <div className="col-12 col-md-6">
                <div className="small text-muted mb-2 fw-bold">Clinical Features ({gd.n_patients} patients)</div>
                <div className="row g-1">
                  {[
                    ['Fractures', gd.fractures_pct],
                    ['Blue Sclerae', gd.blue_sclerae_pct],
                    ['Dentinogenesis Imperfecta', gd.dentinogenesis_imperfecta_pct],
                    ['Hearing Loss', gd.hearing_loss_pct],
                    ['Short Stature', gd.short_stature_pct],
                    ['Scoliosis', gd.scoliosis_pct],
                    ['Joint Contractures', gd.joint_contractures_pct],
                    ['Hyperplastic Callus', gd.hyperplastic_callus_pct],
                    ['Basilar Invagination', gd.basilar_invagination_pct],
                  ].map(([label, pct]) => (
                    <div key={label} className="col-6 col-sm-4">
                      <div className="small">
                        <span className="fw-bold"
                          style={{ color: pct >= 80 ? '#b71c1c' : pct >= 40 ? '#e65100' : '#424242' }}>
                          {pct}%
                        </span>
                        {' '}{label}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="small text-muted mt-2">
                  Sex: M {gd.sex_distribution?.M || '-'} / F {gd.sex_distribution?.F || '-'} ·
                  Avg onset: {gd.avg_age_at_onset} yr · Diag delay: {gd.avg_diagnosis_delay_years} yr
                </div>
              </div>
              <div className="col-12 col-md-6">
                <div className="small fw-bold text-muted mb-2">Hallmarks</div>
                {gd.hallmarks.map((h, i) => (
                  <div key={i} className="small mb-1" style={{ fontSize: '0.76rem' }}>
                    <span style={{ color: GENE_COLORS[gd.gene] }}>▸</span> {h}
                  </div>
                ))}
                <div className="small fw-bold text-muted mt-2 mb-1">Treatment Alerts</div>
                {gd.treatment_alerts.map((a, i) => (
                  <div key={i} className="alert py-1 px-2 mb-1 border-0"
                    style={{ background: '#fff3e0', fontSize: '0.74rem',
                      borderLeft: `3px solid ${GENE_COLORS[gd.gene]}` }}>
                    ⚠ {a}
                  </div>
                ))}
              </div>
            </div>
            {/* Etiology breakdown */}
            {gd.etiology_distribution && (
              <div className="mt-3">
                <div className="small fw-bold text-muted mb-1">Variant Distribution</div>
                <div className="d-flex flex-wrap gap-2">
                  {Object.entries(gd.etiology_distribution).map(([et, cnt]) => (
                    <span key={et} className="badge"
                      style={{ background: GENE_COLORS[gd.gene] || '#607d8b',
                        fontSize: '0.7rem', opacity: 0.85 }}>
                      {et.length > 60 ? et.substring(0, 57) + '…' : et}: {cnt}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── DEFINITIONS TAB ── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h6 className="text-uppercase text-muted mb-3 small">Clinical Definitions &amp; Molecular Mechanisms</h6>
      {data.definitions.map((def, i) => (
        <div key={i} className="card mb-3 shadow-sm border-0">
          <div className="card-header py-2 px-3"
            style={{ background: Object.values(GENE_COLORS)[i % 8], color: '#fff' }}>
            <strong style={{ fontSize: '0.9rem' }}>{def.term}</strong>
          </div>
          <div className="card-body p-3">
            <p className="mb-0" style={{ fontSize: '0.83rem', lineHeight: '1.65' }}>{def.definition}</p>
          </div>
        </div>
      ))}
      {data.cascade_testing_note && (
        <div className="alert border-start border-4 mt-3"
          style={{ borderColor: '#1a237e', background: '#e8eaf6', fontSize: '0.82rem' }}>
          <strong>CASCADE TESTING NOTE:</strong> {data.cascade_testing_note}
        </div>
      )}
    </div>
  );
}

/* ── MAIN PAGE ── */
export default function OIAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/oi-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/oi-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/oi-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4">
      <div className="row mb-3">
        <div className="col">
          <h2 className="mb-1" style={{ color: '#1a237e' }}>
            🦴 OI-Atlas — Complete 8-Gene Osteogenesis Imperfecta Atlas
          </h2>
          <p className="text-muted small mb-0">
            COL1A1 · COL1A2 · IFITM5 · SERPINF1 · CRTAP · P3H1 · FKBP10 · WNT1 ·
            320 patients (8×40, seeds 1374–1381) · OI type I–VIII, XI, XV + Bruck syndrome
          </p>
        </div>
      </div>

      {/* Alert strip */}
      <div className="alert border-0 py-2 mb-3" style={{ background: '#ffebee', fontSize: '0.8rem' }}>
        <strong>⚠ Critical Alerts:</strong>{' '}
        <AlertBadge text="IFITM5 HYPERPLASTIC CALLUS ≠ OSTEOSARCOMA" color="#b71c1c" />
        <AlertBadge text="SERPINF1: SERUM PEDF UNDETECTABLE = DIAGNOSTIC" color="#1b5e20" />
        <AlertBadge text="CRTAP/P3H1: RHIZOMELIC + WHITE SCLERAE" color="#4e342e" />
        <AlertBadge text="FKBP10 BRUCK: OI + CONTRACTURES" color="#37474f" />
        <AlertBadge text="WNT1-OI XV: BRAIN MRI MANDATORY" color="#4a148c" />
        <AlertBadge text="P3H1 Arg989Cys: 15-20% SEVERE OI AFRICAN AMERICANS" color="#006064" />
        <AlertBadge text="SERPINF1: BISPHOSPHONATES LESS EFFECTIVE" color="#1b5e20" />
        <AlertBadge text="c.-14C>T NOT IN STANDARD PANELS" color="#b71c1c" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <GeneTableTab data={overview} />}
      {tab === 2 && <ClinicalAtlasTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
