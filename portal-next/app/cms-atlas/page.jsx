'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// CMS Atlas color palette — neuromuscular junction / cholinergic
const COLOR  = '#1a237e';  // deep navy — NMJ / neurology
const LIGHT  = '#e8eaf6';  // indigo tint
const COLOR2 = '#b71c1c';  // deep red — contraindicated / severe
const COLOR3 = '#1b5e20';  // deep green — safe / first-line
const COLOR4 = '#e65100';  // orange — caution / warning
const COLOR5 = '#4a148c';  // purple — glycosylation
const COLOR6 = '#37474f';  // blue-grey — ion channel
const COLOR7 = '#006064';  // teal — presynaptic

const GENE_COLORS = {
  CHRNE: '#1a237e',  // most common, postsynaptic AChR (deep navy)
  RAPSN: '#283593',  // AChR clustering, N88K (indigo)
  DOK7:  '#e65100',  // limb-girdle, salbutamol (orange — caution pyridostigmine)
  COLQ:  '#b71c1c',  // AChE deficient, pyridostigmine ABSOLUTELY CI (red)
  CHAT:  '#006064',  // presynaptic, episodic apnea (teal)
  GFPT1: '#4a148c',  // glycosylation CMS (purple)
  AGRN:  '#1b5e20',  // presynaptic agrin (green)
  SCN4A: '#37474f',  // ion channel Nav1.4 (blue-grey)
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

function BarRow({ label, pct, color = COLOR, note }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between mb-1">
        <span className="small fw-semibold">{label}</span>
        <span className="small text-muted">{typeof pct === 'number' ? `${pct}%` : pct}{note ? ` — ${note}` : ''}</span>
      </div>
      {typeof pct === 'number' && (
        <div className="progress" style={{ height: 8 }}>
          <div className="progress-bar" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color }} />
        </div>
      )}
    </div>
  );
}

function AlertBox({ type = 'info', title, children }) {
  const icons = { danger: '🚨', warning: '⚠️', info: 'ℹ️', success: '✅' };
  return (
    <div className={`alert alert-${type} py-2 px-3 mb-3`}>
      <strong>{icons[type]} {title}</strong>
      <div className="small mt-1">{children}</div>
    </div>
  );
}

function Loading() {
  return <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /><div className="mt-2 text-muted small">Loading CMS-Atlas…</div></div>;
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger">{msg}</div>;
}

// ── Tab: Overview ─────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const cf = ov.clinical_features_prevalence || {};
  const sev = ov.severity || {};

  return (
    <div>
      <AlertBox type="danger" title="COLQ-CMS: Pyridostigmine ABSOLUTELY CONTRAINDICATED">
        COLQ-CMS = AChE-deficient NMJ. AChE already absent → pyridostigmine (AChEI) causes
        depolarisation block → paradoxical severe worsening → potentially fatal cholinergic crisis.
        Pathognomonic sign: SLOW PUPILLARY LIGHT RESPONSE (measure in clinic with torch in dim room).
        Treatment: Ephedrine ± salbutamol ONLY.
      </AlertBox>
      <AlertBox type="warning" title="DOK7-CMS: Pyridostigmine WORSENS — Salbutamol First-Line">
        DOK7 myasthenia = limb-girdle CMS (proximal weakness + neck flexors). Pyridostigmine typically
        counterproductive (reduces MuSK-DOK7-mediated AChR density). SALBUTAMOL (oral 2-4 mg TDS) or
        ephedrine first-line — β2-agonists upregulate MuSK-DOK7-AChR pathway.
        Most common allele: c.1124_1127dupTGCC (>50% of pathogenic alleles).
      </AlertBox>

      <h6 className="fw-bold mb-2" style={{ color: COLOR }}>
        Cohort KPIs — {ov.total_patients} Patients (8×40, Seeds {ov.seed_range})
      </h6>
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={ov.total_patients} color={COLOR} />
        <KPI label="Genes" value={ov.genes_covered} color={COLOR} />
        <KPI label="Mean Onset" value={`${ov.mean_onset_age_y}y`} color={COLOR2} />
        <KPI label="Severe" value={`${sev.severe_pct}%`} color={COLOR2} />
        <KPI label="Moderate" value={`${sev.moderate_pct}%`} color={COLOR4} />
        <KPI label="Mild" value={`${sev.mild_pct}%`} color={COLOR3} />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>
              Clinical Features (% of 320 patients)
            </div>
            <div className="card-body">
              <BarRow label="Ptosis" pct={cf.ptosis_pct} color={COLOR} note="CHRNE/RAPSN/COLQ most" />
              <BarRow label="Ophthalmoplegia" pct={cf.ophthalmoplegia_pct} color={COLOR} note="COLQ/CHRNE most" />
              <BarRow label="Bulbar weakness" pct={cf.bulbar_weakness_pct} color={COLOR2} note="CHAT/RAPSN/CHRNE" />
              <BarRow label="Respiratory crises" pct={cf.respiratory_crisis_pct} color={COLOR2} note="CHAT/RAPSN/COLQ" />
              <BarRow label="Limb-girdle pattern" pct={cf.limb_girdle_pattern_pct} color={COLOR4} note="DOK7/GFPT1/AGRN" />
              <BarRow label="Arthrogryposis (neonatal)" pct={cf.arthrogryposis_pct} color={COLOR5} note="RAPSN most common" />
              <BarRow label="Slow pupils (COLQ)" pct={cf.slow_pupils_pct} color={COLOR2} note="COLQ pathognomonic" />
              <BarRow label="Cold sensitivity (SCN4A)" pct={cf.cold_sensitivity_pct} color={COLOR6} note="SCN4A Nav1.4" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>
              CMS Categories by Anatomical Defect
            </div>
            <div className="card-body">
              {ov.cms_category_breakdown && Object.entries(ov.cms_category_breakdown).map(([cat, genes]) => (
                <div key={cat} className="mb-3">
                  <div className="small fw-semibold text-muted mb-1">{cat}</div>
                  <div>
                    {genes.map(g => (
                      <span key={g} className="badge me-1" style={{ backgroundColor: GENE_COLORS[g] || COLOR, fontSize: '0.72rem' }}>{g}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>
          Key Teaching Points
        </div>
        <div className="card-body">
          <ul className="small mb-0 ps-3">
            {(ov.key_teaching_points || []).map((pt, i) => (
              <li key={i} className="mb-1">{pt}</li>
            ))}
          </ul>
        </div>
      </div>

      {ov.drug_alerts && ov.drug_alerts.length > 0 && (
        <div className="card shadow-sm mb-3" style={{ borderColor: '#b71c1c' }}>
          <div className="card-header fw-semibold" style={{ backgroundColor: '#ffebee', color: '#b71c1c' }}>
            🚨 Drug Alerts — CMS Treatment Rules
          </div>
          <div className="card-body">
            <ul className="small mb-0 ps-3">
              {ov.drug_alerts.map((a, i) => <li key={i} className="mb-1">{a}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Tab: Gene Table ───────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const { genes } = data;
  return (
    <div>
      <AlertBox type="danger" title="Treatment Rule Matrix — Read Before Prescribing">
        COLQ = pyridostigmine ABSOLUTELY CI (red) · DOK7 = pyridostigmine WORSENS (orange) ·
        SCN4A-GOF = quinidine/mexiletine only · CHAT = 3,4-DAP + sick-day plan ·
        CHRNE/RAPSN/GFPT1 = pyridostigmine ± 3,4-DAP first-line (safe)
      </AlertBox>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle small">
          <thead style={{ background: LIGHT }}>
            <tr>
              <th style={{ color: COLOR }}>Gene</th>
              <th>CMS Type</th>
              <th>Locus</th>
              <th>Inh.</th>
              <th>Onset (y)</th>
              <th>Severe%</th>
              <th>Ptosis%</th>
              <th>Bulbar%</th>
              <th>Resp. Crisis%</th>
              <th>Pyr. Safe?</th>
              <th>3,4-DAP?</th>
              <th>Salbutamol?</th>
              <th>Quinidine?</th>
            </tr>
          </thead>
          <tbody>
            {genes.map(g => {
              const cf = g.clinical_features || {};
              const sev = g.severity_distribution || {};
              const onset = g.onset_range_y || [];
              const pyrSafe = g.pyridostigmine_safe;
              return (
                <tr key={g.gene}>
                  <td>
                    <span className="badge rounded-pill" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR, fontSize: '0.75rem' }}>
                      {g.gene}
                    </span>
                  </td>
                  <td className="fw-semibold" style={{ color: GENE_COLORS[g.gene] || COLOR, maxWidth: 160, fontSize: '0.72rem' }}>
                    {g.cms_group}
                    <div className="text-muted fw-normal" style={{ fontSize: '0.68rem' }}>
                      {g.protein} · OMIM #{g.omim_disease}
                    </div>
                  </td>
                  <td className="text-muted">{g.locus}</td>
                  <td>
                    <span className={`badge ${g.inheritance?.includes('Dominant') ? 'bg-warning text-dark' : 'bg-secondary'}`} style={{ fontSize: '0.7rem' }}>
                      {g.inheritance?.includes('Dominant') ? 'AD' : 'AR'}
                    </span>
                  </td>
                  <td className="text-muted" style={{ fontSize: '0.7rem' }}>
                    {onset[0] !== undefined && onset[1] !== undefined ? `${onset[0]}–${onset[1]}` : '—'}
                  </td>
                  <td>
                    <span className={`badge ${sev.severe_pct > 40 ? 'bg-danger' : sev.severe_pct > 20 ? 'bg-warning text-dark' : 'bg-success'}`}>
                      {sev.severe_pct}%
                    </span>
                  </td>
                  <td><span className="badge bg-light text-dark">{cf.ptosis_pct}%</span></td>
                  <td><span className={`badge ${cf.bulbar_weakness_pct > 50 ? 'bg-warning text-dark' : 'bg-light text-dark'}`}>{cf.bulbar_weakness_pct}%</span></td>
                  <td><span className={`badge ${cf.respiratory_crisis_pct > 40 ? 'bg-danger' : cf.respiratory_crisis_pct > 20 ? 'bg-warning text-dark' : 'bg-light text-dark'}`}>{cf.respiratory_crisis_pct}%</span></td>
                  <td>
                    {pyrSafe === false
                      ? <span className="badge bg-danger" style={{ fontSize: '0.72rem' }}>❌ CI / WORSENS</span>
                      : <span className="badge bg-success" style={{ fontSize: '0.72rem' }}>✅ Safe</span>}
                  </td>
                  <td>
                    {g.three_four_dap_indicated
                      ? <span className="badge bg-primary" style={{ fontSize: '0.72rem' }}>✅</span>
                      : <span className="badge bg-light text-dark" style={{ fontSize: '0.72rem' }}>—</span>}
                  </td>
                  <td>
                    {g.salbutamol_indicated
                      ? <span className="badge" style={{ backgroundColor: COLOR4, fontSize: '0.72rem' }}>✅ 1st-line</span>
                      : <span className="badge bg-light text-dark" style={{ fontSize: '0.72rem' }}>—</span>}
                  </td>
                  <td>
                    {g.quinidine_indicated
                      ? <span className="badge" style={{ backgroundColor: COLOR6, fontSize: '0.72rem' }}>✅</span>
                      : <span className="badge bg-light text-dark" style={{ fontSize: '0.72rem' }}>—</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Tab: Clinical Atlas ───────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const { genes } = data;
  return (
    <div>
      <AlertBox type="info" title="CMS Diagnostic Approach">
        Step 1: Exclude autoimmune MG (AChR/MuSK/LRP4 antibodies). Step 2: EMG — decrement ≥10% at 3Hz;
        RCMAP (double CMAP) = COLQ. Step 3: SFEMG (most sensitive). Step 4: CMS gene panel.
        Step 5: if slow pupils → COLQ; limb-girdle → DOK7/GFPT1; neonatal arthrogryposis → RAPSN; episodic apnea → CHAT.
      </AlertBox>
      {genes.map(g => (
        <div key={g.gene} className="card shadow-sm mb-4">
          <div className="card-header d-flex align-items-center gap-2" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR }}>
            <span className="badge bg-white fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR, fontSize: '0.85rem' }}>{g.gene}</span>
            <span className="fw-semibold text-white">{g.protein}</span>
            <span className="badge bg-white text-dark ms-auto" style={{ fontSize: '0.7rem' }}>{g.locus} · OMIM #{g.omim_disease}</span>
          </div>
          <div className="card-body">
            <div className="row g-3">
              <div className="col-md-6">
                <div className="mb-2">
                  <span className="fw-semibold small" style={{ color: GENE_COLORS[g.gene] || COLOR }}>CMS Type: </span>
                  <span className="small">{g.cms_group}</span>
                </div>
                <div className="mb-2">
                  <span className="fw-semibold small">Inheritance: </span>
                  <span className="small text-muted">{g.inheritance}</span>
                </div>
                <div className="mb-2">
                  <span className="fw-semibold small">Size: </span>
                  <span className="small text-muted">{g.aa} · {g.kDa}</span>
                </div>
                <div className="mb-3">
                  <div className="fw-semibold small mb-1">Phenotype</div>
                  <div className="small text-muted" style={{ lineHeight: 1.5 }}>{g.phenotype}</div>
                </div>
              </div>
              <div className="col-md-6">
                <div className="mb-2">
                  <div className="fw-semibold small mb-1">Molecular Mechanism</div>
                  <div className="small text-muted" style={{ lineHeight: 1.5, fontSize: '0.72rem' }}>{g.gene_class?.substring(0, 400)}{g.gene_class?.length > 400 ? '…' : ''}</div>
                </div>
                <div className="mb-2">
                  <div className="fw-semibold small mb-1" style={{ color: GENE_COLORS[g.gene] || COLOR }}>Clinical Features Distribution (n=40)</div>
                  <div>
                    {g.clinical_features && Object.entries(g.clinical_features).map(([k, v]) => (
                      <div key={k} className="mb-1">
                        <div className="d-flex justify-content-between mb-0" style={{ fontSize: '0.72rem' }}>
                          <span>{k.replace(/_pct$/, '').replace(/_/g, ' ')}</span>
                          <span className="text-muted">{v}%</span>
                        </div>
                        <div className="progress" style={{ height: 5 }}>
                          <div className="progress-bar" style={{ width: `${Math.min(v, 100)}%`, backgroundColor: GENE_COLORS[g.gene] || COLOR }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Treatment */}
            <div className="mb-3">
              <div className="fw-semibold small mb-1">
                Treatment Options
                {!g.pyridostigmine_safe && (
                  <span className="badge bg-danger ms-2" style={{ fontSize: '0.68rem' }}>
                    ❌ PYRIDOSTIGMINE {g.gene === 'COLQ' ? 'ABSOLUTELY CI' : 'WORSENS / CI'}
                  </span>
                )}
                {g.salbutamol_indicated && (
                  <span className="badge ms-1" style={{ backgroundColor: COLOR4, fontSize: '0.68rem' }}>
                    Salbutamol FIRST-LINE
                  </span>
                )}
                {g.quinidine_indicated && (
                  <span className="badge ms-1" style={{ backgroundColor: COLOR6, fontSize: '0.68rem' }}>
                    Quinidine/Mexiletine
                  </span>
                )}
              </div>
              <ul className="small mb-0 ps-3 text-muted">
                {g.treatment_options?.map((t, i) => <li key={i} className="mb-1">{t}</li>)}
              </ul>
            </div>

            {/* DDx */}
            <div>
              <div className="fw-semibold small mb-1">Key DDx</div>
              <ul className="small mb-0 ps-3 text-muted">
                {g.key_ddx?.map((d, i) => <li key={i} className="mb-1">{d}</li>)}
              </ul>
            </div>

            {/* Sample patients */}
            {g.sample_patients && g.sample_patients.length > 0 && (
              <div className="mt-3">
                <div className="fw-semibold small mb-2" style={{ color: COLOR }}>Sample Patients (n=3 of 40)</div>
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0" style={{ fontSize: '0.7rem' }}>
                    <thead style={{ background: LIGHT }}>
                      <tr>
                        <th>ID</th><th>Sex</th><th>Onset(y)</th><th>Sev.</th>
                        <th>Ptosis</th><th>Ophthalmo</th><th>Bulbar</th><th>Resp.</th>
                        <th>LG</th><th>Slow Pupil</th><th>Treatment</th>
                      </tr>
                    </thead>
                    <tbody>
                      {g.sample_patients.map(p => (
                        <tr key={p.id}>
                          <td className="text-muted">{p.id.split('-').slice(-1)[0]}</td>
                          <td>{p.sex}</td>
                          <td>{p.onset_age_y}</td>
                          <td>
                            <span className={`badge ${p.severity === 'Severe' ? 'bg-danger' : p.severity === 'Moderate' ? 'bg-warning text-dark' : 'bg-success'}`} style={{ fontSize: '0.65rem' }}>
                              {p.severity}
                            </span>
                          </td>
                          <td>{p.ptosis ? '✅' : '—'}</td>
                          <td>{p.ophthalmoplegia ? '✅' : '—'}</td>
                          <td>{p.bulbar_weakness ? '✅' : '—'}</td>
                          <td>{p.respiratory_crisis ? '🚨' : '—'}</td>
                          <td>{p.limb_girdle_pattern ? '✅' : '—'}</td>
                          <td>{p.slow_pupils ? '🔵' : '—'}</td>
                          <td className="text-muted">{p.current_treatment}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const { definitions } = data;
  return (
    <div>
      <AlertBox type="info" title="CMS vs MG — Critical Distinction">
        CMS = GENETIC (AChR/MuSK/LRP4 antibodies NEGATIVE). MG = AUTOIMMUNE (antibodies positive).
        NEVER diagnose CMS as seronegative MG without full CMS gene panel. Treatment is fundamentally different.
        Wrong treatment (pyridostigmine in COLQ-CMS) = iatrogenic worsening.
      </AlertBox>
      <div className="row g-3">
        {definitions.map((d, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header py-2" style={{ backgroundColor: LIGHT }}>
                <span className="fw-bold small" style={{ color: COLOR }}>{d.term}</span>
              </div>
              <div className="card-body py-2">
                <p className="small text-muted mb-0" style={{ lineHeight: 1.55 }}>{d.definition}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────
export default function CMSAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/cms-atlas`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message));
  }, []);

  return (
    <div className="container-fluid py-3 px-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3 pb-2" style={{ borderBottom: `3px solid ${COLOR}` }}>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>CMS-Atlas</h4>
          <div className="small text-muted">
            Complete 8-Gene Congenital Myasthenic Syndromes Atlas ·
            CHRNE · RAPSN · DOK7 · COLQ · CHAT · GFPT1 · AGRN · SCN4A
          </div>
          <div className="small text-muted">
            320 patients (8×40, seeds 1022–1029) · NMJ transmission disorders (genetic, not autoimmune)
          </div>
        </div>
        <div className="ms-auto d-flex gap-2 flex-wrap">
          <span className="badge" style={{ backgroundColor: COLOR2 }}>COLQ: Pyridostigmine ABSOLUTELY CI</span>
          <span className="badge" style={{ backgroundColor: COLOR4 }}>DOK7: Salbutamol First-Line</span>
          <span className="badge" style={{ backgroundColor: COLOR7 }}>CHAT: Episodic Apnea</span>
        </div>
      </div>

      {error && <ErrorMsg msg={error} />}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-semibold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
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
