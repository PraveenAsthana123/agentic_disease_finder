'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// CM Atlas color palette — congenital myopathy / neonatal / muscle fibre
const COLOR  = '#1a237e';  // deep navy — congenital / serious
const LIGHT  = '#e8eaf6';  // indigo tint
const COLOR2 = '#b71c1c';  // deep red — MH / danger / critical
const COLOR3 = '#1b5e20';  // deep green — AR / benign
const COLOR4 = '#e65100';  // orange — caution / respiratory
const COLOR5 = '#4a148c';  // purple — X-linked / severe
const COLOR6 = '#37474f';  // blue-grey — AD / moderate
const COLOR7 = '#006064';  // teal — thin filament / nemaline

const GENE_COLORS = {
  RYR1:    '#b71c1c',  // MH risk — danger red
  NEB:     '#1b5e20',  // largest gene, AR — green
  ACTA1:   '#1a237e',  // de novo dominant, severe — navy
  TPM2:    '#006064',  // milder cap disease — teal
  MYH7:    '#e65100',  // cardiac allelic — orange
  SELENON: '#4a148c',  // rigid spine, respiratory — purple
  MTM1:    '#880e4f',  // X-linked neonatal severe — deep pink
  DNM2:    '#37474f',  // AD milder, ptosis — blue-grey
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
  return <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /><div className="mt-2 text-muted small">Loading CM-Atlas…</div></div>;
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
  const cat = ov.cm_category_breakdown || {};

  return (
    <div>
      <h5 className="fw-bold mb-1" style={{ color: COLOR }}>{ov.full_name}</h5>
      <p className="text-muted small mb-3">{ov.subtitle}</p>
      <p className="mb-3">{ov.description}</p>

      {/* Drug alerts */}
      {(ov.drug_alerts || []).map((a, i) => (
        <AlertBox key={i}
          type={a.includes('ABSOLUTELY') || a.includes('VOLATILE') || a.includes('Succinylcholine') ? 'danger' : 'warning'}
          title="Drug / Treatment Alert">
          {a}
        </AlertBox>
      ))}

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={ov.total_patients} color={COLOR} />
        <KPI label="Genes Covered" value={ov.genes_covered} color={COLOR3} />
        <KPI label="Patients/Gene" value={ov.patients_per_gene} color={COLOR5} />
        <KPI label="Mean Onset (y)" value={ov.mean_onset_age_y} color={COLOR4} />
        <KPI label="Mean CK (IU/L)" value={(ov.mean_ck_iu_l || 0).toLocaleString()} color={COLOR6} />
        <KPI label="Seeds" value={ov.seed_range} color={COLOR6} />
      </div>

      <div className="row g-3 mb-4">
        {/* Severity */}
        <div className="col-md-4">
          <div className="card h-100">
            <div className="card-header fw-bold small" style={{ backgroundColor: LIGHT }}>Severity Distribution</div>
            <div className="card-body">
              <BarRow label="Mild" pct={sev.mild_pct} color={COLOR3} />
              <BarRow label="Moderate" pct={sev.moderate_pct} color={COLOR4} />
              <BarRow label="Severe" pct={sev.severe_pct} color={COLOR2} />
            </div>
          </div>
        </div>

        {/* Clinical features */}
        <div className="col-md-8">
          <div className="card h-100">
            <div className="card-header fw-bold small" style={{ backgroundColor: LIGHT }}>Clinical Feature Prevalence (cohort-wide)</div>
            <div className="card-body">
              <div className="row">
                <div className="col-md-6">
                  <BarRow label="Respiratory Decline" pct={cf.respiratory_decline_pct} color={COLOR4} />
                  <BarRow label="Ventilator Dependent" pct={cf.ventilator_dependent_pct} color={COLOR2} />
                  <BarRow label="MH Event (periop)" pct={cf.mh_event_pct} color={COLOR2} />
                  <BarRow label="Cardiac Event" pct={cf.cardiac_event_pct} color={COLOR2} />
                  <BarRow label="Contractures" pct={cf.contractures_pct} color={COLOR6} />
                </div>
                <div className="col-md-6">
                  <BarRow label="Scoliosis" pct={cf.scoliosis_pct} color={COLOR} />
                  <BarRow label="Facial Weakness" pct={cf.facial_weakness_pct} color={COLOR3} />
                  <BarRow label="Ptosis" pct={cf.ptosis_pct} color={COLOR5} />
                  <BarRow label="Ophthalmoparesis" pct={cf.ophthalmoparesis_pct} color={COLOR7} />
                  <BarRow label="Rigid Spine" pct={cf.rigid_spine_pct} color={COLOR4} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Category breakdown */}
      <div className="card mb-4">
        <div className="card-header fw-bold small" style={{ backgroundColor: LIGHT }}>Congenital Myopathy Subtype Categories</div>
        <div className="card-body">
          <div className="row g-2">
            {Object.entries(cat).map(([cat_name, genes]) => (
              <div key={cat_name} className="col-md-6 col-lg-3">
                <div className="border rounded p-2 h-100">
                  <div className="small fw-semibold text-muted mb-1">{cat_name}</div>
                  <div className="d-flex flex-wrap gap-1">
                    {(genes || []).map(g => (
                      <span key={g} className="badge" style={{ backgroundColor: GENE_COLORS[g] || COLOR }}>{g}</span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Key teaching points */}
      <div className="card mb-4">
        <div className="card-header fw-bold small" style={{ backgroundColor: LIGHT }}>Key Clinical Teaching Points</div>
        <div className="card-body">
          <ul className="mb-0 ps-3">
            {(ov.key_teaching_points || []).map((pt, i) => (
              <li key={i} className="small mb-2">{pt}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

// ── Tab: Gene Table ────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <p className="text-muted small mb-3">Per-gene clinical summary — 8 CM genes, 40 patients each.</p>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle small">
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Locus</th><th>Protein / aa</th><th>Inheritance</th>
              <th>CM Type</th><th>Onset (y)</th><th>Mean CK</th>
              <th>Resp.</th><th>MH Risk</th><th>Neonatal</th><th>Severity (Mod+Sev %)</th>
            </tr>
          </thead>
          <tbody>
            {(data.genes || []).map(g => {
              const modSev = (g.severity_distribution?.moderate_pct || 0) + (g.severity_distribution?.severe_pct || 0);
              return (
                <tr key={g.gene}>
                  <td>
                    <span className="badge fw-bold" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</span>
                  </td>
                  <td className="font-monospace small">{g.locus}</td>
                  <td>{g.aa}</td>
                  <td>
                    <span className={`badge ${g.ad_inheritance ? 'bg-warning text-dark' : g.gene === 'MTM1' ? 'bg-danger' : 'bg-primary'}`}>
                      {g.gene === 'MTM1' ? 'XLR' : g.ad_inheritance ? 'AD' : 'AR'}
                    </span>
                  </td>
                  <td className="small text-muted">{g.cm_type?.split(' —')[0]}</td>
                  <td>{g.onset_range_y?.[0]}–{g.onset_range_y?.[1]}</td>
                  <td className="fw-semibold" style={{ color: COLOR6 }}>
                    {(g.mean_ck_iu_l || 0).toLocaleString()}
                  </td>
                  <td>
                    {g.respiratory_risk
                      ? <span className="badge bg-warning text-dark">Monitor</span>
                      : <span className="badge bg-secondary">Spared</span>}
                  </td>
                  <td>
                    {g.mh_risk
                      ? <span className="badge bg-danger">MH RISK</span>
                      : <span className="badge bg-secondary">No</span>}
                  </td>
                  <td>
                    {g.neonatal_severe
                      ? <span className="badge bg-danger">Neonatal</span>
                      : <span className="badge bg-secondary">Later</span>}
                  </td>
                  <td>
                    <div className="progress" style={{ height: 10, minWidth: 80 }}>
                      <div className="progress-bar" style={{ width: `${Math.min(modSev, 100)}%`, backgroundColor: GENE_COLORS[g.gene] || COLOR }} />
                    </div>
                    <small className="text-muted">{modSev.toFixed(0)}%</small>
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

// ── Tab: Clinical Atlas ────────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  const [sel, setSel] = useState(null);
  if (!data) return <Loading />;
  const genes = data.genes || [];
  const active = sel ? genes.find(g => g.gene === sel) : genes[0];

  return (
    <div className="row g-3">
      {/* Gene selector */}
      <div className="col-md-3">
        <div className="list-group">
          {genes.map(g => (
            <button
              key={g.gene}
              className={`list-group-item list-group-item-action d-flex justify-content-between align-items-center ${(sel || genes[0]?.gene) === g.gene ? 'active' : ''}`}
              style={(sel || genes[0]?.gene) === g.gene ? { backgroundColor: GENE_COLORS[g.gene] || COLOR, borderColor: GENE_COLORS[g.gene] || COLOR } : {}}
              onClick={() => setSel(g.gene)}
            >
              <span className="fw-bold">{g.gene}</span>
              <div className="d-flex flex-column align-items-end gap-1">
                <span className={`badge ${g.gene === 'MTM1' ? 'bg-danger' : g.ad_inheritance ? 'bg-warning text-dark' : 'bg-light text-dark'}`} style={{ fontSize: '0.6rem' }}>
                  {g.gene === 'MTM1' ? 'XLR' : g.ad_inheritance ? 'AD' : 'AR'}
                </span>
                {g.mh_risk && <span className="badge bg-danger" style={{ fontSize: '0.55rem' }}>MH</span>}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Detail panel */}
      {active && (
        <div className="col-md-9">
          <div className="card">
            <div className="card-header" style={{ backgroundColor: GENE_COLORS[active.gene] || COLOR, color: 'white' }}>
              <h5 className="mb-0">{active.gene} — {active.protein}</h5>
              <small>{active.alias}</small>
            </div>
            <div className="card-body">
              {/* Gene class */}
              <h6 className="fw-bold mt-2">Molecular Mechanism</h6>
              <p className="small">{active.gene_class}</p>

              <h6 className="fw-bold mt-3">Clinical Phenotype</h6>
              <p className="small">{active.phenotype}</p>

              <h6 className="fw-bold mt-3">Disease Summary</h6>
              <p className="small">{active.disease}</p>

              {/* Flags */}
              <div className="d-flex flex-wrap gap-2 mb-3">
                {active.mh_risk && <span className="badge bg-danger">🚨 MH RISK — Avoid Volatile Agents + Succinylcholine</span>}
                {active.neonatal_severe && <span className="badge bg-danger">Neonatal Severe</span>}
                {active.respiratory_risk && <span className="badge bg-warning text-dark">Respiratory Monitor</span>}
                {active.rigid_spine && <span className="badge" style={{ backgroundColor: '#4a148c', color: 'white' }}>Rigid Spine</span>}
                {active.cardiac_risk && <span className="badge bg-danger">Cardiac Mandatory</span>}
                {active.ophthalmoparesis && <span className="badge bg-info text-dark">Ophthalmoparesis</span>}
                {active.ptosis && <span className="badge bg-secondary">Ptosis</span>}
                {active.contractures && <span className="badge bg-secondary">Contractures</span>}
                {active.ad_inheritance && active.gene !== 'MTM1' && <span className="badge bg-warning text-dark">Autosomal Dominant</span>}
                {active.gene === 'MTM1' && <span className="badge bg-danger">X-Linked Recessive</span>}
              </div>

              {/* Clinical features */}
              <div className="row g-3 mb-3">
                <div className="col-md-6">
                  <div className="card bg-light">
                    <div className="card-body py-2">
                      <div className="fw-bold small mb-2">Clinical Feature Rates</div>
                      <BarRow label="Respiratory Decline" pct={active.clinical_features?.respiratory_decline_pct} color={COLOR4} />
                      <BarRow label="Ventilator Dependent" pct={active.clinical_features?.ventilator_dependent_pct} color={COLOR2} />
                      <BarRow label="MH Event (periop)" pct={active.clinical_features?.mh_event_pct} color={COLOR2} />
                      <BarRow label="Cardiac Event" pct={active.clinical_features?.cardiac_event_pct} color={COLOR2} />
                      <BarRow label="Contractures" pct={active.clinical_features?.contractures_pct} color={COLOR6} />
                    </div>
                  </div>
                </div>
                <div className="col-md-6">
                  <div className="card bg-light">
                    <div className="card-body py-2">
                      <div className="fw-bold small mb-2">Structural / Eye / Spine</div>
                      <BarRow label="Scoliosis" pct={active.clinical_features?.scoliosis_pct} color={COLOR} />
                      <BarRow label="Facial Weakness" pct={active.clinical_features?.facial_weakness_pct} color={COLOR3} />
                      <BarRow label="Ptosis" pct={active.clinical_features?.ptosis_pct} color={COLOR5} />
                      <BarRow label="Ophthalmoparesis" pct={active.clinical_features?.ophthalmoparesis_pct} color={COLOR7} />
                      <BarRow label="Rigid Spine" pct={active.clinical_features?.rigid_spine_pct} color={COLOR4} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Treatment */}
              <h6 className="fw-bold">Treatment Options</h6>
              <ul className="small mb-3 ps-3">
                {(active.treatment_options || []).map((t, i) => (
                  <li key={i} className={
                    t.toUpperCase().includes('ABSOLUTELY') || t.toUpperCase().includes('AVOID') ||
                    t.toUpperCase().includes('CONTRAINDICATED') || t.toUpperCase().includes('MANDATORY')
                      ? 'text-danger fw-semibold mb-1' : 'mb-1'
                  }>{t}</li>
                ))}
              </ul>

              {/* DDx */}
              <h6 className="fw-bold">Key Differential Diagnoses</h6>
              <ul className="small mb-3 ps-3">
                {(active.key_ddx || []).map((d, i) => <li key={i} className="mb-1">{d}</li>)}
              </ul>

              {/* Sample patients */}
              <h6 className="fw-bold">Sample Patients (first 3 of 40)</h6>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead className="table-light">
                    <tr><th>ID</th><th>Sex</th><th>Onset (y)</th><th>Severity</th><th>CK</th><th>Resp.</th><th>Treatment</th></tr>
                  </thead>
                  <tbody>
                    {(active.sample_patients || []).map(p => (
                      <tr key={p.id}>
                        <td className="font-monospace small">{p.id}</td>
                        <td>{p.sex}</td>
                        <td>{p.onset_age_y}</td>
                        <td>
                          <span className={`badge ${p.severity === 'Severe' ? 'bg-danger' : p.severity === 'Moderate' ? 'bg-warning text-dark' : 'bg-success'}`}>
                            {p.severity}
                          </span>
                        </td>
                        <td>{(p.ck_iu_l || 0).toLocaleString()}</td>
                        <td>{p.respiratory_decline ? '✓' : '—'}</td>
                        <td className="small">{p.current_treatment}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <p className="text-muted small mb-3">Clinical and molecular definitions for Congenital Myopathy subtypes.</p>
      {(data.definitions || []).map((d, i) => (
        <div key={i} className="mb-3">
          <h6 className="fw-bold" style={{ color: COLOR }}>{d.term}</h6>
          <p className="small mb-0">{d.definition}</p>
          <hr className="my-2" />
        </div>
      ))}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────
export default function CMAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState({});
  const [errors, setErrors] = useState({});

  async function load(key, url, setter) {
    if (loading[key] || (key === 'overview' && overview) || (key === 'breakdown' && breakdown) || (key === 'definitions' && definitions)) return;
    setLoading(l => ({ ...l, [key]: true }));
    try {
      const r = await fetch(`${API}${url}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setter(await r.json());
    } catch (e) {
      setErrors(er => ({ ...er, [key]: e.message }));
    } finally {
      setLoading(l => ({ ...l, [key]: false }));
    }
  }

  useEffect(() => { load('overview', '/api/cm-atlas/overview', setOverview); }, []);
  useEffect(() => {
    if (tab === 1 || tab === 2) load('breakdown', '/api/cm-atlas/breakdown', setBreakdown);
    if (tab === 3) load('definitions', '/api/cm-atlas/definitions', setDefinitions);
  }, [tab]);

  const tabContent = [
    <OverviewTab key="ov" data={overview} />,
    <GeneTableTab key="gt" data={breakdown} />,
    <ClinicalAtlasTab key="ca" data={breakdown} />,
    <DefinitionsTab key="df" data={definitions} />,
  ];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3">
        <div>
          <h4 className="fw-bold mb-0" style={{ color: COLOR }}>🧬 CM-Atlas</h4>
          <div className="text-muted small">
            Complete 8-Gene Congenital Myopathy Atlas ·
            RYR1 · NEB · ACTA1 · TPM2 · MYH7 · SELENON · MTM1 · DNM2 ·
            320 patients (8×40, seeds 1038–1045)
          </div>
        </div>
      </div>

      {/* Critical alert banner */}
      <div className="alert alert-danger py-2 px-3 mb-3 small">
        <strong>🚨 Critical Treatment Rules:</strong>{' '}
        <strong>RYR1:</strong> MALIGNANT HYPERTHERMIA — volatile anaesthetics + succinylcholine ABSOLUTELY AVOID (dantrolene IV immediately if MH crisis) ·
        <strong>SELENON:</strong> NIV MANDATORY EARLY (respiratory failure disproportionate to limb weakness; start when FVC &lt;60%) ·
        <strong>MTM1:</strong> Ventilator from birth most males ·
        <strong>ALL:</strong> Succinylcholine CONTRAINDICATED
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* Content */}
      {errors[['overview','breakdown','breakdown','definitions'][tab]] && (
        <ErrorMsg msg={errors[['overview','breakdown','breakdown','definitions'][tab]]} />
      )}
      {tabContent[tab]}
    </div>
  );
}
