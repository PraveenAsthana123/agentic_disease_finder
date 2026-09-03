'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Leigh vs MELAS vs Exercise', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#1a237e';   // deep indigo — mtDNA CI / maternal inheritance
const LIGHT  = '#e8eaf6';
const COLOR2 = '#283593';
const COLOR3 = '#b71c1c';   // dark red — Leigh / severe CI
const COLOR4 = '#e65100';   // deep orange — MELAS-like overlap
const COLOR5 = '#2e7d32';   // green — exercise / treatments

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Bar({ label, value, color = COLOR }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        {title && <h6 className="fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>}
        {children}
      </div>
    </div>
  );
}

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const s = data.cohort_statistics || {};
  const feats = data.cohort_summary_features || [];
  const pheno_dist = data.phenotype_distribution || [];
  const mol_feats = data.key_molecular_features || [];
  const alerts = data.clinical_alerts || [];

  return (
    <div>
      {/* No-LHON / Junction Bridge Banner */}
      <div className="alert fw-bold mb-4" style={{ backgroundColor: '#fff3e0', borderLeft: `5px solid ${COLOR3}`, color: '#bf360c' }}>
        🔴 MT-ND3: NO PRIMARY LHON MUTATION — unlike ND1 (m.3460G&gt;A #2), ND4 (m.11778G&gt;A #1), ND6 (m.14484T&gt;C #3).
        SMALLEST H-strand CI subunit (115 aa, 3 TM). Junction bridge: N-module ↔ proximal membrane arm.
        Dominant phenotypes: <strong>Leigh Syndrome · MELAS-like Overlap · Exercise Intolerance · KSS/CPEO</strong>
      </div>

      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Patients" value={data.n_patients} color={COLOR} />
        <KPI label="Avg CI Activity" value={`${s.avg_ci_activity_pct}%`} color={COLOR3} />
        <KPI label="Avg Lactate" value={`${s.avg_lactic_acid_mmolL} mmol/L`} color={COLOR4} />
        <KPI label="Leigh MRI" value={`${s.leigh_mri_pct}%`} color={COLOR3} />
        <KPI label="Lactic Acidosis" value={`${s.lactic_acidosis_pct}%`} color={COLOR4} />
        <KPI label="Stroke-like (MELAS)" value={`${s.stroke_like_pct}%`} color={COLOR4} />
        <KPI label="Exercise Intol." value={`${s.exercise_intolerance_pct}%`} color={COLOR5} />
        <KPI label="Avg Heteroplasmy" value={`${s.avg_heteroplasmy_blood_pct}%`} color={COLOR2} />
        <KPI label="Deceased" value={`${s.deceased_pct}%`} color="#616161" />
        <KPI label="TM Helices" value={data.tm_helices} color={COLOR} />
        <KPI label="aa Length" value={`${data.aa_length} aa`} color={COLOR} />
        <KPI label="MW (kDa)" value={`${data.molecular_weight_kda} kDa`} color={COLOR} />
      </div>

      {/* Gene Info */}
      <SectionCard title="Gene & Protein" borderColor={COLOR}>
        <div className="row g-2 small">
          <div className="col-md-6"><strong>Gene:</strong> {data.gene} ({data.omim_gene})</div>
          <div className="col-md-6"><strong>Protein:</strong> {data.protein}</div>
          <div className="col-md-6"><strong>Module/Position:</strong> {data.module}</div>
          <div className="col-md-6"><strong>rCRS positions:</strong> {data.rcrs_positions} ({data.strand})</div>
          <div className="col-md-6"><strong>Inheritance:</strong> {data.inheritance}</div>
          <div className="col-md-6"><strong>Primary Disease:</strong> {data.primary_disease}</div>
          <div className="col-12 mt-2 text-muted fst-italic">{data.no_lhon_note}</div>
        </div>
      </SectionCard>

      {/* Phenotype distribution */}
      {pheno_dist.length > 0 && (
        <SectionCard title="Phenotype Distribution" borderColor={COLOR3}>
          <div className="row">
            <div className="col-md-6">
              {pheno_dist.map(p => (
                <Bar key={p.phenotype} label={p.phenotype} value={p.pct}
                  color={p.phenotype.includes('Leigh') ? COLOR3 : p.phenotype.includes('MELAS') ? COLOR4 : p.phenotype.includes('Exercise') ? COLOR5 : COLOR2} />
              ))}
            </div>
            <div className="col-md-6">
              <Bar label="Lactic Acidosis" value={s.lactic_acidosis_pct} color={COLOR4} />
              <Bar label="Hypotonia" value={s.hypotonia_pct} color={COLOR2} />
              <Bar label="Developmental Delay" value={s.developmental_delay_pct} color={COLOR2} />
              <Bar label="Seizures" value={s.seizures_pct} color={COLOR3} />
              <Bar label="Encephalopathy" value={s.encephalopathy_pct} color={COLOR3} />
              <Bar label="Ragged Red Fibres" value={s.ragged_red_fibres_pct} color={COLOR5} />
              <Bar label="Respiratory Failure" value={s.respiratory_failure_pct} color={COLOR3} />
              <Bar label="KSS/CPEO" value={s.cpeo_pct} color={COLOR2} />
            </div>
          </div>
        </SectionCard>
      )}

      {/* Molecular features */}
      {mol_feats.length > 0 && (
        <SectionCard title="Key Molecular Features" borderColor={COLOR2}>
          <ul className="list-unstyled mb-0">
            {mol_feats.map((f, i) => <li key={i} className="mb-1 small">• {f}</li>)}
          </ul>
        </SectionCard>
      )}

      {/* Summary features */}
      {feats.length > 0 && (
        <SectionCard title="Cohort Summary" borderColor={COLOR5}>
          <ul className="list-unstyled mb-0">
            {feats.map((f, i) => <li key={i} className="mb-1 small">• {f}</li>)}
          </ul>
        </SectionCard>
      )}

      {/* Clinical alerts */}
      {alerts.length > 0 && (
        <SectionCard title="Clinical Alerts" borderColor={COLOR3}>
          {alerts.map((a, i) => (
            <div key={i} className="alert alert-warning py-2 mb-2 small">{a}</div>
          ))}
        </SectionCard>
      )}
    </div>
  );
}

// ── Tab: Leigh vs MELAS vs Exercise ──────────────────────────────────────────
function PhenotypesTab({ data, bdata }) {
  if (!data || !bdata) return <p className="text-muted">Loading…</p>;
  const s = data.cohort_statistics || {};
  const ci_bands = bdata.ci_activity_bands || [];
  const het_bands = bdata.heteroplasmy_bands || [];
  const pheno_by_var = bdata.phenotype_by_variant || [];
  const variants = bdata.variant_breakdown || [];

  return (
    <div>
      <SectionCard title="Phenotype–CI Activity Spectrum" borderColor={COLOR3}>
        <div className="row g-3 small">
          <div className="col-md-4">
            <div className="p-3 rounded" style={{ backgroundColor: '#ffebee', border: `1px solid ${COLOR3}` }}>
              <strong className="d-block mb-2" style={{ color: COLOR3 }}>Leigh Syndrome</strong>
              <div>CI Activity: 5–25% residual</div>
              <div>Heteroplasmy: ≥70–80%</div>
              <div>MRI: bilateral symmetric BG/brainstem</div>
              <div>Onset: infantile (3–18 months)</div>
              <div>Prevalence: {s.leigh_mri_pct}%</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-3 rounded" style={{ backgroundColor: '#fff3e0', border: `1px solid ${COLOR4}` }}>
              <strong className="d-block mb-2" style={{ color: COLOR4 }}>MELAS-like Overlap</strong>
              <div>Stroke-like episodes (NOT thrombotic)</div>
              <div>NO tPA (contraindicated)</div>
              <div>Heteroplasmy: 50–70%</div>
              <div>Thiamine + hydration acute</div>
              <div>Prevalence: {s.stroke_like_pct}%</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-3 rounded" style={{ backgroundColor: '#e8f5e9', border: `1px solid ${COLOR5}` }}>
              <strong className="d-block mb-2" style={{ color: COLOR5 }}>Exercise Intolerance</strong>
              <div>CI: 25–45% residual</div>
              <div>Adult onset (20–40 yr)</div>
              <div>RRF on Gomori trichrome</div>
              <div>COX-positive RRF (unlike KSS)</div>
              <div>Prevalence: {s.exercise_intolerance_pct}%</div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* CI Activity bands */}
      {ci_bands.length > 0 && (
        <SectionCard title="CI Activity Distribution" borderColor={COLOR3}>
          {ci_bands.map(b => (
            <Bar key={b.band} label={b.band} value={b.pct} color={b.pct > 30 ? COLOR3 : COLOR5} />
          ))}
        </SectionCard>
      )}

      {/* Heteroplasmy bands */}
      {het_bands.length > 0 && (
        <SectionCard title="Blood Heteroplasmy Distribution" borderColor={COLOR2}>
          {het_bands.map(b => (
            <Bar key={b.band} label={b.band} value={b.pct} color={COLOR2} />
          ))}
        </SectionCard>
      )}

      {/* Phenotype by variant */}
      {pheno_by_var.length > 0 && (
        <SectionCard title="Phenotype by Variant" borderColor={COLOR4}>
          <div className="table-responsive">
            <table className="table table-sm small mb-0">
              <thead><tr>
                <th>Variant</th><th>Dominant Phenotype</th><th>CI Activity</th><th>Heteroplasmy</th><th>n</th>
              </tr></thead>
              <tbody>
                {pheno_by_var.map(r => (
                  <tr key={r.variant}>
                    <td><code>{r.variant}</code></td>
                    <td>{r.dominant_phenotype}</td>
                    <td>{r.ci_activity_range}</td>
                    <td>{r.heteroplasmy_range}</td>
                    <td>{r.n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {/* Variant breakdown */}
      {variants.length > 0 && (
        <SectionCard title="Variant Breakdown" borderColor={COLOR}>
          {variants.map(v => (
            <Bar key={v.variant} label={`${v.variant} — ${v.phenotype}`} value={v.pct} color={COLOR} />
          ))}
        </SectionCard>
      )}
    </div>
  );
}

// ── Tab: DDx & Treatment ─────────────────────────────────────────────────────
function TreatmentTab({ data, bdata }) {
  if (!data || !bdata) return <p className="text-muted">Loading…</p>;
  const abs_ci = data.absolute_contraindications || [];
  const mandatory = data.mandatory_empiric_treatments || [];
  const level_c = data.level_c_treatments || [];
  const preferred_aed = data.preferred_aed;
  const ddx = bdata.differential_diagnosis || [];

  return (
    <div>
      {/* Absolute contraindications */}
      {abs_ci.length > 0 && (
        <SectionCard title="🚫 Absolute Contraindications" borderColor={COLOR3}>
          <div className="row g-2">
            {abs_ci.map((c, i) => (
              <div key={i} className="col-md-6">
                <div className="alert alert-danger py-2 mb-0 small">{c}</div>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {/* Mandatory empiric treatments */}
      {mandatory.length > 0 && (
        <SectionCard title="✅ Mandatory Empiric Treatments" borderColor={COLOR5}>
          <div className="row g-2">
            {mandatory.map((t, i) => (
              <div key={i} className="col-md-6">
                <div className="alert alert-success py-2 mb-0 small">{t}</div>
              </div>
            ))}
          </div>
          {preferred_aed && (
            <div className="alert alert-info py-2 mt-2 small">
              <strong>Preferred AED:</strong> {preferred_aed}
            </div>
          )}
        </SectionCard>
      )}

      {/* Level C treatments */}
      {level_c.length > 0 && (
        <SectionCard title="Level C Treatments" borderColor={COLOR4}>
          <ul className="list-unstyled mb-0">
            {level_c.map((t, i) => <li key={i} className="mb-1 small">• {t}</li>)}
          </ul>
        </SectionCard>
      )}

      {/* DDx */}
      {ddx.length > 0 && (
        <SectionCard title="Differential Diagnosis" borderColor={COLOR2}>
          <div className="table-responsive">
            <table className="table table-sm small mb-0">
              <thead><tr>
                <th>Entity</th><th>Key Distinguishing Feature</th>
              </tr></thead>
              <tbody>
                {ddx.map((d, i) => (
                  <tr key={i}>
                    <td><strong>{d.entity}</strong></td>
                    <td>{d.key_distinguisher}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {/* WES + KSS notes */}
      <SectionCard title="Diagnostic Notes" borderColor={COLOR2}>
        <ul className="list-unstyled mb-0 small">
          <li className="mb-2">⚠️ <strong>WES MISSES MT-ND3</strong> — mtDNA H-strand coverage often absent from WES panels; dedicated mtDNA sequencing (muscle biopsy preferred over blood) is required</li>
          <li className="mb-2">🫀 <strong>KSS Cardiac Monitoring</strong> — annual Holter/ECG mandatory; cardiac conduction block → pacemaker threshold PR ≥240 ms</li>
          <li className="mb-2">🔴 <strong>NO tPA in MELAS-like SLE</strong> — stroke-like episodes are NOT thrombotic; tPA causes cerebral haemorrhage</li>
          <li className="mb-2">💉 <strong>GIR 6–8 mg/kg/min</strong> — never fast; acute decompensation risk with fasting in ALL heteroplasmy levels</li>
          <li>🧬 <strong>Heteroplasmy threshold</strong> — below 50%: subclinical; 50–70%: exercise intolerance/MELAS; above 70%: Leigh/severe CI</li>
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const refs = data.key_references || [];
  const variants = data.key_variants || [];
  const thresh = data.heteroplasmy_thresholds || {};
  const monitor = data.specialist_monitoring || [];

  return (
    <div>
      <SectionCard title="Gene Summary" borderColor={COLOR}>
        <div className="row g-2 small">
          <div className="col-md-6"><strong>Full name:</strong> {data.full_name}</div>
          <div className="col-md-6"><strong>Protein:</strong> {data.protein_name}</div>
          <div className="col-md-6"><strong>OMIM Gene:</strong> {data.omim_gene}</div>
          <div className="col-md-6"><strong>Locus:</strong> {data.rcrs_positions} ({data.strand})</div>
          <div className="col-md-6"><strong>aa length:</strong> {data.aa_length} aa</div>
          <div className="col-md-6"><strong>MW:</strong> {data.molecular_weight_kda} kDa</div>
          <div className="col-md-6"><strong>TM helices:</strong> {data.tm_helices}</div>
          <div className="col-md-6"><strong>Module:</strong> {data.module}</div>
          <div className="col-md-12 mt-2"><strong>No primary LHON:</strong> {data.no_primary_lhon}</div>
          <div className="col-md-12"><strong>WES coverage:</strong> {data.wes_coverage}</div>
        </div>
      </SectionCard>

      {/* OMIM diseases */}
      {data.omim_diseases && (
        <SectionCard title="OMIM Disease Entries" borderColor={COLOR3}>
          <ul className="list-unstyled mb-0 small">
            {data.omim_diseases.map((d, i) => <li key={i} className="mb-1">• {d}</li>)}
          </ul>
        </SectionCard>
      )}

      {/* Key variants */}
      {variants.length > 0 && (
        <SectionCard title="Key Variants" borderColor={COLOR4}>
          <div className="table-responsive">
            <table className="table table-sm small mb-0">
              <thead><tr>
                <th>Variant</th><th>Protein Change</th><th>Phenotype</th><th>%</th>
              </tr></thead>
              <tbody>
                {variants.map(v => (
                  <tr key={v.variant}>
                    <td><code>{v.variant}</code></td>
                    <td>{v.protein_change}</td>
                    <td>{v.phenotype}</td>
                    <td>{v.cohort_pct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {/* Heteroplasmy thresholds */}
      {Object.keys(thresh).length > 0 && (
        <SectionCard title="Heteroplasmy Thresholds" borderColor={COLOR2}>
          <ul className="list-unstyled mb-0 small">
            {Object.entries(thresh).map(([k, v]) => (
              <li key={k} className="mb-1">• <strong>{k}:</strong> {v}</li>
            ))}
          </ul>
        </SectionCard>
      )}

      {/* Specialist monitoring */}
      {monitor.length > 0 && (
        <SectionCard title="Specialist Monitoring" borderColor={COLOR5}>
          <ul className="list-unstyled mb-0 small">
            {monitor.map((m, i) => <li key={i} className="mb-1">• {m}</li>)}
          </ul>
        </SectionCard>
      )}

      {/* References */}
      {refs.length > 0 && (
        <SectionCard title="Key References" borderColor={COLOR}>
          <ul className="list-unstyled mb-0 small">
            {refs.map((r, i) => <li key={i} className="mb-1">• {r}</li>)}
          </ul>
        </SectionCard>
      )}

      <p className="text-muted small mt-3">Cohort: {data.n_patients} patients · seed {data.cohort_seed} · Generated {data.generated}</p>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function MTND3Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/mtnd3/overview`).then(r => r.json()).then(setOverview).catch(() => setError('Backend unreachable'));
    fetch(`${API}/api/mtnd3/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/mtnd3/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div className="rounded-circle d-flex align-items-center justify-content-center fw-bold text-white"
          style={{ width: 48, height: 48, background: COLOR, fontSize: 18 }}>ND3</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>MT-ND3 Dashboard</h4>
          <small className="text-muted">SMALLEST H-strand CI Subunit · N-Module/Membrane-Arm Junction Bridge · 115 aa · 3 TM helices · NO Primary LHON · Maternal Inheritance</small>
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PhenotypesTab data={overview} bdata={breakdown} />}
      {tab === 2 && <TreatmentTab data={overview} bdata={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
