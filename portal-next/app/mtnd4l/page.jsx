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
      {/* Hairpin Connector / No-LHON Banner */}
      <div className="alert fw-bold mb-4" style={{ backgroundColor: '#fff3e0', borderLeft: `5px solid ${COLOR3}`, color: '#bf360c' }}>
        🔴 MT-ND4L: SMALLEST CI mtDNA SUBUNIT (98 aa, 3 TM) — NO PRIMARY LHON despite proximity to ND4 (m.11778G&gt;A #1).
        ND4/ND4L Antiporter Hairpin Connector — braces ND4 central proton channel at PP-b assembly checkpoint.
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
          {pheno_dist.map((p, i) => (
            <div key={i} className="mb-2">
              <div className="d-flex justify-content-between small mb-1">
                <span>{p.phenotype}</span><span className="text-muted">{p.count} pts</span>
              </div>
              <div className="progress" style={{ height: 10 }}>
                <div className="progress-bar" style={{ width: `${(p.count / data.n_patients) * 100}%`, backgroundColor: COLOR3 }} />
              </div>
            </div>
          ))}
        </SectionCard>
      )}

      {/* Clinical feature bars */}
      <SectionCard title="Clinical Features (% of 40-patient cohort)" borderColor={COLOR2}>
        <div className="row">
          <div className="col-md-6">
            {feats.slice(0, Math.ceil(feats.length / 2)).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={f.pct > 60 ? COLOR3 : f.pct > 30 ? COLOR4 : COLOR5} />
            ))}
          </div>
          <div className="col-md-6">
            {feats.slice(Math.ceil(feats.length / 2)).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={f.pct > 60 ? COLOR3 : f.pct > 30 ? COLOR4 : COLOR5} />
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Molecular features */}
      {mol_feats.length > 0 && (
        <SectionCard title="Key Molecular Features — ND4/ND4L Antiporter Hairpin Connector" borderColor={COLOR}>
          {mol_feats.map((f, i) => (
            <div key={i} className="mb-3 pb-3 border-bottom">
              <div className="fw-semibold small mb-1" style={{ color: COLOR }}>{f.feature}</div>
              <div className="small text-muted">{f.significance}</div>
            </div>
          ))}
        </SectionCard>
      )}

      {/* CI Module Size Comparison */}
      <SectionCard title="All 7 mtDNA-Encoded CI Subunits — Size Comparison (ND4L is SMALLEST)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}>
              <th>Subunit</th><th>aa</th><th>kDa</th><th>TM</th><th>Module</th><th>Primary LHON?</th>
            </tr></thead>
            <tbody>
              {[
                { g: 'MT-ND4L', aa: 98, kDa: 10.7, tm: 3, mod: 'ND4/ND4L Hairpin Connector (THIS GENE)', lhon: '❌ None', highlight: true },
                { g: 'MT-ND3', aa: 115, kDa: 13.1, tm: 3, mod: 'N-Module/Membrane Arm Junction Bridge', lhon: '❌ None' },
                { g: 'MT-ND6', aa: 174, kDa: 19.6, tm: 5, mod: 'Distal Arm Tip (L-strand, inverted)', lhon: '✅ m.14484T>C #3 (~14%)' },
                { g: 'MT-ND1', aa: 318, kDa: 36, tm: 8, mod: 'Proximal Membrane Arm ND1-Module', lhon: '✅ m.3460G>A #2 (~15%)' },
                { g: 'MT-ND2', aa: 347, kDa: 39.5, tm: 13, mod: 'Proximal-Middle Antiporter', lhon: '❌ None' },
                { g: 'MT-ND4', aa: 459, kDa: 51.7, tm: 13, mod: 'Central Antiporter Module', lhon: '✅ m.11778G>A #1 (~70%)' },
                { g: 'MT-ND5', aa: 603, kDa: 67.9, tm: 16, mod: 'Distal Antiporter + Lateral Helix βH1', lhon: '❌ None' },
              ].map((r, i) => (
                <tr key={i} style={r.highlight ? { backgroundColor: '#fffde7', fontWeight: 600 } : {}}>
                  <td>{r.g}</td><td>{r.aa}</td><td>{r.kDa}</td><td>{r.tm}</td>
                  <td>{r.mod}</td><td>{r.lhon}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Clinical alerts */}
      {alerts.length > 0 && (
        <SectionCard title="Clinical Alerts" borderColor={COLOR3}>
          {alerts.map((a, i) => (
            <div key={i} className="alert alert-light border-start border-3 mb-2 py-2 small"
              style={{ borderColor: a.startsWith('🔴') ? COLOR3 : a.startsWith('⚠️') ? '#e65100' : COLOR }}>
              {a}
            </div>
          ))}
        </SectionCard>
      )}
    </div>
  );
}

// ── Tab: Leigh vs MELAS vs Exercise ──────────────────────────────────────────
function LeighVsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  const variants = data.variant_breakdown || [];
  const ci_bands = data.ci_activity_bands || {};
  const het_bands = data.heteroplasmy_bands || {};
  const outcomes = data.outcome_distribution || [];
  const assembly = data.assembly_pathway || [];
  const pts = data.patient_table || [];

  return (
    <div>
      {/* Variant breakdown table */}
      <SectionCard title="Variant Breakdown — MT-ND4L Pathogenic Variants" borderColor={COLOR3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}>
              <th>Variant</th><th>Protein</th><th>Domain</th><th>n</th>
              <th>Avg CI%</th><th>Avg Heter%</th><th>Leigh MRI%</th>
              <th>Stroke%</th><th>Exercise%</th>
            </tr></thead>
            <tbody>
              {variants.map((v, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{v.variant}</td>
                  <td>{v.protein}</td>
                  <td className="text-muted" style={{ maxWidth: 180 }}>{v.domain}</td>
                  <td>{v.n_patients}</td>
                  <td style={{ color: v.avg_ci_activity_pct < 15 ? COLOR3 : v.avg_ci_activity_pct < 30 ? COLOR4 : COLOR5 }}>
                    {v.avg_ci_activity_pct}%
                  </td>
                  <td>{v.avg_heteroplasmy_pct}%</td>
                  <td>{v.leigh_mri_pct}%</td>
                  <td>{v.stroke_like_pct}%</td>
                  <td>{v.exercise_intolerance_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* CI activity & heteroplasmy bands side by side */}
      <div className="row mb-4">
        <div className="col-md-6">
          <SectionCard title="CI Activity Distribution" borderColor={COLOR3}>
            {Object.entries(ci_bands).map(([band, count]) => (
              <Bar key={band} label={`CI ${band}`} value={Math.round((count / 40) * 100)}
                color={band === '<10%' ? COLOR3 : band === '10-25%' ? COLOR4 : COLOR5} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Heteroplasmy Distribution (blood)" borderColor={COLOR2}>
            {Object.entries(het_bands).map(([band, count]) => (
              <Bar key={band} label={`${band} heteroplasmy`} value={Math.round((count / 40) * 100)}
                color={band === '>80%' ? COLOR3 : band === '60-80%' ? COLOR4 : COLOR2} />
            ))}
          </SectionCard>
        </div>
      </div>

      {/* CI Assembly pathway */}
      {assembly.length > 0 && (
        <SectionCard title="CI Assembly Pathway — Where MT-ND4L Mutations Stall (PP-b Checkpoint)" borderColor={COLOR}>
          {assembly.map((step, i) => (
            <div key={i} className="d-flex gap-3 mb-3 pb-3 border-bottom">
              <div className="fw-bold text-white rounded-circle d-flex align-items-center justify-content-center"
                style={{ minWidth: 32, height: 32, backgroundColor: i === 2 ? COLOR3 : COLOR }}>
                {step.step}
              </div>
              <div>
                <div className="fw-semibold small" style={{ color: i === 2 ? COLOR3 : COLOR }}>
                  {step.intermediate}
                  {i === 2 && <span className="ms-2 badge bg-danger">MT-ND4L STALL POINT</span>}
                </div>
                <div className="text-muted small">{step.components}</div>
                <div className="small mt-1">{step.note}</div>
              </div>
            </div>
          ))}
        </SectionCard>
      )}

      {/* Outcomes */}
      <SectionCard title="Outcome Distribution (40-patient cohort)" borderColor={COLOR2}>
        {outcomes.map((o, i) => (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span>{o.outcome}</span><span className="text-muted">{o.count} pts</span>
            </div>
            <div className="progress" style={{ height: 10 }}>
              <div className="progress-bar" style={{
                width: `${(o.count / 40) * 100}%`,
                backgroundColor: o.outcome.includes('Deceased') ? '#616161' : COLOR2
              }} />
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Patient table (first 15) */}
      <SectionCard title="Patient Cohort Sample (seed-771)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}>
              <th>ID</th><th>Phenotype</th><th>Variant</th><th>Heter%</th>
              <th>CI%</th><th>Lactate</th><th>Leigh MRI</th><th>Stroke-like</th><th>Outcome</th>
            </tr></thead>
            <tbody>
              {pts.slice(0, 15).map((p, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{p.id}</td>
                  <td style={{ maxWidth: 180, fontSize: '0.75rem' }}>{p.phenotype}</td>
                  <td>{p.variant}</td>
                  <td>{p.heteroplasmy_pct}%</td>
                  <td style={{ color: p.ci_pct < 15 ? COLOR3 : p.ci_pct < 30 ? COLOR4 : COLOR5 }}>
                    {p.ci_pct}%
                  </td>
                  <td>{p.lactate}</td>
                  <td>{p.leigh_mri ? '✅' : '—'}</td>
                  <td>{p.stroke_like ? '✅' : '—'}</td>
                  <td style={{ fontSize: '0.72rem' }}>{p.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-muted small mt-2">Showing 15 of {pts.length} patients (seed-771, synthetic cohort)</p>
      </SectionCard>
    </div>
  );
}

// ── Tab: DDx & Treatment ──────────────────────────────────────────────────────
function DDxTab({ overview, breakdown }) {
  const ddx = breakdown?.differential_diagnosis || [];
  const contraindications = overview?.absolute_contraindications || [];
  const mandatory = overview?.mandatory_empiric_treatments || [];
  const levelc = overview?.level_c_treatments || [];
  const alerts = overview?.clinical_alerts || [];

  return (
    <div>
      {/* Differential Diagnosis */}
      <SectionCard title="Differential Diagnosis — MT-ND4L vs Key Mimics" borderColor={COLOR3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}>
              <th>Entity</th><th>Distinguishing Feature</th><th>Key Test</th>
            </tr></thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ minWidth: 160 }}>{d.entity}</td>
                  <td>{d.distinguishing_feature}</td>
                  <td className="text-muted">{d.key_test}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Absolute contraindications */}
      <SectionCard title="Absolute Contraindications" borderColor={COLOR3}>
        <ul className="mb-0 small">
          {contraindications.map((c, i) => (
            <li key={i} className="mb-1"><span className="text-danger fw-semibold">🚫 </span>{c}</li>
          ))}
        </ul>
      </SectionCard>

      {/* Mandatory empiric treatments */}
      <SectionCard title="Mandatory Empiric Treatments" borderColor={COLOR5}>
        <ul className="mb-0 small">
          {mandatory.map((t, i) => (
            <li key={i} className="mb-1"><span style={{ color: COLOR5 }}>✅ </span>{t}</li>
          ))}
        </ul>
      </SectionCard>

      {/* Level C treatments */}
      <SectionCard title="Level C Treatments" borderColor={COLOR2}>
        <ul className="mb-0 small">
          {levelc.map((t, i) => (
            <li key={i} className="mb-1"><span style={{ color: COLOR2 }}>🔵 </span>{t}</li>
          ))}
        </ul>
        {overview?.preferred_aed && (
          <div className="mt-2 small"><strong>Preferred AED:</strong> {overview.preferred_aed}</div>
        )}
      </SectionCard>

      {/* CI residual range guide */}
      <SectionCard title="CI Residual Activity Guide — Phenotype Thresholds" borderColor={COLOR4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}>
              <th>CI Activity</th><th>Phenotype</th><th>Heteroplasmy</th>
            </tr></thead>
            <tbody>
              <tr><td className="fw-semibold" style={{ color: COLOR3 }}>5–22%</td><td>Leigh Syndrome (severe)</td><td>≥70%</td></tr>
              <tr><td className="fw-semibold" style={{ color: COLOR4 }}>8–26%</td><td>MELAS-like Overlap (stroke-like)</td><td>≥65%</td></tr>
              <tr><td className="fw-semibold" style={{ color: '#f9a825' }}>18–38%</td><td>Leigh Moderate / Exercise (Thr43Ala)</td><td>40–85%</td></tr>
              <tr><td className="fw-semibold" style={{ color: COLOR5 }}>28–48%</td><td>Exercise Intolerance Myopathy</td><td>50–74%</td></tr>
              <tr><td className="fw-semibold" style={{ color: COLOR2 }}>22–50%</td><td>KSS / CPEO (large deletion)</td><td>variable</td></tr>
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <div>
      <SectionCard title="Gene & Protein" borderColor={COLOR}>
        <div className="row g-2 small">
          {[
            ['Gene', data.gene], ['OMIM Gene', data.omim_gene],
            ['Full Name', data.full_name], ['Protein', data.protein_name],
            ['aa Length', `${data.aa_length} aa`], ['MW', `${data.molecular_weight_kda} kDa`],
            ['TM Helices', data.tm_helices], ['rCRS', data.rcrs_positions],
            ['Strand', data.strand], ['Module', data.module],
          ].map(([k, v], i) => (
            <div key={i} className="col-md-6"><strong>{k}:</strong> {v}</div>
          ))}
          <div className="col-12 mt-1 text-muted fst-italic">{data.size_context}</div>
        </div>
      </SectionCard>

      <SectionCard title="Assembly Position" borderColor={COLOR}>
        <p className="small mb-0">{data.assembly_position}</p>
      </SectionCard>

      <SectionCard title="No Primary LHON" borderColor={COLOR3}>
        <p className="small mb-0 text-danger fw-semibold">{data.no_primary_lhon}</p>
      </SectionCard>

      <SectionCard title="Key Variants" borderColor={COLOR4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}><th>Variant</th><th>Details</th></tr></thead>
            <tbody>
              {Object.entries(data.key_variants || {}).map(([variant, detail], i) => (
                <tr key={i}>
                  <td className="fw-semibold">{variant}</td>
                  <td>{detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Biochemical Fingerprint" borderColor={COLOR2}>
        {Object.entries(data.biochemical_fingerprint || {}).map(([k, v], i) => (
          <div key={i} className="mb-2 small">
            <strong className="text-capitalize">{k.replace(/_/g, ' ')}:</strong> {v}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Heteroplasmy Thresholds" borderColor={COLOR2}>
        {Object.entries(data.heteroplasmy_thresholds || {}).map(([k, v], i) => (
          <div key={i} className="mb-2 small">
            <strong>{k.replace(/_/g, ' ')}:</strong> {v}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Recommended Treatments" borderColor={COLOR5}>
        {Object.entries(data.recommended_treatments || {}).map(([k, v], i) => (
          <div key={i} className="mb-1 small">
            <strong>{k.replace(/_/g, ' ').toUpperCase()}:</strong> {v}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Absolute Contraindications" borderColor={COLOR3}>
        {Object.entries(data.absolute_contraindications || {}).map(([k, v], i) => (
          <div key={i} className="mb-1 small text-danger">
            <strong>🚫 {k}:</strong> {v}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Specialist Monitoring" borderColor={COLOR}>
        {Object.entries(data.specialist_monitoring || {}).map(([k, v], i) => (
          <div key={i} className="mb-1 small">
            <strong>{k}:</strong> {v}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="KSS Annual Holter Policy" borderColor={COLOR3}>
        <p className="small mb-0">{data.kss_holter_policy}</p>
      </SectionCard>

      <SectionCard title="WES Coverage Warning" borderColor={COLOR3}>
        <p className="small mb-0 fw-semibold text-danger">{data.wes_coverage}</p>
      </SectionCard>

      <SectionCard title="Key References" borderColor={COLOR2}>
        <ol className="small mb-0">
          {(data.key_references || []).map((r, i) => (
            <li key={i} className="mb-1">{r}</li>
          ))}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MTND4LPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/mtnd4l/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setErr(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1 && !breakdown)
      fetch(`${API}/api/mtnd4l/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setErr(e.message));
    if (tab === 3 && !defs)
      fetch(`${API}/api/mtnd4l/definitions`)
        .then(r => r.json()).then(setDefs).catch(e => setErr(e.message));
  }, [tab]);

  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;

  return (
    <div className="container-fluid py-4">
      <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
        MT-ND4L — SMALLEST CI mtDNA Subunit — ND4/ND4L Antiporter Hairpin Connector
      </h4>
      <p className="text-muted small mb-3">
        98 aa · 10.7 kDa · 3 TM helices · H-strand rCRS 10470-10766 · OMIM *516004 ·
        Leigh Syndrome / MELAS-Leigh Overlap / Exercise Intolerance — NO primary LHON ·
        40-patient cohort (seed-771)
      </p>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-semibold' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <LeighVsTab data={breakdown} />}
      {tab === 2 && <DDxTab overview={overview} breakdown={breakdown} />}
      {tab === 3 && <DefinitionsTab data={defs} />}
    </div>
  );
}
