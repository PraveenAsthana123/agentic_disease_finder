'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Leigh vs MELAS vs Exercise', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#1a237e';   // deep indigo — mtDNA CI / maternal inheritance
const LIGHT  = '#e8eaf6';
const COLOR2 = '#283593';
const COLOR3 = '#b71c1c';   // dark red — Leigh / severe CI (no LHON primary)
const COLOR4 = '#e65100';   // deep orange — MELAS-like overlap
const COLOR5 = '#2e7d32';   // green — treatments / exercise phenotype

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
  const onset = data.onset_distribution || {};
  const mol_feats = data.key_molecular_features || [];
  const alerts = data.clinical_alerts || [];

  return (
    <div>
      {/* No-LHON Banner */}
      <div className="alert fw-bold mb-4" style={{ backgroundColor: '#fff3e0', borderLeft: `5px solid ${COLOR3}`, color: '#bf360c' }}>
        🔴 MT-ND2: NO PRIMARY LHON MUTATION — unlike ND1 (m.3460G>A #2), ND4 (m.11778G>A #1), ND6 (m.14484T>C #3).
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
        <KPI label="Protein Size" value={`${data.aa_length} aa`} color={COLOR} />
        <KPI label="Module" value="Prox-Middle Antiporter" color={COLOR} />
      </div>

      {/* Gene Banner */}
      <SectionCard title="MT-ND2 — Proximal-Middle Antiporter Module" borderColor={COLOR}>
        <div className="row">
          <div className="col-md-6">
            <table className="table table-sm table-bordered small mb-0">
              <tbody>
                <tr><td className="fw-bold">Gene</td><td>{data.gene} ({data.omim_gene})</td></tr>
                <tr><td className="fw-bold">Protein</td><td>{data.protein}</td></tr>
                <tr><td className="fw-bold">Module</td><td>{data.module}</td></tr>
                <tr><td className="fw-bold">Genome</td><td>mtDNA H-strand, rCRS 4470–5511</td></tr>
                <tr><td className="fw-bold">Inheritance</td><td>{data.inheritance}</td></tr>
                <tr><td className="fw-bold">Primary Disease</td><td>{data.primary_disease}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="col-md-6">
            <div className="p-3 rounded" style={{ backgroundColor: LIGHT }}>
              <div className="fw-bold small mb-2" style={{ color: COLOR }}>⚠️ No Primary LHON</div>
              <p className="small mb-2">{data.no_lhon_note}</p>
              <div className="fw-bold small mb-1">CI Residual Range:</div>
              <code className="small">{data.ci_residual_range}</code>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Clinical Features */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Features — 40-Patient Cohort" borderColor={COLOR3}>
            {feats.slice(0, 9).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct}
                color={f.pct > 60 ? COLOR3 : f.pct > 30 ? COLOR4 : COLOR5} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Clinical Features (continued)" borderColor={COLOR3}>
            {feats.slice(9).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct}
                color={f.pct > 60 ? COLOR3 : f.pct > 30 ? COLOR4 : COLOR5} />
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Phenotype Distribution + Onset */}
      <div className="row">
        <div className="col-md-7">
          <SectionCard title="Phenotype Distribution" borderColor={COLOR4}>
            {pheno_dist.map(p => (
              <div key={p.phenotype} className="d-flex justify-content-between align-items-center mb-2">
                <span className="small">{p.phenotype}</span>
                <span className="badge rounded-pill" style={{ backgroundColor: COLOR4 }}>{p.count}</span>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-5">
          <SectionCard title="Age of Onset Distribution" borderColor={COLOR2}>
            {Object.entries(onset).map(([label, count]) => (
              <div key={label} className="d-flex justify-content-between mb-2 small">
                <span>{label.replace(/_/g, ' ')}</span>
                <span className="fw-bold">{count}</span>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Molecular Features */}
      <SectionCard title="Key Molecular Features — ND2-Module Architecture" borderColor={COLOR}>
        {mol_feats.map(f => (
          <div key={f.feature} className="mb-3 p-2 rounded" style={{ backgroundColor: LIGHT }}>
            <div className="fw-bold small" style={{ color: COLOR }}>{f.feature}</div>
            <div className="small text-muted">{f.significance}</div>
          </div>
        ))}
      </SectionCard>

      {/* Clinical Alerts */}
      <SectionCard title="Clinical Alerts — MT-ND2" borderColor={COLOR3}>
        {alerts.map((a, i) => (
          <div key={i} className="alert alert-light border-start border-4 mb-2 py-2 small"
            style={{ borderColor: a.startsWith('🔴') ? COLOR3 : a.startsWith('⚠️') ? COLOR4 : COLOR }}>
            {a}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Leigh vs MELAS vs Exercise ─────────────────────────────────────────
function PhenotypeTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const variants = data.variant_breakdown || [];
  const assembly = data.assembly_pathway || [];
  const het_bands = data.heteroplasmy_bands || {};
  const ci_bands = data.ci_activity_bands || {};
  const outcomes = data.outcome_distribution || [];

  return (
    <div>
      <SectionCard title="Heteroplasmy Threshold vs Phenotype" borderColor={COLOR2}>
        <div className="row">
          <div className="col-md-4">
            <div className="p-3 rounded mb-3" style={{ backgroundColor: '#ffebee' }}>
              <div className="fw-bold small text-danger mb-1">Leigh Syndrome</div>
              <div className="small">Heteroplasmy ≥70% (blood); severe CI 5-25% residual; bilateral BG/brainstem T2; infantile onset 3-18 months</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-3 rounded mb-3" style={{ backgroundColor: '#fff3e0' }}>
              <div className="fw-bold small text-warning-emphasis mb-1">MELAS-like Overlap</div>
              <div className="small">Heteroplasmy ≥70%; CI 8-28%; stroke-like episodes + Leigh MRI; IV L-Arg ONLY in acute SLE; NOT tPA</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-3 rounded mb-3" style={{ backgroundColor: '#e8f5e9' }}>
              <div className="fw-bold small text-success mb-1">Exercise Intolerance</div>
              <div className="small">Heteroplasmy 50-75%; CI 25-48%; adult onset; ragged-red fibres 55%; exercise lactate diagnostic</div>
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: COLOR2 }}>Heteroplasmy Distribution (blood)</div>
            {Object.entries(het_bands).map(([b, c]) => (
              <Bar key={b} label={b} value={Math.round(c * 100 / 40)}
                color={b === '>80%' ? COLOR3 : b === '60-80%' ? COLOR4 : COLOR5} />
            ))}
          </div>
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: COLOR2 }}>CI Activity Distribution</div>
            {Object.entries(ci_bands).map(([b, c]) => (
              <Bar key={b} label={b} value={Math.round(c * 100 / 40)}
                color={b === '<10%' ? COLOR3 : b === '10-25%' ? COLOR4 : COLOR5} />
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Variant Table */}
      <SectionCard title="Variant Breakdown — 5 Pathogenic/Likely-Pathogenic Variants" borderColor={COLOR3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: COLOR, color: '#fff' }}>
              <tr>
                <th>Variant</th><th>Protein</th><th>Domain</th><th>N</th>
                <th>Avg CI%</th><th>Avg Het%</th><th>Leigh MRI%</th>
                <th>Stroke%</th><th>Exercise%</th><th>Resp Fail%</th>
              </tr>
            </thead>
            <tbody>
              {variants.map(v => (
                <tr key={v.variant}>
                  <td><code>{v.variant}</code></td>
                  <td>{v.protein}</td>
                  <td className="text-muted">{v.domain.substring(0, 45)}…</td>
                  <td>{v.n_patients}</td>
                  <td style={{ color: v.avg_ci_activity_pct < 25 ? COLOR3 : v.avg_ci_activity_pct < 40 ? COLOR4 : COLOR5 }}>
                    {v.avg_ci_activity_pct}%
                  </td>
                  <td>{v.avg_heteroplasmy_pct}%</td>
                  <td>{v.leigh_mri_pct}%</td>
                  <td>{v.stroke_like_pct}%</td>
                  <td>{v.exercise_intolerance_pct}%</td>
                  <td>{v.respiratory_failure_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Assembly Pathway */}
      <SectionCard title="CI Assembly Pathway — MT-ND2 Joins Step 2 (After ND1-Module)" borderColor={COLOR}>
        {assembly.map(step => (
          <div key={step.step} className="d-flex align-items-start mb-3">
            <div className="rounded-circle d-flex align-items-center justify-content-center me-3 flex-shrink-0"
              style={{ width: 36, height: 36, backgroundColor: step.step === 2 ? COLOR3 : COLOR, color: '#fff', fontSize: 14, fontWeight: 'bold' }}>
              {step.step}
            </div>
            <div>
              <div className="fw-bold small" style={{ color: step.step === 2 ? COLOR3 : COLOR }}>
                {step.intermediate}
                {step.step === 2 && <span className="badge ms-2 text-white" style={{ backgroundColor: COLOR3, fontSize: 10 }}>MT-ND2 joins here</span>}
              </div>
              <div className="small text-muted">{step.components}</div>
              <div className="small">{step.note}</div>
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Outcomes */}
      <SectionCard title="Outcome Distribution" borderColor={COLOR2}>
        {outcomes.map(o => (
          <div key={o.outcome} className="d-flex justify-content-between align-items-center mb-2 small">
            <span>{o.outcome}</span>
            <span className="badge rounded-pill" style={{ backgroundColor: o.outcome.includes('Deceased') ? COLOR3 : COLOR }}>{o.count}</span>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: DDx & Treatment ──────────────────────────────────────────────────────
function DdxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ddx = data.differential_diagnosis || [];

  return (
    <div>
      {/* Absolute CIs */}
      <SectionCard title="⛔ Absolute Contraindications — MT-ND2 All Phenotypes" borderColor={COLOR3}>
        <div className="row">
          {(data.variant_breakdown && data.variant_breakdown[0] && data.variant_breakdown[0].notes) ? null : null}
        </div>
        {[
          ['Metformin', 'CI inhibitor — additive lethal in isolated CI deficiency'],
          ['VPA / Valproic Acid', 'CoA sequestration + POLG inhibition + hepatotoxicity — triple danger in Leigh/MELAS'],
          ['Linezolid', 'Inhibits mt-23S rRNA → ALL 7 ND subunit synthesis blocked including ND2; CI assembly collapses'],
          ['Chloramphenicol', 'Mt-ribosome inhibitor — all ND subunit synthesis including ND2 impaired'],
          ['Propofol', 'PRIS (propofol infusion syndrome) + CI inhibition — compounding isolated CI deficiency'],
          ['Ketogenic Diet', 'Severe isolated CI: beta-oxidation FADH2 backlog worsens CI bottleneck — CONTRAINDICATED Leigh/MELAS'],
          ['IV tPA', 'CONTRAINDICATED in MELAS-like stroke-like episodes — NOT thrombotic; haemorrhage risk'],
          ['Fasting / prolonged NPO', 'GIR 6-8 mg/kg/min MANDATORY in Leigh/MELAS crisis — NEVER fast'],
        ].map(([drug, reason]) => (
          <div key={drug} className="d-flex mb-2 p-2 rounded" style={{ backgroundColor: '#ffebee' }}>
            <div className="fw-bold me-3 text-danger small" style={{ minWidth: 180 }}>{drug}</div>
            <div className="small text-muted">{reason}</div>
          </div>
        ))}
      </SectionCard>

      {/* Treatments */}
      <SectionCard title="✅ Recommended Treatments" borderColor={COLOR5}>
        {[
          ['Thiamine B1 (MANDATORY empiric)', 'Level B — 10-20 mg/kg IV in Leigh/MELAS crisis; oral 100-300 mg/day; PDH cofactor; empiric before confirmatory testing', COLOR5],
          ['Biotin (MANDATORY empiric)', '5-20 mg/day empiric pending BTD/SLC19A3 (BTBGD) exclusion — treatable Leigh-like mimic', COLOR5],
          ['GIR 6-8 mg/kg/min', 'Continuous glucose infusion rate — NEVER fast in Leigh/MELAS crisis; prevents catabolism', COLOR5],
          ['CoQ10 ubiquinol (Level C)', '10-20 mg/kg/day — electron shuttle CI bypass pathway support', COLOR2],
          ['Riboflavin B2 (Level C)', 'FAD/FMN CI cofactor — lower evidence than ACAD9 (no FAD domain in ND2 unlike ACAD9)', COLOR2],
          ['Succinate (investigational)', 'CI bypass via FADH2-independent CII route; palliative in Leigh crisis — not standard of care', COLOR2],
          ['LEV (Levetiracetam)', 'Preferred AED for MT-ND2 Leigh/MELAS seizures — avoid VPA (absolute CI); avoid phenobarbital (CI caution)', COLOR5],
          ['L-Arginine IV (MELAS SLE only)', 'ONLY for acute MELAS-like stroke-like episodes — NOT for Leigh-only phenotype; NOT tPA', COLOR4],
          ['KSS: Annual Holter / Pacemaker', 'Annual ECG + Holter for large deletion KSS; pacemaker threshold PR >240 ms or Mobitz II/CHB', COLOR4],
        ].map(([drug, reason, col]) => (
          <div key={drug} className="d-flex mb-2 p-2 rounded" style={{ backgroundColor: '#f1f8e9' }}>
            <div className="fw-bold me-3 small" style={{ minWidth: 220, color: col }}>{drug}</div>
            <div className="small text-muted">{reason}</div>
          </div>
        ))}
      </SectionCard>

      {/* DDx Table */}
      <SectionCard title="Differential Diagnosis" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: COLOR2, color: '#fff' }}>
              <tr><th>Entity</th><th>Distinguishing Features</th><th>Key Test</th></tr>
            </thead>
            <tbody>
              {ddx.map(d => (
                <tr key={d.entity}>
                  <td className="fw-bold" style={{ color: COLOR3 }}>{d.entity}</td>
                  <td>{d.distinguishing_feature}</td>
                  <td className="text-muted">{d.key_test}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────────────
function DefsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <div>
      <SectionCard title="Gene & Protein Definitions" borderColor={COLOR}>
        <table className="table table-sm table-bordered small">
          <tbody>
            {[
              ['Gene', data.gene],
              ['OMIM Gene', data.omim_gene],
              ['Full Name', data.full_name],
              ['Protein', data.protein_name],
              ['Length', `${data.aa_length} aa, ${data.molecular_weight_kda} kDa`],
              ['TM Helices', `${data.tm_helices} TM helices`],
              ['rCRS Position', data.rcrs_positions],
              ['Strand', data.strand],
              ['Module', data.module],
              ['Antiporter Rank', data.antiporter_rank],
              ['Assembly Position', data.assembly_position],
              ['No Primary LHON', data.no_primary_lhon],
            ].map(([k, v]) => (
              <tr key={k}><td className="fw-bold">{k}</td><td>{v}</td></tr>
            ))}
          </tbody>
        </table>
      </SectionCard>

      <SectionCard title="Key Pathogenic Variants" borderColor={COLOR3}>
        <table className="table table-sm table-bordered small">
          <thead style={{ backgroundColor: COLOR3, color: '#fff' }}>
            <tr><th>Variant</th><th>Details</th></tr>
          </thead>
          <tbody>
            {Object.entries(data.key_variants || {}).map(([k, v]) => (
              <tr key={k}><td><code>{k}</code></td><td>{v}</td></tr>
            ))}
          </tbody>
        </table>
      </SectionCard>

      <SectionCard title="Biochemical Fingerprint" borderColor={COLOR2}>
        {Object.entries(data.biochemical_fingerprint || {}).map(([k, v]) => (
          <div key={k} className="mb-3 p-2 rounded" style={{ backgroundColor: LIGHT }}>
            <div className="fw-bold small" style={{ color: COLOR }}>{k.replace(/_/g, ' ')}</div>
            <div className="small">{v}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Heteroplasmy Thresholds" borderColor={COLOR2}>
        {Object.entries(data.heteroplasmy_thresholds || {}).map(([k, v]) => (
          <div key={k} className="mb-2 p-2 rounded" style={{ backgroundColor: LIGHT }}>
            <div className="fw-bold small" style={{ color: COLOR2 }}>{k.replace(/_/g, ' ')}</div>
            <div className="small">{v}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Specialist Monitoring" borderColor={COLOR5}>
        <table className="table table-sm table-bordered small">
          <thead style={{ backgroundColor: COLOR5, color: '#fff' }}>
            <tr><th>Specialty</th><th>Monitoring Protocol</th></tr>
          </thead>
          <tbody>
            {Object.entries(data.specialist_monitoring || {}).map(([k, v]) => (
              <tr key={k}><td className="fw-bold">{k}</td><td>{v}</td></tr>
            ))}
          </tbody>
        </table>
      </SectionCard>

      <SectionCard title="WES Coverage" borderColor={COLOR3}>
        <div className="alert alert-warning small">{data.wes_coverage}</div>
      </SectionCard>

      <SectionCard title="Key References" borderColor={COLOR2}>
        <ul className="small ps-3">
          {(data.key_references || []).map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function Mtnd2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mtnd2/overview`).then(r => r.json()),
      fetch(`${API}/api/mtnd2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mtnd2/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefs(df); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return (
    <div className="container py-4">
      <div className="alert alert-danger">Error loading MT-ND2 data: {err}</div>
    </div>
  );

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4 p-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${COLOR} 0%, ${COLOR3} 100%)` }}>
        <h4 className="mb-1">🧬 MT-ND2 — Proximal-Middle Antiporter Module</h4>
        <div className="small opacity-90">
          Leigh Syndrome · MELAS-Leigh Overlap · Exercise Intolerance Myopathy · KSS/CPEO (Large Deletion) —
          347 aa · 39.5 kDa · 13 TM Helices · H-strand mtDNA rCRS 4470–5511 · NO Primary LHON ·
          OMIM *516001 · Maternal Inheritance · 40-patient cohort seed-763
        </div>
      </div>

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PhenotypeTab data={breakdown} />}
      {tab === 2 && <DdxTab data={breakdown} />}
      {tab === 3 && <DefsTab data={defs} />}
    </div>
  );
}
