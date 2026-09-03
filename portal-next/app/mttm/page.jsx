'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Cohort', 'DDx & Management', 'Definitions'];
const COLOR  = '#1565c0';   // deep blue — tRNA-Met / H-strand / SIXTH tRNA
const LIGHT  = '#e3f2fd';
const COLOR2 = '#1976d2';   // medium blue — CPEO / myopathy
const COLOR3 = '#b71c1c';   // dark red — absolute CIs / initiation block alert
const COLOR4 = '#1b5e20';   // dark green — biochemical fingerprint / OXPHOS
const COLOR5 = '#4a148c';   // deep indigo — MARS2 DDx / nuclear / dual function

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
  const hmap = data.heteroplasmy_clinical_map || [];
  const gf = data.gene_facts || {};
  const dual_alert = data.dual_function_alert || {};
  const h_note = data.h_strand_note || {};
  const bf = data.biochemical_fingerprint || {};

  return (
    <>
      {/* Gene header */}
      <SectionCard borderColor={COLOR}>
        <div className="d-flex align-items-center gap-3 mb-2">
          <span style={{ fontSize: 40 }}>🧬</span>
          <div>
            <h4 className="fw-bold mb-0" style={{ color: COLOR }}>{data.title}</h4>
            <div className="text-muted small">{data.subtitle}</div>
            <span className="badge" style={{ backgroundColor: COLOR, fontSize: '0.75rem' }}>{data.omim}</span>
          </div>
        </div>
      </SectionCard>

      {/* Dual-function alert — prominent */}
      <div className="alert border-0 mb-4" style={{ backgroundColor: '#f3e5f5', borderLeft: `6px solid ${COLOR5}` }}>
        <strong style={{ color: COLOR5 }}>⚡ DUAL-FUNCTION tRNA — INITIATOR + ELONGATOR — UNIQUE IN mt-GENOME:</strong>
        <br />
        MT-TM encodes the <strong>ONLY mitochondrial methionine tRNA</strong>, serving as both
        initiator (N-formyl-Met for AUG start codons) AND elongator (Met at AUG+AUA via modified
        CAU anticodon). At <strong>&gt;75–80% heteroplasmy</strong>, MT-TM mutations impair
        translation <strong>INITIATION across all 13 mtDNA-encoded subunits</strong> — uniquely
        severe among mt-tRNA mutations.
      </div>

      {/* H-strand note */}
      <div className="alert border-0 mb-4" style={{ backgroundColor: '#e8f5e9', borderLeft: `6px solid ${COLOR4}` }}>
        <strong style={{ color: COLOR4 }}>✅ H-STRAND ENCODED — No L-strand NGS Pitfall:</strong>
        <br />
        MT-TM is H-strand encoded (unlike adjacent <strong>MT-TQ which is L-strand</strong>).
        Standard NGS H-strand variant calling correctly detects MT-TM variants.
        Verify the 2 nt MT-TQ/MT-TM junction (rCRS 4400–4402) to avoid strand-assignment confusion.
      </div>

      {/* KPIs */}
      <SectionCard title="40-Patient Cohort — Key Statistics (seed-811)" borderColor={COLOR2}>
        <div className="row g-2">
          <KPI label="Patients"                   value={s.n_patients} />
          <KPI label="Avg Blood Heteroplasmy"     value={`${s.avg_heteroplasmy_blood_pct}%`} />
          <KPI label="Avg CI Activity"            value={`${s.avg_ci_activity_pct_normal}%`}  color={COLOR3} />
          <KPI label="Avg CIV Activity"           value={`${s.avg_civ_activity_pct_normal}%`} color={COLOR3} />
          <KPI label="CII (Nuclear — NORMAL)"     value={`${s.avg_cii_activity_pct_normal}%`} color="#2e7d32" />
          <KPI label="CPEO"                       value={`${s.pct_cpeo}%`}                    color={COLOR2} />
          <KPI label="Myopathy"                   value={`${s.pct_myopathy}%`}               color={COLOR2} />
          <KPI label="Cardiomyopathy"             value={`${s.pct_cardiomyopathy}%`}         color={COLOR3} />
          <KPI label="Exercise Intolerance"       value={`${s.pct_exercise_intolerance}%`}   color={COLOR} />
          <KPI label="SNHL"                       value={`${s.pct_snhl}%`}                   color={COLOR5} />
          <KPI label="Ragged-Red Fibres"          value={`${s.pct_ragged_red_fibres}%`}      color={COLOR} />
          <KPI label="Avg Onset (yr)"             value={s.avg_age_onset_yr} />
        </div>
      </SectionCard>

      {/* Gene facts */}
      <SectionCard title="Gene Facts — MT-TM" borderColor={COLOR}>
        <div className="row g-2 small">
          {Object.entries(gf).map(([k, v]) => (
            <div key={k} className="col-12 col-md-6">
              <span className="fw-semibold text-capitalize">{k.replace(/_/g,' ')}: </span>
              <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Biochemical fingerprint */}
      <SectionCard title="Biochemical Fingerprint — Combined CI+CIV Deficiency (CII NORMAL)" borderColor={COLOR4}>
        <div className="row g-3">
          {[
            { label: 'Complex I',  val: bf.complex_i,  color: COLOR3 },
            { label: 'Complex II (nuclear — NORMAL)', val: bf.complex_ii, color: '#2e7d32' },
            { label: 'Complex IV', val: bf.complex_iv, color: COLOR3 },
          ].map(c => (
            <div key={c.label} className="col-12 col-md-4">
              <div className="card text-center shadow-sm">
                <div className="card-body py-2">
                  <div className="fw-bold" style={{ color: c.color }}>{c.label}</div>
                  <div className="small text-muted">{c.val}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
        <p className="small text-muted mt-3 mb-0">{bf.mechanism}</p>
      </SectionCard>

      {/* Phenotype distribution */}
      <SectionCard title="Variant Phenotype Distribution — 40-patient cohort" borderColor={COLOR2}>
        {pheno_dist.map((p, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between small mb-1">
              <span><strong>{p.variant}</strong> — {p.phenotype}</span>
              <span className="text-muted">{p.pct}%</span>
            </div>
            <div className="progress mb-1" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${p.pct}%`, backgroundColor: COLOR }} />
            </div>
            <div className="text-muted" style={{ fontSize: '0.72rem' }}>{p.position}</div>
          </div>
        ))}
      </SectionCard>

      {/* Heteroplasmy–phenotype map */}
      <SectionCard title="Heteroplasmy–Phenotype Threshold Map" borderColor={COLOR5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: LIGHT }}>
              <tr>
                <th>Heteroplasmy (%)</th>
                <th>Expected Phenotype</th>
              </tr>
            </thead>
            <tbody>
              {hmap.map((h, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{h.threshold_pct}</td>
                  <td>{h.expected_phenotype}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Clinical alerts */}
      <SectionCard title="Clinical Alerts" borderColor={COLOR3}>
        {alerts.map((a, i) => (
          <div key={i} className="alert mb-2 py-2 border-0" style={{ backgroundColor: i % 2 === 0 ? '#fce4ec' : '#e8eaf6' }}>
            <strong style={{ color: COLOR3 }}>{a.alert}:</strong> {a.detail}
          </div>
        ))}
      </SectionCard>

      {/* Key molecular features */}
      <SectionCard title="Key Molecular Features" borderColor={COLOR}>
        <ul className="mb-0 small">
          {mol_feats.map((f, i) => <li key={i}>{f}</li>)}
        </ul>
      </SectionCard>
    </>
  );
}

// ── Tab: Variants & Cohort ────────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const vb = data.variant_breakdown || [];
  const mgmt = data.management_by_variant || [];
  const aci = data.absolute_contraindications || [];
  const safe = data.safe_interventions || [];
  const s = data.cohort_statistics || {};

  return (
    <>
      <SectionCard title="Variant Breakdown — 40-patient cohort (seed-811)" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead style={{ backgroundColor: LIGHT }}>
              <tr>
                <th>Variant</th><th>N</th><th>% cohort</th>
                <th>Avg Hetero</th><th>Avg CI%</th><th>Avg CIV%</th>
                <th>CPEO%</th><th>Myop%</th><th>Cardio%</th>
              </tr>
            </thead>
            <tbody>
              {vb.map((v, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ color: COLOR }}>{v.variant}</td>
                  <td>{v.n}</td>
                  <td>{v.pct_of_cohort}%</td>
                  <td>{v.avg_heteroplasmy}%</td>
                  <td style={{ color: COLOR3 }}>{v.avg_ci_pct_normal}%</td>
                  <td style={{ color: COLOR3 }}>{v.avg_civ_pct_normal}%</td>
                  <td>{v.pct_cpeo}%</td>
                  <td>{v.pct_myopathy}%</td>
                  <td>{v.pct_cardiomyopathy}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Cohort Summary Statistics" borderColor={COLOR2}>
        <div className="row g-2">
          <KPI label="Patients"              value={s.n_patients} />
          <KPI label="Avg Heteroplasmy"      value={`${s.avg_heteroplasmy_blood_pct}%`} />
          <KPI label="Avg CI%"               value={`${s.avg_ci_activity_pct_normal}%`}  color={COLOR3} />
          <KPI label="Avg CIV%"              value={`${s.avg_civ_activity_pct_normal}%`} color={COLOR3} />
          <KPI label="CII% (NORMAL)"         value={`${s.avg_cii_activity_pct_normal}%`} color={COLOR4} />
          <KPI label="CPEO"                  value={`${s.pct_cpeo}%`}                    color={COLOR2} />
        </div>
        {[
          { label: 'CPEO',                    value: s.pct_cpeo },
          { label: 'Myopathy',                value: s.pct_myopathy },
          { label: 'Exercise Intol.',          value: s.pct_exercise_intolerance },
          { label: 'Cardiomyopathy',           value: s.pct_cardiomyopathy, color: COLOR3 },
          { label: 'SNHL',                     value: s.pct_snhl, color: COLOR5 },
          { label: 'Lactic Acidosis',          value: s.pct_lactic_acidosis },
          { label: 'Leigh-like MRI',           value: s.pct_leigh_like_mri },
          { label: 'Ragged-Red Fibres',        value: s.pct_ragged_red_fibres },
          { label: 'Initiation Block (>75%)',  value: s.pct_initiation_block_high_hetero, color: COLOR5 },
        ].map((b, i) => (
          <Bar key={i} label={b.label} value={b.value} color={b.color || COLOR} />
        ))}
      </SectionCard>

      <SectionCard title="Management by Variant" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: LIGHT }}>
              <tr><th>Variant</th><th>CPEO Risk</th><th>Cardio Risk</th><th>Key Action</th></tr>
            </thead>
            <tbody>
              {mgmt.map((m, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ color: COLOR }}>{m.variant}</td>
                  <td>{m.cpeo_risk}</td>
                  <td>{m.cardio_risk}</td>
                  <td>{m.key_action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Absolute Contraindications" borderColor={COLOR3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#fce4ec' }}>
              <tr><th>Drug</th><th>Reason</th></tr>
            </thead>
            <tbody>
              {aci.map((a, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ color: COLOR3 }}>{a.drug}</td>
                  <td>{a.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Safe Interventions (evidence-based)" borderColor={COLOR4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: LIGHT }}>
              <tr><th>Intervention</th><th>Evidence</th></tr>
            </thead>
            <tbody>
              {safe.map((s, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ color: COLOR4 }}>{s.intervention}</td>
                  <td>{s.evidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: DDx & Management ─────────────────────────────────────────────────────
function DdxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ddx = data.ddx_table || [];

  return (
    <>
      <SectionCard title="Differential Diagnosis Table" borderColor={COLOR5}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead style={{ backgroundColor: '#e8eaf6' }}>
              <tr>
                <th>Entity</th><th>Inheritance</th><th>Phenotype</th>
                <th>Biochemistry</th><th>NGS</th><th>Distinctive</th>
              </tr>
            </thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i} style={d.entity.includes('MT-TM') ? { backgroundColor: LIGHT, fontWeight: 600 } : {}}>
                  <td style={{ color: d.entity.includes('MT-TM') ? COLOR : 'inherit' }}>{d.entity}</td>
                  <td>{d.inheritance}</td>
                  <td>{d.phenotype}</td>
                  <td>{d.biochemistry}</td>
                  <td>{d.ngs}</td>
                  <td>{d.distinctive}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="KEY DDx Exclusion Rules — MT-TM" borderColor={COLOR3}>
        {[
          { rule: 'NO stroke-like episodes', note: 'Stroke-like episodes → MT-TL1 / MELAS (m.3243A>G); NOT MT-TM' },
          { rule: 'NO myoclonic epilepsy / MSL', note: 'Myoclonic epilepsy + multiple symmetrical lipomatosis → MT-TK / MERRF; NOT MT-TM' },
          { rule: 'NO isolated HCM without CPEO', note: 'Isolated HCM at low heteroplasmy → MT-TI (m.4300A>G); NOT MT-TM' },
          { rule: 'NO MIDM', note: 'Maternally inherited diabetes + deafness → MT-TE (m.14709T>C); NOT MT-TM' },
          { rule: 'ARSAL DDx for spastic ataxia + leukoencephalopathy', note: 'MARS2 biallelic → ARSAL; AR not maternal; cerebellar ataxia NOT ophthalmoplegia; WES-detectable' },
          { rule: 'BTBGD MANDATORY EXCLUSION', note: 'SLC19A3 biallelic → biotin+thiamine-responsive Leigh mimic; treat first before attributing Leigh MRI to MT-TM' },
        ].map((item, i) => (
          <div key={i} className="alert mb-2 py-2 border-0" style={{ backgroundColor: i % 2 === 0 ? '#fce4ec' : '#e8eaf6' }}>
            <strong style={{ color: COLOR3 }}>✗ {item.rule}:</strong> {item.note}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Perioperative / Emergency Protocol" borderColor={COLOR2}>
        <ul className="mb-0 small">
          <li><strong>GIR 6–8 mg/kg/min MANDATORY</strong> during any fasting period — prevent catabolic crisis</li>
          <li><strong>NEVER fast</strong> MT-TM patients without IV glucose support</li>
          <li><strong>Propofol ABSOLUTE CI</strong> — use inhalational anaesthesia (sevoflurane) instead</li>
          <li><strong>Avoid Metformin</strong> — CI inhibitor → fatal lactic acidosis</li>
          <li><strong>Linezolid CI</strong> — inhibits mt-23S rRNA translation; use alternative antibiotics</li>
          <li><strong>Lactate monitoring</strong> — baseline + intra/post-operative; target &lt;2.0 mmol/L</li>
          <li><strong>Metabolic crisis</strong> — IV dextrose + Thiamine + Biotin immediately; ICU admission</li>
          <li><strong>High-heteroplasmy patients (&gt;75%)</strong> — initiation block risk; ICU-level metabolic monitoring</li>
        </ul>
      </SectionCard>

      <SectionCard title="Surveillance Schedule by System" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: LIGHT }}>
              <tr><th>System</th><th>Test</th><th>Frequency</th><th>Trigger</th></tr>
            </thead>
            <tbody>
              {[
                { sys: 'Ophthalmology', test: 'Orthoptic exam, ptosis/ophthalmoplegia assessment', freq: 'Annual', trigger: 'All MT-TM patients' },
                { sys: 'Neurology',     test: 'Clinical + lactate + CK', freq: 'Annual',  trigger: 'All MT-TM patients' },
                { sys: 'Cardiology',    test: 'Echo + Holter', freq: 'Annual', trigger: 'm.4460GA or any cardiomyopathy' },
                { sys: 'Audiology',     test: 'Pure-tone audiogram', freq: 'Annual', trigger: 'm.4450TC or SNHL' },
                { sys: 'Neurology MRI', test: 'Brain MRI with DWI', freq: 'PRN', trigger: 'm.4460GA + high hetero; Leigh-like screening' },
                { sys: 'Endocrine',     test: 'Glucose + HbA1c', freq: 'Annual', trigger: 'Large deletions (KSS risk)' },
                { sys: 'Muscle biopsy', test: 'Histochemistry + OXPHOS enzymes', freq: 'Once', trigger: 'Diagnostic confirmation' },
                { sys: 'Metabolic ICU', test: 'Lactate + glucose + blood gas', freq: 'Crisis', trigger: 'Any heteroplasmy >75% + acute intercurrent illness' },
              ].map((r, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{r.sys}</td>
                  <td>{r.test}</td>
                  <td>{r.freq}</td>
                  <td>{r.trigger}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const sections = [
    { title: 'Gene Definitions',        key: 'gene_definitions',       color: COLOR },
    { title: 'Biochemical Definitions', key: 'biochemical_definitions', color: COLOR4 },
    { title: 'Clinical Definitions',    key: 'clinical_definitions',   color: COLOR2 },
    { title: 'NGS / Lab Definitions',   key: 'ngs_definitions',        color: COLOR3 },
    { title: 'Drug Definitions',        key: 'drug_definitions',       color: COLOR5 },
  ];

  return (
    <>
      {sections.map(s => (
        data[s.key] && (
          <SectionCard key={s.key} title={s.title} borderColor={s.color}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small mb-0">
                <thead style={{ backgroundColor: LIGHT }}>
                  <tr><th style={{ width: '28%' }}>Term</th><th>Definition</th></tr>
                </thead>
                <tbody>
                  {data[s.key].map((d, i) => (
                    <tr key={i}>
                      <td className="fw-semibold" style={{ color: s.color }}>{d.term}</td>
                      <td>{d.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        )
      ))}

      {data.references && (
        <SectionCard title="Key References" borderColor={COLOR5}>
          <ul className="mb-0 small">
            {data.references.map((r, i) => (
              <li key={i}><strong>{r.ref}:</strong> {r.citation}</li>
            ))}
          </ul>
        </SectionCard>
      )}
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MttmPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mttm/overview`).then(r => r.json()),
      fetch(`${API}/api/mttm/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mttm/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefs(df);
    }).catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">{err}</div>;

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: 32, color: COLOR }}>🧬</span>
        <div>
          <h3 className="fw-bold mb-0" style={{ color: COLOR }}>MT-TM — tRNA-Met Expert Dashboard</h3>
          <small className="text-muted">
            Combined CI+CIV Deficiency · CPEO · Myopathy ·{' '}
            <strong style={{ color: COLOR5 }}>Dual Initiator+Elongator</strong> ·{' '}
            <strong style={{ color: COLOR4 }}>H-strand rCRS 4402–4469</strong> ·{' '}
            SIXTH tRNA · seed-811
          </small>
        </div>
      </div>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <VariantsTab data={breakdown} />}
      {tab === 2 && <DdxTab data={breakdown} />}
      {tab === 3 && <DefsTab data={defs} />}
    </div>
  );
}
