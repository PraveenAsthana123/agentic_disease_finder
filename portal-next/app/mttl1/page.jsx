'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'MELAS vs MIDD vs KSS', 'DDx & SLE Management', 'Definitions'];
const COLOR  = '#00695c';   // deep teal — tRNA / MELAS (mt-translation gene)
const LIGHT  = '#e0f2f1';
const COLOR2 = '#00838f';   // teal-cyan — MELAS phenotype
const COLOR3 = '#b71c1c';   // dark red — stroke-like episode / severe
const COLOR4 = '#1565c0';   // deep blue — MIDD / diabetes
const COLOR5 = '#4a148c';   // deep purple — pan-OXPHOS / biochemical fingerprint

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

  return (
    <div>
      {/* Banner */}
      <div className="alert fw-bold mb-4" style={{ backgroundColor: LIGHT, borderLeft: `5px solid ${COLOR}`, color: COLOR }}>
        🟢 MT-TL1: tRNA-LEU(UUR) — <strong>m.3243A&gt;G: MOST COMMON pathogenic mtDNA variant (~1 in 400 adults)</strong>.
        Pan-OXPHOS deficiency (CI+CIII+CIV; CII NORMAL = mt-translation fingerprint).
        <span className="text-danger"> IV tPA ABSOLUTE CI</span> — stroke-like episodes are NOT thrombotic.
        IV L-Arginine Level B for acute SLE. Metformin ABSOLUTE CI (including MIDD).
      </div>

      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Patients" value={data.n_patients} color={COLOR} />
        <KPI label="Avg CI Activity" value={`${s.avg_ci_activity_pct}%`} color={COLOR3} />
        <KPI label="Avg Lactate" value={`${s.avg_lactic_acid_mmolL} mmol/L`} color={COLOR2} />
        <KPI label="Stroke-like Episodes" value={`${s.stroke_like_episode_pct}%`} color={COLOR3} />
        <KPI label="SNHL" value={`${s.sensorineural_hearing_loss_pct}%`} color={COLOR2} />
        <KPI label="Diabetes (DM)" value={`${s.diabetes_mellitus_pct}%`} color={COLOR4} />
        <KPI label="Seizures" value={`${s.seizures_pct}%`} color={COLOR3} />
        <KPI label="Exercise Intolerance" value={`${s.exercise_intolerance_pct}%`} color={COLOR2} />
        <KPI label="MELAS cortical MRI" value={`${s.melas_mri_cortical_pct}%`} color={COLOR} />
      </div>

      {/* Clinical Alerts */}
      <SectionCard title="⚠️ Critical Clinical Alerts" borderColor={COLOR3}>
        {alerts.map((a, i) => (
          <div key={i} className="mb-2 p-2 rounded small" style={{ background: i < 3 ? '#ffebee' : i < 5 ? '#fff8e1' : '#e8f5e9' }}>
            {a}
          </div>
        ))}
      </SectionCard>

      {/* Heteroplasmy-Phenotype Map */}
      <SectionCard title="Heteroplasmy → Clinical Phenotype Map (m.3243A>G — use URINE, not blood)" borderColor={COLOR4}>
        <div className="small text-muted mb-2">Blood underestimates by 20-30%. Urine epithelial cells = preferred test.</div>
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead style={{ backgroundColor: COLOR4, color: 'white' }}>
              <tr>
                <th>Heteroplasmy Tier</th>
                <th>Clinical Phenotype</th>
                <th>SLE Risk</th>
                <th>Metformin CI?</th>
              </tr>
            </thead>
            <tbody>
              {hmap.map((row, i) => (
                <tr key={i} style={{ backgroundColor: i === 4 ? '#ffebee' : i === 3 ? '#fff3e0' : i <= 1 ? '#e8f5e9' : 'white' }}>
                  <td className="fw-bold small">{row.tier}</td>
                  <td className="small">{row.phenotype}</td>
                  <td className="small">{row.sle_risk}</td>
                  <td className="fw-bold small text-danger">{row.metformin_ci}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Phenotype Distribution */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Phenotype Distribution (40 patients, seed-783)" borderColor={COLOR}>
            {pheno_dist.map((pd, i) => (
              <div key={i} className="d-flex justify-content-between small mb-1 border-bottom pb-1">
                <span>{pd.phenotype}</span>
                <span className="fw-bold" style={{ color: COLOR }}>{pd.count}</span>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Clinical Feature Prevalence" borderColor={COLOR2}>
            {feats.slice(0, 10).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={f.pct > 70 ? COLOR3 : f.pct > 40 ? COLOR2 : COLOR} />
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Molecular Features */}
      <SectionCard title="Key Molecular Features — MT-TL1 tRNA-Leu(UUR)" borderColor={COLOR5}>
        {mol_feats.map((mf, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f3e5f5' }}>
            <div className="fw-bold small" style={{ color: COLOR5 }}>{mf.feature}</div>
            <div className="text-muted" style={{ fontSize: '0.78rem' }}>{mf.significance}</div>
          </div>
        ))}
      </SectionCard>

      {/* Absolute CIs */}
      <SectionCard title="Absolute Contraindications (ALL MT-TL1 patients)" borderColor={COLOR3}>
        <ul className="mb-0 small">
          {(data.absolute_contraindications || []).map((ci, i) => (
            <li key={i} className="mb-1 text-danger fw-bold">{ci}</li>
          ))}
        </ul>
      </SectionCard>

      {/* Treatments */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Mandatory Acute Treatments" borderColor={COLOR}>
            <ul className="mb-0 small">
              {(data.mandatory_acute_treatments || []).map((t, i) => (
                <li key={i} className="mb-1">{t}</li>
              ))}
            </ul>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Maintenance Treatments (Level B/C)" borderColor={COLOR2}>
            <ul className="mb-0 small">
              {(data.maintenance_treatments || []).map((t, i) => (
                <li key={i} className="mb-1">{t}</li>
              ))}
            </ul>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

// ── Tab: MELAS vs MIDD vs KSS ─────────────────────────────────────────────────
function PhenotypeTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const vb = data.variant_breakdown || [];
  const pheno = data.phenotype_distribution || {};
  const hetBands = data.heteroplasmy_bands_urine || {};
  const hetBandsBlood = data.heteroplasmy_bands_blood || {};
  const ciBands = data.ci_activity_bands || {};
  const pts = data.patient_table || [];

  return (
    <div>
      {/* Variant Breakdown */}
      <SectionCard title="Variant Breakdown by MT-TL1 locus" borderColor={COLOR}>
        {vb.map((v, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ background: '#f1f8e9', borderLeft: `4px solid ${COLOR}` }}>
            <div className="fw-bold" style={{ color: COLOR }}>{v.variant}</div>
            <div className="small text-muted mb-1">{v.domain}</div>
            <div className="small mb-2">{v.severity}</div>
            <div className="row small">
              <div className="col-md-4">
                <b>n:</b> {v.n_patients} | <b>SLE:</b> {v.sle_pct}% | <b>SNHL:</b> {v.snhl_pct}%
              </div>
              <div className="col-md-4">
                <b>Diabetes:</b> {v.diabetes_pct}% | <b>LA:</b> {v.lactic_acidosis_pct}%
              </div>
              <div className="col-md-4">
                <b>Avg CI:</b> {v.avg_ci_activity_pct}% | <b>Urine heter:</b> {v.avg_heteroplasmy_urine_pct}%
              </div>
            </div>
            <div className="small text-muted mt-1" style={{ fontSize: '0.75rem' }}>{v.notes}</div>
          </div>
        ))}
      </SectionCard>

      {/* Heteroplasmy Bands */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Heteroplasmy Distribution — Urine (preferred)" borderColor={COLOR4}>
            {Object.entries(hetBands).map(([band, count]) => (
              <div key={band} className="d-flex justify-content-between small border-bottom pb-1 mb-1">
                <span className="fw-bold">{band}</span>
                <span>{count} patients</span>
              </div>
            ))}
            <div className="text-muted small mt-2">Urine epithelial cells: 20-30% higher than blood; preferred for accurate threshold.</div>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Heteroplasmy Distribution — Blood (underestimates)" borderColor={COLOR2}>
            {Object.entries(hetBandsBlood).map(([band, count]) => (
              <div key={band} className="d-flex justify-content-between small border-bottom pb-1 mb-1">
                <span className="fw-bold">{band}</span>
                <span>{count} patients</span>
              </div>
            ))}
            <div className="text-muted small mt-2">Blood underestimates due to clonal haematopoiesis. Never use blood alone for phenotype classification.</div>
          </SectionCard>
        </div>
      </div>

      {/* CI Activity Bands */}
      <SectionCard title="Complex I Activity Bands (pan-OXPHOS fingerprint)" borderColor={COLOR3}>
        <div className="row small">
          {Object.entries(ciBands).map(([band, count]) => (
            <div key={band} className="col-md-3 text-center mb-2">
              <div className="fw-bold fs-5" style={{ color: band === '<15%' ? COLOR3 : band === '15-30%' ? COLOR2 : COLOR }}>
                {count}
              </div>
              <div className="text-muted">{band} CI activity</div>
            </div>
          ))}
        </div>
        <div className="text-muted small mt-2">CII (SDH) NORMAL in all MT-TL1 patients — nuclear-encoded; pan-OXPHOS fingerprint = CI+CIII+CIV reduced with CII normal.</div>
      </SectionCard>

      {/* Patient Table */}
      <SectionCard title={`Patient Table — ${pts.length} patients (seed-783)`} borderColor={COLOR}>
        <div className="table-responsive" style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table className="table table-sm table-bordered table-hover">
            <thead style={{ backgroundColor: COLOR, color: 'white', position: 'sticky', top: 0 }}>
              <tr>
                <th>ID</th><th>Phenotype</th><th>Variant</th>
                <th>Urine Het%</th><th>Blood Het%</th><th>CI%</th>
                <th>Lactate</th><th>SLE</th><th>SNHL</th><th>DM</th>
              </tr>
            </thead>
            <tbody>
              {pts.map((p, i) => (
                <tr key={i} style={{ backgroundColor: p.sle ? '#ffebee' : 'white' }}>
                  <td className="small">{p.id}</td>
                  <td className="small" style={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.phenotype}</td>
                  <td className="small" style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.variant}</td>
                  <td className="small">{p.heteroplasmy_urine_pct}%</td>
                  <td className="small">{p.heteroplasmy_blood_pct}%</td>
                  <td className="small">{p.ci_pct}%</td>
                  <td className="small">{p.lactate}</td>
                  <td className="small">{p.sle ? '✓' : '—'}</td>
                  <td className="small">{p.snhl ? '✓' : '—'}</td>
                  <td className="small">{p.diabetes ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: DDx & SLE Management ─────────────────────��───────────────────────────
function TreatmentTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ddx = data.differential_diagnosis || [];
  const sleMgmt = data.sle_management_table || [];

  return (
    <div>
      {/* SLE Management */}
      <SectionCard title="SLE (Stroke-like Episode) Management Protocol" borderColor={COLOR3}>
        <div className="alert text-danger fw-bold mb-3" style={{ background: '#ffebee' }}>
          ⛔ IV tPA / Thrombolytics: ABSOLUTE CONTRAINDICATION — SLE are NOT thrombotic. MRA/MRV NORMAL.
          IV tPA → haemorrhagic transformation without benefit. FATAL in confirmed/suspected MT-TL1.
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead style={{ backgroundColor: COLOR3, color: 'white' }}>
              <tr>
                <th>Phase</th><th>Treatment</th><th>Evidence</th><th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {sleMgmt.map((row, i) => (
                <tr key={i} style={{ backgroundColor: row.evidence === 'Absolute CI' ? '#ffebee' : row.evidence.includes('Level B') ? '#e8f5e9' : 'white' }}>
                  <td className="small fw-bold">{row.phase}</td>
                  <td className="small" style={{ color: row.evidence === 'Absolute CI' ? COLOR3 : 'inherit', fontWeight: row.evidence === 'Absolute CI' ? 'bold' : 'normal' }}>{row.treatment}</td>
                  <td className="small fw-bold" style={{ color: row.evidence === 'Absolute CI' ? COLOR3 : row.evidence.includes('Level B') ? '#2e7d32' : 'inherit' }}>{row.evidence}</td>
                  <td className="small">{row.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Differential Diagnosis */}
      <SectionCard title="Differential Diagnosis — Key Distinguishing Features" borderColor={COLOR}>
        {ddx.map((d, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: '#e0f2f1', borderLeft: `4px solid ${COLOR}` }}>
            <div className="fw-bold small" style={{ color: COLOR }}>{d.entity}</div>
            <div className="small mt-1">{d.distinguishing_feature}</div>
            <div className="small text-muted mt-1"><b>Key test:</b> {d.key_test}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const aci = data.absolute_contraindications || {};
  const tx = data.recommended_treatments || {};
  const mon = data.specialist_monitoring || {};
  const refs = data.key_references || [];

  const defFields = [
    ['Gene', 'full_name'],
    ['OMIM Gene', 'omim_gene'],
    ['tRNA Type', 'trna_type'],
    ['Anticodon', 'anticodon'],
    ['rCRS Positions', 'rcrs_positions'],
    ['Strand', 'strand'],
    ['Pan-OXPHOS', 'pan_oxphos'],
    ['MELAS Definition', 'melas_definition'],
    ['SLE Definition', 'sle_definition'],
    ['MIDD Definition', 'midd_definition'],
    ['Pan-OXPHOS Mechanism', 'pan_oxphos_definition'],
    ['Urine Heteroplasmy', 'heteroplasmy_urine_definition'],
    ['L-Arginine Mechanism', 'arginine_mechanism'],
    ['IV tPA CI', 'tpa_ci_definition'],
    ['m.3243A>G', 'm3243ag_definition'],
    ['Ragged Red Fibres', 'ragged_red_fibres'],
    ['WES Coverage', 'wes_coverage'],
  ];

  return (
    <div>
      <SectionCard title="Gene & Disease Definitions" borderColor={COLOR}>
        {defFields.map(([label, key]) => data[key] ? (
          <div key={key} className="mb-2 pb-2 border-bottom">
            <div className="fw-bold small" style={{ color: COLOR }}>{label}</div>
            <div className="small">{typeof data[key] === 'string' ? data[key] : JSON.stringify(data[key])}</div>
          </div>
        ) : null)}
      </SectionCard>

      <SectionCard title="Absolute Contraindications" borderColor={COLOR3}>
        {Object.entries(aci).map(([drug, reason]) => (
          <div key={drug} className="mb-2 p-2 rounded" style={{ background: '#ffebee' }}>
            <div className="fw-bold small text-danger">{drug}</div>
            <div className="small">{reason}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Recommended Treatments" borderColor={COLOR2}>
        {Object.entries(tx).map(([drug, desc]) => (
          <div key={drug} className="mb-2 pb-1 border-bottom">
            <div className="fw-bold small" style={{ color: COLOR2 }}>{drug}</div>
            <div className="small">{desc}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Specialist Monitoring" borderColor={COLOR4}>
        {Object.entries(mon).map(([spec, detail]) => (
          <div key={spec} className="mb-2 pb-1 border-bottom">
            <div className="fw-bold small" style={{ color: COLOR4 }}>{spec}</div>
            <div className="small">{detail}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Key References" borderColor={COLOR}>
        <ol className="small mb-0">
          {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ol>
      </SectionCard>

      <SectionCard title="OMIM Diseases" borderColor={COLOR5}>
        {Object.entries(data.omim_diseases || {}).map(([k, v]) => (
          <div key={k} className="mb-2 pb-1 border-bottom">
            <div className="fw-bold small" style={{ color: COLOR5 }}>{k.toUpperCase()}</div>
            <div className="small">{v}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function MTTL1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mttl1/overview`).then(r => r.json()),
      fetch(`${API}/api/mttl1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mttl1/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div className="container-fluid py-4">
      {/* Header */}
      <div className="mb-4 p-3 rounded" style={{ background: `linear-gradient(135deg, ${COLOR} 0%, ${COLOR2} 100%)`, color: 'white' }}>
        <h3 className="mb-1 fw-bold">
          🟢 MT-TL1 — tRNA-Leu(UUR) — MELAS / MIDD / Pan-OXPHOS
        </h3>
        <div className="small opacity-90">
          MT-TL1 (OMIM *590050) · rCRS H-strand 3230–3304 · m.3243A&gt;G: most common pathogenic mtDNA variant (~1 in 400 adults) ·
          Pan-OXPHOS (CI+CIII+CIV; CII NORMAL) · Stroke-like episodes (SLE) NOT thrombotic · IV tPA ABSOLUTE CI ·
          IV L-Arginine Level B (acute SLE) · Metformin ABSOLUTE CI (all m.3243A&gt;G including MIDD) · 40-patient cohort seed-783
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* Tab Content */}
      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PhenotypeTab data={breakdown} />}
      {tab === 2 && <TreatmentTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
