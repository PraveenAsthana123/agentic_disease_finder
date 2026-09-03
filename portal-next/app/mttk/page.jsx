'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'MERRF Variants & Cohort', 'DDx & Management', 'Definitions'];
const COLOR  = '#4a148c';   // deep purple — tRNA / MERRF (mt-translation gene)
const LIGHT  = '#f3e5f5';
const COLOR2 = '#7b1fa2';   // purple — myoclonic epilepsy
const COLOR3 = '#b71c1c';   // dark red — VPA CI / severe
const COLOR4 = '#1565c0';   // deep blue — metabolic / CI+CIV
const COLOR5 = '#2e7d32';   // green — MSL / lipomatosis

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
        🟣 MT-TK: tRNA-LYS — <strong>m.8344A&gt;G: MOST COMMON MERRF mutation (~80% of MERRF worldwide)</strong>.
        Pan-OXPHOS deficiency (CI+CIV predominantly; CII NORMAL = mt-translation fingerprint).
        <span className="text-danger"> VPA ABSOLUTE CI</span> — use LEV for myoclonic seizures.
        NO Stroke-like Episodes (distinguishes MERRF from MELAS). MSL (Multiple Symmetrical Lipomatosis) PATHOGNOMONIC.
      </div>

      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Patients" value={data.n_patients} color={COLOR} />
        <KPI label="Avg CI Activity" value={`${s.avg_ci_activity_pct}%`} color={COLOR3} />
        <KPI label="Avg CIV Activity" value={`${s.avg_civ_activity_pct}%`} color={COLOR4} />
        <KPI label="Avg Lactate" value={`${s.avg_lactic_acid_mmolL} mmol/L`} color={COLOR2} />
        <KPI label="Myoclonus" value={`${s.myoclonus_pct}%`} color={COLOR} />
        <KPI label="Cerebellar Ataxia" value={`${s.cerebellar_ataxia_pct}%`} color={COLOR2} />
        <KPI label="Ragged Red Fibres" value={`${s.ragged_red_fibres_pct}%`} color={COLOR} />
        <KPI label="MSL (Lipomatosis)" value={`${s.msl_pct}%`} color={COLOR5} />
        <KPI label="SLE (Stroke-like)" value={`${s.stroke_like_episode_pct}% ✓ absent`} color="#2e7d32" />
      </div>

      {/* Clinical Alerts */}
      <SectionCard title="⚠️ Critical Clinical Alerts" borderColor={COLOR3}>
        {alerts.map((a, i) => (
          <div key={i} className="mb-2 p-2 rounded small" style={{ background: i < 2 ? '#ffebee' : i < 5 ? '#fff8e1' : '#e8f5e9' }}>
            {a}
          </div>
        ))}
      </SectionCard>

      {/* Heteroplasmy-Phenotype Map */}
      <SectionCard title="Heteroplasmy → Clinical Phenotype Map (m.8344A>G — muscle preferred over blood)" borderColor={COLOR4}>
        <div className="small text-muted mb-2">Blood underestimates by ~10-15%. Muscle biopsy preferred in equivocal blood cases.</div>
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead style={{ backgroundColor: COLOR4, color: 'white' }}>
              <tr>
                <th>Heteroplasmy Tier</th>
                <th>Clinical Phenotype</th>
                <th>MERRF Severity</th>
                <th>VPA CI?</th>
              </tr>
            </thead>
            <tbody>
              {hmap.map((row, i) => (
                <tr key={i} style={{ backgroundColor: i === 3 ? '#ffebee' : i === 2 ? '#fff3e0' : i === 1 ? '#fff8e1' : '#f3e5f5' }}>
                  <td className="fw-bold small">{row.tier}</td>
                  <td className="small">{row.phenotype}</td>
                  <td className="small">{row.merrf_severity}</td>
                  <td className="fw-bold small text-danger">{row.vpa_ci}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Phenotype Distribution + Features */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Phenotype Distribution (40 patients, seed-787)" borderColor={COLOR}>
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
            {feats.slice(0, 12).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={f.feature.includes('SLE') ? '#2e7d32' : f.pct > 70 ? COLOR : f.pct > 40 ? COLOR2 : COLOR4} />
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Molecular Features */}
      <SectionCard title="Key Molecular Features — MT-TK tRNA-Lys" borderColor={COLOR4}>
        {mol_feats.map((mf, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f3e5f5' }}>
            <div className="fw-bold small" style={{ color: COLOR }}>{mf.feature}</div>
            <div className="text-muted" style={{ fontSize: '0.78rem' }}>{mf.significance}</div>
          </div>
        ))}
      </SectionCard>

      {/* Absolute CIs */}
      <SectionCard title="Absolute Contraindications (ALL MT-TK patients)" borderColor={COLOR3}>
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

// ── Tab: MERRF Variants & Cohort ──────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const vb = data.variant_breakdown || [];
  const hetBandsBlood = data.heteroplasmy_bands_blood || {};
  const hetBandsMuscle = data.heteroplasmy_bands_muscle || {};
  const ciBands = data.ci_activity_bands || {};
  const pts = data.patient_table || [];

  return (
    <div>
      {/* Variant Breakdown */}
      <SectionCard title="Variant Breakdown by MT-TK locus" borderColor={COLOR}>
        {vb.map((v, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ background: '#f3e5f5', borderLeft: `4px solid ${COLOR}` }}>
            <div className="fw-bold" style={{ color: COLOR }}>{v.variant}</div>
            <div className="small text-muted mb-1">{v.domain}</div>
            <div className="small mb-2">{v.severity}</div>
            <div className="row small">
              <div className="col-md-4">
                <b>n:</b> {v.n_patients} | <b>Myoclonus:</b> {v.myoclonus_pct}% | <b>Ataxia:</b> {v.ataxia_pct}%
              </div>
              <div className="col-md-4">
                <b>RRF:</b> {v.rrf_pct}% | <b>MSL:</b> {v.msl_pct}% | <b>SNHL:</b> {v.snhl_pct}%
              </div>
              <div className="col-md-4">
                <b>Avg CI:</b> {v.avg_ci_activity_pct}% | <b>Avg CIV:</b> {v.avg_civ_activity_pct}%
              </div>
            </div>
            <div className="small text-muted mt-1" style={{ fontSize: '0.75rem' }}>{v.notes}</div>
          </div>
        ))}
      </SectionCard>

      {/* Heteroplasmy Bands */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Heteroplasmy Distribution — Blood (underestimates ~10-15%)" borderColor={COLOR4}>
            {Object.entries(hetBandsBlood).map(([band, count]) => (
              <div key={band} className="d-flex justify-content-between small border-bottom pb-1 mb-1">
                <span className="fw-bold">{band}</span>
                <span>{count} patients</span>
              </div>
            ))}
            <div className="text-muted small mt-2">Blood underestimates due to clonal haematopoiesis. Use muscle in equivocal cases.</div>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Heteroplasmy Distribution — Muscle (preferred for threshold)" borderColor={COLOR}>
            {Object.entries(hetBandsMuscle).map(([band, count]) => (
              <div key={band} className="d-flex justify-content-between small border-bottom pb-1 mb-1">
                <span className="fw-bold">{band}</span>
                <span>{count} patients</span>
              </div>
            ))}
            <div className="text-muted small mt-2">Post-mitotic muscle retains original heteroplasmy; ~10-15% higher than blood in MERRF.</div>
          </SectionCard>
        </div>
      </div>

      {/* CI Activity Bands */}
      <SectionCard title="Complex I Activity Bands (CI+CIV pan-OXPHOS; CII NORMAL)" borderColor={COLOR3}>
        <div className="row small">
          {Object.entries(ciBands).map(([band, count]) => (
            <div key={band} className="col-md-3 text-center mb-2">
              <div className="fw-bold fs-5" style={{ color: band === '<20%' ? COLOR3 : band === '20-40%' ? COLOR2 : COLOR }}>
                {count}
              </div>
              <div className="text-muted">{band} CI activity</div>
            </div>
          ))}
        </div>
        <div className="text-muted small mt-2">CII (SDH) NORMAL in all MT-TK patients — nuclear-encoded; CI+CIV pan-OXPHOS fingerprint differs from isolated CI (MT-ND1-6) or isolated CIV (SURF1/SCO2).</div>
      </SectionCard>

      {/* Patient Table */}
      <SectionCard title={`Patient Table — ${pts.length} patients (seed-787)`} borderColor={COLOR}>
        <div className="table-responsive" style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table className="table table-sm table-bordered table-hover">
            <thead style={{ backgroundColor: COLOR, color: 'white', position: 'sticky', top: 0 }}>
              <tr>
                <th>ID</th><th>Phenotype</th><th>Variant</th>
                <th>Blood Het%</th><th>Muscle Het%</th><th>CI%</th><th>CIV%</th>
                <th>Lactate</th><th>Myoclonus</th><th>Ataxia</th><th>RRF</th><th>MSL</th>
              </tr>
            </thead>
            <tbody>
              {pts.map((p, i) => (
                <tr key={i} style={{ backgroundColor: p.msl ? '#f3e5f5' : 'white' }}>
                  <td className="small">{p.id}</td>
                  <td className="small" style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.phenotype}</td>
                  <td className="small" style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.variant}</td>
                  <td className="small">{p.heteroplasmy_blood_pct}%</td>
                  <td className="small">{p.heteroplasmy_muscle_pct}%</td>
                  <td className="small">{p.ci_pct}%</td>
                  <td className="small">{p.civ_pct}%</td>
                  <td className="small">{p.lactate}</td>
                  <td className="small">{p.myoclonus ? '✓' : '—'}</td>
                  <td className="small">{p.ataxia ? '✓' : '—'}</td>
                  <td className="small">{p.rrf ? '✓' : '—'}</td>
                  <td className="small" style={{ color: p.msl ? COLOR5 : 'inherit', fontWeight: p.msl ? 'bold' : 'normal' }}>{p.msl ? 'MSL✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: DDx & Management ─────────────────────────────────────────────────────
function TreatmentTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ddx = data.differential_diagnosis || [];
  const mgmt = data.merrf_management_table || [];

  return (
    <div>
      {/* Management */}
      <SectionCard title="MERRF Management Protocol" borderColor={COLOR3}>
        <div className="alert text-danger fw-bold mb-3" style={{ background: '#ffebee' }}>
          ⛔ VPA / Valproic Acid: ABSOLUTE CONTRAINDICATION — mt-ribosome inhibition + CoA sequestration → acute mt crisis.
          ALWAYS use LEV for myoclonic seizures in MERRF / MT-TK. NEVER prescribe VPA in any tRNA-Lys carrier.
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead style={{ backgroundColor: COLOR, color: 'white' }}>
              <tr>
                <th>Phase</th><th>Treatment</th><th>Evidence</th><th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {mgmt.map((row, i) => (
                <tr key={i} style={{ backgroundColor: row.evidence === 'Absolute CI' ? '#ffebee' : row.evidence.includes('Level C') ? '#f3e5f5' : 'white' }}>
                  <td className="small fw-bold">{row.phase}</td>
                  <td className="small" style={{ color: row.evidence === 'Absolute CI' ? COLOR3 : 'inherit', fontWeight: row.evidence === 'Absolute CI' ? 'bold' : 'normal' }}>{row.treatment}</td>
                  <td className="small fw-bold" style={{ color: row.evidence === 'Absolute CI' ? COLOR3 : row.evidence.includes('Level C') ? COLOR : '#2e7d32' }}>{row.evidence}</td>
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
          <div key={i} className="mb-3 p-3 rounded" style={{ background: '#f3e5f5', borderLeft: `4px solid ${COLOR}` }}>
            <div className="fw-bold small" style={{ color: COLOR }}>{d.entity}</div>
            <div className="small mt-1">{d.distinguishing_feature}</div>
            <div className="small text-muted mt-1"><b>Key test:</b> {d.key_test}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const aci = data.absolute_contraindications || {};
  const tx = data.recommended_treatments || {};
  const mon = data.specialist_monitoring || {};
  const refs = data.key_references || [];

  const defFields = [
    ['Gene', 'full_name'],
    ['OMIM Gene', 'omim_gene'],
    ['tRNA Type', 'protein_name'],
    ['Anticodon', 'anticodon'],
    ['rCRS Positions', 'rcrs_positions'],
    ['Strand', 'strand'],
    ['MERRF Definition', 'merrf_definition'],
    ['MSL Definition', 'msl_definition'],
    ['Pan-OXPHOS (CI+CIV)', 'pan_oxphos_definition'],
    ['Muscle Heteroplasmy', 'heteroplasmy_muscle_definition'],
    ['Progressive Myoclonic Epilepsy', 'progressive_myoclonic_epilepsy_definition'],
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

      <SectionCard title="Key Variants" borderColor={COLOR}>
        {Object.entries(data.key_variants || {}).map(([k, v]) => (
          <div key={k} className="mb-2 pb-1 border-bottom">
            <div className="fw-bold small" style={{ color: COLOR }}>{k}</div>
            <div className="small">{v}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function MTTKPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mttk/overview`).then(r => r.json()),
      fetch(`${API}/api/mttk/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mttk/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div className="container-fluid py-4">
      {/* Header */}
      <div className="mb-4 p-3 rounded" style={{ background: `linear-gradient(135deg, ${COLOR} 0%, ${COLOR2} 100%)`, color: 'white' }}>
        <h3 className="mb-1 fw-bold">
          🟣 MT-TK — tRNA-Lys — MERRF / MSL / Pan-OXPHOS (CI+CIV)
        </h3>
        <div className="small opacity-90">
          MT-TK (OMIM *590060) · rCRS H-strand 8295–8364 · m.8344A&gt;G: most common MERRF mutation (~80% worldwide) ·
          Pan-OXPHOS (CI+CIV; CII NORMAL = mt-translation fingerprint) · Myoclonic epilepsy + Cerebellar ataxia + RRF ·
          MSL (Madelung's) PATHOGNOMONIC · NO Stroke-like Episodes (unlike MELAS/MT-TL1) ·
          VPA ABSOLUTE CI — use LEV · Metformin ABSOLUTE CI · 40-patient cohort seed-787
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
      {tab === 1 && <VariantsTab data={breakdown} />}
      {tab === 2 && <TreatmentTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
