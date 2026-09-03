'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Cohort', 'DDx & Management', 'Definitions'];
const COLOR  = '#4a148c';   // deep purple — tRNA-Asp / CPEO-dominant / H-strand adjacent MT-CO2
const LIGHT  = '#f3e5f5';
const COLOR2 = '#6a1b9a';   // medium purple — CPEO / myopathy
const COLOR3 = '#b71c1c';   // dark red — absolute CIs / contraindications
const COLOR4 = '#1b5e20';   // dark green — biochemical fingerprint / OXPHOS
const COLOR5 = '#0d47a1';   // deep blue — DARS2 DDx / LBSL white matter

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
  const s        = data.cohort_statistics || {};
  const feats    = data.cohort_summary_features || [];
  const pheno_dist = data.phenotype_distribution || [];
  const hmap     = data.heteroplasmy_clinical_map || [];
  const mol_feats = data.key_molecular_features || [];
  const alerts   = data.clinical_alerts || [];
  const gf       = data.gene_facts || {};

  return (
    <>
      {/* Gene facts banner */}
      <SectionCard title="Gene Facts — MT-TD tRNA-Asp" borderColor={COLOR}>
        <div className="row g-2 small">
          {Object.entries(gf).map(([k, v]) => (
            <div className="col-md-6" key={k}>
              <span className="fw-bold text-capitalize">{k.replace(/_/g,' ')}: </span>
              <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* DARS2 DDx note */}
      {data.dars2_ddx_note && (
        <div className="alert border-0 mb-4" style={{ backgroundColor: '#e3f2fd', borderLeft: `5px solid ${COLOR5}` }}>
          <strong style={{ color: COLOR5 }}>🧬 {data.dars2_ddx_note.note_class}</strong>
          <p className="mb-0 small mt-1">{data.dars2_ddx_note.detail}</p>
        </div>
      )}

      {/* MT-CO2 boundary note */}
      {data.mttd_co2_boundary_note && (
        <div className="alert border-0 mb-4" style={{ backgroundColor: '#fff3e0', borderLeft: `5px solid #e65100` }}>
          <strong style={{ color: '#e65100' }}>🔬 {data.mttd_co2_boundary_note.note_class}</strong>
          <p className="mb-0 small mt-1">{data.mttd_co2_boundary_note.detail}</p>
        </div>
      )}

      {/* KPI row */}
      <div className="row mb-2">
        <KPI label="Patients (n)" value={s.n_patients} />
        <KPI label="Avg Heteroplasmy" value={`${s.avg_heteroplasmy_blood_pct}%`} />
        <KPI label="CPEO" value={`${s.pct_cpeo}%`} color={COLOR} />
        <KPI label="Myopathy" value={`${s.pct_myopathy}%`} color={COLOR2} />
        <KPI label="Avg CI Activity" value={`${s.avg_ci_activity_pct_normal}%`} color={COLOR4} />
        <KPI label="Avg CIV Activity" value={`${s.avg_civ_activity_pct_normal}%`} color={COLOR4} />
      </div>

      {/* Biochemical fingerprint */}
      {data.biochemical_fingerprint && (
        <SectionCard title="Biochemical Fingerprint — Combined CI+CIV Deficiency (CII NORMAL)" borderColor={COLOR4}>
          <p className="small mb-2"><strong>{data.biochemical_fingerprint.summary}</strong></p>
          <div className="row g-2 small">
            <div className="col-md-4">
              <div className="p-2 rounded" style={{ backgroundColor: '#ffebee' }}>
                <strong style={{ color: COLOR3 }}>Complex I (CI)</strong>
                <div>{data.biochemical_fingerprint.complex_i}</div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="p-2 rounded" style={{ backgroundColor: '#e8f5e9' }}>
                <strong style={{ color: COLOR4 }}>Complex II (CII)</strong>
                <div>{data.biochemical_fingerprint.complex_ii}</div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="p-2 rounded" style={{ backgroundColor: '#ffebee' }}>
                <strong style={{ color: COLOR3 }}>Complex IV (CIV)</strong>
                <div>{data.biochemical_fingerprint.complex_iv}</div>
              </div>
            </div>
          </div>
          <p className="small text-muted mt-2 mb-0">{data.biochemical_fingerprint.mechanism}</p>
        </SectionCard>
      )}

      {/* Cohort features */}
      <SectionCard title="40-Patient Cohort — Clinical Features" borderColor={COLOR}>
        {feats.map((f, i) => (
          <div key={i} className="mb-1">
            <Bar label={`${f.feature} — ${f.note}`} value={parseFloat(f.value)} color={COLOR} />
          </div>
        ))}
      </SectionCard>

      {/* Phenotype distribution */}
      <SectionCard title="Variant Phenotype Distribution" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead><tr>
              <th>Variant</th><th>% Cohort</th><th>Phenotype</th><th>Position</th>
            </tr></thead>
            <tbody>
              {pheno_dist.map((r, i) => (
                <tr key={i}>
                  <td><code>{r.variant}</code></td>
                  <td><span className="badge" style={{ backgroundColor: COLOR }}>{r.pct}%</span></td>
                  <td>{r.phenotype}</td>
                  <td className="text-muted">{r.position}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Heteroplasmy-clinical map */}
      <SectionCard title="Heteroplasmy Threshold → Clinical Phenotype" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr><th>Heteroplasmy</th><th>Expected Phenotype</th></tr></thead>
            <tbody>
              {hmap.map((r, i) => (
                <tr key={i}>
                  <td><strong>{r.threshold_pct}</strong></td>
                  <td>{r.expected_phenotype}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Key molecular features */}
      <SectionCard title="Key Molecular Features" borderColor={COLOR4}>
        <ul className="list-unstyled small mb-0">
          {mol_feats.map((f, i) => <li key={i} className="mb-1">• {f}</li>)}
        </ul>
      </SectionCard>

      {/* Clinical alerts */}
      <SectionCard title="Clinical Alerts" borderColor={COLOR3}>
        {alerts.map((a, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ backgroundColor: '#fff3e0', borderLeft: `4px solid ${COLOR3}` }}>
            <strong className="small" style={{ color: COLOR3 }}>⚠ {a.alert}</strong>
            <p className="mb-0 small text-muted mt-1">{a.detail}</p>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Variants & Cohort ────────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const vb   = data.variant_breakdown || [];
  const mgmt = data.management_by_variant || [];
  const aci  = data.absolute_contraindications || [];
  const safe = data.safe_interventions || [];

  return (
    <>
      <SectionCard title="Variant Breakdown — 40-Patient Cohort (seed-825)" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead><tr>
              <th>Variant</th><th>n</th><th>% Cohort</th>
              <th>Avg Hetero</th><th>Avg CI%</th><th>Avg CIV%</th>
              <th>CPEO%</th><th>SNHL%</th><th>Myo%</th><th>Cardio%</th>
            </tr></thead>
            <tbody>
              {vb.map((r, i) => (
                <tr key={i}>
                  <td><code>{r.variant}</code></td>
                  <td>{r.n}</td>
                  <td><span className="badge" style={{ backgroundColor: COLOR }}>{r.pct_of_cohort}%</span></td>
                  <td>{r.avg_heteroplasmy}%</td>
                  <td style={{ color: COLOR3 }}>{r.avg_ci_pct_normal}%</td>
                  <td style={{ color: COLOR3 }}>{r.avg_civ_pct_normal}%</td>
                  <td>{r.pct_cpeo}%</td>
                  <td>{r.pct_snhl}%</td>
                  <td>{r.pct_myopathy}%</td>
                  <td>{r.pct_cardiomyopathy}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <div className="row g-3">
        <div className="col-md-6">
          <SectionCard title="Absolute Contraindications" borderColor={COLOR3}>
            {aci.map((d, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ backgroundColor: '#ffebee' }}>
                <strong className="small" style={{ color: COLOR3 }}>{d.drug}</strong>
                <p className="mb-0 small text-muted">{d.reason}</p>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Safe Interventions" borderColor={COLOR4}>
            {safe.map((d, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ backgroundColor: '#e8f5e9' }}>
                <strong className="small" style={{ color: COLOR4 }}>{d.intervention}</strong>
                <p className="mb-0 small text-muted">{d.evidence}</p>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Management by Variant" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead><tr>
              <th>Variant</th><th>CPEO Risk</th><th>Cardio Risk</th><th>SNHL Risk</th><th>Key Action</th>
            </tr></thead>
            <tbody>
              {mgmt.map((r, i) => (
                <tr key={i}>
                  <td><code>{r.variant}</code></td>
                  <td><span className="badge" style={{ backgroundColor: r.cpeo_risk === 'High' ? COLOR3 : r.cpeo_risk === 'Moderate' ? '#e65100' : COLOR4 }}>{r.cpeo_risk}</span></td>
                  <td><span className="badge" style={{ backgroundColor: r.cardio_risk === 'High' ? COLOR3 : r.cardio_risk === 'Moderate' ? '#e65100' : COLOR4 }}>{r.cardio_risk}</span></td>
                  <td><span className="badge" style={{ backgroundColor: r.snhl_risk === 'High' ? COLOR3 : r.snhl_risk === 'Moderate' ? '#e65100' : COLOR }}>{r.snhl_risk}</span></td>
                  <td className="small">{r.key_action}</td>
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
function DDxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ddx = data.ddx_table || [];

  return (
    <>
      <SectionCard title="Differential Diagnosis Table" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: LIGHT }}>
              <tr>
                <th>Entity</th><th>Inheritance</th><th>Phenotype</th>
                <th>Biochemistry</th><th>NGS Strategy</th><th>Distinctive Feature</th>
              </tr>
            </thead>
            <tbody>
              {ddx.map((r, i) => (
                <tr key={i} style={r.entity && r.entity.startsWith('MT-TD') ? { backgroundColor: LIGHT, fontWeight: 600 } : {}}>
                  <td>{r.entity}</td>
                  <td>{r.inheritance}</td>
                  <td>{r.phenotype}</td>
                  <td>{r.biochemistry}</td>
                  <td>{r.ngs}</td>
                  <td>{r.distinctive}</td>
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
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  const sections = [
    { key: 'gene_definitions',       title: 'Gene & Genomic Definitions',        color: COLOR  },
    { key: 'biochemical_definitions', title: 'Biochemical Definitions',           color: COLOR4 },
    { key: 'clinical_definitions',   title: 'Clinical Definitions',              color: COLOR2 },
    { key: 'ngs_definitions',        title: 'NGS / Variant-Calling Definitions', color: COLOR5 },
    { key: 'drug_definitions',       title: 'Drug / Treatment Definitions',      color: COLOR3 },
    { key: 'references',             title: 'Key References',                    color: COLOR  },
  ];

  return (
    <>
      {sections.map(({ key, title, color }) => {
        const rows = data[key];
        if (!rows || !rows.length) return null;
        const isRef = key === 'references';
        return (
          <SectionCard key={key} title={title} borderColor={color}>
            <div className="table-responsive">
              <table className="table table-sm table-hover small mb-0">
                <thead><tr>
                  <th style={{ width: '30%' }}>{isRef ? 'Reference' : 'Term'}</th>
                  <th>{isRef ? 'Full Citation' : 'Definition'}</th>
                </tr></thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i}>
                      <td><strong>{r.term || r.ref}</strong></td>
                      <td className="text-muted">{r.definition || r.citation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        );
      })}
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MTTDPage() {
  const [tab, setTab]     = useState(0);
  const [over, setOver]   = useState(null);
  const [brkd, setBrkd]   = useState(null);
  const [defs, setDefs]   = useState(null);
  const [err,  setErr]    = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mttd/overview`).then(r => r.json()),
      fetch(`${API}/api/mttd/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mttd/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOver(o); setBrkd(b); setDefs(d); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div className="container-fluid py-4">
      {/* Header */}
      <div className="mb-4 p-3 rounded" style={{ background: `linear-gradient(135deg, ${COLOR} 0%, ${COLOR2} 100%)`, color: '#fff' }}>
        <h4 className="fw-bold mb-1">🧬 MT-TD — tRNA-Asp (GUC Anticodon)</h4>
        <p className="mb-1 small opacity-75">
          CPEO-Dominant · Combined CI+CIV Deficiency · MT-CO2 Shared Boundary — THIRTEENTH tRNA — rCRS 7518–7585
        </p>
        <div className="d-flex gap-3 flex-wrap small">
          <span className="badge bg-light text-dark">OMIM *590015</span>
          <span className="badge bg-light text-dark">H-strand rCRS 7518–7585</span>
          <span className="badge bg-light text-dark">40-patient cohort seed-825</span>
          <span className="badge bg-light text-dark">0nt gap — MT-CO2 adjacency</span>
          <span className="badge bg-light text-dark">DARS2 DDx — LBSL White Matter</span>
        </div>
      </div>

      {err && <div className="alert alert-danger">Error: {err}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { borderBottom: `3px solid ${COLOR}`, color: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab   data={over} />}
      {tab === 1 && <VariantsTab   data={brkd} />}
      {tab === 2 && <DDxTab        data={brkd} />}
      {tab === 3 && <DefinitionsTab data={defs} />}
    </div>
  );
}
