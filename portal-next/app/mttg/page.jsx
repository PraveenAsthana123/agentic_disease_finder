'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Cohort', 'DDx & Management', 'Definitions'];
const COLOR  = '#00695c';   // dark teal — tRNA-Gly / CPEO-dominant / H-strand sandwiched CO3-ND3
const LIGHT  = '#e0f2f1';
const COLOR2 = '#00796b';   // medium teal — CPEO / myopathy
const COLOR3 = '#b71c1c';   // dark red — absolute CIs / contraindications
const COLOR4 = '#1b5e20';   // dark green — biochemical fingerprint / OXPHOS
const COLOR5 = '#0d47a1';   // deep blue — GARS2 DDx / nuclear

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
  const pheno    = data.phenotype_distribution || [];
  const hmap     = data.heteroplasmy_clinical_map || [];
  const mol      = data.key_molecular_features || [];
  const alerts   = data.clinical_alerts || [];
  const gf       = data.gene_facts || {};
  const biochem  = data.biochemical_fingerprint || {};
  const gars2    = data.gars2_ddx_note || {};
  const boundary = data.mttg_co3_nd3_boundary_note || {};

  return (
    <>
      {/* Gene identity banner */}
      <SectionCard title="MT-TG — tRNA-Gly (GCC Anticodon) | 15th mt-tRNA | H-strand rCRS 9991–10058 | OMIM *590035">
        <div className="row g-2 small">
          {Object.entries(gf).map(([k, v]) => (
            <div key={k} className="col-12 col-md-6">
              <span className="fw-semibold text-capitalize">{k.replace(/_/g, ' ')}: </span>
              <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Patients (cohort)" value={s.n_patients}                        color={COLOR} />
        <KPI label="Avg Heteroplasmy" value={`${s.avg_heteroplasmy_blood_pct}%`}   color={COLOR} />
        <KPI label="Avg CI Activity"  value={`${s.avg_ci_activity_pct_normal}%`}   color={COLOR2} />
        <KPI label="Avg CIV Activity" value={`${s.avg_civ_activity_pct_normal}%`}  color={COLOR2} />
        <KPI label="CPEO"             value={`${s.pct_cpeo}%`}                     color={COLOR} />
        <KPI label="Myopathy"         value={`${s.pct_myopathy}%`}                 color={COLOR2} />
        <KPI label="SNHL"             value={`${s.pct_snhl}%`}                     color={COLOR} />
        <KPI label="Cardiomyopathy"   value={`${s.pct_cardiomyopathy}%`}           color={COLOR3} />
      </div>

      {/* Biochemical fingerprint */}
      <SectionCard title="Biochemical Fingerprint — Combined CI+CIV Deficiency (CII NORMAL)" borderColor={COLOR4}>
        <p className="text-muted small">{biochem.summary}</p>
        <div className="row g-2 small">
          {['complex_i', 'complex_ii', 'complex_iv', 'mechanism'].map(k => biochem[k] && (
            <div key={k} className="col-12 col-md-6">
              <span className="fw-semibold text-capitalize">{k.replace(/_/g, ' ')}: </span>
              <span className="text-muted">{biochem[k]}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* CO3/ND3 boundary note */}
      <SectionCard title={boundary.note_class} borderColor={COLOR2}>
        <p className="small text-muted mb-0">{boundary.detail}</p>
      </SectionCard>

      {/* GARS2 DDx */}
      <SectionCard title={gars2.note_class} borderColor={COLOR5}>
        <p className="small text-muted mb-0">{gars2.detail}</p>
      </SectionCard>

      {/* Phenotype bars */}
      <SectionCard title="Cohort Feature Prevalence">
        {feats.map(f => (
          <div key={f.feature} className="mb-3">
            <Bar label={f.feature} value={parseInt(f.value) || 0} />
            <div className="text-muted small ms-1">{f.note}</div>
          </div>
        ))}
      </SectionCard>

      {/* Variant distribution */}
      <SectionCard title="Variant Phenotype Distribution">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr>
              <th>Variant</th><th>Cohort %</th><th>Phenotype</th><th>tRNA Position</th>
            </tr></thead>
            <tbody>
              {pheno.map(r => (
                <tr key={r.variant}>
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

      {/* Heteroplasmy map */}
      <SectionCard title="Heteroplasmy Threshold → Clinical Phenotype Map">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Blood Heteroplasmy</th><th>Expected Phenotype</th></tr></thead>
            <tbody>
              {hmap.map(r => (
                <tr key={r.threshold_pct}>
                  <td className="fw-semibold">{r.threshold_pct}</td>
                  <td className="text-muted">{r.expected_phenotype}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Molecular features */}
      <SectionCard title="Key Molecular & Genomic Features">
        <ul className="small text-muted mb-0">
          {mol.map((m, i) => <li key={i}>{m}</li>)}
        </ul>
      </SectionCard>

      {/* Clinical alerts */}
      <SectionCard title="Clinical Alerts & Absolute Contraindications" borderColor={COLOR3}>
        {alerts.map(a => (
          <div key={a.alert} className="mb-3 p-2 rounded" style={{ backgroundColor: '#ffebee', border: `1px solid ${COLOR3}` }}>
            <div className="fw-bold small" style={{ color: COLOR3 }}>⚠ {a.alert}</div>
            <div className="text-muted small mt-1">{a.detail}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Variants & Cohort ────────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const vb   = data.variant_breakdown    || [];
  const pt   = data.per_patient_table    || [];
  const tx   = data.treatment_protocol   || {};

  return (
    <>
      <SectionCard title="Per-Variant Breakdown (40-patient cohort, seed-827)">
        {vb.map(v => (
          <div key={v.variant} className="mb-4 p-3 rounded" style={{ background: LIGHT }}>
            <div className="d-flex justify-content-between align-items-center mb-2">
              <h6 className="mb-0 fw-bold" style={{ color: COLOR }}><code>{v.variant}</code> — {v.structural_position}</h6>
              <span className="badge" style={{ backgroundColor: COLOR }}>{v.pct_of_cohort}% · n={v.n}</span>
            </div>
            <div className="row g-2 small mb-2">
              <div className="col-6 col-md-3"><span className="fw-semibold">Avg Heteroplasmy:</span> {v.avg_heteroplasmy}%</div>
              <div className="col-6 col-md-3"><span className="fw-semibold">Avg CI:</span> {v.avg_ci_pct_normal}%</div>
              <div className="col-6 col-md-3"><span className="fw-semibold">Avg CIV:</span> {v.avg_civ_pct_normal}%</div>
              <div className="col-6 col-md-3"><span className="fw-semibold">CPEO:</span> {v.pct_cpeo}% · Myo: {v.pct_myopathy}%</div>
              <div className="col-6 col-md-3"><span className="fw-semibold">SNHL:</span> {v.pct_snhl}%</div>
              <div className="col-6 col-md-3"><span className="fw-semibold">Cardio:</span> {v.pct_cardiomyopathy}%</div>
            </div>
            <p className="small text-muted mb-0">{v.note}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Treatment Protocol">
        <div className="mb-3">
          <div className="fw-bold small mb-1" style={{ color: '#f57c00' }}>Mandatory Exclusion First:</div>
          <p className="small text-muted">{tx.mandatory_exclusion}</p>
        </div>
        <div className="mb-3">
          <div className="fw-bold small mb-1" style={{ color: COLOR3 }}>Absolute Contraindications:</div>
          <ul className="small text-muted mb-0">{(tx.absolute_ci || []).map((x, i) => <li key={i}>{x}</li>)}</ul>
        </div>
        <div className="mb-3">
          <div className="fw-bold small mb-1">Empiric Supplements (Level C):</div>
          <ul className="small text-muted mb-0">{(tx.empiric_supplements || []).map((x, i) => <li key={i}>{x}</li>)}</ul>
        </div>
        <div className="mb-3">
          <div className="fw-bold small mb-1">Monitoring Schedule:</div>
          <ul className="small text-muted mb-0">{(tx.monitoring || []).map((x, i) => <li key={i}>{x}</li>)}</ul>
        </div>
        <div className="mb-2">
          <span className="fw-semibold small">Preferred AED: </span>
          <span className="small text-muted">{tx.preferred_aed}</span>
        </div>
        <div className="mb-2">
          <span className="fw-semibold small">Cardiac: </span>
          <span className="small text-muted">{tx.cardiac}</span>
        </div>
        <div>
          <span className="fw-semibold small">Crisis management: </span>
          <span className="small text-muted">{tx.crisis_management}</span>
        </div>
      </SectionCard>

      <SectionCard title="Per-Patient Table (40 patients)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-light">
              <tr>
                <th>PID</th><th>Variant</th><th>Hetero%</th><th>CI%</th><th>CIV%</th>
                <th>CPEO</th><th>SNHL</th><th>Myo</th><th>Cardio</th><th>KSS</th>
              </tr>
            </thead>
            <tbody>
              {pt.map(p => (
                <tr key={p.pid}>
                  <td>{p.pid}</td>
                  <td><code className="small">{p.variant.length > 20 ? p.variant.slice(0, 20) + '…' : p.variant}</code></td>
                  <td>{p.heteroplasmy_pct}</td>
                  <td>{p.ci_pct_normal}</td>
                  <td>{p.civ_pct_normal}</td>
                  <td>{p.cpeo ? '✓' : '–'}</td>
                  <td>{p.snhl ? '✓' : '–'}</td>
                  <td>{p.myopathy ? '✓' : '–'}</td>
                  <td>{p.cardiomyopathy ? '✓' : '–'}</td>
                  <td>{p.kss_large_del ? <span style={{ color: COLOR3 }}>KSS</span> : '–'}</td>
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
      <SectionCard title="Differential Diagnosis Table">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead className="table-light">
              <tr>
                <th>Entity</th><th>Inheritance</th><th>Phenotype</th><th>OXPHOS</th><th>DDx Clue</th><th>WES</th>
              </tr>
            </thead>
            <tbody>
              {ddx.map(r => (
                <tr key={r.entity}>
                  <td className="fw-semibold">{r.entity}</td>
                  <td>{r.inheritance}</td>
                  <td className="text-muted">{r.phenotype}</td>
                  <td className="text-muted">{r.oxphos}</td>
                  <td style={{ color: COLOR }}>{r.ddx_clue}</td>
                  <td>{r.wes.startsWith('MISSED') ? <span style={{ color: COLOR3 }}>{r.wes}</span> : <span style={{ color: '#1b5e20' }}>{r.wes}</span>}</td>
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
    { key: 'gene_definitions',       label: 'Gene & Genomic Definitions' },
    { key: 'biochemical_definitions', label: 'Biochemical Definitions' },
    { key: 'clinical_definitions',    label: 'Clinical Definitions' },
    { key: 'ngs_definitions',         label: 'NGS & Heteroplasmy Definitions' },
    { key: 'drug_definitions',        label: 'Drug / Contraindication Definitions' },
    { key: 'references',              label: 'References' },
  ];
  return (
    <>
      {sections.map(sec => {
        const items = data[sec.key] || [];
        if (!items.length) return null;
        return (
          <SectionCard key={sec.key} title={sec.label}>
            <dl className="row small mb-0">
              {items.map(item => (
                <div key={item.term || item.ref} className="col-12 mb-2">
                  <dt className="fw-semibold" style={{ color: COLOR }}>{item.term || item.ref}</dt>
                  <dd className="text-muted ms-3 mb-0">{item.definition || item.citation}</dd>
                </div>
              ))}
            </dl>
          </SectionCard>
        );
      })}
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MttgPage() {
  const [tab,     setTab]     = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs,    setDefs]    = useState(null);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    fetch(`${API}/api/mttg/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2) {
      fetch(`${API}/api/mttg/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 3) {
      fetch(`${API}/api/mttg/definitions`)
        .then(r => r.json()).then(setDefs).catch(() => {});
    }
  }, [tab]);

  return (
    <div>
      <div className="py-3 mb-4 rounded" style={{ background: `linear-gradient(135deg, ${COLOR} 0%, ${COLOR2} 100%)`, color: '#fff' }}>
        <div className="container-fluid px-4">
          <h4 className="mb-1 fw-bold">🧬 MT-TG — tRNA-Gly (GCC Anticodon)</h4>
          <p className="mb-1 small opacity-75">
            Combined CI+CIV Deficiency · CPEO + Myopathy · 15th mt-tRNA · H-strand rCRS 9991–10058 ·
            0nt gap MT-CO3 (5') + MT-ND3 (3') · GARS2 Nuclear DDx · KSS (common 4977bp deletion) ·
            OMIM *590035 · 40-patient cohort seed-827
          </p>
        </div>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active' : ''}`}
              style={tab === i ? { borderColor: COLOR, borderBottomColor: '#fff', color: COLOR, fontWeight: 600 } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab   data={overview} />}
      {tab === 1 && <VariantsTab   data={breakdown} />}
      {tab === 2 && <DdxTab        data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={defs} />}
    </div>
  );
}
