'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Cohort', 'DDx & Management', 'Definitions'];
const COLOR  = '#880e4f';   // deep magenta-crimson — tRNA-Ile / m.4300AG isolated HCM
const LIGHT  = '#fce4ec';
const COLOR2 = '#ad1457';   // medium crimson — CPEO / myopathy
const COLOR3 = '#b71c1c';   // dark red — absolute CIs / severe
const COLOR4 = '#1b5e20';   // dark green — biochemical fingerprint / OXPHOS
const COLOR5 = '#1a237e';   // deep indigo — IARS2 DDx / cardiac alerts

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

      {/* KPIs */}
      <SectionCard title="40-Patient Cohort — Key Statistics (seed-807)" borderColor={COLOR2}>
        <div className="row g-2">
          <KPI label="Patients" value={s.n_patients} />
          <KPI label="Avg Blood Heteroplasmy" value={`${s.avg_heteroplasmy_blood_pct}%`} />
          <KPI label="Avg CI Activity" value={`${s.avg_ci_activity_pct_normal}%`} color={COLOR4} />
          <KPI label="Avg CIV Activity" value={`${s.avg_civ_activity_pct_normal}%`} color={COLOR4} />
          <KPI label="CII (Nuclear — NORMAL)" value={`${s.avg_cii_activity_pct_normal}%`} color="#2e7d32" />
          <KPI label="CPEO" value={`${s.pct_cpeo}%`} color={COLOR2} />
          <KPI label="Myopathy" value={`${s.pct_myopathy}%`} color={COLOR2} />
          <KPI label="Cardiomyopathy" value={`${s.pct_cardiomyopathy}%`} color={COLOR3} />
          <KPI label="Isolated HCM (m.4300AG)" value={`${s.pct_isolated_hcm_m4300ag}%`} color={COLOR3} />
          <KPI label="SNHL" value={`${s.pct_snhl}%`} color={COLOR5} />
          <KPI label="Ragged-Red Fibres" value={`${s.pct_ragged_red_fibres}%`} color={COLOR} />
          <KPI label="Avg Onset (yr)" value={s.avg_age_onset_yr} />
        </div>
      </SectionCard>

      {/* DISTINCTIVE FEATURE alert */}
      <div className="alert border-0 mb-4" style={{ backgroundColor: '#fce4ec', borderLeft: `6px solid ${COLOR3}` }}>
        <strong style={{ color: COLOR3 }}>⚠️ MOST DISTINCTIVE MT-TI FEATURE — m.4300A&gt;G ISOLATED HCM:</strong>
        <br />
        m.4300A&gt;G is the <strong>only</strong> mt-tRNA mutation documented to cause <strong>isolated hypertrophic
        cardiomyopathy without CPEO</strong> as the dominant phenotype at low heteroplasmy. Cardiac heteroplasmy
        is amplified 10–25% above blood. <strong>Annual echo + Holter MANDATORY</strong> for ALL m.4300A&gt;G
        carriers from diagnosis — HCM can precede any neuromuscular symptoms by years.{' '}
        <strong style={{ color: COLOR3 }}>AMIODARONE ABSOLUTE CI</strong> — use beta-blockers only.
      </div>

      {/* MT-TQ overlap alert */}
      <div className="alert border-0 mb-4" style={{ backgroundColor: '#e8f5e9', borderLeft: `6px solid ${COLOR4}` }}>
        <strong style={{ color: COLOR4 }}>ℹ️ MT-TI–MT-TQ JUNCTION OVERLAP (rCRS 4329–4331):</strong>
        <br />
        MT-TI (H-strand 4263–4331) overlaps MT-TQ (L-strand 4329–4400) at 3 nucleotides.
        Large deletions spanning this boundary simultaneously impair <strong>tRNA-Ile AND tRNA-Gln</strong>,
        producing compound OXPHOS deficiency exceeding single-tRNA loss. Always verify
        L-strand NGS coverage at 4329–4400 when investigating suspected MT-TI large deletions.
      </div>

      {/* Phenotype distribution */}
      <SectionCard title="Phenotype Distribution — 40 Patients" borderColor={COLOR2}>
        {pheno_dist.map((p, i) => (
          <Bar key={i} label={`${p.phenotype} (n=${p.count})`} value={p.pct} color={i === 0 ? COLOR3 : COLOR2} />
        ))}
      </SectionCard>

      {/* Molecular features */}
      <SectionCard title="Key Molecular Features" borderColor={COLOR4}>
        <ul className="mb-0 small">
          {mol_feats.map((f, i) => <li key={i} className="mb-1">{f}</li>)}
        </ul>
      </SectionCard>

      {/* Clinical alerts */}
      <SectionCard title="Clinical Alerts" borderColor={COLOR3}>
        <div className="row g-2">
          {alerts.map((a, i) => (
            <div key={i} className="col-12 col-md-6">
              <div className="p-2 rounded" style={{ backgroundColor: '#fce4ec', border: `1px solid ${COLOR3}` }}>
                <div className="fw-bold small" style={{ color: COLOR3 }}>{a.alert}</div>
                <div className="text-muted" style={{ fontSize: '0.78rem' }}>{a.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Heteroplasmy map */}
      <SectionCard title="Heteroplasmy → Clinical Severity Map" borderColor={COLOR5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: LIGHT }}>
              <tr>
                <th>Heteroplasmy Range</th>
                <th>Phenotype</th>
                <th>Management</th>
              </tr>
            </thead>
            <tbody>
              {hmap.map((row, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ color: COLOR }}>{row.range}</td>
                  <td>{row.phenotype}</td>
                  <td>{row.management}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Variants & Cohort ────────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const variants = data.variant_summaries || [];
  const patients = data.per_patient || [];
  const triggers = data.trigger_rates || [];
  const treatments = data.treatment_info || [];
  const fp = data.biochemical_fingerprint || {};

  return (
    <>
      <SectionCard title="Variant Summaries" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: LIGHT }}>
              <tr>
                <th>Variant</th><th>Region</th><th>n</th>
                <th>Avg Hetero %</th><th>CI %</th><th>CIV %</th>
                <th>CPEO %</th><th>HCM %</th><th>Cardio %</th><th>SNHL %</th>
              </tr>
            </thead>
            <tbody>
              {variants.map((v, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ color: i === 2 ? COLOR3 : COLOR }}>{v.variant}</td>
                  <td>{v.region}</td>
                  <td>{v.n}</td>
                  <td>{v.avg_heteroplasmy_blood_pct}%</td>
                  <td>{v.avg_ci_activity_pct}%</td>
                  <td>{v.avg_civ_activity_pct}%</td>
                  <td>{v.pct_cpeo}%</td>
                  <td className={i === 2 ? 'fw-bold' : ''} style={{ color: i === 2 ? COLOR3 : undefined }}>{v.pct_isolated_hcm}%</td>
                  <td>{v.pct_cardiomyopathy}%</td>
                  <td>{v.pct_snhl}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-2 small text-muted">* HCM % = isolated HCM (m.4300AG column). Cardio % includes all forms of cardiomyopathy.</div>
      </SectionCard>

      {/* Biochemical fingerprint */}
      <SectionCard title="Biochemical Fingerprint — OXPHOS Activities" borderColor={COLOR4}>
        <div className="row g-3">
          <div className="col-md-4 text-center">
            <div className="fw-bold fs-4" style={{ color: COLOR3 }}>CI: {fp.CI_pct_normal}%</div>
            <div className="text-muted small">Complex I (mtDNA-encoded ND1–6, ND4L)</div>
          </div>
          <div className="col-md-4 text-center">
            <div className="fw-bold fs-4" style={{ color: COLOR3 }}>CIV: {fp.CIV_pct_normal}%</div>
            <div className="text-muted small">Complex IV (mtDNA-encoded CO1–3)</div>
          </div>
          <div className="col-md-4 text-center">
            <div className="fw-bold fs-4" style={{ color: '#2e7d32' }}>CII: {fp.CII_pct_normal}%</div>
            <div className="text-muted small">Complex II — NUCLEAR (CII NORMAL = mt-translation fingerprint)</div>
          </div>
        </div>
        <div className="mt-3 small">
          <div><strong>Pattern:</strong> {fp.pattern}</div>
          <div className="mt-1"><strong>BN-PAGE:</strong> {fp.BN_PAGE}</div>
          <div className="mt-1"><strong>Histochemistry:</strong> {fp.muscle_histochemistry}</div>
          <div className="mt-1 text-muted"><strong>NGS note:</strong> {fp.H_strand_note}</div>
        </div>
      </SectionCard>

      {/* Per-patient */}
      <SectionCard title="Per-Patient Cohort Table" borderColor={COLOR2}>
        <div className="table-responsive" style={{ maxHeight: 380, overflowY: 'auto' }}>
          <table className="table table-sm table-striped small mb-0">
            <thead style={{ backgroundColor: LIGHT, position: 'sticky', top: 0 }}>
              <tr>
                <th>ID</th><th>Variant</th><th>Sex</th><th>Onset</th>
                <th>Hetero %</th><th>CI %</th><th>CIV %</th><th>Lactate</th>
                <th>CPEO</th><th>Myo</th><th>Cardio</th><th>Iso-HCM</th><th>SNHL</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td style={{ fontSize: '0.7rem', color: COLOR }}>{p.variant.replace('Large deletion', 'Del')}</td>
                  <td>{p.sex}</td>
                  <td>{p.age_onset_yr} yr</td>
                  <td>{p.heteroplasmy_blood_pct}%</td>
                  <td style={{ color: COLOR3 }}>{p.ci_pct}%</td>
                  <td style={{ color: COLOR3 }}>{p.civ_pct}%</td>
                  <td>{p.lactate_mmol_L}</td>
                  <td>{p.cpeo ? '✓' : '–'}</td>
                  <td>{p.myopathy ? '✓' : '–'}</td>
                  <td>{p.cardiomyopathy ? '✓' : '–'}</td>
                  <td style={{ color: COLOR3, fontWeight: p.isolated_hcm ? 'bold' : 'normal' }}>{p.isolated_hcm ? '✓HCM' : '–'}</td>
                  <td>{p.snhl ? '✓' : '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Triggers */}
      <SectionCard title="Crisis Triggers — Relative Rates in Cohort" borderColor={COLOR3}>
        {triggers.slice(0, 8).map((t, i) => (
          <Bar key={i} label={t.trigger} value={t.pct} color={i < 3 ? COLOR3 : COLOR2} />
        ))}
      </SectionCard>

      {/* Treatments */}
      <SectionCard title="Evidence-Based Treatments" borderColor={COLOR4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#e8f5e9' }}>
              <tr><th>Agent</th><th>Evidence</th><th>Note</th></tr>
            </thead>
            <tbody>
              {treatments.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.agent}</td>
                  <td><span className="badge" style={{ backgroundColor: COLOR4 }}>{t.evidence}</span></td>
                  <td className="small">{t.note}</td>
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
  const ddx = data.ddx_comparison || [];
  const ci = data.contraindication_info || [];

  return (
    <>
      <SectionCard title="Differential Diagnosis" borderColor={COLOR5}>
        {ddx.map((d, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ backgroundColor: i % 2 === 0 ? LIGHT : '#f3e5f5', border: `1px solid ${COLOR5}` }}>
            <div className="fw-bold small" style={{ color: COLOR5 }}>{d.gene}</div>
            <div className="small"><strong>Disease:</strong> {d.disease}</div>
            <div className="small"><strong>OXPHOS:</strong> {d.oxphos}</div>
            <div className="small text-muted"><strong>Key DDx:</strong> {d.distinguisher}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Absolute Contraindications & Cautions" borderColor={COLOR3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#fce4ec' }}>
              <tr><th>Agent</th><th>Category</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {ci.map((c, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ color: COLOR3 }}>{c.agent}</td>
                  <td><span className="badge" style={{ backgroundColor: c.category.includes('ABSOLUTE') ? COLOR3 : '#e65100' }}>{c.category}</span></td>
                  <td className="small">{c.rationale}</td>
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
  const gb = data.gene_biology || {};
  const ct = data.clinical_terms || {};
  const ph = data.pharmacology || {};
  const refs = data.key_references || [];

  const Section = ({ title, obj, color }) => (
    <SectionCard title={title} borderColor={color}>
      {Object.entries(obj).map(([k, v], i) => (
        <div key={i} className="mb-3">
          <div className="fw-bold small" style={{ color }}>{k.replace(/_/g, ' ')}</div>
          <div className="small text-muted">{typeof v === 'object' ? JSON.stringify(v, null, 2) : v}</div>
        </div>
      ))}
    </SectionCard>
  );

  return (
    <>
      <Section title="Gene Biology" obj={gb} color={COLOR4} />
      <Section title="Clinical Terminology" obj={ct} color={COLOR2} />
      <Section title="Pharmacology" obj={ph} color={COLOR3} />
      <SectionCard title="Key References" borderColor={COLOR}>
        <ol className="small mb-0">
          {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ol>
      </SectionCard>
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function MTTIPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      fetch(`${API}/api/mtti/overview`).then(r => r.json()),
      fetch(`${API}/api/mtti/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mtti/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: 32 }}>🧬</span>
        <div>
          <h3 className="fw-bold mb-0" style={{ color: COLOR }}>
            MT-TI — tRNA-Ile — m.4300A&gt;G Isolated HCM / CPEO / Myopathy / Combined CI+CIV Deficiency
          </h3>
          <div className="text-muted small">
            FOURTH tRNA in mt-genome · H-strand rCRS 4263–4331 · MT-TQ L-strand overlap at 4329–4331 · IARS2 nuclear DDx (CAGSSS) · seed-807
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {loading && <div className="text-muted">Loading MT-TI data…</div>}
      {error && <div className="alert alert-danger">{error}</div>}

      {!loading && !error && (
        <>
          {tab === 0 && <OverviewTab data={overview} />}
          {tab === 1 && <VariantsTab data={breakdown} />}
          {tab === 2 && <DDxTab data={breakdown} />}
          {tab === 3 && <DefinitionsTab data={definitions} />}
        </>
      )}
    </div>
  );
}
