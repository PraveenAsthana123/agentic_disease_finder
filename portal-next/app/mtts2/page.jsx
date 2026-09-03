'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Cohort', 'DDx & Management', 'Definitions'];
const COLOR  = '#01579b';   // deep blue — tRNA-Ser(AGY) / CPEO+Myopathy+SNHL
const LIGHT  = '#e3f2fd';
const COLOR2 = '#0277bd';   // medium blue — CPEO / ophthalmoplegia
const COLOR3 = '#b71c1c';   // dark red — absolute CIs / severe
const COLOR4 = '#1b5e20';   // dark green — biochemical fingerprint / OXPHOS
const COLOR5 = '#4a148c';   // deep purple — treatment / SNHL management

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
      {/* Gene header */}
      <div className="p-3 mb-4 rounded" style={{ background: LIGHT, borderLeft: `5px solid ${COLOR}` }}>
        <div className="d-flex flex-wrap align-items-start gap-3">
          <div>
            <h4 className="fw-bold mb-1" style={{ color: COLOR }}>MT-TS2 — tRNA-Ser(AGY)</h4>
            <div className="text-muted small">
              <span className="badge me-1" style={{ background: COLOR }}>OMIM *590085</span>
              <span className="badge me-1" style={{ background: COLOR2 }}>Combined CI+CIV Deficiency</span>
              <span className="badge me-1 bg-dark">SHORTEST mt-tRNA (59 nt)</span>
              <span className="badge me-1" style={{ background: COLOR5 }}>CPEO / Myopathy / SNHL</span>
            </div>
            <p className="mt-2 mb-0 small">
              MT-TS2 encodes mitochondrial tRNA-Ser(AGY) (GCU anticodon, 59 nt) — H-strand rCRS 12207–12265 —
              immediately after MT-TH (12138–12206) and before MT-TL2 (12266–12336).
              At <strong>59 nt, MT-TS2 is the SHORTEST human mitochondrial tRNA</strong>.
              Mutations cause <strong>combined CI + CIV deficiency</strong> (mt-translation fingerprint: CII NORMAL).
              Key features: CPEO, myopathy, SNHL (can be isolated at low heteroplasmy).
              <strong className="text-danger"> NO myoclonic epilepsy</strong> (vs MT-TK MERRF) ·
              <strong className="text-danger"> NO stroke-like episodes</strong> (vs MT-TL1 MELAS) ·
              <strong className="text-danger"> NO MSL</strong> (vs MT-TK MERRF) ·
              Less cardiomyopathy than MT-TH.
            </p>
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Patients" value={s.n_patients} />
        <KPI label="Mean Heteroplasmy (blood)" value={`${s.avg_heteroplasmy_blood_pct}%`} color={COLOR2} />
        <KPI label="Mean CI Activity" value={`${s.avg_ci_activity_pct_normal}%`} color={COLOR3} />
        <KPI label="Mean CIV Activity" value={`${s.avg_civ_activity_pct_normal}%`} color={COLOR3} />
        <KPI label="CII (NORMAL)" value={`${s.avg_cii_activity_pct_normal}%`} color={COLOR4} />
        <KPI label="CPEO" value={`${s.pct_cpeo}%`} color={COLOR2} />
        <KPI label="Myopathy" value={`${s.pct_myopathy}%`} color={COLOR} />
        <KPI label="SNHL" value={`${s.pct_snhl}%`} color={COLOR5} />
        <KPI label="Cardiomyopathy" value={`${s.pct_cardiomyopathy}%`} color={COLOR3} />
        <KPI label="Mean Onset (yr)" value={s.avg_age_onset_yr} />
      </div>

      {/* Biochemical fingerprint + Heteroplasmy map */}
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <SectionCard title="🔬 OXPHOS Biochemical Fingerprint (mt-Translation)" borderColor={COLOR4}>
            <div className="p-2 rounded mb-2" style={{ background: '#e8f5e9' }}>
              <strong>CI + CIV REDUCED — CII NORMAL</strong>
              <div className="text-muted small">mt-translation fingerprint: all 7 CI ND-subunits + all 3 CIV CO-subunits are mtDNA-encoded; CII (SDH) is nuclear-encoded → NORMAL despite pan-tRNA defect</div>
            </div>
            <Bar label="CI Activity (% normal)" value={s.avg_ci_activity_pct_normal} color={COLOR3} />
            <Bar label="CIV Activity (% normal)" value={s.avg_civ_activity_pct_normal} color={COLOR3} />
            <Bar label="CII Activity (% normal — NORMAL)" value={Math.min(s.avg_cii_activity_pct_normal, 100)} color={COLOR4} />
            <div className="alert alert-info mt-2 py-2 small">
              BN-PAGE: CI + CIV bands absent/reduced · CII band present · Muscle histology: RRF on Gomori trichrome · COX-negative fibres at moderate-high heteroplasmy
            </div>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="📊 Heteroplasmy Threshold Map (blood; muscle ~10-15% higher)" borderColor={COLOR2}>
            {hmap.map((row, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff' }}>
                <div className="d-flex justify-content-between">
                  <strong className="small" style={{ color: COLOR }}>{row.range}</strong>
                </div>
                <div className="text-muted small">{row.phenotype}</div>
                <div className="text-success small">{row.management}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Unique feature highlight */}
      <div className="alert mb-4" style={{ background: '#fff8e1', borderLeft: `5px solid #f57f17` }}>
        <strong style={{ color: '#e65100' }}>⚡ UNIQUE: SHORTEST Mitochondrial tRNA Gene</strong>
        <div className="small mt-1">
          MT-TS2 at <strong>59 nt</strong> is shorter than MT-TH (69 nt), MT-TK (69 nt), MT-TL1 (74 nt), and most other mt-tRNAs (69–75 nt).
          The compact structure (truncated D-arm + compressed variable loop) means a single point mutation at any position
          destabilises the entire fold more readily. This accounts for the broad phenotypic spectrum:
          <strong> isolated SNHL at low heteroplasmy (&lt;50%) all the way to multisystem encephalomyopathy at high heteroplasmy (&gt;80%)</strong>.
          Isolated SNHL presenting years before CPEO/myopathy is a distinctive MT-TS2 clinical feature.
        </div>
      </div>

      {/* Phenotype distribution */}
      <SectionCard title="📈 Cohort Phenotype Distribution (n=40, seed-793)" borderColor={COLOR}>
        <div className="row">
          {pheno_dist.map((ph, i) => (
            <div key={i} className="col-md-6 mb-2">
              <Bar label={ph.phenotype} value={ph.pct} color={[COLOR, COLOR2, COLOR3, COLOR4, COLOR5][i % 5]} />
              <div className="text-muted" style={{ fontSize: '0.72rem' }}>n={ph.count}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Key molecular features */}
      <SectionCard title="🧬 Key Molecular & Clinical Features" borderColor={COLOR2}>
        <ul className="list-unstyled mb-0">
          {mol_feats.map((f, i) => (
            <li key={i} className="mb-1 small">
              <span style={{ color: COLOR }}>▸</span> {f}
            </li>
          ))}
        </ul>
      </SectionCard>

      {/* Clinical alerts */}
      <SectionCard title="⚠️ Critical Clinical Alerts" borderColor={COLOR3}>
        {alerts.map((a, i) => (
          <div key={i} className="mb-2 p-2 rounded" style={{ background: i % 2 === 0 ? '#fff3e0' : '#fce4ec' }}>
            <strong className="text-danger small">{a.alert}</strong>
            <div className="text-muted small">{a.detail}</div>
          </div>
        ))}
      </SectionCard>

      {/* Cohort summary bullets */}
      <SectionCard title="📋 Cohort Summary" borderColor={COLOR4}>
        <ul className="list-group list-group-flush">
          {feats.map((f, i) => (
            <li key={i} className="list-group-item py-1 small">{f}</li>
          ))}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Tab: Variants & Cohort ─────────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const variants = data.variant_summaries || [];
  const per_patient = data.per_patient || [];
  const triggers = data.trigger_rates || [];
  const fingerprint = data.biochemical_fingerprint || {};

  return (
    <div>
      {/* Biochemical fingerprint summary */}
      <SectionCard title="🔬 Biochemical Fingerprint Summary" borderColor={COLOR4}>
        <div className="row g-2">
          {Object.entries(fingerprint).map(([k, v], i) => (
            <div key={i} className="col-md-6">
              <div className="p-2 rounded" style={{ background: LIGHT }}>
                <strong className="small">{k.replace(/_/g, ' ')}</strong>
                <div className="text-muted small">{v}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Variant summaries */}
      <SectionCard title="🧬 Variant-Level Analysis" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>Variant</th><th>Region</th><th>n</th><th>Het% (blood)</th>
                <th>CI%</th><th>CIV%</th><th>CPEO%</th><th>Myopathy%</th><th>SNHL%</th><th>Cardio%</th>
              </tr>
            </thead>
            <tbody>
              {variants.map((v, i) => (
                <tr key={i}>
                  <td><strong style={{ color: COLOR }}>{v.variant}</strong></td>
                  <td className="small text-muted">{v.region}</td>
                  <td>{v.n}</td>
                  <td>{v.avg_heteroplasmy_blood_pct}%</td>
                  <td style={{ color: COLOR3 }}>{v.avg_ci_activity_pct}%</td>
                  <td style={{ color: COLOR3 }}>{v.avg_civ_activity_pct}%</td>
                  <td>{v.pct_cpeo}%</td>
                  <td>{v.pct_myopathy}%</td>
                  <td>{v.pct_snhl}%</td>
                  <td>{v.pct_cardiomyopathy}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {variants.map((v, i) => (
          <div key={i} className="mb-2 p-2 rounded small" style={{ background: i % 2 === 0 ? LIGHT : '#fff' }}>
            <strong style={{ color: COLOR }}>{v.variant}</strong> — {v.note}
          </div>
        ))}
      </SectionCard>

      {/* Trigger rates */}
      <SectionCard title="⚡ Crisis Trigger Rates" borderColor={COLOR2}>
        {triggers.map((t, i) => (
          <Bar key={i} label={t.trigger} value={t.pct} color={[COLOR3, COLOR, COLOR2, COLOR4][i % 4]} />
        ))}
      </SectionCard>

      {/* Per-patient table */}
      <SectionCard title="🗂️ Per-Patient Data (40 patients, seed-793)" borderColor={COLOR}>
        <div className="table-responsive" style={{ maxHeight: 420, overflowY: 'auto' }}>
          <table className="table table-sm table-striped">
            <thead style={{ background: LIGHT, position: 'sticky', top: 0 }}>
              <tr>
                <th>ID</th><th>Variant</th><th>Sex</th><th>Onset</th><th>Het%</th>
                <th>CI%</th><th>CIV%</th><th>CII%</th><th>Lactate</th>
                <th>CPEO</th><th>Myopathy</th><th>KSS</th><th>SNHL</th><th>RRF</th>
              </tr>
            </thead>
            <tbody>
              {per_patient.map((p, i) => (
                <tr key={i}>
                  <td className="small">{p.id}</td>
                  <td className="small" style={{ color: COLOR }}>{p.variant}</td>
                  <td>{p.sex}</td>
                  <td>{p.age_onset_yr}y</td>
                  <td>{p.heteroplasmy_blood_pct}%</td>
                  <td style={{ color: COLOR3 }}>{p.ci_pct}%</td>
                  <td style={{ color: COLOR3 }}>{p.civ_pct}%</td>
                  <td style={{ color: COLOR4 }}>{p.cii_pct}%</td>
                  <td>{p.lactate_mmol_L}</td>
                  <td>{p.cpeo ? '✓' : '–'}</td>
                  <td>{p.myopathy ? '✓' : '–'}</td>
                  <td>{p.kss ? '✓' : '–'}</td>
                  <td>{p.snhl ? '✓' : '–'}</td>
                  <td>{p.ragged_red_fibres ? '✓' : '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: DDx & Management ──────────────────────────────────────────────────────
function DDxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ddx = data.ddx_comparison || [];
  const tx = data.treatment_info || [];
  const ci = data.contraindication_info || [];

  return (
    <div>
      {/* DDx comparison */}
      <SectionCard title="🔍 Differential Diagnosis — Key Distinguishers" borderColor={COLOR2}>
        {ddx.map((d, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff' }}>
            <div className="d-flex align-items-center gap-2 mb-1">
              <strong style={{ color: COLOR }}>{d.gene}</strong>
              <span className="badge" style={{ background: COLOR2 }}>{d.disease}</span>
              <span className="badge bg-secondary">{d.oxphos}</span>
            </div>
            <div className="small text-muted">{d.distinguisher}</div>
          </div>
        ))}
      </SectionCard>

      {/* Absolute contraindications */}
      <SectionCard title="🚫 Absolute Contraindications & High-Risk Drugs" borderColor={COLOR3}>
        {ci.map((c, i) => (
          <div key={i} className="mb-2 p-2 rounded" style={{ background: i % 2 === 0 ? '#fce4ec' : '#fff3e0' }}>
            <div className="d-flex align-items-center gap-2">
              <strong className="text-danger">{c.agent}</strong>
              <span className={`badge ${c.category.includes('ABSOLUTE') ? 'bg-danger' : 'bg-warning text-dark'}`}>{c.category}</span>
            </div>
            <div className="small text-muted">{c.rationale}</div>
          </div>
        ))}
      </SectionCard>

      {/* Treatments */}
      <SectionCard title="💊 Evidence-Based Management" borderColor={COLOR5}>
        {tx.map((t, i) => (
          <div key={i} className="mb-2 p-2 rounded" style={{ background: i % 2 === 0 ? '#f3e5f5' : '#fff' }}>
            <div className="d-flex align-items-center gap-2">
              <strong style={{ color: COLOR5 }}>{t.agent}</strong>
              <span className="badge" style={{ background: COLOR5 }}>{t.evidence}</span>
            </div>
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const bio = data.gene_biology || {};
  const clin = data.clinical_terms || {};
  const pharm = data.pharmacology || {};
  const refs = data.key_references || [];

  return (
    <div>
      <SectionCard title="🧬 Gene Biology" borderColor={COLOR}>
        {Object.entries(bio).map(([k, v], i) => (
          <div key={i} className="mb-2 p-2 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff' }}>
            <strong className="small" style={{ color: COLOR }}>{k.replace(/_/g, ' ')}</strong>
            <div className="text-muted small">{v}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📖 Clinical Terms" borderColor={COLOR2}>
        {Object.entries(clin).map(([k, v], i) => (
          <div key={i} className="mb-2 p-2 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff' }}>
            <strong className="small" style={{ color: COLOR2 }}>{k.replace(/_/g, ' ')}</strong>
            <div className="text-muted small">{v}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="💊 Pharmacology" borderColor={COLOR4}>
        {pharm.absolute_ci && (
          <div className="mb-3">
            <strong className="text-danger small">Absolute Contraindications:</strong>
            {Object.entries(pharm.absolute_ci).map(([drug, reason], i) => (
              <div key={i} className="small text-muted ms-2">• <strong>{drug}</strong>: {reason}</div>
            ))}
          </div>
        )}
        {pharm.preferred_aed && (
          <div className="mb-2 small"><strong style={{ color: COLOR5 }}>Preferred AED:</strong> {pharm.preferred_aed}</div>
        )}
        {pharm.emergency_protocol && (
          <div className="mb-2 small"><strong style={{ color: COLOR3 }}>Emergency Protocol:</strong> {pharm.emergency_protocol}</div>
        )}
        {pharm.aminoglycoside_note && (
          <div className="mb-2 small"><strong className="text-warning">Aminoglycoside Warning:</strong> {pharm.aminoglycoside_note}</div>
        )}
      </SectionCard>

      <SectionCard title="📚 Key References" borderColor={COLOR5}>
        <ul className="list-unstyled mb-0">
          {refs.map((r, i) => (
            <li key={i} className="mb-1 small">
              <span style={{ color: COLOR }}>▸</span> {r}
            </li>
          ))}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function Mtts2Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/mtts2/overview`).then(r => r.json()),
      fetch(`${API}/api/mtts2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mtts2/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: 28 }}>🧬</span>
        <div>
          <h3 className="mb-0 fw-bold" style={{ color: COLOR }}>MT-TS2 — tRNA-Ser(AGY)</h3>
          <div className="text-muted small">
            Combined CI+CIV Deficiency · CPEO · Myopathy · SNHL (isolated at low Het%) ·
            SHORTEST mt-tRNA (59 nt) · m.12258C&gt;A · H-strand rCRS 12207–12265 · Maternal Inheritance
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger">Error: {error}</div>}
      {loading && <div className="text-muted">Loading…</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${activeTab === i ? 'active fw-bold' : ''}`}
              style={activeTab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setActiveTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {activeTab === 0 && <OverviewTab data={overview} />}
      {activeTab === 1 && <VariantsTab data={breakdown} />}
      {activeTab === 2 && <DDxTab data={breakdown} />}
      {activeTab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
