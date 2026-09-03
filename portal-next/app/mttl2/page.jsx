'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Cohort', 'DDx & Management', 'Definitions'];
const COLOR  = '#006064';   // deep teal — tRNA-Leu(CUN) / CPEO+Myopathy+Optic Atrophy
const LIGHT  = '#e0f7fa';
const COLOR2 = '#00838f';   // medium teal — CPEO / ophthalmoplegia
const COLOR3 = '#b71c1c';   // dark red — absolute CIs / severe
const COLOR4 = '#1b5e20';   // dark green — biochemical fingerprint / OXPHOS
const COLOR5 = '#4a148c';   // deep purple — optic atrophy / treatment

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
            <h4 className="fw-bold mb-1" style={{ color: COLOR }}>MT-TL2 — tRNA-Leu(CUN)</h4>
            <div className="text-muted small">
              <span className="badge me-1" style={{ background: COLOR }}>OMIM *590055</span>
              <span className="badge me-1" style={{ background: COLOR2 }}>Combined CI+CIV Deficiency</span>
              <span className="badge me-1 bg-dark">CUN codons: CUA/CUC/CUG/CUU</span>
              <span className="badge me-1" style={{ background: COLOR5 }}>CPEO / Myopathy / Optic Atrophy</span>
            </div>
            <p className="mt-2 mb-0 small">
              MT-TL2 encodes mitochondrial tRNA-Leu(CUN) (UAA anticodon, 71 nt) — H-strand rCRS 12266–12336 —
              immediately after MT-TS2 (12207–12265) and before MT-ND5 (12337–14148).
              One of <strong>two mt-tRNA-Leu genes</strong>: MT-TL1 (UUR/MELAS) vs MT-TL2 (CUN/CPEO-Optic Atrophy).
              Mutations cause <strong>combined CI + CIV deficiency</strong> (mt-translation fingerprint: CII NORMAL).
              Key features: CPEO, myopathy, optic atrophy (DISTINCTIVE — m.12308A>G).
              <strong className="text-danger"> NO stroke-like episodes</strong> (KEY DDx from MT-TL1 MELAS — same anticodon, different disease) ·
              <strong className="text-danger"> NO myoclonic epilepsy</strong> (vs MT-TK MERRF) ·
              <strong className="text-danger"> NO MSL</strong> (vs MT-TK MERRF).
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
        <KPI label="Optic Atrophy" value={`${s.pct_optic_atrophy}%`} color={COLOR5} />
        <KPI label="SNHL" value={`${s.pct_snhl}%`} color={COLOR2} />
        <KPI label="Cardiomyopathy" value={`${s.pct_cardiomyopathy}%`} color={COLOR3} />
        <KPI label="Mean Onset (yr)" value={s.avg_age_onset_yr} />
      </div>

      {/* Two leucine tRNA distinction */}
      <SectionCard title="Two Mitochondrial Leucine tRNA Genes — Critical Distinction" borderColor={COLOR5}>
        <div className="row">
          <div className="col-md-6">
            <div className="p-3 rounded mb-2" style={{ background: '#e8f5e9', border: '2px solid #2e7d32' }}>
              <div className="fw-bold small" style={{ color: '#2e7d32' }}>MT-TL2 — THIS GENE (rCRS 12266–12336)</div>
              <div className="small mt-1">CUN codons (CUA/CUC/CUG/CUU) · UAA anticodon · 71 nt</div>
              <div className="small">Disease: CPEO + Myopathy + <strong>Optic Atrophy</strong></div>
              <div className="small text-muted">NO stroke-like episodes · NO MIDD · OXPHOS: CI+CIV reduced</div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-3 rounded mb-2" style={{ background: '#fff3e0', border: '2px solid #e65100' }}>
              <div className="fw-bold small" style={{ color: '#e65100' }}>MT-TL1 — MELAS GENE (rCRS 3230–3304)</div>
              <div className="small mt-1">UUR codons (UUA/UUG) · UAA anticodon · 74 nt</div>
              <div className="small">Disease: MELAS + MIDD + <strong>Stroke-like Episodes</strong></div>
              <div className="small text-muted">IV L-arginine · No optic atrophy primary · CI+CIII+CIV pan-OXPHOS</div>
            </div>
          </div>
        </div>
        <p className="small text-muted mb-0 mt-2">
          Both have UAA anticodon but decode <em>different</em> Leu codons. The clinical phenotypes are completely different.
          Never confuse m.12311T&gt;C (MT-TL2 = CPEO) with m.3243A&gt;G (MT-TL1 = MELAS).
        </p>
      </SectionCard>

      {/* Heteroplasmy map */}
      <SectionCard title="Heteroplasmy–Phenotype Map (m.12311T>C; blood underestimates by 10–15%)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Blood heteroplasmy</th><th>Clinical phenotype</th><th>Management</th>
            </tr></thead>
            <tbody>
              {hmap.map((r, i) => (
                <tr key={i}>
                  <td className="fw-bold">{r.range}</td>
                  <td>{r.phenotype}</td>
                  <td className="text-muted">{r.management}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Clinical alerts */}
      <SectionCard title="Clinical Alerts — Contraindications" borderColor={COLOR3}>
        <div className="row">
          {alerts.map((a, i) => (
            <div className="col-md-6 mb-2" key={i}>
              <div className="p-2 rounded" style={{ background: '#fce4ec', border: '1px solid #c62828' }}>
                <div className="fw-bold small" style={{ color: COLOR3 }}>{a.alert}</div>
                <div className="small text-muted">{a.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Phenotype distribution */}
      <SectionCard title="Phenotype Distribution (40-patient cohort, seed-795)" borderColor={COLOR}>
        {pheno_dist.map((p, i) => (
          <Bar key={i} label={`${p.phenotype} (n=${p.count})`} value={p.pct} color={COLOR} />
        ))}
      </SectionCard>

      {/* Molecular features */}
      <SectionCard title="Key Molecular Features" borderColor={COLOR4}>
        <ul className="small mb-0">
          {mol_feats.map((f, i) => <li key={i} className="mb-1">{f}</li>)}
        </ul>
      </SectionCard>

      {/* Cohort summary */}
      <SectionCard title="Cohort Summary" borderColor={COLOR2}>
        <ul className="small mb-0">
          {feats.map((f, i) => <li key={i} className="mb-1">{f}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Tab: Variants & Cohort ─────────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const vs = data.variant_summaries || [];
  const pts = data.per_patient || [];
  const triggers = data.trigger_rates || [];
  const txs = data.treatment_info || [];
  const bf = data.biochemical_fingerprint || {};

  return (
    <div>
      {/* Variant summaries */}
      <SectionCard title="Variant Summaries by Mutation" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Variant</th><th>Region</th><th>N</th>
              <th>Het%</th><th>CI%</th><th>CIV%</th>
              <th>CPEO%</th><th>Myopathy%</th><th>Optic%</th><th>SNHL%</th><th>Cardio%</th>
            </tr></thead>
            <tbody>
              {vs.map((v, i) => (
                <tr key={i}>
                  <td className="fw-bold">{v.variant}</td>
                  <td>{v.region}</td>
                  <td>{v.n}</td>
                  <td>{v.avg_heteroplasmy_blood_pct}</td>
                  <td style={{ color: COLOR3 }}>{v.avg_ci_activity_pct}</td>
                  <td style={{ color: COLOR3 }}>{v.avg_civ_activity_pct}</td>
                  <td>{v.pct_cpeo}%</td>
                  <td>{v.pct_myopathy}%</td>
                  <td style={{ color: COLOR5 }}>{v.pct_optic_atrophy}%</td>
                  <td>{v.pct_snhl}%</td>
                  <td>{v.pct_cardiomyopathy}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3">
          {vs.map((v, i) => (
            <div key={i} className="mb-2 p-2 rounded" style={{ background: LIGHT }}>
              <span className="fw-bold small me-2">{v.variant}</span>
              <span className="text-muted small">{v.note}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Biochemical fingerprint */}
      <SectionCard title="Biochemical Fingerprint (BN-PAGE / Spectrophotometry)" borderColor={COLOR4}>
        <div className="row">
          {[
            { label: 'CI Activity', val: bf.CI_pct_normal, color: COLOR3 },
            { label: 'CIV Activity', val: bf.CIV_pct_normal, color: COLOR3 },
            { label: 'CII Activity (NORMAL)', val: bf.CII_pct_normal, color: COLOR4 },
          ].map((x, i) => (
            <div className="col-md-4 mb-2" key={i}>
              <div className="p-2 rounded text-center" style={{ background: LIGHT }}>
                <div className="fw-bold" style={{ color: x.color }}>{x.val}%</div>
                <div className="small text-muted">{x.label}</div>
              </div>
            </div>
          ))}
        </div>
        <div className="small mt-2">
          <div><strong>Pattern:</strong> {bf.pattern}</div>
          <div><strong>BN-PAGE:</strong> {bf.BN_PAGE}</div>
          <div><strong>Histochem:</strong> {bf.muscle_histochemistry}</div>
        </div>
      </SectionCard>

      {/* Per-patient table (first 20) */}
      <SectionCard title="Per-Patient Cohort Data (40 patients, seed-795 — first 20 shown)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>ID</th><th>Variant</th><th>Sex</th><th>Onset</th>
              <th>Het%</th><th>CI%</th><th>CIV%</th><th>CII%</th>
              <th>Lactate</th><th>CPEO</th><th>Myo</th><th>Optic</th><th>SNHL</th><th>RRF</th>
            </tr></thead>
            <tbody>
              {pts.slice(0, 20).map((p, i) => (
                <tr key={i}>
                  <td className="fw-bold">{p.id}</td>
                  <td>{p.variant}</td>
                  <td>{p.sex}</td>
                  <td>{p.age_onset_yr}yr</td>
                  <td>{p.heteroplasmy_blood_pct}%</td>
                  <td style={{ color: COLOR3 }}>{p.ci_pct}</td>
                  <td style={{ color: COLOR3 }}>{p.civ_pct}</td>
                  <td style={{ color: COLOR4 }}>{p.cii_pct}</td>
                  <td>{p.lactate_mmol_L}</td>
                  <td>{p.cpeo ? '✓' : '—'}</td>
                  <td>{p.myopathy ? '✓' : '—'}</td>
                  <td style={{ color: COLOR5 }}>{p.optic_atrophy ? '✓' : '—'}</td>
                  <td>{p.snhl ? '✓' : '—'}</td>
                  <td>{p.ragged_red_fibres ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Trigger rates */}
      <SectionCard title="Crisis Trigger Rates" borderColor={COLOR3}>
        {triggers.map((t, i) => (
          <Bar key={i} label={t.trigger} value={t.pct} color={COLOR3} />
        ))}
      </SectionCard>

      {/* Treatments */}
      <SectionCard title="Treatment Summary" borderColor={COLOR4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Agent</th><th>Evidence</th><th>Note</th>
            </tr></thead>
            <tbody>
              {txs.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.agent}</td>
                  <td><span className="badge" style={{ background: COLOR }}>{t.evidence}</span></td>
                  <td className="text-muted">{t.note}</td>
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
function DDxTab({ bdata }) {
  if (!bdata) return <p className="text-muted">Loading…</p>;
  const ddx = bdata.ddx_comparison || [];
  const ci_info = bdata.contraindication_info || [];

  return (
    <div>
      {/* DDx table */}
      <SectionCard title="Differential Diagnosis — MT-TL2 vs Key Mimics" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Gene / Disease</th><th>Primary Disease</th><th>OXPHOS</th><th>Key Distinguisher from MT-TL2</th>
            </tr></thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i} style={{ background: d.gene === 'MT-TL1' ? '#fff8e1' : 'inherit' }}>
                  <td className="fw-bold">{d.gene}</td>
                  <td>{d.disease}</td>
                  <td>{d.oxphos}</td>
                  <td className="text-muted">{d.distinguisher}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small text-warning fw-bold mt-2 mb-0">
          ⚠ MT-TL1 is the MOST CRITICAL DDx — both are mt-tRNA-Leu genes with UAA anticodon.
          MT-TL1 = MELAS (stroke-like); MT-TL2 = CPEO + optic atrophy. NEVER use IV tPA in MT-TL2.
        </p>
      </SectionCard>

      {/* Contraindication details */}
      <SectionCard title="Contraindications — Detailed Rationale" borderColor={COLOR3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Agent</th><th>Category</th><th>Rationale</th>
            </tr></thead>
            <tbody>
              {ci_info.map((c, i) => (
                <tr key={i}>
                  <td className="fw-bold">{c.agent}</td>
                  <td><span className="badge" style={{ background: c.category.includes('ABSOLUTE') ? COLOR3 : '#e65100' }}>{c.category}</span></td>
                  <td className="text-muted">{c.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Optic atrophy management */}
      <SectionCard title="Optic Atrophy Management — m.12308A>G Specific" borderColor={COLOR5}>
        <div className="row">
          {[
            { title: 'Annual OCT', text: 'Retinal nerve fibre layer thickness — early atrophy detection before symptomatic visual loss' },
            { title: 'VEP Annually', text: 'Visual evoked potentials — prolonged P100 latency precedes VA loss; track progression' },
            { title: 'Neuro-ophthalmology', text: 'Early referral — establish baseline acuity + colour vision + contrast sensitivity; monitor every 6 months' },
            { title: 'Idebenone 900 mg/day', text: 'Level C — analogous to LHON; may slow RGC atrophy; weaker evidence than LHON-specific trials' },
            { title: 'Avoid Ethambutol', text: 'Optic nerve toxin — synergistic toxicity with pre-existing MT-TL2 RGC vulnerability; use alternative antibiotics' },
            { title: 'Avoid Amiodarone', text: 'Mitochondrial membrane toxin; optic neuropathy side-effect; worsens CI+CIV insufficiency' },
          ].map((item, i) => (
            <div className="col-md-6 mb-2" key={i}>
              <div className="p-2 rounded" style={{ background: '#f3e5f5', border: `1px solid ${COLOR5}` }}>
                <div className="fw-bold small" style={{ color: COLOR5 }}>{item.title}</div>
                <div className="small text-muted">{item.text}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Emergency protocol */}
      <SectionCard title="Acute Crisis Protocol" borderColor={COLOR3}>
        <div className="row">
          {[
            { step: '1. GIR 6–8 mg/kg/min', detail: 'IV dextrose — NEVER fast; maintain normoglycaemia; prevents catabolism' },
            { step: '2. IV Thiamine 10-20 mg/kg', detail: 'Empiric — PDH/KGDH cofactor; give BEFORE glucose if Wernicke risk; continue oral 100-300 mg/day' },
            { step: '3. Avoid Propofol', detail: 'Use sevoflurane for anaesthesia — PRIS risk amplified in CI+CIV deficiency' },
            { step: '4. Bicarbonate if pH <7.2', detail: 'Target pH >7.2; do not fully correct lactic acidosis (overtreatment risk); treat underlying cause' },
            { step: '5. Avoid VPA', detail: 'Use LEV for seizures — VPA absolutely contraindicated; no CBZ/OXC in myopathy' },
            { step: '6. Avoid Metformin', detail: 'Discontinue immediately if patient on metformin — fatal lactic acidosis in CI deficiency' },
          ].map((item, i) => (
            <div className="col-md-6 mb-2" key={i}>
              <div className="p-2 rounded" style={{ background: '#fce4ec', border: `1px solid ${COLOR3}` }}>
                <div className="fw-bold small" style={{ color: COLOR3 }}>{item.step}</div>
                <div className="small text-muted">{item.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const gb = data.gene_biology || {};
  const ct = data.clinical_terms || {};
  const ph = data.pharmacology || {};
  const refs = data.key_references || [];

  return (
    <div>
      <SectionCard title="Gene Biology" borderColor={COLOR4}>
        {Object.entries(gb).map(([k, v], i) => (
          <div key={i} className="mb-3">
            <div className="fw-bold small" style={{ color: COLOR4 }}>{k.replace(/_/g, ' ')}</div>
            <div className="small text-muted">{v}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Terms" borderColor={COLOR2}>
        {Object.entries(ct).map(([k, v], i) => (
          <div key={i} className="mb-3">
            <div className="fw-bold small" style={{ color: COLOR2 }}>{k.replace(/_/g, ' ')}</div>
            <div className="small text-muted">{v}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Pharmacology" borderColor={COLOR3}>
        {ph.preferred_aed && (
          <div className="mb-2">
            <span className="fw-bold small">Preferred AED: </span>
            <span className="small">{ph.preferred_aed}</span>
          </div>
        )}
        {ph.emergency_protocol && (
          <div className="mb-2">
            <span className="fw-bold small">Emergency Protocol: </span>
            <span className="small">{ph.emergency_protocol}</span>
          </div>
        )}
        {ph.optic_atrophy_note && (
          <div className="mb-2">
            <span className="fw-bold small" style={{ color: COLOR5 }}>Optic Atrophy Rx: </span>
            <span className="small">{ph.optic_atrophy_note}</span>
          </div>
        )}
        {ph.absolute_ci && (
          <div className="mt-3">
            <div className="fw-bold small mb-2" style={{ color: COLOR3 }}>Absolute Contraindications:</div>
            {Object.entries(ph.absolute_ci).map(([drug, reason], i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: '#fce4ec' }}>
                <span className="fw-bold small me-2" style={{ color: COLOR3 }}>{drug}:</span>
                <span className="small text-muted">{reason}</span>
              </div>
            ))}
          </div>
        )}
      </SectionCard>

      <SectionCard title="Key References" borderColor={COLOR}>
        <ol className="small mb-0">
          {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function MTTL2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mttl2/overview`).then(r => r.json()),
      fetch(`${API}/api/mttl2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mttl2/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => {
      setOverview(o);
      setBreakdown(b);
      setDefinitions(d);
    }).catch(e => setError(String(e)));
  }, []);

  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  return (
    <div className="container-fluid py-4">
      <h2 className="fw-bold mb-1" style={{ color: COLOR }}>
        🧬 MT-TL2 — tRNA-Leu(CUN) Dashboard
      </h2>
      <p className="text-muted mb-3 small">
        Combined CI+CIV Deficiency · CPEO / Myopathy / Optic Atrophy · m.12311T&gt;C most common ·
        m.12308A&gt;G CPEO+Optic Atrophy distinctive · rCRS H-strand 12266–12336 ·
        Flanked by MT-TS2 (12207–12265) and MT-ND5 (12337–14148)
      </p>

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <VariantsTab data={breakdown} />}
      {tab === 2 && <DDxTab bdata={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
