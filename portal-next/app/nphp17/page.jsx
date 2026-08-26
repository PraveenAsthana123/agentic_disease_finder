'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'MAPKBP1 Architecture & Genetics', 'Definitions'];

// NPHP17 colour scheme — MAPKBP1 / JNK scaffold / NPHP4 supercomplex / pure renal ultra-rare
const ACCENT  = '#1b5e20';   // deep green — NPHP4 supercomplex; TZ scaffold platform
const ACCENT2 = '#b71c1c';   // deep red — JNK/MAPK pathway; stress-kinase axis; apoptosis
const ACCENT3 = '#0d47a1';   // deep blue — 2q13.3 locus; NPHP1 DDx confusion; chromosome arm
const ACCENT4 = '#4a148c';   // deep purple — MAPKBP1 scaffold; ankyrin-repeat + leucine-zipper
const ACCENT5 = '#e65100';   // burnt orange — ESRD; CKD progression; tubular loss
const ACCENT6 = '#37474f';   // dark slate — molecular architecture; ultra-rare classification
const ACCENT7 = '#f57f17';   // amber — misdiagnosis (NPHP1 MLPA / ADPKD confusion)
const ACCENT8 = '#006064';   // dark teal — NPHP4 co-sequencing; supercomplex partners

const SEED = 373;
const _COHORT_SIZE = 40;

function KPI({ label, value, color }) {
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

function Alert({ color, children }) {
  return (
    <div className="alert mb-2" style={{ background: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 6 }}>
      {children}
    </div>
  );
}

function Section({ title, color, children }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

function Badge({ text, color }) {
  return <span className="badge me-1" style={{ background: color, fontSize: '0.72em' }}>{text}</span>;
}

function Bar({ label, value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{value}</span>
      </div>
      <div className="progress" style={{ height: 8, borderRadius: 4 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color, borderRadius: 4 }} />
      </div>
    </div>
  );
}

export default function NPHP17Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp17/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp17/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp17/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading NPHP17 cohort…</div>;
  if (error)   return <div className="container py-5 text-danger">Error: {error}</div>;
  if (!ov)     return null;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', border: `2px solid ${ACCENT}` }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div style={{ flex: 1, minWidth: 260 }}>
            <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
              &#x1f9ec; Nephronophthisis Type 17 (NPHP17) — MAPKBP1 / JNK Scaffold · NPHP4 Supercomplex · Ultra-Rare
            </h4>
            <div className="small text-muted mb-1">
              <strong>MAPKBP1</strong> (JIP4/SPAG9) · 2q13.3 · ~1,388 aa · JNK/MAPK scaffold ·
              NPHP4 supercomplex interactor · pure renal TIN + corticomedullary cysts ·
              ESRD median ~14–16yr (adolescent) · 0% laterality · 0% retinal · ultra-rare
            </div>
            <div className="small">
              <Badge text="OMIM *610889" color={ACCENT} />
              <Badge text="#616140 NPHP17" color={ACCENT} />
              <Badge text="AR biallelic LOF" color={ACCENT6} />
              <Badge text="2q13.3" color={ACCENT3} />
              <Badge text="NPHP4 supercomplex" color={ACCENT4} />
              <Badge text="JNK/MAPK scaffold" color={ACCENT2} />
              <Badge text="NO situs · NO retinal · NO CHF · NO Joubert" color={ACCENT} />
              <Badge text="Ultra-rare ~25–35 families" color={ACCENT6} />
            </div>
          </div>
          <div className="d-flex gap-2 flex-wrap">
            <span className="badge px-3 py-2" style={{ background: ACCENT, fontSize: '0.8em' }}>
              ESRD median ~14–16yr
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT2, fontSize: '0.8em' }}>
              JNK/MAPK TZ scaffold
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT3, fontSize: '0.8em' }}>
              2q13.3 ≠ NPHP1 (2q13)
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT5, fontSize: '0.8em' }}>
              ESRD/Tx {ov.pct_esrd_or_transplant}%
            </span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert color={ACCENT2}>
        <strong>&#x1f9ec; ONLY NPHP CAUSED BY JNK/MAPK SCAFFOLD LOSS AT THE TRANSITION ZONE:</strong> MAPKBP1 (JIP4)
        is the JNK-interacting scaffold protein 4 that bridges stress-kinase (JNK/MAPK) signalling
        to the ciliary transition zone via the NPHP4 supercomplex (NPHP1·NPHP4·NPHP8·RPGRIP1L).
        Loss → impaired JNK-mediated stress-response at TZ → tubular epithelial apoptosis
        dysregulation → progressive tubulointerstitial nephritis (TIN) → ESRD. Unique mechanism:
        kinase-scaffold dysfunction at TZ rather than structural TZ or IFT machinery loss.
        MAPKBP1 also regulates kinesin-1/KIF5B-dependent ciliary transport.
      </Alert>
      <Alert color={ACCENT3}>
        <strong>&#x26a0;&#xfe0f; 2q13.3 CHROMOSOME ARM TRAP — NPHP1 MLPA DOES NOT DETECT MAPKBP1:</strong> MAPKBP1 maps
        to 2q13.3; NPHP1 maps to 2q13 — same chromosome arm, DIFFERENT genes. Standard NPHP1
        290kb deletion MLPA covers NPHP1 only and CANNOT detect MAPKBP1 variants. Because both
        are on chromosome 2q13 region, targeted deletion panels may report "2q13 normal" while
        missing MAPKBP1. WES is the ONLY reliable diagnostic method. Any NPHP1-MLPA-negative
        patient with juvenile CKD + TIN pattern requires comprehensive NPHP gene panel via WES.
        NPHP4 must be co-sequenced — MAPKBP1 is a direct NPHP4-interacting protein.
      </Alert>
      <Alert color={ACCENT4}>
        <strong>&#x1f9ec; NPHP4 SUPERCOMPLEX — CO-SEQUENCING MANDATORY:</strong> MAPKBP1 interacts directly with
        NPHP4 (nephrocystin-4) as part of the NPHP1·NPHP4·NPHP8/RPGRIP1L·NPHP4 supercomplex at
        the transition zone. Always co-sequence NPHP4 when MAPKBP1 biallelic variants found.
        If MAPKBP1 has only ONE pathogenic allele: WES full NPHP4 supercomplex panel mandatory
        (NPHP1, NPHP4, NPHP8/RPGRIP1L). Digenic MAPKBP1+NPHP4 variants are a theoretical
        possibility given direct protein interaction. NPHP4 loss causes a similar pure renal
        phenotype — clinical phenotype alone cannot distinguish NPHP4 from NPHP17.
      </Alert>
      <Alert color={ACCENT7}>
        <strong>&#x26a0;&#xfe0f; MOST COMMON MISDIAGNOSIS — NPHP1 MLPA NEGATIVE ({ov.pct_misdiagnosed_nphp1}%) + ADPKD CONFUSION ({ov.pct_misdiagnosed_adpkd}%):</strong> NPHP1
        290kb MLPA is standard first-line → DOES NOT detect MAPKBP1. Bilateral corticomedullary
        cysts cause ADPKD/PKD1 misdiagnosis (~20%) before AR pattern is recognised. Alport
        syndrome is confused when haematuria + CKD co-exist. Ultra-rarity (~25–35 families
        globally) means most clinicians have never seen NPHP17 — systematic WES NPHP panel
        is the only reliable diagnostic route. MAPKBP1 must appear on all comprehensive NPHP
        and renal ciliopathy gene panels.
      </Alert>
      <Alert color={ACCENT}>
        <strong>&#x2705; PURE RENAL NPHP — RENAL TRANSPLANT CURATIVE · NO EXTRA-RENAL MONITORING:</strong> MAPKBP1
        is NOT expressed in retinal photoreceptors (ERG normal — no ophthalmology monitoring),
        biliary epithelium (no CHF — no liver monitoring), cerebellar neurons (no Joubert/MTS),
        nodal cilia (0% situs inversus — no cardiac evaluation required), or brain neurons
        (no intellectual disability). Cell-autonomous TZ defect → NO recurrence post-transplant.
        Excellent graft outcomes. Pre-emptive transplant preferred. NPHP17 is one of the
        cleanest pure renal NPHP phenotypes — focused nephrology management only.
      </Alert>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Cohort (n)" value={ov.cohort_n} color={ACCENT} />
            <KPI label="Median GFR" value={`${ov.median_gfr} ml/min`} color={ACCENT} />
            <KPI label="Median Hb" value={`${ov.median_hb} g/dL`} color={ACCENT6} />
            <KPI label="Median age Dx" value={`${ov.median_age_renal_dx}yr`} color={ACCENT} />
            <KPI label="ESRD/Transplant" value={`${ov.pct_esrd_or_transplant}%`} color={ACCENT5} />
            <KPI label="Situs inversus" value="0%" color={ACCENT} />
            <KPI label="Retinal" value="0%" color={ACCENT} />
            <KPI label="CHF" value="0%" color={ACCENT} />
            <KPI label="Joubert/MTS" value="0%" color={ACCENT} />
            <KPI label="NPHP1 MLPA misdiag" value={`${ov.pct_misdiagnosed_nphp1}%`} color={ACCENT7} />
            <KPI label="ADPKD misdiag" value={`${ov.pct_misdiagnosed_adpkd}%`} color={ACCENT7} />
            <KPI label="Ultra-rare families" value="~25–35" color={ACCENT6} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="&#x1f9ec; MAPKBP1 — JNK Scaffold & NPHP4 Supercomplex (2q13.3)" color={ACCENT6}>
                <div className="small text-muted mb-2">
                  MAPKBP1 (JIP4; ~1,388 aa) is the JNK/MAPK-interacting scaffold protein 4
                  that links stress-kinase signalling to the NPHP4 supercomplex at the
                  transition zone. Ankyrin-repeat domain (protein interaction) + leucine-zipper
                  (dimerisation) + C-terminal scaffold domain (KIF5B/kinesin-1 binding).
                  Loss → JNK dysregulation at TZ → tubular apoptosis/survival imbalance →
                  TIN + corticomedullary cysts → ESRD. Only NPHP gene encoding a JNK scaffold.
                </div>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>MAPKBP1 (also JIP4 / SPAG9)</td></tr>
                    <tr><td className="fw-bold">Location</td><td>2q13.3 (distinct from NPHP1 at 2q13)</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>~1,388 aa · JNK/MAPK scaffold · NPHP4 supercomplex interactor · kinesin-1 binding</td></tr>
                    <tr><td className="fw-bold">Domains</td><td>Ankyrin repeats (N-term): protein interaction · Leucine-zipper: dimerisation · C-term scaffold: KIF5B/kinesin-1 · JNK-binding domain</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*610889</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#616140 (NPHP17)</td></tr>
                    <tr><td className="fw-bold">Key partners</td><td>NPHP4, NPHP1, NPHP8/RPGRIP1L (supercomplex); JNK/SAPK; KIF5B/kinesin-1</td></tr>
                    <tr><td className="fw-bold">Mechanism</td><td>JNK scaffold at TZ → stress-response → tubular survival; loss → apoptosis dysregulation → TIN → ESRD</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/1,000,000–2,000,000; ~25–35 published families (2026; ultra-rare)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive · biallelic LOF</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="&#x1f6a8; NPHP17 Hallmark Features" color={ACCENT}>
                {[
                  ['ONLY NPHP CAUSED BY JNK/MAPK SCAFFOLD LOSS — distinct mechanism from all other NPHP subtypes', ACCENT2,
                   'MAPKBP1/JIP4 scaffolds JNK (c-Jun N-terminal kinase) stress-response signalling at the transition zone via NPHP4. Loss → TZ JNK signal loss → tubular epithelial apoptosis/survival imbalance → progressive TIN + cysts → ESRD. No other NPHP gene encodes a JNK scaffold; unique mechanistic class within the ciliopathy spectrum'],
                  ['2q13.3 ≠ NPHP1 2q13 — chromosome arm trap; NPHP1 MLPA misses MAPKBP1', ACCENT3,
                   'MAPKBP1 at 2q13.3 and NPHP1 at 2q13 are on the same chromosome arm but different genes. Standard NPHP1 290kb MLPA covers NPHP1 deletion only. Any NPHP1-MLPA-negative patient with TIN pattern needs WES including MAPKBP1. 2q13 "normal" result on targeted deletion panel does NOT exclude NPHP17'],
                  [`Pure renal NPHP17 — ESRD median ~14–16yr (adolescent onset)`, ACCENT,
                   'TIN + corticomedullary cysts + concentrating defect (polyuria first). Adolescent-onset pattern slightly later than NPHP1 (~13yr). ESRD median ~14–16yr. Small echogenic kidneys on USS. Renal transplant CURATIVE; NO recurrence (cell-autonomous TZ defect). Pre-emptive transplant preferred'],
                  ['NPHP4 supercomplex — always co-sequence NPHP1/NPHP4/NPHP8', ACCENT4,
                   'MAPKBP1 is a direct NPHP4-interacting protein. Always co-sequence NPHP4, NPHP1, NPHP8/RPGRIP1L when MAPKBP1 variants found. Digenic MAPKBP1+NPHP4 theoretically possible given direct protein interaction. NPHP4 loss alone causes phenotypically identical pure renal NPHP — genotype determines subtype'],
                  ['Ultra-rare — ~25–35 families worldwide (rarest common NPHP subtype)', ACCENT6,
                   'NPHP17 is among the ultra-rare NPHP subtypes. Most paediatric nephrologists will never encounter a case. Systematic comprehensive WES NPHP panel is the only way to diagnose. Registry enrolment (RareCare / EURO-RDI / NPHP consortium) essential for all families. Research collaboration mandatory for genotype-phenotype refinement'],
                  [`NO situs · NO retinal · NO CHF · NO Joubert · NO ID — cleanest NPHP phenotype`, ACCENT,
                   'MAPKBP1 absent from nodal cilia (0% situs inversus), photoreceptors (ERG normal; no retinal monitoring), biliary epithelium (no CHF; no liver monitoring), cerebellar neurons (no Joubert/MTS), brain neurons (no ID). Pure renal ± concentrating defect only. Focused nephrology care; no multi-organ surveillance required'],
                  ['ADPKD and Alport misdiagnosis — bilateral cysts and haematuria confuse workup', ACCENT7,
                   'Bilateral corticomedullary cysts on USS prompt PKD1/PKD2 testing before AR inheritance pattern is recognised. Haematuria + CKD → Alport (COL4A gene) testing before NPHP panel. Ultra-rarity means NPHP17 is not on initial differential. WES NPHP panel after negative NPHP1 MLPA resolves diagnosis'],
                  ['Renal transplant CURATIVE — excellent outcomes, no disease recurrence', ACCENT,
                   'Cell-autonomous TZ defect: donor kidney has functional MAPKBP1 → normal TZ JNK signalling → tubular identity maintained → no TIN in graft. Excellent long-term outcomes post-transplant. Living-related donor evaluation: renal USS + genetic carrier screening (heterozygous carriers have normal kidneys)'],
                ].map(([title, color, sub], i) => (
                  <div key={i} className="mb-2 p-2 rounded" style={{ background: color + '12', borderLeft: `3px solid ${color}` }}>
                    <div className="fw-bold small" style={{ color }}>{title}</div>
                    <div className="text-muted" style={{ fontSize: '0.75em' }}>{sub}</div>
                  </div>
                ))}
              </Section>
            </div>
          </div>

          {/* Sample patients */}
          <Section title={`&#x1f4cb; Sample Patients (first 8 of ${_COHORT_SIZE} · seed=${SEED})`} color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small">
                <thead>
                  <tr style={{ background: ACCENT + '22' }}>
                    <th>ID</th><th>Ethnicity</th><th>CKD Stage</th>
                    <th>Age Dx (yr)</th><th>GFR</th>
                    <th>Kidney USS</th><th>First Symptom</th><th>Misdiagnosis</th>
                  </tr>
                </thead>
                <tbody>
                  {ov.patients.map(p => (
                    <tr key={p.id}>
                      <td><span className="badge" style={{ background: ACCENT }}>{p.id}</span></td>
                      <td style={{ fontSize: '0.72em' }}>{p.ethnicity.split('(')[0].trim()}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.ckd_stage.split('(')[0].trim()}</td>
                      <td>{p.age_renal_dx_yr}</td>
                      <td>{p.gfr_now_ml_min}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.kidney_uss.split('(')[0].trim().slice(0, 28)}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.first_symptom.split('(')[0].trim().slice(0, 28)}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.prior_misdiagnosis.split('(')[0].trim().slice(0, 28)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── Tab 1: Diagnostic Breakdown ── */}
      {tab === 1 && bk && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="CKD Stage / Renal Status Distribution" color={ACCENT}>
              {Object.entries(bk.ckd_stage_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="Kidney Phenotype on USS" color={ACCENT}>
              {Object.entries(bk.kidney_phenotype).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
            <Section title="Urine Osmolality Tiers (Tubular Concentrating Defect)" color={ACCENT}>
              {Object.entries(bk.urine_osmolality_tiers).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Prior Misdiagnosis (NPHP1 MLPA · ADPKD · Alport)" color={ACCENT7}>
              {Object.entries(bk.prior_misdiagnosis).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
            <Section title="First Presenting Symptom" color={ACCENT}>
              {Object.entries(bk.first_symptom_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="Ethnicity Distribution" color={ACCENT6}>
              {Object.entries(bk.ethnicity).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
            <Section title="GFR Slope Tiers (Progression Rate)" color={ACCENT6}>
              {Object.entries(bk.gfr_slope_tiers).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
          </div>
        </div>
      )}

      {/* ── Tab 2: MAPKBP1 Architecture & Genetics ── */}
      {tab === 2 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="&#x1f9ec; MAPKBP1 Architecture, JNK Pathway & NPHP4 Interaction" color={ACCENT4}>
              {df.genetic_architecture && Object.entries(df.genetic_architecture).filter(([k]) => k !== 'key_variants').map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT4}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{typeof v === 'string' ? v : JSON.stringify(v)}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="&#x1f9ea; Key Variants (MAPKBP1 NPHP17)" color={ACCENT}>
              {df.genetic_architecture && df.genetic_architecture.key_variants && df.genetic_architecture.key_variants.map((v, i) => (
                <div key={i} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT}` }}>
                  <div className="small text-muted">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="&#x1f4ca; NPHP Subtype Comparison" color={ACCENT6}>
              {df.nphp_comparison && Object.entries(df.nphp_comparison).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded"
                  style={{
                    background: k.includes('★') ? ACCENT + '18' : ACCENT6 + '08',
                    borderLeft: `3px solid ${k.includes('★') ? ACCENT : ACCENT6}`
                  }}>
                  <div className="fw-bold small" style={{ color: k.includes('★') ? ACCENT : ACCENT6 }}>{k}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-12">
            <Section title="&#x1fa7a; Differential Diagnosis Table" color={ACCENT7}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead style={{ background: ACCENT7 + '18' }}>
                    <tr><th>Condition</th><th>Key Distinguishing Features from NPHP17</th></tr>
                  </thead>
                  <tbody>
                    {df.ddx_table && Object.entries(df.ddx_table).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ color: ACCENT7, minWidth: 200 }}>{k}</td>
                        <td className="small">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* ── Tab 3: Definitions ── */}
      {tab === 3 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Disease Definition" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {['disease','omim_gene','omim_disease','chromosome','inheritance','prevalence'].map(k => (
                    <tr key={k}>
                      <td className="fw-bold text-capitalize" style={{ width: 140 }}>{k.replace(/_/g,' ')}</td>
                      <td>{df[k]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="Mechanism" color={ACCENT2}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT + '06', lineHeight: 1.7 }}>
                {df.mechanism}
              </div>
            </Section>
            <Section title="&#x1f3e5; Treatment" color={ACCENT}>
              {df.treatment && Object.entries(df.treatment).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Key Clinical Features" color={ACCENT3}>
              {df.key_clinical_features && Object.entries(df.key_clinical_features).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT3 + '08', borderLeft: `3px solid ${ACCENT3}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="Diagnostic Criteria" color={ACCENT6}>
              {df.diagnostic_criteria && Object.entries(df.diagnostic_criteria).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT6 + '08', borderLeft: `3px solid ${ACCENT6}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT6 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="Prognosis" color={ACCENT}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT + '06', lineHeight: 1.7 }}>
                {df.prognosis}
              </div>
            </Section>
            <Section title="Cohort Note" color={ACCENT6}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT6 + '06', lineHeight: 1.6 }}>
                {df.cohort_note}
              </div>
            </Section>
          </div>
        </div>
      )}

      <div className="mt-4 pt-3 border-top">
        <Link href="/" className="btn btn-sm btn-outline-secondary me-2">&#x2190; Portal Home</Link>
        <Link href="/nphp16" className="btn btn-sm btn-outline-primary">&#x2190; NPHP16 (ANKS6/IC Scaffold)</Link>
      </div>
    </div>
  );
}
