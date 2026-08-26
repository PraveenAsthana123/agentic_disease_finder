'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Genetics & Allele Spectrum', 'Definitions'];

// NPHP12 colour scheme — TTC21B/IFT139; IFT-A retrograde complex; indigo-slate
const ACCENT  = '#1a237e';   // deep indigo — IFT-A retrograde complex; TTC21B; pure renal identity
const ACCENT2 = '#1b5e20';   // deep green — renal transplant curative / IFT-A cell-autonomous
const ACCENT3 = '#b71c1c';   // deep crimson — Jeune/ATD4 thoracic dystrophy (severe alleles)
const ACCENT4 = '#004d40';   // dark teal — IFT-A retrograde complex biochemistry; dynein-2
const ACCENT5 = '#827717';   // dark amber-olive — allele spectrum: hypomorphic → NPHP12; null → ATD4
const ACCENT6 = '#37474f';   // dark slate — IFT-A interactome; WDR19; ciliary tip IFT-plug
const ACCENT7 = '#880e4f';   // dark magenta — ADPKD misdiagnosis (dominant inheritance assumed)
const ACCENT8 = '#4a148c';   // deep purple — IFT retrograde mechanism; Hedgehog pathway imbalance

const SEED = 363;
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

export default function NPHP12Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp12/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp12/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp12/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading NPHP12 cohort…</div>;
  if (error)   return <div className="container py-5 text-danger">Error: {error}</div>;
  if (!ov)     return null;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', border: `2px solid ${ACCENT}` }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div style={{ flex: 1, minWidth: 260 }}>
            <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
              &#x1f9ec; Nephronophthisis Type 12 (NPHP12) — TTC21B / IFT139 / IFT-A Retrograde Complex
            </h4>
            <div className="small text-muted mb-1">
              <strong>TTC21B</strong> · 2q24.3 · 1,317 aa · IFT139 · TPR-repeat protein ·
              IFT-A retrograde complex subunit · pure renal ciliopathy ·
              Jeune/ATD4 thoracic dystrophy (biallelic null) · only NPHP caused by IFT-A deficiency
            </div>
            <div className="small">
              <Badge text="OMIM *612014" color={ACCENT} />
              <Badge text="#613820 NPHP12" color={ACCENT} />
              <Badge text="#611263 ATD4/Jeune" color={ACCENT3} />
              <Badge text="AR biallelic LOF" color={ACCENT6} />
              <Badge text="2q24.3" color={ACCENT6} />
              <Badge text="IFT-A retrograde" color={ACCENT4} />
              <Badge text="NO CHF · NO Joubert · NO retinal" color={ACCENT2} />
            </div>
          </div>
          <div className="d-flex gap-2 flex-wrap">
            <span className="badge px-3 py-2" style={{ background: ACCENT, fontSize: '0.8em' }}>
              ESRD median ~11–15yr
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT3, fontSize: '0.8em' }}>
              Jeune/ATD4 {ov.pct_atd4_jeune}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT5, fontSize: '0.8em' }}>
              Skeletal {ov.pct_skeletal_involvement}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT2, fontSize: '0.8em' }}>
              Retinal 0–8%
            </span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert color={ACCENT7}>
        <strong>&#x26a0;&#xfe0f; MOST COMMON MISDIAGNOSIS — ADPKD (autosomal dominant PKD1/PKD2):</strong> Cystic
        echogenic kidneys → dominant inheritance assumed → PKD1/PKD2 tested → NPHP12 missed as AR pattern
        overlooked. KEY DDx: ADPKD = enlarged kidneys + adult dominant family history; NPHP12 = small echogenic +
        AR + juvenile ESRD + concentrating defect. TTC21B MUST be on ALL NPHP extended panels — standard NPHP1
        MLPA (290kb deletion) does NOT detect TTC21B mutations; WES mandatory.
      </Alert>
      <Alert color={ACCENT3}>
        <strong>&#x1f9b4; JEUNE / ATD4 SPECTRUM — BIALLELIC NULL TTC21B ({ov.pct_atd4_jeune}% of cohort):</strong> Narrow
        thorax + short ribs + shortened limbs + polydactyly (~12%) → neonatal/infantile respiratory failure.
        Allele classification CRITICAL: null × null → ATD4; hypomorphic × null → NPHP12 pure; hypomorphic ×
        hypomorphic → mildest NPHP12. VEPTR/MAGEC thoracic expansion for survivors. Annual CXR in ALL NPHP12.
      </Alert>
      <Alert color={ACCENT4}>
        <strong>&#x1f504; IFT-A RETROGRADE — UNIQUE MECHANISM AMONG NPHP1–12:</strong> NPHP12 is the ONLY NPHP
        subtype caused by IFT-A retrograde complex loss. IFT-B anterograde particles accumulate at ciliary tip →
        IFT-plug (ultrastructure on TEM) → impaired Hedgehog/Gli3 recycling → TIN + ESRD. Direct IFT-A binding
        partner: WDR19/IFT144 (NPHP13) — always co-sequence WDR19 when TTC21B found; digenic TTC21B/WDR19
        heterozygosity can cause ciliopathy phenotype.
      </Alert>
      <Alert color={ACCENT5}>
        <strong>&#x1f9ec; ALLELE SPECTRUM — HYPOMORPHIC vs NULL:</strong> p.Ala428Val (most common; gnomAD
        carrier ~1/600) is a hypomorphic allele → pure NPHP12 when compound het with null. Biallelic null
        (nonsense/frameshift/deletion) → ATD4/Jeune. Allele classification drives prognosis, respiratory
        management, and PGT-M risk counselling. Standard NPHP1 MLPA misses TTC21B — WES + CNV (2q24.3) mandatory.
      </Alert>
      <Alert color={ACCENT2}>
        <strong>&#x2705; RENAL TRANSPLANT = CURATIVE:</strong> Cell-autonomous IFT-A retrograde defect. NO
        recurrence. Excellent graft outcomes. ATD4 respiratory management enables survival to transplantable age
        in Jeune patients with modern VEPTR/MAGEC thoracic expansion.
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
            <KPI label="ESRD/Transplant" value={`${ov.pct_esrd_or_transplant}%`} color={ACCENT} />
            <KPI label="Skeletal (any)" value={`${ov.pct_skeletal_involvement}%`} color={ACCENT3} />
            <KPI label="ATD4/Jeune" value={`${ov.pct_atd4_jeune}%`} color={ACCENT3} />
            <KPI label="Retinal (minor)" value={`${ov.pct_retinal_involvement}%`} color={ACCENT2} />
            <KPI label="ADPKD misdiag" value={`${ov.pct_misdiagnosed_as_adpkd}%`} color={ACCENT7} />
            <KPI label="Polyuria first" value={`${ov.pct_polyuria_first_symptom}%`} color={ACCENT6} />
            <KPI label="CHF" value="0%" color={ACCENT2} />
            <KPI label="Joubert" value="0%" color={ACCENT2} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="&#x1f9ec; TTC21B — IFT-A Retrograde Complex (IFT139)" color={ACCENT6}>
                <div className="small text-muted mb-2">
                  TTC21B (IFT139; 1,317 aa) contains 18 tetratricopeptide repeat (TPR) motifs arranged in a
                  right-handed superhelix, forming the structural backbone of the IFT-A retrograde complex.
                  The IFT-A complex (IFT144/WDR19 + IFT140 + IFT122 + IFT139/TTC21B + IFT43) mediates
                  retrograde transport from the ciliary tip to the basal body via dynein-2 (DYNC2H1).
                  Loss of TTC21B → retrograde failure → IFT-B particle accumulation → IFT-plug →
                  Hedgehog/Gli3 imbalance → TIN → ESRD.
                </div>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>TTC21B (IFT139 / NPHPF1 / THM1)</td></tr>
                    <tr><td className="fw-bold">Location</td><td>2q24.3</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>1,317 aa · ~148 kDa · 18× TPR superhelix · IFT-A core subunit</td></tr>
                    <tr><td className="fw-bold">Domains</td><td>N-terminal dimerisation (1–100) + 18-TPR superhelix (101–1,000) + C-terminal dynein-2/cargo-adaptor (1,001–1,317)</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*612014</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#613820 (NPHP12) · #611263 (ATD4 / Jeune Syndrome 4)</td></tr>
                    <tr><td className="fw-bold">IFT-A partners</td><td>WDR19 (IFT144/NPHP13), IFT140 (WDPCP), IFT122 (WDR10), IFT43 (C14orf179)</td></tr>
                    <tr><td className="fw-bold">Motor</td><td>Dynein-2 (DYNC2H1 + DYNC2LI1 + WDR34 + WDR60)</td></tr>
                    <tr><td className="fw-bold">Mechanism</td><td>Retrograde IFT failure → IFT-B anterograde accumulation → IFT-plug at ciliary tip → Hedgehog/Gli3 imbalance → TIN + ESRD</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/500,000–1,000,000 (NPHP12 pure renal)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive · biallelic LOF</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="&#x1f6a8; NPHP12 Hallmark Features" color={ACCENT}>
                {[
                  ['Pure renal (83–85%) — NPHP12 phenotype', ACCENT,
                   'TIN + corticomedullary cysts + concentrating defect; ESRD median ~11–15yr; small echogenic kidneys; NO extra-renal (most)'],
                  [`ATD4/Jeune thoracic dystrophy (${ov.pct_atd4_jeune}%) — biallelic null alleles`, ACCENT3,
                   'Narrow thorax + short ribs + shortened limbs + polydactyly (12%); neonatal/infantile respiratory failure; VEPTR/MAGEC thoracic expansion; annual CXR in ALL NPHP12'],
                  ['ONLY NPHP caused by IFT-A retrograde complex loss', ACCENT4,
                   'Retrograde IFT failure → IFT-B anterograde accumulation at ciliary tip → IFT-plug ultrastructure → Hedgehog/Gli3 imbalance; distinct from TZ scaffold, photoreceptor-CC, centrosomal NPHP subtypes'],
                  ['p.Ala428Val — most common NPHP12 hypomorphic allele', ACCENT5,
                   'c.1283C>T; pan-ethnic; gnomAD European carrier ~1/600; 18% of NPHP12 alleles; partial IFT-A function retained; pure NPHP12 when compound het with null'],
                  ['NO retinal dystrophy in NPHP12 (ERG normal >92%)', ACCENT2,
                   'TTC21B not expressed in photoreceptors at critical threshold; critical negative feature distinguishing from NPHP5 (100% retinal), NPHP6 (65%), NPHP10 (57%)'],
                  ['NO CHF, NO Joubert, NO situs, NO pancreatic', ACCENT2,
                   'TTC21B absent from biliary epithelium + cerebellar vermis + nodal cilia + pancreatic ducts; simplifies extra-renal workup vs NPHP2/3/9/11'],
                  ['WDR19 (IFT144/NPHP13) — direct IFT-A binding partner', ACCENT6,
                   'TTC21B TPR domain contacts WDR19 β-propeller; always co-sequence WDR19 when TTC21B found; digenic TTC21B/WDR19 heterozygosity causes ciliopathy; CED spectrum if WDR19 biallelic'],
                  ['Renal transplant CURATIVE', ACCENT2,
                   'Cell-autonomous IFT-A retrograde defect; NO recurrence; excellent outcomes; ATD4 respiratory management enables survival to transplantable age'],
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
                    <th>Skeletal</th><th>Retinal</th><th>First Symptom</th>
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
                      <td>
                        {p.atd4_jeune
                          ? <span className="badge" style={{ background: ACCENT3 }}>ATD4 ✓</span>
                          : p.skeletal_involvement
                          ? <span className="badge" style={{ background: ACCENT5 }}>mild ✓</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                      <td>
                        {p.retinal_involvement
                          ? <span className="badge" style={{ background: ACCENT8 }}>ERG ✓</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                      <td style={{ fontSize: '0.72em' }}>{p.first_symptom.split('(')[0].trim().slice(0, 30)}</td>
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
            <Section title="Kidney Phenotype Distribution" color={ACCENT}>
              {Object.entries(bk.kidney_phenotype_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="Skeletal Status Distribution (ATD4 / Jeune)" color={ACCENT3}>
              {Object.entries(bk.skeletal_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
            <Section title="Retinal Status Distribution" color={ACCENT8}>
              {Object.entries(bk.retinal_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="CKD Stage Distribution" color={ACCENT6}>
              {Object.entries(bk.ckd_stage_current).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
            <Section title="RRT / Transplant Status" color={ACCENT2}>
              {Object.entries(bk.rrt_transplant_status).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
              ))}
            </Section>
            <Section title="Prior Misdiagnosis (most common: ADPKD)" color={ACCENT7}>
              {Object.entries(bk.prior_misdiagnosis).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
            <Section title="Age at Renal Diagnosis — Tiers" color={ACCENT}>
              {Object.entries(bk.age_at_renal_dx_tiers).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
          </div>
          <div className="col-12">
            <div className="row g-3">
              <div className="col-md-4">
                <Section title="Ethnicity Distribution" color={ACCENT6}>
                  {Object.entries(bk.ethnicity).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                  ))}
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="Urine Osmolality Tiers (Tubular Defect)" color={ACCENT}>
                  {Object.entries(bk.urine_osmolality_tiers).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                  ))}
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="GFR Slope Tiers (Progression Rate)" color={ACCENT6}>
                  {Object.entries(bk.gfr_slope_tiers).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                  ))}
                </Section>
              </div>
            </div>
          </div>
          <div className="col-12">
            <Section title="First Presenting Symptom" color={ACCENT}>
              <div className="row g-2">
                {Object.entries(bk.first_symptom_distribution).map(([k, v]) => (
                  <div key={k} className="col-md-6">
                    <Bar label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                  </div>
                ))}
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* ── Tab 2: Genetics & Allele Spectrum ── */}
      {tab === 2 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="&#x1f9ec; IFT-A Complex Architecture & TTC21B TPR Domain" color={ACCENT6}>
              {df.genetic_architecture && Object.entries(df.genetic_architecture).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="&#x1f9ea; Key Variants (TTC21B NPHP12 / ATD4)" color={ACCENT3}>
              {df.key_variants && df.key_variants.map((v, i) => (
                <div key={i} className="mb-2 p-2 rounded" style={{ background: ACCENT3 + '08', borderLeft: `3px solid ${ACCENT3}` }}>
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
                    <tr><th>Condition</th><th>Key Distinguishing Features from NPHP12</th></tr>
                  </thead>
                  <tbody>
                    {df.ddx_table && Object.entries(df.ddx_table).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ color: ACCENT7, minWidth: 160 }}>{k}</td>
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
            <Section title="Mechanism" color={ACCENT8}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT + '06', lineHeight: 1.7 }}>
                {df.mechanism}
              </div>
            </Section>
            <Section title="&#x1f3e5; Treatment" color={ACCENT2}>
              {df.treatment && Object.entries(df.treatment).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT2 + '08', borderLeft: `3px solid ${ACCENT2}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</div>
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
            <Section title="Diagnostic Criteria" color={ACCENT4}>
              {df.diagnostic_criteria && Object.entries(df.diagnostic_criteria).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT4 + '08', borderLeft: `3px solid ${ACCENT4}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}</div>
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
        <Link href="/nphp11" className="btn btn-sm btn-outline-primary">&#x2190; NPHP11 (TMEM67)</Link>
      </div>
    </div>
  );
}
