'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'IC Architecture & Genetics', 'Definitions'];

// NPHP16 colour scheme — ANKS6 / inversin compartment / situs inversus / teal-amber-indigo-slate
const ACCENT  = '#004d40';   // deep teal-green — inversin compartment; IC scaffold platform
const ACCENT2 = '#827717';   // dark amber-olive — situs inversus; left-right axis; nodal cilia
const ACCENT3 = '#1a237e';   // deep indigo — NEK8-ANKS6 phospho-axis; IC kinase module
const ACCENT4 = '#4e342e';   // dark brown — Wnt switch; PCP→canonical β-catenin regulation
const ACCENT5 = '#880e4f';   // dark magenta — ESRD; CKD progression; concentrating defect
const ACCENT6 = '#37474f';   // dark slate — molecular architecture; IC tetramer details
const ACCENT7 = '#e65100';   // burnt orange — misdiagnosis (NPHP1 MLPA / NPHP2 confusion)
const ACCENT8 = '#006064';   // dark teal — IC tetramer co-sequencing; INVS+NPHP3+NEK8

const SEED = 371;
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

export default function NPHP16Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp16/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp16/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp16/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading NPHP16 cohort…</div>;
  if (error)   return <div className="container py-5 text-danger">Error: {error}</div>;
  if (!ov)     return null;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', border: `2px solid ${ACCENT}` }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div style={{ flex: 1, minWidth: 260 }}>
            <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
              &#x1f9ec; Nephronophthisis Type 16 (NPHP16) — ANKS6 / Inversin Compartment Ciliopathy
            </h4>
            <div className="small text-muted mb-1">
              <strong>ANKS6</strong> · 9q22.33 · 982 aa · inversin compartment (IC) scaffold ·
              NEK8 phospho-target (linker aa 541–720) · IC tetramer: INVS+ANKS6+NPHP3+NEK8 ·
              Wnt-pathway switch · situs inversus {ov.pct_situs_inversus}% (IC laterality defect)
            </div>
            <div className="small">
              <Badge text="OMIM *615803" color={ACCENT} />
              <Badge text="#615862 NPHP16" color={ACCENT} />
              <Badge text="AR biallelic LOF" color={ACCENT6} />
              <Badge text="9q22.33" color={ACCENT6} />
              <Badge text="IC scaffold" color={ACCENT3} />
              <Badge text={`Situs inversus ${ov.pct_situs_inversus}%`} color={ACCENT2} />
              <Badge text="NO retinal · NO CHF · NO Joubert" color={ACCENT} />
            </div>
          </div>
          <div className="d-flex gap-2 flex-wrap">
            <span className="badge px-3 py-2" style={{ background: ACCENT, fontSize: '0.8em' }}>
              ESRD median ~13yr
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT2, fontSize: '0.8em' }}>
              Situs inversus {ov.pct_any_laterality}% any laterality
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT3, fontSize: '0.8em' }}>
              NEK8 phospho-target
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT5, fontSize: '0.8em' }}>
              ESRD/Tx {ov.pct_esrd_or_transplant}%
            </span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert color={ACCENT3}>
        <strong>&#x1f9ec; INVERSIN COMPARTMENT (IC) NPHP — DISTINCT FROM TZ-SCAFFOLD AND DAP NPHP:</strong> ANKS6
        localises to the inversin compartment (IC) — a ciliary subdomain proximal to the TZ gate,
        housing the obligate IC tetramer: INVS (NPHP2) · ANKS6 (NPHP16) · NPHP3 · NEK8 (NPHP9).
        NEK8 phosphorylates ANKS6 linker (aa 541–720) → stabilises IC scaffold. IC intact →
        suppresses canonical Wnt/β-catenin in tubular epithelium → maintains tubular identity.
        ANKS6 loss → IC collapse → canonical Wnt up-regulation → tubular EMT → TIN → ESRD.
        Mechanistically distinct from TZ-scaffold NPHP (NPHP1/4/8/15) and DAP NPHP (CEP164/NPHP15).
      </Alert>
      <Alert color={ACCENT2}>
        <strong>&#x21c4; SITUS INVERSUS IN {ov.pct_situs_inversus}% — MOST DIAGNOSTICALLY DISCRIMINATING EXTRA-RENAL FEATURE:</strong> The
        IC in embryonic node cilia controls left-right signalling (PCP-mediated nodal flow).
        ANKS6 loss → incomplete IC laterality defect → situs inversus in ~20–30% (incomplete
        penetrance). Second NPHP subtype (after NPHP2/INVS) with significant situs inversus.
        CRITICAL DDx: situs inversus + CKD onset &gt;3yr → NPHP16 (ANKS6). If onset &lt;3yr →
        NPHP2 (INVS). Document organ situs at diagnosis — mandatory before any surgical procedure.
        Echocardiography mandatory in situs ambiguus (structural CHD possible).
      </Alert>
      <Alert color={ACCENT8}>
        <strong>&#x1f4cb; IC TETRAMER CO-SEQUENCING MANDATORY — ALWAYS SEQUENCE INVS+ANKS6+NPHP3+NEK8:</strong> The
        inversin compartment is a functional unit of four proteins. Digenic IC variants are reported.
        If ANKS6 biallelic confirmed → still sequence INVS (NPHP2), NPHP3, NEK8 (NPHP9) for digenic
        modifiers. If ANKS6 has only ONE pathogenic allele → WES full IC panel mandatory. NEK8 kinase
        loss phenocopies ANKS6 loss (IC collapse via same pathway). NPHP3 adds CHF risk (absent in
        pure NPHP16). Functional IC tetramer integrity = clinical phenotype determinant.
      </Alert>
      <Alert color={ACCENT7}>
        <strong>&#x26a0;&#xfe0f; MOST COMMON MISDIAGNOSIS — NPHP1 MLPA (MISSES 9q22.33) + NPHP2 CONFUSION (WRONG ONSET AGE):</strong> NPHP1
        290kb deletion MLPA is the standard first-line NPHP test → does NOT detect ANKS6 on
        9q22.33. Situs inversus cases are frequently mis-attributed to NPHP2/INVS (infantile onset
        ESRD &lt;3yr, situs inversus &gt;85%) — key discriminator is onset age: NPHP16 juvenile (~13yr).
        Kartagener/PCD is misdiagnosed when situs + renal are both present — absent bronchiectasis
        excludes PCD. WES including ANKS6 is the only reliable test. ANKS6 must appear on all
        NPHP + situs + IC ciliopathy gene panels.
      </Alert>
      <Alert color={ACCENT}>
        <strong>&#x2705; RENAL TRANSPLANT CURATIVE · NO RETINAL · NO CHF — PURE RENAL ± LATERALITY:</strong> Cell-autonomous
        IC defect → NO recurrence in transplanted kidney (donor cells have functional ANKS6 → IC
        intact → tubular identity maintained). Excellent graft outcomes. ANKS6 NOT expressed in
        photoreceptors (ERG normal — no retinal monitoring needed). NOT expressed in biliary
        epithelium (no CHF). NOT expressed in cerebellar neurons (no Joubert/MTS). Not neuronal
        (no ID). NPHP16 = pure renal ciliopathy ± laterality defect — one of the cleanest NPHP phenotypes.
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
            <KPI label="Situs inversus" value={`${ov.pct_situs_inversus}%`} color={ACCENT2} />
            <KPI label="Any laterality" value={`${ov.pct_any_laterality}%`} color={ACCENT2} />
            <KPI label="Retinal" value="0%" color={ACCENT} />
            <KPI label="CHF" value="0%" color={ACCENT} />
            <KPI label="NPHP1 misdiag" value={`${ov.pct_misdiagnosed_nphp1}%`} color={ACCENT7} />
            <KPI label="NPHP2 misdiag" value={`${ov.pct_misdiagnosed_nphp2}%`} color={ACCENT7} />
            <KPI label="Joubert/MTS" value="0%" color={ACCENT} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="&#x1f9ec; ANKS6 — Inversin Compartment Scaffold & NPHP16 (9q22.33)" color={ACCENT6}>
                <div className="small text-muted mb-2">
                  ANKS6 (982 aa; ~110 kDa) is an obligate scaffold of the inversin compartment (IC) —
                  the ciliary subdomain proximal to the transition zone. IC tetramer: INVS (NPHP2) +
                  ANKS6 (NPHP16) + NPHP3 + NEK8 (NPHP9). NEK8 phosphorylates ANKS6 linker (aa 541–720)
                  → maintains IC. IC collapse → Wnt dysregulation → TIN → ESRD. IC in nodal cilia →
                  left-right signalling → situs inversus (20–30% incomplete penetrance).
                </div>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>ANKS6 (also PKDR1)</td></tr>
                    <tr><td className="fw-bold">Location</td><td>9q22.33</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>982 aa · ~110 kDa · inversin compartment (IC) scaffold · 12 ankyrin repeats + SAM domain</td></tr>
                    <tr><td className="fw-bold">Domains</td><td>AnkR (1–540): INVS+NPHP3 binding · Linker (541–720): NEK8 phospho-targets · SAM (721–982): BICC1+oligomerisation</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*615803</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#615862 (NPHP16)</td></tr>
                    <tr><td className="fw-bold">Key partners</td><td>INVS (NPHP2), NPHP3, NEK8 (NPHP9), BICC1 — IC functional tetramer</td></tr>
                    <tr><td className="fw-bold">Mechanism</td><td>IC scaffold → Wnt switch (canonical suppression) → tubular identity; loss → Wnt up → EMT → TIN → ESRD</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/500,000–1,000,000; ~60–80 published families (2026)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive · biallelic LOF</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="&#x1f6a8; NPHP16 Hallmark Features" color={ACCENT}>
                {[
                  ['INVERSIN COMPARTMENT (IC) NPHP — distinct from TZ-scaffold and DAP NPHP', ACCENT3,
                   'ANKS6 localises to IC (proximal TZ subdomain). IC tetramer: INVS+ANKS6+NPHP3+NEK8. NEK8 phospho-maintains IC. IC controls Wnt switch → tubular identity. Loss → canonical Wnt → EMT → TIN + cysts → ESRD. Mechanistically distinct from all other NPHP subtypes'],
                  [`Situs inversus — ${ov.pct_situs_inversus}% totalis + ${ov.pct_situs_ambiguus}% ambiguus (any laterality ${ov.pct_any_laterality}%)`, ACCENT2,
                   'IC in embryonic node cilia controls left-right axis (PCP-mediated nodal flow). ANKS6 loss → incomplete laterality defect → ~20–30% situs inversus totalis; ~5% ambiguus. Second NPHP subtype with significant laterality defects (after NPHP2 >85%). Onset age discriminates from NPHP2 (ESRD <3yr vs ~13yr)'],
                  [`NPHP16 renal phenotype — ESRD median ~13yr`, ACCENT,
                   'TIN + corticomedullary cysts + concentrating defect (polyuria first). Small echogenic kidneys on USS. Similar to NPHP1 (~13yr) and NPHP15 (~13–15yr). Renal transplant CURATIVE; NO recurrence (cell-autonomous IC defect in native kidneys)'],
                  ['IC tetramer co-sequencing — INVS+ANKS6+NPHP3+NEK8 mandatory', ACCENT8,
                   'IC is a functional unit. Always sequence all four IC genes together. Digenic IC variants reported. NEK8 loss phenocopies ANKS6 loss. NPHP3 co-variant adds CHF risk. If ANKS6 single allele: WES complete IC panel mandatory. BICC1 (SAM-domain partner) on extended panel'],
                  ['NEK8 phosphorylates ANKS6 — IC kinase-scaffold axis', ACCENT3,
                   'NEK8 (NPHP9) phosphorylates ANKS6 linker region (Ser residues, aa 541–720) → stabilises IC scaffold integrity. NEK8 kinase-dead phenocopies ANKS6 loss. Linker missense variants in ANKS6 may disrupt NEK8 phospho-site → IC collapse without complete domain loss'],
                  [`NO retinal · NO CHF · NO Joubert · NO ID — KEY DDx`, ACCENT,
                   'ANKS6 absent from photoreceptors (ERG normal; no retinal monitoring), biliary epithelium (no CHF), cerebellar neurons (no MTS/Joubert), brain neurons (no ID). Situs inversus + juvenile CKD + normal ERG + no CHF → NPHP16. Retinal/CHF presence → reconsider NPHP5/6/10/15 or NPHP3/11'],
                  ['NPHP1 MLPA + NPHP2 confusion — most common misdiagnoses', ACCENT7,
                   'NPHP1 290kb MLPA (standard first-line) does NOT detect ANKS6 9q22.33. NPHP2/INVS confused when situs inversus present — discriminate by ONSET AGE (NPHP2 infantile ESRD <3yr; NPHP16 juvenile ~13yr). Kartagener/PCD excluded by absent bronchiectasis/sinusitis. WES mandatory'],
                  ['Renal transplant CURATIVE — pure renal ± laterality, excellent outcomes', ACCENT,
                   'Cell-autonomous IC defect; NO recurrence post-transplant. Donor cells have functional ANKS6 → IC intact → tubular identity maintained → no TIN in graft. Living-related donors must have genetic screening (carrier evaluation). Pre-emptive transplant preferred when feasible'],
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
                    <th>Situs</th><th>First Symptom</th><th>Misdiagnosis</th>
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
                        {p.situs_inversus
                          ? <span className="badge" style={{ background: ACCENT2 }}>SI ✓</span>
                          : p.situs_ambiguus
                          ? <span className="badge" style={{ background: ACCENT7 }}>Amb ✓</span>
                          : <span className="text-muted small">solitus</span>}
                      </td>
                      <td style={{ fontSize: '0.72em' }}>{p.first_symptom.split('(')[0].trim().slice(0, 30)}</td>
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
            <Section title="Situs / Laterality Distribution (IC nodal cilia defect)" color={ACCENT2}>
              {Object.entries(bk.situs_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
              ))}
            </Section>
            <Section title="Kidney Phenotype on USS" color={ACCENT}>
              {Object.entries(bk.kidney_phenotype).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Prior Misdiagnosis (NPHP1 MLPA / NPHP2 most common)" color={ACCENT7}>
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
          </div>
          <div className="col-12">
            <div className="row g-3">
              <div className="col-md-6">
                <Section title="Urine Osmolality Tiers (Tubular Concentrating Defect)" color={ACCENT}>
                  {Object.entries(bk.urine_osmolality_tiers).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                  ))}
                </Section>
              </div>
              <div className="col-md-6">
                <Section title="GFR Slope Tiers (Progression Rate)" color={ACCENT6}>
                  {Object.entries(bk.gfr_slope_tiers).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                  ))}
                </Section>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 2: IC Architecture & Genetics ── */}
      {tab === 2 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="&#x1f9ec; ANKS6 Inversin Compartment Architecture & NEK8 Mechanism" color={ACCENT3}>
              {df.genetic_architecture && Object.entries(df.genetic_architecture).filter(([k]) => k !== 'key_variants').map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT3}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{typeof v === 'string' ? v : JSON.stringify(v)}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="&#x1f9ea; Key Variants (ANKS6 NPHP16)" color={ACCENT}>
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
                    <tr><th>Condition</th><th>Key Distinguishing Features from NPHP16</th></tr>
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
            <Section title="Mechanism" color={ACCENT3}>
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
            <Section title="Key Clinical Features" color={ACCENT2}>
              {df.key_clinical_features && Object.entries(df.key_clinical_features).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT2 + '08', borderLeft: `3px solid ${ACCENT2}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</div>
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
        <Link href="/nphp15" className="btn btn-sm btn-outline-primary">&#x2190; NPHP15 (CEP164/SLS)</Link>
      </div>
    </div>
  );
}
