'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Genetics & DDR Mechanism', 'Definitions'];

// NPHP14 colour scheme — ZNF423/EBFAZ; nuclear DDR protein; JBTS19; zinc fingers; dark-walnut + navy + teal
const ACCENT  = '#3e2723';   // dark walnut — ZNF423 nuclear zinc fingers; DDR uniqueness; grounding colour
const ACCENT2 = '#1a237e';   // deep indigo — JBTS19 cerebellar/Joubert; MTS; brain MRI
const ACCENT3 = '#1b5e20';   // deep forest green — NPHP14 renal; transplant curative
const ACCENT4 = '#4a148c';   // deep purple — DDR pathway (ATM-PARP1); unique mechanism
const ACCENT5 = '#880e4f';   // dark magenta — ESRD; CKD progression
const ACCENT6 = '#37474f';   // dark slate — molecular details; complex architecture
const ACCENT7 = '#e65100';   // burnt orange — misdiagnosis (NPHP1/CEP290 first)
const ACCENT8 = '#006064';   // dark teal — CEP290 direct binding partner; BBS10/BBS12 transcription

const SEED = 367;
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

export default function NPHP14Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp14/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp14/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp14/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading NPHP14 cohort…</div>;
  if (error)   return <div className="container py-5 text-danger">Error: {error}</div>;
  if (!ov)     return null;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', border: `2px solid ${ACCENT}` }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div style={{ flex: 1, minWidth: 260 }}>
            <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
              &#x1f9ec; Nephronophthisis Type 14 / Joubert Syndrome 19 (NPHP14 / JBTS19) — ZNF423 / EBFAZ / OAZ
            </h4>
            <div className="small text-muted mb-1">
              <strong>ZNF423</strong> · 16q12.1 · 1,284 aa · 30 C2H2 zinc fingers · nuclear DDR protein ·
              ONLY NPHP caused by a DNA damage response protein · ciliogenesis via ATM-PARP1 centrosomal DDR ·
              CEP290 direct binding partner · BBS10/BBS12 transcription activator
            </div>
            <div className="small">
              <Badge text="OMIM *604085" color={ACCENT} />
              <Badge text="#614844 NPHP14/JBTS19" color={ACCENT} />
              <Badge text="AR biallelic LOF" color={ACCENT6} />
              <Badge text="16q12.1" color={ACCENT6} />
              <Badge text="DDR nuclear protein" color={ACCENT4} />
              <Badge text="JBTS19 MTS 40–50%" color={ACCENT2} />
              <Badge text="NO retinal · NO CHF · NO situs" color={ACCENT3} />
            </div>
          </div>
          <div className="d-flex gap-2 flex-wrap">
            <span className="badge px-3 py-2" style={{ background: ACCENT, fontSize: '0.8em' }}>
              ESRD median ~13–18yr
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT2, fontSize: '0.8em' }}>
              JBTS19 full {ov.pct_jbts19_full}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT5, fontSize: '0.8em' }}>
              ID impairment {ov.pct_intellectual_impairment}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT3, fontSize: '0.8em' }}>
              No retinal {100 - ov.pct_retinal_involvement}%
            </span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert color={ACCENT4}>
        <strong>&#x1f9ec; ONLY NPHP CAUSED BY A DNA DAMAGE RESPONSE PROTEIN — ZNF423/ATM-PARP1:</strong> All other
        NPHP genes encode structural ciliary/TZ/centrosomal components. ZNF423 is a nuclear transcriptional
        regulator that promotes ciliogenesis by resolving centrosomal DNA damage via ATM-PARP1 signalling in
        quiescent (G0) cells. ZNF423 null → persistent γH2AX foci at centrosomes → ciliogenesis initiation failure
        → NPHP14. Mechanistically unique among all &gt;20 NPHP subtypes. WES mandatory — standard NPHP panels
        without ZNF423 will miss this diagnosis.
      </Alert>
      <Alert color={ACCENT2}>
        <strong>&#x1f9e0; JOUBERT SYNDROME 19 (JBTS19) IN 40–50%:</strong> Molar Tooth Sign on axial MRI +
        cerebellar vermis hypoplasia + oculomotor apraxia (OMA; head-thrusting) + ataxia + intellectual disability.
        Brain MRI MANDATORY at diagnosis for all ZNF423 biallelic confirmed patients. Biallelic truncating alleles =
        full JBTS19; hypomorphic alleles = pure NPHP14 (no MTS). Cerebellar features do NOT improve post-transplant —
        cell-autonomous neuronal DDR defect. Developmental paediatrics + special education referral immediately.
      </Alert>
      <Alert color={ACCENT8}>
        <strong>&#x1f517; CEP290 DIRECT BINDING PARTNER — SEQUENCE BOTH:</strong> ZNF423 directly binds CEP290
        (NPHP6 gene) at the centrosome/TZ — the only NPHP gene pair with direct protein-protein interaction
        as a functional unit. CEP290 is the most common Joubert gene (40% of JBTS) and is typically tested
        first → ZNF423 missed. KEY DISTINGUISHER: NPHP14 has NO retinal dystrophy (ZNF423 not in photoreceptors);
        CEP290/NPHP6 has 65% retinal dystrophy (LCA-like). Always co-sequence CEP290 when ZNF423 found.
        ZNF423 also activates BBS10/BBS12 transcription (partial BBSome dysfunction without full BBS phenotype).
      </Alert>
      <Alert color={ACCENT7}>
        <strong>&#x26a0;&#xfe0f; MOST COMMON MISDIAGNOSIS — NPHP1 DELETION MLPA:</strong> NPHP1 MLPA
        (290kb deletion test) is the standard first-line test for suspected NPHP → does NOT detect ZNF423
        on 16q12.1. NPHP1 has NO Joubert MTS (critical: MLPA-negative + Joubert → immediately order WES,
        not sequential single-gene testing). CEP290 tested second on JBTS panels → ZNF423 missed if
        no retinal and CEP290 negative. WES is the only reliable test for NPHP14 diagnosis.
      </Alert>
      <Alert color={ACCENT3}>
        <strong>&#x1f6ab; NO RETINAL DYSTROPHY — KEY DDx vs CEP290/NPHP6:</strong> ZNF423 is not expressed
        in photoreceptors at disease-relevant levels → ERG normal in &gt;93% of NPHP14 patients.
        This is the critical differentiator from CEP290/NPHP6 (65% retinal dystrophy). If rod-cone changes
        are found in a suspected NPHP14 patient, consider digenic CEP290 mutation or alternative diagnosis.
        NO CHF (ZNF423 absent biliary), NO situs inversus (absent nodal cilia), NO ectodermal features
        (absent hair/nail — unlike NPHP13/CED1).
      </Alert>
      <Alert color={ACCENT5}>
        <strong>&#x2705; RENAL TRANSPLANT = CURATIVE FOR NPHP14:</strong> Cell-autonomous DDR defect in
        native kidneys — NO recurrence in transplanted kidney. Excellent graft outcomes.
        JBTS19 neurological features (cerebellar, intellectual disability, OMA) are INDEPENDENT of renal outcome
        — transplant does NOT cure or improve these. JBTS19 neurology needs lifelong independent management:
        physiotherapy, OT, speech therapy, special education, disability support.
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
            <KPI label="JBTS19 full" value={`${ov.pct_jbts19_full}%`} color={ACCENT2} />
            <KPI label="JBTS19 any" value={`${ov.pct_jbts19_any}%`} color={ACCENT2} />
            <KPI label="Intellectual Dx" value={`${ov.pct_intellectual_impairment}%`} color={ACCENT5} />
            <KPI label="Retinal" value={`${ov.pct_retinal_involvement}%`} color={ACCENT3} />
            <KPI label="CHF" value={`${ov.pct_chf_involvement}%`} color={ACCENT3} />
            <KPI label="NPHP1 misdiag" value={`${ov.pct_misdiagnosed_as_nphp1}%`} color={ACCENT7} />
            <KPI label="Polyuria first" value={`${ov.pct_polyuria_first}%`} color={ACCENT6} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="&#x1f9ec; ZNF423 — Nuclear DDR Protein & NPHP14 (16q12.1)" color={ACCENT6}>
                <div className="small text-muted mb-2">
                  ZNF423 / EBFAZ / OAZ (1,284 aa; 30 C2H2-type Krüppel zinc fingers in 4 clusters) is a
                  nuclear transcriptional regulator that promotes ciliogenesis via the DNA damage response (DDR):
                  ZNF423 resolves centrosomal DNA damage in quiescent cells via ATM-PARP1 signalling, enabling
                  basal body maturation and axoneme nucleation. Loss → persistent centrosomal γH2AX → ciliogenesis
                  block → NPHP14 TIN. ZNF423 also directly binds CEP290 (NPHP6) and activates BBS10/BBS12 transcription.
                </div>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>ZNF423 (EBFAZ / OAZ / ROAZ / ZNF467p)</td></tr>
                    <tr><td className="fw-bold">Location</td><td>16q12.1</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>1,284 aa · ~145 kDa · 30 C2H2 Krüppel zinc fingers · 4 clusters · nuclear protein</td></tr>
                    <tr><td className="fw-bold">Domains</td><td>ZF1 (aa 91–250): Smad2/3 + EBF1 · ZF2 (251–500): SMAD4 + ROR2 · Central (501–700): RAR/RXR + p300 · ZF3 (701–950): CEP290 binding + centrosomal · ZF4 (951–1,284): BBS10/12 + ATM-PARP1-DDR</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*604085</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#614844 (NPHP14 / JBTS19 — same entry)</td></tr>
                    <tr><td className="fw-bold">Key partners</td><td>CEP290 (NPHP6; direct binding ZF3), PARP1 (DDR), ATM (DDR kinase), BBS10 + BBS12 (transcription)</td></tr>
                    <tr><td className="fw-bold">Mechanism</td><td>Centrosomal DDR (ATM-PARP1) + BBS10/12 transcription → ciliogenesis; loss → γH2AX persistence → NPHP14</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/700,000–1,500,000; ~60–80 published families (2026)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive · biallelic LOF</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="&#x1f6a8; NPHP14/JBTS19 Hallmark Features" color={ACCENT}>
                {[
                  ['ONLY NPHP CAUSED BY DNA DAMAGE RESPONSE PROTEIN', ACCENT4,
                   'ZNF423 resolves centrosomal DNA damage in G0 cells via ATM-PARP1; loss → persistent γH2AX at centrosome → ciliogenesis failure → NPHP14. All other NPHP genes encode structural ciliary/TZ/centrosomal components — ZNF423 is the UNIQUE DDR mechanism'],
                  [`JOUBERT SYNDROME 19 (JBTS19) — ${ov.pct_jbts19_full}% full MTS`, ACCENT2,
                   'Molar Tooth Sign (MTS) + cerebellar vermis hypoplasia + OMA + ataxia + intellectual disability. Biallelic truncating alleles. Brain MRI mandatory. Cerebellar features do NOT improve with renal transplant. Highest JBTS rate among NPHP subtypes lacking retinal dystrophy'],
                  [`Intellectual disability (${ov.pct_intellectual_impairment}%) — JBTS19 correlation`, ACCENT5,
                   'Mild-to-moderate ID in 25–35%; correlates with JBTS19 alleles and MTS severity. Special education + developmental paediatrics + OT + speech therapy. Cognitive outcome independent of renal transplant — lifelong support needed'],
                  [`NPHP14 renal phenotype — ESRD median ~13–18yr`, ACCENT3,
                   'TIN + corticomedullary cysts + concentrating defect (polyuria first). Small echogenic kidneys. Later ESRD than NPHP12 (~11–15yr). Renal transplant CURATIVE; NO recurrence (cell-autonomous DDR defect)'],
                  ['CEP290 (NPHP6) direct binding partner — co-sequence mandatory', ACCENT8,
                   'ZNF423 ZF cluster 3 directly binds CEP290 — only functional NPHP gene pair with direct protein contact. NPHP14 and NPHP6 phenocopies (both JBTS) but KEY DDx: NPHP6 has 65% retinal dystrophy; NPHP14 has NONE. Always co-sequence CEP290 when ZNF423 found'],
                  [`NO retinal (${100 - ov.pct_retinal_involvement}% normal ERG) — KEY DDx vs CEP290`, ACCENT3,
                   'ZNF423 absent in photoreceptors → ERG normal >93%. Critical differentiator from CEP290 (65% retinal) and SDCCAG8 (50–60% retinal). If ERG abnormal → consider digenic CEP290. Absence of retinal + Joubert = think ZNF423 (not CEP290)'],
                  ['BBS10/BBS12 transcription activator — partial BBSome link', ACCENT8,
                   'ZNF423 ZF cluster 4 transcriptionally activates BBS10 and BBS12 (peripheral BBSome subunits). ZNF423 null → partial BBSome dysfunction → ciliary signalling defect. No full BBS phenotype (no obesity, no polydactyly) — distinguishes NPHP14 from BBS'],
                  ['Renal transplant CURATIVE for NPHP14 — JBTS19 neurology independent', ACCENT3,
                   'Cell-autonomous DDR defect; NO recurrence in transplant. JBTS19 neurological features (cerebellar, ID, OMA) are INDEPENDENT of renal outcome — transplant does NOT improve cerebellar hypoplasia or intellectual disability'],
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
                    <th>JBTS19</th><th>Intellect</th><th>First Symptom</th>
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
                        {p.jbts19_full
                          ? <span className="badge" style={{ background: ACCENT2 }}>MTS ✓</span>
                          : p.jbts19_any
                          ? <span className="badge" style={{ background: ACCENT5 }}>partial ✓</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                      <td>
                        {p.intellectual_impairment
                          ? <span className="badge" style={{ background: ACCENT5 }}>ID ✓</span>
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
            <Section title="CKD Stage / Renal Status Distribution" color={ACCENT}>
              {Object.entries(bk.ckd_stage_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="JBTS19 / Joubert Syndrome Distribution" color={ACCENT2}>
              {Object.entries(bk.jbts19_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
              ))}
            </Section>
            <Section title="Intellectual Disability Distribution" color={ACCENT5}>
              {Object.entries(bk.intellectual_dx_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Retinal Status Distribution (expected: &gt;93% normal)" color={ACCENT3}>
              {Object.entries(bk.retinal_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
            <Section title="CHF / Hepatic Status Distribution (expected: &gt;96% absent)" color={ACCENT3}>
              {Object.entries(bk.chf_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
            <Section title="Prior Misdiagnosis (most common: NPHP1 MLPA / CEP290)" color={ACCENT7}>
              {Object.entries(bk.prior_misdiagnosis).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
            <Section title="First Presenting Symptom" color={ACCENT}>
              {Object.entries(bk.first_symptom_distribution).map(([k, v]) => (
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
        </div>
      )}

      {/* ── Tab 2: Genetics & DDR Mechanism ── */}
      {tab === 2 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="&#x1f9ec; ZNF423 Zinc Finger Architecture & DDR-Ciliogenesis Mechanism" color={ACCENT4}>
              {df.genetic_architecture && Object.entries(df.genetic_architecture).filter(([k]) => k !== 'key_variants').map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT4}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{typeof v === 'string' ? v : JSON.stringify(v)}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="&#x1f9ea; Key Variants (ZNF423 NPHP14 / JBTS19)" color={ACCENT}>
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
                    <tr><th>Condition</th><th>Key Distinguishing Features from NPHP14 / JBTS19</th></tr>
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
            <Section title="Mechanism" color={ACCENT4}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT + '06', lineHeight: 1.7 }}>
                {df.mechanism}
              </div>
            </Section>
            <Section title="&#x1f3e5; Treatment" color={ACCENT3}>
              {df.treatment && Object.entries(df.treatment).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT3 + '08', borderLeft: `3px solid ${ACCENT3}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}</div>
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
        <Link href="/nphp13" className="btn btn-sm btn-outline-primary">&#x2190; NPHP13 (WDR19/CED1)</Link>
      </div>
    </div>
  );
}
