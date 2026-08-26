'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'DAP Architecture & Genetics', 'Definitions'];

// NPHP15 colour scheme — CEP164 / distal appendages / SLS / cobalt-sienna-teal-slate
const ACCENT  = '#01579b';   // deep cobalt — CEP164 distal appendage; ciliogenesis initiation
const ACCENT2 = '#bf360c';   // deep sienna — SLS retinal dystrophy; rod-cone degeneration
const ACCENT3 = '#1b5e20';   // deep forest green — NPHP15 renal; transplant curative
const ACCENT4 = '#006064';   // dark teal — TTBK2 phospho-axis; DAP-II docking; molecular mechanism
const ACCENT5 = '#880e4f';   // dark magenta — ESRD; CKD progression; concentrating defect
const ACCENT6 = '#37474f';   // dark slate — molecular architecture; DAP hierarchy details
const ACCENT7 = '#e65100';   // burnt orange — misdiagnosis (NPHP1 MLPA / LCA workup)
const ACCENT8 = '#4a148c';   // deep purple — NPHP1 direct binding; DAP-TZ bridge; NPHP4

const SEED = 369;
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

export default function NPHP15Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp15/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp15/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp15/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading NPHP15 cohort…</div>;
  if (error)   return <div className="container py-5 text-danger">Error: {error}</div>;
  if (!ov)     return null;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', border: `2px solid ${ACCENT}` }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div style={{ flex: 1, minWidth: 260 }}>
            <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
              &#x1f9ec; Nephronophthisis Type 15 (NPHP15) — CEP164 / Distal Appendage Ciliopathy
            </h4>
            <div className="small text-muted mb-1">
              <strong>CEP164</strong> · 11q23.3 · 1,460 aa · distal appendage scaffold (transition fiber) ·
              TTBK2 phospho-target (Ser172) · vesicle docking → CP110 removal → ciliogenesis ·
              NPHP1 direct binding partner · Senior-Løken Syndrome (SLS) 35–40%
            </div>
            <div className="small">
              <Badge text="OMIM *614848" color={ACCENT} />
              <Badge text="#614845 NPHP15" color={ACCENT} />
              <Badge text="AR biallelic LOF" color={ACCENT6} />
              <Badge text="11q23.3" color={ACCENT6} />
              <Badge text="DAP initiator" color={ACCENT4} />
              <Badge text={`SLS retinal ${ov.pct_sls_full}%`} color={ACCENT2} />
              <Badge text="NO Joubert · NO CHF · NO ID" color={ACCENT3} />
            </div>
          </div>
          <div className="d-flex gap-2 flex-wrap">
            <span className="badge px-3 py-2" style={{ background: ACCENT, fontSize: '0.8em' }}>
              ESRD median ~13–15yr
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT2, fontSize: '0.8em' }}>
              SLS {ov.pct_retinal_involvement}% retinal
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT3, fontSize: '0.8em' }}>
              No Joubert · No ID
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT5, fontSize: '0.8em' }}>
              ESRD/Tx {ov.pct_esrd_or_transplant}%
            </span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert color={ACCENT4}>
        <strong>&#x1f9ec; DISTAL APPENDAGE (TRANSITION FIBER) NPHP — UNIQUE CILIOGENESIS INITIATION DEFECT:</strong> CEP164
        is a core scaffold of the distal appendages (DAP) of the mother centriole — the ring of 9 symmetric
        transition fibers required for vesicle docking at the ciliary pit. TTBK2 phosphorylates CEP164 Ser172
        → activates Rab8a vesicle docking → CP110 cap removal → axoneme elongation. CEP164 loss → vesicle-docking
        arrest → persistent CP110 → ciliogenesis failure → NPHP15 TIN + cysts → ESRD. Mechanistically distinct
        from TZ-scaffold NPHP (NPHP1/4/8) and DDR-NPHP (NPHP14/ZNF423).
      </Alert>
      <Alert color={ACCENT2}>
        <strong>&#x1f441; SENIOR-LØKEN SYNDROME (SLS) IN 35–40% — RETINAL DYSTROPHY + NPHP15:</strong> CEP164 IS
        expressed in photoreceptor connecting cilia (unlike ZNF423/NPHP14) → rod-cone degeneration → SLS.
        ERG flat/extinguished; nystagmus from infancy; progressive visual loss. Retinal does NOT improve
        post-transplant — cell-autonomous photoreceptor CEP164 defect. Annual ophthalmology ERG mandatory
        in ALL NPHP15 patients regardless of initial visual symptoms — SLS may appear after renal onset.
        LCA-like severe retinal in ~10% (null×null alleles). Low-vision support services immediately.
      </Alert>
      <Alert color={ACCENT8}>
        <strong>&#x1f517; NPHP1 DIRECT BINDING PARTNER — CO-SEQUENCE MANDATORY:</strong> CEP164 WW-like
        domain (aa 561–820) directly binds NPHP1 (Nephrocystin-1) at the DAP-TZ interface — CEP164
        physically links the distal appendage platform to the TZ NPHP scaffold. Always co-sequence NPHP1
        when CEP164 found (digenic CEP164+NPHP1 heterozygosity may cause ciliopathy). CEP164 also binds
        NPHP4 and RPGRIP1L (NPHP8) — include these on targeted re-sequencing. CEP164 is the ONLY NPHP
        gene that directly bridges the DAP to the TZ gate assembly.
      </Alert>
      <Alert color={ACCENT7}>
        <strong>&#x26a0;&#xfe0f; MOST COMMON MISDIAGNOSIS — NPHP1 DELETION MLPA + LCA WORKUP:</strong> NPHP1
        MLPA (290kb deletion test) is the standard first-line test for suspected NPHP → does NOT detect
        CEP164 on 11q23.3. SLS cases are frequently worked up as pure LCA (retinal-only panels miss NPHP15
        renal involvement). Annual renal USS + urine osmolality MANDATORY in ALL LCA patients — ESRD in
        NPHP15 may present years after retinal onset. WES including CEP164 is the only reliable test.
        NPHP5/IQCB1 is the first SLS gene tested → CEP164 missed when NPHP5 negative.
      </Alert>
      <Alert color={ACCENT3}>
        <strong>&#x1f6ab; NO JOUBERT · NO CHF · NO INTELLECTUAL DISABILITY — KEY DDx:</strong> CEP164 is NOT
        a Joubert gene. No Molar Tooth Sign, no cerebellar vermis hypoplasia — if these are present,
        the diagnosis is NOT NPHP15 (consider CEP290/NPHP6 or RPGRIP1L/NPHP8). No CHF (CEP164 absent
        biliary epithelium). No intellectual disability (CEP164 not expressed in neurons). Normal cognitive
        development expected. These absences are critical DDx markers for NPHP15.
      </Alert>
      <Alert color={ACCENT5}>
        <strong>&#x2705; RENAL TRANSPLANT = CURATIVE — RETINAL IS INDEPENDENT:</strong> Cell-autonomous
        DAP defect in native kidneys → NO recurrence in transplanted kidney (transplanted cells have
        normal CEP164 → ciliogenesis normal). Excellent graft outcomes. HOWEVER: SLS retinal dystrophy
        is also cell-autonomous in photoreceptors — transplant does NOT improve retinal. Retinal
        management and renal management must be planned and coordinated but are INDEPENDENT.
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
            <KPI label="SLS (full)" value={`${ov.pct_sls_full}%`} color={ACCENT2} />
            <KPI label="Any retinal" value={`${ov.pct_retinal_involvement}%`} color={ACCENT2} />
            <KPI label="LCA-like" value={`${ov.pct_lca_like}%`} color={ACCENT2} />
            <KPI label="CHF" value={`${ov.pct_chf_involvement}%`} color={ACCENT3} />
            <KPI label="NPHP1 misdiag" value={`${ov.pct_misdiagnosed_as_nphp1}%`} color={ACCENT7} />
            <KPI label="LCA misdiag" value={`${ov.pct_misdiagnosed_as_lca}%`} color={ACCENT7} />
            <KPI label="Visual first" value={`${ov.pct_visual_first}%`} color={ACCENT2} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="&#x1f9ec; CEP164 — Distal Appendage Scaffold & NPHP15 (11q23.3)" color={ACCENT6}>
                <div className="small text-muted mb-2">
                  CEP164 (1,460 aa; ~165 kDa) is a core scaffold protein of the distal appendages (DAP /
                  transition fibers) of the mother centriole. It is the central hub of the DAP ring (CEP83→SCLT1→
                  CEP164→FBF1→LRRC45 hierarchy). TTBK2 phosphorylates CEP164 Ser172 to initiate ciliogenesis:
                  → Rab8a vesicle docking → CP110 cap removal → axoneme elongation. Loss → vesicle-docking arrest
                  → NPHP15 TIN + corticomedullary cysts → ESRD. CEP164 also directly binds NPHP1 + NPHP4 (TZ bridge).
                </div>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>CEP164 (Centrosomal Protein 164)</td></tr>
                    <tr><td className="fw-bold">Location</td><td>11q23.3</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>1,460 aa · ~165 kDa · distal appendage (DAP) scaffold · 9-fold symmetric ring of mother centriole</td></tr>
                    <tr><td className="fw-bold">Domains</td><td>N-CC (1–220): RABIN8 · DAP-I (221–560): SCLT1+CEP83+TTBK2 Ser172 · WW-like (561–820): NPHP1+NPHP4 · DAP-II (821–1120): FBF1+Rab8a · C-CC (1121–1460): LRRC45+NPHP8</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*614848</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#614845 (NPHP15)</td></tr>
                    <tr><td className="fw-bold">Key partners</td><td>NPHP1 (WW-like direct binding), NPHP4, TTBK2 (kinase), RABIN8, Rab8a, FBF1, CEP83, SCLT1, RPGRIP1L/NPHP8</td></tr>
                    <tr><td className="fw-bold">Mechanism</td><td>TTBK2→CEP164 Ser172→RABIN8→Rab8a vesicle docking→CP110 removal→axoneme; loss → ciliogenesis arrest</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/300,000–600,000; ~80–100 published families (2026)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive · biallelic LOF</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="&#x1f6a8; NPHP15 Hallmark Features" color={ACCENT}>
                {[
                  ['DISTAL APPENDAGE (TRANSITION FIBER) NPHP — DAP initiator mechanism', ACCENT4,
                   'CEP164 = central hub of the 9-fold DAP ring at the mother centriole. TTBK2 phosphorylates CEP164 Ser172 → Rab8a vesicle docking → CP110 removal → ciliogenesis. Distinct from TZ-scaffold NPHP (NPHP1/4/8) and DDR-NPHP (NPHP14). CEP164 loss → vesicle-docking arrest → CP110 persists → no axoneme'],
                  [`Senior-Løken Syndrome (SLS) — ${ov.pct_retinal_involvement}% retinal (${ov.pct_sls_full}% full SLS)`, ACCENT2,
                   'CEP164 expressed in photoreceptor connecting cilia → rod-cone degeneration → SLS. ERG flat/extinguished; nystagmus; photophobia; progressive visual loss. LCA-like in ~10% (null×null). Retinal does NOT improve post-transplant — cell-autonomous photoreceptor defect. Annual ERG mandatory ALL patients'],
                  [`NPHP15 renal phenotype — ESRD median ~13–15yr`, ACCENT3,
                   'TIN + corticomedullary cysts + concentrating defect (polyuria first). Small echogenic kidneys on USS. Similar to NPHP1 (~13yr); earlier than NPHP14 (~13–18yr). Renal transplant CURATIVE; NO recurrence (cell-autonomous DAP defect)'],
                  ['NPHP1 direct binding partner — co-sequence mandatory', ACCENT8,
                   'CEP164 WW-like domain (aa 561–820) directly binds NPHP1 at the DAP-TZ interface — physically links distal appendage to TZ gate. Also binds NPHP4. Always co-sequence NPHP1+NPHP4+NPHP8 when CEP164 found. Digenic CEP164+NPHP1 heterozygosity may cause ciliopathy'],
                  ['TTBK2 phosphorylation target — Ser172 ciliogenesis master switch', ACCENT4,
                   'TTBK2 phosphorylates CEP164 Ser172 (DAP-I domain) → activates CEP164 as vesicle-docking hub. CEP164 is the only NPHP gene product directly phosphorylated by TTBK2. Missense variants in Ser172 vicinity disrupt TTBK2→RABIN8→Rab8a cascade'],
                  [`NO Joubert · NO ID — key DDx vs NPHP6/NPHP14`, ACCENT3,
                   'CEP164 is NOT a Joubert gene. No MTS, no cerebellar hypoplasia, no intellectual disability. If Joubert present → reconsider toward CEP290 (NPHP6) or RPGRIP1L (NPHP8). Normal cognitive development in NPHP15. Key differentiator from NPHP14/JBTS19 (ID 25–35%)'],
                  ['NO CHF · NO situs — DAP protein not in biliary/nodal cilia', ACCENT3,
                   'CEP164 absent in biliary epithelium (no CHF) and nodal cilia (no situs inversus). No BBS features (no obesity, no polydactyly). No ectodermal features. Absence of CHF + situs + Joubert + ID is the NPHP15 defining exclusion cluster'],
                  ['Renal transplant CURATIVE — retinal management independent', ACCENT3,
                   'Cell-autonomous DAP defect; NO recurrence post-transplant. SLS retinal also cell-autonomous in photoreceptors — transplant does NOT cure retinal dystrophy. Retinal and renal must both be planned: ophthalmology + nephrology coordinated, independent outcome trajectories'],
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
                    <th>Retinal/SLS</th><th>First Symptom</th><th>Misdiagnosis</th>
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
                        {p.sls_full
                          ? <span className="badge" style={{ background: ACCENT2 }}>SLS ✓</span>
                          : p.lca_like
                          ? <span className="badge" style={{ background: ACCENT5 }}>LCA ✓</span>
                          : p.retinal_involvement
                          ? <span className="badge" style={{ background: ACCENT7 }}>subclinical</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                      <td style={{ fontSize: '0.72em' }}>{p.first_symptom.split('(')[0].trim().slice(0, 30)}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.prior_misdiagnosis.split('(')[0].trim().slice(0, 25)}</td>
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
            <Section title="Retinal / SLS Status Distribution" color={ACCENT2}>
              {Object.entries(bk.retinal_status_distribution).map(([k, v]) => (
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
            <Section title="Prior Misdiagnosis (NPHP1 MLPA / LCA workup most common)" color={ACCENT7}>
              {Object.entries(bk.prior_misdiagnosis).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
            <Section title="First Presenting Symptom" color={ACCENT}>
              {Object.entries(bk.first_symptom_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="CHF / Hepatic Status (expected: &gt;98% absent)" color={ACCENT3}>
              {Object.entries(bk.chf_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
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

      {/* ── Tab 2: DAP Architecture & Genetics ── */}
      {tab === 2 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="&#x1f9ec; CEP164 Distal Appendage Architecture & TTBK2 Mechanism" color={ACCENT4}>
              {df.genetic_architecture && Object.entries(df.genetic_architecture).filter(([k]) => k !== 'key_variants').map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT4}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{typeof v === 'string' ? v : JSON.stringify(v)}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="&#x1f9ea; Key Variants (CEP164 NPHP15)" color={ACCENT}>
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
                    <tr><th>Condition</th><th>Key Distinguishing Features from NPHP15</th></tr>
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
        <Link href="/nphp14" className="btn btn-sm btn-outline-primary">&#x2190; NPHP14 (ZNF423/JBTS19)</Link>
      </div>
    </div>
  );
}
