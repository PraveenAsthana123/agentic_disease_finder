'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Genetics & Allele Spectrum', 'Definitions'];

// NPHP9 colour scheme — deep-crimson-sienna (NEK8; pediatric; multi-organ; rarest NPHP)
const ACCENT  = '#b71c1c';   // deep crimson — NPHP9 rarity / severity; earliest ESRD after NPHP2
const ACCENT2 = '#1b5e20';   // deep green — transplant curative / excellent renal outcome
const ACCENT3 = '#e65100';   // deep orange — situs inversus / laterality defects
const ACCENT4 = '#4a148c';   // deep purple — congenital hepatic fibrosis
const ACCENT5 = '#880e4f';   // dark magenta — pancreatic ductal ectasia (unique feature)
const ACCENT6 = '#1a237e';   // deep indigo — epidemiology / ultra-rare
const ACCENT7 = '#bf360c';   // deep burnt orange — ARPKD misdiagnosis (most common error)
const ACCENT8 = '#006064';   // deep teal — NEK8–ANKS6–BICC1 IFT-zone module

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

export default function NPHP9Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp9/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp9/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp9/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading NPHP9 cohort…</div>;
  if (error)   return <div className="container py-5 text-danger">Error: {error}</div>;
  if (!ov)     return null;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', border: `2px solid ${ACCENT}` }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div style={{ flex: 1, minWidth: 260 }}>
            <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
              &#x1f9ec; Nephronophthisis Type 9 (NPHP9)
            </h4>
            <div className="small text-muted mb-1">
              <strong>NEK8</strong> · 17q11.2 · 692 aa · NIMA kinase family · IFT-zone / inversin compartment
              · N-terminal kinase domain (aa 1–285) + C-terminal RCC1-like domain (aa 286–692)
            </div>
            <div className="small">
              <Badge text="OMIM *613312" color={ACCENT} />
              <Badge text="#613824 NPHP9" color={ACCENT} />
              <Badge text="AR biallelic LOF" color={ACCENT6} />
              <Badge text="17q11.2" color={ACCENT8} />
              <Badge text="Rarest NPHP subtype" color="#c62828" />
            </div>
          </div>
          <div className="d-flex gap-2 flex-wrap">
            <span className="badge px-3 py-2" style={{ background: ACCENT, fontSize: '0.8em' }}>
              ESRD median ~10–13yr
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT4, fontSize: '0.8em' }}>
              CHF {ov.pct_hepatic_fibrosis}% (biliary)
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT5, fontSize: '0.8em' }}>
              Pancreatic {ov.pct_pancreatic_involvement}% (UNIQUE)
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT3, fontSize: '0.8em' }}>
              Situs inversus {ov.pct_situs_inversus}%
            </span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert color={ACCENT7}>
        <strong>&#x26a0;&#xfe0f; MOST COMMON MISDIAGNOSIS — ARPKD:</strong> Enlarged echogenic kidneys +
        CHF + situs inversus → paediatric team assumes PKHD1. PKHD1 negative on WES → add NEK8 and full
        NPHP panel. ARPKD and NPHP9 are phenocopies on USS; genetics is the only discriminator.
      </Alert>
      <Alert color={ACCENT5}>
        <strong>&#x1f941; PANCREATIC DUCTAL ECTASIA — NPHP9 UNIQUE FEATURE:</strong> Only NPHP subtype with
        pancreatic ductal involvement (15–25%). ABSENT in NPHP1-8. MRCP if ductal ectasia on USS. NEK8
        expressed in pancreatic ductal cells — LOF → ductal ectasia / cysts. Monitor exocrine function annually.
      </Alert>
      <Alert color={ACCENT3}>
        <strong>&#x1f9ed; SITUS INVERSUS — cardiac work-up MANDATORY:</strong> 25–30% of NPHP9. Situs ambiguus
        / heterotaxy in 6% → complex CHD (TAPVR/ASD/VSD). Echocardiogram + chest X-ray if any laterality
        defect found. Surgical sequencing: cardiac repair before renal transplant listing.
      </Alert>
      <Alert color={ACCENT2}>
        <strong>&#x2705; RENAL TRANSPLANT = CURATIVE:</strong> Cell-autonomous IFT-zone defect. NO recurrence.
        Excellent graft outcomes. Living donor preferred. Hepatic / pancreatic involvement independent —
        does not worsen post-transplant (cell-autonomous biliary + ductal expression).
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
            <KPI label="Median Hb" value={`${ov.median_hb} g/dL`} color="#5d4037" />
            <KPI label="Median age Dx" value={`${ov.median_age_renal_dx}yr`} color={ACCENT} />
            <KPI label="ESRD/Transplant" value={`${ov.pct_esrd_or_transplant}%`} color={ACCENT} />
            <KPI label="Situs inversus" value={`${ov.pct_situs_inversus}%`} color={ACCENT3} />
            <KPI label="CHF" value={`${ov.pct_hepatic_fibrosis}%`} color={ACCENT4} />
            <KPI label="Pancreatic" value={`${ov.pct_pancreatic_involvement}%`} color={ACCENT5} />
            <KPI label="Cardiac defect" value={`${ov.pct_cardiac_defect}%`} color={ACCENT3} />
            <KPI label="Enlarged kidneys (early)" value={`${ov.pct_enlarged_kidneys_early}%`} color={ACCENT7} />
            <KPI label="Polyuria first" value={`${ov.pct_polyuria_first_symptom}%`} color={ACCENT8} />
            <KPI label="ARPKD misdiag" value={`${ov.pct_misdiagnosed_as_arpkd}%`} color={ACCENT7} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="&#x1f9ec; NEK8–ANKS6–BICC1 IFT-Zone Module" color={ACCENT8}>
                <div className="small text-muted mb-2">
                  NEK8 is the kinase component of the IFT-zone (inversin compartment) complex.
                  It phosphorylates BICC1 to restrain mTOR/Wnt cystogenesis signalling.
                </div>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>NEK8 (Never In Mitosis A Related Kinase 8)</td></tr>
                    <tr><td className="fw-bold">Location</td><td>17q11.2</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>692 aa · ~78 kDa · NIMA-family kinase</td></tr>
                    <tr><td className="fw-bold">Domains</td><td>Kinase (1–285) + RCC1-like 7-β-propeller (286–692)</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*613312</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#613824 (Nephronophthisis 9)</td></tr>
                    <tr><td className="fw-bold">Complex</td><td>NEK8–ANKS6 (NPHP16)–BICC1 at inversin compartment</td></tr>
                    <tr><td className="fw-bold">Pathway</td><td>mTORC1/Rheb (BICC1) + Wnt canonical (DVL2) + DDR (RPA)</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/1,000,000–2,000,000 (rarest NPHP subtype)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive · biallelic LOF</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="&#x1f6a8; NPHP9 Hallmark Features" color={ACCENT}>
                {[
                  [`Pancreatic ductal ectasia ${ov.pct_pancreatic_involvement}% (UNIQUE)`, ACCENT5,
                   'ABSENT in ALL other NPHP subtypes — NPHP9 exclusive; NEK8 expressed pancreatic ducts'],
                  [`Congenital hepatic fibrosis ${ov.pct_hepatic_fibrosis}%`, ACCENT4,
                   'Biliary ductal plate malformation; portal HTN; similar to NPHP2/3'],
                  [`Situs inversus / heterotaxy ${ov.pct_situs_inversus}%`, ACCENT3,
                   'Nodal cilia expression; less penetrant than NPHP2 (35%); cardiac work-up mandatory'],
                  [`Kidneys enlarged early ${ov.pct_enlarged_kidneys_early}%`, ACCENT7,
                   'ARPKD-like USS; evolves to small fibrotic; most common misdiagnosis trap'],
                  ['NO retinal dystrophy (0%)', ACCENT2,
                   'NEK8 absent from photoreceptors; pure TIN + multi-organ ciliopathy'],
                  ['NO Molar Tooth Sign (0%)', ACCENT2,
                   'NEK8 not a Joubert gene; no cerebellar vermis hypoplasia; no MTS'],
                  [`ESRD median ~${ov.median_age_renal_dx}yr (cohort)`, ACCENT,
                   'Earlier than NPHP3-8; later than NPHP2 (3yr); range 4–25yr'],
                  [`Renal transplant CURATIVE`, ACCENT2,
                   'Cell-autonomous IFT-zone defect; NO recurrence; excellent outcomes'],
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
                    <th>Age Dx (yr)</th><th>GFR</th><th>Situs</th>
                    <th>CHF</th><th>Pancreatic</th><th>First Symptom</th>
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
                          ? <span className="badge" style={{ background: ACCENT3 }}>Situs ✓</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                      <td>
                        {p.hepatic_fibrosis
                          ? <span className="badge" style={{ background: ACCENT4 }}>CHF ✓</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                      <td>
                        {p.pancreatic_involvement
                          ? <span className="badge" style={{ background: ACCENT5 }}>Panc ✓</span>
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
            <Section title="Kidney Phenotype Distribution" color={ACCENT7}>
              {Object.entries(bk.kidney_phenotype_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
            <Section title="Situs / Laterality Distribution" color={ACCENT3}>
              {Object.entries(bk.situs_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
            <Section title="Hepatic Status (CHF) Distribution" color={ACCENT4}>
              {Object.entries(bk.hepatic_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
              ))}
            </Section>
            <Section title="Pancreatic Involvement Distribution" color={ACCENT5}>
              {Object.entries(bk.pancreatic_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
              ))}
            </Section>
            <Section title="Cardiac Status Distribution" color={ACCENT3}>
              {Object.entries(bk.cardiac_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="CKD Stage (current)" color={ACCENT}>
              {Object.entries(bk.ckd_stage_current).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="RRT / Transplant Status" color={ACCENT2}>
              {Object.entries(bk.rrt_transplant_status).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
              ))}
            </Section>
            <Section title="Prior Misdiagnosis" color={ACCENT7}>
              {Object.entries(bk.prior_misdiagnosis).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
            <Section title="First Symptom" color={ACCENT8}>
              {Object.entries(bk.first_symptom_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
              ))}
            </Section>
            <Section title="Growth Status" color="#5d4037">
              {Object.entries(bk.growth_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color="#5d4037" />
              ))}
            </Section>
          </div>

          <div className="col-12">
            <div className="row g-3">
              <div className="col-md-4">
                <Section title="Age at Renal Dx Tiers" color={ACCENT}>
                  {Object.entries(bk.age_at_renal_dx_tiers).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                  ))}
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="GFR Slope (ml/min/yr)" color={ACCENT}>
                  {Object.entries(bk.gfr_slope_tiers).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                  ))}
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="Urine Osmolality (mOsm/kg)" color={ACCENT8}>
                  {Object.entries(bk.urine_osmolality_tiers).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                  ))}
                </Section>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 2: Genetics ── */}
      {tab === 2 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Allele Classes / Variant Distribution" color={ACCENT}>
              {bk && Object.entries(bk.gene_distribution).map(([k, v]) => (
                <Bar key={k} label={k.slice(0, 60)} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="Ethnicity Distribution" color={ACCENT6}>
              {bk && Object.entries(bk.ethnicity).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Key Variants (NEK8 / NPHP9)" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>#</th><th>Variant</th></tr></thead>
                <tbody>
                  {df.key_variants.map((v, i) => (
                    <tr key={i}>
                      <td><Badge text={i + 1} color={ACCENT} /></td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="Genetic Architecture" color={ACCENT8}>
              {df.genetic_architecture && Object.entries(df.genetic_architecture).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <span className="fw-bold small" style={{ color: ACCENT8 }}>{k.replace(/_/g, ' ')}: </span>
                  <span className="small text-muted">{v}</span>
                </div>
              ))}
            </Section>
          </div>

          <div className="col-12">
            <Section title="NPHP Subtype Comparison (including NPHP9)" color={ACCENT}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead style={{ background: ACCENT + '22' }}>
                    <tr><th>Subtype / Gene</th><th>Note</th></tr>
                  </thead>
                  <tbody>
                    {df.nphp_comparison && Object.entries(df.nphp_comparison).map(([k, v]) => (
                      <tr key={k} style={k.includes('★') ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                        <td style={{ whiteSpace: 'nowrap', color: k.includes('★') ? ACCENT : 'inherit' }}>{k}</td>
                        <td>{v}</td>
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
                  {[
                    ['Disease', df.disease],
                    ['OMIM Gene', df.omim_gene],
                    ['OMIM Disease', df.omim_disease],
                    ['Chromosome', df.chromosome],
                    ['Inheritance', df.inheritance],
                    ['Prevalence', df.prevalence],
                  ].map(([k, v]) => (
                    <tr key={k}><td className="fw-bold">{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="Mechanism" color={ACCENT8}>
              <p className="small text-muted">{df.mechanism}</p>
            </Section>
            <Section title="Diagnostic Criteria" color={ACCENT}>
              {df.diagnostic_criteria && Object.entries(df.diagnostic_criteria).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <span className="fw-bold small" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}: </span>
                  <span className="small text-muted">{v}</span>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Key Clinical Features" color={ACCENT5}>
              {df.key_clinical_features && Object.entries(df.key_clinical_features).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT5 + '10', borderLeft: `3px solid ${ACCENT5}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT5 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="small text-muted">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="DDx Table" color={ACCENT7}>
              {df.ddx_table && Object.entries(df.ddx_table).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <span className="fw-bold small" style={{ color: ACCENT7 }}>{k}: </span>
                  <span className="small text-muted">{v}</span>
                </div>
              ))}
            </Section>
            <Section title="Treatment" color={ACCENT2}>
              {df.treatment && Object.entries(df.treatment).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <span className="fw-bold small" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}: </span>
                  <span className="small text-muted">{v}</span>
                </div>
              ))}
            </Section>
            <Section title="Prognosis" color={ACCENT}>
              <p className="small text-muted">{df.prognosis}</p>
            </Section>
            <Section title="Cohort Note" color="#757575">
              <p className="small text-muted fst-italic">{df.cohort_note}</p>
            </Section>
          </div>
        </div>
      )}

      {/* Footer nav */}
      <div className="mt-4 pt-3 border-top d-flex gap-2 flex-wrap">
        <Link href="/nphp8" className="btn btn-outline-secondary btn-sm">← NPHP8 (RPGRIP1L)</Link>
        <Link href="/" className="btn btn-outline-primary btn-sm">&#x1f3e0; Portal Home</Link>
      </div>
    </div>
  );
}

const SEED = 357;
