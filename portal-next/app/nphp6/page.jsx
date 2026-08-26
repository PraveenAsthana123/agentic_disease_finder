'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Genetics & Subtype', 'Definitions'];

// NPHP6 colour scheme — deep purple-gold-crimson-green (CEP290 hub; broadest allele spectrum)
const ACCENT  = '#4a148c';   // deep purple — CEP290 / broadest spectrum / hub gene
const ACCENT2 = '#1b5e20';   // deep green — transplant curative / excellent renal outcome
const ACCENT3 = '#1a237e';   // dark indigo — genetics / allele spectrum / 4 phenotypes
const ACCENT4 = '#b71c1c';   // deep red — ESRD / Meckel end of spectrum (severe)
const ACCENT5 = '#37474f';   // dark slate — renal/tubular component
const ACCENT6 = '#4e342e';   // dark brown — epidemiology / heterogeneous
const ACCENT7 = '#e65100';   // deep orange — LCA misdiagnosis (most critical error)
const ACCENT8 = '#880e4f';   // dark magenta — visual impairment / nystagmus / LCA-like retinal

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

export default function NPHP6Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp6/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp6/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp6/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="container-fluid py-3 px-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🧬 Nephronophthisis Type 6 / Senior-Løken Syndrome 6
        </h4>
        <div className="text-muted small">
          CEP290 — Centrosomal Protein 290kDa · 12q21.32 · TZ Matrix Scaffold · Broadest Allele-Phenotype Spectrum of Any NPHP Gene
        </div>
        <div className="mt-1">
          <Badge text="CEP290 *610142" color={ACCENT} />
          <Badge text="12q21.32" color={ACCENT3} />
          <Badge text="OMIM #610189" color={ACCENT5} />
          <Badge text="AR Biallelic LOF" color={ACCENT6} />
          <Badge text="~1/100k–1/300k" color={ACCENT5} />
          <Badge text="SLS6 / NPHP6" color={ACCENT} />
          <Badge text="LCA10 ≠ NPHP6" color={ACCENT7} />
          <Badge text="No Molar Tooth Sign" color={ACCENT4} />
        </div>
      </div>

      {/* Critical Alert — allele spectrum */}
      <Alert color={ACCENT}>
        <strong style={{ color: ACCENT }}>⚠️ CEP290 HUB GENE — 4 DISTINCT PHENOTYPES, ONE GENE:</strong>
        <ul className="mb-0 mt-1 small">
          <li><strong>IVS26+1655A>G homozygous → LCA10</strong> (retinal ONLY; no renal; sepofarsen therapeutic target)</li>
          <li><strong>Truncating + missense → NPHP6/SLS6</strong> (THIS dashboard — renal + severe retinal; NO Molar Tooth Sign)</li>
          <li><strong>Intermediate alleles → Joubert Syndrome 5 (JBTS5)</strong> (Molar Tooth Sign on MRI; cerebellar hypoplasia)</li>
          <li><strong>Biallelic null → Meckel-Gruber MKS4</strong> (lethal prenatal; encephalocele + cysts + polydactyly)</li>
        </ul>
        <div className="mt-1 small fw-bold" style={{ color: ACCENT4 }}>
          Brain MRI MANDATORY in every CEP290 patient — presence of Molar Tooth Sign = Joubert, NOT NPHP6.
        </div>
      </Alert>

      <Alert color={ACCENT7}>
        <strong style={{ color: ACCENT7 }}>🔴 MOST COMMON MISDIAGNOSIS: LCA (Leber Congenital Amaurosis)</strong>
        {' '}— Severe early retinal dystrophy labelled LCA; renal workup omitted; ESRD arrives years later unmonitored.
        Every LCA patient requires renal function screen + NPHP6/IQCB1 on gene panel.
      </Alert>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {loading && <div className="text-muted">Loading…</div>}
      {error   && <div className="alert alert-danger">Error: {error}</div>}

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && ov && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Median GFR (ml/min)" value={ov.median_gfr} color={ACCENT4} />
            <KPI label="Median Hb (g/dL)" value={ov.median_hb} color={ACCENT2} />
            <KPI label="Median Age Retinal Dx" value={`${ov.median_age_retinal_dx} yr`} color={ACCENT8} />
            <KPI label="Median Age Renal Dx" value={`${ov.median_age_renal_dx} yr`} color={ACCENT4} />
            <KPI label="ESRD/Transplant %" value={`${ov.pct_esrd_or_transplant}%`} color={ACCENT} />
            <KPI label="Nystagmus %" value={`${ov.pct_nystagmus}%`} color={ACCENT8} />
            <KPI label="Severe Retinal %" value={`${ov.pct_severe_retinal}%`} color={ACCENT7} />
            <KPI label="Misdiag. as LCA %" value={`${ov.pct_misdiagnosed_as_lca}%`} color={ACCENT7} />
            <KPI label="Median U-Osm" value={`${ov.median_uosm} mosm`} color={ACCENT5} />
            <KPI label="Mean SBP" value={`${ov.mean_sbp} mmHg`} color={ACCENT3} />
            <KPI label="Cohort n" value={ov.cohort_n} color={ACCENT6} />
            <KPI label="IVS26 Carrier %" value={`${ov.pct_ivs26_carrier}%`} color={ACCENT3} />
          </div>

          {/* Allele spectrum summary */}
          <Section title="🧬 CEP290 Allele-Phenotype Spectrum (same gene, 4 phenotypes)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small mb-0">
                <thead style={{ background: ACCENT + '15' }}>
                  <tr>
                    <th>Allele Type</th>
                    <th>Phenotype</th>
                    <th>Retinal</th>
                    <th>Renal</th>
                    <th>Brain MRI</th>
                    <th>Therapy</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="fw-bold" style={{ color: ACCENT7 }}>IVS26 homozygous</td>
                    <td>LCA10 (retinal ONLY)</td>
                    <td style={{ color: ACCENT8 }}>Severe early</td>
                    <td style={{ color: ACCENT2 }}>NONE</td>
                    <td>Normal</td>
                    <td style={{ color: ACCENT2 }}>Sepofarsen/QR-110</td>
                  </tr>
                  <tr style={{ background: ACCENT + '10' }}>
                    <td className="fw-bold" style={{ color: ACCENT }}>Truncating + missense ★</td>
                    <td>NPHP6/SLS6 (THIS)</td>
                    <td style={{ color: ACCENT8 }}>Severe LCA-like</td>
                    <td style={{ color: ACCENT4 }}>ESRD 13–15yr</td>
                    <td>Normal (NO MTS)</td>
                    <td style={{ color: ACCENT4 }}>Transplant (renal)</td>
                  </tr>
                  <tr>
                    <td className="fw-bold" style={{ color: ACCENT3 }}>Intermediate alleles</td>
                    <td>Joubert Syndrome 5</td>
                    <td style={{ color: ACCENT8 }}>Variable</td>
                    <td style={{ color: ACCENT4 }}>NPHP-like</td>
                    <td style={{ color: ACCENT4 }}>Molar Tooth Sign ✓</td>
                    <td>Multidisciplinary</td>
                  </tr>
                  <tr>
                    <td className="fw-bold" style={{ color: ACCENT4 }}>Biallelic null</td>
                    <td>Meckel-Gruber MKS4</td>
                    <td style={{ color: ACCENT4 }}>Dysplastic</td>
                    <td style={{ color: ACCENT4 }}>Lethal</td>
                    <td style={{ color: ACCENT4 }}>Encephalocele</td>
                    <td style={{ color: ACCENT4 }}>Supportive only</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* NPHP6 vs NPHP5 Comparison */}
          <Section title="🔬 NPHP6 vs NPHP5 (most similar phenotype — critical DDx)" color={ACCENT8}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small mb-0">
                <thead style={{ background: ACCENT8 + '15' }}>
                  <tr>
                    <th>Feature</th>
                    <th style={{ color: ACCENT8 }}>NPHP6 (CEP290)</th>
                    <th style={{ color: '#00695c' }}>NPHP5 (IQCB1)</th>
                    <th>NPHP1</th>
                    <th>Joubert JBTS5</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td className="fw-bold">Gene</td>
                    <td style={{ color: ACCENT }}>CEP290 / 12q21.32</td>
                    <td>IQCB1 / 3q21.1</td>
                    <td>NPHP1 / 2q13</td>
                    <td style={{ color: ACCENT }}>CEP290 / 12q21.32</td></tr>
                  <tr><td className="fw-bold">ESRD median</td>
                    <td style={{ color: ACCENT4 }}>~13–15 yr</td>
                    <td style={{ color: ACCENT4 }}>~13 yr</td>
                    <td>~13 yr</td>
                    <td>Variable</td></tr>
                  <tr><td className="fw-bold">Retinal severity</td>
                    <td style={{ color: ACCENT8 }}>LCA-like (severe)</td>
                    <td style={{ color: ACCENT8 }}>LCA-like (severe)</td>
                    <td>Mild (10–15%)</td>
                    <td>Variable by subtype</td></tr>
                  <tr><td className="fw-bold">Nystagmus</td>
                    <td style={{ color: ACCENT8 }}>~65%</td>
                    <td style={{ color: ACCENT8 }}>75–80%</td>
                    <td>Rare</td>
                    <td>Common</td></tr>
                  <tr><td className="fw-bold">Molar Tooth Sign</td>
                    <td style={{ color: ACCENT2 }}>ABSENT (NPHP6)</td>
                    <td style={{ color: ACCENT2 }}>ABSENT</td>
                    <td style={{ color: ACCENT2 }}>ABSENT</td>
                    <td style={{ color: ACCENT4 }}>PRESENT ★</td></tr>
                  <tr><td className="fw-bold">Allele spectrum</td>
                    <td style={{ color: ACCENT }}>BROADEST (4 phenotypes)</td>
                    <td>Narrow (SLS+NPHP only)</td>
                    <td>Narrow (del 80%)</td>
                    <td>Same as NPHP6</td></tr>
                  <tr><td className="fw-bold">RPGR interaction</td>
                    <td>Indirect (via PCM1)</td>
                    <td style={{ color: '#00695c' }}>DIRECT binding</td>
                    <td>None direct</td>
                    <td>Same as NPHP6</td></tr>
                  <tr><td className="fw-bold">Sepofarsen/ASO</td>
                    <td style={{ color: ACCENT7 }}>IVS26 allele ONLY</td>
                    <td>Not applicable</td>
                    <td>Not applicable</td>
                    <td style={{ color: ACCENT7 }}>IVS26 subtype only</td></tr>
                  <tr><td className="fw-bold">Common misdiagnosis</td>
                    <td style={{ color: ACCENT7 }}>LCA / Joubert</td>
                    <td style={{ color: ACCENT7 }}>LCA</td>
                    <td>ADPKD</td>
                    <td>NPHP6</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* Patient Sample */}
          <Section title={`👥 Sample Patients (first 8 of ${_COHORT_SIZE})`} color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-hover small mb-0">
                <thead style={{ background: ACCENT + '15' }}>
                  <tr>
                    <th>ID</th><th>Gene/Allele</th><th>Ethnicity</th>
                    <th>Retinal Dx</th><th>Renal Dx</th><th>GFR Now</th>
                    <th>U-Osm</th><th>Visual Acuity</th><th>Nystagmus</th><th>RRT/Tx</th>
                  </tr>
                </thead>
                <tbody>
                  {(ov.patients || []).map(p => (
                    <tr key={p.id}>
                      <td><code style={{ fontSize: '0.75em' }}>{p.id}</code></td>
                      <td style={{ maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={p.gene}>{p.gene?.split('—')?.slice(-1)[0]?.trim()?.split('/')[0]?.trim()}</td>
                      <td>{p.ethnicity?.split('(')[0]?.trim()}</td>
                      <td style={{ color: ACCENT8 }}>{p.age_retinal_dx_yr} yr</td>
                      <td style={{ color: ACCENT4 }}>{p.age_renal_dx_yr} yr</td>
                      <td style={{ color: p.gfr_now_ml_min < 15 ? ACCENT4 : p.gfr_now_ml_min < 30 ? ACCENT7 : ACCENT2 }}>
                        {p.gfr_now_ml_min}
                      </td>
                      <td style={{ color: p.urine_osmolality_mosm < 150 ? ACCENT4 : ACCENT5 }}>
                        {p.urine_osmolality_mosm}
                      </td>
                      <td style={{ color: p.visual_acuity?.includes('Light') || p.visual_acuity?.includes('CF') || p.visual_acuity?.includes('HM') ? ACCENT8 : ACCENT6,
                                   maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={p.visual_acuity}>
                        {p.visual_acuity?.split('(')[0]?.trim()}
                      </td>
                      <td style={{ color: p.ocular_motor?.includes('Nystagmus') ? ACCENT8 : ACCENT6 }}>
                        {p.ocular_motor?.includes('Nystagmus') ? '👁 Nys' : '—'}
                      </td>
                      <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={p.rrt_or_transplant}>{p.rrt_or_transplant?.split('—')[0]?.trim()?.split('(')[0]?.trim()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 1: Diagnostic Breakdown ── */}
      {tab === 1 && bk && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="🫘 Kidney Size Distribution" color={ACCENT5}>
                {Object.entries(bk.kidney_size_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
                ))}
              </Section>

              <Section title="👁 Retinal Involvement (LCA-like / SLS6)" color={ACCENT8}>
                {Object.entries(bk.retinal_involvement || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                ))}
              </Section>

              <Section title="👁 Ocular Motor / Nystagmus" color={ACCENT8}>
                {Object.entries(bk.ocular_motor_nystagmus || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                ))}
              </Section>

              <Section title="🔍 Visual Acuity Distribution" color={ACCENT3}>
                {Object.entries(bk.visual_acuity_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>

              <Section title="📊 CKD Stage (Current Visit)" color={ACCENT4}>
                {Object.entries(bk.ckd_stage_current || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="⏱️ Age at Retinal Dx Tiers" color={ACCENT8}>
                {Object.entries(bk.age_at_retinal_dx_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                ))}
              </Section>

              <Section title="⏱️ Age at Renal Dx Tiers" color={ACCENT4}>
                {Object.entries(bk.age_at_renal_dx_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
              </Section>

              <Section title="💧 Urine Osmolality Tiers (mosm/kg)" color={ACCENT5}>
                {Object.entries(bk.urine_osmolality_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
                ))}
              </Section>

              <Section title="💊 RRT / Transplant Status" color={ACCENT2}>
                {Object.entries(bk.rrt_transplant_status || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
                ))}
              </Section>

              <Section title="📉 GFR Slope (ml/min/yr)" color={ACCENT3}>
                {Object.entries(bk.gfr_slope_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>

              <Section title="⚠️ Prior Misdiagnosis" color={ACCENT7}>
                {Object.entries(bk.prior_misdiagnosis || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
                ))}
              </Section>
            </div>
          </div>

          <div className="row">
            <div className="col-md-6">
              <Section title="🧬 CEP290 Allele Distribution" color={ACCENT3}>
                {Object.entries(bk.gene_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="🌍 Ethnicity" color={ACCENT6}>
                {Object.entries(bk.ethnicity || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                ))}
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: Genetics & Subtype ── */}
      {tab === 2 && df && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="🧬 Genetic Architecture (CEP290/12q21.32)" color={ACCENT3}>
                {Object.entries(df.genetic_architecture || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="🔍 Founder Variants" color={ACCENT3}>
                <ul className="small mb-0">
                  {(df.founder_variants || []).map((v, i) => <li key={i} className="mb-1">{v}</li>)}
                </ul>
              </Section>

              <Section title="🎯 CEP290 Allele → Phenotype Map" color={ACCENT}>
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: ACCENT + '15' }}>
                    <tr><th>Allele Type</th><th>Phenotype Outcome</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(df.cep290_allele_spectrum || {}).map(([k, v]) => (
                      <tr key={k} style={k.includes('Truncating') ? { background: ACCENT + '15' } : {}}>
                        <td className="fw-bold small" style={{ color: k.includes('Truncating') ? ACCENT : ACCENT6, whiteSpace: 'nowrap', maxWidth: 150 }}>{k.replace(/_/g, ' ')}</td>
                        <td className="small">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="🦠 NPHP Subtype Comparison" color={ACCENT}>
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: ACCENT + '15' }}>
                    <tr><th>Subtype / Gene</th><th>Key Distinguishing Feature</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(df.nphp_comparison || {}).map(([k, v]) => (
                      <tr key={k} style={k.includes('NPHP6') ? { background: ACCENT + '15' } : {}}>
                        <td className="fw-bold" style={{ color: k.includes('NPHP6') ? ACCENT : ACCENT6, whiteSpace: 'nowrap' }}>{k}</td>
                        <td className="small">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>

              <Section title="🔍 Differential Diagnosis" color={ACCENT7}>
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: ACCENT7 + '15' }}>
                    <tr><th>Condition</th><th>Key Differentiators</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(df.ddx_table || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ color: ACCENT7, whiteSpace: 'nowrap', maxWidth: 160 }}>{k.replace(/_/g, ' ')}</td>
                        <td className="small">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && df && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="📖 Disease Overview" color={ACCENT}>
                <table className="table table-sm table-bordered small mb-3">
                  <tbody>
                    <tr><td className="fw-bold">Disease</td><td>{df.disease}</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>{df.omim_gene}</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>{df.omim_disease}</td></tr>
                    <tr><td className="fw-bold">Chromosome</td><td>{df.chromosome}</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>{df.inheritance}</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>{df.prevalence}</td></tr>
                  </tbody>
                </table>
                <div className="small p-2 rounded" style={{ background: ACCENT + '08', border: `1px solid ${ACCENT}30` }}>
                  <b>Mechanism:</b> {df.mechanism}
                </div>
              </Section>

              <Section title="🏥 Diagnostic Criteria" color={ACCENT3}>
                {Object.entries(df.diagnostic_criteria || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="🩺 Key Clinical Features" color={ACCENT2}>
                {Object.entries(df.key_clinical_features || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="💊 Treatment" color={ACCENT4}>
                {Object.entries(df.treatment || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="📈 Prognosis" color={ACCENT2}>
                <div className="small p-2 rounded" style={{ background: ACCENT2 + '10', border: `1px solid ${ACCENT2}40` }}>
                  {df.prognosis}
                </div>
              </Section>

              <div className="text-muted small mt-2">{df.cohort_note}</div>
            </div>
          </div>
        </div>
      )}

      <div className="mt-3 pt-2 border-top">
        <Link href="/" className="text-muted small">← Back to Portal</Link>
      </div>
    </div>
  );
}
