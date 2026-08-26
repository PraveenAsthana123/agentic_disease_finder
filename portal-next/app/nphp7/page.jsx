'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Genetics & Subtype', 'Definitions'];

// NPHP7 colour scheme — steel-teal-amber-green (GLIS2; pure renal; zinc finger; rare)
const ACCENT  = '#006064';   // deep teal — GLIS2 kidney-enriched / pure renal; zinc finger
const ACCENT2 = '#1b5e20';   // deep green — transplant curative / excellent renal outcome
const ACCENT3 = '#01579b';   // dark blue — genetics / founder alleles / WES mandatory
const ACCENT4 = '#bf360c';   // deep burnt orange — ESRD / renal progression
const ACCENT5 = '#37474f';   // dark slate — renal/tubular component
const ACCENT6 = '#4a148c';   // deep purple — epidemiology / very rare ~1/1M
const ACCENT7 = '#e65100';   // deep orange — ADPKD misdiagnosis (most common error)
const ACCENT8 = '#558b2f';   // dark olive — pure renal positive feature (no extra-renal)

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

export default function NPHP7Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp7/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp7/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp7/definitions`).then(r => r.json()),
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
          🧬 Nephronophthisis Type 7 (GLIS2/NPHP7)
        </h4>
        <div className="text-muted small">
          GLIS2 — GLIS Family Zinc Finger 2 · 16p13.3 · 525 aa · Ciliary Krüppel-like Transcription Factor · Pure Renal Phenotype · Very Rare ~1/1,000,000
        </div>
        <div className="mt-1">
          <Badge text="GLIS2 *608539" color={ACCENT} />
          <Badge text="16p13.3" color={ACCENT3} />
          <Badge text="OMIM #611498" color={ACCENT5} />
          <Badge text="AR Biallelic LOF" color={ACCENT6} />
          <Badge text="~1/1,000,000" color={ACCENT6} />
          <Badge text="PURE RENAL" color={ACCENT8} />
          <Badge text="NO Retinal" color={ACCENT2} />
          <Badge text="NO Hepatic" color={ACCENT2} />
          <Badge text="NO Situs" color={ACCENT2} />
        </div>
      </div>

      {/* Critical Alert — pure renal / no extra-renal */}
      <Alert color={ACCENT}>
        <strong style={{ color: ACCENT }}>✅ NPHP7 IS PURE RENAL — NO EXTRA-RENAL FEATURES:</strong>
        <ul className="mb-0 mt-1 small">
          <li><strong>NO retinal dystrophy, NO nystagmus</strong> — GLIS2 absent from photoreceptors (unlike NPHP4/5/6 SLS subtypes)</li>
          <li><strong>NO hepatic fibrosis</strong> — GLIS2 absent from biliary epithelium (unlike NPHP2/3)</li>
          <li><strong>NO situs inversus</strong> — GLIS2 absent from nodal cilia (unlike NPHP2/3)</li>
          <li><strong>NO Molar Tooth Sign, NO cerebellar features</strong> — not a Joubert spectrum gene</li>
        </ul>
        <div className="mt-1 small fw-bold" style={{ color: ACCENT2 }}>
          No ophthalmology ERG / no hepatic USS / no cardiac ECG surveillance required — renal monitoring ONLY.
        </div>
      </Alert>

      <Alert color={ACCENT7}>
        <strong style={{ color: ACCENT7 }}>🔴 MOST COMMON MISDIAGNOSIS: ADPKD</strong>
        {' '}— Small corticomedullary cysts on ultrasound → assumed autosomal dominant PKD; wrong inheritance assumed; PKD1/PKD2 sequencing first; delay in correct AR NPHP genetics.
        Always consider NPHP7/WES if: AR family history, childhood polyuria, small kidneys, TIN on biopsy.
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
            <KPI label="Median Age Renal Dx" value={`${ov.median_age_renal_dx} yr`} color={ACCENT4} />
            <KPI label="ESRD/Transplant %" value={`${ov.pct_esrd_or_transplant}%`} color={ACCENT} />
            <KPI label="Polyuria 1st Symptom %" value={`${ov.pct_polyuria_first_symptom}%`} color={ACCENT5} />
            <KPI label="Misdiag. as ADPKD %" value={`${ov.pct_misdiagnosed_as_adpkd}%`} color={ACCENT7} />
            <KPI label="Median U-Osm" value={`${ov.median_uosm} mosm`} color={ACCENT5} />
            <KPI label="Mean SBP" value={`${ov.mean_sbp} mmHg`} color={ACCENT3} />
            <KPI label="Founder p.Ala279Val %" value={`${ov.pct_founder_ala279val}%`} color={ACCENT3} />
            <KPI label="Retinal Dystrophy %" value={`${ov.pct_retinal_dystrophy}%`} color={ACCENT8} />
            <KPI label="Hepatic Fibrosis %" value={`${ov.pct_hepatic_fibrosis}%`} color={ACCENT8} />
            <KPI label="Cohort n" value={ov.cohort_n} color={ACCENT6} />
          </div>

          {/* Pure renal feature summary */}
          <Section title="🏥 NPHP7 Feature Profile — Pure Renal vs Other NPHP Subtypes" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small mb-0">
                <thead style={{ background: ACCENT + '15' }}>
                  <tr>
                    <th>Subtype</th>
                    <th>Gene / Locus</th>
                    <th>ESRD (yr)</th>
                    <th>Retinal</th>
                    <th>CHF</th>
                    <th>Situs</th>
                    <th>Joubert</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>NPHP1</td><td>NPHP1 / 2q13</td><td>~13</td>
                    <td>10–15%</td><td>No</td><td>No</td><td>No</td>
                  </tr>
                  <tr>
                    <td>NPHP2</td><td>INVS / 9q31.1</td><td>~3</td>
                    <td>No</td><td>55%</td><td>35%</td><td>No</td>
                  </tr>
                  <tr>
                    <td>NPHP3</td><td>NPHP3 / 3q22.1</td><td>~19</td>
                    <td>No</td><td>45%</td><td>15%</td><td>No</td>
                  </tr>
                  <tr>
                    <td>NPHP4 / SLS4</td><td>NPHP4 / 1p36</td><td>17–20</td>
                    <td>15–20%</td><td>No</td><td>No</td><td>rare</td>
                  </tr>
                  <tr>
                    <td>NPHP5 / SLS5</td><td>IQCB1 / 3q21.1</td><td>~13</td>
                    <td>Severe LCA-like</td><td>No</td><td>No</td><td>No</td>
                  </tr>
                  <tr>
                    <td>NPHP6 / SLS6</td><td>CEP290 / 12q21.32</td><td>13–15</td>
                    <td>Severe LCA-like</td><td>No</td><td>No</td><td>JBTS5 same gene</td>
                  </tr>
                  <tr style={{ background: ACCENT + '12', fontWeight: 700 }}>
                    <td style={{ color: ACCENT }}>NPHP7 ★</td>
                    <td style={{ color: ACCENT }}>GLIS2 / 16p13.3</td>
                    <td>16–20</td>
                    <td style={{ color: ACCENT2 }}>❌ None</td>
                    <td style={{ color: ACCENT2 }}>❌ None</td>
                    <td style={{ color: ACCENT2 }}>❌ None</td>
                    <td style={{ color: ACCENT2 }}>❌ None</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* Sample patients */}
          <Section title="👥 Sample Patients (first 8)" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-hover small mb-0">
                <thead style={{ background: ACCENT3 + '15' }}>
                  <tr>
                    <th>ID</th><th>GFR</th><th>Age Dx (yr)</th>
                    <th>CKD Stage</th><th>1st Symptom</th><th>Prior Misdiag.</th>
                  </tr>
                </thead>
                <tbody>
                  {ov.patients.map(p => (
                    <tr key={p.id}>
                      <td className="fw-bold" style={{ color: ACCENT }}>{p.id}</td>
                      <td>{p.gfr_now_ml_min}</td>
                      <td>{p.age_renal_dx_yr}</td>
                      <td><span className="badge" style={{ background: ACCENT5, fontSize: '0.68em' }}>{p.ckd_stage.split('(')[0].trim()}</span></td>
                      <td>{p.first_symptom.split('(')[0].trim()}</td>
                      <td className="small text-muted">{p.prior_misdiagnosis.split('(')[0].trim()}</td>
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
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="🔬 Kidney Size Distribution" color={ACCENT}>
              {Object.entries(bk.kidney_size_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="📊 CKD Stage Distribution" color={ACCENT4}>
              {Object.entries(bk.ckd_stage_current).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
              ))}
            </Section>
            <Section title="🏥 RRT / Transplant Status" color={ACCENT2}>
              {Object.entries(bk.rrt_transplant_status).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
              ))}
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="⚠️ Prior Misdiagnosis" color={ACCENT7}>
              {Object.entries(bk.prior_misdiagnosis).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
            <Section title="📅 Age at Renal Diagnosis" color={ACCENT3}>
              {Object.entries(bk.age_at_renal_dx_tiers).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
            <Section title="💧 Urine Osmolality (Concentrating Defect)" color={ACCENT5}>
              {Object.entries(bk.urine_osmolality_tiers).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
              ))}
            </Section>
            <Section title="📉 GFR Slope (Progression Rate)" color={ACCENT4}>
              {Object.entries(bk.gfr_slope_tiers).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
              ))}
            </Section>
          </div>

          <div className="col-12">
            <Section title="🌍 Ethnicity Distribution" color={ACCENT6}>
              <div className="row">
                {Object.entries(bk.ethnicity).map(([k, v]) => (
                  <div key={k} className="col-md-6">
                    <Bar label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                  </div>
                ))}
              </div>
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="🌱 Growth Status (Paediatric CKD)" color={ACCENT8}>
              {Object.entries(bk.growth_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="🩺 First Symptom at Presentation" color={ACCENT}>
              {Object.entries(bk.first_symptom_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
          </div>

          <div className="col-12">
            <Section title="🧬 GLIS2 Allele Distribution (NPHP7 Genotypes)" color={ACCENT3}>
              {Object.entries(bk.gene_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 2: Genetics & Subtype ── */}
      {tab === 2 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="🧬 Genetic Architecture — GLIS2/NPHP7" color={ACCENT}>
              {df.genetic_architecture && Object.entries(df.genetic_architecture).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <div className="small fw-bold" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}</div>
                  <div className="small text-muted">{v}</div>
                </div>
              ))}
            </Section>

            <Section title="📌 Founder Variants" color={ACCENT3}>
              {df.founder_variants && df.founder_variants.map((v, i) => (
                <div key={i} className="mb-1 small">
                  <Badge text={i + 1} color={ACCENT3} /> {v}
                </div>
              ))}
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="🔍 Differential Diagnosis Table" color={ACCENT7}>
              {df.ddx_table && Object.entries(df.ddx_table).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT7 + '08', border: `1px solid ${ACCENT7}30` }}>
                  <div className="small fw-bold" style={{ color: ACCENT7 }}>{k}</div>
                  <div className="small text-muted">{v}</div>
                </div>
              ))}
            </Section>
          </div>

          <div className="col-12">
            <Section title="📊 NPHP Subtype Comparison — NPHP7 Position" color={ACCENT5}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: ACCENT5 + '15' }}>
                    <tr><th>Subtype</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {df.nphp_comparison && Object.entries(df.nphp_comparison).map(([k, v]) => (
                      <tr key={k} style={k.includes('★') ? { background: ACCENT + '12', fontWeight: 700 } : {}}>
                        <td style={k.includes('★') ? { color: ACCENT } : {}}>{k}</td>
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

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="📋 Disease Summary" color={ACCENT}>
              <table className="table table-sm small mb-2">
                <tbody>
                  {[
                    ['Disease', df.disease],
                    ['OMIM Gene', df.omim_gene],
                    ['OMIM Disease', df.omim_disease],
                    ['Chromosome', df.chromosome],
                    ['Inheritance', df.inheritance],
                    ['Prevalence', df.prevalence],
                  ].map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-muted" style={{ width: '35%' }}>{k}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>

            <Section title="⚙️ Mechanism" color={ACCENT3}>
              <p className="small">{df.mechanism}</p>
            </Section>

            <Section title="💊 Treatment" color={ACCENT2}>
              {df.treatment && Object.entries(df.treatment).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <div className="small fw-bold" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="small text-muted">{v}</div>
                </div>
              ))}
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="🔑 Key Clinical Features" color={ACCENT4}>
              {df.key_clinical_features && Object.entries(df.key_clinical_features).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT4 + '08', border: `1px solid ${ACCENT4}25` }}>
                  <div className="small fw-bold" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="small">{v}</div>
                </div>
              ))}
            </Section>

            <Section title="🏥 Diagnostic Criteria" color={ACCENT5}>
              {df.diagnostic_criteria && Object.entries(df.diagnostic_criteria).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <div className="small fw-bold" style={{ color: ACCENT5 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="small text-muted">{v}</div>
                </div>
              ))}
            </Section>

            <Section title="📈 Prognosis" color={ACCENT2}>
              <p className="small">{df.prognosis}</p>
            </Section>

            <Section title="📝 Cohort Note" color={ACCENT6}>
              <p className="small text-muted">{df.cohort_note}</p>
            </Section>
          </div>
        </div>
      )}

      {/* Footer nav */}
      <div className="mt-4 pt-3 border-top d-flex gap-3 flex-wrap small">
        <Link href="/nphp6" style={{ color: ACCENT }}>← NPHP6 / CEP290 (SLS6)</Link>
        <Link href="/" style={{ color: ACCENT5 }}>Home</Link>
      </div>
    </div>
  );
}
