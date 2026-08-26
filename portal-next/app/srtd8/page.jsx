'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'WDR60 Dynein-2 & Retrograde IFT', 'Definitions'];

// SRTD8 colour scheme — WDR60 / dynein-2 intermediate chain / narrow thorax / Jeune ATD8
const ACCENT  = '#1565c0';   // deep blue — WDR60 β-propeller; dynein-2 structural subunit
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax / neonatal respiratory; severity
const ACCENT3 = '#2e7d32';   // deep green — renal TIN; secondary ESRD; transplant outcome
const ACCENT4 = '#4a148c';   // deep purple — retinal rod-cone dystrophy; secondary
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic fibrosis; ductal plate malformation
const ACCENT6 = '#37474f';   // dark slate — β-propeller WD40 domain; molecular architecture
const ACCENT7 = '#f57f17';   // amber — misdiagnosis alerts; EM club cilia; diagnostic
const ACCENT8 = '#880e4f';   // deep pink — polydactyly; postaxial; VEPTR surgery

const SEED = 383;

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

export default function SRTD8Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/srtd8/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd8/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd8/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error)   return <div className="alert alert-danger m-4">API error: {error}</div>;
  if (!ov)     return null;

  const k = ov.kpis || {};

  return (
    <div className="container-fluid py-3 px-3" style={{ maxWidth: 1400 }}>
      {/* Header */}
      <div className="mb-3">
        <div className="d-flex align-items-center gap-2 flex-wrap mb-1">
          <span className="badge" style={{ background: ACCENT, fontSize: '0.8em' }}>SRTD8</span>
          <span className="badge" style={{ background: ACCENT2, fontSize: '0.8em' }}>ATD8</span>
          <span className="badge bg-secondary" style={{ fontSize: '0.8em' }}>OMIM #615503</span>
          <span className="badge bg-secondary" style={{ fontSize: '0.8em' }}>WDR60 *615462</span>
          <span className="badge bg-secondary" style={{ fontSize: '0.8em' }}>7q36.3</span>
          <span className="badge bg-secondary" style={{ fontSize: '0.8em' }}>Autosomal Recessive</span>
          <Link href="/" className="btn btn-sm btn-outline-secondary ms-auto" style={{ fontSize: '0.75em' }}>← Portal Home</Link>
        </div>
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          WDR60 Short-Rib Thoracic Dysplasia 8 (SRTD8 / Jeune ATD8)
        </h4>
        <div className="text-muted small mt-1">
          <strong>WDR60</strong> — WD Repeat Domain 60 · 1,173 aa · 7q36.3 · Dynein-2 WD40 β-propeller intermediate chain ·
          Second most common dynein-2-subunit SRTD gene (~5–10%) after DYNC2H1/SRTD3 (~50%) ·
          Retrograde IFT failure → Hedgehog signalling failure → <strong>narrow thorax (primary pathognomonic)</strong> ·
          Seed-{SEED} · 40-patient cohort
        </div>
      </div>

      {/* KPI strip */}
      <div className="row g-2 mb-3">
        <KPI label="Cohort N"           value={k.cohort_size}        color={ACCENT}  />
        <KPI label="Thorax Severe"      value={`${k.thorax_severe_n} (${k.thorax_severe_pct}%)`} color={ACCENT2} />
        <KPI label="Polydactyly"        value={`${k.polydactyly_n} (${k.polydactyly_pct}%)`}    color={ACCENT8} />
        <KPI label="Renal Involvement"  value={`${k.renal_any_n} (${k.renal_any_pct}%)`}         color={ACCENT3} />
        <KPI label="Retinal Dystrophy"  value={`${k.retinal_any_n} (${k.retinal_any_pct}%)`}     color={ACCENT4} />
        <KPI label="Hepatic CHF"        value={`${k.hepatic_chf_n} (${k.hepatic_chf_pct}%)`}    color={ACCENT5} />
        <KPI label="VEPTR Surgery"      value={`${k.veptr_any_n} (${k.veptr_any_pct}%)`}         color={ACCENT8} />
        <KPI label="Renal Transplant"   value={k.transplant_done_n}  color={ACCENT3} />
        <KPI label="Prior Misdiagnosis" value={`${k.misdiagnosis_n} (${k.misdiagnosis_pct}%)`}  color={ACCENT7} />
      </div>

      {/* Alert banners */}
      <Alert color={ACCENT2}>
        <strong>PRIMARY FEATURE — NARROW THORAX:</strong> Short horizontal ribs, narrow bell-shaped chest, neonatal respiratory failure.
        This is the <em>presenting</em> and <em>pathognomonic</em> feature — NOT secondary. Renal TIN and retinal dystrophy occur only in survivors.
        Contrast NPHP1-20 where ESRD is the primary feature.
      </Alert>
      <Alert color={ACCENT}>
        <strong>DYNEIN-2 SUBUNIT — WDR60:</strong> WDR60 is the WD40 β-propeller intermediate chain of cytoplasmic dynein-2. It sits
        OPPOSITE WDR34 (SRTD11) on the dynein-2 tail and directly contacts TCTEX1D2 (SRTD17) and DYNC2H1 (SRTD3 heavy chain).
        WDR60 loss → dynein-2 destabilisation → retrograde IFT failure → IFT-B pile-up at ciliary tips ("club/bulging" cilia on EM).
      </Alert>
      <Alert color={ACCENT7}>
        <strong>DIAGNOSTIC EM FINDING:</strong> Club/bulging ciliary tips on electron microscopy of nasal brush biopsy — IFT-B cargo
        accumulation at the ciliary tip. Seen in all dynein-2 subunit SRTDs (SRTD3/8/11/15/17). NOT seen in TZ-type NPHP or PCD (dynein-arm defect).
      </Alert>
      <Alert color={ACCENT3}>
        <strong>NO SITUS INVERSUS · NO JOUBERT MTS:</strong> WDR60 functions only in non-motile primary cilia (9+0) — not in nodal motile cilia.
        Renal transplant is <strong>CURATIVE</strong> (no recurrence — cell-autonomous IFT defect; donor cilia are normal).
      </Alert>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ───────────────────────────────────────────────────── */}
      {tab === 0 && (
        <div className="row g-3">
          {/* Gene + Disease card */}
          <div className="col-md-6">
            <div className="card h-100 shadow-sm">
              <div className="card-header fw-bold" style={{ background: ACCENT + '22', color: ACCENT }}>
                WDR60 Gene & SRTD8 Disease Card
              </div>
              <div className="card-body small">
                <table className="table table-sm table-borderless mb-0">
                  <tbody>
                    <tr><td className="fw-bold text-nowrap">Gene</td><td>WDR60 (*615462)</td></tr>
                    <tr><td className="fw-bold text-nowrap">Chromosome</td><td>7q36.3</td></tr>
                    <tr><td className="fw-bold text-nowrap">Protein</td><td>1,173 aa · 7-blade WD40 β-propeller + DYNC2H1-tail binding + TCTEX1D2 interface</td></tr>
                    <tr><td className="fw-bold text-nowrap">Function</td><td>Dynein-2 WD-repeat intermediate chain; stabilises dynein-2 tail complex alongside WDR34 (SRTD11); retrograde IFT</td></tr>
                    <tr><td className="fw-bold text-nowrap">Disease</td><td>SRTD8 / ATD8 (#615503)</td></tr>
                    <tr><td className="fw-bold text-nowrap">Inheritance</td><td>Autosomal Recessive (biallelic LOF)</td></tr>
                    <tr><td className="fw-bold text-nowrap">Prevalence</td><td>~1/200,000–500,000 · ~50–75 families worldwide (2026)</td></tr>
                    <tr><td className="fw-bold text-nowrap">Freq in SRTD</td><td>~5–10% of molecularly confirmed SRTD (2nd most common dynein-2-subunit gene)</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Mechanism card */}
          <div className="col-md-6">
            <div className="card h-100 shadow-sm">
              <div className="card-header fw-bold" style={{ background: ACCENT6 + '22', color: ACCENT6 }}>
                Pathomechanism — WDR60 Loss → Retrograde IFT Failure
              </div>
              <div className="card-body small">
                <ol className="mb-0 ps-3" style={{ lineHeight: 1.8 }}>
                  <li>WDR60 β-propeller lost → dynein-2 tail destabilised (TCTEX1D2/DYNLRB1 light chains detach)</li>
                  <li>Retrograde IFT fails → IFT-B anterograde cargo accumulates at ciliary tip</li>
                  <li>EM: "club/bulging" ciliary tip — IFT-B pile-up (diagnostic finding)</li>
                  <li>Hedgehog transducers (PTCH1, SMO, GLI3) cannot return from tip → Hh signalling impaired</li>
                  <li>GLI3 processing to repressor (GLI3R) fails → Hh target genes dysregulated</li>
                  <li>Chondrocytes: Ihh/Shh failure → short ribs, narrow thorax, short limbs, ± polydactyly</li>
                  <li>Renal tubular cells (survivors): TIN + cysts → ESRD (secondary)</li>
                  <li>Retinal photoreceptors (survivors): rod-cone dystrophy (~20–25%)</li>
                  <li>Biliary cholangiocytes (survivors): CHF (~8%)</li>
                </ol>
              </div>
            </div>
          </div>

          {/* Dynein-2 complex subunit table */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold" style={{ background: ACCENT + '22', color: ACCENT }}>
                Cytoplasmic Dynein-2 Complex — All SRTD-Associated Subunits
              </div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead style={{ background: ACCENT + '11' }}>
                      <tr>
                        <th>Subunit</th><th>Role in Dynein-2</th><th>SRTD#</th><th>OMIM Gene</th><th>Chromosome</th><th>Frequency in SRTD</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(ov.dynein2_subunit_table || []).map((r, i) => (
                        <tr key={i} style={r.subunit === 'WDR60' ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                          <td>{r.subunit}{r.subunit === 'WDR60' ? ' ← THIS DISEASE' : ''}</td>
                          <td className="small">{r.role}</td>
                          <td>{r.srtd}</td>
                          <td>{r.omim_gene}</td>
                          <td>{r.chr}</td>
                          <td>{r.freq}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Age at diagnosis + sex */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT2 + '22', color: ACCENT2 }}>Age at Diagnosis</div>
              <div className="card-body small">
                {Object.entries(ov.age_distribution || {}).map(([k2, v]) => (
                  <Bar key={k2} label={k2.replace(/_/g, ' ')} value={v} max={COHORT_N} color={ACCENT2} />
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT3 + '22', color: ACCENT3 }}>Sex Distribution</div>
              <div className="card-body small">
                {Object.entries(ov.sex_split || {}).map(([s, v]) => (
                  <Bar key={s} label={s} value={v} max={COHORT_N} color={ACCENT3} />
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT7 + '22', color: ACCENT7 }}>Key Clinical Alerts</div>
              <div className="card-body small">
                <ul className="mb-0 ps-3" style={{ lineHeight: 1.8 }}>
                  <li>WDR60 = <strong>2nd most common dynein-2-subunit SRTD</strong> (~5–10%)</li>
                  <li>EM club/bulging cilia tip = diagnostic IFT-B pile-up</li>
                  <li>Renal USS annually from age 5</li>
                  <li>Annual ERG from age 3 (rod-cone ~20–25%)</li>
                  <li>VEPTR/MAGEC for thoracic insufficiency</li>
                  <li><strong>NO</strong> situs inversus · <strong>NO</strong> Joubert MTS</li>
                  <li>Renal transplant = <strong>CURATIVE</strong> (no recurrence)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 1: Diagnostic Breakdown ───────────────────────────────────────── */}
      {tab === 1 && bk && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT2 + '22', color: ACCENT2 }}>Thorax Severity Distribution</div>
              <div className="card-body small">
                {(bk.thorax_distribution || []).map(r => <Bar key={r.label} label={r.label} value={r.n} max={COHORT_N} color={ACCENT2} />)}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT8 + '22', color: ACCENT8 }}>Polydactyly Distribution</div>
              <div className="card-body small">
                {(bk.polydactyly_distribution || []).map(r => <Bar key={r.label} label={r.label} value={r.n} max={COHORT_N} color={ACCENT8} />)}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT3 + '22', color: ACCENT3 }}>Renal Status Distribution</div>
              <div className="card-body small">
                {(bk.renal_distribution || []).map(r => <Bar key={r.label} label={r.label} value={r.n} max={COHORT_N} color={ACCENT3} />)}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT3 + '22', color: ACCENT3 }}>CKD Stage at Last Review</div>
              <div className="card-body small">
                {(bk.ckd_stage_distribution || []).map(r => <Bar key={r.label} label={r.label} value={r.n} max={COHORT_N} color={ACCENT3} />)}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT4 + '22', color: ACCENT4 }}>Retinal Status</div>
              <div className="card-body small">
                {(bk.retinal_distribution || []).map(r => <Bar key={r.label} label={r.label} value={r.n} max={COHORT_N} color={ACCENT4} />)}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT5 + '22', color: ACCENT5 }}>Hepatic / CHF Status</div>
              <div className="card-body small">
                {(bk.hepatic_distribution || []).map(r => <Bar key={r.label} label={r.label} value={r.n} max={COHORT_N} color={ACCENT5} />)}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT8 + '22', color: ACCENT8 }}>VEPTR Surgical Status</div>
              <div className="card-body small">
                {(bk.veptr_distribution || []).map(r => <Bar key={r.label} label={r.label} value={r.n} max={COHORT_N} color={ACCENT8} />)}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT + '22', color: ACCENT }}>Allele Class</div>
              <div className="card-body small">
                {(bk.allele_class_summary || []).map(r => <Bar key={r.label} label={r.label} value={r.n} max={COHORT_N} color={ACCENT} />)}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT7 + '22', color: ACCENT7 }}>Misdiagnosis History</div>
              <div className="card-body small">
                {(bk.misdiagnosis_distribution || []).map(r => <Bar key={r.label} label={r.label} value={r.n} max={COHORT_N} color={ACCENT7} />)}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT6 + '22', color: ACCENT6 }}>Presenting Symptom / Route to Diagnosis</div>
              <div className="card-body small">
                {(bk.presentation_distribution || []).map(r => <Bar key={r.label} label={r.label} value={r.n} max={COHORT_N} color={ACCENT6} />)}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT + '22', color: ACCENT }}>Respiratory Management</div>
              <div className="card-body small">
                {(bk.respiratory_management || []).map(r => <Bar key={r.label} label={r.label} value={r.n} max={COHORT_N} color={ACCENT} />)}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small" style={{ background: ACCENT3 + '22', color: ACCENT3 }}>Renal Treatment</div>
              <div className="card-body small">
                {(bk.treatment_renal || []).map(r => <Bar key={r.label} label={r.label} value={r.n} max={COHORT_N} color={ACCENT3} />)}
              </div>
            </div>
          </div>

          {/* Top variants */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold small" style={{ background: ACCENT + '22', color: ACCENT }}>Most Frequent WDR60 Variants (cohort)</div>
              <div className="card-body small">
                <div className="row g-2">
                  {(bk.top_variants || []).map(r => (
                    <div key={r.variant} className="col-12 col-md-6">
                      <Bar label={r.variant} value={r.n} max={COHORT_N} color={ACCENT} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Ethnicity */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold small" style={{ background: ACCENT6 + '22', color: ACCENT6 }}>Ethnicity Distribution</div>
              <div className="card-body small">
                <div className="row g-2">
                  {(bk.ethnicity_distribution || []).map(r => (
                    <div key={r.ethnicity} className="col-12 col-md-6">
                      <Bar label={r.ethnicity} value={r.n} max={COHORT_N} color={ACCENT6} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Sample patients */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold small" style={{ background: ACCENT6 + '22', color: ACCENT6 }}>Sample Patient Records (first 8)</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0 small">
                    <thead style={{ background: ACCENT6 + '11' }}>
                      <tr>
                        <th>ID</th><th>Sex</th><th>Age Dx</th><th>Ethnicity</th>
                        <th>Thorax</th><th>Renal</th><th>Retinal</th><th>Allele</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bk.patients_sample || []).map(p => (
                        <tr key={p.id}>
                          <td><code>{p.id}</code></td>
                          <td>{p.sex}</td>
                          <td>{p.age_at_dx}y</td>
                          <td className="small">{p.ethnicity.split('(')[0].trim()}</td>
                          <td className="small">{p.thorax.split('—')[0].trim()}</td>
                          <td className="small">{p.renal.split('—')[0].trim()}</td>
                          <td className="small">{p.retinal.split('(')[0].trim()}</td>
                          <td className="small">{p.allele_class.split('(')[0].trim()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 2: WDR60 Dynein-2 & Retrograde IFT ───────────────────────────── */}
      {tab === 2 && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: ACCENT + '22', color: ACCENT }}>
                WDR60 Protein Domain Architecture (1,173 aa)
              </div>
              <div className="card-body small">
                {[
                  { domain: 'WD repeat 1–7 / β-propeller (aa ~50–620)', color: ACCENT,  desc: 'Seven-bladed WD40 β-propeller; protein–protein interaction scaffold within dynein-2 tail; WD3–WD6 harbour most pathogenic missense variants; blade missense → moderate SRTD8 (~40–60% residual retrograde activity)' },
                  { domain: 'DYNC2H1-tail binding region (aa ~620–850)', color: ACCENT2, desc: 'Central unstructured region + amphipathic helix contacting DYNC2H1 stem/tail; variants here → severe dynein-2 complex destabilisation' },
                  { domain: 'TCTEX1D2 / DYNLRB1 interface (aa ~850–950)', color: ACCENT4, desc: 'C-terminal α-helical region binding light chains TCTEX1D2 (SRTD17) and DYNLRB1; truncating variants beyond aa 850 → complete light-chain detachment → near-null dynein-2' },
                  { domain: 'C-terminal disordered tail (aa ~950–1173)', color: ACCENT6, desc: 'Low-complexity region; hypomorphic missense here → mildest SRTD8 (β-propeller intact; partial complex assembly preserved)' },
                ].map(d => (
                  <div key={d.domain} className="mb-3 p-2 rounded" style={{ background: d.color + '12', borderLeft: `4px solid ${d.color}` }}>
                    <div className="fw-bold mb-1" style={{ color: d.color }}>{d.domain}</div>
                    <div className="text-muted">{d.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: ACCENT6 + '22', color: ACCENT6 }}>
                Dynein-2 Complex Position of WDR60
              </div>
              <div className="card-body small">
                <div className="p-2 rounded mb-3" style={{ background: ACCENT + '12', border: `1px solid ${ACCENT}33`, fontFamily: 'monospace', fontSize: '0.8em', lineHeight: 1.9 }}>
                  {`DYNEIN-2 RETROGRADE IFT COMPLEX`}<br/>
                  {`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`}<br/>
                  {`[DYNC2H1 ×2]  AAA+ ring motor (SRTD3) — force generator`}<br/>
                  {`      |`}<br/>
                  {`[DYNC2LI1]    light intermediate chain (SRTD15)`}<br/>
                  {`      |`}<br/>
                  {`   ┌──┴──┐`}<br/>
                  {`[WDR34] [WDR60] ← β-propeller pair (SRTD11 | SRTD8)`}<br/>
                  {`            |`}<br/>
                  {`       [TCTEX1D2] light chain (SRTD17)`}<br/>
                  {`       [DYNLRB1/2] roadblock chains`}<br/>
                  {`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`}<br/>
                  {`WDR60 loss → tail destabilises → TCTEX1D2+DYNLRB detach`}<br/>
                  {`→ retrograde IFT fails → club cilia tips → Hh failure`}
                </div>
                <div className="text-muted">
                  <strong>WDR34 vs WDR60:</strong> Both are WD40 β-propeller intermediate chains on opposite sides of the
                  dynein-2 tail. WDR34 (SRTD11) sits adjacent to DYNC2LI1; WDR60 (SRTD8) sits adjacent to TCTEX1D2.
                  Together they form the structural "arms" that stabilise dynein-2 cargo adaptor contacts during retrograde IFT.
                </div>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: ACCENT2 + '22', color: ACCENT2 }}>
                Retrograde IFT Failure — Step-by-Step
              </div>
              <div className="card-body small">
                <ol className="mb-0 ps-3" style={{ lineHeight: 2 }}>
                  <li>IFT trains (IFT-A + IFT-B + cargo) assembled at ciliary base</li>
                  <li>Anterograde movement (base→tip) powered by kinesin-2 (KIF3A/KIF3B/KAP)</li>
                  <li>At ciliary tip: IFT-B unloads tubulin + membrane cargoes</li>
                  <li>IFT-A + Hh transducers (PTCH1, SMO, GLI3) loaded onto dynein-2 for return</li>
                  <li><strong>WDR60 absent → dynein-2 unstable → retrograde stalls</strong></li>
                  <li>IFT-B + signalling cargo pile up at tip → "club" cilia on EM</li>
                  <li>PTCH1 cannot leave tip → SMO blocked → GLI3 full-length accumulates</li>
                  <li>GLI3 repressor (GLI3R) not generated → Hh targets dysregulated</li>
                  <li>Chondrocytes: Ihh/Shh failure → short ribs, narrow thorax, polydactyly</li>
                  <li>Renal tubules (survivors): TIN + cysts → ESRD</li>
                </ol>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: ACCENT7 + '22', color: ACCENT7 }}>
                Key Pathogenic WDR60 Variants
              </div>
              <div className="card-body small p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead style={{ background: ACCENT7 + '11' }}>
                    <tr><th>Variant</th><th>Domain</th><th>Consequence</th><th>Ethnicity</th></tr>
                  </thead>
                  <tbody>
                    {df && (df.key_variants || []).map(v => (
                      <tr key={v.variant}>
                        <td><code>{v.variant}</code></td>
                        <td className="small">{v.domain}</td>
                        <td className="small">{v.consequence}</td>
                        <td className="small">{v.ethnicity}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold" style={{ background: ACCENT3 + '22', color: ACCENT3 }}>
                SRTD8 vs NPHP — Mechanistic Distinction
              </div>
              <div className="card-body small">
                <div className="row g-3">
                  <div className="col-md-6">
                    <div className="p-2 rounded" style={{ background: ACCENT2 + '12', borderLeft: `4px solid ${ACCENT2}` }}>
                      <div className="fw-bold mb-1" style={{ color: ACCENT2 }}>SRTD8 / WDR60 (THIS DISEASE)</div>
                      <ul className="mb-0 ps-3">
                        <li><strong>Primary:</strong> NARROW THORAX (neonatal lethal risk)</li>
                        <li>Mechanism: retrograde IFT motor chain loss → dynein-2 collapse</li>
                        <li>Renal TIN + retinal: SECONDARY in survivors</li>
                        <li>Non-motile primary cilia (9+0)</li>
                        <li>NO situs inversus · NO Joubert MTS</li>
                        <li>First-line treatment: VEPTR/MAGEC growing rods</li>
                      </ul>
                    </div>
                  </div>
                  <div className="col-md-6">
                    <div className="p-2 rounded" style={{ background: ACCENT3 + '12', borderLeft: `4px solid ${ACCENT3}` }}>
                      <div className="fw-bold mb-1" style={{ color: ACCENT3 }}>NPHP1-20 (Nephronophthisis)</div>
                      <ul className="mb-0 ps-3">
                        <li><strong>Primary:</strong> ESRD (renal is the presenting feature)</li>
                        <li>Mechanism: transition zone (TZ), distal appendage, or IFT-A defect</li>
                        <li>Narrow thorax RARE (only NPHP12/TTC21B, NPHP13/WDR19, ~7–10%)</li>
                        <li>Non-motile primary cilia (9+0)</li>
                        <li>NO situs inversus (most types)</li>
                        <li>First-line treatment: renal transplant (curative)</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold" style={{ background: ACCENT8 + '22', color: ACCENT8 }}>
                Differential Diagnosis Table
              </div>
              <div className="card-body p-0 small">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead style={{ background: ACCENT8 + '11' }}>
                      <tr><th>Disease</th><th>Key Distinguishing Feature</th></tr>
                    </thead>
                    <tbody>
                      {df && (df.ddx_table || []).map(r => (
                        <tr key={r.disease}>
                          <td className="fw-bold text-nowrap">{r.disease}</td>
                          <td>{r.key_difference}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 3: Definitions ─────────────────────────────────────────────────── */}
      {tab === 3 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: ACCENT + '22', color: ACCENT }}>Gene Card — WDR60</div>
              <div className="card-body small">
                <table className="table table-sm table-borderless mb-0">
                  <tbody>
                    {Object.entries(df.gene_card || {}).map(([k2, v]) => (
                      <tr key={k2}><td className="fw-bold text-capitalize text-nowrap" style={{ width: '35%' }}>{k2.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: ACCENT2 + '22', color: ACCENT2 }}>Disease Card — SRTD8</div>
              <div className="card-body small">
                <table className="table table-sm table-borderless mb-0">
                  <tbody>
                    {Object.entries(df.disease_card || {}).map(([k2, v]) => (
                      <tr key={k2}><td className="fw-bold text-capitalize text-nowrap" style={{ width: '35%' }}>{k2.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold" style={{ background: ACCENT6 + '22', color: ACCENT6 }}>Mechanism Glossary</div>
              <div className="card-body small">
                <div className="row g-2">
                  {(df.mechanism_glossary || []).map(g => (
                    <div key={g.term} className="col-md-6">
                      <div className="p-2 rounded" style={{ background: ACCENT6 + '0d', borderLeft: `3px solid ${ACCENT6}` }}>
                        <div className="fw-bold mb-1" style={{ color: ACCENT6 }}>{g.term}</div>
                        <div className="text-muted">{g.definition}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: ACCENT7 + '22', color: ACCENT7 }}>Diagnostic Workup</div>
              <div className="card-body small">
                <ol className="mb-0 ps-3" style={{ lineHeight: 1.9 }}>
                  {(df.diagnostic_workup || []).map((step, i) => <li key={i}>{step}</li>)}
                </ol>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: ACCENT3 + '22', color: ACCENT3 }}>Treatment Summary</div>
              <div className="card-body small">
                <ol className="mb-0 ps-3" style={{ lineHeight: 1.9 }}>
                  {(df.treatment_summary || []).map((step, i) => <li key={i}>{step}</li>)}
                </ol>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="text-muted small mt-4 pt-2 border-top">
        SRTD8 · WDR60 (*615462) · OMIM #615503 · 7q36.3 · Dynein-2 WD40 β-propeller intermediate chain ·
        40-patient cohort · Seed {SEED} · Autosomal Recessive · ~1/200,000–500,000 · ~50–75 families worldwide (2026) ·
        Narrow thorax = primary pathognomonic feature · VEPTR/MAGEC surgical · Renal transplant curative · No disease-modifying therapy 2026
      </div>
    </div>
  );
}
