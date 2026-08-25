'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Respiratory', 'Treatments & Genetics', 'Definitions'];

// Kagami-Ogata colour scheme — amber/orange (severe/lethal neonatal; coat-hanger ribs)
const ACCENT  = '#e65100';   // deep orange — coat-hanger ribs / severity / lethal
const ACCENT2 = '#bf360c';   // dark red-orange — neonatal mortality
const ACCENT3 = '#1b5e20';   // deep green — KEY POSITIVES / surveillance
const ACCENT4 = '#b71c1c';   // deep red — absolute risks / hepatoblastoma
const ACCENT5 = '#0d47a1';   // dark blue — genetics / methylation
const ACCENT6 = '#4a148c';   // purple — imprinting (same locus as Temple, opposite parent)
const ACCENT7 = '#37474f';   // dark slate — epidemiology
const ACCENT8 = '#004d40';   // teal — Temple Syndrome contrast

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

export default function KagamiPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/kagami/overview`).then(r => r.json()),
      fetch(`${API}/api/kagami/breakdown`).then(r => r.json()),
      fetch(`${API}/api/kagami/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error) return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  const kpi = overview?.kpis || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3" style={{ borderLeft: `6px solid ${ACCENT}`, paddingLeft: 14 }}>
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>🧬 Kagami-Ogata Syndrome (KOS14)</h4>
        <div className="text-muted small">
          DLK1 / RTL1 / MEG3 · 14q32.3 · Genomic Imprinting (Maternal LOF) · OMIM #608149
          <span className="ms-3 badge" style={{ background: ACCENT6 }}>Imprinting Disorder</span>
          <span className="ms-2 badge" style={{ background: ACCENT2 }}>Opposite of Temple Syndrome</span>
        </div>
        <div className="mt-1">
          <small>
            <span className="badge me-2" style={{ background: ACCENT }}>Coat-Hanger Ribs = PATHOGNOMONIC</span>
            <span className="badge me-2" style={{ background: ACCENT4 }}>Hepatoblastoma Risk ~5%</span>
            <span className="badge me-2" style={{ background: ACCENT7 }}>~1:50,000–100,000</span>
            <span className="badge me-2" style={{ background: ACCENT2 }}>Neonatal Mortality ~30%</span>
          </small>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottom: `3px solid ${ACCENT}` } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && (
        <div>
          {/* Critical alerts */}
          <Alert color={ACCENT2}>
            <strong>⚠️ PATHOGNOMONIC:</strong> Coat-hanger ribs (horizontal bell-shaped thorax on CXR) in &gt;95% —
            the sine qua non of Kagami-Ogata Syndrome. Absent in Temple Syndrome (same 14q32.3 locus, opposite parent).
          </Alert>
          <Alert color={ACCENT4}>
            <strong>🔴 HEPATOBLASTOMA:</strong> MEG3 absent → reduced p53 → ~5% lifetime hepatoblastoma risk.
            AFP q3-6mo + liver ultrasound q6mo MANDATORY in ALL survivors. VPA moderate risk — confounds AFP.
          </Alert>
          <Alert color={ACCENT}>
            <strong>🧬 SAME LOCUS AS TEMPLE SYNDROME:</strong> 14q32.3 — Maternal LOF (KOS14) vs Paternal LOF (Temple).
            IG-DMR: 100% methylated in KOS14 (both alleles paternal pattern) vs 0% in Temple Syndrome.
            SNP array ALONE misses ~50% (UPD + epimutation) — methylation test FIRST.
          </Alert>

          {/* KPIs */}
          <div className="row mb-3">
            <KPI label="Neonatal Mortality" value={`${kpi.neonatal_mortality_pct ?? '—'}%`} color={ACCENT2} />
            <KPI label="Coat-Hanger Ribs" value={`${kpi.coat_hanger_ribs_pct ?? '—'}%`} color={ACCENT} />
            <KPI label="Polyhydramnios" value={`${kpi.polyhydramnios_pct ?? '—'}%`} color={ACCENT5} />
            <KPI label="Macrosomia (LGA)" value={`${kpi.macrosomia_pct ?? '—'}%`} color={ACCENT6} />
            <KPI label="Preterm" value={`${kpi.preterm_pct ?? '—'}%`} color={ACCENT7} />
            <KPI label="Epilepsy (All)" value={`${kpi.epilepsy_pct ?? '—'}%`} color={ACCENT3} />
            <KPI label="Hepatoblastoma" value={`${kpi.hepatoblastoma_detected_pct ?? '—'}%`} color={ACCENT4} />
            <KPI label="Placentomegaly" value={`${kpi.placentomegaly_pct ?? '—'}%`} color={ACCENT7} />
            <KPI label="DLK1 Serum (% norm)" value={`${kpi.avg_dlk1_serum_pct_of_normal ?? '—'}%`} color={ACCENT} />
            <KPI label="IG-DMR Methylation" value={`${kpi.avg_igdmr_methylation_pct ?? '—'}%`} color={ACCENT5} />
          </div>

          <div className="row">
            {/* Mechanism distribution */}
            <div className="col-md-4 mb-3">
              <Section title="Genetic Mechanism Distribution" color={ACCENT5}>
                {overview?.mechanism_distribution && Object.entries(overview.mechanism_distribution).map(([k, v]) => (
                  <div key={k} className="mb-1">
                    <div className="d-flex justify-content-between small">
                      <span>{k.replace(/_/g,' ')}</span><span className="fw-bold">{v}/40</span>
                    </div>
                    <div className="progress" style={{ height: 6 }}>
                      <div className="progress-bar" style={{ width: `${v*2.5}%`, background: ACCENT5 }} />
                    </div>
                  </div>
                ))}
              </Section>
            </div>

            {/* Phenotype distribution */}
            <div className="col-md-4 mb-3">
              <Section title="Phenotypic Class Distribution" color={ACCENT2}>
                {overview?.phenotype_distribution && Object.entries(overview.phenotype_distribution).map(([k, v]) => (
                  <div key={k} className="mb-1">
                    <div className="d-flex justify-content-between small">
                      <span>{k}</span><span className="fw-bold">{v}/40</span>
                    </div>
                    <div className="progress" style={{ height: 6 }}>
                      <div className="progress-bar" style={{ width: `${v*2.5}%`, background: k === 'Severe-Lethal' ? ACCENT2 : k === 'Severe-Surviving' ? ACCENT : ACCENT3 }} />
                    </div>
                  </div>
                ))}
              </Section>
            </div>

            {/* Seizure types */}
            <div className="col-md-4 mb-3">
              <Section title="Seizure Types (Survivors with Epilepsy)" color={ACCENT3}>
                {overview?.seizure_types && Object.keys(overview.seizure_types).length > 0
                  ? Object.entries(overview.seizure_types).map(([k, v]) => (
                    <div key={k} className="mb-1">
                      <div className="d-flex justify-content-between small">
                        <span>{k}</span><span className="fw-bold">{v}</span>
                      </div>
                      <div className="progress" style={{ height: 6 }}>
                        <div className="progress-bar" style={{ width: `${v*25}%`, background: ACCENT3 }} />
                      </div>
                    </div>
                  ))
                  : <div className="text-muted small">Epilepsy is secondary (HIE); most patients have no seizures</div>
                }
              </Section>
            </div>
          </div>

          {/* Key facts */}
          <Section title="Key Clinical Facts — KOS14" color={ACCENT}>
            <div className="row">
              {(overview?.key_facts || []).map((f, i) => (
                <div key={i} className="col-md-6 mb-1">
                  <small>• {f}</small>
                </div>
              ))}
            </div>
          </Section>

          {/* Imprinting contrast */}
          <Section title="14q32.3 Imprinting: KOS14 vs Temple Syndrome — Opposite Phenotypes at Same Locus" color={ACCENT6}>
            <div className="row">
              <div className="col-md-6">
                <div className="p-2 rounded mb-2" style={{ background: ACCENT + '18', border: `2px solid ${ACCENT}` }}>
                  <strong style={{ color: ACCENT }}>🧬 Kagami-Ogata Syndrome (KOS14)</strong>
                  <ul className="mb-0 small mt-1">
                    <li>MATERNAL LOF at 14q32.3</li>
                    <li>DLK1: 2× excess (paternally expressed, unopposed)</li>
                    <li>RTL1: 2× excess → placentomegaly</li>
                    <li>MEG3: absent → ↓p53 → hepatoblastoma ~5%</li>
                    <li>IG-DMR methylation: 100% (both alleles = paternal)</li>
                    <li>Macrosomia (LGA) + coat-hanger ribs</li>
                    <li>Neonatal respiratory failure, high mortality</li>
                    <li>Epilepsy 15-25% (secondary HIE)</li>
                  </ul>
                </div>
              </div>
              <div className="col-md-6">
                <div className="p-2 rounded mb-2" style={{ background: ACCENT8 + '18', border: `2px solid ${ACCENT8}` }}>
                  <strong style={{ color: ACCENT8 }}>🧬 Temple Syndrome (TS14)</strong>
                  <ul className="mb-0 small mt-1">
                    <li>PATERNAL LOF at 14q32.3</li>
                    <li>DLK1: absent (paternally expressed, no paternal allele)</li>
                    <li>RTL1: absent (paternal)</li>
                    <li>MEG3: biallelic expression (maternal present × 2)</li>
                    <li>IG-DMR methylation: 0-10% (both alleles = maternal)</li>
                    <li>SGA + truncal obesity + CPP in females</li>
                    <li>Neonatal hypotonia, low mortality</li>
                    <li>Epilepsy 20-30% (primary, focal, mild)</li>
                  </ul>
                </div>
              </div>
            </div>
            <Alert color={ACCENT6}>
              <strong>Key:</strong> Same locus (14q32.3), opposite parent → opposite phenotypes.
              This is the cardinal demonstration of genomic imprinting. A deletion at 14q32.3
              inherited from FATHER → Temple Syndrome. Same deletion inherited from MOTHER → KOS14.
            </Alert>
          </Section>
        </div>
      )}

      {/* ── Tab 1: Patients & Respiratory ── */}
      {tab === 1 && (
        <div>
          <Section title="Phenotypic Classes" color={ACCENT2}>
            {(breakdown?.phenotypes || []).map((ph, i) => (
              <div key={i} className="card mb-2 shadow-sm">
                <div className="card-body py-2">
                  <div className="d-flex justify-content-between align-items-center mb-1">
                    <strong style={{ color: ph.group === 'Severe-Lethal' ? ACCENT2 : ph.group === 'Severe-Surviving' ? ACCENT : ACCENT3 }}>
                      {ph.group} ({ph.pct}%)
                    </strong>
                    <span className="badge" style={{ background: ph.group === 'Severe-Lethal' ? ACCENT2 : ACCENT }}>
                      {ph.survival}
                    </span>
                  </div>
                  <div className="small text-muted mb-1">{ph.description}</div>
                  <div className="d-flex flex-wrap gap-1">
                    {(ph.key_features || []).map((f, j) => (
                      <span key={j} className="badge" style={{ background: ACCENT7, fontSize: '0.7rem' }}>{f}</span>
                    ))}
                  </div>
                  {ph.epilepsy_pct > 0 && (
                    <div className="mt-1 small"><strong style={{ color: ACCENT3 }}>Epilepsy in survivors: {ph.epilepsy_pct}%</strong></div>
                  )}
                </div>
              </div>
            ))}
          </Section>

          <Section title="Genetic Mechanisms" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-hover small">
                <thead style={{ background: ACCENT5 + '22' }}>
                  <tr>
                    <th>Mechanism</th><th>Freq %</th><th>First Test</th><th>CN Change</th><th>Recurrence</th><th>DLK1</th><th>RTL1</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.mechanisms || []).map((m, i) => (
                    <tr key={i}>
                      <td><strong>{m.name}</strong><br/><span className="text-muted">{m.notes}</span></td>
                      <td className="fw-bold" style={{ color: ACCENT5 }}>{m.frequency_pct}%</td>
                      <td>{m.first_line_test}</td>
                      <td>{m.cn_change}</td>
                      <td className="fw-bold" style={{ color: m.recurrence_risk_pct >= 50 ? ACCENT2 : ACCENT3 }}>{m.recurrence_risk_pct}%</td>
                      <td></td>
                      <td></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Molecular Subtypes / Variants" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-hover small">
                <thead style={{ background: ACCENT + '22' }}>
                  <tr>
                    <th>Subtype</th><th>Freq %</th><th>Phenotype</th><th>DLK1</th><th>RTL1</th><th>MEG3</th><th>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.variants || []).map((v, i) => (
                    <tr key={i}>
                      <td><strong>{v.variant}</strong></td>
                      <td style={{ color: ACCENT }}>{v.frequency_pct}%</td>
                      <td>{v.phenotype_class}</td>
                      <td className="fw-bold" style={{ color: ACCENT2 }}>{v.dlk1_level}</td>
                      <td className="fw-bold" style={{ color: ACCENT2 }}>{v.rtl1_level}</td>
                      <td className="fw-bold" style={{ color: ACCENT4 }}>{v.meg3_level}</td>
                      <td className="small text-muted">{v.mechanism_detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="EEG Patterns in KOS14 Survivors" color={ACCENT3}>
            <div className="row">
              {(breakdown?.eeg_patterns || []).map((e, i) => (
                <div key={i} className="col-md-6 mb-2">
                  <div className="card h-100 shadow-sm">
                    <div className="card-body py-2">
                      <div className="fw-bold small" style={{ color: ACCENT3 }}>{e.pattern}</div>
                      <div className="small text-muted">{e.context}</div>
                      <div className="small"><strong>Significance:</strong> {e.significance}</div>
                      <span className="badge" style={{ background: ACCENT7, fontSize: '0.65rem' }}>{e.frequency}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Patient Cohort (40 patients, seed-293)" color={ACCENT7}>
            <div className="table-responsive" style={{ maxHeight: 340, overflowY: 'auto' }}>
              <table className="table table-sm table-hover small">
                <thead style={{ background: ACCENT7 + '22', position: 'sticky', top: 0 }}>
                  <tr>
                    <th>ID</th><th>Sex</th><th>Phenotype</th><th>Mechanism</th><th>Coat-Hanger</th>
                    <th>Macrosomia</th><th>Polyhydramnios</th><th>Preterm</th>
                    <th>Epilepsy</th><th>Seizure Type</th><th>AED</th><th>Hepatoblastoma</th>
                    <th>DLK1 %norm</th><th>IG-DMR %</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.patients || []).map((p, i) => (
                    <tr key={i} style={{ opacity: p.survived_neonatal ? 1 : 0.5 }}>
                      <td style={{ color: ACCENT7 }}>{p.id}</td>
                      <td>{p.sex}</td>
                      <td>
                        <span className="badge" style={{
                          background: p.phenotype_group === 'Severe-Lethal' ? ACCENT2 : p.phenotype_group === 'Severe-Surviving' ? ACCENT : ACCENT3,
                          fontSize: '0.65rem'
                        }}>{p.phenotype_group}</span>
                      </td>
                      <td style={{ color: ACCENT5 }}>{p.mechanism}</td>
                      <td style={{ color: p.coat_hanger_ribs ? ACCENT : ACCENT3 }}>{p.coat_hanger_ribs ? '✓' : '–'}</td>
                      <td>{p.macrosomia ? '✓' : '–'}</td>
                      <td>{p.polyhydramnios ? '✓' : '–'}</td>
                      <td>{p.preterm ? '✓' : '–'}</td>
                      <td style={{ color: p.has_epilepsy ? ACCENT3 : ACCENT7 }}>{p.has_epilepsy ? '✓' : '–'}</td>
                      <td>{p.seizure_type || '–'}</td>
                      <td>{p.current_aed || '–'}</td>
                      <td style={{ color: p.hepatoblastoma_detected ? ACCENT4 : ACCENT7 }}>{p.hepatoblastoma_detected ? '⚠️' : '–'}</td>
                      <td className="fw-bold" style={{ color: ACCENT }}>{p.dlk1_serum_pct_of_normal}%</td>
                      <td className="fw-bold" style={{ color: ACCENT5 }}>{p.igdmr_methylation_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="small text-muted mt-1">Greyed rows = neonatal death (Severe-Lethal class)</div>
          </Section>
        </div>
      )}

      {/* ── Tab 2: Treatments & Genetics ── */}
      {tab === 2 && (
        <div>
          <Section title="Treatment Protocols" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-hover small">
                <thead style={{ background: ACCENT3 + '22' }}>
                  <tr>
                    <th>Treatment</th><th>Level</th><th>Indication</th><th>Notes</th><th>CI?</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.treatments || []).map((t, i) => (
                    <tr key={i} style={{ opacity: t.contraindicated ? 0.6 : 1 }}>
                      <td><strong style={{ color: t.contraindicated ? ACCENT4 : ACCENT3 }}>{t.treatment}</strong></td>
                      <td>
                        <span className="badge" style={{
                          background: t.level === 'A' ? ACCENT3 : t.level === 'B' ? ACCENT5 : t.level === 'C' ? ACCENT7 : ACCENT4
                        }}>{t.level}</span>
                      </td>
                      <td>{t.indication}</td>
                      <td className="small text-muted">{t.notes}</td>
                      <td>{t.contraindicated ? <span style={{ color: ACCENT4 }}>🚫 N/A</span> : '–'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Drug Risks in KOS14" color={ACCENT4}>
            <Alert color={ACCENT}>
              <strong>No AED is ABSOLUTELY contraindicated in KOS14</strong> (contrast Angelman: CBZ/OXC absolute CI).
              Epilepsy in KOS14 is secondary (HIE). VPA is MODERATE risk — hepatic monitoring overlaps
              hepatoblastoma AFP surveillance. Prefer LEV or LTG as first-line.
            </Alert>
            <div className="table-responsive">
              <table className="table table-sm table-hover small">
                <thead style={{ background: ACCENT4 + '22' }}>
                  <tr>
                    <th>Drug</th><th>Risk Level</th><th>Risk Type</th><th>Mechanism</th><th>Recommendation</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.drug_risks || []).map((r, i) => (
                    <tr key={i}>
                      <td><strong style={{ color: r.risk_level === 'HIGH' ? ACCENT2 : r.risk_level === 'MODERATE' ? ACCENT : ACCENT3 }}>{r.drug}</strong></td>
                      <td>
                        <span className="badge" style={{
                          background: r.risk_level === 'HIGH' ? ACCENT2 : r.risk_level === 'MODERATE' ? ACCENT : ACCENT3
                        }}>{r.risk_level}</span>
                      </td>
                      <td>{r.risk_type}</td>
                      <td className="small text-muted">{r.mechanism}</td>
                      <td className="small">{r.recommendation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Hepatoblastoma Surveillance Protocol" color={ACCENT4}>
            <Alert color={ACCENT4}>
              <strong>🔴 MANDATORY in ALL KOS14 survivors:</strong> AFP every 3-6 months + liver ultrasound every 6 months.
              Hepatoblastoma peak incidence age 0-4. Lifetime annual surveillance after age 5.
              AFP must be interpreted with age-specific paediatric norms (AFP normally very high at birth, falls by age 1-2).
            </Alert>
            <div className="row">
              <div className="col-md-6">
                <table className="table table-sm small">
                  <thead><tr><th>Age</th><th>AFP</th><th>Liver US</th></tr></thead>
                  <tbody>
                    <tr><td>0-5 years</td><td className="fw-bold" style={{ color: ACCENT4 }}>Every 3 months</td><td>Every 6 months</td></tr>
                    <tr><td>5-10 years</td><td>Every 6 months</td><td>Every 6-12 months</td></tr>
                    <tr><td>&gt;10 years</td><td>Annual</td><td>Annual</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="col-md-6">
                <div className="small">
                  <div><strong style={{ color: ACCENT4 }}>⚠️ AFP interpretation:</strong></div>
                  <div>Newborn normal AFP: ~100,000 ng/mL</div>
                  <div>Falls to adult levels (&lt;10 ng/mL) by age 12-18 months</div>
                  <div>Concern: failure to decline OR rise from nadir → urgent oncology referral</div>
                  <div className="mt-1"><strong>VPA note:</strong> VPA can cause mild AFP elevation → prefer LEV/LTG to avoid confounding</div>
                </div>
              </div>
            </div>
          </Section>

          <Section title="Diagnostic Algorithm — Coat-Hanger Rib Phenotype" color={ACCENT5}>
            <div className="row">
              <div className="col-md-8">
                <div className="p-2 rounded" style={{ background: ACCENT5 + '12', border: `1px solid ${ACCENT5}` }}>
                  <div className="small fw-bold mb-2" style={{ color: ACCENT5 }}>Step-by-step diagnostic workup</div>
                  <div className="small">
                    <div className="mb-1">1️⃣ <strong>CXR showing coat-hanger / horizontal ribs in neonate</strong> → suspect KOS14</div>
                    <div className="mb-1">2️⃣ <strong>Methylation analysis 14q32.3 (IG-DMR)</strong> — FIRST TEST
                      <ul className="mb-0">
                        <li>100% methylation → KOS14 confirmed; proceed to mechanism identification</li>
                        <li>50% (normal) → KOS14 unlikely; consider other diagnoses</li>
                      </ul>
                    </div>
                    <div className="mb-1">3️⃣ <strong>SNP array (chr14)</strong>
                      <ul className="mb-0">
                        <li>LOH chr14 without CN change → upd(14)pat (40%)</li>
                        <li>CN loss at 14q32.3 → maternal deletion (40%) → parental SNP array to determine inheritance</li>
                        <li>Normal CN → epimutation (10%) or paternal duplication if gain</li>
                      </ul>
                    </div>
                    <div className="mb-1">4️⃣ <strong>Parental studies</strong> — maternal deletion from mother → 50% recurrence risk</div>
                    <div className="mb-1">5️⃣ <strong>Initiate hepatoblastoma surveillance</strong> immediately once KOS14 confirmed</div>
                    <div className="mb-1">6️⃣ <strong>DLK1 serum protein</strong> — elevated (emerging biomarker; not yet routine 2026)</div>
                  </div>
                </div>
              </div>
              <div className="col-md-4">
                <div className="p-2 rounded" style={{ background: ACCENT2 + '12', border: `1px solid ${ACCENT2}` }}>
                  <div className="small fw-bold mb-2" style={{ color: ACCENT2 }}>Key pitfalls</div>
                  <div className="small">
                    <div className="mb-1">❌ SNP array ALONE misses 50% (UPD = no CN change; epimutation = no CN change)</div>
                    <div className="mb-1">❌ Methylation test for 15q11 (PWS) is NORMAL in KOS14</div>
                    <div className="mb-1">❌ BWS workup (11p15.5) is NORMAL in KOS14</div>
                    <div className="mb-1">✓ Coat-hanger ribs on CXR → ORDER 14q32.3 methylation</div>
                  </div>
                </div>
              </div>
            </div>
          </Section>
        </div>
      )}

      {/* ── Tab 3: Definitions ── */}
      {tab === 3 && (
        <div>
          {definitions && Object.entries(definitions).map(([section, entries]) => (
            <Section key={section} title={section.replace(/_/g, ' ')} color={ACCENT5}>
              {Object.entries(entries).map(([key, value]) => (
                <div key={key} className="mb-3">
                  <div className="fw-bold small" style={{ color: ACCENT }}>{key.replace(/_/g, ' ')}</div>
                  <div className="small" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>{value}</div>
                </div>
              ))}
            </Section>
          ))}
        </div>
      )}

      <div className="mt-3 text-muted small">
        Cohort: {overview?.cohort_n} patients · Seed {overview?.seed} · KOS14 dashboard v1 · 2026-08-25 ·
        <Link href="/temple" className="ms-2" style={{ color: ACCENT8 }}>↔ Temple Syndrome (same locus, opposite parent)</Link>
      </div>
    </div>
  );
}
