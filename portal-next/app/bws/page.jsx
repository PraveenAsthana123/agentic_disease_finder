'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Tumors', 'Treatments & Genetics', 'Definitions'];

// BWS colour scheme — deep orange/red-orange (overgrowth; tumor risk; macroglossia)
const ACCENT  = '#bf360c';   // deep orange-red — overgrowth / BWS identity
const ACCENT2 = '#e64a19';   // orange — macrosomia / macroglossia / organomegaly
const ACCENT3 = '#b71c1c';   // deep red — DANGER / tumor risk / hyperinsulinism
const ACCENT4 = '#1a237e';   // deep indigo — genetics / ICR mechanism
const ACCENT5 = '#1b5e20';   // deep green — GH contraindicated / diazoxide first-line
const ACCENT6 = '#4a148c';   // purple — imprinting / same-locus-opposite SRS
const ACCENT7 = '#37474f';   // dark slate — epidemiology
const ACCENT8 = '#00695c';   // dark teal — SRS contrast (SRS uses teal)

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

export default function BWSPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/bws/overview`).then(r => r.json()),
      fetch(`${API}/api/bws/breakdown`).then(r => r.json()),
      fetch(`${API}/api/bws/definitions`).then(r => r.json()),
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
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>🧬 Beckwith-Wiedemann Syndrome (BWS)</h4>
        <div className="text-muted small">
          IGF2 / H19 / CDKN1C / KCNQ1OT1 · 11p15.5 · Genomic Imprinting (Maternal LOF) · OMIM #130650
          <span className="ms-3 badge" style={{ background: ACCENT6 }}>Imprinting Disorder</span>
          <span className="ms-2 badge" style={{ background: ACCENT8 }}>Opposite of SRS</span>
          <span className="ms-2 badge" style={{ background: ACCENT3 }}>Tumor Surveillance Mandatory</span>
        </div>
        <div className="text-muted small mt-1">
          ICR2 (KCNQ1OT1) hypomethylation (~50%) → Biallelic CDKN1C silencing + excess IGF2 → OVERGROWTH · Wilms 7–29% · Prevalence ~1:10,500–1:13,700
        </div>
      </div>

      {/* KPI strip */}
      <div className="row mb-3 g-2">
        <KPI label="Cohort (n)" value={kpi.total_patients} color={ACCENT} />
        <KPI label="ICR2 Hypo (~50%)" value={`${kpi.icr2_hypo_pct}%`} color={ACCENT4} />
        <KPI label="UPD11p15 (~20%)" value={`${kpi.upd11p15pat_pct}%`} color={ACCENT4} />
        <KPI label="Macroglossia" value={`${kpi.macroglossia_pct}%`} color={ACCENT2} />
        <KPI label="Omphalocele" value={`${kpi.omphalocele_pct}%`} color={ACCENT2} />
        <KPI label="Hemihypertrophy" value={`${kpi.hemihypertrophy_pct}%`} color={ACCENT} />
        <KPI label="Hyperinsulinism" value={`${kpi.hyperinsulinism_pct}%`} color={ACCENT3} />
        <KPI label="Wilms Tumor" value={`${kpi.wilms_pct}%`} color={ACCENT3} />
        <KPI label="Tumor Surveillance" value={`${kpi.tumor_surveillance_pct}%`} color={ACCENT5} />
        <KPI label="Mean Wt SDS" value={`+${kpi.mean_birth_weight_sds}`} color={ACCENT2} />
        <KPI label="Mean Dx (y)" value={kpi.mean_age_diagnosis_y} color={ACCENT7} />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && (
        <div>
          {/* Key Alerts */}
          <Section title="🚨 Critical Clinical Alerts" color={ACCENT3}>
            {(overview?.key_alerts || []).map((a, i) => (
              <Alert key={i} color={a.level === 'DANGER' ? ACCENT3 : a.level === 'WARN' ? '#e65100' : ACCENT4}>
                <span className="badge me-2" style={{ background: a.level === 'DANGER' ? ACCENT3 : a.level === 'WARN' ? '#e65100' : ACCENT4 }}>{a.level}</span>
                {a.msg}
              </Alert>
            ))}
          </Section>

          {/* Cardinal features */}
          <Section title="Cardinal Clinical Features" color={ACCENT2}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT2 + '22' }}>
                  <tr><th>Feature</th><th>Prevalence</th><th>Notes</th></tr>
                </thead>
                <tbody>
                  {(overview?.cardinal_features || []).map((f, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{f.feature}</td>
                      <td><span className="badge" style={{ background: ACCENT }}>{f.prevalence}</span></td>
                      <td className="text-muted">{f.notes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Tumor risks */}
          <Section title="🎯 Tumor Risks (MANDATORY Surveillance)" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT3 + '22' }}>
                  <tr><th>Tumor</th><th>Overall BWS Risk</th><th>Highest Subtype</th><th>Surveillance</th></tr>
                </thead>
                <tbody>
                  {(overview?.tumor_risks || []).map((t, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{t.tumor}</td>
                      <td><span className="badge" style={{ background: ACCENT3 }}>{t.overall_bws_risk}</span></td>
                      <td className="text-muted small">{t.highest_subtype}</td>
                      <td className="text-muted small">{t.surveillance}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Mechanism breakdown */}
          <div className="row">
            <div className="col-md-6">
              <Section title="Mechanism Distribution (n=40)" color={ACCENT4}>
                {Object.entries(overview?.mechanism_breakdown || {}).map(([k, v]) => (
                  <div key={k} className="d-flex justify-content-between mb-1 small">
                    <span>{k}</span>
                    <span className="badge" style={{ background: ACCENT4 }}>{v} pts</span>
                  </div>
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Phenotype Groups" color={ACCENT}>
                {Object.entries(overview?.phenotype_breakdown || {}).map(([k, v]) => (
                  <div key={k} className="d-flex justify-content-between mb-1 small">
                    <span>{k}</span>
                    <span className="badge" style={{ background: ACCENT }}>{v} pts</span>
                  </div>
                ))}
              </Section>
            </div>
          </div>

          {/* Diagnostic pathway */}
          <Section title="Diagnostic Pathway" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm small">
                <thead style={{ background: ACCENT7 + '22' }}>
                  <tr><th>Step</th><th>Test</th><th>Yield</th><th>Detects</th></tr>
                </thead>
                <tbody>
                  {(overview?.diagnostic_pathway || []).map((s, i) => (
                    <tr key={i}>
                      <td className="fw-bold text-center" style={{ color: ACCENT }}>{s.step}</td>
                      <td>{s.test}</td>
                      <td><span className="badge" style={{ background: ACCENT7 }}>{s.yield || s.threshold}</span></td>
                      <td className="text-muted">{s.detects}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* BWS vs SRS Comparison */}
          <Section title="🔬 BWS vs SRS — Same Locus 11p15.5 · Opposite Phenotypes" color={ACCENT6}>
            <div className="alert" style={{ background: ACCENT6 + '12', borderLeft: `4px solid ${ACCENT6}`, borderRadius: 6 }}>
              <strong>Canonical Imprinting Proof:</strong> SAME LOCUS (11p15.5) — OPPOSITE PARENT — OPPOSITE PHENOTYPE
            </div>
            <div className="table-responsive">
              <table className="table table-sm small">
                <thead style={{ background: ACCENT6 + '22' }}>
                  <tr><th>Property</th><th style={{ color: ACCENT }}>BWS (Maternal LOF)</th><th style={{ color: ACCENT8 }}>SRS (Paternal LOF)</th></tr>
                </thead>
                <tbody>
                  {Object.entries(breakdown?.compared_with_srs || {}).filter(([k]) => k !== 'principle').map(([k, v]) => {
                    const parts = v.split('|');
                    return (
                      <tr key={k}>
                        <td className="fw-bold text-muted small">{k.replace(/_/g, ' ')}</td>
                        <td style={{ color: ACCENT }}>{parts[0]?.trim()}</td>
                        <td style={{ color: ACCENT8 }}>{parts[1]?.trim()}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── Tab 1: Patients & Tumors ── */}
      {tab === 1 && (
        <div>
          <Section title="Tumor Surveillance Protocol" color={ACCENT3}>
            {breakdown?.tumor_surveillance_protocol && (
              <div className="row">
                <div className="col-md-6">
                  <div className="card mb-3" style={{ borderLeft: `4px solid ${ACCENT3}` }}>
                    <div className="card-body py-2">
                      <h6 className="fw-bold" style={{ color: ACCENT3 }}>Wilms Tumor (Nephroblastoma)</h6>
                      <div className="small">
                        <div><strong>Imaging:</strong> {breakdown.tumor_surveillance_protocol.wilms.imaging}</div>
                        <div><strong>Age 0–4:</strong> {breakdown.tumor_surveillance_protocol.wilms.frequency_age_0_4}</div>
                        <div><strong>Age 4–8:</strong> {breakdown.tumor_surveillance_protocol.wilms.frequency_age_4_8}</div>
                        <div><strong>Stop:</strong> Age {breakdown.tumor_surveillance_protocol.wilms.stop_age}</div>
                        <div className="text-muted mt-1">{breakdown.tumor_surveillance_protocol.wilms.notes}</div>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="col-md-6">
                  <div className="card mb-3" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
                    <div className="card-body py-2">
                      <h6 className="fw-bold" style={{ color: ACCENT2 }}>Hepatoblastoma</h6>
                      <div className="small">
                        <div><strong>Biomarker:</strong> {breakdown.tumor_surveillance_protocol.hepatoblastoma.biomarker}</div>
                        <div><strong>Age 0–4:</strong> {breakdown.tumor_surveillance_protocol.hepatoblastoma.frequency_age_0_4}</div>
                        <div><strong>Stop:</strong> Age {breakdown.tumor_surveillance_protocol.hepatoblastoma.stop_age}</div>
                        <div className="text-muted mt-1">{breakdown.tumor_surveillance_protocol.hepatoblastoma.notes}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <Alert color={ACCENT3}>
              <strong>ALL BWS patients require tumor surveillance regardless of molecular mechanism</strong> —
              surveillance clinic at specialist centre recommended. Abdominal US + AFP every 3 months from birth.
            </Alert>
          </Section>

          <Section title="Cohort Patient Table" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small">
                <thead style={{ background: ACCENT7 + '22' }}>
                  <tr>
                    <th>ID</th><th>Mechanism</th><th>Phenotype</th><th>Sex</th>
                    <th>Wt SDS</th><th>Macroglossia</th><th>Omphalocele</th>
                    <th>Hemihypertrophy</th><th>Hyperinsulinism</th><th>Wilms</th>
                    <th>Diazoxide</th><th>Dx (y)</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.patients || []).map(p => (
                    <tr key={p.id}>
                      <td className="fw-bold" style={{ color: ACCENT }}>{p.id}</td>
                      <td className="small text-muted" style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.mechanism}</td>
                      <td className="small">{p.phenotype_group}</td>
                      <td>{p.sex}</td>
                      <td style={{ color: ACCENT2 }}>+{p.birth_weight_sds}</td>
                      <td>{p.macroglossia ? <span className="badge bg-warning text-dark">Yes</span> : 'No'}</td>
                      <td>{p.omphalocele ? <span className="badge" style={{ background: ACCENT }}>Yes</span> : 'No'}</td>
                      <td>{p.hemihypertrophy ? <span className="badge" style={{ background: ACCENT2 }}>Yes</span> : 'No'}</td>
                      <td>{p.neonatal_hyperinsulinism ? <span className="badge" style={{ background: ACCENT3 }}>Yes</span> : 'No'}</td>
                      <td>{p.wilms_tumor ? <span className="badge bg-danger">Yes</span> : 'No'}</td>
                      <td>{p.diazoxide_used ? <span className="badge" style={{ background: ACCENT5 }}>Yes</span> : 'No'}</td>
                      <td>{p.age_diagnosis_y}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Biomarker Thresholds" color={ACCENT4}>
            <div className="row">
              {Object.entries(breakdown?.biomarker_thresholds || {}).map(([k, v]) => (
                <div key={k} className="col-md-6 mb-2">
                  <div className="d-flex justify-content-between small p-2 rounded" style={{ background: ACCENT4 + '10' }}>
                    <span className="text-muted">{k.replace(/_/g, ' ')}</span>
                    <span className="fw-bold" style={{ color: ACCENT4 }}>{v}</span>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* ── Tab 2: Treatments & Genetics ── */}
      {tab === 2 && (
        <div>
          <Section title="Management Protocols" color={ACCENT}>
            {(breakdown?.management_protocols || []).map((m, i) => (
              <div key={i} className="card mb-2" style={{ borderLeft: `4px solid ${m.level === 'DANGER' ? ACCENT3 : m.level === 'MANDATORY' ? ACCENT3 : m.level === 'CONTRAINDICATED' ? '#b71c1c' : m.level === 'SURGICAL' ? ACCENT2 : ACCENT7}` }}>
                <div className="card-body py-2">
                  <div className="d-flex justify-content-between align-items-start">
                    <span className="fw-bold small">{m.domain}</span>
                    <span className="badge" style={{ background: m.level === 'DANGER' || m.level === 'MANDATORY' || m.level === 'CONTRAINDICATED' ? ACCENT3 : ACCENT7 }}>{m.level}</span>
                  </div>
                  <div className="small mt-1">{m.protocol}</div>
                  <div className="text-muted small mt-1">📋 {m.monitoring}</div>
                </div>
              </div>
            ))}
          </Section>

          <Section title="AED Guide (Seizures in BWS — mostly hypoglycemic, not primary epilepsy)" color={ACCENT4}>
            <Alert color={ACCENT3}>
              <strong>IMPORTANT:</strong> BWS seizures are usually hypoglycemic (hyperinsulinism). Treat the cause with <strong>diazoxide</strong> first. AEDs treat primary epilepsy, which is rare in BWS.
            </Alert>
            {(breakdown?.aed_guide || []).map((a, i) => (
              <div key={i} className="card mb-2" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
                <div className="card-body py-2">
                  <div className="fw-bold small" style={{ color: ACCENT4 }}>{a.drug} <span className="fw-normal text-muted">({a.class})</span></div>
                  <div className="small mt-1">BWS: {a.bws_evidence}</div>
                  <div className="small text-warning-emphasis mt-1">⚠ {a.key_alert}</div>
                  {a.ci_in_bws && <div className="small text-muted mt-1">CI: {a.ci_in_bws}</div>}
                </div>
              </div>
            ))}
          </Section>

          <Section title="Genetic Mechanisms — Molecular Details" color={ACCENT4}>
            {(breakdown?.mechanism_details || []).map((m, i) => (
              <div key={i} className="card mb-2">
                <div className="card-body py-2">
                  <div className="fw-bold small" style={{ color: ACCENT4 }}>{m.label} <span className="badge ms-2" style={{ background: ACCENT }}>{m.pct}%</span></div>
                  <div className="row mt-1 small">
                    <div className="col-md-4"><span className="text-muted">IGF2: </span>{m.igf2_status}</div>
                    <div className="col-md-4"><span className="text-muted">CDKN1C: </span>{m.cdkn1c_status}</div>
                    <div className="col-md-4"><span className="text-muted">Wilms risk: </span><strong style={{ color: ACCENT3 }}>{m.wilms_risk_pct}%</strong></div>
                  </div>
                  <div className="small text-muted mt-1">{m.notes}</div>
                  <div className="small mt-1">Recurrence: <strong>{m.recurrence_pct}%</strong></div>
                </div>
              </div>
            ))}
          </Section>

          <Section title="Differential Diagnoses" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm small">
                <thead style={{ background: ACCENT7 + '22' }}>
                  <tr><th>Disease</th><th>Locus</th><th>Key Contrast with BWS</th></tr>
                </thead>
                <tbody>
                  {(breakdown?.differentials || []).map((d, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{d.disease}</td>
                      <td><code>{d.locus}</code></td>
                      <td className="text-muted small">{d.key_contrast}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── Tab 3: Definitions ── */}
      {tab === 3 && (
        <div>
          <Section title="Gene Definitions" color={ACCENT4}>
            {(definitions?.gene_definitions || []).map((g, i) => (
              <div key={i} className="card mb-3">
                <div className="card-header py-1" style={{ background: ACCENT4 + '22' }}>
                  <strong style={{ color: ACCENT4 }}>{g.gene}</strong>
                  <span className="ms-2 text-muted small">{g.omim} · {g.location}</span>
                </div>
                <div className="card-body py-2 small">
                  <div><span className="text-muted">Protein: </span>{g.protein}</div>
                  <div><span className="text-muted">Expression: </span>{g.expression}</div>
                  <div><span className="text-muted">Function: </span>{g.function}</div>
                  <div style={{ color: ACCENT }}><span className="text-muted">In BWS: </span>{g.in_bws}</div>
                  <div style={{ color: ACCENT8 }}><span className="text-muted">Contrast SRS: </span>{g.contrast_srs}</div>
                </div>
              </div>
            ))}
          </Section>

          <Section title="Imprinting Concepts" color={ACCENT6}>
            {(definitions?.imprinting_concepts || []).map((c, i) => (
              <div key={i} className="mb-3 p-3 rounded" style={{ background: ACCENT6 + '10', borderLeft: `3px solid ${ACCENT6}` }}>
                <div className="fw-bold small" style={{ color: ACCENT6 }}>{c.term}</div>
                <div className="small mt-1">{c.definition}</div>
              </div>
            ))}
          </Section>

          <Section title="Drug Classes" color={ACCENT5}>
            {(definitions?.drug_classes || []).map((d, i) => (
              <div key={i} className="card mb-2">
                <div className="card-body py-2 small">
                  <div className="fw-bold" style={{ color: ACCENT5 }}>{d.drug} <span className="fw-normal text-muted">— {d.examples}</span></div>
                  <div><span className="text-muted">Mechanism: </span>{d.mechanism}</div>
                  <div style={{ color: ACCENT5 }}><span className="text-muted">BWS evidence: </span>{d.evidence_bws}</div>
                  <div className="text-muted"><span>Monitoring: </span>{d.monitoring}</div>
                </div>
              </div>
            ))}
          </Section>

          <Section title="Key Facts — Exam Summary" color={ACCENT3}>
            <ol className="small">
              {(definitions?.key_facts_exam || []).map((f, i) => (
                <li key={i} className="mb-2">{f}</li>
              ))}
            </ol>
          </Section>
        </div>
      )}

      <div className="text-muted small mt-3">
        BWS Dashboard · OMIM #130650 · 11p15.5 · Genomic Imprinting (Maternal LOF) · Same locus as SRS · Seed {overview?.seed} · n={overview?.cohort_size} synthetic patients · Updated {overview?.updated_at?.slice(0, 10)}
      </div>
    </div>
  );
}
