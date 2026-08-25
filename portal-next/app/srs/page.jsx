'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Growth', 'Treatments & Genetics', 'Definitions'];

// SRS colour scheme — teal/cyan-green (growth restriction; fragile neonates; brain sparing)
const ACCENT  = '#00695c';   // dark teal — growth restriction / SRS identity
const ACCENT2 = '#004d40';   // deeper teal — severe SGA / neonatal fragility
const ACCENT3 = '#1b5e20';   // deep green — KEY positives / GH therapy / cornstarch mandatory
const ACCENT4 = '#b71c1c';   // deep red — DANGER / fasting CI / KD CI
const ACCENT5 = '#0d47a1';   // dark blue — genetics / ICR1 mechanism
const ACCENT6 = '#4a148c';   // purple — imprinting / same-locus-opposite BWS
const ACCENT7 = '#37474f';   // dark slate — epidemiology
const ACCENT8 = '#e65100';   // deep orange — BWS contrast

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

export default function SRSPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srs/overview`).then(r => r.json()),
      fetch(`${API}/api/srs/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srs/definitions`).then(r => r.json()),
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
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>🧬 Silver-Russell Syndrome (SRS)</h4>
        <div className="text-muted small">
          IGF2 / H19 / CDKN1C · 11p15.5 · Genomic Imprinting (Paternal LOF) · OMIM #180860
          <span className="ms-3 badge" style={{ background: ACCENT6 }}>Imprinting Disorder</span>
          <span className="ms-2 badge" style={{ background: ACCENT8 }}>Opposite of BWS</span>
        </div>
        <div className="text-muted small mt-1">
          H19-ICR1 hypomethylation (paternal, ~45%) → Biallelic IGF2 silencing → Profound growth restriction · Prevalence ~1:30,000–1:100,000
        </div>
      </div>

      {/* KPI strip */}
      <div className="row mb-3 g-2">
        <KPI label="Cohort (n)" value={kpi.total_patients} color={ACCENT} />
        <KPI label="ICR1 Hypo (~45%)" value={`${kpi.icr1_hypo_pct}%`} color={ACCENT5} />
        <KPI label="SGA (universal)" value={`${kpi.sga_universal_pct}%`} color={ACCENT2} />
        <KPI label="Hemihypotrophy" value={`${kpi.hemihypotrophy_pct}%`} color={ACCENT} />
        <KPI label="Neonatal Hypo" value={`${kpi.neonatal_hypoglycemia_pct}%`} color={ACCENT4} />
        <KPI label="Epilepsy" value={`${kpi.epilepsy_pct}%`} color={ACCENT4} />
        <KPI label="GH Therapy" value={`${kpi.gh_therapy_pct}%`} color={ACCENT3} />
        <KPI label="CPP" value={`${kpi.cpp_pct}%`} color={ACCENT7} />
        <KPI label="Mean Wt SDS" value={kpi.mean_birth_weight_sds} color={ACCENT2} />
        <KPI label="Mean ICR1 Meth" value={`${kpi.mean_icr1_meth}%`} color={ACCENT5} />
        <KPI label="Mean Dx (y)" value={kpi.mean_age_diagnosis_y} color={ACCENT7} />
        <KPI label="UPD7mat (~10%)" value={`${kpi.upd7mat_pct}%`} color={ACCENT6} />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          {/* Key alerts */}
          <Section title="⚠️ Critical Clinical Alerts" color={ACCENT4}>
            {(overview?.key_alerts || []).map((a, i) => (
              <Alert key={i} color={a.level === 'DANGER' ? ACCENT4 : a.level === 'WARN' ? '#f57f17' : ACCENT3}>
                <strong>{a.level}:</strong> {a.msg}
              </Alert>
            ))}
          </Section>

          {/* Mechanism */}
          <Section title="🔬 Imprinting Mechanism — Why Paternal LOF Causes SRS" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT5 + '22' }}>
                  <tr><th>Component</th><th>Normal State</th><th>SRS (Paternal LOF)</th><th>Result</th></tr>
                </thead>
                <tbody>
                  <tr><td><strong>ICR1 (Paternal)</strong></td><td>METHYLATED → blocks CTCF</td><td style={{ color: ACCENT4 }}>UNMETHYLATED → CTCF binds</td><td style={{ color: ACCENT4 }}>IGF2 enhancers blocked on paternal allele</td></tr>
                  <tr><td><strong>ICR1 (Maternal)</strong></td><td>UNMETHYLATED → CTCF binds → IGF2 silenced</td><td>UNMETHYLATED (unchanged)</td><td>IGF2 silenced on maternal allele (unchanged)</td></tr>
                  <tr><td><strong>IGF2</strong></td><td>Expressed from PATERNAL only</td><td style={{ color: ACCENT4 }}>SILENCED BIALLELICALLY</td><td style={{ color: ACCENT4, fontWeight: 'bold' }}>ZERO IGF2 → profound growth restriction</td></tr>
                  <tr><td><strong>H19</strong></td><td>Expressed MATERNALLY only</td><td style={{ color: ACCENT4 }}>BIALLELICALLY expressed</td><td>Excess H19 lncRNA → additional growth suppression</td></tr>
                  <tr><td><strong>CDKN1C</strong></td><td>Expressed MATERNALLY (growth brake)</td><td>INTACT (ICR2 unaffected)</td><td style={{ color: ACCENT3 }}>Normal in ICR1-hypomethylation type</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* NH Criteria */}
          <Section title="📋 Netchine-Harbison Clinical Criteria (Diagnosis ≥4/6)" color={ACCENT}>
            <div className="row g-2">
              {(overview?.nh_criteria || []).map((c, i) => (
                <div key={i} className="col-md-6">
                  <div className="card p-2 shadow-sm h-100" style={{ borderLeft: `3px solid ${ACCENT}` }}>
                    <div className="fw-bold small">#{i + 1} {c.criterion}</div>
                    <div className="text-muted small">Prevalence: {c.prevalence}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="alert mt-2 small" style={{ background: ACCENT + '15', borderLeft: `4px solid ${ACCENT}` }}>
              <strong>Threshold:</strong> ≥4/6 criteria = investigate. ≥4/6 + negative molecular = <em>clinical SRS diagnosis</em> (no molecular diagnosis required for treatment).
            </div>
          </Section>

          {/* Diagnostic pathway */}
          <Section title="🔍 Diagnostic Pathway" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT5 + '22' }}>
                  <tr><th>Step</th><th>Test</th><th>Yield</th><th>Detects</th></tr>
                </thead>
                <tbody>
                  {(overview?.diagnostic_pathway || []).map(s => (
                    <tr key={s.step}>
                      <td><strong>{s.step}</strong></td>
                      <td>{s.test}</td>
                      <td>{s.yield || s.threshold || '—'}</td>
                      <td>{s.detects}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Mechanism breakdown */}
          <Section title="🧬 Genetic Mechanisms — 40-Patient Cohort (seed-295)" color={ACCENT6}>
            <div className="row g-2">
              {(breakdown?.mechanism_details || []).map(m => (
                <div key={m.id} className="col-md-6 col-lg-4">
                  <div className="card p-2 shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT6}` }}>
                    <div className="fw-bold small">{m.label}</div>
                    <div className="text-muted small">n={m.n} · {m.pct}%</div>
                    <div className="small mt-1">ICR1 meth ~{m.methylation_pct}% | Asymmetry {Math.round(m.asymmetry_rate * 100)}% | Hypo {Math.round(m.hypo_rate * 100)}%</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Same-locus pair */}
          <Section title="⚖️ Same-Locus Opposite-Phenotype: SRS vs BWS (11p15.5)" color={ACCENT8}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT8 + '22' }}>
                  <tr><th>Feature</th><th style={{ color: ACCENT }}>SRS (Paternal LOF)</th><th style={{ color: ACCENT8 }}>BWS (Maternal LOF)</th></tr>
                </thead>
                <tbody>
                  {breakdown?.compared_with_bws && Object.entries(breakdown.compared_with_bws).map(([k, v]) => (
                    k !== 'principle' && (
                      <tr key={k}>
                        <td className="fw-bold">{k.replace(/_/g, ' ')}</td>
                        <td>{typeof v === 'string' ? v.split(' | ')[0] : ''}</td>
                        <td>{typeof v === 'string' ? v.split(' | ')[1] : ''}</td>
                      </tr>
                    )
                  ))}
                </tbody>
              </table>
            </div>
            <Alert color={ACCENT6}>
              <strong>Imprinting Proof:</strong> {breakdown?.compared_with_bws?.principle}
            </Alert>
          </Section>
        </div>
      )}

      {/* ── TAB 1: Patients & Growth ── */}
      {tab === 1 && (
        <div>
          <Section title="👥 40-Patient Cohort (seed-295) — Individual Records" color={ACCENT}>
            <div className="table-responsive" style={{ maxHeight: 450, overflowY: 'auto' }}>
              <table className="table table-sm table-striped table-bordered small mb-0">
                <thead className="sticky-top" style={{ background: ACCENT + 'dd', color: '#fff' }}>
                  <tr>
                    <th>ID</th><th>Sex</th><th>Mechanism</th><th>Phenotype</th>
                    <th>NH</th><th>BW SDS</th><th>HT SDS</th>
                    <th>ICR1%</th><th>Asymm</th><th>Hypo</th><th>CPP</th><th>GH</th><th>Epilepsy</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.patients || []).map(p => (
                    <tr key={p.id}>
                      <td><strong>{p.id}</strong></td>
                      <td>{p.sex}</td>
                      <td style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.mechanism}</td>
                      <td>{p.phenotype_group}</td>
                      <td><span className="badge" style={{ background: p.nh_score >= 5 ? ACCENT4 : p.nh_score >= 4 ? '#f57f17' : ACCENT3 }}>{p.nh_score}/6</span></td>
                      <td style={{ color: ACCENT4 }}>{p.birth_weight_sds}</td>
                      <td style={{ color: ACCENT2 }}>{p.current_height_sds}</td>
                      <td style={{ color: p.icr1_methylation_pct < 30 ? ACCENT4 : p.icr1_methylation_pct < 45 ? '#f57f17' : ACCENT3 }}>{p.icr1_methylation_pct}%</td>
                      <td>{p.hemihypotrophy ? '✅' : '—'}</td>
                      <td>{p.neonatal_hypoglycemia ? <span style={{ color: ACCENT4 }}>⚠️</span> : '—'}</td>
                      <td>{p.cpp ? '✅' : '—'}</td>
                      <td>{p.gh_therapy ? <span style={{ color: ACCENT3 }}>GH</span> : '—'}</td>
                      <td>{p.epilepsy ? <span style={{ color: ACCENT4 }}>⚡</span> : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="📊 Biomarker Thresholds" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT5 + '22' }}>
                  <tr><th>Biomarker</th><th>Value / Threshold</th></tr>
                </thead>
                <tbody>
                  {breakdown?.biomarker_thresholds && Object.entries(breakdown.biomarker_thresholds).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold">{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="⚖️ SRS vs Temple Syndrome Comparison" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT7 + '22' }}>
                  <tr><th>Feature</th><th style={{ color: ACCENT }}>SRS (11p15.5)</th><th style={{ color: '#880e4f' }}>Temple (14q32.3)</th></tr>
                </thead>
                <tbody>
                  {breakdown?.compared_with_temple && Object.entries(breakdown.compared_with_temple).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold">{k.replace(/_/g, ' ')}</td>
                      <td>{typeof v === 'string' ? v.split(' | ')[0] : v}</td>
                      <td>{typeof v === 'string' ? v.split(' | ')[1] : ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 2: Treatments & Genetics ── */}
      {tab === 2 && (
        <div>
          <Section title="💊 AED / Drug Guide" color={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT4 + '22' }}>
                  <tr><th>Drug</th><th>Evidence</th><th>Role</th><th>Rationale</th></tr>
                </thead>
                <tbody>
                  {(breakdown?.aed_guide || []).map((a, i) => (
                    <tr key={i} style={{ background: a.level === 'CONTRAINDICATED' ? ACCENT4 + '18' : a.level === 'AVOID' ? '#f57f1722' : a.level === 'A' ? ACCENT3 + '18' : '' }}>
                      <td className="fw-bold">{a.name}</td>
                      <td><span className="badge" style={{ background: a.level === 'CONTRAINDICATED' ? ACCENT4 : a.level === 'AVOID' ? '#e65100' : a.level === 'A' ? ACCENT3 : '#0d47a1' }}>{a.level}</span></td>
                      <td>{a.role}</td>
                      <td>{a.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="🏥 Management Protocols" color={ACCENT3}>
            <div className="row g-2">
              {(breakdown?.management_protocols || []).map((m, i) => (
                <div key={i} className="col-md-6">
                  <div className="card p-2 shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT3}` }}>
                    <div className="fw-bold small">{m.category}</div>
                    <div className="text-muted small">{m.intervention} — {m.dose}</div>
                    <div className="small"><span className="badge me-1" style={{ background: ACCENT5 }}>Level {m.evidence.split(' ')[1]?.replace('(','')?.replace(')','') || m.evidence.substring(6,13)}</span>{m.outcome}</div>
                    <div className="text-muted small mt-1">Monitor: {m.monitoring}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="🔬 Genetic Variants" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT6 + '22' }}>
                  <tr><th>Variant / Mechanism</th><th>Frequency</th><th>ICR1 Methylation</th><th>Severity</th><th>Recurrence</th><th>Description</th></tr>
                </thead>
                <tbody>
                  {(breakdown?.variants || []).map((v, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{v.variant}</td>
                      <td>{v.freq_pct}%</td>
                      <td>{v.methylation}</td>
                      <td>{v.severity}</td>
                      <td>{v.recurrence}</td>
                      <td style={{ maxWidth: 200 }}>{v.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="⚖️ Differential Diagnoses" color={ACCENT7}>
            {(breakdown?.differentials || []).map((d, i) => (
              <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT7}` }}>
                <div className="card-body py-2 px-3 small">
                  <div className="fw-bold">{d.condition}</div>
                  <div><strong>Shared:</strong> {d.shared}</div>
                  <div><strong>SRS-specific:</strong> {d.srs_unique}</div>
                  <div><strong>Other-specific:</strong> {d[Object.keys(d).find(k => k.includes('unique') && k !== 'srs_unique')] || d.other_unique}</div>
                  <div style={{ color: ACCENT4 }}><strong>Key discriminator:</strong> {d.key_discriminator}</div>
                  <div className="text-muted">{d.verdict}</div>
                </div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && (
        <div>
          <Section title="🧬 Gene Definitions" color={ACCENT5}>
            {(definitions?.gene_definitions || []).map((g, i) => (
              <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT5}` }}>
                <div className="card-body py-2 px-3 small">
                  <div className="fw-bold">{g.gene} <span className="text-muted">(OMIM {g.omim})</span></div>
                  <div><strong>Protein:</strong> {g.protein}</div>
                  <div><strong>Location:</strong> {g.location}</div>
                  <div><strong>Expression:</strong> {g.expression}</div>
                  <div><strong>Function:</strong> {g.function}</div>
                  <div style={{ color: ACCENT4 }}><strong>In SRS:</strong> {g.in_srs}</div>
                  <div style={{ color: ACCENT8 }}><strong>Contrast BWS:</strong> {g.contrast_bws}</div>
                </div>
              </div>
            ))}
          </Section>

          <Section title="📚 Imprinting Concepts" color={ACCENT6}>
            {(definitions?.imprinting_concepts || []).map((c, i) => (
              <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT6}` }}>
                <div className="card-body py-2 px-3 small">
                  <div className="fw-bold">{c.term}</div>
                  <div>{c.definition}</div>
                </div>
              </div>
            ))}
          </Section>

          <Section title="💊 Drug Classes" color={ACCENT3}>
            {(definitions?.drug_classes || []).map((d, i) => (
              <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT3}` }}>
                <div className="card-body py-2 px-3 small">
                  <div className="fw-bold">{d.drug} <span className="text-muted">({d.examples})</span></div>
                  <div><strong>Mechanism:</strong> {d.mechanism}</div>
                  <div style={{ color: ACCENT3 }}><strong>Evidence in SRS:</strong> {d.evidence_srs}</div>
                  <div><strong>Monitoring:</strong> {d.monitoring}</div>
                </div>
              </div>
            ))}
          </Section>

          <Section title="📝 Key Facts for Exam / Board Review" color={ACCENT4}>
            <ul className="small mb-0">
              {(definitions?.key_facts_exam || []).map((f, i) => (
                <li key={i} className="mb-1">{f}</li>
              ))}
            </ul>
          </Section>

          <div className="text-muted small mt-3">
            Disease: {definitions?.disease} · OMIM {definitions?.omim} · Updated: {definitions?.updated_at}
          </div>
        </div>
      )}
    </div>
  );
}
