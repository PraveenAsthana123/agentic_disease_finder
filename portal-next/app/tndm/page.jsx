'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Cohort & Mechanisms', 'Treatment & Diagnostics', 'Definitions'];

// TNDM colour scheme — amber/gold (neonatal diabetes; transient; insulin/glucose)
const ACCENT  = '#e65100';   // deep orange — neonatal diabetes; glucose emergency
const ACCENT2 = '#f57f17';   // amber — transient course; remission
const ACCENT3 = '#b71c1c';   // deep red — DANGER: ABCC8/KCNJ11 → sulfo immediately
const ACCENT4 = '#1a237e';   // deep indigo — genetics / imprinting mechanism
const ACCENT5 = '#1b5e20';   // deep green — sulfonylurea response / remission
const ACCENT6 = '#4a148c';   // purple — imprinting / PLAGL1
const ACCENT7 = '#37474f';   // dark slate — epidemiology
const ACCENT8 = '#004d40';   // dark teal — contrasts (BWS = orange; TNDM = teal-green-contrasted)

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

export default function TNDMPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/tndm/overview`).then(r => r.json()),
      fetch(`${API}/api/tndm/breakdown`).then(r => r.json()),
      fetch(`${API}/api/tndm/definitions`).then(r => r.json()),
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
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>🧬 Transient Neonatal Diabetes Mellitus (TNDM1)</h4>
        <div className="text-muted small">
          PLAGL1 / HYMAI · 6q24.2 · Genomic Imprinting (Paternal GOF / Maternal LOF) · OMIM #601410
          <span className="ms-3 badge" style={{ background: ACCENT6 }}>Imprinting Disorder</span>
          <span className="ms-2 badge" style={{ background: ACCENT5 }}>Transient → Remits</span>
          <span className="ms-2 badge" style={{ background: ACCENT3 }}>Sulfo @ Relapse</span>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          {/* Clinical alerts */}
          {(overview?.clinical_alerts || []).map((a, i) => (
            <Alert key={i} color={a.level === 'danger' ? ACCENT3 : a.level === 'warning' ? ACCENT2 : ACCENT5}>
              <strong>{a.level === 'danger' ? '🚨 ' : a.level === 'warning' ? '⚠️ ' : 'ℹ️ '}</strong>{a.msg}
            </Alert>
          ))}

          {/* KPI row */}
          <div className="row mt-3">
            <KPI label="Patients (cohort)" value={kpi.total_patients}       color={ACCENT7} />
            <KPI label="Pat dup 6q24 (~40%)" value={kpi.mechanism_patdup}   color={ACCENT4} />
            <KPI label="UPD6pat (~40%)"    value={kpi.mechanism_upd6pat}    color={ACCENT4} />
            <KPI label="Mat DMR hypo (~20%)" value={kpi.mechanism_mat_hypometh} color={ACCENT6} />
            <KPI label="Remission/Relapse" value={kpi.remission_or_relapse} color={ACCENT5} />
            <KPI label="Relapse on Sulfo"  value={kpi.relapse_on_sulfo}     color={ACCENT5} />
            <KPI label="Neonatal active"   value={kpi.neonatal_active}      color={ACCENT} />
            <KPI label="Macroglossia %"    value={`${kpi.macroglossia_pct}%`} color={ACCENT2} />
            <KPI label="IUGR/SGA %"        value={`${kpi.iugr_sga_pct}%`}  color={ACCENT2} />
            <KPI label="Umbilical hernia%" value={`${kpi.umbilical_pct}%`} color={ACCENT2} />
            <KPI label="Ab-negative (not T1D)" value={`${kpi.antibody_negative_pct}%`} color={ACCENT5} />
            <KPI label="Sulfo response"    value={kpi.sulfonylurea_response} color={ACCENT5} />
          </div>

          {/* Mechanism note */}
          <Section title="Imprinting Mechanism — Why PLAGL1 Excess Causes TNDM" color={ACCENT6}>
            <p className="small mb-2">{overview?.mechanism_note}</p>
            <div className="row g-2">
              {[
                { mech: "Paternal dup 6q24 (~40%)", detail: "Extra paternal 6q24 → 2 active PLAGL1 copies + 1 silent maternal. Familial if father carries.", color: ACCENT4 },
                { mech: "Paternal UPD6 (~40%)", detail: "Two paternal chr6 → 2 active PLAGL1, 0 maternal. Sporadic. SNP array shows LOH on chr6.", color: ACCENT4 },
                { mech: "Maternal DMR hypometh (~20%)", detail: "Maternal 6q24 DMR loses methylation → normally silent maternal PLAGL1/HYMAI expressed. Check for MLID.", color: ACCENT6 },
              ].map((m, i) => (
                <div key={i} className="col-md-4">
                  <div className="p-2 rounded" style={{ background: m.color + '12', border: `1px solid ${m.color}40` }}>
                    <div className="fw-bold small" style={{ color: m.color }}>{m.mech}</div>
                    <div className="text-muted" style={{ fontSize: '0.78rem' }}>{m.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Natural history */}
          <Section title="Natural History — 4 Clinical Phases" color={ACCENT}>
            <div className="row g-2">
              {[
                { phase: "1. Neonatal (0-3 mo)", note: "Severe hyperglycemia, IUGR, polyuria, dehydration. Insulin-requiring. Macroglossia 30%.", color: ACCENT3 },
                { phase: "2. Remission (3-18 mo)", note: "Insulin requirements fall to zero. PLAGL1 expression drops developmentally. 100% remit.", color: ACCENT5 },
                { phase: "3. Latency (childhood)", note: "Glucose normal. BUT first-phase insulin secretion is impaired lifelong. Annual HbA1c from age 5.", color: ACCENT7 },
                { phase: "4. Relapse (~50%, adol/adult)", note: "T2D-like diabetes. Sulfonylurea HIGHLY EFFECTIVE (85-95%). NOT T1D.", color: ACCENT2 },
              ].map((p, i) => (
                <div key={i} className="col-md-3">
                  <div className="p-2 rounded h-100" style={{ background: p.color + '12', border: `1px solid ${p.color}40` }}>
                    <div className="fw-bold small" style={{ color: p.color }}>{p.phase}</div>
                    <div className="text-muted" style={{ fontSize: '0.78rem' }}>{p.note}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Treatment ladder */}
          <Section title="Treatment Ladder" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead><tr style={{ background: ACCENT5 + '22' }}>
                  <th>Step</th><th>Treatment</th><th>Phase</th><th>Evidence</th>
                </tr></thead>
                <tbody>
                  {(overview?.treatment_ladder || []).map((r, i) => (
                    <tr key={i}>
                      <td>{r.step}</td>
                      <td className="fw-bold">{r.tx}</td>
                      <td>{r.phase}</td>
                      <td style={{ color: ACCENT5 }}>{r.evidence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Key contrasts */}
          <Section title="Key Contrasts — TNDM1 vs TNDM2/3 vs T1D vs BWS" color={ACCENT8}>
            <div className="row g-2">
              {[
                { label: "TNDM1 (6q24, PLAGL1)", pts: ["Imprinting disorder", "Sulfo works at RELAPSE (not neonatal)", "100% remit", "Ab-negative"], color: ACCENT6 },
                { label: "TNDM2/3 (ABCC8/KCNJ11)", pts: ["K-ATP channel GOF", "SULFO FIRST LINE — even in NEONATE", "~50% remit", "Ab-negative"], color: ACCENT3 },
                { label: "T1D (autoimmune)", pts: ["HLA-mediated beta-cell destruction", "Sulfo DOES NOT WORK", "Does not remit", "Ab-POSITIVE (GADA, ZnT8)"], color: ACCENT3 },
                { label: "BWS (11p15.5, IGF2)", pts: ["Overgrowth, macrosomia", "Hyperinsulinism (NOT diabetes)", "Wilms/hepatoblastoma risk", "Macroglossia shared — different mechanism"], color: ACCENT8 },
              ].map((c, i) => (
                <div key={i} className="col-md-3">
                  <div className="p-2 rounded h-100" style={{ background: c.color + '12', border: `1px solid ${c.color}40` }}>
                    <div className="fw-bold small mb-1" style={{ color: c.color }}>{c.label}</div>
                    <ul className="mb-0 ps-3" style={{ fontSize: '0.76rem', color: '#444' }}>
                      {c.pts.map((pt, j) => <li key={j}>{pt}</li>)}
                    </ul>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 1: Cohort & Mechanisms ── */}
      {tab === 1 && (
        <div>
          <Section title="Mechanism Distribution (40-patient cohort, seed 299)" color={ACCENT4}>
            <div className="row g-3">
              <div className="col-md-4">
                <div className="card shadow-sm">
                  <div className="card-header fw-bold small" style={{ background: ACCENT4 + '18', color: ACCENT4 }}>Mechanism</div>
                  <div className="card-body p-2">
                    {Object.entries(breakdown?.mechanism_distribution || {}).map(([k, v]) => (
                      <div key={k} className="d-flex justify-content-between small mb-1">
                        <span>{k}</span><span className="fw-bold" style={{ color: ACCENT4 }}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="col-md-4">
                <div className="card shadow-sm">
                  <div className="card-header fw-bold small" style={{ background: ACCENT5 + '18', color: ACCENT5 }}>Clinical Outcome</div>
                  <div className="card-body p-2">
                    {Object.entries(breakdown?.outcome_distribution || {}).map(([k, v]) => (
                      <div key={k} className="d-flex justify-content-between small mb-1">
                        <span>{k}</span><span className="fw-bold" style={{ color: ACCENT5 }}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="col-md-4">
                <div className="card shadow-sm">
                  <div className="card-header fw-bold small" style={{ background: ACCENT + '18', color: ACCENT }}>Clinical Features</div>
                  <div className="card-body p-2">
                    {Object.entries(breakdown?.feature_rates_pct || {}).map(([k, v]) => (
                      <div key={k} className="d-flex justify-content-between small mb-1">
                        <span>{k}</span><span className="fw-bold" style={{ color: ACCENT }}>{v}%</span>
                      </div>
                    ))}
                    <div className="d-flex justify-content-between small mb-1">
                      <span>Median remission</span>
                      <span className="fw-bold" style={{ color: ACCENT5 }}>{breakdown?.median_remission_months} mo</span>
                    </div>
                    <div className="d-flex justify-content-between small mb-1">
                      <span>Avg onset</span>
                      <span className="fw-bold" style={{ color: ACCENT }}>{breakdown?.avg_onset_weeks} wk</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Section>

          <Section title="Neonatal Diabetes Type Comparison" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead><tr style={{ background: ACCENT3 + '22' }}>
                  <th>Entity</th><th>Gene</th><th>Mechanism</th><th>Sulfo Response</th><th>Remits?</th><th>Antibodies</th>
                </tr></thead>
                <tbody>
                  {(breakdown?.neonatal_diabetes_comparison || []).map((r, i) => (
                    <tr key={i} style={{ background: r.entity.includes('TNDM1') ? ACCENT6 + '10' : '' }}>
                      <td className="fw-bold">{r.entity}</td>
                      <td>{r.gene}</td>
                      <td>{r.mechanism}</td>
                      <td style={{ color: r.sulfo_response.includes('Yes') ? ACCENT5 : ACCENT3 }}>{r.sulfo_response}</td>
                      <td style={{ color: r.remits.includes('Yes') || r.remits.includes('100') ? ACCENT5 : ACCENT3 }}>{r.remits}</td>
                      <td style={{ color: r.antibodies === 'Negative' ? ACCENT5 : ACCENT3 }}>{r.antibodies}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Patient Cohort Table (n=40)" color={ACCENT7}>
            <div className="table-responsive" style={{ maxHeight: 380, overflowY: 'auto' }}>
              <table className="table table-sm table-striped small">
                <thead className="sticky-top" style={{ background: '#fff' }}><tr>
                  <th>ID</th><th>Sex</th><th>Mechanism</th><th>Onset (wk)</th>
                  <th>BW SDS</th><th>Macroglossia</th><th>Umbilical</th><th>IUGR</th>
                  <th>Outcome</th><th>Remission (mo)</th><th>Relapse (yr)</th>
                </tr></thead>
                <tbody>
                  {(breakdown?.cohort_table || []).map((p, i) => (
                    <tr key={i}>
                      <td>{p.id}</td><td>{p.sex}</td>
                      <td style={{ fontSize: '0.72rem' }}>{p.mechanism.replace('Paternal duplication 6q24','Pat dup').replace('Paternal UPD6 (upd(6)pat)','UPD6pat').replace('Maternal DMR hypomethylation','Mat DMR hypo')}</td>
                      <td>{p.onset_wk}</td>
                      <td style={{ color: p.birth_wt_sds < -2 ? ACCENT3 : ACCENT7 }}>{p.birth_wt_sds}</td>
                      <td>{p.macroglossia ? '✓' : '—'}</td>
                      <td>{p.umbilical_hernia ? '✓' : '—'}</td>
                      <td>{p.iugr ? '✓' : '—'}</td>
                      <td style={{ fontSize: '0.72rem', color: p.outcome.includes('sulfo') ? ACCENT5 : p.outcome.includes('Neonatal') ? ACCENT : ACCENT7 }}>{p.outcome}</td>
                      <td>{p.remission_mo ?? '—'}</td>
                      <td>{p.relapse_age_yr ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 2: Treatment & Diagnostics ── */}
      {tab === 2 && (
        <div>
          <Alert color={ACCENT3}>
            <strong>🚨 CRITICAL:</strong> Test ALL neonatal diabetes for ABCC8 + KCNJ11 mutations.
            K-ATP gain-of-function = sulfonylurea FIRST LINE immediately, even in neonate —
            do NOT wait for genetic confirmation if clinical suspicion is high.
          </Alert>

          <Section title="Diagnostic Sequence" color={ACCENT4}>
            <div className="row g-2">
              {[
                { step: "1st: MS-MLPA (6q24)", detail: "Methylation-sensitive MLPA at PLAGL1/HYMAI DMR — detects paternal duplication + maternal hypomethylation. First-line for TNDM1.", color: ACCENT4 },
                { step: "2nd: SNP Array", detail: "Genome-wide SNP array — identifies UPD6pat (chr6 LOH). Normal array does NOT rule out DMR epimutation.", color: ACCENT4 },
                { step: "3rd: ABCC8 + KCNJ11 seq", detail: "Mandatory in all neonatal DM — K-ATP mutation = different disease, different treatment (sulfo 1st line immediately).", color: ACCENT3 },
                { step: "4th: Antibody panel", detail: "GADA, IA-2, ZnT8 — NEGATIVE in TNDM. Positive = T1D → insulin lifelong, sulfo won't work.", color: ACCENT5 },
                { step: "5th: C-peptide", detail: "Low/undetectable neonatal phase; re-check in remission and annually for relapse. Rise = relapse starting.", color: ACCENT2 },
              ].map((d, i) => (
                <div key={i} className="col-md-4 mb-2">
                  <div className="p-2 rounded h-100" style={{ background: d.color + '12', border: `1px solid ${d.color}40` }}>
                    <div className="fw-bold small" style={{ color: d.color }}>{d.step}</div>
                    <div className="text-muted" style={{ fontSize: '0.78rem' }}>{d.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Treatment — Neonatal Phase" color={ACCENT}>
            <div className="row g-2">
              {[
                { tx: "Insulin (IV continuous)", note: "0.01-0.1 U/kg/hr; target glucose 70-180 mg/dL; adjust for feeds", ok: true },
                { tx: "SC insulin (alternative)", note: "Rapid-acting SC; multiple daily injections once oral feeds established", ok: true },
                { tx: "Wean insulin (remission)", note: "When daily requirements drop to <0.1 U/kg/day over weeks", ok: true },
                { tx: "Immunosuppression", note: "CONTRAINDICATED — TNDM is NOT autoimmune. No role for steroids, mycophenolate, or anti-CD20.", ok: false },
              ].map((r, i) => (
                <div key={i} className="col-md-3">
                  <div className="p-2 rounded" style={{ background: (r.ok ? ACCENT5 : ACCENT3) + '12', border: `1px solid ${(r.ok ? ACCENT5 : ACCENT3)}40` }}>
                    <div className="fw-bold small" style={{ color: r.ok ? ACCENT5 : ACCENT3 }}>{r.ok ? '✓' : '✗'} {r.tx}</div>
                    <div className="text-muted" style={{ fontSize: '0.76rem' }}>{r.note}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Treatment — Relapse Phase (Adolescent/Adult)" color={ACCENT5}>
            <div className="row g-2">
              {[
                { tx: "Sulfonylurea (1st line)", note: "Glyburide or glipizide. ~85-95% response rate. Stimulates K-ATP-independent insulin secretion. Level A.", ok: true },
                { tx: "Metformin (adjunct)", note: "Insulin sensitiser adjunct if sulfonylurea insufficient. Level C. May reduce relapse progression.", ok: true },
                { tx: "Insulin (2nd line)", note: "Reserve for sulfonylurea failure, pregnancy, complications. Works but sulfo preferred.", ok: true },
                { tx: "Stop sulfo without monitoring", note: "Lifelong glucose surveillance required — ~50% relapse in adult life. Never discharge from diabetes follow-up.", ok: false },
              ].map((r, i) => (
                <div key={i} className="col-md-3">
                  <div className="p-2 rounded" style={{ background: (r.ok ? ACCENT5 : ACCENT3) + '12', border: `1px solid ${(r.ok ? ACCENT5 : ACCENT3)}40` }}>
                    <div className="fw-bold small" style={{ color: r.ok ? ACCENT5 : ACCENT3 }}>{r.ok ? '✓' : '✗'} {r.tx}</div>
                    <div className="text-muted" style={{ fontSize: '0.76rem' }}>{r.note}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Surveillance Schedule" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead><tr style={{ background: ACCENT7 + '22' }}><th>Test</th><th>Frequency</th><th>Rationale</th></tr></thead>
                <tbody>
                  {[
                    ["HbA1c + fasting glucose", "Annually from age 5 (remission phase)", "Early relapse detection; first-phase impaired lifelong"],
                    ["OGTT (75 g)", "Every 2-3 yr from adolescence", "Detect relapse as glucose tolerance deteriorates before overt DM"],
                    ["C-peptide (fasting)", "At remission; annually thereafter", "Confirms remission; rising = relapse flag"],
                    ["Genetic counselling", "At diagnosis; before each pregnancy", "Pat dup → 50% if father carries; UPD/DMR → <5%; ART counselling for MLID risk"],
                    ["MLID panel (if mat hypometh)", "Once at diagnosis", "Maternal DMR hypometh ~20% cases: check KCNQ1OT1, H19, SNRPN DMRs for multi-locus defect"],
                  ].map(([test, freq, reason], i) => (
                    <tr key={i}>
                      <td className="fw-bold">{test}</td><td>{freq}</td>
                      <td className="text-muted" style={{ fontSize: '0.78rem' }}>{reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && (
        <div>
          <Section title="Disease Overview" color={ACCENT}>
            <dl className="row small">
              <dt className="col-sm-3">Full name</dt>
              <dd className="col-sm-9">{definitions?.disease_overview?.full_name}</dd>
              <dt className="col-sm-3">OMIM</dt>
              <dd className="col-sm-9">{definitions?.disease_overview?.omim}</dd>
              <dt className="col-sm-3">Locus</dt>
              <dd className="col-sm-9">{definitions?.disease_overview?.locus}</dd>
              <dt className="col-sm-3">Genes</dt>
              <dd className="col-sm-9">{Object.entries(definitions?.disease_overview?.mim_genes || {}).map(([g, o]) => `${g} (${o})`).join(', ')}</dd>
              <dt className="col-sm-3">Inheritance</dt>
              <dd className="col-sm-9">{definitions?.disease_overview?.inheritance}</dd>
              <dt className="col-sm-3">Prevalence</dt>
              <dd className="col-sm-9">{definitions?.disease_overview?.prevalence}</dd>
            </dl>
          </Section>

          <Section title="Gene Definitions" color={ACCENT4}>
            {Object.entries(definitions?.genes || {}).map(([gene, info]) => (
              <div key={gene} className="mb-3 p-3 rounded" style={{ background: ACCENT4 + '08', border: `1px solid ${ACCENT4}30` }}>
                <div className="fw-bold" style={{ color: ACCENT4 }}>{gene} — {info.full_name}</div>
                <div className="small text-muted mt-1">{info.function}</div>
                <div className="small mt-1"><strong>Key:</strong> {info.key_fact}</div>
              </div>
            ))}
          </Section>

          <Section title="Mechanism Details" color={ACCENT6}>
            {Object.entries(definitions?.mechanisms || {}).map(([key, m]) => (
              <div key={key} className="mb-3 p-3 rounded" style={{ background: ACCENT6 + '08', border: `1px solid ${ACCENT6}30` }}>
                <div className="fw-bold small" style={{ color: ACCENT6 }}>{key.replace(/_/g, ' ').toUpperCase()} — {m.frequency}</div>
                <div className="small text-muted">{m.description}</div>
                {m.recurrence && <div className="small"><strong>Recurrence:</strong> {m.recurrence}</div>}
                {m.note && <div className="small text-warning">{m.note}</div>}
                <div className="small"><strong>Testing:</strong> {m.testing}</div>
              </div>
            ))}
          </Section>

          <Section title="MLID & ART Risk Notes" color={ACCENT2}>
            <Alert color={ACCENT2}>
              <strong>MLID (Multi-Locus Imprinting Disturbance):</strong> {definitions?.mlid_note}
            </Alert>
            <Alert color={ACCENT2}>
              <strong>ART Risk:</strong> {definitions?.art_risk}
            </Alert>
          </Section>
        </div>
      )}

      <div className="text-muted small mt-4 border-top pt-2">
        TNDM1 — PLAGL1/HYMAI · 6q24.2 · OMIM #601410 · Genomic Imprinting (Paternal GOF / Maternal DMR Hypomethylation) ·
        Seed-{overview?.seed} · {overview?.generated} ·{' '}
        <Link href="/" className="text-decoration-none">← Portal Home</Link>
      </div>
    </div>
  );
}
