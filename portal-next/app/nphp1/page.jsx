'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'NPHP1 Molecular Biology & Diagnosis', 'Definitions'];

// NPHP1 colour scheme — juvenile ESRD / 290kb deletion / TZ Y-link / most common NPHP
const ACCENT  = '#0d47a1';   // deep blue — NPHP1 gene / most common / TZ Y-link scaffold
const ACCENT2 = '#880e4f';   // deep pink — Senior-Løken Syndrome 1 (SLS1) / retinal dystrophy
const ACCENT3 = '#1b5e20';   // deep green — MLPA / 290kb deletion / diagnostic first line
const ACCENT4 = '#e65100';   // burnt orange — ESRD / CKD progression / transplant
const ACCENT5 = '#4a148c';   // deep purple — NPHP1-4-8 supercomplex / TZ biology
const ACCENT6 = '#37474f';   // dark slate — molecular architecture / TPR domains
const ACCENT7 = '#f57f17';   // amber — misdiagnosis alerts / DDx traps
const ACCENT8 = '#b71c1c';   // deep red — JBTS4 / Joubert overlap

const SEED = 341;
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
  return (
    <span className="badge me-1 mb-1" style={{ background: color, fontSize: '0.72rem' }}>{text}</span>
  );
}

function BarRow({ label, count, total, color }) {
  const pct = total ? Math.round((count / total) * 100) : 0;
  return (
    <div className="mb-1">
      <div className="d-flex justify-content-between small mb-0">
        <span className="text-truncate" style={{ maxWidth: '70%' }}>{label}</span>
        <span className="fw-bold">{count} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 6 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function NPHP1Dashboard() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp1/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp1/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3">Loading NPHP1 Dashboard…</p></div>;
  if (error)   return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '14', borderLeft: `5px solid ${ACCENT}` }}>
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🧬 Nephronophthisis Type 1 (NPHP1 — Juvenile Nephronophthisis)
        </h4>
        <div className="small text-muted">
          <strong>Gene:</strong> NPHP1 (2q13) · Nephrocystin-1, 736 aa · TZ Y-link scaffold (NPHP1-4-8 supercomplex) &nbsp;|&nbsp;
          <strong>OMIM:</strong> *607100 (gene) · #256100 (NPHP1) · #266900 (SLS1) &nbsp;|&nbsp;
          <strong>Cohort:</strong> {_COHORT_SIZE} patients · seed-{SEED}
        </div>
        <div className="mt-1">
          <Badge text="MOST COMMON NPHP (80-85%)" color={ACCENT} />
          <Badge text="290kb del 2q13 — MLPA FIRST LINE" color={ACCENT3} />
          <Badge text="Juvenile ESRD median 13yr" color={ACCENT4} />
          <Badge text="Senior-Løken SLS1 10-15%" color={ACCENT2} />
          <Badge text="NO situs inversus · NO CHF" color={ACCENT6} />
          <Badge text="Renal Tx CURATIVE" color="#2e7d32" />
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ───────────────────────────────────────────────── */}
      {tab === 0 && overview && (
        <div>
          {/* KPI row */}
          <div className="row g-2 mb-3">
            <KPI label="Cohort" value={overview.n_patients} color={ACCENT} />
            <KPI label="Avg GFR (ml/min)" value={overview.avg_gfr_ml_min} color={ACCENT4} />
            <KPI label="Avg Age Dx (yr)" value={overview.avg_age_renal_dx_yr} color={ACCENT} />
            <KPI label="Avg Hb (g/dL)" value={overview.avg_hb_g_dl} color={ACCENT2} />
            <KPI label="ESRD %" value={`${overview.pct_esrd}%`} color={ACCENT4} />
            <KPI label="290kb del Hom %" value={`${overview.pct_290kb_hom_del}%`} color={ACCENT3} />
            <KPI label="MLPA+ %" value={`${overview.pct_mlpa_positive}%`} color={ACCENT3} />
            <KPI label="SLS1 Retinal %" value={`${overview.pct_sls1_retinal}%`} color={ACCENT2} />
            <KPI label="JBTS4 %" value={`${overview.pct_jbts4}%`} color={ACCENT8} />
            <KPI label="Concentrating Def %" value={`${overview.pct_concentrating_defect}%`} color={ACCENT5} />
            <KPI label="ESRD Median" value="13 yr" color={ACCENT4} />
            <KPI label="ESRD Range" value="4–20 yr" color={ACCENT6} />
          </div>

          {/* Critical alerts */}
          <Alert color={ACCENT3}>
            <strong>🔬 MLPA P369 IS FIRST LINE:</strong> Send MLPA (MRC-Holland P369) for ALL juvenile
            CKD + small echogenic kidneys — detects the 290 kb homozygous deletion (66-80% of NPHP1).
            NEVER stop at MLPA-negative: 20-34% have compound het SNVs requiring WES + CNV analysis.
          </Alert>
          <Alert color={ACCENT2}>
            <strong>👁️ ANNUAL ERG MANDATORY:</strong> 10-15% have subclinical Senior-Løken Syndrome 1
            (SLS1) — rod-cone dystrophy. Retina does NOT improve after renal transplant (cell-autonomous).
            Ophthalmology + ERG annually for all NPHP1 patients.
          </Alert>
          <Alert color={ACCENT7}>
            <strong>⚠️ DIAGNOSTIC TRAPS:</strong> (1) MLPA-negative ≠ ruled out — sequence NPHP1 for SNVs.
            (2) ADPKD often suspected first — adult-onset, macrocysts, HTN, AD inheritance are hallmarks of ADPKD;
            NPHP1 is childhood-onset, small kidneys, normotensive. (3) TIN biopsy — always send NPHP1 MLPA
            before labelling "idiopathic TIN."
          </Alert>
          <Alert color={ACCENT4}>
            <strong>🏥 RENAL TRANSPLANT = CURATIVE:</strong> NPHP1 is cell-autonomous — NO recurrence in graft.
            Living-related donors (obligate heterozygotes) are SAFE (single allele sufficient). Coordinate
            transplant planning from CKD stage 3b onwards.
          </Alert>

          {/* Key facts */}
          <Section title="📋 Key Clinical Facts" color={ACCENT}>
            <div className="row">
              <div className="col-md-6">
                <ul className="list-group list-group-flush small">
                  {(overview.key_facts || []).slice(0, 4).map((f, i) => (
                    <li key={i} className="list-group-item px-0 py-1">✅ {f}</li>
                  ))}
                </ul>
              </div>
              <div className="col-md-6">
                <ul className="list-group list-group-flush small">
                  {(overview.key_facts || []).slice(4).map((f, i) => (
                    <li key={i} className="list-group-item px-0 py-1">✅ {f}</li>
                  ))}
                </ul>
              </div>
            </div>
          </Section>

          {/* Disease summary */}
          <div className="row">
            <div className="col-md-6">
              <Section title="🧬 NPHP1 Disease Profile" color={ACCENT}>
                <table className="table table-sm small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>{overview.gene}</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>{overview.omim_gene}</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>{overview.omim_disease}</td></tr>
                    <tr><td className="fw-bold">ESRD Median</td><td>{overview.esrd_median_yr} yr (range {overview.esrd_range_yr})</td></tr>
                    <tr><td className="fw-bold">First Symptom</td><td>{overview.typical_first_symptom}</td></tr>
                    <tr><td className="fw-bold">Dx First Line</td><td style={{ color: ACCENT3 }}>{overview.diagnostic_first_line}</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="🚦 Feature Presence / Absence (vs other NPHP)" color={ACCENT5}>
                <table className="table table-sm small">
                  <tbody>
                    <tr><td>Situs inversus</td><td style={{ color: '#d32f2f' }}><strong>ABSENT</strong> — NPHP1 not in nodal cilia</td></tr>
                    <tr><td>Congenital hepatic fibrosis</td><td style={{ color: '#d32f2f' }}><strong>ABSENT</strong> — not biliary</td></tr>
                    <tr><td>Polydactyly</td><td style={{ color: '#d32f2f' }}><strong>ABSENT</strong> — no skeletal features</td></tr>
                    <tr><td>Intellectual disability</td><td style={{ color: '#d32f2f' }}><strong>ABSENT</strong> (pure renal NPHP1)</td></tr>
                    <tr><td>Retinal dystrophy (SLS1)</td><td style={{ color: ACCENT2 }}><strong>10-15%</strong> (Senior-Løken SLS1)</td></tr>
                    <tr><td>Joubert (JBTS4)</td><td style={{ color: ACCENT8 }}><strong>~5%</strong> (MTS + NPHP1)</td></tr>
                    <tr><td>Concentrating defect</td><td style={{ color: ACCENT5 }}><strong>UNIVERSAL</strong> — earliest sign</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 1: Diagnostic Breakdown ───────────────────────────────────── */}
      {tab === 1 && breakdown && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="🧬 Genetic Architecture (MLPA + WES)" color={ACCENT3}>
                {Object.entries(breakdown.genetic_architecture || {}).map(([k, v]) => (
                  <BarRow key={k} label={k} count={v} total={breakdown.cohort_size} color={ACCENT3} />
                ))}
              </Section>
              <Section title="🩺 CKD Stage Distribution" color={ACCENT4}>
                {Object.entries(breakdown.ckd_stage || {}).map(([k, v]) => (
                  <BarRow key={k} label={k} count={v} total={breakdown.cohort_size} color={ACCENT4} />
                ))}
              </Section>
              <Section title="🌍 Ethnicity" color={ACCENT}>
                {Object.entries(breakdown.ethnicity || {}).map(([k, v]) => (
                  <BarRow key={k} label={k} count={v} total={breakdown.cohort_size} color={ACCENT} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="👁️ Senior-Løken Syndrome 1 (SLS1) Status" color={ACCENT2}>
                {Object.entries(breakdown.sls1_status || {}).map(([k, v]) => (
                  <BarRow key={k} label={k} count={v} total={breakdown.cohort_size} color={ACCENT2} />
                ))}
              </Section>
              <Section title="🧠 Joubert Syndrome 4 (JBTS4) Status" color={ACCENT8}>
                {Object.entries(breakdown.jbts4_status || {}).map(([k, v]) => (
                  <BarRow key={k} label={k} count={v} total={breakdown.cohort_size} color={ACCENT8} />
                ))}
              </Section>
              <Section title="🔍 Prior Misdiagnosis" color={ACCENT7}>
                {Object.entries(breakdown.prior_misdiagnosis || {}).map(([k, v]) => (
                  <BarRow key={k} label={k} count={v} total={breakdown.cohort_size} color={ACCENT7} />
                ))}
              </Section>
              <Section title="⚗️ Urine Osmolality (Concentrating Defect)" color={ACCENT5}>
                {Object.entries(breakdown.urine_osm || {}).map(([k, v]) => (
                  <BarRow key={k} label={k} count={v} total={breakdown.cohort_size} color={ACCENT5} />
                ))}
              </Section>
              <Section title="🩻 Kidney USS Pattern" color={ACCENT6}>
                {Object.entries(breakdown.kidney_uss || {}).map(([k, v]) => (
                  <BarRow key={k} label={k} count={v} total={breakdown.cohort_size} color={ACCENT6} />
                ))}
              </Section>
            </div>
          </div>

          {/* Per-patient table */}
          <Section title="👤 Per-Patient Summary (40 patients)" color={ACCENT}>
            <div style={{ overflowX: 'auto' }}>
              <table className="table table-sm table-striped small">
                <thead>
                  <tr>
                    <th>ID</th><th>Age Dx</th><th>GFR</th><th>Hb</th>
                    <th>CKD Stage</th><th>Genetics</th><th>SLS1</th><th>Misdiagnosis</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.per_patient || []).map(p => (
                    <tr key={p.id}>
                      <td className="fw-bold">{p.id}</td>
                      <td>{p.age_dx_yr} yr</td>
                      <td>{p.gfr}</td>
                      <td>{p.hb}</td>
                      <td className="text-truncate" style={{ maxWidth: 160 }}>{p.ckd_stage}</td>
                      <td className="text-truncate" style={{ maxWidth: 200 }}>{p.genetic}</td>
                      <td className="text-truncate" style={{ maxWidth: 140 }}>{p.sls1}</td>
                      <td className="text-truncate" style={{ maxWidth: 160 }}>{p.misdiag}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 2: Molecular Biology & Diagnosis ─────────────────────────── */}
      {tab === 2 && definitions && (
        <div className="row">
          <div className="col-md-6">
            <Section title="🔬 Molecular Architecture: NPHP1 / Nephrocystin-1" color={ACCENT5}>
              <p className="small">{definitions.mechanism}</p>
            </Section>
            <Section title="🧬 Genetic Architecture & Diagnostic Strategy" color={ACCENT3}>
              <table className="table table-sm small">
                <tbody>
                  {Object.entries(definitions.genetic_architecture || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-capitalize" style={{ width: '35%' }}>{k.replace(/_/g,' ')}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🏥 Treatment & Renal Transplant" color={ACCENT4}>
              <table className="table table-sm small">
                <tbody>
                  {Object.entries(definitions.treatment || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-capitalize" style={{ width: '35%' }}>{k.replace(/_/g,' ')}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="🔑 Key Pathogenic Variants" color={ACCENT3}>
              <ul className="list-group list-group-flush small">
                {(definitions.key_variants || []).map((v, i) => (
                  <li key={i} className="list-group-item px-0 py-1">🧬 {v}</li>
                ))}
              </ul>
            </Section>
            <Section title="⚖️ NPHP Subtype Comparison" color={ACCENT}>
              {Object.entries(definitions.nphp_comparison || {}).map(([k, v]) => (
                <div key={k} className="small mb-1 p-1 rounded" style={{ background: ACCENT + '0a' }}>
                  <strong>{k}:</strong> {v}
                </div>
              ))}
            </Section>
            <Section title="🔍 Differential Diagnosis" color={ACCENT7}>
              {Object.entries(definitions.ddx_table || {}).map(([k, v]) => (
                <div key={k} className="small mb-2 p-2 rounded" style={{ background: ACCENT7 + '10', border: `1px solid ${ACCENT7}30` }}>
                  <div className="fw-bold">{k}</div>
                  <div className="text-muted">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="📊 Prognosis" color={ACCENT4}>
              <p className="small">{definitions.prognosis}</p>
              <div className="small text-muted fst-italic">{definitions.cohort_note}</div>
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ───────────────────────────────────────────── */}
      {tab === 3 && definitions && (
        <div className="row">
          <div className="col-md-6">
            <Section title="📖 Disease & Gene" color={ACCENT}>
              <table className="table table-sm small">
                <tbody>
                  <tr><td className="fw-bold">Disease</td><td>{definitions.disease}</td></tr>
                  <tr><td className="fw-bold">OMIM Gene</td><td>{definitions.omim_gene}</td></tr>
                  <tr><td className="fw-bold">OMIM Disease</td><td>{definitions.omim_disease}</td></tr>
                  <tr><td className="fw-bold">Chromosome</td><td>{definitions.chromosome}</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td>{definitions.inheritance}</td></tr>
                  <tr><td className="fw-bold">Prevalence</td><td>{definitions.prevalence}</td></tr>
                </tbody>
              </table>
            </Section>
            <Section title="🩺 Key Clinical Features" color={ACCENT5}>
              <table className="table table-sm small">
                <tbody>
                  {Object.entries(definitions.key_clinical_features || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-capitalize" style={{ width: '35%' }}>{k.replace(/_/g,' ')}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="🔬 Diagnostic Criteria" color={ACCENT3}>
              <table className="table table-sm small">
                <tbody>
                  {Object.entries(definitions.diagnostic_criteria || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-capitalize" style={{ width: '35%' }}>{k.replace(/_/g,' ')}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🌐 NPHP Comparison Table" color={ACCENT}>
              {Object.entries(definitions.nphp_comparison || {}).map(([k, v]) => (
                <div key={k} className="small mb-2 p-2 rounded" style={{ background: ACCENT + '0d' }}>
                  <span className="fw-bold">{k}:</span> {v}
                </div>
              ))}
            </Section>
            <div className="alert small" style={{ background: ACCENT3 + '15', borderLeft: `4px solid ${ACCENT3}` }}>
              <strong>💡 Cohort Note:</strong> {definitions.cohort_note}
            </div>
          </div>
        </div>
      )}

      {/* Back navigation */}
      <div className="mt-4 pt-3 border-top">
        <Link href="/nphp" className="btn btn-sm btn-outline-secondary me-2">← NPHP Overview</Link>
        <Link href="/nphp2" className="btn btn-sm btn-outline-secondary me-2">NPHP2 →</Link>
        <Link href="/" className="btn btn-sm btn-outline-secondary">← Home</Link>
      </div>
    </div>
  );
}
