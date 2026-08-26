'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Variants & Diagnostics', 'Definitions'];

// BBS10 colour scheme — deep teal / amber / cobalt (BCC β-ring; highest frequency; BCC upstream folding)
const ACCENT  = '#004d40';   // deep teal — BCC chaperonin; upstream folding complex
const ACCENT2 = '#e65100';   // deep amber/orange — high frequency (20% of BBS); prominent
const ACCENT3 = '#1a237e';   // deep cobalt — BCC β-ring structural role; equatorial domain
const ACCENT4 = '#880e4f';   // dark rose — rod-cone degeneration; retinal
const ACCENT5 = '#01579b';   // dark blue — renal anomaly
const ACCENT6 = '#37474f';   // dark slate — epidemiology; AR inheritance
const ACCENT7 = '#4e342e';   // dark brown — cognitive/learning disability
const ACCENT8 = '#558b2f';   // dark olive — obesity; LepR mis-trafficking

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
        <span>{label}</span><span className="fw-bold">{value} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function BBS10Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/bbs10/overview`).then(r => r.json()),
      fetch(`${API}/api/bbs10/breakdown`).then(r => r.json()),
      fetch(`${API}/api/bbs10/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="d-flex align-items-center justify-content-center" style={{ minHeight: 320 }}><div className="spinner-border" style={{ color: ACCENT }} /><span className="ms-3 text-muted">Loading BBS10 dashboard…</span></div>;
  if (error)   return <div className="alert alert-danger m-4">Error: {error}</div>;

  const ov = overview;
  const br = breakdown;
  const df = definitions;
  const n  = _COHORT_SIZE;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-start gap-3 mb-3">
        <div>
          <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
            &#x1f9ec; BBS10 — Bardet-Biedl Syndrome Type 10
          </h4>
          <div className="text-muted small">
            <Badge text="BBS10 / C12orf58" color={ACCENT} />
            <Badge text="Chr 12q21.2" color={ACCENT2} />
            <Badge text="723 aa" color={ACCENT3} />
            <Badge text="OMIM *610148 / #209900" color={ACCENT6} />
            <Badge text="BCC β-Ring Subunit" color={ACCENT} />
            <Badge text="~20% of All BBS" color={ACCENT2} />
            <Badge text="AR Biallelic LOF" color={ACCENT6} />
          </div>
        </div>
        <Link href="/" className="btn btn-sm btn-outline-secondary ms-auto">← Portal</Link>
      </div>

      {/* Frequency alert — BBS10 is among the most common */}
      <Alert color={ACCENT2}>
        <strong>High-Frequency BBS Subtype:</strong> BBS10 accounts for <strong>~20% of all Bardet-Biedl Syndrome</strong> cases worldwide — tied with BBS1 as the most common BBS subtype. Together, BBS1 + BBS10 explain ~40–45% of all BBS. BBS10 is the dominant BBS gene in European and North African (Maghrebi) populations.
      </Alert>

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0 — Overview ── */}
      {tab === 0 && ov && (
        <>
          <Section title="Mechanism — BCC β-Ring Structural Subunit" color={ACCENT}>
            <p className="small mb-2">{ov.mechanism}</p>
            <Alert color={ACCENT3}>
              <strong>BCC β-Ring Role:</strong> BBS10 (723 aa; 12q21.2) is the structural β-ring subunit of the BBSome Chaperonin Complex (BCC: MKKS · BBS10 · BBS12). BCC co-folds BBS2 and BBS7 WD40 β-propellers before BBSome assembly. BBS10 LOF → BCC trimer collapses → MKKS monomeric (REDUCED IF, not absent) → BBS2/BBS7 misfold → Step 0 fails → no BBSome → full BBS.
            </Alert>
          </Section>

          {/* KPI cards */}
          <Section title={`Cohort KPIs — ${n} Patients (Educational/Synthetic · Seed ${ov.seed})`} color={ACCENT2}>
            <div className="row g-2">
              <KPI label="Polydactyly"       value={`${ov.kpis.polydactyly_n} (${ov.kpis.polydactyly_pct}%)`}       color={ACCENT3} />
              <KPI label="Obesity"           value={`${ov.kpis.obesity_n} (${ov.kpis.obesity_pct}%)`}               color={ACCENT8} />
              <KPI label="Cognitive/LD"      value={`${ov.kpis.cognitive_n} (${ov.kpis.cognitive_pct}%)`}           color={ACCENT7} />
              <KPI label="Renal (any)"       value={`${ov.kpis.renal_any_n} (${ov.kpis.renal_any_pct}%)`}          color={ACCENT5} />
              <KPI label="Hypogonadism"      value={`${ov.kpis.hypogonadism_n} (${ov.kpis.hypogonadism_pct}%)`}   color={ACCENT6} />
              <KPI label="Anosmia"           value={`${ov.kpis.anosmia_n} (${ov.kpis.anosmia_pct}%)`}             color={ACCENT} />
              <KPI label="CHD"               value={`${ov.kpis.chd_n} (${ov.kpis.chd_pct}%)`}                     color={ACCENT4} />
              <KPI label="Retinal End-Stage" value={`${ov.kpis.retinal_endstage_n} (${ov.kpis.retinal_endstage_pct}%)`} color={ACCENT4} />
              <KPI label="Tri-allelic BBS"   value={`${ov.kpis.triallelic_n} (${ov.kpis.triallelic_pct}%)`}       color={ACCENT2} />
              <KPI label="Misdiagnosis"      value={`${ov.kpis.misdiagnosis_n} (${ov.kpis.misdiagnosis_pct}%)`}   color={ACCENT6} />
              <KPI label="ESRD"              value={`${ov.kpis.esrd_n}`}                                            color={ACCENT5} />
            </div>
          </Section>

          {/* Key distinction */}
          <Section title="Key Distinction — BBS10 vs BBS6/MKKS and Other BBS Types" color={ACCENT3}>
            <p className="small">{ov.key_distinction}</p>
          </Section>

          {/* BBS pathway table */}
          <Section title="BBS Pathway — All Known Types (BBS1–10)" color={ACCENT2}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT2 + '22' }}>
                  <tr><th>Gene</th><th>Role</th><th>Function Class</th><th>OMIM</th><th>Freq (% BBS)</th></tr>
                </thead>
                <tbody>
                  {ov.bbs_pathway_comparison.map((r, i) => (
                    <tr key={i} style={r.gene.includes('[THIS]') ? { background: ACCENT + '22', fontWeight: 700 } : {}}>
                      <td>{r.gene}</td><td>{r.role}</td><td>{r.function_class}</td>
                      <td>{r.omim}</td><td>{r.frequency_pct}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Retinal + Age */}
          <div className="row">
            <div className="col-md-6">
              <Section title="Retinal Stage Distribution" color={ACCENT4}>
                {ov.retinal_distribution.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={n} color={ACCENT4} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Age at Diagnosis" color={ACCENT6}>
                {[
                  { label: '0–4 yr',   n: ov.age_distribution.dx_0_4yr },
                  { label: '5–11 yr',  n: ov.age_distribution.dx_5_11yr },
                  { label: '12–17 yr', n: ov.age_distribution.dx_12_17yr },
                  { label: '18+ yr',   n: ov.age_distribution.dx_18plus },
                ].map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={n} color={ACCENT6} />
                ))}
              </Section>
            </div>
          </div>
        </>
      )}

      {/* ── TAB 1 — Multi-System Breakdown ── */}
      {tab === 1 && br && (
        <>
          <Section title="Systemic Feature Burden" color={ACCENT}>
            {br.systemic_burden.map((r, i) => (
              <Bar key={i} label={r.feature} value={r.n} max={n} color={i % 2 === 0 ? ACCENT : ACCENT2} />
            ))}
          </Section>

          <div className="row">
            <div className="col-md-6">
              <Section title="Ethnicity Distribution" color={ACCENT2}>
                {br.ethnicity_distribution.map((r, i) => (
                  <Bar key={i} label={r.ethnicity} value={r.n} max={n} color={ACCENT2} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Allele Class" color={ACCENT3}>
                {br.allele_class_summary.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={n} color={ACCENT3} />
                ))}
              </Section>
            </div>
          </div>

          <div className="row">
            <div className="col-md-6">
              <Section title="Retinal Stage" color={ACCENT4}>
                {br.retinal_stage_distribution.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={n} color={ACCENT4} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Polydactyly Type" color={ACCENT3}>
                {br.polydactyly_distribution.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={n} color={ACCENT3} />
                ))}
              </Section>
            </div>
          </div>

          <div className="row">
            <div className="col-md-6">
              <Section title="Renal Phenotype" color={ACCENT5}>
                {br.renal_distribution.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={n} color={ACCENT5} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Presentation Age" color={ACCENT6}>
                {br.presentation_distribution.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={n} color={ACCENT6} />
                ))}
              </Section>
            </div>
          </div>

          <div className="row">
            <div className="col-md-6">
              <Section title="Misdiagnosis Pattern" color={ACCENT7}>
                {br.misdiagnosis_distribution.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={n} color={ACCENT7} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Top Variants Observed" color={ACCENT}>
                {br.top_variants.map((r, i) => (
                  <Bar key={i} label={r.variant} value={r.n} max={n * 2} color={ACCENT} />
                ))}
              </Section>
            </div>
          </div>
        </>
      )}

      {/* ── TAB 2 — Variants & Diagnostics ── */}
      {tab === 2 && df && (
        <>
          <Section title="Key Pathogenic Variants — BBS10 (C12orf58)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '22' }}>
                  <tr><th>Variant</th><th>Domain</th><th>Consequence</th><th>Ethnicity</th></tr>
                </thead>
                <tbody>
                  {df.key_variants.map((v, i) => (
                    <tr key={i}>
                      <td><code>{v.variant}</code></td>
                      <td>{v.domain}</td>
                      <td>{v.consequence}</td>
                      <td><Badge text={v.ethnicity} color={ACCENT2} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Diagnostic Workup — BBS10" color={ACCENT2}>
            <ol className="small ps-3">
              {df.diagnostic_workup.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
            </ol>
          </Section>

          <Section title="Treatment Summary" color={ACCENT3}>
            <ol className="small ps-3">
              {df.treatment_summary.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
            </ol>
          </Section>

          <Section title="Differential Diagnosis Table" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT6 + '22' }}>
                  <tr><th>Condition</th><th>Key Difference from BBS10</th></tr>
                </thead>
                <tbody>
                  {df.ddx_table.map((r, i) => (
                    <tr key={i}><td><strong>{r.disease}</strong></td><td>{r.key_difference}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </>
      )}

      {/* ── TAB 3 — Definitions ── */}
      {tab === 3 && df && (
        <>
          <Section title="Gene Card — BBS10" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(df.gene_card).map(([k, v]) => (
                    <tr key={k}><th style={{ width: '22%', background: ACCENT + '11' }}>{k.replace(/_/g, ' ')}</th><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Disease Card — Bardet-Biedl Syndrome Type 10" color={ACCENT2}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(df.disease_card).map(([k, v]) => (
                    <tr key={k}><th style={{ width: '22%', background: ACCENT2 + '11' }}>{k.replace(/_/g, ' ')}</th><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Mechanism Glossary" color={ACCENT3}>
            {df.mechanism_glossary.map((g, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ background: ACCENT3 + '0d', border: `1px solid ${ACCENT3}33` }}>
                <div className="fw-bold small mb-1" style={{ color: ACCENT3 }}>{g.term}</div>
                <div className="small">{g.definition}</div>
              </div>
            ))}
          </Section>
        </>
      )}

      <div className="text-muted mt-4" style={{ fontSize: '0.72em' }}>
        Educational/synthetic cohort · {_COHORT_SIZE} patients · Seed 351 · BBS10 / C12orf58 · Chr 12q21.2 · OMIM *610148 · #209900 · BCC β-Ring Structural Subunit · ~20% of All BBS
      </div>
    </div>
  );
}
