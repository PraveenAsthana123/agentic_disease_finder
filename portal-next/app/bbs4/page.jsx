'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Variants & Diagnostics', 'Definitions'];

// BBS4 colour scheme — deep violet-teal-amber-rose (TPR scaffold; PCM1 satellite anchor)
const ACCENT  = '#4a148c';   // deep violet — BBS4 TPR scaffold; pericentriolar satellite
const ACCENT2 = '#006064';   // dark teal — pericentriolar material; PCM1 anchor
const ACCENT3 = '#e65100';   // deep orange — polydactyly; cardinal feature
const ACCENT4 = '#880e4f';   // dark rose — rod-cone degeneration; retinal
const ACCENT5 = '#1a237e';   // navy — renal anomaly
const ACCENT6 = '#37474f';   // dark slate — epidemiology; AR inheritance
const ACCENT7 = '#4e342e';   // dark brown — cognitive/learning disability
const ACCENT8 = '#bf360c';   // burnt orange — obesity; LepR mis-trafficking

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

export default function BBS4Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/bbs4/overview`).then(r => r.json()),
      fetch(`${API}/api/bbs4/breakdown`).then(r => r.json()),
      fetch(`${API}/api/bbs4/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading BBS4 / TPR-Scaffold dashboard…</div>;
  if (error)   return <div className="container py-5 text-center text-danger">Error: {error}</div>;

  const kpi = overview?.kpis || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', borderLeft: `5px solid ${ACCENT}` }}>
        <div className="d-flex flex-wrap align-items-center gap-2 mb-1">
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>BBS4 — Bardet-Biedl Syndrome Type 4</h4>
          <Badge text="BBS4 / *600374" color={ACCENT} />
          <Badge text="Chr 15q24.1" color={ACCENT6} />
          <Badge text="AR Biallelic LOF" color={ACCENT4} />
          <Badge text="~2–4% of BBS" color={ACCENT2} />
          <Badge text="BBSome PCM1 Satellite Anchor (TPR)" color={ACCENT2} />
          <Badge text="40-patient cohort · seed-339" color={ACCENT6} />
        </div>
        <p className="mb-0 small text-muted">
          BBS4 protein contains <strong>8 TPR (tetratricopeptide repeat) domains</strong> that tether the assembling BBSome
          to <strong>PCM1 pericentriolar satellites</strong> — the only BBSome subunit with this function.
          BBS4 LOF → BBSome cannot anchor to PCM1 → BBS5–BBIP10 peripheral module fails to integrate →
          BBSome partial intermediates accumulate → GPCR ciliary trafficking fails → multi-system BBS.
          Unlike BBS3 (ARL6 GTPase), BBS4 failure is an <strong>earlier biogenesis defect</strong>
          (satellite anchoring) upstream of membrane recruitment.
        </p>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button className={`nav-link${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          <Alert color={ACCENT}>
            <strong>Unique mechanism:</strong> BBS4 is the <strong>only BBSome subunit that directly tethers the assembling complex to PCM1 pericentriolar satellites</strong>.
            Anti-BBS4 immunofluorescence in patient fibroblasts (absence of PCM1-colocalised BBS4 puncta) is a
            <em> diagnostic tool unique to BBS4</em> — not available for any other BBS subtype.
          </Alert>

          {/* KPI row 1: clinical features */}
          <div className="row g-2 mb-2">
            <KPI label="Polydactyly" value={`${kpi.polydactyly_n}/${_COHORT_SIZE} (${kpi.polydactyly_pct}%)`} color={ACCENT3} />
            <KPI label="Obesity ≥ BMI 29" value={`${kpi.obesity_n}/${_COHORT_SIZE} (${kpi.obesity_pct}%)`} color={ACCENT8} />
            <KPI label="Cognitive/LD" value={`${kpi.cognitive_n}/${_COHORT_SIZE} (${kpi.cognitive_pct}%)`} color={ACCENT7} />
            <KPI label="Renal Anomaly" value={`${kpi.renal_any_n}/${_COHORT_SIZE} (${kpi.renal_any_pct}%)`} color={ACCENT5} />
            <KPI label="Hypogonadism" value={`${kpi.hypogonadism_n}/${_COHORT_SIZE} (${kpi.hypogonadism_pct}%)`} color={ACCENT4} />
            <KPI label="Anosmia/Hyposmia" value={`${kpi.anosmia_n}/${_COHORT_SIZE} (${kpi.anosmia_pct}%)`} color={ACCENT6} />
          </div>
          <div className="row g-2 mb-3">
            <KPI label="Retinal End-Stage" value={`${kpi.retinal_endstage_n}/${_COHORT_SIZE} (${kpi.retinal_endstage_pct}%)`} color={ACCENT4} />
            <KPI label="CHD" value={`${kpi.chd_n}/${_COHORT_SIZE} (${kpi.chd_pct}%)`} color={ACCENT6} />
            <KPI label="Tri-allelic BBS" value={`${kpi.triallelic_n}/${_COHORT_SIZE} (${kpi.triallelic_pct}%)`} color={ACCENT2} />
            <KPI label="Initial Misdiagnosis" value={`${kpi.misdiagnosis_n}/${_COHORT_SIZE} (${kpi.misdiagnosis_pct}%)`} color={ACCENT3} />
            <KPI label="ESRD Cases" value={kpi.esrd_n} color={ACCENT5} />
            <KPI label="Cohort N" value={`${overview?.cohort_n} (seed ${overview?.seed})`} color={ACCENT6} />
          </div>

          {/* Mechanism */}
          <Section title="Molecular Mechanism — BBS4 TPR Scaffold & PCM1 Pericentriolar Satellite Anchoring" color={ACCENT}>
            <div className="small p-3 rounded" style={{ background: ACCENT + '0d', border: `1px solid ${ACCENT}33` }}>
              {overview?.mechanism}
            </div>
          </Section>

          {/* Key distinction */}
          <Section title="Key Distinction — BBS4 vs BBS1 / BBS2 / BBS3" color={ACCENT2}>
            <div className="small p-3 rounded" style={{ background: ACCENT2 + '0d', border: `1px solid ${ACCENT2}33` }}>
              {overview?.key_distinction}
            </div>
          </Section>

          {/* BBS pathway comparison table */}
          <Section title="BBS Pathway Functional Comparison — BBS4 Position" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '18' }}>
                  <tr>
                    <th>Gene</th>
                    <th>Functional Role</th>
                    <th>Function Class</th>
                    <th>OMIM</th>
                    <th>BBS Frequency</th>
                  </tr>
                </thead>
                <tbody>
                  {overview?.bbs_pathway_comparison?.map((row, i) => (
                    <tr key={i} style={row.gene?.includes('THIS') ? { background: ACCENT + '18', fontWeight: 600 } : {}}>
                      <td><code>{row.gene}</code></td>
                      <td>{row.role}</td>
                      <td><Badge text={row.function_class} color={row.gene?.includes('THIS') ? ACCENT : ACCENT6} /></td>
                      <td><code>{row.omim}</code></td>
                      <td>{row.frequency_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Retinal + Age distribution */}
          <div className="row g-3">
            <div className="col-md-6">
              <Section title="Retinal Dystrophy Stage Distribution (Rod-Cone, Rod First)" color={ACCENT4}>
                {overview?.retinal_distribution?.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Age at Diagnosis Distribution" color={ACCENT}>
                {[
                  { label: '0–4 yr (infant/preschool)', val: overview?.age_distribution?.dx_0_4yr },
                  { label: '5–11 yr (childhood)',        val: overview?.age_distribution?.dx_5_11yr },
                  { label: '12–17 yr (adolescent)',      val: overview?.age_distribution?.dx_12_17yr },
                  { label: '18+ yr (adult)',             val: overview?.age_distribution?.dx_18plus },
                ].map((r, i) => (
                  <Bar key={i} label={r.label} value={r.val || 0} max={_COHORT_SIZE} color={ACCENT} />
                ))}
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 1: Multi-System Breakdown ── */}
      {tab === 1 && breakdown && (
        <div>
          <Alert color={ACCENT2}>
            <strong>BBS4 systemic burden:</strong> Rod-cone dystrophy (100%) → polydactyly (65%) → obesity (80%) → hypogonadism (65%) → anosmia (60%) → cognitive impairment (50%) → renal anomaly (45%) → CHD (6%). Slightly higher cognitive/renal penetrance vs BBS3 (ARL6) — consistent with BBS4's earlier biogenesis defect affecting more ciliary cell types.
          </Alert>

          {/* Systemic burden */}
          <Section title="Systemic Feature Burden (Cohort N=40)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '18' }}>
                  <tr><th>Feature</th><th>N</th><th>Pct</th><th>Bar</th></tr>
                </thead>
                <tbody>
                  {breakdown.systemic_burden?.map((r, i) => (
                    <tr key={i}>
                      <td>{r.feature}</td>
                      <td>{r.n}</td>
                      <td>{r.pct}%</td>
                      <td style={{ width: 160 }}>
                        <div className="progress" style={{ height: 8 }}>
                          <div className="progress-bar" style={{ width: `${r.pct}%`, background: ACCENT }} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <div className="row g-3">
            {/* Ethnicity */}
            <div className="col-md-6">
              <Section title="Ethnicity Distribution" color={ACCENT6}>
                {breakdown.ethnicity_distribution?.map((r, i) => (
                  <Bar key={i} label={r.ethnicity} value={r.n} max={_COHORT_SIZE} color={ACCENT6} />
                ))}
              </Section>
            </div>
            {/* Allele class */}
            <div className="col-md-6">
              <Section title="Allele Class Summary" color={ACCENT}>
                {breakdown.allele_class_summary?.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT} />
                ))}
              </Section>
            </div>
          </div>

          <div className="row g-3">
            {/* Retinal stage */}
            <div className="col-md-6">
              <Section title="Retinal Stage at Diagnosis" color={ACCENT4}>
                {breakdown.retinal_stage_distribution?.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
              </Section>
            </div>
            {/* Renal */}
            <div className="col-md-6">
              <Section title="Renal Anomaly Distribution" color={ACCENT5}>
                {breakdown.renal_distribution?.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT5} />
                ))}
              </Section>
            </div>
          </div>

          <div className="row g-3">
            {/* Polydactyly */}
            <div className="col-md-6">
              <Section title="Polydactyly Distribution" color={ACCENT3}>
                {breakdown.polydactyly_distribution?.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>
            </div>
            {/* Presentation age */}
            <div className="col-md-6">
              <Section title="Age at Presentation" color={ACCENT2}>
                {breakdown.presentation_distribution?.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT2} />
                ))}
              </Section>
            </div>
          </div>

          {/* Misdiagnosis */}
          <Section title="Initial Misdiagnosis Distribution" color={ACCENT3}>
            {breakdown.misdiagnosis_distribution?.map((r, i) => (
              <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT3} />
            ))}
          </Section>

          {/* Top variants */}
          <Section title="Top Recurrent Variants (Educational Cohort)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '18' }}>
                  <tr><th>Variant</th><th>N</th></tr>
                </thead>
                <tbody>
                  {breakdown.top_variants?.map((v, i) => (
                    <tr key={i}>
                      <td>{v.variant}</td>
                      <td>{v.n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 2: Variants & Diagnostics ── */}
      {tab === 2 && definitions && (
        <div>
          <Alert color={ACCENT}>
            <strong>Diagnostic key:</strong> Full BBS gene panel (20–24 genes) mandatory — BBS4 clinically indistinguishable from BBS1/BBS2/BBS3 without genetics. BBS4 VUS: classify by TPR domain position (TPR1–4 = BBS5 interface; TPR5–8 = PCM1/BBIP10 anchor). Confirm LOF with anti-BBS4 IF in patient fibroblasts (absence of PCM1-colocalised puncta = functional null).
          </Alert>

          {/* Key variants table */}
          <Section title="Key BBS4 Pathogenic Variants" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '18' }}>
                  <tr><th>Variant</th><th>Domain / Position</th><th>Consequence</th><th>Ethnicity</th></tr>
                </thead>
                <tbody>
                  {definitions.key_variants?.map((v, i) => (
                    <tr key={i}>
                      <td><code className="fw-bold">{v.variant}</code></td>
                      <td>{v.domain}</td>
                      <td>{v.consequence}</td>
                      <td>{v.ethnicity}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Diagnostic workup */}
          <Section title="Diagnostic Workup — BBS4" color={ACCENT2}>
            <ol className="small ps-3">
              {definitions.diagnostic_workup?.map((step, i) => (
                <li key={i} className="mb-1">{step.replace(/^\d+\.\s*/, '')}</li>
              ))}
            </ol>
          </Section>

          {/* Treatment summary */}
          <Section title="Management Summary" color={ACCENT5}>
            <ol className="small ps-3">
              {definitions.treatment_summary?.map((step, i) => (
                <li key={i} className="mb-1">{step.replace(/^\d+\.\s*/, '')}</li>
              ))}
            </ol>
          </Section>

          {/* DDx table */}
          <Section title="Differential Diagnosis" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT3 + '18' }}>
                  <tr><th>Disease</th><th>Key Differentiator from BBS4</th></tr>
                </thead>
                <tbody>
                  {definitions.ddx_table?.map((r, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{r.disease}</td>
                      <td>{r.key_difference}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && definitions && (
        <div>
          {/* Gene card */}
          <Section title="Gene Card — BBS4 (*600374)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(definitions.gene_card || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-semibold text-nowrap" style={{ width: 180, color: ACCENT }}>{k.replace(/_/g, ' ')}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Disease card */}
          <Section title="Disease Card — BBS4 (#209900)" color={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(definitions.disease_card || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-semibold text-nowrap" style={{ width: 180, color: ACCENT4 }}>{k.replace(/_/g, ' ')}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Mechanism glossary */}
          <Section title="Mechanism Glossary — BBS4 TPR Domains, PCM1 Satellites & BBSome Biogenesis" color={ACCENT2}>
            {definitions.mechanism_glossary?.map((g, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT2}33`, background: ACCENT2 + '08' }}>
                <div className="fw-bold small mb-1" style={{ color: ACCENT2 }}>{g.term}</div>
                <div className="small">{g.definition}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* Footer breadcrumb */}
      <div className="mt-4 pt-3 border-top small text-muted">
        <Link href="/" className="text-decoration-none">← Home</Link>
        {' · '}
        <Link href="/bbs" className="text-decoration-none">BBS1 (BBS1)</Link>
        {' · '}
        <Link href="/bbs2" className="text-decoration-none">BBS2 (BBS2)</Link>
        {' · '}
        <Link href="/bbs3" className="text-decoration-none">BBS3 (ARL6)</Link>
        {' · '}
        <strong>BBS4 (BBS4)</strong>
        {' · '}
        <span>OMIM Gene: <a href="https://www.omim.org/entry/600374" target="_blank" rel="noreferrer" className="text-decoration-none">*600374</a></span>
        {' · '}
        <span>Disease OMIM: <a href="https://www.omim.org/entry/209900" target="_blank" rel="noreferrer" className="text-decoration-none">#209900</a></span>
      </div>
    </div>
  );
}
