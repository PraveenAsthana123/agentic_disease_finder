'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Variants & Diagnostics', 'Definitions'];

// BBS7 colour scheme — steel-blue / indigo / teal (WD40 structural; core dimer; obligate partner of BBS2)
const ACCENT  = '#1a237e';   // deep indigo — structural core dimer; WD40 beta-propeller
const ACCENT2 = '#004d40';   // deep teal — BBS2-BBS7 obligate heterodimer
const ACCENT3 = '#e65100';   // deep orange — polydactyly; cardinal feature
const ACCENT4 = '#880e4f';   // dark rose — rod-cone degeneration; retinal
const ACCENT5 = '#01579b';   // dark blue — renal anomaly
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

export default function BBS7Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/bbs7/overview`).then(r => r.json()),
      fetch(`${API}/api/bbs7/breakdown`).then(r => r.json()),
      fetch(`${API}/api/bbs7/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading BBS7 Core-Dimer dashboard…</div>;
  if (error)   return <div className="container py-5 text-center text-danger">Error: {error}</div>;

  const kpi = overview?.kpis || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', borderLeft: `5px solid ${ACCENT}` }}>
        <div className="d-flex flex-wrap align-items-center gap-2 mb-1">
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>BBS7 — Bardet-Biedl Syndrome Type 7</h4>
          <Badge text="BBS7 / *607590" color={ACCENT} />
          <Badge text="Chr 4q27" color={ACCENT6} />
          <Badge text="AR Biallelic LOF" color={ACCENT4} />
          <Badge text="~2–4% of BBS" color={ACCENT2} />
          <Badge text="BBSome Core Dimer (BBS2-BBS7)" color={ACCENT2} />
          <Badge text="40-patient cohort · seed-345" color={ACCENT6} />
        </div>
        <p className="mb-0 small text-muted">
          BBS7 encodes a <strong>672 aa WD40 beta-propeller protein</strong> (also called BBS2L1 — BBS2-like protein 1)
          that is the <strong>obligate structural dimer partner of BBS2</strong> in the BBSome octamer core module.
          The BBS2-BBS7 heterodimer is Step 0 of BBSome assembly — without it, the full BBSome octamer cannot form.
          BBS7 LOF phenotype is <strong>clinically and cellularly indistinguishable from BBS2 LOF</strong> (both show
          absent BBS2 IF + absent BBS7 IF with <em>normal MKKS IF</em> — the key contrast with BBS6/MKKS).
          <strong>Gene panel is the only discriminator between BBS2 and BBS7.</strong>{' '}
          Tri-allelic BBS (~7%) most commonly involves BBS2 as the third allele — reflecting core-dimer partner co-dependence.
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
            <strong>Core dimer assembly step (Step 0):</strong> BBS7 and BBS2 form an <strong>obligate WD40 beta-propeller heterodimer</strong> — neither
            protein is stable without the other. BBS7 LOF → orphaned BBS2 degraded by proteasome → BBSome core module fails →
            identical downstream cascade to BBS2 LOF. Critical: anti-MKKS IF is <strong>NORMAL</strong> in BBS7 LOF
            (BCC chaperonin successfully folded BBS7; failure is at assembly, not folding) — this distinguishes BBS7 from BBS6/MKKS.
            BBS2 vs BBS7 cannot be distinguished by any clinical or cellular marker — <strong>gene panel only</strong>.
          </Alert>

          {/* KPI row 1: clinical features */}
          <div className="row g-2 mb-2">
            <KPI label="Polydactyly" value={`${kpi.polydactyly_n}/${_COHORT_SIZE} (${kpi.polydactyly_pct}%)`} color={ACCENT3} />
            <KPI label="Obesity ≥ BMI 28" value={`${kpi.obesity_n}/${_COHORT_SIZE} (${kpi.obesity_pct}%)`} color={ACCENT8} />
            <KPI label="Cognitive/LD" value={`${kpi.cognitive_n}/${_COHORT_SIZE} (${kpi.cognitive_pct}%)`} color={ACCENT7} />
            <KPI label="Renal Anomaly" value={`${kpi.renal_any_n}/${_COHORT_SIZE} (${kpi.renal_any_pct}%)`} color={ACCENT5} />
            <KPI label="Hypogonadism" value={`${kpi.hypogonadism_n}/${_COHORT_SIZE} (${kpi.hypogonadism_pct}%)`} color={ACCENT4} />
            <KPI label="Anosmia/Hyposmia" value={`${kpi.anosmia_n}/${_COHORT_SIZE} (${kpi.anosmia_pct}%)`} color={ACCENT6} />
          </div>
          <div className="row g-2 mb-3">
            <KPI label="CHD" value={`${kpi.chd_n}/${_COHORT_SIZE} (${kpi.chd_pct}%)`} color={ACCENT4} />
            <KPI label="Retinal End-Stage" value={`${kpi.retinal_endstage_n}/${_COHORT_SIZE} (${kpi.retinal_endstage_pct}%)`} color={ACCENT4} />
            <KPI label="Tri-allelic BBS" value={`${kpi.triallelic_n}/${_COHORT_SIZE} (${kpi.triallelic_pct}%)`} color={ACCENT2} />
            <KPI label="Initial Misdiagnosis" value={`${kpi.misdiagnosis_n}/${_COHORT_SIZE} (${kpi.misdiagnosis_pct}%)`} color={ACCENT3} />
            <KPI label="ESRD Cases" value={kpi.esrd_n} color={ACCENT5} />
            <KPI label="Cohort N" value={`${overview?.cohort_n} (seed ${overview?.seed})`} color={ACCENT6} />
          </div>

          {/* Mechanism */}
          <Section title="Molecular Mechanism — BBS7 as BBS2 Obligate Dimer Partner & BBSome Core Module Step 0" color={ACCENT}>
            <div className="small p-3 rounded" style={{ background: ACCENT + '0d', border: `1px solid ${ACCENT}33` }}>
              {overview?.mechanism}
            </div>
          </Section>

          {/* Key distinction */}
          <Section title="Key Distinction — BBS7 vs BBS1 / BBS2 / BBS3 / BBS4 / BBS5 / BBS6" color={ACCENT2}>
            <div className="small p-3 rounded" style={{ background: ACCENT2 + '0d', border: `1px solid ${ACCENT2}33` }}>
              {overview?.key_distinction}
            </div>
          </Section>

          {/* BBS pathway comparison table */}
          <Section title="BBS Pathway Functional Comparison — BBS7 Position (Core Dimer with BBS2)" color={ACCENT}>
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
            <strong>BBS7 systemic burden:</strong> Rod-cone dystrophy (100%) → polydactyly (~63%) → obesity (~80%) →
            hypogonadism (~60%) → anosmia (~55%) → cognitive impairment (~45%) → renal anomaly (~40%) → CHD (~5%).
            Clinical profile is <strong>identical to BBS2</strong> — no phenotypic marker distinguishes these two core-dimer partners.
            Tri-allelic rate (~7%) is highest for BBS2 as third allele (dimer partner relationship).
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
              <Section title="Ethnicity Distribution (no dominant founder allele)" color={ACCENT6}>
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
            <strong>Diagnostic key:</strong> Full BBS gene panel (20–24 genes) mandatory — BBS7 is clinically and cellularly
            indistinguishable from BBS2 LOF. Both show absent BBS2 + absent BBS7 IF with <strong>NORMAL MKKS IF</strong>.
            If MKKS IF is absent → suspect BBS6 (MKKS) LOF, not BBS7. Tri-allelic screen: always co-test BBS2
            (core-dimer partner; most common third allele ~7%) and BBS9 (bridge subunit).
          </Alert>

          {/* Key variants table */}
          <Section title="Key BBS7 Pathogenic Variants" color={ACCENT}>
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
          <Section title="Diagnostic Workup — BBS7" color={ACCENT2}>
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
                  <tr><th>Disease</th><th>Key Differentiator from BBS7</th></tr>
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
          <Section title="Gene Card — BBS7 (*607590)" color={ACCENT}>
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
          <Section title="Disease Card — BBS7 / Bardet-Biedl Syndrome (#209900)" color={ACCENT4}>
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
          <Section title="Mechanism Glossary — BBS2-BBS7 Obligate Dimer, BBS9 Bridge & Tri-allelic BBS" color={ACCENT2}>
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
        <Link href="/bbs4" className="text-decoration-none">BBS4 (BBS4)</Link>
        {' · '}
        <Link href="/bbs5" className="text-decoration-none">BBS5 (BBS5)</Link>
        {' · '}
        <Link href="/bbs6" className="text-decoration-none">BBS6 (MKKS)</Link>
        {' · '}
        <strong>BBS7 (BBS7)</strong>
        {' · '}
        <span>OMIM Gene: <a href="https://www.omim.org/entry/607590" target="_blank" rel="noreferrer" className="text-decoration-none">*607590</a></span>
        {' · '}
        <span>Disease OMIM: <a href="https://www.omim.org/entry/209900" target="_blank" rel="noreferrer" className="text-decoration-none">#209900</a></span>
      </div>
    </div>
  );
}
