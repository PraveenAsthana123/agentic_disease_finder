'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Variants & Diagnostics', 'Definitions'];

// BBS12 colour scheme — deep indigo / forest green / burnt sienna (BCC γ-ring; consanguineous; MENA/North African)
const ACCENT  = '#1a237e';   // deep indigo — BCC chaperonin γ-ring; upstream folding complex
const ACCENT2 = '#1b5e20';   // forest green — enriched in consanguineous populations; North African
const ACCENT3 = '#bf360c';   // burnt sienna / deep orange — polydactyly; distinct phenotype
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
      <div style={{ background: '#e9ecef', borderRadius: 4, height: 10 }}>
        <div style={{ width: `${pct}%`, background: color, borderRadius: 4, height: 10 }} />
      </div>
    </div>
  );
}

function useFetch(path) {
  const [data, setData] = useState(null);
  const [err, setErr]   = useState(null);
  useEffect(() => {
    fetch(`${API}${path}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(setData)
      .catch(e => setErr(String(e)));
  }, [path]);
  return { data, err };
}

// ── Overview Tab ─────────────────────────────────────────────────────────────
function OverviewTab() {
  const { data, err } = useFetch('/api/bbs12/overview');
  if (err)  return <div className="alert alert-danger">Error: {err}</div>;
  if (!data) return <div className="text-muted p-4">Loading…</div>;
  const k = data.kpis;

  return (
    <div>
      {/* Fingerprint alert */}
      <Alert color={ACCENT}>
        <strong>🔬 BBS12 IF Fingerprint (BCC γ-Ring LOF):</strong>{' '}
        BBS12 <strong>ABSENT</strong> · MKKS <strong>REDUCED</strong> (not absent) ·
        BBS10 <strong>REDUCED</strong> · BBS2 <strong>ABSENT</strong> — almost identical to BBS10 LOF;
        gene panel is the definitive discriminator between BBS10 LOF and BBS12 LOF.
      </Alert>
      <Alert color={ACCENT2}>
        <strong>🌍 Population:</strong>{' '}
        BBS12 is enriched in <strong>consanguineous North African (Maghrebi)</strong> and{' '}
        <strong>Middle Eastern</strong> families. Identified by Stoetzel et al. 2007 (Nat Genet) in
        Tunisian/Moroccan cohorts. <strong>Arg140Cys</strong> is the North African founder allele.
        ~5–8% of all BBS worldwide; 4th–5th most common BBS gene.
      </Alert>

      {/* KPI row */}
      <Section title="Cohort KPIs — 40 Patients (seed 353)" color={ACCENT}>
        <div className="row g-2">
          <KPI label="Polydactyly"    value={`${k.polydactyly_n} (${k.polydactyly_pct}%)`}    color={ACCENT3} />
          <KPI label="Obesity"        value={`${k.obesity_n} (${k.obesity_pct}%)`}             color={ACCENT8} />
          <KPI label="Cognitive/LD"   value={`${k.cognitive_n} (${k.cognitive_pct}%)`}         color={ACCENT7} />
          <KPI label="Hypogonadism"   value={`${k.hypogonadism_n} (${k.hypogonadism_pct}%)`}  color={ACCENT6} />
          <KPI label="Anosmia"        value={`${k.anosmia_n} (${k.anosmia_pct}%)`}            color={ACCENT2} />
          <KPI label="Renal (any)"    value={`${k.renal_any_n} (${k.renal_any_pct}%)`}        color={ACCENT5} />
          <KPI label="CHD"            value={`${k.chd_n} (${k.chd_pct}%)`}                    color={ACCENT4} />
          <KPI label="Retinal End-Stage" value={`${k.retinal_endstage_n} (${k.retinal_endstage_pct}%)`} color={ACCENT4} />
          <KPI label="Tri-Allelic BBS" value={`${k.triallelic_n} (${k.triallelic_pct}%)`}    color={ACCENT} />
          <KPI label="Consanguinity"  value={`${k.consanguinity_n} (${k.consanguinity_pct}%)`} color={ACCENT2} />
          <KPI label="Misdiagnosis"   value={`${k.misdiagnosis_n} (${k.misdiagnosis_pct}%)`}  color={ACCENT6} />
          <KPI label="ESRD"           value={`${k.esrd_n}`}                                    color={ACCENT5} />
        </div>
      </Section>

      {/* Mechanism */}
      <Section title="Molecular Mechanism — BCC γ-Ring Structural Subunit" color={ACCENT2}>
        <p className="small">{data.mechanism}</p>
      </Section>

      {/* Key distinction */}
      <Section title="Key Clinical Distinction vs BBS6 / BBS10 / BBS12 — IF Panel Logic" color={ACCENT3}>
        <p className="small">{data.key_distinction}</p>
      </Section>

      {/* Retinal distribution */}
      <Section title="Retinal Degeneration Stage Distribution" color={ACCENT4}>
        {data.retinal_distribution.map(r => (
          <Bar key={r.label} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT4} />
        ))}
      </Section>

      {/* Age at dx */}
      <Section title="Age at Diagnosis Distribution" color={ACCENT6}>
        <div className="row">
          {Object.entries(data.age_distribution).map(([k, v]) => (
            <div key={k} className="col-6 col-md-3 mb-2 text-center">
              <div className="fw-bold" style={{ color: ACCENT6, fontSize: '1.3em' }}>{v}</div>
              <div className="text-muted small">{k.replace(/_/g,' ').replace('dx ','Dx ')}</div>
            </div>
          ))}
        </div>
      </Section>

      {/* BBS pathway comparison */}
      <Section title="BBS Pathway Comparison — BBS1–BBS12 Mechanism Classes" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-light">
              <tr>
                <th>Gene</th><th>Role</th><th>Function Class</th><th>OMIM</th><th>Frequency</th>
              </tr>
            </thead>
            <tbody>
              {data.bbs_pathway_comparison.map(r => (
                <tr key={r.gene} style={r.gene.includes('[THIS]') ? { background: ACCENT + '18', fontWeight: 600 } : {}}>
                  <td>{r.gene}</td>
                  <td>{r.role}</td>
                  <td>{r.function_class}</td>
                  <td>{r.omim}</td>
                  <td>{r.frequency_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

// ── Breakdown Tab ─────────────────────────────────────────────────────────────
function BreakdownTab() {
  const { data, err } = useFetch('/api/bbs12/breakdown');
  if (err)  return <div className="alert alert-danger">Error: {err}</div>;
  if (!data) return <div className="text-muted p-4">Loading…</div>;

  return (
    <div>
      {/* Systemic burden */}
      <Section title="Systemic Feature Burden" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped small">
            <thead className="table-light">
              <tr><th>Feature</th><th>N</th><th>%</th></tr>
            </thead>
            <tbody>
              {data.systemic_burden.map(r => (
                <tr key={r.feature}>
                  <td>{r.feature}</td>
                  <td>{r.n}</td>
                  <td>{r.pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Ethnicity */}
      <Section title="Ethnicity Distribution" color={ACCENT2}>
        {data.ethnicity_distribution.map(r => (
          <Bar key={r.ethnicity} label={r.ethnicity} value={r.n} max={40} color={ACCENT2} />
        ))}
      </Section>

      {/* Allele class */}
      <Section title="Allele Class Summary" color={ACCENT3}>
        {data.allele_class_summary.map(r => (
          <Bar key={r.label} label={r.label} value={r.n} max={40} color={ACCENT3} />
        ))}
      </Section>

      <div className="row">
        <div className="col-md-6">
          {/* Retinal stage */}
          <Section title="Retinal Stage" color={ACCENT4}>
            {data.retinal_stage_distribution.map(r => (
              <Bar key={r.label} label={r.label} value={r.n} max={40} color={ACCENT4} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          {/* Renal */}
          <Section title="Renal Distribution" color={ACCENT5}>
            {data.renal_distribution.map(r => (
              <Bar key={r.label} label={r.label} value={r.n} max={40} color={ACCENT5} />
            ))}
          </Section>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          {/* Polydactyly */}
          <Section title="Polydactyly Type" color={ACCENT3}>
            {data.polydactyly_distribution.map(r => (
              <Bar key={r.label} label={r.label} value={r.n} max={40} color={ACCENT3} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          {/* Presentation */}
          <Section title="Presentation Age" color={ACCENT6}>
            {data.presentation_distribution.map(r => (
              <Bar key={r.label} label={r.label} value={r.n} max={40} color={ACCENT6} />
            ))}
          </Section>
        </div>
      </div>

      {/* Misdiagnosis */}
      <Section title="Initial Misdiagnosis" color={ACCENT7}>
        {data.misdiagnosis_distribution.map(r => (
          <Bar key={r.label} label={r.label} value={r.n} max={40} color={ACCENT7} />
        ))}
      </Section>

      {/* Top variants */}
      <Section title="Most Frequent Variants (by cohort allele count)" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-light"><tr><th>Variant</th><th>Allele Count</th></tr></thead>
            <tbody>
              {data.top_variants.map(r => (
                <tr key={r.variant}><td>{r.variant}</td><td>{r.n}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

// ── Variants & Diagnostics Tab ────────────────────────────────────────────────
function VariantsTab() {
  const { data, err } = useFetch('/api/bbs12/definitions');
  if (err)  return <div className="alert alert-danger">Error: {err}</div>;
  if (!data) return <div className="text-muted p-4">Loading…</div>;

  return (
    <div>
      {/* Key variants */}
      <Section title="Key Variants — BBS12 (729 aa; Chr 4q27)" color={ACCENT}>
        {data.key_variants.map(v => (
          <div key={v.variant} className="mb-3 p-3 rounded" style={{ background: ACCENT + '0a', border: `1px solid ${ACCENT}40` }}>
            <div className="fw-bold mb-1" style={{ color: ACCENT }}>{v.variant}</div>
            <div className="small mb-1"><Badge text="Domain" color={ACCENT6} /> {v.domain}</div>
            <div className="small mb-1"><Badge text="Consequence" color={ACCENT3} /> {v.consequence}</div>
            <div className="small"><Badge text="Ethnicity" color={ACCENT2} /> {v.ethnicity}</div>
          </div>
        ))}
      </Section>

      {/* Diagnostic workup */}
      <Section title="Diagnostic Workup — BCC γ-Ring LOF" color={ACCENT2}>
        <ol className="small ps-3">
          {data.diagnostic_workup.map((s, i) => <li key={i} className="mb-1">{s.replace(/^\d+\.\s/,'')}</li>)}
        </ol>
      </Section>

      {/* Treatment */}
      <Section title="Treatment Summary" color={ACCENT3}>
        <ol className="small ps-3">
          {data.treatment_summary.map((s, i) => <li key={i} className="mb-1">{s.replace(/^\d+\.\s/,'')}</li>)}
        </ol>
      </Section>

      {/* DDx */}
      <Section title="Differential Diagnosis vs BBS10 / BBS6 / Alström" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-light"><tr><th>Disease</th><th>Key Discriminator</th></tr></thead>
            <tbody>
              {data.ddx_table.map(r => (
                <tr key={r.disease}><td className="fw-bold">{r.disease}</td><td>{r.key_difference}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab() {
  const { data, err } = useFetch('/api/bbs12/definitions');
  if (err)  return <div className="alert alert-danger">Error: {err}</div>;
  if (!data) return <div className="text-muted p-4">Loading…</div>;

  return (
    <div>
      {/* Gene card */}
      <Section title="Gene Card — BBS12" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <tbody>
              {Object.entries(data.gene_card).map(([k, v]) => (
                <tr key={k}><td className="fw-bold text-nowrap" style={{ color: ACCENT }}>{k.replace(/_/g,' ')}</td><td>{v}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Disease card */}
      <Section title="Disease Card — Bardet-Biedl Syndrome Type 12 (#209900)" color={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <tbody>
              {Object.entries(data.disease_card).map(([k, v]) => (
                <tr key={k}><td className="fw-bold text-nowrap" style={{ color: ACCENT2 }}>{k.replace(/_/g,' ')}</td><td>{v}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Mechanism glossary */}
      <Section title="Mechanism Glossary" color={ACCENT3}>
        {data.mechanism_glossary.map(g => (
          <div key={g.term} className="mb-3 p-2 rounded" style={{ background: ACCENT3 + '0a', border: `1px solid ${ACCENT3}30` }}>
            <div className="fw-bold small mb-1" style={{ color: ACCENT3 }}>{g.term}</div>
            <div className="small">{g.definition}</div>
          </div>
        ))}
      </Section>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function BBS12Page() {
  const [tab, setTab] = useState(0);

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3">
        <div className="d-flex align-items-center gap-2 mb-1">
          <span style={{ fontSize: '1.5em' }}>🧬</span>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
            BBS12 — Bardet-Biedl Syndrome Type 12
          </h4>
          <span className="badge ms-2" style={{ background: ACCENT, fontSize: '0.75em' }}>BCC γ-Ring</span>
          <span className="badge ms-1" style={{ background: ACCENT2, fontSize: '0.75em' }}>4q27</span>
          <span className="badge ms-1" style={{ background: ACCENT3, fontSize: '0.75em' }}>~5–8% BBS</span>
        </div>
        <div className="text-muted small">
          BBS12 · OMIM *610188 · 729 aa · Chr 4q27 · BCC (BBSome Chaperonin Complex) γ-Ring Subunit ·
          MKKS(α)·BBS10(β)·BBS12(γ) trimer · co-folds BBS2/BBS7 WD40 barrels (BBSome Step 0) ·
          AR biallelic LOF · Enriched in North African/Maghrebi and MENA consanguineous families ·
          40-patient cohort (seed 353)
        </div>
        <div className="mt-1">
          <Badge text="BCC γ-Ring LOF" color={ACCENT} />
          <Badge text="IF: BBS12↓↓ + MKKS↓ + BBS10↓ + BBS2↓↓" color={ACCENT2} />
          <Badge text="BBS10 most common 3rd allele" color={ACCENT3} />
          <Badge text="North African founder: Arg140Cys" color={ACCENT6} />
          <Badge text="AR · #209900" color={ACCENT5} />
        </div>
      </div>

      {/* BCC trimer alert */}
      <div className="alert mb-3 small" style={{ background: ACCENT + '12', border: `1px solid ${ACCENT}40`, borderRadius: 8 }}>
        <strong>BCC Trimer (MKKS · BBS10 · BBS12):</strong>{' '}
        BBS12 completes the BCC heterotrimer as the γ-ring structural subunit. Its N-terminal/apical domain contacts
        BBS10 β-ring (Arg140 interface bridge); its equatorial domain docks to MKKS α-subunit equatorial surface
        (Arg445 salt-bridge). LOF → BCC γ-ring absent → MKKS and BBS10 present as REDUCED (partial degradation
        of monomers) → BCC cannot co-fold BBS2/BBS7 → BBSome Step 0 fails → GPCR cargo mis-trafficking → full BBS.
        <strong> IF almost identical to BBS10 LOF</strong>; gene panel is the definitive discriminator.
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT, fontWeight: 600 } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab />}
      {tab === 1 && <BreakdownTab />}
      {tab === 2 && <VariantsTab />}
      {tab === 3 && <DefinitionsTab />}

      <div className="mt-4 text-muted small">
        <Link href="/" className="me-2">← Home</Link>
        <span>BBS12 · OMIM *610188 · 40 patients · seed 353 · BCC γ-Ring Subunit · #209900</span>
      </div>
    </div>
  );
}
