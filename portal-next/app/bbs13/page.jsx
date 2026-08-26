'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Variants & Diagnostics', 'Definitions'];

// BBS13 colour scheme — forest green / deep plum / teal (TZ scaffold; NPHP-MKS module; Meckel spectrum)
const ACCENT  = '#1b5e20';   // forest green — transition zone scaffolding; NPHP-MKS module
const ACCENT2 = '#4a148c';   // deep plum — allele class spectrum (Meckel↔BBS13↔JBTS28 triad)
const ACCENT3 = '#e65100';   // deep orange — polydactyly
const ACCENT4 = '#880e4f';   // dark rose — rod-cone dystrophy; retinal
const ACCENT5 = '#006064';   // dark teal — nephronophthisis; NPHP renal pattern
const ACCENT6 = '#37474f';   // dark slate — epidemiology; AR inheritance
const ACCENT7 = '#4e342e';   // dark brown — cognitive/LD; liver fibrosis
const ACCENT8 = '#33691e';   // darker olive — obesity; LepR TZ pathway

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

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab() {
  const { data: ov, err: ovErr } = useFetch('/api/bbs13/overview');

  if (ovErr) return <div className="alert alert-danger">Failed to load overview: {ovErr}</div>;
  if (!ov)   return <div className="text-muted p-3">Loading overview…</div>;

  const kc = ov.key_counts || {};

  return (
    <div>
      {/* Unique mechanism banner */}
      <Alert color={ACCENT2}>
        <strong>BBS13 (MKS1) — First BBS Gene NOT in BBSome or BCC.</strong>{' '}
        MKS1 is a <strong>Transition Zone (TZ) scaffold</strong> (NPHP-MKS-JBTS module). The BBSome
        assembles normally — <em>BBS2, BBS4, BBS8, BBS9, MKKS all NORMAL on IF</em>.
        Only MKS1 is absent. GPCR mis-trafficking occurs via <strong>TZ gate leakiness</strong>,
        not BBSome assembly failure. <strong>Allele class controls disease tier:</strong> null/null →
        Meckel-Gruber (lethal); hypomorphic → BBS13; hypomorphic/truncating → BBS13 or JBTS28.
      </Alert>

      {/* Gene card */}
      <div className="card mb-4 shadow-sm border-0">
        <div className="card-body">
          <div className="row g-2 text-center">
            {[
              ['Gene', 'MKS1', ACCENT],
              ['OMIM Gene', '*609883', ACCENT2],
              ['OMIM Disease', '#613464', ACCENT3],
              ['Locus', '17q22', ACCENT4],
              ['Protein', '559 aa', ACCENT5],
              ['Module', 'TZ scaffold (NPHP-MKS-JBTS)', ACCENT6],
              ['BBS freq.', '~1% all BBS', ACCENT7],
              ['Allelic diseases', 'MKS1 / JBTS28 / BBS13', ACCENT8],
            ].map(([label, value, color]) => (
              <KPI key={label} label={label} value={value} color={color} />
            ))}
          </div>
        </div>
      </div>

      {/* Systemic burden KPIs */}
      <Section title={`Cardinal Feature Penetrance — Cohort N=${_COHORT_SIZE}`} color={ACCENT}>
        <div className="row g-2 text-center">
          {[
            ['Polydactyly', `${kc.polydactyly_n} (${kc.polydactyly_pct}%)`, ACCENT3],
            ['Obesity (TZ/LepR)', `${kc.obesity_n} (${kc.obesity_pct}%)`, ACCENT8],
            ['Renal anomaly', `${kc.renal_any_n} (${kc.renal_any_pct}%)`, ACCENT5],
            ['NPHP pattern', `${kc.nphp_n} (${kc.nphp_pct}%)`, ACCENT5],
            ['Hypogonadism', `${kc.hypogonadism_n} (${kc.hypogonadism_pct}%)`, ACCENT6],
            ['Cognitive/LD', `${kc.cognitive_n} (${kc.cognitive_pct}%)`, ACCENT7],
            ['Anosmia', `${kc.anosmia_n} (${kc.anosmia_pct}%)`, ACCENT4],
            ['Liver fibrosis', `${kc.liver_fibrosis_n} (${kc.liver_fibrosis_pct}%)`, ACCENT7],
            ['JBTS28 overlap', `${kc.jbts_overlap_n} (${kc.jbts_overlap_pct}%)`, ACCENT2],
            ['CHD', `${kc.chd_n} (${kc.chd_pct}%)`, ACCENT4],
            ['Retinal end-stage', `${kc.retinal_endstage_n} (${kc.retinal_endstage_pct}%)`, ACCENT4],
            ['Tri-allelic BBS', `${kc.triallelic_n} (${kc.triallelic_pct}%)`, ACCENT2],
          ].map(([label, value, color]) => (
            <KPI key={label} label={label} value={value} color={color} />
          ))}
        </div>
      </Section>

      {/* BBS13 Unique Features */}
      <Section title="BBS13-Unique Diagnostic Features" color={ACCENT2}>
        <div className="row g-3">
          <div className="col-md-6">
            <Alert color={ACCENT}>
              <strong>BBSome IF NORMAL (unique):</strong> BBS2 · BBS4 · BBS8 · BBS9 · MKKS all present
              at normal levels. Only <strong>MKS1 absent</strong> + NPHP4 deranged at TZ.
              This is the opposite of BBS1–12 where BBSome/BCC subunits are absent.
            </Alert>
            <Alert color={ACCENT2}>
              <strong>Allele-class severity triad:</strong> Null/null → Meckel-Gruber (lethal) ·
              Hypomorphic/hypomorphic → BBS13 · Hypomorphic/truncating → BBS13 or JBTS28.
              Classify both alleles as truncating vs hypomorphic — critical for prognosis and family planning.
            </Alert>
          </div>
          <div className="col-md-6">
            <Alert color={ACCENT5}>
              <strong>Renal: NPHP pattern dominant</strong> (tubular atrophy, interstitial fibrosis,
              concentrating defect) — distinct from BBS1–12 (cystic/structural).
              ESRD risk ~{kc.esrd_pct}% ({kc.esrd_n}/{_COHORT_SIZE}). Annual GFR + urine concentrating ability.
            </Alert>
            <Alert color={ACCENT7}>
              <strong>Liver fibrosis</strong> in {kc.liver_fibrosis_pct}% ({kc.liver_fibrosis_n}/{_COHORT_SIZE}) —
              ductal plate malformation / congenital hepatic fibrosis (Meckel overlap feature).
              Unique to BBS13 among all BBS types. Liver USS + fibroscan at diagnosis.
            </Alert>
          </div>
        </div>
      </Section>

      {/* Retinal stages */}
      <Section title="Retinal Stage Distribution (Rod-First ERG — Same Pattern as BBS1–12)" color={ACCENT4}>
        <div className="row g-2">
          {ov.retinal_stage_distribution && Object.entries(ov.retinal_stage_distribution).map(([stage, cnt]) => (
            <div key={stage} className="col-6 col-md-3">
              <div className="card text-center shadow-sm">
                <div className="card-body py-2">
                  <div className="fw-bold fs-5" style={{ color: ACCENT4 }}>{cnt}</div>
                  <div className="small text-muted">{stage}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* Diagnosis age */}
      <Section title="Age at Diagnosis Distribution" color={ACCENT6}>
        <div className="row g-2">
          {ov.dx_age_distribution && Object.entries(ov.dx_age_distribution).map(([age, cnt]) => (
            <div key={age} className="col-6 col-md-3">
              <div className="card text-center shadow-sm">
                <div className="card-body py-2">
                  <div className="fw-bold fs-4" style={{ color: ACCENT6 }}>{cnt}</div>
                  <div className="small text-muted">{age}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* Consanguinity note */}
      <Alert color={ACCENT6}>
        <strong>Consanguinity:</strong> {kc.consanguinity_pct}% ({kc.consanguinity_n}/{_COHORT_SIZE}) — enriched in
        MENA, South Asian, and North African consanguineous families. Pan-ethnic; no single dominant
        founder allele (contrast BBS12 Arg140Cys North African founder).
      </Alert>

      {/* JBTS28 overlap note */}
      {kc.jbts_overlap_n > 0 && (
        <Alert color={ACCENT2}>
          <strong>JBTS28 overlap:</strong> {kc.jbts_overlap_n} patients ({kc.jbts_overlap_pct}%) show Joubert syndrome
          features (molar tooth sign; cerebellar vermis hypoplasia) — allele class: hypomorphic/truncating or
          splice/truncating. MRI brain indicated when allele class is intermediate severity.
        </Alert>
      )}
    </div>
  );
}

// ── Tab: Breakdown ────────────────────────────────────────────────────────────
function BreakdownTab() {
  const { data: bk, err } = useFetch('/api/bbs13/breakdown');
  if (err)  return <div className="alert alert-danger">Failed to load breakdown: {err}</div>;
  if (!bk)  return <div className="text-muted p-3">Loading breakdown…</div>;

  const n = bk.cohort_n || _COHORT_SIZE;

  return (
    <div>
      {/* Systemic feature bars */}
      <Section title="Multi-System Feature Penetrance" color={ACCENT}>
        {(bk.systemic_burden || []).map(({ feature, n: fn, pct }) => (
          <Bar key={feature} label={feature} value={fn} max={n}
               color={
                 feature.includes('Retinal') ? ACCENT4 :
                 feature.includes('Obesity') ? ACCENT8 :
                 feature.includes('Poly') ? ACCENT3 :
                 feature.includes('Renal') || feature.includes('NPHP') || feature.includes('ESRD') ? ACCENT5 :
                 feature.includes('Liver') ? ACCENT7 :
                 feature.includes('Joubert') ? ACCENT2 :
                 ACCENT6
               } />
        ))}
      </Section>

      <div className="row g-4">
        {/* Ethnicity */}
        <div className="col-md-6">
          <Section title="Ethnicity Distribution" color={ACCENT2}>
            {(bk.ethnicity_distribution || []).map(({ ethnicity, n: en }) => (
              <Bar key={ethnicity} label={ethnicity} value={en} max={n} color={ACCENT2} />
            ))}
          </Section>
        </div>

        {/* Allele class */}
        <div className="col-md-6">
          <Section title="Allele Class (BBS13 Severity Spectrum)" color={ACCENT2}>
            {(bk.allele_class_summary || []).map(({ label, n: an }) => (
              <Bar key={label} label={label} value={an} max={n} color={ACCENT2} />
            ))}
          </Section>
        </div>

        {/* Retinal stage */}
        <div className="col-md-6">
          <Section title="Retinal Stage" color={ACCENT4}>
            {(bk.retinal_stage_distribution || []).map(({ label, n: rn }) => (
              <Bar key={label} label={label} value={rn} max={n} color={ACCENT4} />
            ))}
          </Section>
        </div>

        {/* Renal — NPHP pattern */}
        <div className="col-md-6">
          <Section title="Renal Pattern Distribution (NPHP Dominant)" color={ACCENT5}>
            {(bk.renal_distribution || []).map(({ label, n: rn }) => (
              <Bar key={label} label={label} value={rn} max={n} color={ACCENT5} />
            ))}
          </Section>
        </div>

        {/* Polydactyly */}
        <div className="col-md-6">
          <Section title="Polydactyly Type" color={ACCENT3}>
            {(bk.polydactyly_distribution || []).map(({ label, n: pn }) => (
              <Bar key={label} label={label} value={pn} max={n} color={ACCENT3} />
            ))}
          </Section>
        </div>

        {/* Presentation age */}
        <div className="col-md-6">
          <Section title="Age at First Presentation" color={ACCENT6}>
            {(bk.presentation_distribution || []).map(({ label, n: pn }) => (
              <Bar key={label} label={label} value={pn} max={n} color={ACCENT6} />
            ))}
          </Section>
        </div>

        {/* Misdiagnosis */}
        <div className="col-md-6">
          <Section title="Initial Misdiagnosis (BBS13-Specific)" color={ACCENT7}>
            {(bk.misdiagnosis_distribution || []).map(({ label, n: mn }) => (
              <Bar key={label} label={label} value={mn} max={n} color={ACCENT7} />
            ))}
          </Section>
        </div>

        {/* Top variants */}
        <div className="col-md-6">
          <Section title="Top MKS1 Variants (Allele Counts)" color={ACCENT}>
            {(bk.top_variants || []).map(({ variant, n: vn }) => (
              <Bar key={variant} label={variant} value={vn} max={n * 2} color={ACCENT} />
            ))}
          </Section>
        </div>
      </div>
    </div>
  );
}

// ── Tab: Variants & Diagnostics ───────────────────────────────────────────────
function VariantsTab() {
  const { data: df, err } = useFetch('/api/bbs13/definitions');
  if (err)  return <div className="alert alert-danger">Failed to load definitions: {err}</div>;
  if (!df)  return <div className="text-muted p-3">Loading definitions…</div>;

  return (
    <div>
      {/* Key variants */}
      <Section title="Key MKS1 Pathogenic Variants" color={ACCENT}>
        {(df.key_variants || []).map((v, i) => (
          <div key={i} className="card mb-3 shadow-sm border-0">
            <div className="card-body">
              <div className="d-flex align-items-start gap-2 mb-2">
                <Badge text={v.variant} color={ACCENT} />
                <Badge text={v.ethnicity} color={ACCENT2} />
              </div>
              <p className="small mb-1"><strong>Domain:</strong> {v.domain}</p>
              <p className="small mb-0"><strong>Consequence:</strong> {v.consequence}</p>
            </div>
          </div>
        ))}
      </Section>

      {/* Diagnostic workup */}
      <Section title="Diagnostic Workup (BBS13-Specific)" color={ACCENT5}>
        <ol className="ps-3 small">
          {(df.diagnostic_workup || []).map((step, i) => (
            <li key={i} className="mb-2">{step}</li>
          ))}
        </ol>
      </Section>

      {/* DDx table */}
      <Section title="Differential Diagnosis" color={ACCENT7}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ background: ACCENT7 + '22' }}>
              <tr>
                <th style={{ color: ACCENT7 }}>Disease</th>
                <th style={{ color: ACCENT7 }}>Key Distinguishing Feature</th>
              </tr>
            </thead>
            <tbody>
              {(df.ddx_table || []).map((row, i) => (
                <tr key={i}>
                  <td className="fw-semibold text-nowrap">{row.disease}</td>
                  <td>{row.key_difference}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Treatment */}
      <Section title="Management Summary" color={ACCENT8}>
        <ol className="ps-3 small">
          {(df.treatment_summary || []).map((step, i) => (
            <li key={i} className="mb-2">{step}</li>
          ))}
        </ol>
      </Section>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab() {
  const { data: df, err } = useFetch('/api/bbs13/definitions');
  if (err)  return <div className="alert alert-danger">Failed to load definitions: {err}</div>;
  if (!df)  return <div className="text-muted p-3">Loading definitions…</div>;

  return (
    <div>
      {/* Gene card */}
      <Section title="MKS1 Gene Card" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <tbody>
              {df.gene_card && Object.entries(df.gene_card).map(([k, v]) => (
                <tr key={k}>
                  <td className="fw-semibold text-nowrap" style={{ color: ACCENT, width: '25%' }}>{k.replace(/_/g, ' ')}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Disease card */}
      <Section title="BBS13 Disease Card" color={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <tbody>
              {df.disease_card && Object.entries(df.disease_card).map(([k, v]) => (
                <tr key={k}>
                  <td className="fw-semibold text-nowrap" style={{ color: ACCENT4, width: '25%' }}>{k.replace(/_/g, ' ')}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Mechanism glossary */}
      <Section title="Mechanism Glossary (TZ Module &amp; BBS13 Biology)" color={ACCENT2}>
        {(df.mechanism_glossary || []).map((entry, i) => (
          <div key={i} className="card mb-3 shadow-sm border-0">
            <div className="card-header fw-bold small" style={{ background: ACCENT2 + '18', color: ACCENT2 }}>
              {entry.term}
            </div>
            <div className="card-body small">
              {entry.definition}
            </div>
          </div>
        ))}
      </Section>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function BBS13Page() {
  const [tab, setTab] = useState(0);

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3">
        <div>
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            🧬 BBS13 — Bardet-Biedl Syndrome Type 13
          </h4>
          <p className="text-muted small mb-0">
            MKS1 · Transition Zone Scaffold · NPHP-MKS-JBTS Module · Chr 17q22 · OMIM *609883 / #613464
            · <em>First BBS gene NOT in BBSome or BCC — BBSome IF normal; TZ gate leaky</em>
          </p>
        </div>
        <Link href="/" className="btn btn-outline-secondary btn-sm ms-auto">← Portal</Link>
      </div>

      {/* Nav tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottom: `2px solid ${ACCENT}` } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* Tab content */}
      {tab === 0 && <OverviewTab />}
      {tab === 1 && <BreakdownTab />}
      {tab === 2 && <VariantsTab />}
      {tab === 3 && <DefinitionsTab />}
    </div>
  );
}
