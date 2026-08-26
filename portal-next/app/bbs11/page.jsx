'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Variants & Diagnostics', 'Definitions'];

// BBS11 colour scheme — teal/jade (E3 ligase unique; NHS-intact; LGMD2H allele overlap)
const ACCENT  = '#004d40';   // deep teal — TRIM32 E3 ligase; unique non-structural BBS gene
const ACCENT2 = '#1b5e20';   // forest green — intact BBSome; downstream ubiquitin mechanism
const ACCENT3 = '#bf360c';   // burnt sienna — polydactyly; congenital
const ACCENT4 = '#880e4f';   // dark rose — rod-cone dystrophy; retinal
const ACCENT5 = '#01579b';   // dark blue — renal anomaly
const ACCENT6 = '#4a148c';   // deep purple — LGMD2H myopathy allele; muscle overlap
const ACCENT7 = '#4e342e';   // dark brown — cognitive/LD
const ACCENT8 = '#558b2f';   // dark olive — obesity; LepR indirect pathway

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
  const { data, err } = useFetch('/api/bbs11/overview');
  if (err)  return <div className="alert alert-danger">API error: {err}</div>;
  if (!data) return <div className="text-muted">Loading…</div>;

  return (
    <div>
      <Alert color={ACCENT}>
        <strong>BBS11 / TRIM32 — Sole E3 Ubiquitin Ligase in the BBS Spectrum.</strong>{' '}
        TRIM32 is the <em>only</em> non-structural BBS gene: it is an E3 ubiquitin ligase
        (RING·B-box·coiled-coil + 6 NHL repeats) that ubiquitinates <strong>HDAC6</strong>,
        Dishevelled, PIMT, actin, and c-Myc. HDAC6 deacetylates ciliary α-tubulin — TRIM32
        LOF stabilises HDAC6 → ciliary tubulin hypoacetylation → impaired IFT without absent BBSome.
      </Alert>
      <Alert color={ACCENT2}>
        <strong>Intact BBSome — Pathognomonic IF Fingerprint.</strong>{' '}
        ALL BBSome subunits are NORMAL by immunofluorescence in BBS11 LOF (BBS1, BBS2, BBS4,
        BBS5, BBS7, BBS8, BBS9, MKKS, BBS10, BBS12 all present). Only TRIM32 IF is ABSENT.
        This is the <em>only</em> BBS gene where a normal BBSome IF does <em>not</em> exclude BBS.
        Always request TRIM32 IF if BBSome is intact but clinical BBS is suspected.
      </Alert>
      <Alert color={ACCENT6}>
        <strong>Allele-Specific LGMD2H Overlap.</strong>{' '}
        RING/B-box linker alleles (p.Pro130Ser; p.Asp487Asn) cause dual BBS + LGMD R11/LGMD2H
        (limb-girdle muscular dystrophy) — proximal weakness, CK 2–5×, dystrophic muscle biopsy.
        NHL repeat alleles (p.Ala502Val; p.Arg394Trp) cause BBS without myopathy.
        Always measure CK and assess proximal strength at BBS11 diagnosis.
      </Alert>

      <div className="row g-2 mb-4">
        <KPI label="Gene"           value="TRIM32"       color={ACCENT}  />
        <KPI label="Chr"            value="9q33.1"       color={ACCENT2} />
        <KPI label="Protein"        value="653 aa"       color={ACCENT3} />
        <KPI label="Function"       value="E3 Ligase"    color={ACCENT}  />
        <KPI label="BBSome"         value="INTACT"       color={ACCENT2} />
        <KPI label="BBS Freq"       value="<2%"          color={ACCENT6} />
        <KPI label="OMIM Gene"      value="*602290"      color={ACCENT4} />
        <KPI label="OMIM Disease"   value="#209900"      color={ACCENT4} />
        <KPI label="Cohort N"       value={_COHORT_SIZE} color={ACCENT5} />
        <KPI label="Tri-Allelic"    value="~4%"          color={ACCENT6} />
        <KPI label="LGMD2H Alleles" value={`${data.myopathy_pct}%`} color={ACCENT6} />
        <KPI label="CHD"            value="~3%"          color={ACCENT3} />
      </div>

      <Section title="TRIM32 Domain Map" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead><tr style={{ background: ACCENT + '22' }}>
              <th>Domain</th><th>Residues</th><th>Function</th><th>BBS vs LGMD2H allele class</th>
            </tr></thead>
            <tbody>
              <tr><td><strong>RING finger</strong></td><td>aa 1–80</td><td>Catalytic E3 ligase (Cys3-His-Cys3 zinc coordination)</td><td>RING mutations → dual BBS+LGMD2H</td></tr>
              <tr><td><strong>B-box zinc-finger</strong></td><td>aa 81–140</td><td>Substrate recognition assistance; Pro130 in RING/B-box linker</td><td>Linker mutations (Pro130Ser) → dual BBS+LGMD2H (LGMD2H founder)</td></tr>
              <tr><td><strong>Coiled-coil</strong></td><td>aa 141–280</td><td>TRIM32 homodimerisation; Arg394 at CC/NHL junction</td><td>Junction mutations → BBS-specific</td></tr>
              <tr><td><strong>NHL repeats 1–3</strong></td><td>aa 281–430</td><td>β-propeller substrate platform (blades 1–3); HDAC6 binding surface</td><td>NHL-1–3 mutations → BBS-specific (no myopathy)</td></tr>
              <tr><td><strong>NHL repeats 4–5</strong></td><td>aa 431–530</td><td>β-propeller blades 4–5; Asp487 (NHL-4/5 junction); Ala502 (NHL-5)</td><td>Asp487Asn → dual BBS+LGMD2H; Ala502Val → BBS-specific</td></tr>
              <tr><td><strong>NHL repeats 5–6 + C-term</strong></td><td>aa 531–653</td><td>β-propeller blades 5–6 completion; Leu619 truncation removes C-term</td><td>Truncating → BBS null (no myopathy)</td></tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Systemic Burden — {_COHORT_SIZE}-Patient Cohort" color={ACCENT3}>
        <div className="row">
          {data.systemic_burden.map(({ feature, n, pct }) => (
            <div key={feature} className="col-md-6">
              <Bar label={feature} value={n} max={_COHORT_SIZE}
                   color={feature.includes('myopath') || feature.includes('LGMD') ? ACCENT6 :
                          feature.includes('polydact') ? ACCENT3 :
                          feature.includes('Obesity') ? ACCENT8 :
                          feature.includes('Retinal') || feature.includes('rod') ? ACCENT4 :
                          feature.includes('Renal') ? ACCENT5 :
                          feature.includes('CHD') ? '#c62828' : ACCENT7} />
            </div>
          ))}
        </div>
      </Section>

      <Section title="Ethnicity Distribution" color={ACCENT2}>
        <div className="row">
          {data.ethnicity_distribution.map(({ ethnicity, n }) => (
            <div key={ethnicity} className="col-md-6">
              <Bar label={ethnicity} value={n} max={_COHORT_SIZE} color={ACCENT2} />
            </div>
          ))}
        </div>
        <div className="small text-muted mt-1">
          European predominance reflects Hutterite/Northern European LGMD2H founder (Pro130Ser) + BBS-specific alleles (Ala502Val).
        </div>
      </Section>

      <Section title="Top Pathogenic Variants" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead><tr style={{ background: ACCENT + '22' }}>
              <th>Variant</th><th>Domain</th><th>Ethnicity</th><th>Allele Class</th>
            </tr></thead>
            <tbody>
              {data.top_variants.map(v => (
                <tr key={v.variant}>
                  <td><code>{v.variant}</code></td>
                  <td>{v.domain}</td>
                  <td>{v.ethnicity}</td>
                  <td>{v.allelic_note
                    ? <Badge text={v.allelic_note} color={ACCENT6} />
                    : <Badge text="BBS-specific" color={ACCENT2} />
                  }</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

// ── Tab: Breakdown ────────────────────────────────────────────────────────────
function BreakdownTab() {
  const { data, err } = useFetch('/api/bbs11/breakdown');
  if (err)  return <div className="alert alert-danger">API error: {err}</div>;
  if (!data) return <div className="text-muted">Loading…</div>;

  const n = data.cohort_n;
  return (
    <div>
      <div className="row">
        <div className="col-md-6">
          <Section title="Systemic Burden" color={ACCENT3}>
            {data.systemic_burden.map(({ feature, n: nn, pct }) => (
              <Bar key={feature} label={`${feature} (${pct}%)`} value={nn} max={n}
                   color={feature.includes('myopath') || feature.includes('LGMD') ? ACCENT6 :
                          feature.includes('polydact') ? ACCENT3 :
                          feature.includes('Obesity') ? ACCENT8 :
                          feature.includes('Retinal') || feature.includes('rod') ? ACCENT4 :
                          feature.includes('Renal') ? ACCENT5 :
                          feature.includes('CHD') ? '#c62828' : ACCENT7} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Ethnicity Distribution" color={ACCENT2}>
            {data.ethnicity_distribution.map(({ ethnicity, n: nn }) => (
              <Bar key={ethnicity} label={ethnicity} value={nn} max={n} color={ACCENT2} />
            ))}
          </Section>
          <Section title="Allele Class Summary" color={ACCENT}>
            {data.allele_class_summary.map(({ label, n: nn }) => (
              <Bar key={label} label={label} value={nn} max={n}
                   color={label.includes('LGMD') ? ACCENT6 : ACCENT} />
            ))}
          </Section>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <Section title="Retinal Disease Stage" color={ACCENT4}>
            {data.retinal_stage_distribution.map(({ label, n: nn }) => (
              <Bar key={label} label={label} value={nn} max={n} color={ACCENT4} />
            ))}
          </Section>
          <Section title="Renal Involvement" color={ACCENT5}>
            {data.renal_distribution.map(({ label, n: nn }) => (
              <Bar key={label} label={label} value={nn} max={n} color={ACCENT5} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Polydactyly Type" color={ACCENT3}>
            {data.polydactyly_distribution.map(({ label, n: nn }) => (
              <Bar key={label} label={label} value={nn} max={n} color={ACCENT3} />
            ))}
          </Section>
          <Section title="Myopathy Features (LGMD2H Alleles)" color={ACCENT6}>
            {data.myopathy_distribution.map(({ label, n: nn }) => (
              <Bar key={label} label={label} value={nn} max={n} color={ACCENT6} />
            ))}
          </Section>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <Section title="Age at Presentation" color={ACCENT2}>
            {data.presentation_distribution.map(({ label, n: nn }) => (
              <Bar key={label} label={label} value={nn} max={n} color={ACCENT2} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Initial Misdiagnosis" color={ACCENT6}>
            {data.misdiagnosis_distribution.map(({ label, n: nn }) => (
              <Bar key={label} label={label} value={nn} max={n} color={ACCENT6} />
            ))}
          </Section>
        </div>
      </div>

      <Section title="Top Variants (by allele count)" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead><tr style={{ background: ACCENT + '22' }}>
              <th>Variant</th><th>Allele Count</th>
            </tr></thead>
            <tbody>
              {data.top_variants.map(({ variant, n: nn }) => (
                <tr key={variant}>
                  <td><code>{variant}</code></td>
                  <td>{nn}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

// ── Tab: Variants & Diagnostics ───────────────────────────────────────────────
function VariantsTab() {
  const { data, err } = useFetch('/api/bbs11/definitions');
  if (err)  return <div className="alert alert-danger">API error: {err}</div>;
  if (!data) return <div className="text-muted">Loading…</div>;

  return (
    <div>
      <Section title="Key Pathogenic Variants" color={ACCENT}>
        {data.key_variants.map(v => (
          <div key={v.variant} className="card mb-3 shadow-sm">
            <div className="card-header py-2" style={{ background: ACCENT + '18' }}>
              <code className="fw-bold">{v.variant}</code>
              <span className="ms-2 small text-muted">{v.domain}</span>
            </div>
            <div className="card-body py-2 small">
              <div><strong>Consequence:</strong> {v.consequence}</div>
              <div className="mt-1"><strong>Ethnicity / frequency:</strong> {v.ethnicity}</div>
            </div>
          </div>
        ))}
      </Section>

      <Section title="Diagnostic Workup — Step-by-Step" color={ACCENT2}>
        <ol className="small">
          {data.diagnostic_workup.map((step, i) => <li key={i} className="mb-1">{step}</li>)}
        </ol>
      </Section>

      <Section title="Differential Diagnosis" color={ACCENT6}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead><tr style={{ background: ACCENT6 + '22' }}>
              <th>Disease</th><th>Key Distinguishing Feature</th>
            </tr></thead>
            <tbody>
              {data.ddx_table.map(row => (
                <tr key={row.disease}>
                  <td className="fw-bold">{row.disease}</td>
                  <td>{row.key_difference}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Treatment Summary" color={ACCENT3}>
        <ol className="small">
          {data.treatment_summary.map((step, i) => <li key={i} className="mb-1">{step}</li>)}
        </ol>
      </Section>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab() {
  const { data, err } = useFetch('/api/bbs11/definitions');
  if (err)  return <div className="alert alert-danger">API error: {err}</div>;
  if (!data) return <div className="text-muted">Loading…</div>;

  return (
    <div>
      <div className="row">
        <div className="col-md-6">
          <Section title="Gene Card — TRIM32" color={ACCENT}>
            <table className="table table-sm table-bordered small">
              <tbody>
                {Object.entries(data.gene_card).map(([k, v]) => (
                  <tr key={k}>
                    <td className="fw-bold text-nowrap" style={{ color: ACCENT }}>{k.replace(/_/g,' ')}</td>
                    <td>{String(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Disease Card — BBS11 / LGMD2H" color={ACCENT4}>
            <table className="table table-sm table-bordered small">
              <tbody>
                {Object.entries(data.disease_card).map(([k, v]) => (
                  <tr key={k}>
                    <td className="fw-bold text-nowrap" style={{ color: ACCENT4 }}>{k.replace(/_/g,' ')}</td>
                    <td>{String(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Section>
        </div>
      </div>

      <Section title="Mechanism Glossary" color={ACCENT2}>
        {data.mechanism_glossary.map(({ term, definition }) => (
          <div key={term} className="mb-3">
            <div className="fw-bold small" style={{ color: ACCENT2 }}>{term}</div>
            <div className="small text-muted">{definition}</div>
          </div>
        ))}
      </Section>
    </div>
  );
}

// ── Root page ─────────────────────────────────────────────────────────────────
export default function BBS11Page() {
  const [tab, setTab] = useState(0);
  const tabContent = [<OverviewTab />, <BreakdownTab />, <VariantsTab />, <DefinitionsTab />];

  return (
    <div className="container-fluid py-3">
      {/* breadcrumb */}
      <nav className="small mb-2">
        <Link href="/" className="text-decoration-none">Home</Link>
        {' / '}
        <Link href="/expert-dashboards" className="text-decoration-none">Expert Dashboards</Link>
        {' / '}
        <span style={{ color: ACCENT }}>BBS11 — TRIM32</span>
      </nav>

      {/* header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', border: `2px solid ${ACCENT}` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: '1.5rem' }}>&#x1f9ec;</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
              BBS11 — Bardet-Biedl Syndrome Type 11 / TRIM32
            </h4>
            <div className="small text-muted">
              <Badge text="E3 Ubiquitin Ligase" color={ACCENT} />
              <Badge text="TRIM32 — *602290" color={ACCENT2} />
              <Badge text="Chr 9q33.1" color={ACCENT3} />
              <Badge text="653 aa" color={ACCENT5} />
              <Badge text="BBSome INTACT" color={ACCENT2} />
              <Badge text="LGMD2H allelic overlap" color={ACCENT6} />
              <Badge text="AR — #209900" color={ACCENT4} />
              <Badge text="<2% BBS" color={ACCENT6} />
            </div>
          </div>
        </div>
        <div className="small mt-2" style={{ color: ACCENT }}>
          <strong>Unique biology:</strong> TRIM32 is the <em>only</em> E3 ubiquitin ligase among all
          BBS-causative proteins. It ubiquitinates HDAC6 (ciliary tubulin deacetylase), Dishevelled
          (Wnt), PIMT, and actin. BBSome is structurally INTACT — all subunits present by IF.
          NHL repeat mutations → BBS only; RING/B-box linker mutations (p.Pro130Ser; p.Asp487Asn) →
          dual BBS + LGMD R11/LGMD2H (limb-girdle muscular dystrophy). Always measure CK. Frequency: &lt;2% of BBS.
        </div>
      </div>

      {/* tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active' : ''}`}
                    style={tab === i ? { color: ACCENT, fontWeight: 700 } : {}}
                    onClick={() => setTab(i)}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      <div>{tabContent[tab]}</div>
    </div>
  );
}
