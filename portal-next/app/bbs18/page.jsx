'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Treatment & Diagnostics', 'Definitions'];

// BBS18 colour scheme — deep teal (BBIP1/BBSome assembly); amber (polydactyly/limb bud);
// forest green (BBSome assembly / BBS9 interface); dark rose (rod-cone degeneration);
// navy (epidemiology / MENA enrichment); dark slate (locus/gene); dark plum (cognitive/LD);
// rust (obesity/LepR trafficking)
const ACCENT  = '#004d40';   // deep teal — BBIP1; BBSome core assembly; CC-domain bridge
const ACCENT2 = '#e65100';   // deep amber — polydactyly; limb bud; BBS assembly cascade
const ACCENT3 = '#1b5e20';   // forest green — BBSome octamer; BBS9/BBS2/BBS7 interface
const ACCENT4 = '#880e4f';   // dark rose — rod-cone degeneration; retinal
const ACCENT5 = '#0d47a1';   // navy — MENA/consanguineous enrichment; epidemiology
const ACCENT6 = '#37474f';   // dark slate — locus 10q25.2; gene/protein annotation
const ACCENT7 = '#4a148c';   // dark plum — cognitive/LD
const ACCENT8 = '#bf360c';   // rust — obesity; LepR ciliary trafficking failure

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
    <div className="mb-1">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{value} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

// ── Overview Tab ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const k = data.key_counts || {};
  const burden = data.systemic_burden || [];
  const maxN = _COHORT_SIZE;

  return (
    <div>
      <Alert color={ACCENT}>
        <strong>BBS18 — BBIP1 (BBSome-Interacting Protein 1)</strong><br />
        BBIP1 is the <strong>8th integral subunit</strong> of the BBSome octamer, acting as the
        coiled-coil bridge between the BBS9 scaffold spine and the BBS2/BBS7 lid.
        LOF abolishes BBSome core assembly — all BBSome subunits (BBS2/4/5/7/8/9)
        are absent from cilia. Cilia form and are full length; only BBSome cargo
        trafficking fails. <Badge text="10q25.2" color={ACCENT6} />
        <Badge text="OMIM *613605" color={ACCENT} />
        <Badge text="#615994" color={ACCENT} />
      </Alert>

      <Alert color={ACCENT5}>
        <strong>Epidemiology:</strong> &lt;1% of all BBS — rarest confirmed BBS type (~10–20 families worldwide 2026).
        MENA/consanguineous enrichment (first report: Leu150del in Turkish consanguineous family, Scheidecker 2014).
        Tri-allelic BBS ~2%.
      </Alert>

      <div className="row g-2 mb-3">
        <KPI label="Cohort N" value={data.cohort_n} color={ACCENT} />
        <KPI label="Polydactyly" value={`${k.polydactyly_pct}%`} color={ACCENT2} />
        <KPI label="Obesity" value={`${k.obesity_pct}%`} color={ACCENT8} />
        <KPI label="Cognitive/LD" value={`${k.cognitive_pct}%`} color={ACCENT7} />
        <KPI label="Hypogonadism" value={`${k.hypogonadism_pct}%`} color={ACCENT3} />
        <KPI label="Anosmia" value={`${k.anosmia_pct}%`} color={ACCENT4} />
        <KPI label="Renal (any)" value={`${k.renal_any_pct}%`} color={ACCENT5} />
        <KPI label="CHD" value={`${k.chd_pct}%`} color={ACCENT6} />
        <KPI label="Tri-allelic" value={`${data.triallelic_pct}%`} color={ACCENT} />
        <KPI label="Worldwide Families" value={data.worldwide_families} color={ACCENT5} />
      </div>

      <Section title="Systemic Burden (Educational Cohort N=40)" color={ACCENT}>
        {burden.map(([feat, n, pct]) => (
          <Bar key={feat} label={feat} value={n} max={maxN} color={ACCENT} />
        ))}
      </Section>

      <Section title="Retinal Stage Distribution" color={ACCENT4}>
        {Object.entries(data.retinal_stage_distribution || {}).map(([stage, n]) => (
          <Bar key={stage} label={stage} value={n} max={maxN} color={ACCENT4} />
        ))}
      </Section>

      <Section title="Age at Diagnosis" color={ACCENT6}>
        {Object.entries(data.dx_age_distribution || {}).map(([bucket, n]) => (
          <Bar key={bucket} label={bucket} value={n} max={maxN} color={ACCENT6} />
        ))}
      </Section>

      <Alert color={ACCENT3}>
        <strong>IF Fingerprint — distinguishes BBS18 (assembly failure):</strong><br />
        BBSome subunits <strong>BBS2/4/5/7/8/9 ALL ABSENT</strong> from cilia (octamer collapse) ·
        BBIP1 absent · MKKS/BCC <strong>NORMAL</strong> (individual subunits fold, octamer does not form) ·
        MKS1/TZ <strong>NORMAL</strong> · Cilia <strong>FULL LENGTH</strong> present ·
        IFT <strong>NORMAL</strong>.
      </Alert>

      <Alert color={ACCENT2}>
        <strong>Mechanistic tier (BBSome assembly cascade):</strong><br />
        BBS18 = <em>assembly failure</em> (upstream) ·
        BBS17 = entry failure (BBSome assembles but cannot enter cilia) ·
        BBS15 = exit failure (BBSome assembles, enters, tip-trapped).
        All three produce absent ciliary BBSome by different mechanisms.
      </Alert>
    </div>
  );
}

// ── Breakdown Tab ─────────────────────────────────────────────────────────────
function BreakdownTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  const maxN = _COHORT_SIZE;

  return (
    <div>
      <Section title="Systemic Burden Detail" color={ACCENT}>
        {(data.systemic_burden || []).map(row => (
          <Bar key={row.feature} label={`${row.feature} (n=${row.n})`} value={row.n} max={maxN} color={ACCENT} />
        ))}
      </Section>

      <Section title="Ethnicity Distribution" color={ACCENT5}>
        {(data.ethnicity_distribution || []).map(row => (
          <Bar key={row.ethnicity} label={row.ethnicity} value={row.n} max={maxN} color={ACCENT5} />
        ))}
      </Section>

      <Section title="Allele Class" color={ACCENT3}>
        {(data.allele_class_summary || []).map(row => (
          <Bar key={row.label} label={row.label} value={row.n} max={maxN} color={ACCENT3} />
        ))}
      </Section>

      <Section title="Retinal Stage" color={ACCENT4}>
        {(data.retinal_stage_distribution || []).map(row => (
          <Bar key={row.label} label={row.label} value={row.n} max={maxN} color={ACCENT4} />
        ))}
      </Section>

      <Section title="Renal Pattern" color={ACCENT5}>
        {(data.renal_distribution || []).map(row => (
          <Bar key={row.label} label={row.label} value={row.n} max={maxN} color={ACCENT5} />
        ))}
        <small className="text-muted">Structural/cystic ~30% (NOT NPHP-dominant — contrast BBS13/14/16)</small>
      </Section>

      <Section title="Polydactyly Distribution" color={ACCENT2}>
        {(data.polydactyly_distribution || []).map(row => (
          <Bar key={row.label} label={row.label} value={row.n} max={maxN} color={ACCENT2} />
        ))}
      </Section>

      <Section title="Presentation Mode" color={ACCENT6}>
        {(data.presentation_distribution || []).map(row => (
          <Bar key={row.label} label={row.label} value={row.n} max={maxN} color={ACCENT6} />
        ))}
      </Section>

      <Section title="Prior Misdiagnosis (before BBS panel)" color={ACCENT8}>
        {(data.misdiagnosis_distribution || []).map(row => (
          <Bar key={row.label} label={row.label} value={row.n} max={maxN} color={ACCENT8} />
        ))}
      </Section>

      <Section title="Top BBIP1 Variants (variant_1 slot)" color={ACCENT}>
        {(data.top_variants || []).map(row => (
          <div key={row.variant} className="d-flex justify-content-between small border-bottom py-1">
            <span className="font-monospace">{row.variant}</span>
            <span className="fw-bold text-muted">n={row.n}</span>
          </div>
        ))}
      </Section>
    </div>
  );
}

// ── Treatment & Diagnostics Tab ───────────────────────────────────────────────
function TreatmentTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <div>
      <Alert color={ACCENT}>
        <strong>BBS18 Diagnostic Strategy</strong><br />
        Extended BBS 24-gene panel required — BBIP1 must be included. Older smaller panels frequently
        miss BBS18. IF confirms BBSome assembly failure (BBS2/BBS9 absent from cilia, cilia present).
        BBSome IP assay (BBS9 pull-down) distinguishes assembly failure (BBS18) from entry/exit failure (BBS17/BBS15).
      </Alert>

      <Section title="Diagnostic Workup" color={ACCENT}>
        {(data.diagnostic_workup || []).map((step, i) => (
          <div key={i} className="mb-2 small border-start ps-2" style={{ borderColor: ACCENT }}>
            {step}
          </div>
        ))}
      </Section>

      <Section title="Treatment Summary" color={ACCENT3}>
        {(data.treatment_summary || []).map((step, i) => (
          <div key={i} className="mb-2 small border-start ps-2" style={{ borderColor: ACCENT3 }}>
            {step}
          </div>
        ))}
      </Section>
    </div>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  const gc  = data.gene_card || {};
  const dc  = data.disease_card || {};
  const kvs = data.key_variants || [];

  return (
    <div>
      <Section title="Gene Card — BBIP1" color={ACCENT}>
        <table className="table table-sm table-bordered small">
          <tbody>
            {Object.entries(gc).map(([k, v]) => (
              <tr key={k}>
                <td className="fw-bold text-nowrap" style={{ width: '30%', color: ACCENT }}>{k}</td>
                <td>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section title="Disease Card — BBS18" color={ACCENT4}>
        <table className="table table-sm table-bordered small">
          <tbody>
            {Object.entries(dc).map(([k, v]) => (
              <tr key={k}>
                <td className="fw-bold text-nowrap" style={{ width: '30%', color: ACCENT4 }}>{k}</td>
                <td>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section title="Key BBIP1 Variants" color={ACCENT3}>
        {kvs.map((v, i) => (
          <div key={i} className="card mb-2 shadow-sm">
            <div className="card-body py-2">
              <div className="fw-bold font-monospace small" style={{ color: ACCENT3 }}>{v.variant}</div>
              <div className="small text-muted"><strong>Domain:</strong> {v.domain}</div>
              <div className="small"><strong>Consequence:</strong> {v.consequence}</div>
              <div className="small text-muted"><strong>Ethnicity:</strong> {v.ethnicity}</div>
            </div>
          </div>
        ))}
      </Section>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function BBS18Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/bbs18/overview`).then(r => r.json()),
      fetch(`${API}/api/bbs18/breakdown`).then(r => r.json()),
      fetch(`${API}/api/bbs18/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2 flex-wrap">
        <Link href="/" className="btn btn-sm btn-outline-secondary">← Home</Link>
        <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
          🧬 BBS18 — Bardet-Biedl Syndrome Type 18 (BBIP1)
        </h4>
        <span className="badge" style={{ background: ACCENT }}>BBSome Assembly Failure</span>
        <span className="badge" style={{ background: ACCENT5 }}>10q25.2</span>
        <span className="badge bg-secondary">AR</span>
        <span className="badge" style={{ background: ACCENT2 }}>8th BBSome Subunit</span>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <BreakdownTab data={breakdown} />}
      {tab === 2 && <TreatmentTab data={definitions} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
