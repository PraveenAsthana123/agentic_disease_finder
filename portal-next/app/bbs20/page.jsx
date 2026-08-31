'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Treatment & Diagnostics', 'Definitions'];

// BBS20 colour scheme — forest green (WDPCP/Fritz PCP effector; ciliogenesis scaffold);
// deep orange (PCP-enriched polydactyly; limb bud BB misdocking);
// teal (BBSome docking at TZ; ciliary gate);
// dark rose (rod-cone degeneration; retinal);
// dark navy (MENA enrichment; consanguinity; epidemiology);
// dark slate (locus 2p15; gene/protein annotation);
// indigo (CPLANE complex; INTU/FUZ interaction);
// rust (obesity; LepR-BBSome retrograde IFT failure)
const ACCENT  = '#1b5e20';   // forest green — WDPCP; Fritz; CPLANE↔BBSome scaffold
const ACCENT2 = '#e65100';   // deep orange — PCP-enriched polydactyly; limb bud
const ACCENT3 = '#00695c';   // dark teal — BBSome TZ docking; ciliary gate
const ACCENT4 = '#880e4f';   // dark rose — rod-cone degeneration; retinal
const ACCENT5 = '#0d47a1';   // dark navy — MENA enrichment; epidemiology
const ACCENT6 = '#37474f';   // dark slate — locus 2p15; gene/protein annotation
const ACCENT7 = '#311b92';   // deep indigo — CPLANE complex; INTU/FUZ contact
const ACCENT8 = '#bf360c';   // rust — obesity; LepR-BBSome trafficking failure

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
        <span>{label}</span>
        <span className="fw-bold">{value} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" role="progressbar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

// ── Overview Tab ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  const kc = data.key_counts || {};

  return (
    <div>
      <Alert color={ACCENT}>
        <strong>WDPCP/Fritz — CPLANE↔BBSome Bridge at Basal Body/TZ.</strong>{' '}
        WDPCP bridges the CPLANE complex (INTURNED + FUZZY) to the BBSome (BBS1 + BBS9) at the
        basal body/transition zone. LOF → INTU ABSENT at BB (pathognomonic IF fingerprint unique
        to BBS20) + BBSome REDUCED in cilia + PCP-enriched polydactyly (~{kc.polydactyly_pct}%).
        Single disease tier — no NPHP/MKS/JBTS allelic variant.
      </Alert>

      <Alert color={ACCENT7}>
        <strong>Diagnostic KEY — INTU Absent at Basal Body.</strong>{' '}
        WDPCP recruits INTURNED (INTU) to the BB; LOF removes INTU from BB in 100% of BBS20 cases.
        Anti-INTU IF: ABSENT at BB — seen in BBS20 only, not BBS17/18/19 or any other BBS type.
        Confirms BBS20 on IF before sequencing result is returned.
      </Alert>

      <Alert color={ACCENT2}>
        <strong>PCP Pathway — Polydactyly Enriched (~{kc.polydactyly_pct}%).</strong>{' '}
        WDPCP is a planar cell polarity effector; BB misdocking geometry in limb bud disrupts
        GLI3R/FL ratio → post-axial polydactyly enriched vs average BBS (~45%).
        Similar mechanism to CPLANE1/JBTS33 but BBS20 lacks MTS / cerebellar features.
      </Alert>

      <div className="row mb-3">
        <KPI label="Cohort (n)" value={data.cohort_n} color={ACCENT} />
        <KPI label="Post-axial Polydactyly" value={`${kc.polydactyly_pct}%`} color={ACCENT2} />
        <KPI label="Obesity" value={`${kc.obesity_pct}%`} color={ACCENT8} />
        <KPI label="Renal (any)" value={`${kc.renal_any_pct}%`} color={ACCENT3} />
        <KPI label="Cognitive / LD" value={`${kc.cognitive_pct}%`} color={ACCENT7} />
        <KPI label="Hypogonadism" value={`${kc.hypogonadism_pct}%`} color={ACCENT4} />
        <KPI label="Anosmia" value={`${kc.anosmia_pct}%`} color={ACCENT6} />
        <KPI label="CHD" value={`${kc.chd_pct}%`} color={ACCENT5} />
        <KPI label="Misdiagnosed (pre-WES)" value={`${kc.misdiagnosis_pct}%`} color={ACCENT8} />
        <KPI label="INTU Absent at BB" value="100%" color={ACCENT7} />
        <KPI label="Gene" value="WDPCP" color={ACCENT} />
        <KPI label="Locus" value={data.locus} color={ACCENT6} />
      </div>

      <Section title="Systemic Burden" color={ACCENT}>
        {(data.systemic_burden || []).map(([feat, n, pct]) => (
          <Bar key={feat} label={feat} value={n} max={data.cohort_n} color={ACCENT} />
        ))}
      </Section>

      <div className="row">
        <div className="col-md-6">
          <Section title="Retinal Stage Distribution" color={ACCENT4}>
            {Object.entries(data.retinal_stage_distribution || {}).map(([k, v]) => (
              <Bar key={k} label={k} value={v} max={data.cohort_n} color={ACCENT4} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Age at Diagnosis" color={ACCENT5}>
            {Object.entries(data.dx_age_distribution || {}).map(([k, v]) => (
              <Bar key={k} label={k} value={v} max={data.cohort_n} color={ACCENT5} />
            ))}
          </Section>
        </div>
      </div>

      <Section title="Gene & Disease Summary" color={ACCENT6}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <tbody>
              <tr><td className="fw-bold">Gene</td><td>{data.gene} ({data.alias})</td></tr>
              <tr><td className="fw-bold">Protein</td><td>{data.protein} ({data.protein_size_aa} aa)</td></tr>
              <tr><td className="fw-bold">Locus</td><td>{data.locus}</td></tr>
              <tr><td className="fw-bold">OMIM Gene</td><td>{data.omim_gene}</td></tr>
              <tr><td className="fw-bold">OMIM Disease</td><td>{data.omim_disease}</td></tr>
              <tr><td className="fw-bold">Inheritance</td><td>{data.inheritance}</td></tr>
              <tr><td className="fw-bold">BBS Frequency</td><td>{data.bbs_frequency_pct}</td></tr>
              <tr><td className="fw-bold">Worldwide Families</td><td>{data.worldwide_families}</td></tr>
              <tr><td className="fw-bold">Disease Tier</td><td>{data.disease_tier}</td></tr>
              <tr><td className="fw-bold">PCP Enrichment</td><td>{data.pcp_enrichment}</td></tr>
              <tr><td className="fw-bold">Tri-allelic BBS</td><td>~{data.triallelic_pct}%</td></tr>
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

// ── Breakdown Tab ─────────────────────────────────────────────────────────────
function BreakdownTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <div>
      <div className="row">
        <div className="col-md-6">
          <Section title="Systemic Burden" color={ACCENT}>
            {(data.systemic_burden || []).map(b => (
              <Bar key={b.feature} label={b.feature} value={b.n} max={_COHORT_SIZE} color={ACCENT} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Ethnicity Distribution" color={ACCENT5}>
            {(data.ethnicity_distribution || []).map(e => (
              <Bar key={e.ethnicity} label={e.ethnicity} value={e.n} max={_COHORT_SIZE} color={ACCENT5} />
            ))}
          </Section>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <Section title="Allele Class" color={ACCENT6}>
            {(data.allele_class_summary || []).map(a => (
              <Bar key={a.label} label={a.label} value={a.n} max={_COHORT_SIZE} color={ACCENT6} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Polydactyly Distribution (PCP-enriched)" color={ACCENT2}>
            {(data.polydactyly_distribution || []).map(p => (
              <Bar key={p.label} label={p.label} value={p.n} max={_COHORT_SIZE} color={ACCENT2} />
            ))}
          </Section>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <Section title="Retinal Stage" color={ACCENT4}>
            {(data.retinal_stage_distribution || []).map(r => (
              <Bar key={r.label} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT4} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Renal Pattern" color={ACCENT3}>
            {(data.renal_distribution || []).map(r => (
              <Bar key={r.label} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT3} />
            ))}
          </Section>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <Section title="First Presentation" color={ACCENT7}>
            {(data.presentation_distribution || []).map(p => (
              <Bar key={p.label} label={p.label} value={p.n} max={_COHORT_SIZE} color={ACCENT7} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Misdiagnosis Patterns" color={ACCENT8}>
            {(data.misdiagnosis_distribution || []).map(m => (
              <Bar key={m.label} label={m.label} value={m.n} max={_COHORT_SIZE} color={ACCENT8} />
            ))}
          </Section>
        </div>
      </div>

      <Section title="Top WDPCP Variants" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead><tr><th>Variant</th><th>n</th><th>%</th></tr></thead>
            <tbody>
              {(data.top_variants || []).map(v => (
                <tr key={v.variant}>
                  <td><code>{v.variant}</code></td>
                  <td>{v.n}</td>
                  <td>{Math.round(v.n / _COHORT_SIZE * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

// ── Treatment Tab ─────────────────────────────────────────────────────────────
function TreatmentTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ts = data.treatment_summary || [];
  const dw = data.diagnostic_workup || [];

  return (
    <div>
      <Alert color={ACCENT7}>
        <strong>IF Fingerprint for BBS20 Confirmation.</strong>{' '}
        WDPCP ABSENT + INTU ABSENT at BB (unique to BBS20) + BBSome REDUCED in cilia +
        IFT88 NORMAL (no tip accumulation) + cilia SHORTENED/misoriented. If IFT88 shows
        tip accumulation → BBS19 (IFT172); if BBSome fully ABSENT → BBS18 (BBIP1);
        if Smo entry FAILS on SAG stimulation → BBS17 (LZTFL1).
      </Alert>

      <Section title="Diagnostic Workup" color={ACCENT}>
        <ol className="small mb-0">
          {dw.map((step, i) => <li key={i} className="mb-2">{step}</li>)}
        </ol>
      </Section>

      <Section title="Treatment Summary" color={ACCENT3}>
        <ol className="small mb-0">
          {ts.map((step, i) => <li key={i} className="mb-2">{step}</li>)}
        </ol>
      </Section>
    </div>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const gc = data.gene_card || {};
  const dc = data.disease_card || {};
  const kv = data.key_variants || [];

  return (
    <div>
      <Section title="Gene Card — WDPCP (Fritz)" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <tbody>
              {Object.entries(gc).map(([k, v]) => (
                <tr key={k}><td className="fw-bold" style={{ width: '30%' }}>{k}</td><td>{v}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Disease Card — BBS20" color={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <tbody>
              {Object.entries(dc).map(([k, v]) => (
                <tr key={k}><td className="fw-bold" style={{ width: '30%' }}>{k}</td><td>{v}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Key WDPCP Variants" color={ACCENT6}>
        {kv.map((v) => (
          <div key={v.variant} className="card mb-2 shadow-sm">
            <div className="card-body py-2 px-3">
              <div className="d-flex align-items-center mb-1">
                <code className="fw-bold me-2" style={{ color: ACCENT }}>{v.variant}</code>
                <Badge text={v.ethnicity} color={ACCENT5} />
              </div>
              <div className="small text-muted mb-1"><strong>Domain:</strong> {v.domain}</div>
              <div className="small">{v.consequence}</div>
            </div>
          </div>
        ))}
      </Section>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function BBS20Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/bbs20/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(() => setError('Failed to load overview'));
    fetch(`${API}/api/bbs20/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(() => {});
    fetch(`${API}/api/bbs20/definitions`)
      .then(r => r.json()).then(setDefinitions)
      .catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-2 flex-wrap gap-2">
        <span style={{ fontSize: '1.6rem' }}>🧬</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
            BBS20 — WDPCP Bardet-Biedl Syndrome Type 20
          </h4>
          <div className="text-muted small">
            Fritz / CPLANE↔BBSome Bridge · INTU Absent at BB · PCP-Enriched Polydactyly · 2p15 · OMIM #617119
          </div>
        </div>
        <div className="ms-auto">
          <Badge text="CPLANE↔BBSome Bridge" color={ACCENT7} />
          <Badge text="INTU Absent at BB" color={ACCENT} />
          <Badge text="PCP Polydactyly" color={ACCENT2} />
          <Badge text="Fritz/WDPCP" color={ACCENT3} />
          <Badge text="AR" color={ACCENT6} />
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link${tab === i ? ' active' : ''}`}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <BreakdownTab data={breakdown} />}
      {tab === 2 && <TreatmentTab data={definitions} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}

      <div className="mt-4 pt-3 border-top text-muted small">
        <strong>BBS20 / WDPCP (Fritz)</strong> · 40-patient educational cohort (seed 371) ·
        Autosomal recessive · Locus 2p15 · OMIM gene *613580 · Disease #617119 ·
        Single disease tier (no NPHP/MKS/JBTS allelic variant) ·
        <Link href="/bbs" className="ms-2">← All BBS Types</Link>
      </div>
    </div>
  );
}
