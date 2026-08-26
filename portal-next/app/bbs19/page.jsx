'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Treatment & Diagnostics', 'Definitions'];

// BBS19 colour scheme — deep indigo (IFT-B2 tip complex; turnaround failure);
// amber (polydactyly/limb); teal (IFT anterograde train);
// dark rose (rod-cone degeneration); navy (epidemiology / MENA enrichment);
// dark slate (locus/gene); dark plum (cognitive/LD); rust (obesity/LepR)
const ACCENT  = '#1a237e';   // deep indigo — IFT172; IFT-B2 tip turnaround; largest IFT-B subunit
const ACCENT2 = '#e65100';   // deep amber — polydactyly; limb bud; tip-accumulation cascade
const ACCENT3 = '#006064';   // dark teal — IFT anterograde train; IFT-B2 complex
const ACCENT4 = '#880e4f';   // dark rose — rod-cone degeneration; retinal
const ACCENT5 = '#0d47a1';   // navy — MENA enrichment; Aldahmesh 2014; epidemiology
const ACCENT6 = '#37474f';   // dark slate — locus 2p23.3; gene/protein annotation
const ACCENT7 = '#4a148c';   // dark plum — cognitive/LD
const ACCENT8 = '#bf360c';   // rust — obesity; LepR retrograde IFT failure

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
        <strong>IFT172 / BBS19</strong> — Bardet-Biedl Syndrome Type 19.{' '}
        IFT172 is the <strong>largest IFT-B subunit (1749 aa)</strong> and the{' '}
        <strong>terminal cap of the IFT-B2 subcomplex</strong>, anchoring
        anterograde IFT trains at the ciliary tip for turnaround.{' '}
        BBS19 is caused by <em>biallelic hypomorphic or null/hypomorphic</em>{' '}
        IFT172 variants; biallelic null alleles cause the more severe{' '}
        <strong>NPHP17 (#616033)</strong> — nephronophthisis with retinal dystrophy.
      </Alert>
      <Alert color={ACCENT3}>
        <strong>Dual IFT-B + BBSome phenotype (unique to BBS19):</strong>{' '}
        IFT172 LOF removes the tip-turnaround cap → anterograde IFT cargo
        (IFT88, IFT-B subunits) accumulates at <strong>bulging ciliary tips</strong>;
        BBSome cannot be retrieved by BBS3/ARL6 → also <strong>tip-trapped</strong>.
        Cilia are <strong>shortened / dysmorphic</strong> — distinct from BBS15
        (full-length, BBSome-only tip-trap) and BBS17/18 (full-length, no IFT-B defect).
      </Alert>

      <div className="row g-2 mb-3">
        <KPI label="Cohort (N)" value={data.cohort_n} color={ACCENT} />
        <KPI label="Gene" value="IFT172" color={ACCENT} />
        <KPI label="Protein (aa)" value="1,749" color={ACCENT6} />
        <KPI label="Locus" value="2p23.3" color={ACCENT6} />
        <KPI label="OMIM Gene" value="*607386" color={ACCENT5} />
        <KPI label="OMIM Disease" value="#615995" color={ACCENT5} />
        <KPI label="OMIM NPHP17" value="#616033" color={ACCENT4} />
        <KPI label="Inheritance" value="AR" color={ACCENT} />
        <KPI label="BBS Freq." value="~0.5–1%" color={ACCENT2} />
        <KPI label="Families 2026" value="30–50" color={ACCENT2} />
        <KPI label="Tri-allelic" value={`${data.triallelic_pct}%`} color={ACCENT7} />
        <KPI label="Worldwide" value="~1/500k–1M" color={ACCENT5} />
      </div>

      <Section title="Systemic Burden — 40-Patient Cohort" color={ACCENT}>
        {burden.map(([label, n, pct]) => (
          <Bar key={label} label={label} value={n} max={maxN} color={
            label.includes('Obesity') ? ACCENT8 :
            label.includes('Retinal') || label.includes('retinal') ? ACCENT4 :
            label.includes('Polydactyly') || label.includes('polydactyly') ? ACCENT2 :
            label.includes('Renal') ? ACCENT3 :
            label.includes('Cogni') ? ACCENT7 :
            label.includes('Hypo') ? ACCENT5 :
            label.includes('Anosmia') ? ACCENT6 :
            label.includes('heart') || label.includes('CHD') ? ACCENT4 : ACCENT
          } />
        ))}
      </Section>

      <div className="row">
        <div className="col-md-6">
          <Section title="Retinal Stage Distribution" color={ACCENT4}>
            {Object.entries(data.retinal_stage_distribution || {}).map(([s, n]) => (
              <Bar key={s} label={s} value={n} max={maxN} color={ACCENT4} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Diagnosis Age Distribution" color={ACCENT5}>
            {Object.entries(data.dx_age_distribution || {}).map(([s, n]) => (
              <Bar key={s} label={s} value={n} max={maxN} color={ACCENT5} />
            ))}
          </Section>
        </div>
      </div>

      <Section title="IFT-B2 Tip-Turnaround Mechanism (BBS19 Unique)" color={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ background: ACCENT3 + '22' }}>
              <tr>
                <th>Component</th><th>BBS19 (IFT172 LOF)</th><th>BBS15 (IFT27 LOF)</th><th>BBS18 (BBIP1 LOF)</th>
              </tr>
            </thead>
            <tbody>
              <tr><td><strong>IFT172 IF</strong></td><td className="text-danger">ABSENT (lost)</td><td className="text-success">Normal</td><td className="text-success">Normal</td></tr>
              <tr><td><strong>IFT88 IF</strong></td><td className="text-warning">Bulging tip accumulation</td><td className="text-success">Normal</td><td className="text-success">Normal</td></tr>
              <tr><td><strong>BBSome (BBS2) IF</strong></td><td className="text-warning">Reduced + tip-trapped</td><td className="text-warning">Tip-trapped (full-length)</td><td className="text-danger">All subunits absent — assembly failure</td></tr>
              <tr><td><strong>MKKS/BCC IF</strong></td><td className="text-success">Normal</td><td className="text-success">Normal</td><td className="text-success">Normal</td></tr>
              <tr><td><strong>MKS1/TZ IF</strong></td><td className="text-success">Normal</td><td className="text-success">Normal</td><td className="text-success">Normal</td></tr>
              <tr><td><strong>Cilia morphology</strong></td><td className="text-danger">Shortened / bulging tips</td><td className="text-success">Full-length</td><td className="text-success">Full-length</td></tr>
              <tr><td><strong>Molecular lesion</strong></td><td>IFT-B2 tip cap lost — turnaround failure + BBSome retrieval blocked</td><td>GTPase lost — BBSome exit failure only</td><td>BBSome assembly failure — no functional octamer</td></tr>
              <tr><td><strong>NPHP tier</strong></td><td className="text-warning">Yes — NPHP17 (biallelic null)</td><td className="text-success">No</td><td className="text-success">No</td></tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Allele-Class → Disease Tier" color={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ background: ACCENT2 + '22' }}>
              <tr><th>Allele Class</th><th>Disease Tier</th><th>Key Features</th></tr>
            </thead>
            <tbody>
              <tr><td>Null / null (two truncating)</td><td className="text-danger fw-bold">NPHP17 (#616033)</td><td>Severe ESRD + retinal dystrophy; BBS pentad absent or minimal</td></tr>
              <tr><td>Null / hypomorphic</td><td className="text-warning fw-bold">BBS19 (#615995)</td><td>Full pentad; renal cysts ~42%; most common BBS19 genotype</td></tr>
              <tr><td>Hypomorphic / hypomorphic</td><td className="text-success fw-bold">Mild BBS19</td><td>Full pentad; renal often spared; slower retinal progression</td></tr>
              <tr><td>Tri-allelic BBS</td><td>BBS19 + modifier</td><td>~3% of families; third allele in a second BBS gene</td></tr>
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
  const maxN = _COHORT_SIZE;

  return (
    <div>
      <div className="row">
        <div className="col-md-6">
          <Section title="Systemic Burden (N=40)" color={ACCENT}>
            {(data.systemic_burden || []).map(({ feature, n, pct }) => (
              <Bar key={feature} label={`${feature} (${pct}%)`} value={n} max={maxN} color={
                feature.includes('Obesity') ? ACCENT8 :
                feature.includes('Polydactyly') ? ACCENT2 :
                feature.includes('Hypo') ? ACCENT5 :
                feature.includes('Cogni') ? ACCENT7 :
                feature.includes('Anosmia') ? ACCENT6 :
                feature.includes('Renal') ? ACCENT3 :
                feature.includes('heart') ? ACCENT4 : ACCENT
              } />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Ethnicity Distribution" color={ACCENT5}>
            {(data.ethnicity_distribution || []).map(({ ethnicity, n }) => (
              <Bar key={ethnicity} label={ethnicity} value={n} max={maxN} color={ACCENT5} />
            ))}
          </Section>
          <Section title="Allele Class" color={ACCENT}>
            {(data.allele_class_summary || []).map(({ label, n }) => (
              <Bar key={label} label={label} value={n} max={maxN} color={ACCENT} />
            ))}
          </Section>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <Section title="Retinal Stage" color={ACCENT4}>
            {(data.retinal_stage_distribution || []).map(({ label, n }) => (
              <Bar key={label} label={label} value={n} max={maxN} color={ACCENT4} />
            ))}
          </Section>
          <Section title="Renal Pattern" color={ACCENT3}>
            {(data.renal_distribution || []).map(({ label, n }) => (
              <Bar key={label} label={label} value={n} max={maxN} color={ACCENT3} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Presentation Mode" color={ACCENT2}>
            {(data.presentation_distribution || []).map(({ label, n }) => (
              <Bar key={label} label={label} value={n} max={maxN} color={ACCENT2} />
            ))}
          </Section>
          <Section title="Prior Misdiagnosis" color={ACCENT7}>
            {(data.misdiagnosis_distribution || []).map(({ label, n }) => (
              <Bar key={label} label={label} value={n} max={maxN} color={ACCENT7} />
            ))}
          </Section>
        </div>
      </div>

      <Section title="Top IFT172 Variants in Cohort" color={ACCENT6}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ background: ACCENT6 + '22' }}>
              <tr><th>Variant</th><th>N (allele 1)</th><th>Notes</th></tr>
            </thead>
            <tbody>
              {(data.top_variants || []).map(({ variant, n }) => (
                <tr key={variant}>
                  <td><code>{variant}</code></td>
                  <td>{n}</td>
                  <td>{
                    variant.includes('Ser1045') ? 'WD40 blade 7; MENA/Saudi; Aldahmesh 2014 first BBS19 allele' :
                    variant.includes('Trp1077') ? 'Truncating null; blades 8–20 lost; pan-ethnic; NPHP17 when biallelic' :
                    variant.includes('Gln914') ? 'IFT-A (IFT144) interface; anterograde-to-retrograde switch disrupted; European' :
                    variant.includes('Arg852') ? 'Tip-anchoring surface; IFT-B2 deanchored; South Asian/MENA' :
                    variant.includes('c.2943') ? 'Splice-donor null; intron 21; pan-ethnic' :
                    variant.includes('Leu1574') ? 'C-term WD40; IFT88 contact; hypomorphic; milder BBS19' :
                    ''
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

// ── Treatment & Diagnostics Tab ───────────────────────────────────────────────
function TreatmentTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const defs = data;

  return (
    <div>
      <Section title="Diagnostic Workup" color={ACCENT}>
        <ol className="small">
          {(defs.diagnostic_workup || []).map((step, i) => (
            <li key={i} className="mb-2">{step}</li>
          ))}
        </ol>
      </Section>
      <Section title="Treatment & Management" color={ACCENT3}>
        <ol className="small">
          {(defs.treatment_summary || []).map((step, i) => (
            <li key={i} className="mb-2">{step}</li>
          ))}
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
      <Section title="Gene Card — IFT172" color={ACCENT}>
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

      <Section title="Disease Card — BBS19 / NPHP17" color={ACCENT4}>
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

      <Section title="Key IFT172 Variants" color={ACCENT6}>
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
export default function BBS19Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/bbs19/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(() => setError('Failed to load overview'));
    fetch(`${API}/api/bbs19/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(() => {});
    fetch(`${API}/api/bbs19/definitions`)
      .then(r => r.json()).then(setDefinitions)
      .catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-2 flex-wrap gap-2">
        <span style={{ fontSize: '1.6rem' }}>🧬</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
            BBS19 — IFT172 Bardet-Biedl Syndrome Type 19
          </h4>
          <div className="text-muted small">
            IFT-B2 Terminal Cap · Tip-Turnaround Failure · Dual IFT-B + BBSome Tip-Trapping · 2p23.3 · OMIM #615995 / NPHP17 #616033
          </div>
        </div>
        <div className="ms-auto">
          <Badge text="IFT-B2 Tip Cap" color={ACCENT3} />
          <Badge text="Tip-Turnaround Failure" color={ACCENT} />
          <Badge text="NPHP17 Overlap" color={ACCENT4} />
          <Badge text="Aldahmesh 2014" color={ACCENT5} />
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
        <strong>BBS19 / IFT172</strong> · 40-patient educational cohort (seed 369) ·
        Autosomal recessive · Locus 2p23.3 · OMIM gene *607386 · Disease #615995 ·
        NPHP17 #616033 (biallelic null) ·
        <Link href="/bbs" className="ms-2">← All BBS Types</Link>
      </div>
    </div>
  );
}
