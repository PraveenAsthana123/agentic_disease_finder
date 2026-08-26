'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Variants & Diagnostics', 'Definitions'];

// BBS14 colour scheme — midnight blue / amber / deep teal (CEP290; TZ Y-link; multi-ciliopathy hub)
const ACCENT  = '#0d47a1';   // midnight blue — TZ Y-link scaffolding; CEP290 structure
const ACCENT2 = '#e65100';   // deep orange — allele class spectrum (MKS4↔LCA10↔BBS14↔JBTS5)
const ACCENT3 = '#1b5e20';   // dark green — polydactyly
const ACCENT4 = '#880e4f';   // dark rose — rod-cone dystrophy; retinal
const ACCENT5 = '#006064';   // dark teal — nephronophthisis; NPHP renal pattern
const ACCENT6 = '#37474f';   // dark slate — epidemiology; AR inheritance
const ACCENT7 = '#4a148c';   // deep purple — CEP290 multi-ciliopathy hub (LCA10/JBTS5/BBS14)
const ACCENT8 = '#bf360c';   // deep amber — obesity; LepR TZ pathway

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
  const { data: ov, err: ovErr } = useFetch('/api/bbs14/overview');

  if (ovErr) return <div className="alert alert-danger">Failed to load overview: {ovErr}</div>;
  if (!ov)   return <div className="text-muted p-3">Loading overview…</div>;

  const kc = ov.key_counts || {};

  return (
    <div>
      {/* Unique mechanism banner */}
      <Alert color={ACCENT2}>
        <strong>BBS14 (CEP290) — Second BBS Gene NOT in BBSome or BCC. Most Common Ciliopathy Gene Overall.</strong>{' '}
        CEP290 is a <strong>Transition Zone (TZ) Y-link scaffold</strong> (NPHP-MKS-JBTS module, distal TZ position).
        The BBSome and MKS sub-module assemble normally —{' '}
        <em>BBS2, BBS4, BBS8, BBS9, MKKS all NORMAL; MKS1 NORMAL on IF</em> (KEY: distinguishes BBS14 from BBS13).
        Only CEP290 is absent; NPHP5/IQCB1 is deranged at TZ.{' '}
        <strong>Same gene causes LCA10, JBTS5, NPHP6, MKS4</strong> — allele class determines disease.
        BBS14 requires ≥1 hypomorphic allele; null/null → MKS4 (lethal).
      </Alert>

      {/* KPI row */}
      <div className="row mb-3">
        <KPI label="Cohort (N)"         value={ov.cohort_n}                               color={ACCENT}  />
        <KPI label="Gene"               value={ov.gene}                                   color={ACCENT}  />
        <KPI label="Locus"              value={ov.locus}                                  color={ACCENT6} />
        <KPI label="OMIM Gene"          value={ov.omim_gene}                              color={ACCENT6} />
        <KPI label="BBS Frequency"      value="<1–2%"                                     color={ACCENT7} />
        <KPI label="Tri-allelic"        value={`~${ov.triallelic_pct}%`}                  color={ACCENT7} />
      </div>
      <div className="row mb-3">
        <KPI label="Polydactyly"        value={`${kc.polydactyly_pct}%`}                  color={ACCENT3} />
        <KPI label="Obesity"            value={`${kc.obesity_pct}%`}                      color={ACCENT8} />
        <KPI label="Renal (any)"        value={`${kc.renal_any_pct}%`}                   color={ACCENT5} />
        <KPI label="NPHP specifically"  value={`${kc.nphp_pct}%`}                        color={ACCENT5} />
        <KPI label="Hypogonadism"       value={`${kc.hypogonadism_pct}%`}                 color={ACCENT4} />
        <KPI label="Cognitive/LD"       value={`${kc.cognitive_pct}%`}                   color={ACCENT7} />
      </div>
      <div className="row mb-4">
        <KPI label="Anosmia"            value={`${kc.anosmia_pct}%`}                     color={ACCENT6} />
        <KPI label="CHD"                value={`${kc.chd_pct}%`}                         color={ACCENT4} />
        <KPI label="JBTS5 overlap"      value={`${kc.jbts_overlap_pct}%`}                color={ACCENT2} />
        <KPI label="LCA10 spectrum"     value={`${kc.lca10_spectrum_pct}%`}              color={ACCENT7} />
        <KPI label="ESRD risk"          value={`${kc.esrd_pct}%`}                        color={ACCENT5} />
        <KPI label="Misdiagnosis"       value={`${kc.misdiagnosis_pct}%`}                color={ACCENT6} />
      </div>

      {/* IF fingerprint — critical diagnostic */}
      <Section title="IF Fingerprint — Critical BBS14 vs BBS13 Distinction" color={ACCENT}>
        <Alert color={ACCENT}>
          <strong>BBSome (BBS2·BBS4·BBS8·BBS9): NORMAL</strong> — BBSome assembles; BCC intact (MKKS/BBS10/BBS12 normal).{' '}
          <strong>MKS1: NORMAL</strong> — MKS sub-module INTACT (unique BBS14 fingerprint vs BBS13 where MKS1 is absent).{' '}
          <strong>CEP290: ABSENT</strong> at ciliary base. <strong>NPHP5/IQCB1: REDUCED</strong> at TZ (docking partner lost).
          TZ Y-links disrupted → gate leaky → GPCR mismigration.
        </Alert>
      </Section>

      {/* Allele class spectrum */}
      <Section title="Allele Class → Disease Tier (CEP290)" color={ACCENT2}>
        <div className="row">
          {[
            { cls: 'Null / Null', tier: 'MKS4 (Meckel-Gruber Variant) — LETHAL', color: '#b71c1c' },
            { cls: 'Hypomorphic / Hypomorphic', tier: 'BBS14 (full BBS) or LCA10 (CC4/5 missense)', color: ACCENT },
            { cls: 'Deep-Intronic / Hypomorphic', tier: 'LCA10-BBS14 spectrum (IVS26+1655A>G)', color: ACCENT7 },
            { cls: 'Hypomorphic / Truncating', tier: 'BBS14 or JBTS5 (variable)', color: ACCENT2 },
          ].map(({ cls, tier, color }) => (
            <div key={cls} className="col-12 col-md-6 mb-2">
              <div className="p-2 rounded border" style={{ borderLeft: `4px solid ${color}` }}>
                <div className="fw-bold small" style={{ color }}>{cls}</div>
                <div className="text-muted small">→ {tier}</div>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* Mechanism */}
      <Section title="Molecular Mechanism" color={ACCENT6}>
        <p className="small mb-1">
          <strong>CEP290</strong> is a 2479 aa coiled-coil scaffold at the ciliary TZ. Its CC4/CC5 domains (aa 1600–2479)
          anchor the TZ Y-links — the structural elements connecting outer doublet microtubules to the ciliary membrane.
          Without CEP290, Y-links fail to form, the TZ diffusion barrier is breached, and non-ciliary membrane proteins
          (including hypothalamic LepR and photoreceptor GPCRs) enter the ciliary compartment aberrantly.
        </p>
        <p className="small mb-1">
          <strong>Mechanism differs from BBS13/MKS1</strong>: CEP290 is at the DISTAL TZ position; MKS1 is more proximal.
          Loss of CEP290 leaves the MKS sub-module intact (MKS1, CC2D2A, B9D1/B9D2 all normal) but disrupts the
          distal Y-link anchoring layer — a distinct TZ architecture failure.
        </p>
        <p className="small mb-0">
          <strong>LepR obesity mechanism</strong>: same downstream endpoint as BBS1-12 BBSome LOF (LepR mislocalises
          → leptin resistance → obesity) but via TZ gate leakiness rather than BBSome IFT-B docking failure.
        </p>
      </Section>

      {/* Systemic burden */}
      <Section title="Systemic Feature Burden" color={ACCENT4}>
        {(ov.systemic_burden || []).map(([feat, n, pct]) => (
          <Bar key={feat} label={feat} value={n} max={ov.cohort_n} color={ACCENT4} />
        ))}
      </Section>

      {/* Retinal stage */}
      <Section title="Retinal Stage Distribution" color={ACCENT4}>
        <div className="row">
          {Object.entries(ov.retinal_stage_distribution || {}).map(([stage, cnt]) => (
            <div key={stage} className="col-12 col-md-6 mb-2">
              <div className="d-flex justify-content-between p-2 rounded" style={{ background: ACCENT4 + '14' }}>
                <span className="small">{stage}</span>
                <strong>{cnt}</strong>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* Dx age */}
      <Section title="Age at Diagnosis Distribution" color={ACCENT6}>
        <div className="row">
          {Object.entries(ov.dx_age_distribution || {}).map(([age, cnt]) => (
            <div key={age} className="col-6 col-md-3 mb-2">
              <div className="text-center p-2 rounded" style={{ background: ACCENT6 + '14' }}>
                <div className="fw-bold">{cnt}</div>
                <div className="text-muted small">{age}</div>
              </div>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}

// ── Tab: Multi-System Breakdown ───────────────────────────────────────────────
function BreakdownTab() {
  const { data: bd, err: bdErr } = useFetch('/api/bbs14/breakdown');

  if (bdErr) return <div className="alert alert-danger">Failed to load breakdown: {bdErr}</div>;
  if (!bd)   return <div className="text-muted p-3">Loading breakdown…</div>;

  return (
    <div>
      <Section title="Systemic Feature Burden (N=40)" color={ACCENT4}>
        {(bd.systemic_burden || []).map(({ feature, n, pct }) => (
          <div key={feature} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span>{feature}</span><span className="fw-bold">{n} ({pct}%)</span>
            </div>
            <div style={{ background: '#e9ecef', borderRadius: 4, height: 10 }}>
              <div style={{ width: `${pct}%`, background: ACCENT4, borderRadius: 4, height: 10 }} />
            </div>
          </div>
        ))}
      </Section>

      <div className="row">
        <div className="col-md-6">
          <Section title="Ethnicity Distribution" color={ACCENT6}>
            {(bd.ethnicity_distribution || []).map(({ ethnicity, n }) => (
              <Bar key={ethnicity} label={ethnicity} value={n} max={40} color={ACCENT6} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Allele Class Summary" color={ACCENT2}>
            {(bd.allele_class_summary || []).map(({ label, n }) => (
              <Bar key={label} label={label} value={n} max={40} color={ACCENT2} />
            ))}
          </Section>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <Section title="Retinal Stage Distribution" color={ACCENT4}>
            {(bd.retinal_stage_distribution || []).map(({ label, n }) => (
              <Bar key={label} label={label} value={n} max={40} color={ACCENT4} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Renal Phenotype Distribution" color={ACCENT5}>
            {(bd.renal_distribution || []).map(({ label, n }) => (
              <Bar key={label} label={label} value={n} max={40} color={ACCENT5} />
            ))}
          </Section>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <Section title="Polydactyly Type Distribution" color={ACCENT3}>
            {(bd.polydactyly_distribution || []).map(({ label, n }) => (
              <Bar key={label} label={label} value={n} max={40} color={ACCENT3} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Presentation Age Distribution" color={ACCENT8}>
            {(bd.presentation_distribution || []).map(({ label, n }) => (
              <Bar key={label} label={label} value={n} max={40} color={ACCENT8} />
            ))}
          </Section>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <Section title="Initial Misdiagnosis Pattern" color={ACCENT7}>
            {(bd.misdiagnosis_distribution || []).map(({ label, n }) => (
              <Bar key={label} label={label} value={n} max={40} color={ACCENT7} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Top Observed Variants (cohort)" color={ACCENT}>
            {(bd.top_variants || []).map(({ variant, n }) => (
              <Bar key={variant} label={variant} value={n} max={40} color={ACCENT} />
            ))}
          </Section>
        </div>
      </div>
    </div>
  );
}

// ── Tab: Variants & Diagnostics ───────────────────────────────────────────────
function VariantsTab() {
  const { data: df, err: dfErr } = useFetch('/api/bbs14/definitions');

  if (dfErr) return <div className="alert alert-danger">Failed to load definitions: {dfErr}</div>;
  if (!df)   return <div className="text-muted p-3">Loading variant data…</div>;

  return (
    <div>
      <Section title="Key CEP290 Variants (BBS14 context)" color={ACCENT}>
        {(df.key_variants || []).map((v, i) => (
          <div key={i} className="card mb-3 shadow-sm">
            <div className="card-body">
              <div className="d-flex align-items-start gap-2 mb-2">
                <Badge text={`V${i+1}`} color={ACCENT} />
                <strong className="small">{v.variant}</strong>
              </div>
              <div className="small mb-1"><span className="text-muted">Domain:</span> {v.domain}</div>
              <div className="small mb-1"><span className="text-muted">Consequence:</span> {v.consequence}</div>
              <div className="small"><span className="text-muted">Population:</span> <Badge text={v.ethnicity} color={ACCENT6} /></div>
            </div>
          </div>
        ))}
      </Section>

      <Section title="Diagnostic Workup (BBS14 / CEP290)" color={ACCENT5}>
        <ol className="small">
          {(df.diagnostic_workup || []).map((step, i) => (
            <li key={i} className="mb-2">{step}</li>
          ))}
        </ol>
      </Section>

      <Section title="Treatment Summary" color={ACCENT8}>
        <ol className="small">
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
  const { data: df, err: dfErr } = useFetch('/api/bbs14/definitions');

  if (dfErr) return <div className="alert alert-danger">Failed to load definitions: {dfErr}</div>;
  if (!df)   return <div className="text-muted p-3">Loading definitions…</div>;

  const gc = df.gene_card || {};
  const dc = df.disease_card || {};

  return (
    <div>
      <Section title="Gene Card — CEP290" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <tbody>
              {Object.entries(gc).map(([k, v]) => (
                <tr key={k}>
                  <td className="fw-semibold text-nowrap" style={{ color: ACCENT, width: 180 }}>{k.replace(/_/g,' ')}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Disease Card — BBS14" color={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <tbody>
              {Object.entries(dc).map(([k, v]) => (
                <tr key={k}>
                  <td className="fw-semibold text-nowrap" style={{ color: ACCENT2, width: 180 }}>{k.replace(/_/g,' ')}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Multi-ciliopathy hub */}
      <Section title="CEP290 Multi-Ciliopathy Allele Spectrum" color={ACCENT7}>
        <Alert color={ACCENT7}>
          <strong>CEP290 is the most commonly mutated ciliopathy gene across all ciliopathies.</strong>{' '}
          Depending on allele class and tissue expression context:{' '}
          <strong>LCA10</strong> (#611755 — ~25% of all LCA; deep intronic IVS26+1655A>G most common);{' '}
          <strong>JBTS5</strong> (~8% of Joubert syndrome; mixed allele class);{' '}
          <strong>NPHP6</strong> (~2% of nephronophthisis; truncating alleles);{' '}
          <strong>MKS4</strong> (#611134 — Meckel-Gruber variant, null/null, lethal);{' '}
          <strong>SLSN6</strong> (Senior-Løken syndrome, ~rare);{' '}
          <strong>BBS14</strong> (<1–2% of all BBS — the rarest clinical tier of CEP290 disease).
        </Alert>
        <div className="row mt-3">
          {[
            { disease: 'LCA10', freq: 'Most common', allele: 'IVS26+1655A>G / hypomorphic', color: ACCENT },
            { disease: 'JBTS5', freq: '~8% of JBTS', allele: 'Hypomorphic/truncating', color: ACCENT7 },
            { disease: 'NPHP6', freq: 'Rare', allele: 'Truncating/truncating (viable)', color: ACCENT5 },
            { disease: 'MKS4', freq: 'Lethal', allele: 'Null/null', color: '#b71c1c' },
            { disease: 'BBS14', freq: '<1–2% BBS', allele: 'Hypomorphic/hypomorphic', color: ACCENT2 },
          ].map(({ disease, freq, allele, color }) => (
            <div key={disease} className="col-12 col-md-6 mb-2">
              <div className="p-2 rounded border" style={{ borderLeft: `4px solid ${color}` }}>
                <div className="fw-bold small" style={{ color }}>{disease}</div>
                <div className="text-muted small">{freq} · {allele}</div>
              </div>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function BBS14Page() {
  const [tab, setTab] = useState(0);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🧬 BBS14 — Bardet-Biedl Syndrome Type 14 (CEP290)
        </h4>
        <div className="text-muted small">
          CEP290 · *610142 · 2479 aa · Chr 12q21.32 · TZ Y-link scaffold (NPHP-MKS-JBTS module, distal position) ·
          Most common ciliopathy gene overall (LCA10/JBTS5/NPHP6/MKS4/BBS14) ·
          Cohort N={_COHORT_SIZE} · seed 359
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab />}
      {tab === 1 && <BreakdownTab />}
      {tab === 2 && <VariantsTab />}
      {tab === 3 && <DefinitionsTab />}
    </div>
  );
}
