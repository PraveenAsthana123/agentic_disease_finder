'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Treatment & Diagnostics', 'Definitions'];

// BBS17 colour scheme — deep indigo (LZTFL1/LZ-domain); amber (polydactyly/Hh-Smo gate);
// forest green (Smo ciliary entry / Hh pathway); dark rose (rod-cone degeneration);
// deep cyan (renal cystic); dark slate (epidemiology); dark plum (cognitive/LD);
// rust (obesity/LepR gate failure)
const ACCENT  = '#311b92';   // deep indigo — LZTFL1; LZ domain; BBSome entry gating
const ACCENT2 = '#ff6f00';   // amber — polydactyly; Shh limb bud; Hh gate
const ACCENT3 = '#1b5e20';   // forest green — Smo ciliary entry; Hh pathway; BBSome rescue
const ACCENT4 = '#880e4f';   // dark rose — rod-cone degeneration; retinal
const ACCENT5 = '#006064';   // deep cyan — renal cystic/structural (NOT NPHP — contrast BBS16)
const ACCENT6 = '#37474f';   // dark slate — epidemiology; locus
const ACCENT7 = '#4a148c';   // dark plum — cognitive/LD
const ACCENT8 = '#bf360c';   // rust — obesity; LepR gate failure; TULP3 axis

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
  const { data: ov, err: ovErr } = useFetch('/api/bbs17/overview');

  if (ovErr) return <div className="alert alert-danger">Failed to load overview: {ovErr}</div>;
  if (!ov)   return <div className="text-muted p-3">Loading overview…</div>;

  const kc = ov.key_counts || {};

  return (
    <div>
      {/* Unique mechanism banner */}
      <Alert color={ACCENT}>
        <strong>BBS17 (LZTFL1) — BBSome Ciliary Entry Gatekeeper Failure.</strong>{' '}
        LZTFL1 (299 aa) tethers the assembled BBSome at the transition zone / ciliary gate and releases it
        upon Hedgehog pathway activation (phosphorylation of Ser228/Thr245 → gate opens → BBSome enters cilia).{' '}
        LZTFL1 LOF → <strong>BBSome CANNOT enter cilia</strong> despite correct assembly at basal body.{' '}
        Cilia are <strong>full length</strong> and present (NOT shortened — contrast BBS16).{' '}
        BBSome is <strong>not trapped in cilia</strong> (contrast BBS15 — exit failure).{' '}
        BBS17 is mechanistically opposite to BBS15: entry failure vs retrograde exit failure.{' '}
        Also disrupts LZTFL1–TULP3 co-scaffold for GPR75/GPR161 ciliary import (dual Hh + cAMP defect).
      </Alert>

      {/* KPI row */}
      <div className="row mb-3">
        <KPI label="Cohort (N)"         value={ov.cohort_n}                           color={ACCENT}  />
        <KPI label="Gene"               value={ov.gene}                               color={ACCENT}  />
        <KPI label="Locus"              value={ov.locus}                              color={ACCENT6} />
        <KPI label="OMIM Gene"          value={ov.omim_gene}                          color={ACCENT6} />
        <KPI label="BBS Frequency"      value="<1%"                                   color={ACCENT7} />
        <KPI label="Tri-allelic"        value={`~${ov.triallelic_pct}%`}              color={ACCENT7} />
      </div>
      <div className="row mb-3">
        <KPI label="Polydactyly"        value={`${kc.polydactyly_pct}%`}              color={ACCENT2} />
        <KPI label="Obesity"            value={`${kc.obesity_pct}%`}                  color={ACCENT8} />
        <KPI label="Renal (any)"        value={`${kc.renal_any_pct}%`}               color={ACCENT5} />
        <KPI label="Hypogonadism"       value={`${kc.hypogonadism_pct}%`}             color={ACCENT4} />
        <KPI label="Cognitive/LD"       value={`${kc.cognitive_pct}%`}               color={ACCENT7} />
        <KPI label="Anosmia"            value={`${kc.anosmia_pct}%`}                 color={ACCENT6} />
      </div>
      <div className="row mb-4">
        <KPI label="CHD"                value={`${kc.chd_pct}%`}                     color={ACCENT4} />
        <KPI label="Retinal End-Stage"  value={`${kc.retinal_endstage_pct}%`}        color={ACCENT4} />
        <KPI label="Misdiagnosis"       value={`${kc.misdiagnosis_pct}%`}            color={ACCENT6} />
        <KPI label="Consanguinity"      value={`${kc.consanguinity_pct}%`}           color={ACCENT6} />
        <KPI label="Families WW"        value="20–35"                                color={ACCENT6} />
        <KPI label="Protein Size"       value="299 aa"                               color={ACCENT}  />
      </div>

      {/* IF fingerprint — critical diagnostic */}
      <Section title="IF Fingerprint — BBS17 Pattern (BBSome Entry Gate Failure)" color={ACCENT}>
        <Alert color={ACCENT}>
          <strong>LZTFL1 ABSENT from ciliary base / TZ</strong> (anti-LZTFL1 IF — normal puncta at ciliary
          base gone).{' '}
          <strong>BBSome REDUCED IN CILIA</strong> (anti-BBS9/BBS2 IF — below normal ciliary body levels;
          NOT absent from basal body where assembly is normal).{' '}
          <strong>Cilia FULL LENGTH</strong> — acetylated α-tubulin IF shows normal-length cilia present
          (NOT shortened/absent as in BBS16).{' '}
          <strong>BBSome NOT trapped at ciliary tip</strong> (contrast BBS15 where anti-BBS2 shows
          base + tip accumulation; BBS17 shows only reduced ciliary body, no tip enrichment).{' '}
          <strong>Smo FAILS to accumulate in cilia</strong> upon SAG stimulation — pathognomonic for
          BBS17; direct test of BBSome entry gate function.{' '}
          <strong>MKKS/BCC: NORMAL</strong>.{' '}
          <strong>MKS1: NORMAL</strong> (TZ scaffold intact — contrast BBS13).
        </Alert>
        <div className="row">
          {[
            { marker: 'LZTFL1 (ciliary base / TZ)', status: 'ABSENT — no basal puncta on IF', color: '#b71c1c' },
            { marker: 'BBSome in cilia (BBS9 / BBS2)', status: 'REDUCED — entry blocked; basal body assembly normal', color: ACCENT },
            { marker: 'Cilia (acetylated α-tubulin)', status: 'FULL LENGTH present — NOT shortened (contrast BBS16)', color: ACCENT3 },
            { marker: 'BBSome at ciliary tip', status: 'NOT trapped/enriched (contrast BBS15 — entry failure, not exit)', color: ACCENT3 },
            { marker: 'Smo in cilia (SAG stimulated)', status: 'BLOCKED — Hh pathway Smo entry fails (pathognomonic BBS17)', color: '#b71c1c' },
            { marker: 'MKKS / BCC', status: 'NORMAL (upstream chaperonin intact)', color: ACCENT3 },
            { marker: 'MKS1 (TZ scaffold)', status: 'NORMAL (TZ Y-links intact — contrast BBS13)', color: ACCENT3 },
          ].map(({ marker, status, color }) => (
            <div key={marker} className="col-12 col-md-6 mb-2">
              <div className="p-2 rounded border" style={{ borderLeft: `4px solid ${color}` }}>
                <div className="fw-bold small" style={{ color }}>{marker}</div>
                <div className="text-muted small">→ {status}</div>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* Mechanism */}
      <Section title="Molecular Mechanism — BBSome Ciliary Entry Gate (LZTFL1–BBSome–TULP3)" color={ACCENT6}>
        <p className="small mb-1">
          <strong>LZTFL1</strong> (299 aa) localises to the distal segment of the transition zone / ciliary gate,
          where it forms a tripartite gate complex with the assembled BBSome (via BBS9/PTHB1 B9 domain) and TULP3
          (tubby-like protein 3). In resting state, LZTFL1 holds the BBSome at the ciliary base, preventing
          unregulated ciliary entry.
        </p>
        <p className="small mb-1">
          <strong>Hh-regulated gate opening:</strong> When the Hedgehog ligand activates Smo, downstream kinases
          (CK2α, DYRK1) phosphorylate LZTFL1 at Ser228 and Thr245. This conformational change releases the
          BBSome, which then enters the cilia and carries Smo and other GPCRs inward for signalling. In parallel,
          TULP3 uses the same LZTFL1 scaffold to import GPR161 (inhibitory cAMP regulator) — also blocked in
          BBS17.
        </p>
        <p className="small mb-0">
          <strong>BBS17 LOF consequence:</strong> No gatekeeper protein → BBSome cannot bind the ciliary entry
          point → BBSome cannot enter cilia even when needed. Ciliary cargo (LepR, Smo, GPR161, olfactory GPCRs)
          cannot be properly trafficked. Phenotypic result: obesity (LepR), retinal degeneration (opsin
          compartmentalisation), anosmia (olfactory GPCR), hypogonadism (gonadotrophin axis GPCRs), and partial
          Hh pathway defect (polydactyly, CHD).
        </p>
      </Section>

      {/* BBS17 vs BBS15 comparison — entry vs exit */}
      <Section title="BBS17 vs BBS15 — Entry Failure vs Exit Failure (Key Mechanistic Pair)" color={ACCENT2}>
        <Alert color={ACCENT2}>
          <strong>BBS17 and BBS15 are mechanistically opposite but phenotypically similar.</strong>{' '}
          Both reduce ciliary BBSome levels, but BBS17 blocks <em>entry</em> while BBS15 traps BBSome at
          the ciliary <em>tip</em> (exit failure). This makes IF the definitive discriminator.
        </Alert>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ background: ACCENT2 + '18' }}>
              <tr>
                <th>Feature</th>
                <th>BBS17 (LZTFL1)</th>
                <th>BBS15 (IFT27/RABL4)</th>
                <th>BBS16 (SDCCAG8)</th>
                <th>BBS13 (MKS1)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="fw-semibold">Failure point</td>
                <td style={{ color: ACCENT }}>BBSome entry gate</td>
                <td style={{ color: ACCENT7 }}>BBSome retrograde exit</td>
                <td style={{ color: '#b71c1c' }}>Ciliogenesis (most upstream)</td>
                <td style={{ color: ACCENT5 }}>TZ Y-link scaffold</td>
              </tr>
              <tr>
                <td className="fw-semibold">Cilia present?</td>
                <td style={{ color: ACCENT3 }}>YES — full length</td>
                <td style={{ color: ACCENT3 }}>YES — full length</td>
                <td style={{ color: '#b71c1c' }}>SHORTENED / ABSENT</td>
                <td style={{ color: ACCENT3 }}>YES (TZ gate leaky)</td>
              </tr>
              <tr>
                <td className="fw-semibold">BBSome in cilia</td>
                <td style={{ color: ACCENT }}>REDUCED (entry blocked)</td>
                <td style={{ color: ACCENT7 }}>TRAPPED at tip (exit blocked)</td>
                <td style={{ color: '#b71c1c' }}>REDUCED at basal body</td>
                <td style={{ color: ACCENT3 }}>NORMAL IF (TZ gate leaky)</td>
              </tr>
              <tr>
                <td className="fw-semibold">BBSome tip enrichment</td>
                <td>No (entry blocked)</td>
                <td style={{ color: ACCENT7 }}>YES — pathognomonic BBS15</td>
                <td>No</td>
                <td>No</td>
              </tr>
              <tr>
                <td className="fw-semibold">MKKS/BCC</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
              </tr>
              <tr>
                <td className="fw-semibold">MKS1</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
                <td style={{ color: ACCENT }}>REDUCED/DELOCALIZED</td>
                <td style={{ color: '#b71c1c' }}>ABSENT (LOF gene)</td>
              </tr>
              <tr>
                <td className="fw-semibold">Smo ciliary entry</td>
                <td style={{ color: '#b71c1c' }}>BLOCKED (pathognomonic)</td>
                <td>Reduced (exit trapping)</td>
                <td>N/A (no cilia)</td>
                <td>Partially impaired</td>
              </tr>
              <tr>
                <td className="fw-semibold">Renal pattern</td>
                <td>Cystic/structural ~42%</td>
                <td>Cystic ~35%</td>
                <td>NPHP-type ~53%</td>
                <td>NPHP-type ~52%</td>
              </tr>
              <tr>
                <td className="fw-semibold">Allele-class tiers</td>
                <td>BBS17 only (single tier)</td>
                <td>BBS15 only (single tier)</td>
                <td>BBS16 / NPHP10 / SLS7</td>
                <td>BBS13 / JBTS28 / MKS1</td>
              </tr>
            </tbody>
          </table>
        </div>
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
  const { data: bd, err: bdErr } = useFetch('/api/bbs17/breakdown');

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
          <Section title="Polydactyly Distribution" color={ACCENT3}>
            {(bd.polydactyly_distribution || []).map(({ label, n }) => (
              <Bar key={label} label={label} value={n} max={40} color={ACCENT3} />
            ))}
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Presentation Mode Distribution" color={ACCENT8}>
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

// ── Tab: Treatment & Diagnostics ──────────────────────────────────────────────
function TreatmentTab() {
  const { data: df, err: dfErr } = useFetch('/api/bbs17/definitions');

  if (dfErr) return <div className="alert alert-danger">Failed to load definitions: {dfErr}</div>;
  if (!df)   return <div className="text-muted p-3">Loading variant data…</div>;

  return (
    <div>
      <Section title="Key LZTFL1 Variants (BBS17 context)" color={ACCENT}>
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

      <Section title="Diagnostic Workup (BBS17 / LZTFL1)" color={ACCENT5}>
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
  const { data: df, err: dfErr } = useFetch('/api/bbs17/definitions');

  if (dfErr) return <div className="alert alert-danger">Failed to load definitions: {dfErr}</div>;
  if (!df)   return <div className="text-muted p-3">Loading definitions…</div>;

  const gc = df.gene_card || {};
  const dc = df.disease_card || {};

  return (
    <div>
      <Section title="Gene Card — LZTFL1 / BBS17 / HIPPI" color={ACCENT}>
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

      <Section title="Disease Card — BBS17" color={ACCENT2}>
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

      {/* BBS module classification */}
      <Section title="BBS17 / LZTFL1 — BBSome Gate Module and Unique Position" color={ACCENT}>
        <Alert color={ACCENT}>
          <strong>LZTFL1/BBS17 is the first BBS gene classified as a BBSome ciliary entry gatekeeper.</strong>{' '}
          All prior BBS genes operate within the BBSome (BBS1-9), in BCC pre-assembly (BBS6/10/12),
          at the TZ scaffold (BBS13/14), in IFT-B retrograde retrieval (BBS15), or in centriolar satellite
          ciliogenesis initiation (BBS16). LZTFL1/BBS17 acts AT the ciliary entry gate — regulating whether
          the already-assembled BBSome can access the ciliary axoneme. This gate function is unique among
          all known BBS mechanisms.
        </Alert>
        <div className="row mt-3">
          {[
            { group: 'BBSome subunits (BBS1-9)', function: 'BBSome structural assembly', examples: 'BBS1, BBS2, BBS4, BBS5, BBS7, BBS8, BBS9', color: ACCENT },
            { group: 'BCC chaperonins (BBS6, BBS10, BBS12)', function: 'BBSome pre-assembly folding', examples: 'MKKS/BBS6, BBS10, BBS12', color: ACCENT3 },
            { group: 'TZ scaffolds (BBS13-14)', function: 'Transition zone Y-link/gate scaffold', examples: 'MKS1/BBS13, CEP290/BBS14', color: ACCENT5 },
            { group: 'IFT-B GTPase (BBS15)', function: 'BBSome retrograde exit from cilia', examples: 'IFT27/RABL4 — exit failure (tip trap)', color: ACCENT7 },
            { group: 'Centriolar satellite (BBS16)', function: 'Ciliogenesis initiation — TZ delivery', examples: 'SDCCAG8/NPHP10 — most upstream mechanism', color: ACCENT6 },
            { group: 'Ciliary entry gatekeeper (BBS17)', function: 'BBSome anterograde entry + TULP3 scaffold', examples: 'LZTFL1/HIPPI — entry failure (opposite of BBS15)', color: ACCENT },
          ].map(({ group, function: fn, examples, color }) => (
            <div key={group} className="col-12 col-md-6 mb-2">
              <div className="p-2 rounded border" style={{ borderLeft: `4px solid ${color}` }}>
                <div className="fw-bold small" style={{ color }}>{group}</div>
                <div className="text-muted small">Function: {fn}</div>
                <div className="text-muted small">Examples: {examples}</div>
              </div>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function BBS17Page() {
  const [tab, setTab] = useState(0);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🧬 BBS17 — LZTFL1 / HIPPI Ciliopathy (BBSome Entry Gate)
        </h4>
        <div className="text-muted small">
          Rod-Cone Dystrophy · Post-Axial Polydactyly · Obesity · Learning Disability · Renal Cystic ·
          LZTFL1 BBSome Entry Gatekeeper · Chr 2p22.1 · OMIM *606568/#615991 · Hh/Smo Gate Failure ·
          Autosomal Recessive · ~1/1,000,000–3,000,000 · Cohort N={_COHORT_SIZE} · seed 365
        </div>
        <div className="mt-1">
          <Badge text="BBS17 *606568" color={ACCENT} />
          <Badge text="LZTFL1 / HIPPI" color={ACCENT6} />
          <Badge text="2p22.1" color={ACCENT6} />
          <Badge text="BBSome Entry Gate" color={ACCENT3} />
          <Badge text="Entry Failure (≠ BBS15 exit)" color={ACCENT2} />
          <Badge text="Hh/Smo blocked" color={ACCENT8} />
          <Badge text="AR" color={ACCENT7} />
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
      {tab === 2 && <TreatmentTab />}
      {tab === 3 && <DefinitionsTab />}
    </div>
  );
}
