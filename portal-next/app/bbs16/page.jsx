'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Treatment & Diagnostics', 'Definitions'];

// BBS16 colour scheme — deep purple (centrosomal satellite/PCM1); deep orange (polydactyly);
// dark green (residual function in hypomorphic); dark rose (rod-cone degeneration);
// dark teal (renal NPHP tubular); dark slate (epidemiology);
// dark brown (cognitive/LD); burnt orange (obesity/ciliogenesis)
const ACCENT  = '#4a148c';   // deep purple — SDCCAG8; centrosomal satellite; PCM1 biology
const ACCENT2 = '#e65100';   // deep orange — polydactyly; limb bud Shh cilia
const ACCENT3 = '#1b5e20';   // dark green — residual function; partial ciliogenesis
const ACCENT4 = '#880e4f';   // dark rose — rod-cone degeneration
const ACCENT5 = '#006064';   // dark teal — renal NPHP tubular/fibrotic
const ACCENT6 = '#37474f';   // dark slate — epidemiology
const ACCENT7 = '#4e342e';   // dark brown — cognitive/LD
const ACCENT8 = '#bf360c';   // burnt orange — obesity; ciliogenesis → LepR absent

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
  const { data: ov, err: ovErr } = useFetch('/api/bbs16/overview');

  if (ovErr) return <div className="alert alert-danger">Failed to load overview: {ovErr}</div>;
  if (!ov)   return <div className="text-muted p-3">Loading overview…</div>;

  const kc = ov.key_counts || {};

  return (
    <div>
      {/* Unique mechanism banner */}
      <Alert color={ACCENT}>
        <strong>BBS16 (SDCCAG8 / NPHP10) — Centriolar Satellite Ciliogenesis Defect.</strong>{' '}
        SDCCAG8 anchors the NPHP module (NPHP1, NPHP4, NPHP5, NPHP8) to PCM1-positive centriolar satellites,
        enabling TZ protein delivery to the ciliary base.{' '}
        SDCCAG8 LOF → <strong>cilia shortened or absent</strong> (ciliogenesis defect — upstream of BBSome, IFT, and TZ).{' '}
        This is <em>the most upstream BBS mechanism characterised to date</em>:{' '}
        no cilia → no downstream ciliary biology. PCM1 satellites disorganised.
        NPHP1/NPHP4 delocalized. Also causes <strong>NPHP10</strong> and <strong>SLS7</strong> in null allele classes.
        MENA/Bedouin enrichment (Otto 2010). Renal = NPHP-type (NOT cystic).
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
        <KPI label="Renal (NPHP-type)"  value={`${kc.renal_any_pct}%`}               color={ACCENT5} />
        <KPI label="Hypogonadism"       value={`${kc.hypogonadism_pct}%`}             color={ACCENT4} />
        <KPI label="Cognitive/LD"       value={`${kc.cognitive_pct}%`}               color={ACCENT7} />
        <KPI label="Anosmia"            value={`${kc.anosmia_pct}%`}                 color={ACCENT6} />
      </div>
      <div className="row mb-4">
        <KPI label="CHD"                value={`${kc.chd_pct}%`}                     color={ACCENT4} />
        <KPI label="Retinal End-Stage"  value={`${kc.retinal_endstage_pct}%`}        color={ACCENT4} />
        <KPI label="Misdiagnosis"       value={`${kc.misdiagnosis_pct}%`}            color={ACCENT6} />
        <KPI label="Consanguinity"      value={`${kc.consanguinity_pct}%`}           color={ACCENT6} />
        <KPI label="SLS7 Overlap"       value={`${kc.sls7_overlap_pct}%`}            color={ACCENT5} />
        <KPI label="Protein Size"       value="713 aa"                               color={ACCENT}  />
      </div>

      {/* IF fingerprint — critical diagnostic */}
      <Section title="IF Fingerprint — BBS16 Pattern (Ciliogenesis Defect)" color={ACCENT}>
        <Alert color={ACCENT}>
          <strong>SDCCAG8 ABSENT from pericentriolar satellites</strong> (anti-SDCCAG8 IF — no satellite puncta).{' '}
          <strong>PCM1 satellites DISORGANISED</strong> (scattered; SDCCAG8 tethering lost — not fully absent, but no ordered array).{' '}
          <strong>NPHP1/NPHP4 DELOCALIZED</strong> from ciliary base (satellite delivery failure).{' '}
          <strong>Cilia SHORTENED or ABSENT</strong> on acetylated α-tubulin staining — the only BBS type where cilia fail to form.{' '}
          <strong>BBSome REDUCED at basal body</strong> (no basal body platform; not specifically trapped — contrast BBS15).{' '}
          <strong>MKKS/BCC: NORMAL</strong> (upstream chaperonin intact — BBSome assembles but cannot anchor).{' '}
          <strong>MKS1: REDUCED/DELOCALIZED</strong> (NPHP module disruption; contrast BBS13 where MKS1 itself is absent).
        </Alert>
        <div className="row">
          {[
            { marker: 'SDCCAG8 (pericentriolar satellites)', status: 'ABSENT — no satellite puncta on IF', color: '#b71c1c' },
            { marker: 'PCM1 satellites', status: 'DISORGANISED — scattered (SDCCAG8 tethering lost)', color: '#b71c1c' },
            { marker: 'NPHP1 / NPHP4', status: 'DELOCALIZED from ciliary base (delivery failure)', color: '#b71c1c' },
            { marker: 'Cilia (α-tubulin)', status: 'SHORTENED or ABSENT — ciliogenesis defect', color: '#b71c1c' },
            { marker: 'BBSome (BBS2·BBS4·BBS8·BBS9)', status: 'REDUCED at basal body (no platform)', color: ACCENT },
            { marker: 'MKKS / BCC', status: 'NORMAL (chaperonin intact)', color: ACCENT3 },
            { marker: 'MKS1 (TZ scaffold)', status: 'REDUCED/DELOCALIZED (NPHP module disruption)', color: ACCENT },
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
      <Section title="Molecular Mechanism — Centriolar Satellite Ciliogenesis Failure" color={ACCENT6}>
        <p className="small mb-1">
          <strong>SDCCAG8</strong> (713 aa) localises to PCM1-positive pericentriolar satellites — dynamic
          membrane-less organelles that transport ciliary components (TZ proteins, IFT subunits, BBSome) to the
          basal body during ciliogenesis initiation. SDCCAG8 anchors the NPHP module (NPHP1, NPHP4, NPHP5/IQCB1,
          NPHP8/RPGRIP1L) to these satellites via CC2 (NPHP1 interaction) and its NPHP-module domain (NPHP4/NPHP8 contacts).
        </p>
        <p className="small mb-1">
          <strong>SDCCAG8 LOF consequence:</strong> Satellites lose their NPHP module cargo → TZ proteins (NPHP1,
          NPHP4, MKS1) cannot be delivered to the forming ciliary base → transition zone fails to assemble properly →
          ciliogenesis aborts early or cilia remain stunted. With no ciliary platform, BBSome, IFT-B, and IFT-A
          also fail to localise to the basal body.
        </p>
        <p className="small mb-0">
          <strong>Mechanistic tier:</strong> BBS16 is the most upstream BBS mechanism — it disrupts the very
          process of building a cilium, before BBSome assembly (BBS1-12), TZ gating (BBS13-14), or retrograde
          IFT retrieval (BBS15) become relevant. All downstream ciliary functions fail as a consequence.
        </p>
      </Section>

      {/* BBS16 vs BBS15 vs BBS13 comparison */}
      <Section title="Mechanistic Comparison: BBS16 vs BBS13 vs BBS14 vs BBS15" color={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ background: ACCENT2 + '18' }}>
              <tr>
                <th>Feature</th>
                <th>BBS16 (SDCCAG8)</th>
                <th>BBS13 (MKS1)</th>
                <th>BBS14 (CEP290)</th>
                <th>BBS15 (IFT27)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="fw-semibold">Module</td>
                <td>Centriolar satellite</td>
                <td>TZ scaffold (Y-links)</td>
                <td>TZ Y-link scaffold</td>
                <td>IFT-B1 GTPase</td>
              </tr>
              <tr>
                <td className="fw-semibold">Cilia formed?</td>
                <td style={{ color: '#b71c1c' }}>SHORTENED/ABSENT</td>
                <td>Present (TZ gate leaky)</td>
                <td>Present (TZ gate leaky)</td>
                <td>Present (retrograde trap)</td>
              </tr>
              <tr>
                <td className="fw-semibold">BBSome</td>
                <td style={{ color: '#b71c1c' }}>REDUCED at basal body</td>
                <td style={{ color: ACCENT3 }}>NORMAL IF</td>
                <td style={{ color: ACCENT3 }}>NORMAL IF</td>
                <td style={{ color: ACCENT }}>TRAPPED in cilia (tip↑)</td>
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
                <td style={{ color: ACCENT }}>REDUCED/DELOCALIZED</td>
                <td style={{ color: '#b71c1c' }}>ABSENT (LOF gene)</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
              </tr>
              <tr>
                <td className="fw-semibold">Renal pattern</td>
                <td>NPHP-type (~53%)</td>
                <td>NPHP-type (~52%)</td>
                <td>NPHP-type (~51%)</td>
                <td>Cystic (~35%)</td>
              </tr>
              <tr>
                <td className="fw-semibold">Liver fibrosis</td>
                <td>No</td>
                <td>Yes (~12–15%)</td>
                <td>No</td>
                <td>No</td>
              </tr>
              <tr>
                <td className="fw-semibold">JBTS overlap</td>
                <td>No</td>
                <td>~8% (JBTS28)</td>
                <td>~10% (JBTS5)</td>
                <td>No</td>
              </tr>
              <tr>
                <td className="fw-semibold">Allele-class disease tier</td>
                <td>BBS16 / NPHP10 / SLS7</td>
                <td>BBS13 / JBTS28 / MKS1</td>
                <td>BBS14 / LCA10 / JBTS5 / MKS4</td>
                <td>BBS15 (single tier)</td>
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
  const { data: bd, err: bdErr } = useFetch('/api/bbs16/breakdown');

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
          <Section title="Renal Phenotype Distribution (NPHP-type)" color={ACCENT5}>
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

// ── Tab: Treatment & Diagnostics ──────────────────────────────────────────────
function TreatmentTab() {
  const { data: df, err: dfErr } = useFetch('/api/bbs16/definitions');

  if (dfErr) return <div className="alert alert-danger">Failed to load definitions: {dfErr}</div>;
  if (!df)   return <div className="text-muted p-3">Loading variant data…</div>;

  return (
    <div>
      <Section title="Key SDCCAG8 Variants (BBS16 context)" color={ACCENT}>
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

      <Section title="Diagnostic Workup (BBS16 / SDCCAG8)" color={ACCENT5}>
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
  const { data: df, err: dfErr } = useFetch('/api/bbs16/definitions');

  if (dfErr) return <div className="alert alert-danger">Failed to load definitions: {dfErr}</div>;
  if (!df)   return <div className="text-muted p-3">Loading definitions…</div>;

  const gc = df.gene_card || {};
  const dc = df.disease_card || {};

  return (
    <div>
      <Section title="Gene Card — SDCCAG8 / NPHP10" color={ACCENT}>
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

      <Section title="Disease Card — BBS16" color={ACCENT2}>
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
      <Section title="BBS16 / SDCCAG8 — Centriolar Satellite Module and Unique Position" color={ACCENT}>
        <Alert color={ACCENT}>
          <strong>SDCCAG8 is the first BBS gene classified as a centriolar satellite ciliogenesis factor.</strong>{' '}
          All prior BBS genes (BBS1-15) operate within the cilium (BBSome), at the ciliary base (TZ scaffold),
          within the IFT machinery, or in BBSome pre-assembly (BCC). SDCCAG8/BBS16 acts before cilia form,
          at the PCM1-positive satellite level, delivering TZ components to the basal body. This makes BBS16
          the most upstream BBS mechanism characterised, and uniquely links satellite biology to BBS pathogenesis.
        </Alert>
        <div className="row mt-3">
          {[
            { group: 'BBSome subunits (BBS1-9)', function: 'BBSome structural assembly', examples: 'BBS1, BBS2, BBS4, BBS5, BBS7, BBS8, BBS9', color: ACCENT },
            { group: 'BCC chaperonins (BBS6, BBS10, BBS12)', function: 'BBSome pre-assembly folding', examples: 'MKKS/BBS6, BBS10, BBS12', color: ACCENT3 },
            { group: 'TZ scaffolds (BBS13-14)', function: 'Transition zone Y-link/gate', examples: 'MKS1/BBS13, CEP290/BBS14', color: ACCENT5 },
            { group: 'IFT-B GTPase (BBS15)', function: 'BBSome retrograde retrieval from cilia', examples: 'IFT27/RABL4', color: ACCENT2 },
            { group: 'Centriolar satellite (BBS16)', function: 'Ciliogenesis initiation — TZ protein delivery', examples: 'SDCCAG8/NPHP10 — most upstream mechanism', color: ACCENT },
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
export default function BBS16Page() {
  const [tab, setTab] = useState(0);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🧬 BBS16 — SDCCAG8 / NPHP10 Ciliopathy (Centriolar Satellite)
        </h4>
        <div className="text-muted small">
          Rod-Cone Dystrophy · Post-Axial Polydactyly · Obesity · Learning Disability · Renal NPHP ·
          SDCCAG8 Centriolar Satellite · Chr 1q43-44 · OMIM *613524/#615993 · Also NPHP10/SLS7 ·
          Ciliogenesis Defect · Autosomal Recessive · ~1/10,000,000–25,000,000 (very rare) ·
          Cohort N={_COHORT_SIZE} · seed 363
        </div>
        <div className="mt-1">
          <Badge text="BBS16 *613524" color={ACCENT} />
          <Badge text="SDCCAG8 / NPHP10" color={ACCENT6} />
          <Badge text="1q43-44" color={ACCENT6} />
          <Badge text="Centriolar Satellite" color={ACCENT3} />
          <Badge text="Ciliogenesis Defect" color={ACCENT2} />
          <Badge text="NPHP-type Renal" color={ACCENT5} />
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
