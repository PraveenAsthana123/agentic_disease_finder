'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Treatment & Diagnostics', 'Definitions'];

// BBS15 colour scheme — deep blue (IFT-B/retrograde); deep orange (polydactyly);
// dark green (BBSome assembled/intact entry); dark rose (rod-cone degeneration);
// dark teal (renal cysts); dark slate (epidemiology);
// dark brown (cognitive/LD); burnt orange (obesity/LepR trapped)
const ACCENT  = '#1565c0';   // deep blue — IFT-B complex; retrograde transport; unique mechanism
const ACCENT2 = '#e65100';   // deep orange — polydactyly; post-axial
const ACCENT3 = '#1b5e20';   // dark green — BBSome intact/assembled; GTPase cycle normal to entry
const ACCENT4 = '#880e4f';   // dark rose — rod-cone degeneration
const ACCENT5 = '#006064';   // dark teal — renal cysts; structural
const ACCENT6 = '#37474f';   // dark slate — epidemiology
const ACCENT7 = '#4e342e';   // dark brown — cognitive/LD
const ACCENT8 = '#bf360c';   // burnt orange — obesity; LepR trapped in cilia

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
  const { data: ov, err: ovErr } = useFetch('/api/bbs15/overview');

  if (ovErr) return <div className="alert alert-danger">Failed to load overview: {ovErr}</div>;
  if (!ov)   return <div className="text-muted p-3">Loading overview…</div>;

  const kc = ov.key_counts || {};

  return (
    <div>
      {/* Unique mechanism banner */}
      <Alert color={ACCENT}>
        <strong>BBS15 (IFT27/RABL4) — The ONLY IFT-B Subunit Classified as a BBS Gene.</strong>{' '}
        IFT27 GTP-bound state recruits the BBSome for <strong>retrograde exit from cilia</strong> via BBS3/ARL6 contact.
        IFT27 LOF → BBSome <strong>assembles normally, enters cilia normally</strong>, but{' '}
        <em>CANNOT EXIT retrograde — BBSome TRAPPED in cilia</em> (elevated ciliary tip signal on IF).{' '}
        This is the <strong>OPPOSITE</strong> of most BBS genes (where BBSome fails to assemble or load).
        GPCR cargo (LepR, photoreceptor opsins) accumulates in cilia → multi-system ciliopathy.
        MKKS/BCC intact · MKS1 intact · IFT25/HSPB11 destabilised.
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
        <KPI label="Worldwide Families" value="~30–50"                               color={ACCENT}  />
        <KPI label="Protein Size"       value="225 aa"                               color={ACCENT}  />
      </div>

      {/* IF fingerprint — critical diagnostic */}
      <Section title="IF Fingerprint — BBS15 Pathognomonic Pattern (Retrograde Trap)" color={ACCENT}>
        <Alert color={ACCENT}>
          <strong>BBSome PRESENT at cilia base (assembled normally)</strong> — unlike BBS2/BBS6 where BBSome absent.{' '}
          <strong>BBSome ELEVATED at ciliary tip</strong> — retrograde retrieval failure; ANTI-PHASE to BBS2/BBS6 (pathognomonic for BBS15).{' '}
          <strong>IFT27 ABSENT from IFT-B complex.</strong>{' '}
          <strong>IFT25/HSPB11 DESTABILISED</strong> (monomeric; cleared by proteasome — IFT27 obligate dimer partner lost).{' '}
          <strong>MKKS/BCC: NORMAL</strong> (upstream chaperonin intact).{' '}
          <strong>MKS1: NORMAL</strong> (TZ scaffold intact — no Joubert overlap expected).
        </Alert>
        <div className="row">
          {[
            { marker: 'BBSome (BBS2·BBS4·BBS8·BBS9)', status: 'PRESENT at base / ELEVATED at tip', color: ACCENT },
            { marker: 'IFT27 / RABL4', status: 'ABSENT from IFT-B complex', color: '#b71c1c' },
            { marker: 'IFT25 / HSPB11', status: 'DESTABILISED (monomeric, proteasome-cleared)', color: '#b71c1c' },
            { marker: 'MKKS / BCC', status: 'NORMAL (chaperonin intact)', color: ACCENT3 },
            { marker: 'MKS1 (TZ scaffold)', status: 'NORMAL (no TZ disruption)', color: ACCENT3 },
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
      <Section title="Molecular Mechanism — Retrograde BBSome Retrieval Failure" color={ACCENT6}>
        <p className="small mb-1">
          <strong>IFT27</strong> is a 225 aa small Rab-like GTPase (IFT-B1 subunit) forming an obligate heterodimer
          with IFT25/HSPB11. In its GTP-bound state, IFT27 contacts BBS3/ARL6 (via Switch I/II) to recruit the
          BBSome for retrograde exit from cilia back to the ciliary base. Without this GTPase-mediated retrieval
          signal, the BBSome is <strong>stranded at the ciliary tip</strong>.
        </p>
        <p className="small mb-1">
          <strong>Why GPCR cargo accumulates:</strong> GPCRs that enter cilia (e.g. LepR, opsins, Smoothened)
          normally hitchhike out via the BBSome (retrograde IFT-B). With BBSome trapped, GPCRs cannot exit →
          aberrant ciliary GPCR accumulation → LepR leptin resistance → obesity; photoreceptor opsin
          mislocalisation → rod-cone degeneration (same endpoints as BBSome assembly failure, different mechanism).
        </p>
        <p className="small mb-0">
          <strong>Contrast with BBS2/BBS6:</strong> In BBS2 LOF (BBSome structural subunit), BBSome fails to
          assemble → absent from cilia entirely (tip signal LOW). In BBS15, BBSome assembles normally →
          trapped in cilia (tip signal HIGH). This mechanistic distinction is directly visible on IF and
          is diagnostically pathognomonic.
        </p>
      </Section>

      {/* BBS15 vs BBS14 vs BBS2 comparison */}
      <Section title="Mechanistic Comparison: BBS15 vs BBS14 vs BBS2" color={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ background: ACCENT2 + '18' }}>
              <tr>
                <th>Feature</th>
                <th>BBS15 (IFT27)</th>
                <th>BBS14 (CEP290)</th>
                <th>BBS2 (BBS2 WD40)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="fw-semibold">Module</td>
                <td>IFT-B1 GTPase</td>
                <td>TZ Y-link scaffold</td>
                <td>BBSome structural subunit</td>
              </tr>
              <tr>
                <td className="fw-semibold">BBSome assembly</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
                <td style={{ color: '#b71c1c' }}>FAILS</td>
              </tr>
              <tr>
                <td className="fw-semibold">BBSome in cilia</td>
                <td style={{ color: '#b71c1c' }}>TRAPPED (tip elevated)</td>
                <td style={{ color: ACCENT3 }}>Normal entry/exit (TZ gate leaky)</td>
                <td style={{ color: ACCENT3 }}>Absent (cannot enter)</td>
              </tr>
              <tr>
                <td className="fw-semibold">IFT25/HSPB11</td>
                <td style={{ color: '#b71c1c' }}>DESTABILISED</td>
                <td style={{ color: ACCENT3 }}>Normal</td>
                <td style={{ color: ACCENT3 }}>Normal</td>
              </tr>
              <tr>
                <td className="fw-semibold">MKKS/BCC</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
              </tr>
              <tr>
                <td className="fw-semibold">MKS1</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
                <td style={{ color: ACCENT3 }}>NORMAL</td>
              </tr>
              <tr>
                <td className="fw-semibold">Renal pattern</td>
                <td>Cystic (~35%)</td>
                <td>NPHP (~51%)</td>
                <td>Cystic (~40%)</td>
              </tr>
              <tr>
                <td className="fw-semibold">Liver fibrosis</td>
                <td>No</td>
                <td>No</td>
                <td>No</td>
              </tr>
              <tr>
                <td className="fw-semibold">JBTS overlap</td>
                <td>No</td>
                <td>~10% (JBTS5)</td>
                <td>No</td>
              </tr>
              <tr>
                <td className="fw-semibold">Obesity</td>
                <td>~72% (LepR trapped)</td>
                <td>~65% (TZ gate leaky)</td>
                <td>~78% (BBSome LOF)</td>
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
  const { data: bd, err: bdErr } = useFetch('/api/bbs15/breakdown');

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

// ── Tab: Treatment & Diagnostics ──────────────────────────────────────────────
function TreatmentTab() {
  const { data: df, err: dfErr } = useFetch('/api/bbs15/definitions');

  if (dfErr) return <div className="alert alert-danger">Failed to load definitions: {dfErr}</div>;
  if (!df)   return <div className="text-muted p-3">Loading variant data…</div>;

  return (
    <div>
      <Section title="Key IFT27 Variants (BBS15 context)" color={ACCENT}>
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

      <Section title="Diagnostic Workup (BBS15 / IFT27)" color={ACCENT5}>
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
  const { data: df, err: dfErr } = useFetch('/api/bbs15/definitions');

  if (dfErr) return <div className="alert alert-danger">Failed to load definitions: {dfErr}</div>;
  if (!df)   return <div className="text-muted p-3">Loading definitions…</div>;

  const gc = df.gene_card || {};
  const dc = df.disease_card || {};

  return (
    <div>
      <Section title="Gene Card — IFT27 / RABL4" color={ACCENT}>
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

      <Section title="Disease Card — BBS15" color={ACCENT2}>
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

      {/* IFT-B module context */}
      <Section title="BBS15 / IFT27 — IFT-B Module Context and Unique Position" color={ACCENT}>
        <Alert color={ACCENT}>
          <strong>IFT27 is the ONLY IFT-B subunit classified as a BBS gene.</strong>{' '}
          All other BBS1-14 genes encode: BBSome structural subunits (BBS1-9, BBS18), BCC chaperonins (BBS6/MKKS,
          BBS10, BBS12), or TZ scaffolding proteins (BBS13/MKS1, BBS14/CEP290). IFT27 uniquely bridges the IFT-B
          complex and the BBSome retrograde retrieval machinery, making it the molecular handshake between
          anterograde IFT and retrograde BBSome trafficking.
        </Alert>
        <div className="row mt-3">
          {[
            { group: 'BBSome subunits (BBS1-9)', function: 'BBSome structural assembly', examples: 'BBS1, BBS2, BBS4, BBS5, BBS7, BBS8, BBS9', color: ACCENT },
            { group: 'BCC chaperonins (BBS6, BBS10, BBS12)', function: 'BBSome pre-assembly folding', examples: 'MKKS/BBS6, BBS10, BBS12', color: ACCENT3 },
            { group: 'TZ scaffolds (BBS13-14)', function: 'Transition zone Y-link/gate', examples: 'MKS1/BBS13, CEP290/BBS14', color: ACCENT5 },
            { group: 'IFT-B GTPase (BBS15)', function: 'BBSome retrograde retrieval from cilia', examples: 'IFT27/RABL4 — unique mechanism', color: ACCENT2 },
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
export default function BBS15Page() {
  const [tab, setTab] = useState(0);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🧬 BBS15 — IFT27 / RABL4 Ciliopathy (IFT-B Retrograde)
        </h4>
        <div className="text-muted small">
          Rod-Cone Dystrophy · Post-Axial Polydactyly · Obesity · Learning Disability · Renal Cysts ·
          IFT27 IFT-B Subunit · Chr 22q12.3 · OMIM *615870/#209900 · Ciliopathy ·
          BBSome Retrograde Trapped · Autosomal Recessive · ~1/10,000,000–30,000,000 (very rare) ·
          Cohort N={_COHORT_SIZE} · seed 361
        </div>
        <div className="mt-1">
          <Badge text="BBS15 *615870" color={ACCENT} />
          <Badge text="IFT27 / RABL4" color={ACCENT6} />
          <Badge text="22q12.3" color={ACCENT6} />
          <Badge text="IFT-B1 GTPase" color={ACCENT3} />
          <Badge text="BBSome Retrograde Trapped" color={ACCENT2} />
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
