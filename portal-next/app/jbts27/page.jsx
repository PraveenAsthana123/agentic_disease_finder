'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'ARMC9 ARM-Platform Pearls', 'Definitions'];

// JBTS27 colour scheme — ARMC9 / Centriolar Satellite / ARM-Repeat / TOGARAM1 Axis / CP110-TTBK2
// Deep teal ARM-platform tones; TOGARAM1 amber; TTBK2/CP110 crimson; satellite indigo
const ACCENT   = '#00695c';   // deep teal — ARMC9 ARM-repeat scaffold / centriolar satellite
const ACCENT2  = '#00838f';   // cyan-teal — ARM repeat core / TOGARAM1 binding
const ACCENT3  = '#1b5e20';   // forest green — cerebellar / neurological
const ACCENT4  = '#0277bd';   // sky blue — renal NPHP-like absent/short tubular cilia
const ACCENT5  = '#37474f';   // slate — domain matrix / tables
const ACCENT6  = '#c62828';   // crimson — CP110 persistence / cilia absent (diagnostic)
const ACCENT7  = '#e65100';   // deep orange — TTBK2 co-recruitment failure / initiation defect
const ACCENT8  = '#ef6c00';   // amber — TOGARAM1 axis / axoneme elongation
const ACCENT9  = '#558b2f';   // olive — hepatic CHF
const ACCENT10 = '#7b1fa2';   // purple — retinal rod-cone absent/short connecting cilia

const SEED = 467;
const N_COHORT = 40;

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
    <div className="alert mb-3" style={{ background: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 6 }}>
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

function Bar({ label, value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{value}</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function JBTS27Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts27/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts27/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts27/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading ARMC9 / JBTS27 dashboard…</p></div>;
  if (error)   return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  const kpi = overview?.kpis || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT2} 60%, ${ACCENT8} 100%)`, color: '#fff' }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div className="fs-1">🧬</div>
          <div>
            <h4 className="mb-0 fw-bold">JBTS27 — ARMC9 Joubert Syndrome Type 27</h4>
            <div className="small opacity-90">
              ARMC9 · Armadillo Repeat-Containing Protein 9 · Centriolar Satellite ARM-Repeat Platform · TOGARAM1 Coupler · CP110–TTBK2 Ciliogenesis Axis
            </div>
            <div className="small opacity-80 mt-1">
              2q37.1 · ~951 aa · OMIM Gene *616948 · Disease #617120 · No MKS Tier · Cilia ABSENT (null) or SHORT (hypomorphic) · MENA Founder Leu247Pro
            </div>
          </div>
          <div className="ms-auto text-end small opacity-80">
            <div>Seed {SEED} · N={N_COHORT}</div>
            <div>AR · Biallelic LOF</div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && overview && (
        <div>
          {/* KPI row */}
          <div className="row g-2 mb-3">
            <KPI label="Cohort (N)" value={kpi.total_patients} color={ACCENT} />
            <KPI label="MTS %" value={`${kpi.mts_pct}%`} color={ACCENT} />
            <KPI label="Ataxia" value={`${kpi.ataxia_pct}%`} color={ACCENT3} />
            <KPI label="Hypotonia" value={`${kpi.hypotonia_pct}%`} color={ACCENT3} />
            <KPI label="OMA" value={`${kpi.oma_pct}%`} color={ACCENT3} />
            <KPI label="Breathing" value={`${kpi.breathing_pct}%`} color={ACCENT3} />
            <KPI label="Retinal" value={`${kpi.retinal_pct}%`} color={ACCENT10} />
            <KPI label="Renal" value={`${kpi.renal_pct}%`} color={ACCENT4} />
            <KPI label="Hepatic" value={`${kpi.hepatic_pct}%`} color={ACCENT9} />
            <KPI label="Polydactyly" value={`${kpi.poly_pct}%`} color={ACCENT5} />
            <KPI label="ID" value={`${kpi.id_pct}%`} color={ACCENT2} />
            <KPI label="Cilia Absent" value={`${kpi.cilia_absent_pct}%`} color={ACCENT6} />
          </div>

          {/* Alerts */}
          {overview.alerts && (
            <div className="mb-3">
              <Alert color={ACCENT}>
                <strong>🛰️ Centriolar Satellite ARM-Platform Mechanism:</strong>{' '}
                {overview.alerts.centriolar_satellite_arm_platform}
              </Alert>
              <Alert color={ACCENT8}>
                <strong>🔗 ARMC9 — TOGARAM1 (JBTS35) Axis:</strong>{' '}
                {overview.alerts.togaram1_axis}
              </Alert>
              <Alert color={ACCENT7}>
                <strong>🔬 CP110–TTBK2 Ciliogenesis Axis:</strong>{' '}
                {overview.alerts.cp110_ttbk2_axis}
              </Alert>
              <Alert color={ACCENT5}>
                <strong>🌍 MENA &amp; European Founder Clusters:</strong>{' '}
                {overview.alerts.mena_european_clusters}
              </Alert>
            </div>
          )}

          {/* Key facts */}
          <Section title="ARMC9 / JBTS27 — Key Clinical Facts" color={ACCENT}>
            <ul className="small mb-0">
              {(overview.key_facts || []).map((f, i) => (
                <li key={i} className="mb-1">{f}</li>
              ))}
            </ul>
          </Section>

          {/* Patient table */}
          <Section title={`Patient Cohort (N=${N_COHORT}, seed ${SEED})`} color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small">
                <thead>
                  <tr>
                    <th>ID</th><th>Age</th><th>Sex</th><th>Ethnicity</th>
                    <th>Allele Class</th><th>Variant</th>
                    <th>MTS</th><th>Ataxia</th><th>Hypotonia</th><th>OMA</th>
                    <th>Breathing</th><th>Retinal</th><th>Renal</th><th>Hepatic</th><th>Poly</th><th>ID</th><th>ESRD</th><th>Cilia¬</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.patients || []).map(p => (
                    <tr key={p.id}>
                      <td className="fw-bold">{p.id}</td>
                      <td>{p.age}</td>
                      <td>{p.sex}</td>
                      <td className="small">{p.ethnicity}</td>
                      <td className="small">{p.allele_class}</td>
                      <td className="small font-monospace">{p.variant}</td>
                      <td>{p.mts ? '✅' : '—'}</td>
                      <td>{p.ataxia ? '✅' : '—'}</td>
                      <td>{p.hypotonia ? '✅' : '—'}</td>
                      <td>{p.oma ? '✅' : '—'}</td>
                      <td>{p.breathing ? '✅' : '—'}</td>
                      <td>{p.retinal ? '✅' : '—'}</td>
                      <td>{p.renal ? '✅' : '—'}</td>
                      <td>{p.hepatic ? '✅' : '—'}</td>
                      <td>{p.poly ? '✅' : '—'}</td>
                      <td>{p.id_flag ? '✅' : '—'}</td>
                      <td>{p.esrd ? '✅' : '—'}</td>
                      <td style={{ color: ACCENT6 }}>{p.cilia_absent ? 'Absent' : 'Short'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="small text-muted">Cilia¬ = cilia absent (biallelic null) vs short (hypomorphic missense). MTS = Molar Tooth Sign. OMA = oculomotor apraxia. ID = intellectual disability.</div>
          </Section>
        </div>
      )}

      {/* ── TAB 1: Diagnostic Breakdown ── */}
      {tab === 1 && breakdown && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Ethnicity Distribution" color={ACCENT}>
              {breakdown.ethnicity_distribution.map((e, i) => (
                <Bar key={i} label={e.ethnicity} value={e.count} max={N_COHORT} color={ACCENT} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Allele Class Distribution" color={ACCENT2}>
              {breakdown.allele_class_distribution.map((a, i) => (
                <Bar key={i} label={a.allele_class} value={a.count} max={N_COHORT} color={ACCENT2} />
              ))}
            </Section>
          </div>
          <div className="col-12">
            <Section title="Phenotype Summary" color={ACCENT3}>
              <div className="row">
                {Object.entries(breakdown.phenotype_summary || {}).map(([key, val]) => (
                  <div key={key} className="col-6 col-md-3 mb-2">
                    <div className="card shadow-sm text-center p-2">
                      <div className="fw-bold" style={{ color: ACCENT }}>{val.n} / {N_COHORT}</div>
                      <div className="small text-muted text-capitalize">{key.replace(/_/g, ' ')}</div>
                      <div className="small text-muted">{val.pct}%</div>
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          </div>
          <div className="col-12">
            <Section title="Notable Variants (ARMC9 — JBTS27 alleles)" color={ACCENT6}>
              {(breakdown.notable_variants || []).map((v, i) => (
                <div key={i} className="card mb-2 shadow-sm">
                  <div className="card-body py-2">
                    <div className="d-flex flex-wrap gap-2 align-items-center mb-1">
                      <span className="fw-bold font-monospace" style={{ color: ACCENT6 }}>{v.name}</span>
                      <span className="badge" style={{ background: ACCENT + '22', color: ACCENT }}>{v.cdna}</span>
                      <span className="badge bg-secondary">{v.severity}</span>
                      <span className="badge" style={{ background: ACCENT8 + '22', color: ACCENT8 }}>{v.population}</span>
                    </div>
                    <div className="small text-muted mb-1"><strong>Domain:</strong> {v.domain}</div>
                    <div className="small">{v.mechanism}</div>
                  </div>
                </div>
              ))}
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 2: ARMC9 ARM-Platform Pearls ── */}
      {tab === 2 && definitions && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="ARMC9 LOF Pathway (JBTS27)" color={ACCENT7}>
              <div className="card shadow-sm p-3 small">
                <div className="mb-2"><strong style={{ color: ACCENT6 }}>Step 1:</strong> ARMC9 biallelic LOF → centriolar satellite function impaired → TTBK2 recruitment to DA outer rim unstable</div>
                <div className="mb-2"><strong style={{ color: ACCENT7 }}>Step 2:</strong> TTBK2 unstable at DA → MPP9 phosphorylation reduced → CP110-CEP97 cap persists at basal body → cilia initiation fails (cilia ABSENT in null alleles)</div>
                <div className="mb-2"><strong style={{ color: ACCENT8 }}>Step 3:</strong> ARMC9–TOGARAM1 complex disrupted (ARM 7–10 groove) → axoneme elongation impaired → cilia short even when they do initiate</div>
                <div className="mb-2"><strong style={{ color: ACCENT2 }}>Step 4:</strong> TULP3–dynein-2 scaffold (ARM 13–15) impaired → retrograde IFT-A cargo loading partially reduced</div>
                <div className="mb-2"><strong style={{ color: ACCENT3 }}>Step 5:</strong> Absent/short cilia → SMO exclusion → Hedgehog abolished → cerebellar granule cell proliferation severely impaired → Molar Tooth Sign</div>
                <div className="text-muted" style={{ borderTop: '1px solid #eee', paddingTop: 8, marginTop: 8 }}>
                  <strong>Key EM/IF:</strong> DA structure INTACT; CP110 PERSISTS at basal body (diagnostic IF marker); TZ B9-gate INTACT in residual cilia; IFT-B subunits: reduced transport (not base or tip accumulation specific).
                </div>
              </div>
            </Section>

            <Section title="ARMC9 Domain Map (~951 aa)" color={ACCENT}>
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead><tr><th>Region</th><th>aa</th><th>Function</th><th>Key Pathogenic Site</th></tr></thead>
                  <tbody>
                    <tr><td>IDR / CSTS</td><td>1–200</td><td>Centriolar satellite targeting; PCM1/SSX2IP binding; NLS</td><td>Leu247Pro (IDR/ARM1 boundary) — satellite targeting + ARM1-4 TOGARAM1 contact</td></tr>
                    <tr><td>ARM 1–6</td><td>201–480</td><td>CCDC66/TOGARAM1 N-terminal contact; ARM scaffold platform</td><td>Val389Gly — ARM-4/5 groove; TOGARAM1 N-CC docking −60%</td></tr>
                    <tr><td>ARM 7–12</td><td>481–720</td><td>Primary TOGARAM1-binding surface; ARM inner groove</td><td>Pro654Leu (ARM 11) −35–50%; Gly710Asp (ARM 12/13 junction) −45%</td></tr>
                    <tr><td>ARM 13–15</td><td>721–780</td><td>TULP3 scaffold; dynein-2 bridge; KIFC1 MT tether</td><td>Gly710Asp allosteric strain</td></tr>
                    <tr><td>SDA anchor</td><td>781–900</td><td>Subdistal appendage matrix; ODF2/CETN2 contact; platform</td><td>Glu877Ter — removes SDA anchor C-terminus + full TTBK2 module</td></tr>
                    <tr><td>TTBK2-CC</td><td>901–951</td><td>TTBK2 co-recruitment; CP110 removal enabler; SDA outer rim</td><td>Glu877Ter truncating — TTBK2 co-recruitment abolished</td></tr>
                  </tbody>
                </table>
              </div>
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="DDx: JBTS27 vs Related Ciliopathies" color={ACCENT5}>
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead>
                    <tr><th>Feature</th><th>JBTS27 ARMC9</th><th>JBTS26 KIAA0556</th><th>JBTS35 TOGARAM1</th><th>JBTS22 CEP83</th></tr>
                  </thead>
                  <tbody>
                    <tr><td><strong>Cilia status</strong></td><td style={{ color: ACCENT6 }}>Absent (null) / Short (hypo)</td><td>Short (always present)</td><td>Short (elongation fail)</td><td>Absent (no DA)</td></tr>
                    <tr><td><strong>Mechanism</strong></td><td>CP110 removal failure (TTBK2 recruit.)</td><td>IFT-B base adapter failure</td><td>TOG axoneme elongation</td><td>DA foundation absent</td></tr>
                    <tr><td><strong>DA structure</strong></td><td>Intact</td><td>Intact</td><td>Intact</td><td>Absent</td></tr>
                    <tr><td><strong>CP110 IF</strong></td><td style={{ color: ACCENT6 }}>Persists (diagnostic)</td><td>Cleared (normal)</td><td>Cleared (normal)</td><td>Persists (no DA)</td></tr>
                    <tr><td><strong>IFT-B EM</strong></td><td>Reduced transport</td><td>BASE accumulation</td><td>TIP accumulation</td><td>Absent cilia</td></tr>
                    <tr><td><strong>TOGARAM1 link</strong></td><td style={{ color: ACCENT8 }}>Direct ARMC9–TOGARAM1 complex</td><td>No direct link</td><td>Same axis (TOGARAM1 = JBTS35)</td><td>No direct link</td></tr>
                    <tr><td><strong>MKS tier</strong></td><td>No</td><td>No</td><td>No</td><td>No</td></tr>
                    <tr><td><strong>Polydactyly</strong></td><td>~18% post-axial</td><td>~12% post-axial</td><td>~15%</td><td>~8%</td></tr>
                    <tr><td><strong>Renal</strong></td><td>~18% NPHP-like</td><td>~15% NPHP-like</td><td>~15%</td><td>~12%</td></tr>
                    <tr><td><strong>Retinal</strong></td><td>~25% rod-cone</td><td>~22% rod-cone</td><td>~22%</td><td>~10%</td></tr>
                  </tbody>
                </table>
              </div>
            </Section>

            <Section title="Surveillance Protocol" color={ACCENT4}>
              <div className="small">
                {Object.entries(definitions.surveillance_protocol || {}).map(([key, val]) => (
                  <div key={key} className="mb-2 p-2 rounded" style={{ background: ACCENT4 + '11' }}>
                    <strong className="text-capitalize">{key.replace(/_/g, ' ')}:</strong> {val}
                  </div>
                ))}
              </div>
            </Section>

            <Section title="Founder Variants" color={ACCENT8}>
              {(definitions.founder_variants || []).map((fv, i) => (
                <div key={i} className="card mb-2 shadow-sm">
                  <div className="card-body py-2 small">
                    <div className="fw-bold font-monospace" style={{ color: ACCENT8 }}>{fv.variant}</div>
                    <div><strong>Population:</strong> {fv.population}</div>
                    <div><strong>Frequency:</strong> {fv.frequency}</div>
                    <div><strong>Domain:</strong> {fv.domain}</div>
                    <div><strong>Severity:</strong> {fv.severity}</div>
                  </div>
                </div>
              ))}
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && definitions && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Gene &amp; Disease Identity" color={ACCENT}>
              <table className="table table-sm small">
                <tbody>
                  <tr><td><strong>Gene</strong></td><td>{definitions.gene_full_name}</td></tr>
                  <tr><td><strong>OMIM Gene</strong></td><td>*{definitions.omim_gene}</td></tr>
                  <tr><td><strong>OMIM Disease</strong></td><td>#{definitions.omim_jbts27}</td></tr>
                  <tr><td><strong>Chromosome</strong></td><td>{definitions.chromosome}</td></tr>
                  <tr><td><strong>Protein Size</strong></td><td>{definitions.protein_size}</td></tr>
                  <tr><td><strong>Inheritance</strong></td><td>{definitions.inheritance}</td></tr>
                  <tr><td><strong>MKS Tier</strong></td><td>{definitions.mks_tier ? '⚠ YES — lethal tier' : '✅ NO — all liveborn'}</td></tr>
                  <tr><td><strong>Frequency</strong></td><td>{definitions.frequency}</td></tr>
                </tbody>
              </table>
            </Section>
            <Section title="Mechanism Class" color={ACCENT7}>
              <div className="small mb-2"><strong>{definitions.mechanism_class}</strong></div>
              <div className="small">{definitions.mechanism_detail}</div>
            </Section>
            <Section title="Cilia Phenotype" color={ACCENT6}>
              <div className="small mb-1"><strong>Status:</strong> {definitions.cilia_phenotype}</div>
              <div className="small mb-1"><strong>Hedgehog Impact:</strong> {definitions.hedgehog_impact}</div>
              <div className="small"><strong>MTS Mechanism:</strong> {definitions.mts_mechanism}</div>
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Key Differential Diagnoses" color={ACCENT5}>
              <ul className="small mb-0">
                {(definitions.key_ddx || []).map((d, i) => (
                  <li key={i} className="mb-2">{d}</li>
                ))}
              </ul>
            </Section>
            <Section title="Treatment Notes" color={ACCENT3}>
              <div className="small">
                {Object.entries(definitions.treatment || {}).map(([key, val]) => (
                  <div key={key} className="mb-2">
                    <strong className="text-capitalize">{key.replace(/_/g, ' ')}:</strong> {val}
                  </div>
                ))}
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* Footer nav */}
      <div className="mt-4 pt-3 border-top d-flex gap-3 flex-wrap small text-muted">
        <Link href="/jbts26" className="text-decoration-none">← JBTS26 KIAA0556</Link>
        <span className="mx-2">|</span>
        <Link href="/" className="text-decoration-none">🏠 Portal Home</Link>
        <span className="mx-2">|</span>
        <span>JBTS27 ARMC9 · Seed {SEED} · N={N_COHORT} · 2q37.1 · OMIM #617120</span>
      </div>
    </div>
  );
}
