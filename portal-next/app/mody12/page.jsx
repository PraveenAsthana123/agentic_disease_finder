'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Cohort & K-ATP GOF', 'Treatment & Comparison', 'Definitions'];

// MODY12 colour scheme — deep amber/gold (K-ATP channel; SUR1; sulfonylurea receptor; ion channel gold)
const ACCENT  = '#e65100';   // deep orange-amber — K-ATP GOF; SUR1 activating mutation; channel open
const ACCENT2 = '#1565c0';   // deep blue — genetics; ABCC8 gene; OMIM; 11p15.1
const ACCENT3 = '#2e7d32';   // dark green — SU response (highest MODY); C-peptide preserved; intact beta-cell
const ACCENT4 = '#b71c1c';   // deep red — T1D misdiagnosis; DKA at presentation; K-ATP GOF
const ACCENT5 = '#4a148c';   // deep violet — K-ATP gating mechanism; NBD1/NBD2; channel octamer
const ACCENT6 = '#37474f';   // dark slate — epidemiology; European enrichment; cohort
const ACCENT7 = '#f57f17';   // amber — SUR1 protein; SU binding site; NBD2 direct closure
const ACCENT8 = '#00838f';   // teal-cyan — PNDM2-TNDM-MODY12 spectrum; neonatal history clue

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
        <span>{label}</span><span className="fw-bold">{value}</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function MODY12Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/mody12/overview`).then(r => r.json()),
      fetch(`${API}/api/mody12/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mody12/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  const kpis = overview?.kpis || {};
  const patients = overview?.patients || [];
  const keyFacts = overview?.key_facts || [];
  const alerts = overview?.alerts || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT2}11)`, border: `1px solid ${ACCENT}44` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: '1.6rem' }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>MODY12 — ABCC8-MODY (SUR1)</h4>
            <div className="text-muted small">Maturity-Onset Diabetes of the Young Type 12 · Chr 11p15.1 · OMIM *600509 · ~1–2% MODY · ABCC8/SUR1 K-ATP Regulatory Subunit GOF · ↑ K-ATP Open → ↓ GSIS · C-Peptide Preserved · SU First-Line (Highest MODY Response ~85–90%) · PNDM2-TNDM-MODY12 Spectrum · Autosomal Dominant</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="ABCC8 *600509" color={ACCENT2} />
            <Badge text="SUR1 K-ATP GOF" color={ACCENT} />
            <Badge text="SU 85–90%" color={ACCENT3} />
            <Badge text="C-pep preserved" color={ACCENT3} />
            <Badge text="PNDM spectrum" color={ACCENT8} />
            <Badge text="DKA ~20% at dx" color={ACCENT4} />
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ─── TAB 0: Overview ─── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row g-2 mb-3">
            <KPI label="Gene" value={kpis.gene} color={ACCENT} />
            <KPI label="Chromosome" value={kpis.chromosome} color={ACCENT2} />
            <KPI label="Mean HbA1c" value={kpis.mean_hba1c} color={ACCENT4} />
            <KPI label="C-Peptide" value="Preserved" color={ACCENT3} />
            <KPI label="SU Excellent" value={kpis.pct_excellent_su} color={ACCENT3} />
            <KPI label="T1D Misdiag." value={kpis.pct_t1d_misdiag} color={ACCENT4} />
            <KPI label="Mean Dx Age" value={kpis.mean_dx_age} color={ACCENT6} />
            <KPI label="DKA at Dx" value={kpis.pct_dka_at_dx} color={ACCENT4} />
            <KPI label="Fasting Glu." value={kpis.mean_fasting_glucose} color={ACCENT} />
            <KPI label="Family Hx +" value={kpis.pct_family_hx} color={ACCENT6} />
            <KPI label="Antibody Neg." value={kpis.pct_antibody_neg} color={ACCENT3} />
            <KPI label="OMIM Gene" value={kpis.omim_gene} color={ACCENT2} />
          </div>

          {/* Alerts */}
          <Section title="⚠ Critical Clinical Alerts" color={ACCENT4}>
            {Object.entries(alerts).map(([k, v]) => (
              <Alert key={k} color={ACCENT4}>
                <strong className="text-capitalize">{k.replace(/_/g, ' ')}:</strong> {v}
              </Alert>
            ))}
          </Section>

          {/* Mechanism */}
          <Section title="🔬 MODY12 Mechanism — ABCC8/SUR1 K-ATP Channel GOF" color={ACCENT}>
            <div className="card border-0 shadow-sm mb-2">
              <div className="card-body">
                <div className="row g-3">
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT }}>Normal K-ATP Gating (GSIS)</div>
                    <ol className="small mb-0">
                      <li>Glucose↑ → glycolysis → ATP/ADP ratio rises</li>
                      <li>SUR1 NBDs sense high ATP → K-ATP <strong>closes</strong></li>
                      <li>Membrane depolarises → Ca²⁺ influx → insulin exocytosis (GSIS)</li>
                      <li>Channel octamer: (Kir6.2)₄·(SUR1)₄ — SUR1 is the gating controller</li>
                    </ol>
                  </div>
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT5 }}>MODY12 GOF (ABCC8 activating missense)</div>
                    <ol className="small mb-0">
                      <li>NBD1/NBD2 mutation → SUR1 constitutively MORE active (channel prefers open state)</li>
                      <li>At high glucose: ATP rise FAILS to fully close mutant K-ATP</li>
                      <li>Reduced Ca²⁺ influx → blunted GSIS → post-meal + fasting hyperglycaemia</li>
                      <li>SU binds SUR1 NBD2 <em>directly</em> → closes K-ATP ATP-independently → excellent response</li>
                    </ol>
                  </div>
                </div>
                <div className="alert mt-2 mb-0 small" style={{ background: ACCENT3 + '18', borderLeft: `3px solid ${ACCENT3}` }}>
                  <strong>Key MODY12 insight:</strong> Beta-cell <em>mass is structurally intact</em> — C-peptide is <strong>preserved</strong>. SU response <strong>85–90%</strong> — highest of all MODY types because SU binds the <em>mutated protein directly</em>, completely bypassing the ATP-sensing defect.
                </div>
              </div>
            </div>
          </Section>

          {/* PNDM spectrum */}
          <Section title="🔄 PNDM2 → TNDM → MODY12 Spectrum (ABCC8 GOF Severity)" color={ACCENT8}>
            <div className="card border-0 shadow-sm mb-2">
              <div className="card-body">
                <div className="row g-3">
                  <div className="col-md-4">
                    <div className="fw-bold small mb-1" style={{ color: ACCENT4 }}>Severe GOF → PNDM2</div>
                    <ul className="small mb-0">
                      <li>Onset: &lt; 6 months</li>
                      <li>Channel barely closable</li>
                      <li>Very high SU dose (0.4–0.8 mg/kg)</li>
                      <li>May still need insulin</li>
                      <li>DEND not typical (cf KCNJ11)</li>
                    </ul>
                  </div>
                  <div className="col-md-4">
                    <div className="fw-bold small mb-1" style={{ color: ACCENT8 }}>Moderate GOF → TNDM</div>
                    <ul className="small mb-0">
                      <li>Transient neonatal DM</li>
                      <li>Remits by 18 months</li>
                      <li>Recurs in teens/adult as MODY12</li>
                      <li>Diagnostic clue: neonatal DM history</li>
                      <li>Standard SU effective when recurs</li>
                    </ul>
                  </div>
                  <div className="col-md-4">
                    <div className="fw-bold small mb-1" style={{ color: ACCENT3 }}>Mild GOF → MODY12</div>
                    <ul className="small mb-0">
                      <li>Onset: teens–adult (mean ~25 yr)</li>
                      <li>Channel closable with standard SU</li>
                      <li>85–90% excellent SU response</li>
                      <li>C-peptide fully preserved</li>
                      <li>DKA possible at presentation (~20%)</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </Section>

          {/* Key facts */}
          <Section title="📋 Key Clinical Facts" color={ACCENT2}>
            <div className="row g-2">
              {keyFacts.map((f, i) => (
                <div key={i} className="col-md-6">
                  <div className="small p-2 rounded" style={{ background: ACCENT2 + '0d', borderLeft: `3px solid ${ACCENT2}` }}>
                    {f}
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* ABCC8 mutations table */}
          <Section title="🧪 ABCC8 Founding Mutations (GOF, MODY12 phenotype)" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT7 + '18' }}>
                    <th>Mutation</th><th>Domain</th><th>GOF Severity</th><th>Reference</th><th>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td><strong>R1380L (c.4139G&gt;T)</strong></td><td>NBD2 Walker A adj.</td><td>Mild</td><td>Babenko 2006 NEJM (French)</td><td>Founding mutation; MODY12 pattern</td></tr>
                  <tr><td><strong>R1380C (c.4138C&gt;T)</strong></td><td>NBD2</td><td>Mild-Mod</td><td>Multiple families</td><td>Adjacent to R1380L; similar phenotype</td></tr>
                  <tr><td>K890T (c.2669A&gt;C)</td><td>NBD1 Walker A</td><td>Moderate</td><td>Ellard 2007</td><td>ATP-binding motif; earlier onset</td></tr>
                  <tr><td>H1023R (c.3068A&gt;G)</td><td>NBD1–NBD2 interface</td><td>Moderate</td><td>European families</td><td>Dimerisation surface disruption</td></tr>
                  <tr><td>L1544P (c.4631T&gt;C)</td><td>NBD2 C-terminal</td><td>Mild</td><td>Rare case reports</td><td>Milder phenotype; adult onset</td></tr>
                  <tr><td>V187D (c.560T&gt;A)</td><td>TMD1</td><td>Rare/moderate</td><td>Rare</td><td>Structural TMD domain</td></tr>
                  <tr><td>Novel_ABCC8_GOF</td><td>Various NBD</td><td>Variable</td><td>Ongoing</td><td>86Rb⁺ assay + patch-clamp mandatory</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* Patient table preview */}
          <Section title="👥 Cohort Preview (first 12 patients, seed=325)" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT6 + '18' }}>
                    <th>#</th><th>Variant</th><th>Dx Age</th><th>HbA1c%</th><th>C-Peptide</th><th>Treatment</th><th>Stage</th><th>FamHx</th><th>DKA</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(p => (
                    <tr key={p.id}>
                      <td>{p.id}</td>
                      <td><code style={{ fontSize: '0.72em' }}>{p.variant}</code></td>
                      <td>{p.age_dx}</td>
                      <td>{p.hba1c}</td>
                      <td style={{ color: ACCENT3 }}>{p.c_peptide}</td>
                      <td>{p.treatment}</td>
                      <td>{p.stage.split('(')[0].trim()}</td>
                      <td>{p.family_hx ? '✓' : '—'}</td>
                      <td>{p.dka_at_dx ? <span style={{ color: ACCENT4 }}>DKA</span> : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ─── TAB 1: Cohort & K-ATP GOF ─── */}
      {tab === 1 && breakdown && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="ABCC8 Variant Distribution" color={ACCENT}>
              {Object.entries(breakdown.variant_distribution).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="K-ATP GOF Severity Tiers" color={ACCENT5}>
              {Object.entries(breakdown.katp_gof_distribution).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
              ))}
              <div className="small text-muted mt-1">Mild GOF most common in MODY12 — severe GOF presents as PNDM2</div>
            </Section>
            <Section title="SU Response Distribution" color={ACCENT3}>
              {Object.entries(breakdown.su_response_distribution).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
              <div className="small text-muted mt-1">Excellent SU response ~58% — highest of all MODY types</div>
            </Section>
            <Section title="Age at Diagnosis Tiers" color={ACCENT6}>
              {Object.entries(breakdown.age_at_diagnosis_tiers).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
              <div className="small text-muted mt-1">Earlier onset than MODY7/11; includes paediatric/young adult cases</div>
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="C-Peptide Tiers (PRESERVED Pattern)" color={ACCENT3}>
              {Object.entries(breakdown.c_peptide_tiers).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
              <div className="small text-muted mt-1">C-peptide preserved — K-ATP gating defect; beta-cell mass intact</div>
            </Section>
            <Section title="Ethnicity Distribution" color={ACCENT6}>
              {Object.entries(breakdown.ethnicity_distribution).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
            <Section title="HbA1c Tiers" color={ACCENT4}>
              {Object.entries(breakdown.hba1c_tiers).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
              ))}
            </Section>
            <Section title="Fasting Glucose Tiers" color={ACCENT}>
              {Object.entries(breakdown.fasting_glucose_tiers).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
              <div className="small text-muted mt-1">Moderately elevated fasting — K-ATP GOF blunts all GSIS phases</div>
            </Section>
          </div>
          <div className="col-12">
            <div className="row g-3">
              <div className="col-md-4">
                <Section title="Disease Stage" color={ACCENT4}>
                  {Object.entries(breakdown.disease_stage_distribution).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                  ))}
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="Treatment Distribution" color={ACCENT3}>
                  {Object.entries(breakdown.treatment_distribution).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                  ))}
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="Prior Misdiagnosis" color={ACCENT4}>
                  {Object.entries(breakdown.misdiagnosis_distribution).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                  ))}
                  <div className="small text-muted mt-1">T1D misdiagnosis ~45% — younger onset + DKA confounds</div>
                </Section>
              </div>
            </div>
            <Section title="Summary Flags" color={ACCENT}>
              <div className="row g-2">
                {Object.entries(breakdown.summary_flags || {}).map(([k, v]) => (
                  <div key={k} className="col-6 col-md-3">
                    <div className="card text-center shadow-sm">
                      <div className="card-body py-2">
                        <div className="fw-bold" style={{ color: ACCENT }}>{v}%</div>
                        <div className="small text-muted">{k.replace(/_/g, ' ')}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* ─── TAB 2: Treatment & Comparison ─── */}
      {tab === 2 && (
        <div>
          <Section title="💊 Treatment Strategy" color={ACCENT3}>
            <div className="row g-3">
              {definitions?.treatment && Object.entries(definitions.treatment).map(([k, v]) => (
                <div key={k} className="col-md-6">
                  <div className="card h-100 border-0 shadow-sm">
                    <div className="card-body">
                      <div className="fw-bold small mb-1" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                      <div className="small">{v}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="🔬 Genetics Testing" color={ACCENT2}>
            <div className="row g-3">
              {definitions?.genetics_testing && Object.entries(definitions.genetics_testing).map(([k, v]) => (
                <div key={k} className="col-md-6">
                  <div className="card h-100 border-0 shadow-sm">
                    <div className="card-body">
                      <div className="fw-bold small mb-1" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</div>
                      <div className="small">{v}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="⚖ MODY12 (ABCC8/SUR1) vs MODY13 (KCNJ11/Kir6.2) — K-ATP Subunit Comparison" color={ACCENT}>
            {definitions?.comparison_mody12_13 && (
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead>
                    <tr style={{ background: ACCENT + '18' }}>
                      <th>Feature</th>
                      {Object.keys(definitions.comparison_mody12_13).map(k => (
                        <th key={k} style={{ color: ACCENT }}>{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {['gene','mechanism','c_peptide','treatment','onset','unique'].map(field => (
                      <tr key={field}>
                        <td className="fw-bold text-capitalize">{field.replace(/_/g, ' ')}</td>
                        {Object.values(definitions.comparison_mody12_13).map((entry, i) => (
                          <td key={i}>{entry[field] || '—'}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>

          <Section title="🧬 Lab Thresholds" color={ACCENT7}>
            {definitions?.lab_thresholds && (
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead><tr style={{ background: ACCENT7 + '18' }}><th>Parameter</th><th>Value / Threshold</th></tr></thead>
                  <tbody>
                    {Object.entries(definitions.lab_thresholds).map(([k, v]) => (
                      <tr key={k}><td className="fw-bold">{k}</td><td>{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>
        </div>
      )}

      {/* ─── TAB 3: Definitions ─── */}
      {tab === 3 && definitions && (
        <div>
          <Section title="Disease Definition" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm small">
                <tbody>
                  {Object.entries(definitions.disease || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-nowrap" style={{ color: ACCENT, width: '22%' }}>{k.replace(/_/g, ' ')}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Genes & Proteins" color={ACCENT2}>
            <div className="row g-3">
              {Object.entries(definitions.genes_and_proteins || {}).map(([k, v]) => (
                <div key={k} className="col-md-6">
                  <div className="card h-100 border-0 shadow-sm">
                    <div className="card-body">
                      <div className="fw-bold small mb-1" style={{ color: ACCENT2 }}>{k}</div>
                      <div className="small">{v}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Clinical Terms" color={ACCENT5}>
            <div className="row g-3">
              {Object.entries(definitions.clinical_terms || {}).map(([k, v]) => (
                <div key={k} className="col-md-6">
                  <div className="card h-100 border-0 shadow-sm">
                    <div className="card-body">
                      <div className="fw-bold small mb-1" style={{ color: ACCENT5 }}>{k}</div>
                      <div className="small">{v}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* Footer nav */}
      <div className="mt-4 pt-3 border-top d-flex gap-2 flex-wrap">
        <Link href="/mody11" className="btn btn-sm btn-outline-secondary">← MODY11 (BLK)</Link>
        <Link href="/mody10" className="btn btn-sm btn-outline-secondary">← MODY10 (INS)</Link>
        <Link href="/" className="btn btn-sm btn-outline-primary">🏠 Portal Home</Link>
      </div>
    </div>
  );
}

const _COHORT_SIZE = 40;
