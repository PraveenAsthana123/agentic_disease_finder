'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Cohort & Kinase Activity', 'Treatment & Comparison', 'Definitions'];

// MODY11 colour scheme — teal/cyan (BLK kinase; Src-family; signalling; B-lymphocyte origin)
const ACCENT  = '#006064';   // dark teal — BLK kinase; Src-family; beta-cell signalling
const ACCENT2 = '#1565c0';   // deep blue — genetics; BLK gene; OMIM; 8p23.1
const ACCENT3 = '#2e7d32';   // dark green — SU response; preserved C-peptide; intact beta-cell mass
const ACCENT4 = '#e65100';   // deep orange — T2D misdiagnosis (50%); late onset; BMI overlap
const ACCENT5 = '#4a148c';   // deep violet — PDX1 phosphorylation; KATP-independent pathway
const ACCENT6 = '#37474f';   // dark slate — epidemiology; European/French enrichment
const ACCENT7 = '#00695c';   // teal-green — BLK kinase activity; hypomorphic; partial LOF
const ACCENT8 = '#bf360c';   // red-orange — GWAS convergence; T2D risk locus rs922879

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

export default function MODY11Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/mody11/overview`).then(r => r.json()),
      fetch(`${API}/api/mody11/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mody11/definitions`).then(r => r.json()),
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
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>MODY11 — BLK-MODY</h4>
            <div className="text-muted small">Maturity-Onset Diabetes of the Young Type 11 · Chr 8p23.1 · OMIM #613375 · ~1% MODY · BLK Src-Family Kinase · ↓ PDX1 Phosphorylation · ↓ KATP-Independent GSIS Amplification · C-Peptide Preserved · SU First-Line · Latest Mean Onset · T2D-GWAS Convergence · Autosomal Dominant</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="BLK *191305" color={ACCENT2} />
            <Badge text="Kinase signalling" color={ACCENT7} />
            <Badge text="SU first-line" color={ACCENT3} />
            <Badge text="C-pep preserved" color={ACCENT3} />
            <Badge text="GWAS convergence" color={ACCENT8} />
            <Badge text="European/French" color={ACCENT6} />
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
            <KPI label="Mean C-Peptide" value="Preserved" color={ACCENT3} />
            <KPI label="SU Response" value={kpis.pct_su_response} color={ACCENT3} />
            <KPI label="T2D Misdiag." value={kpis.pct_t2d_misdiag} color={ACCENT4} />
            <KPI label="Mean Dx Age" value={kpis.mean_dx_age} color={ACCENT6} />
            <KPI label="Mean BMI" value={kpis.mean_bmi} color={ACCENT8} />
            <KPI label="Mean 2h PPG" value={kpis.mean_2h_pp_glucose} color={ACCENT5} />
            <KPI label="Family Hx +" value={kpis.pct_family_hx} color={ACCENT6} />
            <KPI label="Antibody Neg." value={kpis.pct_antibody_neg} color={ACCENT3} />
            <KPI label="OMIM Disease" value={kpis.omim_disease} color={ACCENT2} />
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
          <Section title="🔬 MODY11 Mechanism — BLK Kinase Signalling Defect" color={ACCENT}>
            <div className="card border-0 shadow-sm mb-2">
              <div className="card-body">
                <div className="row g-3">
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT }}>BLK → PDX1 Phosphorylation Axis</div>
                    <ol className="small mb-0">
                      <li>BLK phosphorylates PDX1 at <strong>Ser269</strong> → PDX1 nuclear retention</li>
                      <li>Nuclear PDX1 drives transcription of <em>INS</em>, <em>GCK</em>, <em>GLUT2</em></li>
                      <li>BLK haploinsufficiency → hypophosphorylated PDX1 → cytoplasmic export</li>
                      <li>Reduced insulin mRNA + protein content in beta cells</li>
                    </ol>
                  </div>
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT5 }}>BLK → KATP-Independent GSIS Amplification</div>
                    <ol className="small mb-0">
                      <li>Glucose↑ → ATP↑ → K-ATP closes → Ca²⁺ influx → <em>first-phase</em> insulin (BLK-independent)</li>
                      <li>BLK amplifies <em>second-phase</em> via cAMP/PKA/incretin pathway (~30% of total GSIS)</li>
                      <li>BLK LOF → blunted second-phase → prominent <strong>post-prandial hyperglycaemia</strong></li>
                      <li>First-phase partially preserved → fasting glucose less severely affected early</li>
                    </ol>
                  </div>
                </div>
                <div className="alert mt-2 mb-0 small" style={{ background: ACCENT3 + '18', borderLeft: `3px solid ${ACCENT3}` }}>
                  <strong>Key MODY11 insight:</strong> Beta-cell <em>mass is structurally intact</em> — the defect is in GSIS <em>signalling</em>, not beta-cell viability. This is why C-peptide is <strong>preserved</strong> and SU works (60–70% response rate).
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

          {/* BLK mutations */}
          <Section title="🧪 BLK Founding Mutations" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT7 + '18' }}>
                    <th>Mutation</th><th>Domain</th><th>Kinase Activity</th><th>Reference</th><th>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td><strong>A71T (c.211G&gt;A)</strong></td><td>SH2 domain</td><td>~50% WT</td><td>Borowiec 2009 PNAS (French)</td><td>Most common founding; hypomorphic</td></tr>
                  <tr><td><strong>P489L (c.1466C&gt;T)</strong></td><td>Kinase catalytic</td><td>~40% WT</td><td>Borowiec 2009 PNAS (French)</td><td>Stronger phenotype; activation loop</td></tr>
                  <tr><td>K469N (c.1407G&gt;C)</td><td>Kinase ATP-binding</td><td>~55% WT</td><td>European families</td><td>Rare; moderate phenotype</td></tr>
                  <tr><td>E313K (c.937G&gt;A)</td><td>SH2-kinase linker</td><td>~65% WT</td><td>French/UK families</td><td>Mildest MODY11 phenotype</td></tr>
                  <tr><td>L326P (c.977T&gt;C)</td><td>SH3-SH2-kinase</td><td>~50% WT</td><td>European</td><td>Interface disruption</td></tr>
                  <tr><td>Novel_hypomorphic_BLK</td><td>Various</td><td>Variable</td><td>Ongoing discovery</td><td>Kinase assay + PDX1-phospho mandatory</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* T2D GWAS convergence */}
          <Section title="📊 T2D GWAS Convergence — BLK Common + Rare Variants" color={ACCENT8}>
            <div className="card border-0 shadow-sm">
              <div className="card-body">
                <div className="row g-3">
                  <div className="col-md-6">
                    <div className="fw-bold small mb-1" style={{ color: ACCENT8 }}>Common Variant (T2D GWAS)</div>
                    <ul className="small mb-0">
                      <li><strong>rs922879</strong> (BLK promoter; MAF ~15%)</li>
                      <li>Reduces BLK expression by ~20%</li>
                      <li>T2D OR ~1.08 per risk allele</li>
                      <li>European-enriched; present in most T2D GRS tools</li>
                    </ul>
                  </div>
                  <div className="col-md-6">
                    <div className="fw-bold small mb-1" style={{ color: ACCENT }}>Rare Variant (MODY11)</div>
                    <ul className="small mb-0">
                      <li>A71T, P489L coding hypomorphs</li>
                      <li>Reduces BLK kinase activity 40–65%</li>
                      <li>High penetrance (~80%); autosomal dominant</li>
                      <li>MODY11 = rare extreme of same BLK biology axis</li>
                    </ul>
                  </div>
                </div>
                <div className="alert mt-2 mb-0 small" style={{ background: ACCENT8 + '15', borderLeft: `3px solid ${ACCENT8}` }}>
                  BLK is one of the few genes bridging common-variant T2D GWAS and rare-variant monogenic MODY — illustrating the quantitative continuum of beta-cell BLK dosage on glycaemic risk.
                </div>
              </div>
            </div>
          </Section>

          {/* Patient table preview */}
          <Section title="👥 Cohort Preview (first 12 patients, seed=323)" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT6 + '18' }}>
                    <th>#</th><th>Variant</th><th>Dx Age</th><th>HbA1c%</th><th>C-Peptide</th><th>Treatment</th><th>Stage</th><th>FamHx</th>
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
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ─── TAB 1: Cohort & Kinase Activity ─── */}
      {tab === 1 && breakdown && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="BLK Variant Distribution" color={ACCENT}>
              {Object.entries(breakdown.variant_distribution).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="BLK Kinase Activity Tiers" color={ACCENT7}>
              {Object.entries(breakdown.kinase_activity_distribution).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
              <div className="small text-muted mt-1">Moderate hypomorph (46–60% WT) most common — reflects A71T + P489L founders</div>
            </Section>
            <Section title="Age at Diagnosis Tiers (Late Onset)" color={ACCENT6}>
              {Object.entries(breakdown.age_at_diagnosis_tiers).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
              <div className="small text-muted mt-1">Mean onset ~35–45 yr — latest of all MODY types; major T2D overlap</div>
            </Section>
            <Section title="Post-Prandial Glucose (2h) Tiers" color={ACCENT5}>
              {Object.entries(breakdown.pp_glucose_tiers).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
              ))}
              <div className="small text-muted mt-1">Second-phase GSIS blunted → elevated 2h post-prandial glucose</div>
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="C-Peptide Tiers (PRESERVED Pattern)" color={ACCENT3}>
              {Object.entries(breakdown.c_peptide_tiers).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
              <div className="small text-muted mt-1">C-peptide preserved — beta-cell mass intact (kinase signalling defect only)</div>
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
            <Section title="BMI Tiers (T2D Phenotypic Overlap)" color={ACCENT8}>
              {Object.entries(breakdown.bmi_tiers).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
              ))}
              <div className="small text-muted mt-1">Higher BMI vs other MODY types → confounds T2D differential</div>
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
                  <div className="small text-muted mt-1">T2D misdiagnosis ~50% — highest of all MODY types</div>
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

          <Section title="⚖ MODY10 vs MODY11 — Parallel Comparison" color={ACCENT}>
            {definitions?.comparison_mody10_11 && (
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead>
                    <tr style={{ background: ACCENT + '18' }}>
                      <th>Feature</th>
                      {Object.keys(definitions.comparison_mody10_11).map(k => (
                        <th key={k} style={{ color: ACCENT }}>{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {['gene','mechanism','c_peptide','treatment','onset','unique'].map(field => (
                      <tr key={field}>
                        <td className="fw-bold text-capitalize">{field.replace(/_/g, ' ')}</td>
                        {Object.values(definitions.comparison_mody10_11).map((entry, i) => (
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
        <Link href="/mody10" className="btn btn-sm btn-outline-secondary">← MODY10 (INS)</Link>
        <Link href="/mody9" className="btn btn-sm btn-outline-secondary">← MODY9 (PAX4)</Link>
        <Link href="/" className="btn btn-sm btn-outline-primary">🏠 Portal Home</Link>
      </div>
    </div>
  );
}

const _COHORT_SIZE = 40;
