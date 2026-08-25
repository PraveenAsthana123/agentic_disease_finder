'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'DIDMOAD Breakdown', 'Treatment & Diagnostics', 'Definitions'];

// Wolfram / WFS1 colour scheme — deep purple-burgundy (ER stress; multi-organ; severe)
const ACCENT  = '#6a1b9a';   // deep purple — wolframin; ER membrane; multi-organ
const ACCENT2 = '#004d40';   // dark teal — WFS1 gene; OMIM; 4p16.1
const ACCENT3 = '#b71c1c';   // deep red — C-peptide falling; beta-cell apoptosis; ER stress
const ACCENT4 = '#1565c0';   // deep blue — optic atrophy; VEP/OCT; retinal ganglion
const ACCENT5 = '#e65100';   // deep orange — DI; DDAVP; water deprivation; DM onset
const ACCENT6 = '#37474f';   // dark slate — epidemiology; AR inheritance; cohort
const ACCENT7 = '#880e4f';   // dark rose — psychiatric; suicidality; depression
const ACCENT8 = '#1b5e20';   // dark green — treatment; insulin; supportive care

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
        <span>{label}</span><span className="fw-bold">{value}</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function WolframPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/wolfram/overview`).then(r => r.json()),
      fetch(`${API}/api/wolfram/breakdown`).then(r => r.json()),
      fetch(`${API}/api/wolfram/definitions`).then(r => r.json()),
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
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT3}11)`, border: `1px solid ${ACCENT}44` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: '1.6rem' }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>Wolfram Syndrome 1 (WFS1 — DIDMOAD)</h4>
            <div className="text-muted small">Diabetes Insipidus · Diabetes Mellitus · Optic Atrophy · Deafness · WFS1/Wolframin ER Glycoprotein · Chr 4p16.1 · OMIM *606201/#222300 · ER-Stress Multi-Organ Apoptosis · C-Peptide Falls · Autosomal Recessive · ~1/770,000</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="WFS1 *606201" color={ACCENT2} />
            <Badge text="DIDMOAD" color={ACCENT} />
            <Badge text="C-pep FALLS" color={ACCENT3} />
            <Badge text="ER-stress LOF" color={ACCENT3} />
            <Badge text="Autosomal Recessive" color={ACCENT6} />
            <Badge text="Multi-organ" color={ACCENT7} />
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
            <KPI label="Syndrome" value={kpis.syndrome} color={ACCENT} />
            <KPI label="Chromosome" value={kpis.chromosome} color={ACCENT2} />
            <KPI label="Inheritance" value="AR (biallelic)" color={ACCENT6} />
            <KPI label="Mean DM Onset" value={kpis.mean_dm_onset} color={ACCENT5} />
            <KPI label="Mean HbA1c" value={kpis.mean_hba1c} color={ACCENT3} />
            <KPI label="DI Present" value={kpis.pct_di} color={ACCENT5} />
            <KPI label="C-Peptide" value="Falls (ER-stress)" color={ACCENT3} />
            <KPI label="T1D Misdiag." value={kpis.pct_t1d_misdiag} color={ACCENT3} />
            <KPI label="Neuro Features" value={kpis.pct_neuro} color={ACCENT4} />
            <KPI label="Psych Comorbid" value={kpis.pct_psych} color={ACCENT7} />
            <KPI label="OMIM Disease" value={kpis.omim_disease} color={ACCENT2} />
          </div>

          {/* Critical Alerts */}
          <Section title="⚠ Critical Clinical Alerts" color={ACCENT3}>
            {Object.entries(alerts).map(([k, v]) => (
              <Alert key={k} color={k.includes('psych') ? ACCENT7 : k.includes('oa') || k.includes('ophthalm') ? ACCENT4 : ACCENT3}>
                <strong className="text-capitalize">{k.replace(/_/g, ' ')}:</strong> {v}
              </Alert>
            ))}
          </Section>

          {/* Mechanism */}
          <Section title="🔬 Wolfram Syndrome Mechanism — WFS1/Wolframin ER-Stress LOF" color={ACCENT}>
            <div className="card border-0 shadow-sm mb-2">
              <div className="card-body">
                <div className="row g-3">
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT }}>Normal Wolframin Function</div>
                    <ol className="small mb-0">
                      <li>WFS1/Wolframin anchors in ER membrane (9 TM domains; 890 aa)</li>
                      <li>Maintains ER Ca²⁺ homeostasis (SERCA pump regulation)</li>
                      <li>Modulates UPR sensors: IRE1α ubiquitination, PERK signalling</li>
                      <li>Enables ER Ca²⁺ micro-domains → insulin GSIS and neuronal function</li>
                    </ol>
                  </div>
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT3 }}>WFS1 Biallelic LOF → Multi-Organ Apoptosis</div>
                    <ol className="small mb-0">
                      <li>Wolframin absent → ER Ca²⁺ homeostasis lost in ALL wolframin-expressing cells</li>
                      <li>Chronic ER stress → unresolvable UPR → PERK-eIF2α-ATF4-<strong>CHOP</strong> → apoptosis</li>
                      <li>Beta-cells: DM (~6 yr); Retinal ganglion: OA (~11 yr); Hypothalamus: DI (~14 yr)</li>
                      <li>Cochlear: SNHL (~16 yr); Brainstem/cerebellum: neurodegeneration (~20–30 yr)</li>
                    </ol>
                  </div>
                </div>
                <div className="alert mt-2 mb-0 small" style={{ background: ACCENT3 + '18', borderLeft: `3px solid ${ACCENT3}` }}>
                  <strong>C-peptide FALLS</strong> progressively — beta-cell ER-stress apoptosis (CHOP). Absolute insulin dependence from DM diagnosis.
                  Contrasts with <em>all</em> MODY types (C-pep preserved) and is similar to MODY10/INS (same CHOP mechanism, earlier onset, multi-organ).
                </div>
              </div>
            </div>
          </Section>

          {/* DIDMOAD temporal cascade */}
          <Section title="⏱ DIDMOAD Temporal Cascade (approximate mean ages)" color={ACCENT5}>
            <div className="card border-0 shadow-sm mb-2">
              <div className="card-body">
                <div className="row g-2">
                  {[
                    { yr: '~6 yr', feature: 'Diabetes Mellitus (DM)', detail: 'Juvenile-onset; antibody-negative; C-peptide falls; insulin required', color: ACCENT5 },
                    { yr: '~11 yr', feature: 'Optic Atrophy (OA)', detail: 'Bilateral progressive optic neuropathy; RNFL loss on OCT; VEP prolonged', color: ACCENT4 },
                    { yr: '~14 yr', feature: 'Diabetes Insipidus (DI)', detail: 'Central; ADH/AVP deficiency; ~70% of patients; DDAVP effective', color: ACCENT5 },
                    { yr: '~16 yr', feature: 'Deafness (SNHL)', detail: 'Sensorineural; high-frequency; ~65% of patients; audiometry monitoring', color: ACCENT6 },
                    { yr: '~20 yr', feature: 'Psychiatric', detail: 'Depression/suicidality (~25%); psychosis (~10%); anxiety; annual review', color: ACCENT7 },
                    { yr: '~20–30 yr', feature: 'Neurological', detail: 'Cerebellar ataxia; brainstem atrophy; dysarthria; dysphagia; autonomic', color: ACCENT },
                  ].map((item, i) => (
                    <div key={i} className="col-md-4">
                      <div className="card h-100 border-0" style={{ background: item.color + '0d', borderLeft: `4px solid ${item.color}` }}>
                        <div className="card-body py-2 px-2">
                          <div className="fw-bold small" style={{ color: item.color }}>{item.yr} — {item.feature}</div>
                          <div className="small text-muted">{item.detail}</div>
                        </div>
                      </div>
                    </div>
                  ))}
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

          {/* WFS1 mutations table */}
          <Section title="🧪 WFS1 Key Mutations (biallelic LOF — common genotypes)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT + '18' }}>
                    <th>Mutation</th><th>Domain</th><th>Population</th><th>Type</th><th>Severity</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td><strong>p.Leu432Pro (c.1295T{'>'C})</strong></td><td>ER insertion domain</td><td>Lebanese founder</td><td>Missense</td><td>Severe (homozygous)</td></tr>
                  <tr><td><strong>c.1236_1239del (p.His412fs)</strong></td><td>Pre-TM5</td><td>European/pan-ethnic</td><td>Frameshift</td><td>Severe (null)</td></tr>
                  <tr><td>p.Arg558His (c.1673G{'>'A})</td><td>TM6</td><td>European</td><td>Missense</td><td>Severe</td></tr>
                  <tr><td>p.Arg821Cys (c.2461C{'>'T})</td><td>TM8</td><td>Central European</td><td>Missense</td><td>Severe</td></tr>
                  <tr><td>p.Arg456His (c.1367G{'>'A})</td><td>TM4</td><td>Turkish founder</td><td>Missense</td><td>Moderate–severe</td></tr>
                  <tr><td>c.2051dupC (p.Gln684fs)</td><td>TM7</td><td>British</td><td>Frameshift</td><td>Severe (loss of TM8-9)</td></tr>
                  <tr><td>p.Val779Met (c.2335G{'>'A})</td><td>TM8</td><td>Various</td><td>Missense (hypomorphic)</td><td>Mild (later onset)</td></tr>
                  <tr><td>Splice site variants</td><td>Various introns</td><td>Pan-ethnic</td><td>Splice-site</td><td>Variable (partial retention)</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* Patient table preview */}
          <Section title="👥 Cohort Preview (first 12 patients, seed=329)" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT6 + '18' }}>
                    <th>#</th><th>Mutation (allele 1)</th><th>Features</th><th>DM Onset</th><th>HbA1c%</th><th>C-Pep</th><th>OA Stage</th><th>Hearing</th><th>DKA@Dx</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(p => (
                    <tr key={p.id}>
                      <td>{p.id}</td>
                      <td><code style={{ fontSize: '0.68em' }}>{p.mutation}</code></td>
                      <td><Badge text={p.features} color={ACCENT} /></td>
                      <td>{p.dm_onset}</td>
                      <td>{p.hba1c}</td>
                      <td style={{ color: ACCENT3 }}>{p.c_peptide}</td>
                      <td><span className="small">{p.oa_stage}</span></td>
                      <td><span className="small">{p.hearing}</span></td>
                      <td>{p.dka_at_dx ? <span style={{ color: ACCENT3 }}>DKA</span> : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ─── TAB 1: DIDMOAD Breakdown ─── */}
      {tab === 1 && breakdown && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="DIDMOAD Feature Distribution" color={ACCENT}>
              {Object.entries(breakdown.feature_distribution).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
              <div className="small text-muted mt-1">DM+OA+DI+D (full DIDMOAD) most common in advanced cohort</div>
            </Section>
            <Section title="WFS1 Mutation Distribution" color={ACCENT}>
              {Object.entries(breakdown.mutation_distribution).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="Optic Atrophy Stage" color={ACCENT4}>
              {Object.entries(breakdown.oa_stage_distribution).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
              ))}
              <div className="small text-muted mt-1">Progressive bilateral optic neuropathy; ~17% severe/profound</div>
            </Section>
            <Section title="DM Onset Tiers" color={ACCENT5}>
              {Object.entries(breakdown.dm_onset_tiers).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
              ))}
              <div className="small text-muted mt-1">Mean ~6 yr; earlier than ALL MODY types; juvenile-onset</div>
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="C-Peptide Tiers (FALLING Pattern)" color={ACCENT3}>
              {Object.entries(breakdown.c_peptide_tiers).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
              <div className="small text-muted mt-1">Falls progressively — ER-stress beta-cell apoptosis (CHOP); absolute insulin dependence</div>
            </Section>
            <Section title="HbA1c Tiers" color={ACCENT3}>
              {Object.entries(breakdown.hba1c_tiers).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
            <Section title="Neurological Status" color={ACCENT}>
              {Object.entries(breakdown.neurological_distribution).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="Psychiatric Comorbidity" color={ACCENT7}>
              {Object.entries(breakdown.psychiatric_distribution).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
              <div className="small text-muted mt-1">Depression with suicidality ~8%; annual psychiatric assessment mandatory</div>
            </Section>
          </div>
          <div className="col-12">
            <div className="row g-3">
              <div className="col-md-3">
                <Section title="Ethnicity" color={ACCENT6}>
                  {Object.entries(breakdown.ethnicity_distribution).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                  ))}
                </Section>
              </div>
              <div className="col-md-3">
                <Section title="Hearing Status" color={ACCENT6}>
                  {Object.entries(breakdown.hearing_distribution).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                  ))}
                </Section>
              </div>
              <div className="col-md-3">
                <Section title="Renal Status" color={ACCENT5}>
                  {Object.entries(breakdown.renal_distribution).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
                  ))}
                </Section>
              </div>
              <div className="col-md-3">
                <Section title="Insulin Delivery" color={ACCENT8}>
                  {Object.entries(breakdown.insulin_delivery).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                  ))}
                </Section>
              </div>
            </div>
            <div className="row g-3">
              <div className="col-md-4">
                <Section title="Prior Misdiagnosis" color={ACCENT3}>
                  {Object.entries(breakdown.misdiagnosis_distribution).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                  ))}
                  <div className="small text-muted mt-1">T1D most common — antibody-negative juvenile DM + OA = Wolfram first</div>
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="Disease Duration" color={ACCENT6}>
                  {Object.entries(breakdown.disease_duration_tiers).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                  ))}
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="Summary Flags" color={ACCENT}>
                  {Object.entries(breakdown.summary_flags || {}).map(([k, v]) => (
                    <div key={k} className="d-flex justify-content-between small mb-1 p-1 rounded" style={{ background: ACCENT + '0d' }}>
                      <span>{k.replace(/_/g, ' ')}</span>
                      <span className="fw-bold" style={{ color: ACCENT }}>{v}%</span>
                    </div>
                  ))}
                </Section>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ─── TAB 2: Treatment & Diagnostics ─── */}
      {tab === 2 && definitions && (
        <div>
          <Section title="💊 Treatment Strategy (Multidisciplinary / Supportive)" color={ACCENT8}>
            <div className="row g-3">
              {definitions.treatment && Object.entries(definitions.treatment).map(([k, v]) => (
                <div key={k} className="col-md-6">
                  <div className="card h-100 border-0 shadow-sm">
                    <div className="card-body">
                      <div className="fw-bold small mb-1" style={{ color: ACCENT8 }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                      <div className="small">{v}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="🔬 Diagnostics" color={ACCENT2}>
            <div className="row g-3">
              {definitions.diagnostics && Object.entries(definitions.diagnostics).map(([k, v]) => (
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

          <Section title="⚖ Wolfram 1 (WFS1) vs MODY10 (INS) — ER-Stress Apoptosis Comparison" color={ACCENT3}>
            {definitions.comparison_mody10_wolfram && (
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead>
                    <tr style={{ background: ACCENT3 + '18' }}>
                      <th>Feature</th>
                      {Object.keys(definitions.comparison_mody10_wolfram).map(k => (
                        <th key={k} style={{ color: ACCENT3 }}>{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {['gene', 'mechanism', 'onset', 'c_peptide', 'inheritance', 'features', 'treatment'].map(field => (
                      <tr key={field}>
                        <td className="fw-bold text-capitalize">{field.replace(/_/g, ' ')}</td>
                        {Object.values(definitions.comparison_mody10_wolfram).map((entry, i) => (
                          <td key={i}>{entry[field] || '—'}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>

          <Section title="🧪 Lab Thresholds" color={ACCENT5}>
            {definitions.lab_thresholds && (
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead><tr style={{ background: ACCENT5 + '18' }}><th>Parameter</th><th>Value / Threshold</th></tr></thead>
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
        <Link href="/mody13" className="btn btn-sm btn-outline-secondary">← MODY13 (KCNJ11)</Link>
        <Link href="/mody12" className="btn btn-sm btn-outline-secondary">← MODY12 (ABCC8)</Link>
        <Link href="/" className="btn btn-sm btn-outline-primary">🏠 Portal Home</Link>
      </div>
    </div>
  );
}
