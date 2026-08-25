'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-Organ Breakdown', 'Treatment & Diagnostics', 'Definitions'];

// Alström / ALMS1 colour scheme — amber-orange-teal (ciliopathy; DCM; cone-rod dystrophy)
const ACCENT  = '#e65100';   // deep orange — ALMS1 ciliopathy; cone-rod dystrophy; DCM
const ACCENT2 = '#006064';   // dark teal — ALMS1 gene; 2p13.1; OMIM
const ACCENT3 = '#1b5e20';   // dark green — C-peptide preserved; insulin resistance; NOT falling
const ACCENT4 = '#4a148c';   // deep purple — retinal cone-rod dystrophy; vision loss
const ACCENT5 = '#bf360c';   // burnt orange — cardiomyopathy; DCM; cardiac
const ACCENT6 = '#37474f';   // dark slate — epidemiology; AR inheritance; cohort
const ACCENT7 = '#880e4f';   // dark rose — renal/hepatic; CKD; NASH; fibrosis
const ACCENT8 = '#004d40';   // deep teal — treatment; metformin; GLP-1RA

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

export default function AlstromPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/alstrom/overview`).then(r => r.json()),
      fetch(`${API}/api/alstrom/breakdown`).then(r => r.json()),
      fetch(`${API}/api/alstrom/definitions`).then(r => r.json()),
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
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT5}11)`, border: `1px solid ${ACCENT}44` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: '1.6rem' }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>Alström Syndrome (ALMS1)</h4>
            <div className="text-muted small">Cone-Rod Dystrophy · Sensorineural Deafness · Dilated Cardiomyopathy · Obesity · T2D-like · ALMS1 Ciliary Protein · Chr 2p13.1 · OMIM *606844/#203800 · Ciliopathy · C-Peptide Preserved · Autosomal Recessive · ~1/1,000,000</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="ALMS1 *606844" color={ACCENT2} />
            <Badge text="Ciliopathy" color={ACCENT} />
            <Badge text="C-pep PRESERVED" color={ACCENT3} />
            <Badge text="DCM infantile" color={ACCENT5} />
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
            <KPI label="Chromosome" value={kpis.chromosome} color={ACCENT2} />
            <KPI label="Inheritance" value="AR (biallelic)" color={ACCENT6} />
            <KPI label="Mean DM Onset" value={kpis.mean_dm_onset} color={ACCENT} />
            <KPI label="Mean HbA1c" value={kpis.mean_hba1c} color={ACCENT} />
            <KPI label="Mean BMI" value={kpis.mean_bmi} color={ACCENT} />
            <KPI label="DCM (any)" value={kpis.pct_dcm} color={ACCENT5} />
            <KPI label="Legal Blindness" value={kpis.pct_legal_blind} color={ACCENT4} />
            <KPI label="CKD" value={kpis.pct_ckd} color={ACCENT7} />
            <KPI label="C-Peptide" value="Preserved/High" color={ACCENT3} />
            <KPI label="T1D Misdiag." value={kpis.pct_t1d_misdiag} color={ACCENT} />
            <KPI label="OMIM Disease" value={kpis.omim_disease} color={ACCENT2} />
          </div>

          {/* Critical Alerts */}
          <Section title="⚠ Critical Clinical Alerts" color={ACCENT5}>
            {Object.entries(alerts).map(([k, v]) => (
              <Alert key={k} color={k.includes('wolfram') ? ACCENT2 : k.includes('dcm') || k.includes('cardiac') ? ACCENT5 : k.includes('retinal') || k.includes('cone') ? ACCENT4 : k.includes('c_peptide') ? ACCENT3 : ACCENT}>
                <strong className="text-capitalize">{k.replace(/_/g, ' ')}:</strong> {v}
              </Alert>
            ))}
          </Section>

          {/* Mechanism */}
          <Section title="🔬 Alström Syndrome Mechanism — ALMS1 Ciliopathy / Primary Cilia Dysfunction" color={ACCENT}>
            <div className="card border-0 shadow-sm mb-2">
              <div className="card-body">
                <div className="row g-3">
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT }}>Normal ALMS1 Function</div>
                    <ol className="small mb-0">
                      <li>ALMS1 localises to centrosome, basal body, and ciliary axoneme (4169 aa)</li>
                      <li>Regulates primary cilia assembly/signalling in photoreceptors, cochlear cells, cardiomyocytes, adipocytes, renal tubule</li>
                      <li>Mediates hedgehog, Wnt, PDGF ciliary signalling pathways</li>
                      <li>Essential for leptin receptor signalling via adipocyte primary cilia → satiety</li>
                    </ol>
                  </div>
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT5 }}>ALMS1 Biallelic LOF → Multi-Organ Ciliopathy</div>
                    <ol className="small mb-0">
                      <li>Ciliary dysfunction → photoreceptor degeneration (cone-rod dystrophy) → blindness</li>
                      <li>Cochlear hair cell cilia dysfunction → sensorineural hearing loss</li>
                      <li>Cardiomyocyte cilia → dilated cardiomyopathy (infantile onset; ~80%)</li>
                      <li>Adipocyte cilia → leptin resistance → hyperphagia → truncal obesity → insulin resistance → T2D-like DM</li>
                    </ol>
                  </div>
                </div>
                <div className="alert mt-2 mb-0 small" style={{ background: ACCENT3 + '18', borderLeft: `3px solid ${ACCENT3}` }}>
                  <strong>C-peptide PRESERVED</strong> (HIGH / hyperinsulinism) — insulin resistance mechanism, NOT beta-cell apoptosis.
                  Contrasts sharply with <em>Wolfram Syndrome</em> (C-pep falls, ER-stress apoptosis) and <em>MODY10</em> (C-pep falls).
                  Alström DM is T2D-like: hyperinsulinism → insulin resistance → eventual beta-cell exhaustion.
                </div>
              </div>
            </div>
          </Section>

          {/* Multi-organ manifestations */}
          <Section title="🫀 Multi-Organ Manifestations (approximate onset)" color={ACCENT5}>
            <div className="card border-0 shadow-sm mb-2">
              <div className="card-body">
                <div className="row g-2">
                  {[
                    { yr: 'Infancy (< 2 yr)', feature: 'Dilated Cardiomyopathy (DCM)', detail: 'Life-threatening; ~60% infantile-onset; some resolve; may recur; echo + BNP monitoring', color: ACCENT5 },
                    { yr: 'Infancy–childhood', feature: 'Cone-Rod Dystrophy', detail: 'Nyctalopia (rod) → cone degeneration → visual impairment → legal blindness by 2nd decade; ERG essential', color: ACCENT4 },
                    { yr: 'Infancy–childhood', feature: 'Sensorineural Hearing Loss', detail: 'Progressive SNHL; high/mid frequency; cochlear hair cell ciliopathy; hearing aids', color: ACCENT6 },
                    { yr: '~16 yr (range 6–35)', feature: 'Type 2-like DM', detail: 'Insulin resistance (NOT autoimmune NOT beta-cell apoptosis); C-pep PRESERVED; metformin/GLP-1RA', color: ACCENT },
                    { yr: '3rd–4th decade', feature: 'CKD / Renal Nephropathy', detail: 'Tubulointerstitial nephropathy; microalbuminuria → CKD → ESRD; ACEi/ARB; transplant', color: ACCENT7 },
                    { yr: '3rd–4th decade', feature: 'Hepatic NASH / Fibrosis', detail: 'NAFLD/NASH (35%); hepatic fibrosis (22%); cirrhosis (11%); liver enzymes monitoring', color: ACCENT7 },
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

          {/* ALMS1 mutations table */}
          <Section title="🧪 ALMS1 Key Mutations (biallelic LOF — common genotypes)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT + '18' }}>
                    <th>Mutation</th><th>Domain</th><th>Population</th><th>Type</th><th>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td><strong>p.Arg3444* (c.10330C{'>'T})</strong></td><td>C-terminal / exon 16</td><td>European</td><td>Nonsense</td><td>Common truncating; exon 16 hotspot</td></tr>
                  <tr><td><strong>p.Gln2051* (c.6151C{'>'T})</strong></td><td>Exon 8</td><td>British/European</td><td>Nonsense</td><td>Truncating; loss of centrosomal targeting</td></tr>
                  <tr><td>p.Ala2172Ser (c.6514G{'>'A})</td><td>Ciliary targeting domain</td><td>Pan-ethnic</td><td>Missense</td><td>Impaired cilia localisation</td></tr>
                  <tr><td>c.10775+1G{'>'A}</td><td>Intron 16 splice</td><td>European</td><td>Splice-site</td><td>Aberrant splicing; frameshift consequence</td></tr>
                  <tr><td>p.Leu1521fs (c.4561delC)</td><td>Exon 8</td><td>Pan-ethnic</td><td>Frameshift</td><td>Early truncation; severe phenotype</td></tr>
                  <tr><td>p.Gly3461Arg (c.10381G{'>'A})</td><td>C-terminal</td><td>North African / Maghreb</td><td>Missense</td><td>North African founder; C-terminal disruption</td></tr>
                  <tr><td>Compound heterozygous ALMS1</td><td>Various</td><td>European</td><td>Compound het</td><td>Most common European genotype (truncating + missense)</td></tr>
                  <tr><td>Exon 16 splice + frameshift</td><td>Exon 16</td><td>Pan-ethnic</td><td>Compound het</td><td>Splice site + frameshift compound heterozygous</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* Patient table preview */}
          <Section title="👥 Cohort Preview (first 12 patients, seed=331)" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT6 + '18' }}>
                    <th>#</th><th>Mutation (allele 1)</th><th>Cardiac Status</th><th>Vision</th><th>DM Onset</th><th>HbA1c%</th><th>C-Pep (nmol/L)</th><th>BMI</th><th>Hearing</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(p => (
                    <tr key={p.id}>
                      <td>{p.id}</td>
                      <td><code style={{ fontSize: '0.68em' }}>{p.mutation}</code></td>
                      <td><Badge text={p.cardiac_status?.split('(')[0].trim() || '—'} color={ACCENT5} /></td>
                      <td><span className="small">{p.retinal_status?.split('(')[0].trim() || '—'}</span></td>
                      <td>{p.dm_onset}</td>
                      <td>{p.hba1c}</td>
                      <td style={{ color: ACCENT3 }}>{p.c_peptide}</td>
                      <td>{p.bmi}</td>
                      <td><span className="small">{p.hearing?.split('(')[0].trim() || '—'}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ─── TAB 1: Multi-Organ Breakdown ─── */}
      {tab === 1 && breakdown && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Cardiac Status (DCM)" color={ACCENT5}>
              {Object.entries(breakdown.cardiac_status || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
              ))}
              <div className="small text-muted mt-1">DCM infantile-onset is most severe; some resolve; recurrence common</div>
            </Section>
            <Section title="Retinal / Vision Status" color={ACCENT4}>
              {Object.entries(breakdown.retinal_status || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
              ))}
              <div className="small text-muted mt-1">Cone-rod dystrophy progresses to legal blindness by 2nd decade in ~23%</div>
            </Section>
            <Section title="ALMS1 Mutation Distribution" color={ACCENT}>
              {Object.entries(breakdown.mutation_distribution || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="DM Onset Tiers" color={ACCENT}>
              {Object.entries(breakdown.dm_onset_tiers || {}).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
              <div className="small text-muted mt-1">Mean ~16 yr; later than Wolfram DM (~6 yr); insulin resistance onset</div>
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="C-Peptide Tiers (PRESERVED / HIGH Pattern)" color={ACCENT3}>
              {Object.entries(breakdown.c_peptide_tiers || {}).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
              <div className="small text-muted mt-1">C-pep PRESERVED/HIGH (hyperinsulinism) — contrasts with Wolfram (falling) and MODY10 (falling)</div>
            </Section>
            <Section title="HbA1c Tiers" color={ACCENT}>
              {Object.entries(breakdown.hba1c_tiers || {}).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="BMI Tiers (Truncal Obesity)" color={ACCENT}>
              {Object.entries(breakdown.bmi_tiers || {}).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
              <div className="small text-muted mt-1">Truncal obesity from leptin resistance (adipocyte ciliary dysfunction); distinguishes from Wolfram (lean)</div>
            </Section>
            <Section title="Hearing Status" color={ACCENT6}>
              {Object.entries(breakdown.hearing_status || {}).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
          </div>
          <div className="col-12">
            <div className="row g-3">
              <div className="col-md-3">
                <Section title="Renal Status" color={ACCENT7}>
                  {Object.entries(breakdown.renal_status || {}).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
                  ))}
                </Section>
              </div>
              <div className="col-md-3">
                <Section title="Hepatic Status" color={ACCENT7}>
                  {Object.entries(breakdown.hepatic_status || {}).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
                  ))}
                </Section>
              </div>
              <div className="col-md-3">
                <Section title="Diabetes Status" color={ACCENT}>
                  {Object.entries(breakdown.diabetes_status || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                  ))}
                </Section>
              </div>
              <div className="col-md-3">
                <Section title="Ethnicity" color={ACCENT6}>
                  {Object.entries(breakdown.ethnicity_distribution || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                  ))}
                </Section>
              </div>
            </div>
            <div className="row g-3">
              <div className="col-md-4">
                <Section title="Prior Misdiagnosis" color={ACCENT}>
                  {Object.entries(breakdown.misdiagnosis_distribution || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                  ))}
                  <div className="small text-muted mt-1">Wolfram and BBS are key differentials; NGS panel distinguishes</div>
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="Summary Flags" color={ACCENT}>
                  {Object.entries(breakdown.summary_flags || {}).map(([k, v]) => (
                    <div key={k} className="d-flex justify-content-between small mb-1 p-1 rounded" style={{ background: ACCENT + '0d' }}>
                      <span>{k.replace(/_/g, ' ')}</span>
                      <span className="fw-bold" style={{ color: ACCENT }}>{v}{typeof v === 'number' && v <= 100 ? '%' : ''}</span>
                    </div>
                  ))}
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="Additional Flags (from Overview KPIs)" color={ACCENT2}>
                  {[
                    { label: 'Mean DM onset', value: kpis.mean_dm_onset },
                    { label: 'Mean HbA1c', value: kpis.mean_hba1c },
                    { label: 'Mean BMI', value: kpis.mean_bmi },
                    { label: 'Mean C-Peptide', value: kpis.mean_c_peptide },
                    { label: 'Hepatic involvement', value: kpis.pct_hepatic },
                    { label: 'Consanguinity', value: kpis.pct_consanguinity },
                  ].map(({ label, value }) => (
                    <div key={label} className="d-flex justify-content-between small mb-1 p-1 rounded" style={{ background: ACCENT2 + '0d' }}>
                      <span>{label}</span>
                      <span className="fw-bold" style={{ color: ACCENT2 }}>{value}</span>
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

          <Section title="⚖ Alström (ALMS1) vs Wolfram (WFS1) — Key Differentials" color={ACCENT5}>
            {definitions.comparison_wolfram_alstrom && (
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead>
                    <tr style={{ background: ACCENT5 + '18' }}>
                      <th>Feature</th>
                      {Object.keys(definitions.comparison_wolfram_alstrom).map(k => (
                        <th key={k} style={{ color: ACCENT5 }}>{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {['gene', 'mechanism', 'c_peptide', 'diabetes_type', 'cardiomyopathy', 'obesity', 'retinal', 'di', 'inheritance', 'treatment'].map(field => (
                      <tr key={field}>
                        <td className="fw-bold text-capitalize">{field.replace(/_/g, ' ')}</td>
                        {Object.values(definitions.comparison_wolfram_alstrom).map((entry, i) => (
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
        <Link href="/wolfram" className="btn btn-sm btn-outline-secondary">← Wolfram Syndrome 1 (WFS1)</Link>
        <Link href="/mody13" className="btn btn-sm btn-outline-secondary">← MODY13 (KCNJ11)</Link>
        <Link href="/" className="btn btn-sm btn-outline-primary">🏠 Portal Home</Link>
      </div>
    </div>
  );
}
