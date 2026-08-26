'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Genetics & Subtype', 'Definitions'];

// PCD colour scheme — teal-cerulean-slate-amber (motile ciliopathy; respiratory; cilia)
const ACCENT  = '#006064';   // dark teal — motile cilia / airways; mucociliary clearance
const ACCENT2 = '#01579b';   // dark cerulean — respiratory tract; bronchiectasis
const ACCENT3 = '#1b5e20';   // dark green — DNAH5 ODA defect (most common gene)
const ACCENT4 = '#e65100';   // deep orange — neonatal respiratory distress; warning
const ACCENT5 = '#4a148c';   // deep purple — situs inversus / Kartagener; laterality
const ACCENT6 = '#37474f';   // dark slate — epidemiology; AR genetics
const ACCENT7 = '#bf360c';   // dark orange-red — bronchiectasis severity / FEV1
const ACCENT8 = '#827717';   // dark amber — hearing loss / OME / male infertility

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
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function PCDPage() {
  const [tab, setTab] = useState(0);
  const [ov, setOv]   = useState(null);
  const [bk, setBk]   = useState(null);
  const [df, setDf]   = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/pcd/overview`).then(r => r.json()),
      fetch(`${API}/api/pcd/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pcd/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="container py-4 text-danger">Error: {err}</div>;
  if (!ov)  return <div className="container py-4 text-muted">Loading PCD dashboard…</div>;

  const kpis = ov.kpis || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          🧬 Primary Ciliary Dyskinesia (PCD / Kartagener Syndrome)
        </h4>
        <div className="text-muted small">
          DNAH5 · 5p15.2 · ODA Motile Ciliopathy · Kartagener Triad · AR Biallelic LOF · OMIM #608644
          &nbsp;|&nbsp; Cohort: {_COHORT_SIZE} patients (seed-{339}) · 3 endpoints verified
        </div>
        <div className="mt-1">
          <Badge text="Motile Ciliopathy" color={ACCENT} />
          <Badge text="Kartagener Syndrome ~50%" color={ACCENT5} />
          <Badge text="Situs Inversus" color={ACCENT5} />
          <Badge text="Bronchiectasis" color={ACCENT2} />
          <Badge text="DNAH5 Most Common" color={ACCENT3} />
          <Badge text="AR" color={ACCENT6} />
          <Badge text="~1/15,000" color={ACCENT6} />
        </div>
      </div>

      {/* Tab Nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              onClick={() => setTab(i)}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row mb-2">
            <KPI label="Cohort (n)" value={kpis.cohort_n ?? _COHORT_SIZE} color={ACCENT} />
            <KPI label="Median Dx Age (yr)" value={kpis.median_age_dx_yr ?? '—'} color={ACCENT2} />
            <KPI label="Mean FEV1 % pred" value={kpis.mean_fev1_pct_pred ? `${kpis.mean_fev1_pct_pred}%` : '—'} color={ACCENT7} />
            <KPI label="Median nNO (nL/min)" value={kpis.median_nno_nl_min ?? '—'} color={ACCENT4} />
            <KPI label="% Situs Inversus" value={kpis.pct_situs_inversus ? `${kpis.pct_situs_inversus}%` : '—'} color={ACCENT5} />
            <KPI label="% Bronchiectasis" value={kpis.pct_bronchiectasis ? `${kpis.pct_bronchiectasis}%` : '—'} color={ACCENT2} />
          </div>
          <div className="row mb-3">
            <KPI label="% NRD (Neonatal)" value={kpis.pct_neonatal_rd ? `${kpis.pct_neonatal_rd}%` : '—'} color={ACCENT4} />
            <KPI label="% Hearing Loss/OME" value={kpis.pct_hearing_loss_ome ? `${kpis.pct_hearing_loss_ome}%` : '—'} color={ACCENT8} />
            <KPI label="% Male Infertility" value={kpis.pct_male_infertility ? `${kpis.pct_male_infertility}%` : '—'} color={ACCENT8} />
            <KPI label="% Consanguineous" value={kpis.pct_consanguineous ? `${kpis.pct_consanguineous}%` : '—'} color={ACCENT6} />
            <KPI label="% DNAH5" value={kpis.pct_dnah5 ? `${kpis.pct_dnah5}%` : '—'} color={ACCENT3} />
            <KPI label="% Hydrocephalus" value={kpis.pct_hydrocephalus ? `${kpis.pct_hydrocephalus}%` : '—'} color={ACCENT6} />
          </div>

          {/* Alerts */}
          <Section title="⚠️ Clinical Alerts" color={ACCENT4}>
            {ov.alerts && Object.entries(ov.alerts).map(([k, v]) => (
              <Alert key={k} color={ACCENT4}>
                <span className="fw-bold">{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}:</span>{' '}{v}
              </Alert>
            ))}
          </Section>

          {/* Key Facts */}
          <Section title="🔬 Key Facts — PCD Clinical & Molecular" color={ACCENT}>
            <ul className="mb-0 small">
              {(ov.key_facts || []).map((f, i) => <li key={i} className="mb-1">{f}</li>)}
            </ul>
          </Section>

          {/* Clinical Profile Summary */}
          <Section title="🫁 PCD Clinical Profile Snapshot" color={ACCENT2}>
            <div className="row">
              <div className="col-md-6">
                <table className="table table-sm table-bordered small mb-0">
                  <tbody>
                    <tr><td className="fw-bold" style={{ color: ACCENT }}>Primary Gene (most common)</td><td>{kpis.gene}</td></tr>
                    <tr><td className="fw-bold">Chromosome</td><td>{kpis.chromosome}</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>{kpis.inheritance}</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>{kpis.prevalence}</td></tr>
                    <tr><td className="fw-bold">Cohort Type</td><td>{kpis.cohort_type}</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="col-md-6">
                <div className="p-2 rounded small" style={{ background: ACCENT + '10', border: `1px solid ${ACCENT}40` }}>
                  <div className="fw-bold mb-1" style={{ color: ACCENT }}>vs. Meckel-Gruber / Joubert (key differences):</div>
                  <ul className="mb-0">
                    <li><b>PCD = MOTILE ciliopathy</b> (dynein arm / axoneme) — airways, sperm, laterality</li>
                    <li><b>MKS / Joubert = PRIMARY (non-motile) ciliopathy</b> (TZ gate) — organogenesis</li>
                    <li>PCD: NOT uniformly lethal; survives to adulthood</li>
                    <li>PCD: Situs inversus ~50%; MKS: situs inversus ~10% (Wnt/nodal secondary)</li>
                    <li>PCD: Retina NORMAL (primary cilia intact); BBS/Joubert: retinal dystrophy</li>
                    <li>PCD: Male infertility (sperm flagella); BBS: hypogonadism (different mechanism)</li>
                  </ul>
                </div>
              </div>
            </div>
          </Section>

          {/* Patient Sample */}
          <Section title={`👥 Sample Patients (first 8 of ${_COHORT_SIZE})`} color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-hover small mb-0">
                <thead style={{ background: ACCENT + '15' }}>
                  <tr>
                    <th>ID</th><th>Gene</th><th>Ethnicity</th>
                    <th>Dx Age</th><th>Situs</th><th>NRD</th>
                    <th>nNO</th><th>FEV1%</th><th>Bronch.</th><th>Prior Dx</th>
                  </tr>
                </thead>
                <tbody>
                  {(ov.patients || []).map(p => (
                    <tr key={p.id}>
                      <td><code style={{ fontSize: '0.75em' }}>{p.id}</code></td>
                      <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={p.gene}>{p.gene?.split('(')[0]?.trim()}</td>
                      <td>{p.ethnicity?.split('(')[0]?.trim()}</td>
                      <td>{p.age_at_diagnosis_yr} yr</td>
                      <td>{p.situs_inversus ? <span style={{ color: ACCENT5 }}>✓ Inv</span> : '—'}</td>
                      <td>{p.neonatal_rd ? <span style={{ color: ACCENT4 }}>✓</span> : '—'}</td>
                      <td style={{ color: p.nasal_no_nl_min < 77 ? ACCENT4 : ACCENT3 }}>
                        {p.nasal_no_nl_min}
                      </td>
                      <td style={{ color: p.fev1_pct_pred < 60 ? ACCENT7 : ACCENT3 }}>
                        {p.fev1_pct_pred}%
                      </td>
                      <td>{p.has_bronchiectasis ? <span style={{ color: ACCENT2 }}>✓</span> : '—'}</td>
                      <td style={{ maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={p.prior_misdiagnosis}>{p.prior_misdiagnosis?.split('(')[0]?.trim()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 1: Diagnostic Breakdown ── */}
      {tab === 1 && bk && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="🫁 Pulmonary Finding" color={ACCENT2}>
                {Object.entries(bk.pulmonary_finding || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
                ))}
              </Section>

              <Section title="🔬 TEM Defect Class" color={ACCENT3}>
                {Object.entries(bk.tem_class || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>

              <Section title="🌀 Ciliary Beat Pattern (HSVM)" color={ACCENT}>
                {Object.entries(bk.beat_pattern || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                ))}
              </Section>

              <Section title="🌍 Situs Distribution" color={ACCENT5}>
                {Object.entries(bk.situs_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
                ))}
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="📊 nasal NO Tiers (nL/min)" color={ACCENT4}>
                {Object.entries(bk.nno_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
              </Section>

              <Section title="💨 FEV1 % Predicted Tiers" color={ACCENT7}>
                {Object.entries(bk.fev1_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
                ))}
              </Section>

              <Section title="⏱️ Age at Diagnosis" color={ACCENT6}>
                {Object.entries(bk.age_at_diagnosis_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                ))}
              </Section>

              <Section title="🤧 Sinus Finding" color={ACCENT8}>
                {Object.entries(bk.sinus_finding || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                ))}
              </Section>

              <Section title="⚕️ Prior Misdiagnosis" color={ACCENT4}>
                {Object.entries(bk.prior_misdiagnosis || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
              </Section>
            </div>
          </div>

          {/* Summary stats row */}
          {bk.summary && (
            <Section title="📈 Cohort Summary Statistics" color={ACCENT}>
              <div className="row">
                {Object.entries(bk.summary).filter(([k]) => k !== 'n').map(([k, v]) => (
                  <div key={k} className="col-6 col-md-3 mb-2">
                    <div className="card shadow-sm text-center p-2">
                      <div className="fw-bold" style={{ color: ACCENT }}>{typeof v === 'number' ? (k.includes('pct') ? `${v}%` : v) : v}</div>
                      <div className="text-muted" style={{ fontSize: '0.72em' }}>{k.replace(/_/g, ' ')}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          )}
        </div>
      )}

      {/* ── TAB 2: Genetics & Subtype ── */}
      {tab === 2 && bk && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="🧬 Gene Distribution (this cohort)" color={ACCENT3}>
                {Object.entries(bk.gene_distribution || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>

              <Section title="🌍 Ethnicity Distribution" color={ACCENT6}>
                {Object.entries(bk.ethnicity || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                ))}
              </Section>
            </div>

            <div className="col-md-6">
              {/* Axoneme defect class reference table */}
              <Section title="🔬 Axoneme Defect Class Reference" color={ACCENT}>
                <table className="table table-sm table-bordered small mb-3">
                  <thead style={{ background: ACCENT + '15' }}>
                    <tr><th>Class</th><th>Genes</th><th>TEM</th><th>Beat</th><th>nNO</th></tr>
                  </thead>
                  <tbody>
                    <tr><td><b>ODA</b></td><td>DNAH5, DNAI1, DNAI2</td><td>ODA absent</td><td>Immotile</td><td style={{color:ACCENT4}}>Very low</td></tr>
                    <tr><td><b>IDA+MTD</b></td><td>CCDC39, CCDC40</td><td>IDA absent + MTD</td><td>Dyskinetic</td><td style={{color:ACCENT4}}>Very low</td></tr>
                    <tr><td><b>RSH</b></td><td>RSPH4A, RSPH9</td><td><b>Normal</b></td><td>Circular</td><td style={{color:ACCENT3}}>Near-normal</td></tr>
                    <tr><td><b>CP</b></td><td>HYDIN</td><td><b>Normal</b></td><td>Abnormal wave</td><td style={{color:ACCENT3}}>Near-normal</td></tr>
                    <tr><td><b>RGMC</b></td><td>CCNO, MCIDAS</td><td>Reduced cilia</td><td>Absent/rare</td><td style={{color:ACCENT4}}>Low</td></tr>
                    <tr><td><b>ODA+IDA</b></td><td>DNAAF1-5, LRRC6</td><td>Both absent</td><td>Static</td><td style={{color:ACCENT4}}>Very low</td></tr>
                  </tbody>
                </table>
                <div className="small text-muted">
                  ⚠️ ~30% of PCD has <b>normal TEM ultrastructure</b> (RSH/CP/DNAH11 subtypes) —
                  TEM alone cannot exclude PCD; gene panel mandatory.
                </div>
              </Section>

              {/* Kartagener note */}
              <Section title="↔️ Kartagener Syndrome & Situs Inversus" color={ACCENT5}>
                <div className="p-2 rounded small" style={{ background: ACCENT5 + '12', border: `1px solid ${ACCENT5}40` }}>
                  <p className="mb-1"><b>Kartagener Triad:</b> PCD + Situs Inversus Totalis + Bronchiectasis + Sinusitis</p>
                  <p className="mb-1"><b>~50% of all PCD</b> have situs inversus — embryonic nodal cilia set left-right axis;
                    when nodal cilia are dysfunctional, laterality is assigned randomly (50% chance inversus).</p>
                  <table className="table table-sm table-bordered mb-0 mt-2">
                    <thead><tr><th>Gene</th><th>Situs Inversus Rate</th></tr></thead>
                    <tbody>
                      <tr><td>DNAH11</td><td style={{color:ACCENT5}}>~85% (highest)</td></tr>
                      <tr><td>DNAH5, DNAI1</td><td>~50%</td></tr>
                      <tr><td>CCDC39, CCDC40</td><td>~40%</td></tr>
                      <tr><td>RSPH4A, RSPH9</td><td style={{color:ACCENT3}}>~5% (very rare)</td></tr>
                      <tr><td>HYDIN</td><td style={{color:ACCENT3}}>Rare</td></tr>
                      <tr><td>CCNO / RGMC</td><td style={{color:ACCENT3}}>&lt;5%</td></tr>
                    </tbody>
                  </table>
                </div>
              </Section>

              {/* Founder variants */}
              <Section title="🏴 Founder Variants by Population" color={ACCENT8}>
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: ACCENT8 + '15' }}>
                    <tr><th>Gene / Variant</th><th>Population</th></tr>
                  </thead>
                  <tbody>
                    <tr><td>DNAH5 c.7915C&gt;T p.Arg2639Ter</td><td>North European (UK, Scandinavia)</td></tr>
                    <tr><td>DNAI1 IVS1+2_3insT</td><td>Middle Eastern / North African</td></tr>
                    <tr><td>CCDC40 c.248delC p.Leu83fs</td><td>Iberian (Spanish/Portuguese)</td></tr>
                    <tr><td>RSPH9 c.1-2A&gt;G</td><td>Ashkenazi Jewish / Middle Eastern</td></tr>
                    <tr><td>CCDC103 p.His154Pro</td><td>South Asian (India/Pakistan)</td></tr>
                    <tr><td>LRRC6 c.223G&gt;A p.Gly75Arg</td><td>North African</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && df && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="📖 Disease Overview" color={ACCENT}>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    {['disease','omim_gene','omim_disease','chromosome','inheritance','prevalence'].map(k => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ color: ACCENT, width: '35%' }}>{k.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</td>
                        <td>{df[k]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="small p-2 rounded" style={{ background: ACCENT + '10', border: `1px solid ${ACCENT}30` }}>
                  {df.mechanism}
                </div>
              </Section>

              <Section title="🧬 Kartagener Syndrome" color={ACCENT5}>
                <div className="small p-2 rounded" style={{ background: ACCENT5 + '10', border: `1px solid ${ACCENT5}30` }}>
                  {df.kartagener_syndrome}
                </div>
              </Section>

              <Section title="🔬 Axoneme Defect Classes" color={ACCENT3}>
                {df.axoneme_defect_classes && Object.entries(df.axoneme_defect_classes).map(([k, v]) => (
                  <div key={k} className="mb-1 small">
                    <span className="fw-bold" style={{ color: ACCENT3 }}>{k}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="⚕️ CF vs PCD Distinction" color={ACCENT7}>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    {df.ddx_vs_cf && Object.entries(df.ddx_vs_cf).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ color: ACCENT7, width:'35%' }}>{k.replace(/_/g,' ')}</td>
                        <td>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="🩺 Diagnostic Criteria" color={ACCENT4}>
                {df.diagnostic_criteria && Object.entries(df.diagnostic_criteria).map(([k, v]) => (
                  <div key={k} className="mb-1 small">
                    <span className="fw-bold" style={{ color: ACCENT4 }}>{k}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="🫁 Key Clinical Features" color={ACCENT2}>
                {df.key_clinical_features && Object.entries(df.key_clinical_features).map(([k, v]) => (
                  <div key={k} className="mb-1 small">
                    <span className="fw-bold" style={{ color: ACCENT2 }}>{k.replace(/_/g,' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="💊 Treatment" color={ACCENT3}>
                {df.treatment && Object.entries(df.treatment).map(([k, v]) => (
                  <div key={k} className="mb-1 small">
                    <span className="fw-bold" style={{ color: ACCENT3 }}>{k.replace(/_/g,' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="🧬 Gene Clinical Pearls" color={ACCENT}>
                {df.key_genes_clinical_pearls && Object.entries(df.key_genes_clinical_pearls).map(([k, v]) => (
                  <div key={k} className="mb-1 small">
                    <span className="fw-bold" style={{ color: ACCENT }}>{k}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="📈 Prognosis" color={ACCENT6}>
                <div className="small p-2 rounded" style={{ background: ACCENT6 + '10', border: `1px solid ${ACCENT6}30` }}>
                  {df.prognosis}
                </div>
              </Section>

              {df.cohort_note && (
                <div className="small text-muted mt-2 p-2 rounded border">{df.cohort_note}</div>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="mt-3 text-muted small border-top pt-2">
        <Link href="/" className="me-3">← Home</Link>
        <Link href="/meckel" className="me-3">← Meckel-Gruber</Link>
        Source: Bush et al. 2022 ERS PCD Guidelines · Shapiro et al. 2021 NEJM ·
        OMIM #608644 (PCD2/DNAH5) · Seed-{339} cohort
      </div>
    </div>
  );
}
