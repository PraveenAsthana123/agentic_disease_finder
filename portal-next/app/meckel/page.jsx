'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Genetics & Diagnosis', 'Definitions'];

// Meckel-Gruber Syndrome colour scheme — deep crimson-slate-charcoal-amber (lethal TZ ciliopathy; prenatal)
const ACCENT  = '#b71c1c';   // dark crimson — lethality; MKS hallmark; encephalocele
const ACCENT2 = '#263238';   // dark blue-grey — prenatal diagnosis; anatomy scan
const ACCENT3 = '#e65100';   // deep orange — renal polycystic; Potter sequence
const ACCENT4 = '#4a148c';   // deep purple — TZ ciliopathy pathway; MKS1/CEP290 allele spectrum
const ACCENT5 = '#1b5e20';   // dark green — PGT-M; recurrence prevention
const ACCENT6 = '#37474f';   // dark slate — epidemiology; AR; consanguinity
const ACCENT7 = '#4e342e';   // dark brown — hepatic CHF; ductal plate malformation
const ACCENT8 = '#827717';   // dark amber — polydactyly; limb; post-axial

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

export default function MeckelPage() {
  const [tab, setTab] = useState(0);
  const [ov, setOv]   = useState(null);
  const [bk, setBk]   = useState(null);
  const [df, setDf]   = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/meckel/overview`).then(r => r.json()),
      fetch(`${API}/api/meckel/breakdown`).then(r => r.json()),
      fetch(`${API}/api/meckel/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="container py-5 text-danger">Error: {err}</div>;
  if (!ov)  return <div className="container py-5 text-muted">Loading Meckel-Gruber Syndrome dashboard…</div>;

  const k = ov.kpis;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', borderLeft: `5px solid ${ACCENT}` }}>
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          &#x1f9ec; Meckel-Gruber Syndrome (MKS)
        </h4>
        <div className="small text-muted">
          <Badge text="MKS1 · 17q22" color={ACCENT} />
          <Badge text="TZ Ciliopathy" color={ACCENT4} />
          <Badge text="LETHAL — Prenatal" color={ACCENT} />
          <Badge text="AR Biallelic LOF" color={ACCENT6} />
          <Badge text="OMIM #249000" color={ACCENT2} />
          <Badge text={`n=${_COHORT_SIZE} prenatal/autopsy`} color={ACCENT6} />
        </div>
        <div className="small mt-1" style={{ color: ACCENT }}>
          <strong>Uniformly lethal TZ ciliopathy — </strong>
          the most severe end of the CEP290/TMEM67/CC2D2A allele severity spectrum.
          Classical Meckel's Triad: Occipital Encephalocele + Bilateral Polycystic Kidneys + Post-Axial Polydactyly.
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

      {/* ── TAB 0: Overview ─────────────────────────────────── */}
      {tab === 0 && (
        <>
          {/* KPIs */}
          <div className="row mb-3">
            <KPI label="Cohort (prenatal/autopsy)" value={k.cohort_n} color={ACCENT6} />
            <KPI label="Always Lethal" value="100%" color={ACCENT} />
            <KPI label="TOPF rate" value={`${k.pct_topf}%`} color={ACCENT2} />
            <KPI label="Consanguineous" value={`${k.pct_consanguineous}%`} color={ACCENT6} />
            <KPI label="Hepatic CHF (autopsy)" value={`${k.pct_hepatic_chf}%`} color={ACCENT7} />
            <KPI label="Potter Sequence" value={`${k.pct_potter_sequence}%`} color={ACCENT3} />
            <KPI label="Pulm Hypoplasia" value={`${k.pct_pulm_hypoplasia}%`} color={ACCENT3} />
            <KPI label="Cardiac Defect" value={`${k.pct_cardiac}%`} color={ACCENT} />
            <KPI label="PGT-M Planned" value={`${k.pct_pgt_planned}%`} color={ACCENT5} />
            <KPI label="Prior Affected Preg" value={`${k.pct_prior_recurrence}%`} color={ACCENT6} />
            <KPI label="MKS1 gene" value={`${k.pct_mks1}%`} color={ACCENT4} />
            <KPI label="TMEM67 gene" value={`${k.pct_tmem67}%`} color={ACCENT4} />
          </div>

          {/* Critical alert banner */}
          <div className="alert fw-bold mb-3" style={{ background: ACCENT + '22', border: `2px solid ${ACCENT}`, borderRadius: 8 }}>
            ⚠️ MECKEL-GRUBER SYNDROME IS UNIFORMLY LETHAL — death occurs in utero (2nd trimester) or within hours–days of birth.
            This cohort represents prenatal diagnoses, terminations of pregnancy, perinatal autopsies, and stillbirths.
            Management is palliative. PGT-M is the primary recurrence-prevention strategy.
          </div>

          {/* Alerts */}
          <Section title="Clinical Decision Alerts" color={ACCENT}>
            {Object.entries(ov.alerts).map(([k, v]) => (
              <Alert key={k} color={ACCENT}>
                <strong>{k.replace(/_/g, ' ').toUpperCase()}:</strong> {v}
              </Alert>
            ))}
          </Section>

          {/* Key facts */}
          <Section title="Key Clinical Facts" color={ACCENT4}>
            <ul className="mb-0">
              {ov.key_facts.map((f, i) => <li key={i} className="small mb-1">{f}</li>)}
            </ul>
          </Section>

          {/* Classical Triad */}
          <Section title="Meckel's Triad — All 3 Always Present" color={ACCENT3}>
            <div className="row g-2">
              {[
                { num: '1', label: 'Occipital Encephalocele', desc: 'Neural tube closure defect; occipital most common (~75%); exencephaly in most severe (~15%); Dandy-Walker (RPGRIP1L)', color: ACCENT },
                { num: '2', label: 'Bilateral Polycystic Kidneys', desc: 'Massive bilateral renal dysplasia (3–4× normal size); cysts replacing nephrons; oligohydramnios → Potter sequence; urine absent', color: ACCENT3 },
                { num: '3', label: 'Post-Axial Polydactyly', desc: 'Extra digit on ulnar/fibular side; bilateral in ~80%; present in hands > feet; distinguishes from Joubert (20–30%) and BBS (70%)', color: ACCENT8 },
              ].map(item => (
                <div key={item.num} className="col-md-4">
                  <div className="card h-100 shadow-sm" style={{ borderTop: `4px solid ${item.color}` }}>
                    <div className="card-body py-2 px-3">
                      <div className="fw-bold" style={{ color: item.color }}>
                        <span className="fs-4 me-2">{item.num}.</span>{item.label}
                      </div>
                      <div className="small text-muted mt-1">{item.desc}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Sample prenatal cases */}
          <Section title={`Cohort Sample — Prenatal Diagnoses / Perinatal Autopsies (n=${_COHORT_SIZE})`} color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small">
                <thead>
                  <tr>
                    <th>#</th><th>Maternal Age</th><th>Fetal Sex</th><th>Gene</th>
                    <th>Encephalocele</th><th>Outcome</th><th>Consanguineous</th><th>PGT-M Planned</th>
                  </tr>
                </thead>
                <tbody>
                  {ov.patients.map(p => (
                    <tr key={p.id}>
                      <td>{p.id}</td>
                      <td>{p.maternal_age}y</td>
                      <td>{p.fetal_sex}</td>
                      <td><small>{p.gene.split('—')[0].trim()}</small></td>
                      <td><small>{p.encephalocele_type.split('(')[0].trim()}</small></td>
                      <td><small>{p.outcome.split('—')[0].trim()}</small></td>
                      <td>{p.consanguineous ? <span style={{ color: ACCENT6 }}>Yes</span> : 'No'}</td>
                      <td>{p.pgt_m_planned ? <span style={{ color: ACCENT5 }}>Yes</span> : 'No'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </>
      )}

      {/* ── TAB 1: Multi-System Breakdown ───────────────────── */}
      {tab === 1 && bk && (
        <>
          <div className="row g-3 mb-3">
            {/* Gene distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold mb-3" style={{ color: ACCENT4 }}>Gene Distribution</h6>
                  {Object.entries(bk.gene_distribution)
                    .sort((a, b) => b[1] - a[1])
                    .map(([g, v]) => (
                      <Bar key={g} label={g} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                    ))}
                </div>
              </div>
            </div>

            {/* Outcome distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Pregnancy Outcomes</h6>
                  {Object.entries(bk.outcome_distribution)
                    .sort((a, b) => b[1] - a[1])
                    .map(([o, v]) => (
                      <Bar key={o} label={o} value={v} max={_COHORT_SIZE} color={ACCENT} />
                    ))}
                </div>
              </div>
            </div>
          </div>

          <div className="row g-3 mb-3">
            {/* Encephalocele types */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Encephalocele Subtype</h6>
                  {Object.entries(bk.encephalocele_types)
                    .sort((a, b) => b[1] - a[1])
                    .map(([e, v]) => (
                      <Bar key={e} label={e} value={v} max={_COHORT_SIZE} color={ACCENT} />
                    ))}
                </div>
              </div>
            </div>

            {/* Renal findings */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold mb-3" style={{ color: ACCENT3 }}>Renal Findings</h6>
                  {Object.entries(bk.renal_distribution)
                    .sort((a, b) => b[1] - a[1])
                    .map(([r, v]) => (
                      <Bar key={r} label={r} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                    ))}
                </div>
              </div>
            </div>
          </div>

          <div className="row g-3 mb-3">
            {/* Hepatic findings */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold mb-3" style={{ color: ACCENT7 }}>Hepatic Findings (Autopsy)</h6>
                  {Object.entries(bk.hepatic_distribution)
                    .sort((a, b) => b[1] - a[1])
                    .map(([h, v]) => (
                      <Bar key={h} label={h} value={v} max={_COHORT_SIZE} color={ACCENT7} />
                    ))}
                </div>
              </div>
            </div>

            {/* Ethnicity */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold mb-3" style={{ color: ACCENT6 }}>Ethnicity Distribution</h6>
                  {Object.entries(bk.ethnicity_distribution)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 12)
                    .map(([e, v]) => (
                      <Bar key={e} label={e} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                    ))}
                </div>
              </div>
            </div>
          </div>

          <div className="row g-3 mb-3">
            {/* GA at delivery */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold mb-3" style={{ color: ACCENT2 }}>Gestational Age at Delivery</h6>
                  {Object.entries(bk.gestational_age_tiers).map(([t, v]) => (
                    <Bar key={t} label={t} value={v} max={_COHORT_SIZE} color={ACCENT2} />
                  ))}
                </div>
              </div>
            </div>

            {/* Differential diagnoses */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold mb-3" style={{ color: ACCENT4 }}>Prior Differential Diagnoses Considered</h6>
                  {Object.entries(bk.differential_diagnoses)
                    .sort((a, b) => b[1] - a[1])
                    .map(([d, v]) => (
                      <Bar key={d} label={d} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                    ))}
                </div>
              </div>
            </div>
          </div>

          {/* MKS vs JBTS comparison table */}
          {bk.mks_vs_jbts_comparison && (
            <Section title="MKS vs Joubert Syndrome — Same Genes, Allele Severity Determines Outcome" color={ACCENT4}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead className="table-dark">
                    <tr>
                      <th>Feature</th>
                      <th style={{ color: '#ff8a80' }}>Meckel-Gruber (MKS)</th>
                      <th style={{ color: '#80cbc4' }}>Joubert Syndrome (JBTS)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bk.mks_vs_jbts_comparison.rows.map((row, i) => (
                      <tr key={i}>
                        <td className="fw-bold">{row.feature}</td>
                        <td style={{ color: ACCENT }}>{row.mks}</td>
                        <td style={{ color: '#00695c' }}>{row.joubert}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="small text-muted">
                CEP290 allele severity model: mild missense → Joubert · IVS26+1655A>G (hypomorphic) → LCA10 / Senior-Løken · severe null (p.Arg151*) → Meckel-Gruber.
                The same gene family encodes three distinct clinical outcomes based entirely on residual protein function.
              </div>
            </Section>
          )}
        </>
      )}

      {/* ── TAB 2: Genetics & Diagnosis ─────────────────────── */}
      {tab === 2 && (
        <>
          {/* CEP290 Allele Severity Spectrum */}
          <Section title="CEP290 Allele Severity Spectrum — Same Gene, 4 Outcomes" color={ACCENT4}>
            <div className="row g-2 mb-3">
              {[
                { label: 'Leber Congenital Amaurosis 10 (LCA10)', allele: 'IVS26+1655A>G (c.2991+1655A>G) — hypomorphic; cryptic exon 26a; partial protein', severity: 'Moderate — eye only', outcome: 'Blind from infancy; life expectancy normal; sepofarsen ASO trial', color: ACCENT5 },
                { label: 'Joubert Syndrome (JBTS)', allele: 'p.Pro1280Leu / compound het mild — partial function retained', severity: 'Moderate — multi-organ, survivable', outcome: 'Brain (MTS) + retinal + renal + hepatic; median survival 30–50+ yr', color: '#00695c' },
                { label: 'Senior-Løken Syndrome', allele: 'IVS26 + mild second allele — eye + nephronophthisis', severity: 'Moderate — eye + kidney', outcome: 'Retinal + NPHP; no MTS; survivable with transplant', color: '#1565c0' },
                { label: 'Meckel-Gruber Syndrome (MKS)', allele: 'p.Arg151* / p.Gln1745* — biallelic null; complete LOF', severity: 'Severe — lethal', outcome: 'Encephalocele + polycystic kidneys + polydactyly; death in utero/hours', color: ACCENT },
              ].map(item => (
                <div key={item.label} className="col-md-3">
                  <div className="card h-100 shadow-sm" style={{ borderTop: `4px solid ${item.color}` }}>
                    <div className="card-body py-2 px-3">
                      <div className="fw-bold small" style={{ color: item.color }}>{item.label}</div>
                      <div className="small text-muted mt-1"><em>Allele:</em> {item.allele}</div>
                      <div className="small mt-1"><strong>Severity:</strong> {item.severity}</div>
                      <div className="small text-muted mt-1">{item.outcome}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="small" style={{ color: ACCENT4 }}>
              <strong>Interpretation:</strong> Genotype–phenotype correlation for CEP290 is allele-severity driven.
              Before diagnosing "isolated LCA10", confirm no encephalocele/renal features; CEP290 panel must include exon 5 (p.Arg151* = Meckel allele).
            </div>
          </Section>

          {/* Key mutations table */}
          <Section title="Key Pathogenic Mutations by Gene" color={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small">
                <thead className="table-dark">
                  <tr>
                    <th>Gene (locus)</th><th>Mutation</th><th>Ethnicity / Enrichment</th><th>Mechanism</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { gene: 'MKS1 (17q22)', mut: 'p.Arg327* (c.979C>T)', eth: 'Finnish founder — most common single MKS1 allele', mech: 'Null — exon 10 truncation; TZ matrix lost' },
                    { gene: 'MKS1 (17q22)', mut: 'c.1408_1409insTA (p.Tyr470*)', eth: 'Pan-ethnic', mech: 'Frameshift; null; common compound het partner' },
                    { gene: 'MKS1 (17q22)', mut: 'IVS15+1G>A', eth: 'Northern European', mech: 'Splice site; intron 15; null; often with p.Arg327*' },
                    { gene: 'TMEM216 (11q12.2)', mut: 'p.Arg73Leu (c.218G>T)', eth: 'Ashkenazi Jewish founder; also holoprosencephaly', mech: 'Missense in TM domain; loss of ciliary membrane anchor; JBTS2 when hypomorphic' },
                    { gene: 'TMEM216 (11q12.2)', mut: 'p.Tyr109* (c.327C>A)', eth: 'Pan-ethnic', mech: 'Nonsense; severe MKS; holoprosencephaly + anophthalmia variant' },
                    { gene: 'TMEM67/MKS3 (8q22.1)', mut: 'p.Cys615Arg (c.1843T>C)', eth: 'North African founder — hepato-renal MKS3 subtype', mech: 'Disulfide bond disruption; TZ transmembrane scaffold loss; JBTS6 when hypomorphic' },
                    { gene: 'TMEM67/MKS3 (8q22.1)', mut: 'p.Gln376* (c.1126C>T)', eth: 'European', mech: 'Truncating; hepato-renal prominent; compound het with p.Cys615Arg' },
                    { gene: 'CEP290 (12q21.32)', mut: 'p.Arg151* (c.451C>T)', eth: 'Finnish / general; exon 5', mech: 'Early null — Meckel-spectrum CEP290; vs IVS26 (mild/LCA10) and p.Arg1933* (JBTS)' },
                    { gene: 'CEP290 (12q21.32)', mut: 'p.Gln1745* / p.Arg1933* (biallelic)', eth: 'European / compound het', mech: 'Biallelic truncating → MKS; each alone (one null + one IVS26) → JBTS or LCA10' },
                    { gene: 'CC2D2A (4p15.33)', mut: 'p.Arg1564* + p.Trp1182* (biallelic null)', eth: 'Middle Eastern / European', mech: 'TZ coiled-coil loss; biallelic null → MKS; hypomorphic → JBTS' },
                    { gene: 'RPGRIP1L (16q12.2)', mut: 'p.Arg649* / splice (exon 14)', eth: 'European', mech: 'TZ structural protein; Dandy-Walker variant; also JBTS7 / NPHP7 when hypomorphic' },
                  ].map((r, i) => (
                    <tr key={i}>
                      <td className="fw-bold" style={{ color: ACCENT4 }}>{r.gene}</td>
                      <td><code>{r.mut}</code></td>
                      <td><small>{r.eth}</small></td>
                      <td><small>{r.mech}</small></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Prenatal diagnosis workflow */}
          <Section title="Prenatal Diagnosis Workflow" color={ACCENT2}>
            <div className="row g-2">
              {[
                { step: '1', title: '18–20 wk Anatomy USS', desc: 'Bilateral large echogenic kidneys + occipital encephalocele + post-axial polydactyly → suspect MKS. Refer to tertiary fetal medicine immediately.', color: ACCENT2 },
                { step: '2', title: 'Exclude Trisomy 13 FIRST', desc: 'Karyotype or SNP microarray MANDATORY before gene panel. Trisomy 13 (Patau) has identical phenotype — cannot distinguish without chromosomes.', color: ACCENT },
                { step: '3', title: 'Fetal MRI', desc: 'Clarifies encephalocele type (occipital vs exencephaly vs Dandy-Walker); corpus callosum; holoprosencephaly (TMEM216). Better brain anatomy than USS.', color: ACCENT4 },
                { step: '4', title: 'Molecular Diagnosis', desc: 'MKS panel (≥30 genes) or WES from amniotic fluid / CVS / chorionic villi / fetal blood. Confirms diagnosis and enables PGT-M planning.', color: ACCENT5 },
                { step: '5', title: 'Counselling & Decision', desc: 'Non-directive counselling: TOPF vs expectant management. Perinatal palliative care plan if continued. Autopsy consent regardless of outcome.', color: ACCENT6 },
                { step: '6', title: 'PGT-M Planning', desc: 'Once causative variant identified: offer PGT-M for next pregnancy (IVF + embryo biopsy). Most effective recurrence prevention (25% AR risk).', color: ACCENT5 },
              ].map(item => (
                <div key={item.step} className="col-md-4">
                  <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${item.color}` }}>
                    <div className="card-body py-2 px-3">
                      <div className="fw-bold" style={{ color: item.color }}>Step {item.step}: {item.title}</div>
                      <div className="small text-muted mt-1">{item.desc}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Differential diagnosis */}
          <Section title="Differential Diagnosis (Prenatal)" color={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-dark">
                  <tr><th>Differential</th><th>Overlapping Features</th><th>How to Exclude</th></tr>
                </thead>
                <tbody>
                  {[
                    { dx: 'Trisomy 13 (Patau)', overlap: 'Renal cysts + holoprosencephaly + polydactyly — identical triad', exclude: 'Karyotype or SNP array → trisomy 13 vs diploid MKS' },
                    { dx: 'Trisomy 18 (Edwards)', overlap: 'Choroid plexus cysts + renal overlap; CNS', exclude: 'Karyotype/microarray; no encephalocele in T18' },
                    { dx: 'Hydrolethalus Syndrome (HYLS1)', overlap: 'Finnish; hydrocephalus + polydactyly + brain malformation; clinically MKS-like', exclude: 'HYLS1 sequencing negative → MKS panel; corpus callosum absent in hydrolethalus' },
                    { dx: 'Smith-Lemli-Opitz (DHCR7)', overlap: 'Polydactyly + brain + renal', exclude: 'Amniotic fluid 7-dehydrocholesterol elevated → SLO; MKS if 7-DHC normal' },
                    { dx: 'Fraser Syndrome (FRAS1/FREM2)', overlap: 'Cryptophthalmos + syndactyly + renal agenesis', exclude: 'FRAS1/FREM2 panel → negative → MKS gene panel' },
                    { dx: 'COACH Syndrome (CC2D2A/TMEM67)', overlap: 'Same genes (CC2D2A, TMEM67) — hypomorphic alleles', exclude: 'COACH is postnatal survivable; MKS lethal; allele class determines' },
                  ].map((r, i) => (
                    <tr key={i}>
                      <td className="fw-bold" style={{ color: ACCENT4 }}>{r.dx}</td>
                      <td><small>{r.overlap}</small></td>
                      <td><small style={{ color: ACCENT5 }}>{r.exclude}</small></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </>
      )}

      {/* ── TAB 3: Definitions ──────────────────────────────── */}
      {tab === 3 && df && (
        <>
          <div className="row g-3">
            {Object.entries(df).map(([section, content]) => (
              <div key={section} className="col-md-6">
                <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT4}` }}>
                  <div className="card-body">
                    <h6 className="fw-bold mb-2" style={{ color: ACCENT4, textTransform: 'capitalize' }}>
                      {section.replace(/_/g, ' ')}
                    </h6>
                    {typeof content === 'object' && !Array.isArray(content) ? (
                      <dl className="mb-0">
                        {Object.entries(content).map(([k, v]) => (
                          <div key={k} className="mb-2">
                            <dt className="small fw-bold" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</dt>
                            {typeof v === 'object' && !Array.isArray(v) ? (
                              <dd className="small text-muted mb-0">
                                <ul className="mb-0 ps-3">
                                  {Object.entries(v).map(([gk, gv]) => (
                                    <li key={gk}><strong>{gk}:</strong> {gv}</li>
                                  ))}
                                </ul>
                              </dd>
                            ) : (
                              <dd className="small text-muted mb-0">{String(v)}</dd>
                            )}
                          </div>
                        ))}
                      </dl>
                    ) : (
                      <p className="small text-muted mb-0">{String(content)}</p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Back nav */}
          <div className="mt-4">
            <Link href="/joubert" className="btn btn-sm btn-outline-secondary me-2">← Joubert Syndrome</Link>
            <Link href="/" className="btn btn-sm btn-outline-primary">Portal Home</Link>
          </div>
        </>
      )}
    </div>
  );
}
