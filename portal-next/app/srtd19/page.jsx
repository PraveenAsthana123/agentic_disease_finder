'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'CEP120 Centriole Elongation & JBTS31', 'Definitions'];

// SRTD19 colour scheme — CEP120 / centriole elongation / JBTS31 allelism / SRTD–JBTS spectrum
const ACCENT  = '#1a237e';   // deep navy — centriole/centrosome; CPAP-CEP120 axis
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax / neonatal respiratory; severity
const ACCENT3 = '#0d47a1';   // deep blue — renal cysts/NPHP-like; transplant curative
const ACCENT4 = '#4a148c';   // deep purple — retinal rod-cone dystrophy; connecting cilium
const ACCENT5 = '#e65100';   // burnt orange — hepatic CHF / ductal plate; secondary
const ACCENT6 = '#37474f';   // dark slate — centriole assembly architecture; molecular
const ACCENT7 = '#00695c';   // deep teal — JBTS31 allelism / cerebellar MTS overlap
const ACCENT8 = '#880e4f';   // deep pink — polydactyly; postaxial; VEPTR/surgery
const ACCENT9 = '#f57f17';   // amber — SRPS spectrum; perinatal lethal severe alleles

const SEED = 415;

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

function SimpleBar({ label, n, total, color }) {
  const pct = total > 0 ? Math.round(n / total * 100) : 0;
  return (
    <div className="mb-1">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{n} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function SRTD19Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOver]   = useState(null);
  const [breakdown, setBreak] = useState(null);
  const [defs, setDefs]       = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd19/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd19/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd19/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOver(o); setBreak(b); setDefs(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err)       return <div className="alert alert-danger m-4">API error: {err}</div>;
  if (!overview) return <div className="text-center p-5"><div className="spinner-border" /></div>;

  const k = overview.kpis;
  const N = overview.cohort_n;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <Link href="/" className="btn btn-sm btn-outline-secondary">← Back</Link>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
            CEP120 Short-Rib Thoracic Dysplasia 19 (SRTD19 / ATD19)
          </h4>
          <div className="text-muted small">
            OMIM #617895 · *613446 · 5q23.2 · 1,007 aa · Centriole Elongation Factor (CCDC100) · AR · ~1/1.5M–4M · seed={SEED}
          </div>
          <div className="text-muted small">
            Also allelic: JBTS31 (Joubert Syndrome Type 31, OMIM #617562) — allele class governs SRTD19 vs JBTS31 spectrum
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          <Alert color={ACCENT}>
            <strong>CEP120 (SRTD19)</strong> is the first major <strong>centriole ELONGATION FACTOR</strong> SRTD gene —
            mechanistically distinct from all IFT-A (SRTD2–18) and dynein-2 (SRTD3) subtypes.
            CEP120 acts <em>upstream</em> of IFT: without a functional basal body, IFT loading itself is impaired.
            Also allelic with <strong>JBTS31</strong> (Joubert Syndrome Type 31) — hypomorphic alleles cause mild MTS instead of SRTD.
          </Alert>

          {/* KPIs */}
          <div className="row g-2 mb-3">
            <KPI label="Narrow Thorax (severe)" value={`${k.thorax_severe_n} (${k.thorax_severe_pct}%)`} color={ACCENT2} />
            <KPI label="Polydactyly (any)"      value={`${k.polydactyly_n} (${k.polydactyly_pct}%)`}   color={ACCENT8} />
            <KPI label="Renal (any)"            value={`${k.renal_any_n} (${k.renal_any_pct}%)`}       color={ACCENT3} />
            <KPI label="Retinal Dystrophy"      value={`${k.retinal_any_n} (${k.retinal_any_pct}%)`}   color={ACCENT4} />
            <KPI label="Hepatic CHF"            value={`${k.hepatic_chf_n} (${k.hepatic_chf_pct}%)`}   color={ACCENT5} />
            <KPI label="JBTS31 Overlap"         value={`${k.jbts31_n} (${k.jbts31_pct}%)`}            color={ACCENT7} />
            <KPI label="VEPTR / MAGEC"          value={`${k.veptr_any_n} (${k.veptr_any_pct}%)`}       color={ACCENT6} />
            <KPI label="SRPS Spectrum"          value={`${k.srps_n} (${k.srps_pct}%)`}                color={ACCENT9} />
            <KPI label="Renal Tx (curative)"    value={`${k.transplant_done_n}`}                       color={ACCENT3} />
            <KPI label="Misdiagnosed initially" value={`${k.misdiagnosis_n} (${k.misdiagnosis_pct}%)`} color={ACCENT9} />
          </div>

          {/* Mechanism */}
          <Section title="Molecular Mechanism — CEP120 Centriole Elongation Failure" color={ACCENT}>
            <p className="small">{overview.mechanism}</p>
          </Section>

          {/* Key Distinction */}
          <Section title="Key Diagnostic Distinction vs Other SRTDs" color={ACCENT2}>
            <p className="small">{overview.key_distinction}</p>
          </Section>

          {/* Sex + Age */}
          <div className="row mb-3">
            <div className="col-md-4">
              <Section title="Sex Distribution" color={ACCENT6}>
                <SimpleBar label="Male"   n={overview.sex_split.M} total={N} color={ACCENT}  />
                <SimpleBar label="Female" n={overview.sex_split.F} total={N} color={ACCENT8} />
              </Section>
            </div>
            <div className="col-md-8">
              <Section title="Age at Diagnosis" color={ACCENT6}>
                <SimpleBar label="0–1 yr (neonatal)"   n={overview.age_distribution.dx_0_1yr}   total={N} color={ACCENT2} />
                <SimpleBar label="2–5 yr (infant)"     n={overview.age_distribution.dx_2_5yr}   total={N} color={ACCENT}  />
                <SimpleBar label="6–10 yr (child)"     n={overview.age_distribution.dx_6_10yr}  total={N} color={ACCENT4} />
                <SimpleBar label="11–16 yr (teen)"     n={overview.age_distribution.dx_11_16yr} total={N} color={ACCENT6} />
              </Section>
            </div>
          </div>

          {/* Centriole Elongation Factor Table */}
          <Section title="Centriole Elongation Pathway — CEP120 Context" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-dark">
                  <tr>
                    <th>Factor</th><th>Role</th><th>Disease (LOF)</th><th>OMIM Gene</th><th>Chr</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.centriole_elongation_table || []).map((r, i) => (
                    <tr key={i} style={r.factor === 'CEP120' ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                      <td>{r.factor}</td>
                      <td>{r.role}</td>
                      <td>{r.disease}</td>
                      <td>{r.omim_gene}</td>
                      <td>{r.chr}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="text-muted small">
              CEP120 highlighted — only SRTD gene in the centriole elongation (CPAP–CEP120–CEP135) pathway.
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 1: DIAGNOSTIC BREAKDOWN ── */}
      {tab === 1 && breakdown && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="Thorax Severity Distribution" color={ACCENT2}>
                {breakdown.thorax_distribution.map((r, i) => (
                  <SimpleBar key={i} label={r.label} n={r.n} total={N}
                    color={r.label.includes('Severe') ? ACCENT2 : r.label.includes('Moderate') ? ACCENT : ACCENT6} />
                ))}
              </Section>

              <Section title="Polydactyly Distribution" color={ACCENT8}>
                {breakdown.polydactyly_distribution.map((r, i) => (
                  <SimpleBar key={i} label={r.label} n={r.n} total={N}
                    color={r.label === 'None' ? ACCENT6 : ACCENT8} />
                ))}
              </Section>

              <Section title="JBTS31 Overlap (Cerebellar / MTS)" color={ACCENT7}>
                {breakdown.jbts31_distribution.map((r, i) => (
                  <SimpleBar key={i} label={r.label} n={r.n} total={N}
                    color={r.label.includes('hypoplasia') ? ACCENT7 : ACCENT6} />
                ))}
                <div className="text-muted small mt-1">Mild cerebellar vermis hypoplasia + MTS in hypomorphic alleles → JBTS31 spectrum</div>
              </Section>

              <Section title="VEPTR / Thoracic Management" color={ACCENT6}>
                {breakdown.veptr_distribution.map((r, i) => (
                  <SimpleBar key={i} label={r.label} n={r.n} total={N}
                    color={r.label.includes('VEPTR') ? ACCENT : r.label.includes('MAGEC') ? ACCENT4 : r.label.includes('SRPS') ? ACCENT9 : ACCENT6} />
                ))}
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="Renal Status" color={ACCENT3}>
                {breakdown.renal_distribution.map((r, i) => (
                  <SimpleBar key={i} label={r.label} n={r.n} total={N}
                    color={r.label === 'None' ? ACCENT6 : ACCENT3} />
                ))}
              </Section>

              <Section title="CKD Stage" color={ACCENT3}>
                {breakdown.ckd_stage_distribution.map((r, i) => (
                  <SimpleBar key={i} label={r.label} n={r.n} total={N}
                    color={r.label.includes('ESRD') ? ACCENT2 : r.label.includes('3–4') ? ACCENT3 : ACCENT6} />
                ))}
              </Section>

              <Section title="Retinal Status" color={ACCENT4}>
                {breakdown.retinal_distribution.map((r, i) => (
                  <SimpleBar key={i} label={r.label} n={r.n} total={N}
                    color={r.label === 'None' ? ACCENT6 : ACCENT4} />
                ))}
              </Section>

              <Section title="Hepatic Status" color={ACCENT5}>
                {breakdown.hepatic_distribution.map((r, i) => (
                  <SimpleBar key={i} label={r.label} n={r.n} total={N}
                    color={r.label === 'None' ? ACCENT6 : ACCENT5} />
                ))}
              </Section>
            </div>
          </div>

          {/* Allele + Ethnicity */}
          <div className="row">
            <div className="col-md-6">
              <Section title="Allele Class Distribution" color={ACCENT}>
                {breakdown.allele_class_summary.map((r, i) => (
                  <SimpleBar key={i} label={r.label} n={r.n} total={N} color={[ACCENT, ACCENT2, ACCENT9, ACCENT4, ACCENT7][i % 5]} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Ethnicity Distribution" color={ACCENT6}>
                {breakdown.ethnicity_distribution.map((r, i) => (
                  <SimpleBar key={i} label={r.ethnicity} n={r.n} total={N} color={[ACCENT, ACCENT2, ACCENT4, ACCENT3, ACCENT5, ACCENT6][i % 6]} />
                ))}
              </Section>
            </div>
          </div>

          {/* Top Variants */}
          <Section title="Most Frequent Pathogenic Variants (Cohort)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small">
                <thead className="table-dark">
                  <tr><th>Variant</th><th>n</th></tr>
                </thead>
                <tbody>
                  {breakdown.top_variants.map((v, i) => (
                    <tr key={i}><td>{v.variant}</td><td><strong>{v.n}</strong></td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Misdiagnosis */}
          <Section title="Initial Misdiagnosis Distribution" color={ACCENT9}>
            {breakdown.misdiagnosis_distribution.map((r, i) => (
              <SimpleBar key={i} label={r.label} n={r.n} total={N}
                color={r.label === 'None' ? ACCENT6 : ACCENT9} />
            ))}
            <Alert color={ACCENT9}>
              <strong>{Math.round((breakdown.misdiagnosis_distribution.filter(r => r.label !== 'None').reduce((s, r) => s + r.n, 0)) / N * 100)}%</strong> were
              initially misdiagnosed (most common: SRTD3/DYNC2H1 or SRTD5/WDR19). Gene panel sequencing with CEP120 included is the ONLY reliable differentiator.
            </Alert>
          </Section>
        </div>
      )}

      {/* ── TAB 2: CEP120 CENTRIOLE ELONGATION & JBTS31 ── */}
      {tab === 2 && defs && (
        <div>
          <Alert color={ACCENT7}>
            <strong>SRTD19–JBTS31 allele-class spectrum:</strong> Biallelic null → SRPS (perinatal lethal);
            biallelic missense → SRTD19 (liveborn, VEPTR); mild hypomorphic → JBTS31 (Joubert, mild MTS).
            This is the same allele-class spectrum principle as SRTD16/JBTS23 (KIAA0586/TALPID3).
          </Alert>

          {/* Domain architecture */}
          <Section title="CEP120 Protein Domain Architecture (1,007 aa)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-dark">
                  <tr><th>Domain</th><th>Residues</th><th>Function</th><th>Pathogenic Cluster</th></tr>
                </thead>
                <tbody>
                  <tr style={{ background: ACCENT + '18' }}>
                    <td><strong>N-terminal CPAP-binding</strong></td><td>aa 1–200</td>
                    <td>Direct CPAP/CENPJ interaction; procentriole elongation seeding</td>
                    <td>Arg200Gln (MENA founder) — most common SRTD19 allele</td>
                  </tr>
                  <tr style={{ background: ACCENT4 + '12' }}>
                    <td><strong>Central coiled-coil 1</strong></td><td>aa 201–550</td>
                    <td>CEP135 + SSNA1/NA14 recruitment; self-oligomerization ring</td>
                    <td>Leu408Pro (South Asian); Arg329Cys (pan-ethnic)</td>
                  </tr>
                  <tr style={{ background: ACCENT7 + '12' }}>
                    <td><strong>Central coiled-coil 2</strong></td><td>aa 551–750</td>
                    <td>TULP3 interaction; ciliary transport coordination</td>
                    <td>Gly605Glu (European); c.1344+1G>A splice</td>
                  </tr>
                  <tr style={{ background: ACCENT9 + '12' }}>
                    <td><strong>C-terminal domain</strong></td><td>aa 751–1,007</td>
                    <td>PCM anchoring; subdistal appendage positioning; basal body membrane anchor</td>
                    <td>Glu788Ter — truncating null → SRPS spectrum</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* Key variants table */}
          <Section title="Key Pathogenic Variants in CEP120" color={ACCENT2}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-dark">
                  <tr><th>Variant</th><th>Domain</th><th>Consequence</th><th>Ethnicity</th></tr>
                </thead>
                <tbody>
                  {defs.key_variants.map((v, i) => (
                    <tr key={i}>
                      <td><code>{v.variant}</code></td>
                      <td>{v.domain}</td>
                      <td>{v.consequence}</td>
                      <td>{v.ethnicity}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Differential diagnosis */}
          <Section title="Differential Diagnosis — CEP120 (SRTD19) vs Other SRTDs" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-dark">
                  <tr><th>Disease</th><th>Key Differentiator</th></tr>
                </thead>
                <tbody>
                  {defs.ddx_table.map((r, i) => (
                    <tr key={i}><td><strong>{r.disease}</strong></td><td>{r.key_difference}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* JBTS31 allelism callout */}
          <Section title="JBTS31 Allelism — CEP120 SRTD–Joubert Spectrum" color={ACCENT7}>
            <Alert color={ACCENT7}>
              <strong>Allele class governs phenotype:</strong>
              <ul className="mb-0 mt-1 small">
                <li><strong>Biallelic null (truncating)</strong> → SRPS spectrum (perinatal lethal narrow thorax)</li>
                <li><strong>Biallelic missense (moderate)</strong> → SRTD19 (liveborn, narrow thorax, VEPTR)</li>
                <li><strong>Mild hypomorphic (any domain)</strong> → JBTS31 (Joubert: mild MTS + cerebellar vermis hypoplasia; no narrow thorax)</li>
              </ul>
            </Alert>
            <p className="small text-muted">
              Same allele-class spectrum principle as KIAA0586/TALPID3 (SRTD16 / JBTS23).
              Brain MRI (molar tooth sign) and skeletal survey together define the clinical tier.
              Gene panel with CEP120 is essential for both SRTD and JBTS panels.
            </p>
          </Section>
        </div>
      )}

      {/* ── TAB 3: DEFINITIONS ── */}
      {tab === 3 && defs && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="Gene Card — CEP120" color={ACCENT}>
                <table className="table table-sm small">
                  <tbody>
                    {Object.entries(defs.gene_card).map(([k, v]) => (
                      <tr key={k}><th className="text-muted" style={{ width: '35%' }}>{k.replace(/_/g,' ')}</th><td>{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Disease Card — SRTD19" color={ACCENT2}>
                <table className="table table-sm small">
                  <tbody>
                    {Object.entries(defs.disease_card).map(([k, v]) => (
                      <tr key={k}><th className="text-muted" style={{ width: '35%' }}>{k.replace(/_/g,' ')}</th><td>{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>
          </div>

          {/* Diagnostic workup */}
          <Section title="Diagnostic Workup" color={ACCENT4}>
            <ol className="small">
              {defs.diagnostic_workup.map((s, i) => <li key={i}>{s}</li>)}
            </ol>
          </Section>

          {/* Mechanism glossary */}
          <Section title="Mechanism Glossary" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-dark"><tr><th>Term</th><th>Definition</th></tr></thead>
                <tbody>
                  {defs.mechanism_glossary.map((g, i) => (
                    <tr key={i}><td><strong>{g.term}</strong></td><td>{g.definition}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Treatment summary */}
          <Section title="Treatment Summary" color={ACCENT3}>
            <ol className="small">
              {defs.treatment_summary.map((s, i) => <li key={i}>{s}</li>)}
            </ol>
            <Alert color={ACCENT3}>
              <strong>Renal transplant is CURATIVE</strong> — CEP120 defect is cell-autonomous. Donor kidney (CEP120+) functions normally. No post-transplant recurrence.
            </Alert>
          </Section>
        </div>
      )}
    </div>
  );
}
