'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Iron & Imaging', 'Treatments', 'Definitions'];
const COLOR = '#1b5e20';   // deep green — MECR/MEPAN (mitochondria, FA synthesis, metabolic)
const LIGHT = '#e8f5e9';

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

function Bar({ label, value, max, color = COLOR }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ variant, text }) {
  const bg = variant === 'danger' ? '#ffebee' : variant === 'warning' ? '#fff8e1' : variant === 'success' ? '#e8f5e9' : '#e8eaf6';
  const border = variant === 'danger' ? '#c62828' : variant === 'warning' ? '#f57f17' : variant === 'success' ? '#2e7d32' : COLOR;
  return (
    <div className="mb-2 p-2 rounded small" style={{ background: bg, borderLeft: `4px solid ${border}` }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview...</div>;
  const kpis = data.kpis || {};
  const phenoDist = data.phenotype_distribution || [];
  const highlights = data.clinical_highlights || [];
  const cis = data.contraindications || [];
  const thresholds = data.thresholds || [];

  return (
    <div>
      <div className="alert py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
        <strong>MECR (1p35.3) — 342 aa mitochondrial trans-2-enoyl-CoA reductase · OMIM Gene 608205 / MEPAN 617282 · AR Biallelic LOF:</strong>{' '}
        Terminal enzyme of mitochondrial fatty acid synthesis type II (mtFAS-II); produces octanoyl-ACP →
        substrate for lipoic acid synthase (LIAS). MECR LOF → no lipoic acid → PDH + alpha-KGDH hypolipoylation
        → pyruvate/lactate accumulation + TCA failure → 3-methylglutaconic aciduria (3-MGA) PATHOGNOMONIC.{' '}
        <strong className="text-danger">VPA ABSOLUTE CI (worsens lipoylation failure → metabolic crisis).</strong>{' '}
        <span className="fw-bold" style={{ color: COLOR }}>
          Israeli Bedouin founder p.Tyr200His (~45% worldwide alleles). LEV PREFERRED (no mitochondrial toxicity).
          Lipoic acid supplementation investigational bypass. GP iron bilateral (NO Eye-of-Tiger — DDx PKAN).
          Optic atrophy 80-90% (early). Cerebellar atrophy 60-70%. 3-MGA in urine 100% of patients.
        </span>
      </div>

      <div className="row g-2 mb-3">
        <KPI label="Cohort (n)" value={kpis.n_patients} color={COLOR} />
        <KPI label="Dystonia" value={`${kpis.dystonia_pct}%`} color={COLOR} />
        <KPI label="Optic Atrophy" value={`${kpis.optic_atrophy_pct}%`} color="#e65100" />
        <KPI label="GP Iron (SWI)" value={`${kpis.gp_iron_pct}%`} color="#bf360c" />
        <KPI label="3-MGA (urine)" value="100%" color="#1a237e" />
        <KPI label="Mean Onset (yr)" value={kpis.mean_onset_yr} color={COLOR} />
        <KPI label="Cerebellar Atrophy" value={`${kpis.cerebellar_atrophy_pct}%`} color="#4a148c" />
        <KPI label="Seizures" value={`${kpis.seizures_pct}%`} color="#b71c1c" />
        <KPI label="Cognitive" value={`${kpis.cognitive_pct}%`} color="#37474f" />
        <KPI label="Lactate ↑" value={`${kpis.lactate_pct}%`} color="#0d47a1" />
        <KPI label="Retinal Dystrophy" value={`${kpis.retinal_pct}%`} color="#880e4f" />
        <KPI label="Mean 3-MGA" value={`${kpis.mean_mga} mmol/cr`} color="#1a237e" />
      </div>

      <SectionCard title="Phenotype Distribution (40 patients, seed-533)">
        <div className="row">
          {phenoDist.map(ph => (
            <div key={ph.phenotype} className="col-6 col-md-3 mb-3">
              <div className="border rounded p-2 text-center h-100">
                <div className="fw-bold fs-5" style={{ color: COLOR }}>{ph.n}</div>
                <div className="small text-muted">{ph.phenotype}</div>
                <div className="badge mt-1" style={{ background: LIGHT, color: COLOR }}>{ph.pct}%</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Clinical Features (% of 40 patients)">
        {highlights.map(h => (
          <div key={h.finding} className="mb-3">
            <Bar label={h.finding} value={h.pct} max={100} />
            <div className="text-muted small ms-1">{h.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Pharmacogenomics — Drug Safety in MEPAN (MECR LOF + Mitochondrial Vulnerability)">
        {cis.map(ci => (
          <Alert
            key={ci.drug}
            variant={
              ci.severity.includes('ABSOLUTE') ? 'danger'
              : ci.severity === 'AVOID' ? 'warning'
              : ci.severity.includes('PREFERRED') ? 'success'
              : 'info'
            }
            text={
              <><strong>{ci.drug} [{ci.severity}]</strong> — {ci.reason}{ci.alternative ? ` → ALT: ${ci.alternative}` : ''}</>
            }
          />
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds & Action Triggers">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ background: LIGHT }}>
              <tr><th>Metric</th><th>Threshold</th><th>Action</th></tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.metric}</td>
                  <td><code>{t.threshold}</code></td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const phenoBreakdown = data.phenotype_breakdown || [];
  const variantBreakdown = data.variant_breakdown || [];
  const treatBreakdown = data.treatment_breakdown || [];
  const mgaBreakdown = data.mga_breakdown || [];
  const vaBreakdown = data.va_breakdown || [];
  const sexDist = data.sex_distribution || [];

  return (
    <div>
      <SectionCard title="Phenotype Breakdown — Clinical Features by Subtype">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>Phenotype</th><th>n</th><th>Onset yr</th><th>Dystonia%</th>
                <th>Optic Atr%</th><th>GP Iron%</th><th>Cerebel%</th>
                <th>Seizures%</th><th>Cognitive%</th><th>Lactate%</th><th>Mean 3-MGA</th>
              </tr>
            </thead>
            <tbody>
              {phenoBreakdown.map(ph => (
                <tr key={ph.phenotype}>
                  <td className="fw-bold">{ph.phenotype}</td>
                  <td>{ph.n}</td>
                  <td>{ph.mean_onset_yr}</td>
                  <td>{ph.dystonia_pct}%</td>
                  <td>{ph.optic_atrophy_pct}%</td>
                  <td>{ph.gp_iron_pct}%</td>
                  <td>{ph.cerebellar_atrophy_pct}%</td>
                  <td>{ph.seizures_pct}%</td>
                  <td>{ph.cognitive_pct}%</td>
                  <td>{ph.lactate_pct}%</td>
                  <td>{ph.mean_mga} mmol/cr</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Variant Distribution (40 allele pairs)">
            {variantBreakdown.map(v => (
              <div key={v.variant} className="mb-2">
                <Bar label={v.variant.split(' — ')[0]} value={v.pct} max={100} />
                <div className="text-muted small ms-1">{v.variant.split(' — ').slice(1).join(' ')}</div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Treatment Distribution">
            {treatBreakdown.map(t => (
              <div key={t.treatment} className="mb-2">
                <Bar label={t.treatment} value={t.pct} max={100} />
              </div>
            ))}
          </SectionCard>
          <SectionCard title="Sex Distribution">
            {sexDist.map(s => (
              <div key={s.sex} className="mb-2">
                <Bar label={s.sex} value={s.pct} max={100} />
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Urine 3-MGA Distribution (mmol/mol creatinine)">
            <div className="mb-2 small text-muted">3-methylglutaconic acid — PATHOGNOMONIC; present 100%; normal &lt;5</div>
            {mgaBreakdown.map(m => (
              <div key={m.range} className="mb-2">
                <Bar label={`${m.range} mmol/cr`} value={m.pct} max={100} color="#1a237e" />
                <div className="text-muted small ms-1">n={m.n}</div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Visual Acuity (LogMAR) Distribution">
            <div className="mb-2 small text-muted">Optic atrophy impact on vision; LogMAR 0=normal, &gt;1=severely impaired</div>
            {vaBreakdown.map(v => (
              <div key={v.category} className="mb-2">
                <Bar label={v.category} value={v.pct} max={100} color="#e65100" />
                <div className="text-muted small ms-1">n={v.n}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

function IronTab({ data: overview }) {
  if (!overview) return <div className="text-center py-4 text-muted">Loading...</div>;
  const kpis = overview.kpis || {};

  const mriFindings = [
    { region: "Globus Pallidus (bilateral)", finding: "SWI/T2* hypointensity", pct: kpis.gp_iron_pct, severity: "Moderate", note: "NO Eye-of-Tiger central bright (DDx PKAN). Uniform bilateral hypointensity. R2*/QSM quantification." },
    { region: "Substantia Nigra", finding: "Mild SWI hypointensity", pct: kpis.sn_iron_pct, severity: "Mild", note: "Less prominent than PKAN/MPAN. Correlates with parkinsonism features (late)." },
    { region: "Cerebellum", finding: "Progressive atrophy", pct: kpis.cerebellar_atrophy_pct, severity: "Moderate-severe", note: "Cortex > white matter; vermis + hemispheres. MRS: reduced NAA (neuronal loss). Ataxia complicates dystonia." },
    { region: "Optic Nerves/Discs", finding: "Pallor (bilateral)", pct: kpis.optic_atrophy_pct, severity: "Moderate-severe", note: "Temporal pallor first (papillomacular bundle). VEP latency prolonged. OCT: RNFL thinning. NOT retinal primary." },
    { region: "Cerebral White Matter", finding: "Normal (no leukodystrophy)", pct: 0, severity: "ABSENT", note: "DDx FAHN/NBIA3 (FA2H) — leukodystrophy prominent in FAHN. Normal WM is a key MEPAN distinguishing feature." },
    { region: "Caudate/Putamen", finding: "Spared (early)", pct: 8, severity: "Rare/late", note: "Unlike MPAN (C19orf12) where striatal iron is common. MEPAN predominantly GP iron." },
  ];

  const ddx = [
    { disease: "PKAN (PANK2/NBIA1)", iron: "GP severe + Eye-of-Tiger", ddx_point: "Eye-of-Tiger sign PRESENT in PKAN (central GP T2 hyperintensity) — ABSENT in MEPAN. PKAN: no 3-MGA; acanthocytes 50%; retinopathy 50%." },
    { disease: "PLAN (PLA2G6/NBIA2)", iron: "GP late/mild; cerebellum dominant", ddx_point: "PLAN: cerebellar cortical atrophy + axonal neuropathy (100%) + ERG very abnormal early. No 3-MGA. Optic atrophy different pattern." },
    { disease: "MPAN (C19orf12/NBIA4)", iron: "GP + SN + striatum", ddx_point: "MPAN: optic atrophy + GP iron — SIMILAR to MEPAN. KEY DDx: MPAN has NO 3-MGA; axonal neuropathy 60%; psychiatric features 40%. C19orf12 vs MECR sequencing." },
    { disease: "BPAN (WDR45/NBIA5)", iron: "SN + GP + T1 halo", ddx_point: "BPAN: T1 GP halo sign PATHOGNOMONIC. X-linked dominant (females 90%). Biphasic (static encephalopathy → sudden parkinsonism/dementia). No 3-MGA." },
    { disease: "FAHN (FA2H/NBIA3)", iron: "GP mild; leukodystrophy dominant", ddx_point: "FAHN: leukodystrophy EARLIEST/MOST PROMINENT — ABSENT in MEPAN. Spastic paraplegia dominant in FAHN. No 3-MGA; no optic atrophy early." },
    { disease: "WSS (DCAF17)", iron: "GP mild; NO cortical iron", ddx_point: "WSS: hypogonadism + alopecia + diabetes — ABSENT in MEPAN. No 3-MGA. Ribosome biogenesis mechanism vs mtFAS-II (MECR)." },
    { disease: "OPA3 (Costeff Type III MGA)", iron: "Usually normal", ddx_point: "OPA3: 3-MGA PRESENT + optic atrophy — DDx closest to MEPAN. KEY: OPA3 has CHOREA (not dystonia dominant), NORMAL brain MRI (no GP iron). OPA3 gene vs MECR." },
  ];

  return (
    <div>
      <div className="alert py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
        <strong>MEPAN MRI Signature:</strong> Bilateral GP SWI hypointensity (NO Eye-of-Tiger) + Cerebellar atrophy + NORMAL white matter.
        Key differentiator: 3-MGA-uria (urine organic acids) present in 100% of MEPAN patients — guides gene sequencing to MECR.
        Annual SWI + cerebellar volumetric MRI + MRS (NAA) mandatory. R2*/QSM for iron quantification.
      </div>

      <SectionCard title="MRI Findings by Region — MEPAN (40-patient cohort)">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: LIGHT }}>
              <tr><th>Region</th><th>Finding</th><th>% Affected</th><th>Severity</th><th>Clinical Note</th></tr>
            </thead>
            <tbody>
              {mriFindings.map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold">{m.region}</td>
                  <td>{m.finding}</td>
                  <td>
                    {m.pct === 0
                      ? <span className="badge bg-success">ABSENT</span>
                      : <span className="fw-bold" style={{ color: m.pct > 70 ? '#b71c1c' : '#e65100' }}>{m.pct}%</span>
                    }
                  </td>
                  <td><span className="badge" style={{ background: m.severity === 'ABSENT' ? '#e8f5e9' : LIGHT, color: COLOR }}>{m.severity}</span></td>
                  <td className="small">{m.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Differential Diagnosis — Iron Accumulation Disorders (NBIA vs MEPAN)">
        <div className="mb-2 small text-muted fw-bold">Key question: bilateral GP iron + dystonia + optic atrophy → MEPAN vs others?</div>
        {ddx.map(d => (
          <div key={d.disease} className="mb-3 p-2 rounded" style={{ background: '#f9f9f9', borderLeft: `3px solid ${COLOR}` }}>
            <div className="d-flex justify-content-between small mb-1">
              <span className="fw-bold">{d.disease}</span>
              <span className="text-muted">{d.iron}</span>
            </div>
            <div className="small">{d.ddx_point}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="3-MGA-uria — Metabolic Fingerprint in MEPAN">
        <div className="row">
          <div className="col-md-6">
            <div className="p-3 rounded mb-3" style={{ background: '#e3f2fd', borderLeft: `4px solid #1565c0` }}>
              <div className="fw-bold small mb-1" style={{ color: '#1565c0' }}>What is 3-MGA (3-Methylglutaconic acid)?</div>
              <div className="small">C6 dicarboxylic acid; generated via isoprenoid/HMG-CoA shunting when mitochondrial function is impaired. Measured on urine organic acid chromatography (GC-MS). Normal: &lt;5 mmol/mol creatinine. MEPAN: 20-100+ mmol/mol creatinine in all patients.</div>
            </div>
            <div className="p-3 rounded" style={{ background: '#fce4ec', borderLeft: `4px solid #c62828` }}>
              <div className="fw-bold small mb-1" style={{ color: '#c62828' }}>3-MGA-uria Type Classification</div>
              <div className="small">
                <div>Type I — AUH (3-MGA hydratase def): isolated, benign</div>
                <div>Type II — Barth (TAZ): cardiomyopathy + neutropenia</div>
                <div>Type III — OPA3/DNAJC19: optic atrophy ± ataxia</div>
                <div className="fw-bold" style={{ color: '#c62828' }}>Type IV — MEPAN/MECR: dystonia + GP iron + optic atrophy (NO cardiac/liver)</div>
                <div>Type V — SERAC1 (MEGDEL): deafness + liver + Leigh MRI</div>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-3 rounded" style={{ background: LIGHT }}>
              <div className="fw-bold small mb-2" style={{ color: COLOR }}>Diagnostic Workup Sequence</div>
              <div className="small">
                {[
                  '1. Urine organic acids (GC-MS): confirm 3-MGA elevated (≥20 mmol/mol creatinine)',
                  '2. Plasma amino acids: glycine (GCS involvement subset)',
                  '3. Plasma lactate/pyruvate: PDH failure indicator',
                  '4. Acylcarnitine profile: C5:1 (3-methylglutaconylcarnitine)',
                  '5. Brain MRI (SWI + T1 + MRS): GP iron + cerebellar atrophy',
                  '6. Ophthalmology: VEP + ERG + OCT (optic atrophy characterisation)',
                  '7. MECR gene sequencing (WES or targeted panel): confirms biallelic variants',
                  '8. POLG screening: mandatory before any AED change',
                  '9. Fibroblast ETR enzyme activity (research, not widely available 2026)',
                ].map((s, i) => <div key={i} className="mb-1">{s}</div>)}
              </div>
            </div>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data: overview }) {
  if (!overview) return <div className="text-center py-4 text-muted">Loading...</div>;
  const cis = overview.contraindications || [];

  const treatments = [
    {
      agent: "Levetiracetam (LEV)", class: "AED — SV2A modulator", evidence: "PREFERRED FIRST-LINE",
      dose: "500-3000 mg/day (paediatric: 20-60 mg/kg/day)", color: "#2e7d32",
      note: "Renal excretion (66% unchanged); no mitochondrial interactions; no CYP450 induction; safe with lipoic acid + riboflavin. Broad-spectrum (myoclonic + focal). First choice in all MEPAN seizures.",
    },
    {
      agent: "VPA (Valproate)", class: "AED — broad-spectrum", evidence: "ABSOLUTE CI",
      dose: "CONTRAINDICATED — no dose is safe", color: "#c62828",
      note: "VPA → CoA sequestration + beta-oxidation inhibition → worsens lipoylation failure → PDH collapse → lactate crisis + hyperammonemia + hepatotoxicity. NEVER use in MEPAN regardless of seizure severity.",
    },
    {
      agent: "Clobazam (CLB)", class: "AED — benzodiazepine (1,5-BZD)", evidence: "Second-line",
      dose: "5-30 mg/day (paediatric: 0.2-1 mg/kg/day)", color: "#f57f17",
      note: "Minimal mitochondrial interaction; useful for focal + myoclonic seizures. Sedation and tolerance monitored. Acceptable second-line if LEV insufficient.",
    },
    {
      agent: "Alpha-Lipoic Acid (R-LA)", class: "mtFAS-II bypass — investigational", evidence: "Level D (rational)",
      dose: "100-600 mg/day R-lipoic acid (no consensus 2026)", color: "#1565c0",
      note: "Exogenous lipoic acid bypasses MECR-dependent synthesis via LIPT1/LIPT2 salvage pathway. In vitro: restores partial PDH lipoylation in MECR-deficient fibroblasts. R-isomer preferred (better mitochondrial uptake). Safe; no major interactions. May slow disease progression — no RCT evidence.",
    },
    {
      agent: "Riboflavin (B2)", class: "Mitochondrial cofactor", evidence: "Supportive",
      dose: "100-200 mg/day", color: "#4a148c",
      note: "FAD-dependent enzyme support (complex I, II, ETF). Safe; cheap; may modestly improve respiratory chain efficiency in mtFAS disorders. Standard supportive therapy in all mitochondrial disease.",
    },
    {
      agent: "CoQ10 (Ubiquinone)", class: "Mitochondrial antioxidant", evidence: "Supportive",
      dose: "10-30 mg/kg/day", color: "#880e4f",
      note: "Electron carrier in respiratory chain; antioxidant; membrane stabiliser. No MECR-specific trial; extrapolated from other mitochondrial diseases. Safe with no major interactions.",
    },
    {
      agent: "Tetrabenazine / Deutetrabenazine", class: "VMAT2 inhibitor — chorea", evidence: "Level D",
      dose: "TBZ: 12.5-100 mg/day; DTBZ: 6-48 mg/day", color: "#37474f",
      note: "VMAT2 inhibition depletes presynaptic dopamine → reduces choreic movements. Level D extrapolated from Huntington. Monitor depression + QTc. Not useful for dystonia component.",
    },
    {
      agent: "Deferiprone (DFP)", class: "Iron chelator — investigational", evidence: "Level D",
      dose: "15-25 mg/kg/day in divided doses", color: "#bf360c",
      note: "Brain-penetrant chelator; reduces GP R2* on MRI (PKAN TIRCON precedent). No MEPAN-specific trial. Adverse effects: agranulocytosis (weekly WBC mandatory), GI. Use only in NBIA centres with informed consent.",
    },
    {
      agent: "GPi-DBS", class: "Neurostimulation — dystonia", evidence: "Level D investigational",
      dose: "Bilateral GPi implant; programming per centre protocol", color: "#00695c",
      note: "Generalised/segmental dystonia target. <5 MEPAN case reports. Metabolic optimisation (lipoic acid + riboflavin) FIRST. Contraindicated in severe cognitive decline or active metabolic decompensation. Anaesthesia risk: avoid propofol (PRIS); use volatile agents.",
    },
  ];

  return (
    <div>
      <div className="alert py-2 small mb-3" style={{ borderLeft: `4px solid #c62828`, background: '#ffebee' }}>
        <strong className="text-danger">CRITICAL — VPA ABSOLUTE CONTRAINDICATION IN MEPAN:</strong>{' '}
        Valproate worsens lipoylation failure (CoA sequestration + beta-oxidation inhibition) → PDH collapse →
        lactic acidosis + hyperammonemia + fulminant hepatic failure. No dose is safe. Switch to LEV immediately.
        POLG1 screening mandatory for all patients (standard mitochondrial disease protocol).
      </div>

      <SectionCard title="Treatment Agents — Evidence & Dosing">
        {treatments.map(t => (
          <div key={t.agent} className="mb-3 p-3 rounded" style={{ borderLeft: `4px solid ${t.color}`, background: '#fafafa' }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold">{t.agent}</span>
              <span className="badge text-white small" style={{ background: t.color }}>{t.evidence}</span>
            </div>
            <div className="small text-muted mb-1">{t.class} | Dose: {t.dose}</div>
            <div className="small">{t.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Treatment Hierarchy — MEPAN Seizure Management">
        <div className="p-3 rounded small" style={{ background: LIGHT }}>
          {[
            '1. STOP VPA immediately if patient arrives on it — switch to LEV bridge.',
            '2. POLG1 gene sequencing — mandatory for all MEPAN patients (standard mitochondrial protocol).',
            '3. LEV first-line AED (myoclonic + focal): start 500 mg/day, titrate to 1500-3000 mg/day.',
            '4. Add CLB (clobazam) if LEV insufficient — second-line, minimal mitochondrial risk.',
            '5. Avoid CBZ/PHT (sodium channel blockers worsen mitochondrial membrane potential).',
            '6. For choreic component: Tetrabenazine (VMAT2 inhibitor, Level D).',
            '7. For dystonia: physio + OT + GPi-DBS candidacy evaluation if severe (metabolic optimised first).',
            '8. Metabolic supplements (parallel, not sequential): Lipoic acid + Riboflavin + CoQ10.',
            '9. Deferiprone: if GP iron severity high + patient counselled on risks (investigational).',
            '10. Annual multidisciplinary review: neurology + metabolic + ophthalmology + neuropsychology.',
          ].map((s, i) => <div key={i} className="mb-1 small">{s}</div>)}
        </div>
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const defs = data.definitions || [];
  const [open, setOpen] = useState(null);

  return (
    <div>
      <div className="mb-3 small text-muted">Click any term to expand the clinical detail.</div>
      {defs.map((d, i) => (
        <div key={d.term} className="mb-2 border rounded" style={{ borderLeft: `3px solid ${COLOR} !important` }}>
          <button
            className="btn btn-link w-100 text-start d-flex justify-content-between align-items-center py-2 px-3 text-decoration-none"
            onClick={() => setOpen(open === i ? null : i)}
          >
            <div>
              <span className="fw-bold small" style={{ color: COLOR }}>{d.term}</span>
              <br /><span className="text-muted small">{d.full}</span>
            </div>
            <span style={{ color: COLOR }}>{open === i ? '▲' : '▼'}</span>
          </button>
          {open === i && (
            <div className="px-3 pb-3 small" style={{ background: LIGHT }}>
              {d.detail}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function MECRPage() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [ov, bk, df] = await Promise.all([
          fetch(`${API}/api/mecr/overview`).then(r => r.json()),
          fetch(`${API}/api/mecr/breakdown`).then(r => r.json()),
          fetch(`${API}/api/mecr/definitions`).then(r => r.json()),
        ]);
        setOverview(ov);
        setBreakdown(bk);
        setDefinitions(df);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const tabContent = [
    <OverviewTab key="ov" data={overview} />,
    <PatientsTab key="bk" data={breakdown} />,
    <IronTab key="ir" data={overview} />,
    <TreatmentsTab key="tx" data={overview} />,
    <DefinitionsTab key="df" data={definitions} />,
  ];

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <div style={{ width: 18, height: 18, background: COLOR, borderRadius: '50%' }} />
        <h5 className="mb-0 fw-bold" style={{ color: COLOR }}>
          MEPAN Syndrome — MECR (Mitochondrial Enoyl-CoA Reductase Deficiency)
        </h5>
        <span className="badge ms-2" style={{ background: LIGHT, color: COLOR }}>
          OMIM 617282 · AR · 1p35.3 · Seed-533
        </span>
      </div>

      <div className="alert py-1 px-3 small mb-3 d-flex gap-3 flex-wrap" style={{ background: '#ffebee', border: 'none' }}>
        <span className="text-danger fw-bold">⚠ VPA ABSOLUTE CI</span>
        <span className="text-warning fw-bold">⚠ CBZ/PHT AVOID</span>
        <span className="text-success fw-bold">✓ LEV PREFERRED</span>
        <span className="fw-bold" style={{ color: '#1565c0' }}>🧪 3-MGA-uria PATHOGNOMONIC (100%)</span>
        <span className="fw-bold" style={{ color: '#1a237e' }}>🔬 Lipoic Acid — Investigational Bypass</span>
        <span className="fw-bold" style={{ color: '#880e4f' }}>👁 Optic Atrophy 80-90%</span>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}
      {loading && <div className="text-center py-3 text-muted small">Loading MEPAN data...</div>}

      <ul className="nav nav-tabs mb-3 flex-wrap">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${activeTab === i ? ' active fw-bold' : ''}`}
              onClick={() => setActiveTab(i)}
              style={activeTab === i ? { color: COLOR, borderBottom: `3px solid ${COLOR}` } : {}}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tabContent[activeTab]}
    </div>
  );
}
