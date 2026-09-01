'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Iron Distribution & Biomarkers', 'Treatments', 'Definitions'];
const COLOR = '#1a237e';   // deep navy-blue — CP/Aceruloplasminemia (copper enzyme + brain iron)
const LIGHT = '#e8eaf6';

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
  const bg = variant === 'danger' ? '#ffebee' : variant === 'warning' ? '#fff8e1' : '#e8f5e9';
  const border = variant === 'danger' ? '#c62828' : variant === 'warning' ? '#f57f17' : '#2e7d32';
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
  const diabetesDist = data.diabetes_distribution || [];
  const retinalDist = data.retinal_distribution || [];
  const highlights = data.clinical_highlights || [];
  const cis = data.contraindications || [];
  const thresholds = data.thresholds || [];

  return (
    <div>
      <div className="alert py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
        <strong>CP (3q23-q24) — 1065aa Ceruloplasmin · OMIM 117700/Aceruloplasminemia 604290 · AR Biallelic LOF:</strong>{' '}
        Multi-copper ferroxidase (6 Cu atoms, 3 cupredoxin domains). CP LOF → ferroxidase deficiency → Fe2+ cannot be exported via FPN1 → iron traps in neurons, RPE, β-cells, hepatocytes.{' '}
        <strong className="text-danger">Classic TRIAD: Brain iron accumulation + Diabetes mellitus (insulin-dependent, β-cell iron toxicity) + Retinal degeneration (RPE iron).</strong>{' '}
        <span className="fw-bold" style={{ color: COLOR }}>
          HIGH serum ferritin (&gt;500 ng/mL) OPPOSITE to FTL (which has LOW ferritin). Serum ceruloplasmin UNDETECTABLE — direct measurable PATHOGNOMONIC marker.
          Cortical T2 hypointensity UNIQUE to CP — not seen in NBIA1-7. Cerebellar ataxia DOMINANT early feature (dentate iron).
          VGB ABSOLUTE CI (additive retinal). VPA/PHT CAUTION (hepatic iron). LEV PREFERRED.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Classic Triad" value={kpis.n_classic} color="#c62828" />
        <KPI label="Partial Triad" value={kpis.n_partial} color="#e65100" />
        <KPI label="Neurological Pred." value={kpis.n_neurological} color="#1565c0" />
        <KPI label="Hepatic Pred." value={kpis.n_hepatic} color="#4a148c" />
        <KPI label="Cerebellar Ataxia" value={`${kpis.cerebellar_ataxia_pct}%`} color="#c62828" />
        <KPI label="Cortical Iron (UNIQUE)" value={`${kpis.cortical_iron_pct}%`} color="#c62828" />
        <KPI label="Dentate Iron" value={`${kpis.dentate_iron_pct}%`} color="#e65100" />
        <KPI label="Hepatic Iron" value={`${kpis.hepatic_iron_pct}%`} color="#e65100" />
        <KPI label="Blepharospasm" value={`${kpis.blepharospasm_pct}%`} color={COLOR} />
        <KPI label="Dementia" value={`${kpis.dementia_pct}%`} color={COLOR} />
        <KPI label="Parkinsonism" value={`${kpis.parkinsonism_pct}%`} color="#4a148c" />
        <KPI label="Seizures" value={`${kpis.has_seizures_pct}%`} color="#e65100" />
        <KPI label="Microcytic Anemia" value={`${kpis.anemia_pct}%`} color="#c62828" />
        <KPI label="Mean Ferritin (ng/mL)" value={kpis.mean_ferritin} color="#c62828" />
        <KPI label="Mean CP (g/L)" value={kpis.mean_ceruloplasmin} color="#c62828" />
      </div>

      {/* Diabetes & Retinal Distributions */}
      <div className="row mb-4">
        <div className="col-md-6">
          <SectionCard title="Diabetes Distribution — β-cell Iron Toxicity">
            {diabetesDist.map(d => (
              <Bar key={d.type} label={`${d.type} (n=${d.n})`} value={d.pct} max={100} color={d.type.includes('Insulin') ? '#c62828' : d.type.includes('Pre') ? '#e65100' : '#388e3c'} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Retinal Status — RPE Iron Degeneration">
            {retinalDist.map(d => (
              <Bar key={d.status} label={`${d.status} (n=${d.n})`} value={d.pct} max={100} color={d.status.includes('Pigmentary') ? '#c62828' : d.status.includes('Early') ? '#f57f17' : '#388e3c'} />
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Clinical Highlights */}
      <SectionCard title="Clinical Feature Frequency">
        {highlights.map(h => (
          <div key={h.finding} className="mb-3">
            <Bar label={h.finding} value={h.pct} max={100} />
            <div className="text-muted" style={{ fontSize: '0.78rem', marginLeft: 4 }}>{h.note}</div>
          </div>
        ))}
      </SectionCard>

      {/* Contraindications */}
      <SectionCard title="Drug Contraindications & Cautions">
        {cis.map(ci => (
          <Alert
            key={ci.drug}
            variant={ci.severity === 'ABSOLUTE' ? 'danger' : ci.severity === 'PREFERRED' ? 'success' : 'warning'}
            text={<><strong>{ci.drug}</strong> — <strong>{ci.severity}</strong>: {ci.reason}. {ci.alternative && <em>Alternative: {ci.alternative}</em>}</>}
          />
        ))}
      </SectionCard>

      {/* Thresholds */}
      <SectionCard title="Clinical Thresholds & Action Points">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-light">
              <tr><th>Metric</th><th>Threshold</th><th>Action</th></tr>
            </thead>
            <tbody>
              {thresholds.map(t => (
                <tr key={t.metric}>
                  <td className="fw-bold">{t.metric}</td>
                  <td><span className="badge" style={{ background: COLOR }}>{t.threshold}</span></td>
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
  const patientTable = data.patient_table || [];

  return (
    <div>
      {/* Phenotype breakdown */}
      <SectionCard title="Phenotype Breakdown — 4 Clinical Subtypes">
        <div className="row">
          {phenoBreakdown.map(ph => (
            <div key={ph.phenotype} className="col-md-6 mb-3">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-body small">
                  <div className="fw-bold mb-2" style={{ color: COLOR }}>{ph.phenotype} — {ph.n} pts ({ph.pct}%)</div>
                  <div>Mean onset: <strong>{ph.mean_onset_yr} yr</strong></div>
                  <div>Cerebellar ataxia: <strong>{ph.cerebellar_ataxia_pct}%</strong></div>
                  <div>Blepharospasm: <strong>{ph.blepharospasm_pct}%</strong></div>
                  <div>Parkinsonism: <strong>{ph.parkinsonism_pct}%</strong></div>
                  <div>Cortical iron: <strong>{ph.cortical_iron_pct}%</strong></div>
                  <div>Hepatic iron: <strong>{ph.hepatic_iron_pct}%</strong></div>
                  <div>Microcytic anemia: <strong>{ph.anemia_pct}%</strong></div>
                  <div>Seizures: <strong>{ph.seizures_pct}%</strong></div>
                  <div>Mean ferritin: <strong>{ph.mean_ferritin} ng/mL</strong></div>
                  <div>Mean CP: <strong>{ph.mean_ceruloplasmin} g/L</strong></div>
                  <div>Mean Hb: <strong>{ph.mean_hb} g/dL</strong></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Variant breakdown */}
      <SectionCard title="CP Variant Distribution (biallelic LOF)">
        {variantBreakdown.map(v => (
          <Bar key={v.variant} label={`${v.variant} (n=${v.n})`} value={v.pct} max={100} />
        ))}
      </SectionCard>

      {/* Treatment breakdown */}
      <SectionCard title="Treatment Distribution">
        {treatBreakdown.map(t => (
          <Bar key={t.treatment} label={`${t.treatment} (n=${t.n})`} value={t.pct} max={100} color="#1565c0" />
        ))}
      </SectionCard>

      {/* Patient table */}
      <SectionCard title="Patient Table — Top 20 by Ferritin">
        <div className="table-responsive">
          <table className="table table-sm table-striped small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Phenotype</th><th>Onset</th><th>Dur</th>
                <th>Ferritin</th><th>CP (g/L)</th><th>Hb</th>
                <th>Ataxia</th><th>Blephar.</th><th>Diabetes</th><th>Seizures</th><th>Treatment</th>
              </tr>
            </thead>
            <tbody>
              {patientTable.map(p => (
                <tr key={p.id}>
                  <td className="fw-bold">{p.id}</td>
                  <td>{p.phenotype}</td>
                  <td>{p.onset_yr}y</td>
                  <td>{p.disease_dur_yr}y</td>
                  <td><strong style={{ color: p.ferritin > 2000 ? '#c62828' : '#e65100' }}>{p.ferritin}</strong></td>
                  <td><strong style={{ color: '#c62828' }}>{p.ceruloplasmin}</strong></td>
                  <td>{p.hb}</td>
                  <td>{p.cerebellar_ataxia ? '✓' : '—'}</td>
                  <td>{p.blepharospasm ? '✓' : '—'}</td>
                  <td className="small">{p.diabetes}</td>
                  <td>{p.has_seizures ? '⚡' : '—'}</td>
                  <td className="small">{p.treatment}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function IronTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const ironRegions = data.iron_regions || [];

  return (
    <div>
      <SectionCard title="Iron Distribution — Unique Pattern in Aceruloplasminemia">
        <div className="alert alert-info small mb-3">
          <strong>Key differentiator from classic NBIA (NBIA1-7):</strong> Aceruloplasminemia is the ONLY iron accumulation disorder
          with diffuse <strong>cortical T2 hypointensity</strong> — caused by loss of GPI-anchored CP in cortical astrocytes.
          Classic NBIA (PKAN/MPAN/FAHN/BPAN/CoPAN/FTL) do NOT have cortical iron involvement.
        </div>
        {ironRegions.map(r => (
          <div key={r.region} className="mb-3">
            <Bar label={r.region} value={r.pct} max={100} color={COLOR} />
            <div className="text-muted" style={{ fontSize: '0.78rem', marginLeft: 4 }}>{r.note}</div>
          </div>
        ))}
        <div className="card mt-3" style={{ background: LIGHT }}>
          <div className="card-body small">
            <strong>MRI Pattern Summary:</strong>
            <ul className="mb-0 mt-1">
              <li><strong>T2/GRE/SWI hypointensity:</strong> Bilateral symmetric — cortex (frontal/parietal) + dentate nuclei + basal ganglia + thalamus</li>
              <li><strong>No eye-of-tiger sign</strong> (DDx PKAN/NBIA1)</li>
              <li><strong>No cavitations</strong> (DDx FTL/Neuroferritinopathy)</li>
              <li><strong>No leukodystrophy</strong> (DDx FAHN/NBIA3)</li>
              <li><strong>Cortical hypointensity</strong> — UNIQUE to CP; absent in all classic NBIA subtypes</li>
              <li><strong>QSM/R2* quantification:</strong> Used to monitor chelation response over time</li>
              <li><strong>Pancreatic iron:</strong> Low T2 signal in pancreas (β-cell iron) — supports diagnosis</li>
              <li><strong>Hepatic iron:</strong> Low T2 signal in liver (similar to hereditary haemochromatosis)</li>
            </ul>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Biomarker Profile — Diagnostic Panel">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-light">
              <tr><th>Biomarker</th><th>CP Finding</th><th>Normal</th><th>DDx Comparator</th></tr>
            </thead>
            <tbody>
              <tr>
                <td className="fw-bold">Serum ceruloplasmin</td>
                <td><strong className="text-danger">Undetectable (&lt;0.2 g/L)</strong></td>
                <td>0.25–0.63 g/L</td>
                <td>FTL: normal; HFE: high; Wilson: low</td>
              </tr>
              <tr>
                <td className="fw-bold">Serum ferritin</td>
                <td><strong className="text-danger">HIGH &gt;500–5000 ng/mL</strong></td>
                <td>&lt;200–300 ng/mL</td>
                <td>FTL (NBIA7): <strong>LOW</strong> &lt;30 ng/mL (OPPOSITE)</td>
              </tr>
              <tr>
                <td className="fw-bold">Serum copper</td>
                <td><strong className="text-warning">Low 0.3–0.6 µmol/L</strong></td>
                <td>11–22 µmol/L</td>
                <td>95% plasma copper bound to CP → low CP → low Cu</td>
              </tr>
              <tr>
                <td className="fw-bold">Plasma ferroxidase</td>
                <td><strong className="text-danger">Undetectable</strong></td>
                <td>Active</td>
                <td>Diagnostic confirmation; most specific test</td>
              </tr>
              <tr>
                <td className="fw-bold">Hemoglobin</td>
                <td>9–12 g/dL (microcytic)</td>
                <td>12–17 g/dL</td>
                <td>Iron deficiency: LOW ferritin + high TIBC; CP: HIGH ferritin + normal TIBC</td>
              </tr>
              <tr>
                <td className="fw-bold">Serum iron / TIBC</td>
                <td>Normal or low-normal</td>
                <td>Normal</td>
                <td>NOT classical iron deficiency — iron trapped in cells, not systemically depleted</td>
              </tr>
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;

  const treatments = [
    {
      name: "Deferiprone (DFP) — Brain-Penetrant Iron Chelation",
      level: "Level D (case series, n=15-30 patients)",
      dose: "25 mg/kg/day in 3 divided doses",
      mechanism: "Lipophilic → crosses BBB → chelates brain Fe3+ → urine excretion",
      note: "Preferred for brain iron reduction (R2* MRI improvement in 60-70%). Weekly CBC (agranulocytosis 1% risk).",
      color: "#1565c0",
    },
    {
      name: "Deferasirox — Oral Chelator (Hepatic Iron Focus)",
      level: "Level D (case reports + small series)",
      dose: "20–30 mg/kg/day orally",
      mechanism: "Oral chelator; primarily hepatic iron clearance; less BBB penetration than DFP",
      note: "Used when hepatic iron dominant or DFP not tolerated. Renal monitoring (eGFR). Some brain iron effect.",
      color: "#1565c0",
    },
    {
      name: "Fresh Frozen Plasma (FFP) — Ferroxidase Restoration",
      level: "Level D (case reports; Miyajima 2003)",
      dose: "10–15 mL/kg IV infusion",
      mechanism: "Contains native CP → restores plasma ferroxidase activity for 12-24h",
      note: "Proof-of-concept therapy. Transient motor improvement documented. Not practical chronic use. Bridge therapy.",
      color: "#4a148c",
    },
    {
      name: "Zinc Supplementation — Iron Absorption Block",
      level: "Empirical / supportive",
      dose: "50 mg elemental zinc TID",
      mechanism: "Induces intestinal metallothionein → blocks duodenal iron absorption → reduces iron loading",
      note: "Secondary prevention. Reduces ongoing iron accumulation. Well-tolerated. Used adjunctively with chelation.",
      color: "#2e7d32",
    },
    {
      name: "Insulin — Diabetes Management",
      level: "Mandatory (β-cell destruction → absolute insulin deficiency)",
      dose: "Multiple daily injections or pump; dose individualised",
      mechanism: "Replaces absent insulin (pancreatic β-cell iron toxicity → β-cell destruction)",
      note: "Oral antidiabetics ineffective (β-cell failure, not insulin resistance). Endocrinology co-management essential.",
      color: "#e65100",
    },
    {
      name: "Vitamin E (alpha-tocopherol) — Antioxidant",
      level: "Supportive / empirical",
      dose: "400–800 IU/day",
      mechanism: "Lipid-soluble antioxidant → reduces Fe2+-mediated Fenton chemistry → neuroprotection",
      note: "No RCT. Adjunctive use with chelation. Generally well-tolerated. May slow oxidative neurodegeneration.",
      color: "#2e7d32",
    },
    {
      name: "Botulinum Toxin Type A — Blepharospasm",
      level: "Level B (established for blepharospasm; extrapolated to CP)",
      dose: "Periocular injection 12.5–25 U/eye; repeat every 12 wk",
      mechanism: "SNARE inhibition → NMJ block → orbicularis oculi relaxation → eyelid opening",
      note: "PREFERRED treatment for blepharospasm in CP. Repeatable. Local effect. No systemic iron interaction.",
      color: "#1565c0",
    },
  ];

  return (
    <div>
      <SectionCard title="Key Contraindications — Priority Reminder">
        <Alert variant="danger" text={<><strong>VGB (vigabatrin) ABSOLUTE CI</strong> — additive retinal toxicity on top of CP retinal iron degeneration → accelerated irreversible visual loss. Ophthalmology baseline + annual ERG/visual field mandatory.</>} />
        <Alert variant="danger" text={<><strong>Iron supplementation (oral/IV) ABSOLUTE CI</strong> — worsens iron overload; iron cannot be exported (CP LOF); increases cellular Fenton chemistry-mediated toxicity.</>} />
        <Alert variant="warning" text={<><strong>VPA CAUTION</strong> — hepatotoxic + hepatic iron; POLG1 screen mandatory before use. <strong>PHT CAUTION</strong> — hepatic CYP2C9/2C19 metabolism altered by iron-loaded liver.</>} />
        <Alert variant="success" text={<><strong>LEV PREFERRED first-line AED</strong> — minimal hepatic metabolism (renal excretion); safe with hepatic iron overload. LTG second-line.</>} />
      </SectionCard>

      {treatments.map(t => (
        <div key={t.name} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${t.color}` }}>
          <div className="card-body small">
            <div className="fw-bold mb-1" style={{ color: t.color }}>{t.name}</div>
            <div><strong>Evidence:</strong> {t.level}</div>
            <div><strong>Dose:</strong> {t.dose}</div>
            <div><strong>Mechanism:</strong> {t.mechanism}</div>
            <div className="text-muted mt-1">{t.note}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const defs = data.definitions || [];
  return (
    <div>
      {defs.map(d => (
        <div key={d.term} className="card mb-3 shadow-sm">
          <div className="card-body">
            <div className="fw-bold small mb-1" style={{ color: COLOR }}>{d.term.replace(/-/g, ' ')}</div>
            <div className="text-muted small mb-1 fst-italic">{d.full}</div>
            <div className="small" style={{ whiteSpace: 'pre-wrap' }}>{d.detail}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function CPPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/cp/overview`).then(r => r.json()),
      fetch(`${API}/api/cp/breakdown`).then(r => r.json()),
      fetch(`${API}/api/cp/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: COLOR }} /></div>;
  if (error) return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  const panels = [
    <OverviewTab key="ov" data={overview} />,
    <PatientsTab key="pt" data={breakdown} />,
    <IronTab key="ir" data={breakdown} />,
    <TreatmentsTab key="tr" data={breakdown} />,
    <DefinitionsTab key="df" data={definitions} />,
  ];

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
          🧬 CP Aceruloplasminemia (Ceruloplasmin Deficiency — OMIM 604290)
        </h4>
        <div className="text-muted small">
          CP (3q23-q24) · 1065aa Multi-Copper Ferroxidase · AR Biallelic LOF · ~100-200 cases worldwide 2026 ·
          Classic Triad: Brain Iron + Diabetes (insulin-dependent) + Retinal Degeneration ·
          HIGH ferritin (OPPOSITE of FTL) · Ceruloplasmin undetectable PATHOGNOMONIC ·
          Cortical iron UNIQUE (not seen in NBIA1-7) · Cerebellar ataxia dominant ·
          40-patient cohort seed-529 · OMIM gene 117700 · Yoshida 1995
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {panels[tab]}
    </div>
  );
}
