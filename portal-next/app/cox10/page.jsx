'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#00695c';   // deep teal — renal/tubulopathy + Leigh/COX10
const LIGHT = '#e0f2f1';

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

function Bar({ label, value, color = COLOR }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ variant, text }) {
  const bg = variant === 'danger' ? '#ffebee' : variant === 'warning' ? '#fff8e1' : variant === 'success' ? '#e8f5e9' : LIGHT;
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

function Spinner() {
  return <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /></div>;
}

// ── Tab 1: Overview ──────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="Gene & Disease Identity">
        <div className="row g-2 small">
          {[
            ['Gene', data.gene],
            ['Protein', data.protein],
            ['Disease', data.disease],
            ['OMIM Gene', data.omim_gene],
            ['OMIM Disease', data.omim_disease],
            ['Locus', data.chromosome],
            ['Inheritance', data.inheritance],
            ['Onset', data.onset],
          ].map(([k, v]) => (
            <div key={k} className="col-12 col-md-6">
              <span className="fw-semibold">{k}:</span> <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Mechanism — COX10 / Heme a Biosynthesis Step 1 / Complex IV COX1 Assembly">
        <p className="small text-muted mb-0">{data.mechanism}</p>
      </SectionCard>

      <SectionCard title="RENAL TUBULOPATHY (FANCONI SYNDROME) — Distinguishing COX10 from SURF1 (Both Cause Isolated COX Deficiency + Leigh Syndrome)" borderColor="#1565c0">
        <p className="small text-muted mb-0" style={{ whiteSpace: 'pre-line' }}>{data.tubulopathy_note}</p>
      </SectionCard>

      <SectionCard title={`KPIs — ${data.cohort}`}>
        <div className="row g-2">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Feature Frequencies (40-patient cohort, seed-599)">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar
            key={feat}
            label={feat}
            value={pct}
            color={
              feat.toLowerCase().includes('cardinal') ? '#004d40' :
              feat.toLowerCase().includes('tubulopathy') || feat.toLowerCase().includes('fanconi') || feat.toLowerCase().includes('glucosuria') ? '#1565c0' :
              feat.toLowerCase().includes('lactic') || feat.toLowerCase().includes('died') ? '#b71c1c' :
              feat.toLowerCase().includes('no ') || feat.toLowerCase().includes('normal') ? '#2e7d32' :
              feat.toLowerCase().includes('alive') ? '#2e7d32' :
              COLOR
            }
          />
        ))}
      </SectionCard>

      <SectionCard title="Absolute Contraindications & Drug Safety" borderColor="#c62828">
        {(data.contraindications || []).map(ci => (
          <div key={ci.drug} className="mb-3">
            <Alert
              variant={ci.severity.startsWith('ABSOLUTE') ? 'danger' : ci.severity.startsWith('AVOID') ? 'warning' : 'warning'}
              text={<><strong>{ci.drug}</strong> — {ci.severity}: {ci.mechanism}</>}
            />
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients ───────────────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="Feature Frequencies (40-patient COX10 cohort, seed-599)">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar
            key={feat}
            label={feat}
            value={pct}
            color={
              feat.toLowerCase().includes('tubulopathy') || feat.toLowerCase().includes('fanconi') || feat.toLowerCase().includes('glucosuria') ? '#1565c0' :
              feat.toLowerCase().includes('lactic') || feat.toLowerCase().includes('died') ? '#b71c1c' :
              feat.toLowerCase().includes('no ') || feat.toLowerCase().includes('normal') ? '#2e7d32' :
              feat.toLowerCase().includes('alive') ? '#2e7d32' :
              feat.toLowerCase().includes('leigh') || feat.toLowerCase().includes('cardinal') ? '#004d40' :
              COLOR
            }
          />
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients, seed-599)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Onset (mo)</th><th>Lactate</th>
                <th>COX%</th><th>Genotype</th><th>Features</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_mo}m</td>
                  <td style={{ color: p.lactate >= 10 ? '#b71c1c' : p.lactate >= 5 ? '#e65100' : '#2e7d32' }}>
                    {p.lactate}
                  </td>
                  <td style={{ color: p.cox_pct < 10 ? '#b71c1c' : '#e65100' }}>{p.cox_pct}%</td>
                  <td className="text-muted" style={{ maxWidth: 160, wordBreak: 'break-word' }}>{p.geno}</td>
                  <td className="text-muted" style={{ maxWidth: 180, wordBreak: 'break-word' }}>{p.features}</td>
                  <td style={{
                    color: p.outcome.startsWith('Died') ? '#b71c1c' : '#2e7d32',
                    maxWidth: 140, wordBreak: 'break-word'
                  }}>{p.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Treatments & DDx ──────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <Spinner />;
  const feats = data.feature_frequencies || {};
  return (
    <div>
      <SectionCard title="COX10 vs SURF1 — Critical DDx (Both Isolated COX Deficiency + Leigh Syndrome)" borderColor="#1565c0">
        <Alert variant="info" text="Both COX10 and SURF1 cause isolated Complex IV deficiency and Leigh-like MRI. They are biochemically identical. KEY DDx: urine amino acids + urine glucose (Fanconi screen). COX10 → Fanconi 60-70%; SURF1 → tubulopathy <10%." />
        <div className="row g-3 small mt-1">
          {[
            ['COX10 Tubulopathy (Fanconi)',    `${feats['Renal Tubulopathy / Fanconi (DISTINGUISHING)'] ?? '~65'}%`, '#1565c0'],
            ['SURF1 Tubulopathy',              '<10%',   '#78909c'],
            ['COX10 Leigh MRI',                `${feats['Leigh / Leigh-like MRI (CARDINAL)'] ?? '~88'}%`,  '#004d40'],
            ['COX10 Hepatopathy',              '0% (KEY DDx SCO1)',  '#2e7d32'],
            ['COX10 HCM',                      '<5% (KEY DDx SCO2)', '#2e7d32'],
            ['COX10 Iron Overload',            '0% (KEY DDx GRACILE)','#2e7d32'],
          ].map(([k, v, c]) => (
            <div key={k} className="col-12 col-md-6">
              <div className="p-2 rounded" style={{ background: LIGHT, borderLeft: `4px solid ${c}` }}>
                <span className="fw-semibold">{k}:</span> <span style={{ color: c }}>{v}</span>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Renal Tubulopathy Management (Fanconi Syndrome in COX10)" borderColor="#1565c0">
        {[
          ['Oral Phosphate Supplementation', '30-50 mg/kg/day in divided doses (phosphopenic rickets prevention)'],
          ['Oral Bicarbonate', '2-5 mEq/kg/day for proximal RTA (bicarbonaturia)'],
          ['Fludrocortisone', '0.05-0.1 mg/day for salt-wasting tubulopathy + volume depletion'],
          ['Active Vitamin D (Calcitriol)', '0.025-0.05 µg/kg/day for phosphopenic rickets'],
          ['Potassium supplementation', 'As needed for hypokalaemia (urinary potassium wasting)'],
          ['GFR monitoring', 'Serial serum creatinine + cystatin C; nephrocalcinosis screen'],
          ['AVOID nephrotoxic drugs', 'Aminoglycosides, NSAIDs, high-dose vancomycin amplify tubulopathy'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: '#1565c0' }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Mitochondrial Cofactor & Supportive Therapy">
        {[
          ['CoQ10 / Ubiquinol (Level C)', '300-600 mg/day adults; 10-30 mg/kg/day children; ubiquinol preferred (reduced form)'],
          ['Riboflavin B2 (Level C)', '100-400 mg/day — cofactor for Complex I (FMN) and Complex II (FAD)'],
          ['Thiamine B1 (Level C — MANDATORY empiric)', '100-300 mg/day — empiric in ALL Leigh until SLC19A3/BTD excluded (TREATABLE mimics)'],
          ['Biotin (Level C — MANDATORY empiric)', '5-20 mg/day empiric until BTD enzyme assay result — biotinidase deficiency is CURABLE Leigh mimic'],
          ['Succinate (Level C)', '6-12 g/day anaplerotic bypass — enters TCA at Complex II, bypasses Complex I restriction'],
          ['Carnitine (Level C)', '50-100 mg/kg/day — secondary carnitine deficiency is common in OXPHOS diseases'],
          ['IV Dextrose GIR 6-8', 'Continuous glucose infusion (6-8 mg/kg/min) during Leigh crisis — NEVER FAST'],
          ['NaHCO3', 'IV bicarbonate for lactic acidosis (pH <7.2 or base excess < −10)'],
          ['NIV / BiPAP', 'Non-invasive ventilation for central respiratory failure; avoid intubation+propofol if possible'],
          ['LEV (preferred AED)', 'First-line seizure control; renal excretion; zero mitochondrial toxicity; IV formulation available'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: COLOR }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Feature Breakdown (for Treatment Planning)" borderColor="#004d40">
        {Object.entries(feats).map(([feat, pct]) => (
          <Bar
            key={feat}
            label={feat}
            value={pct}
            color={
              feat.toLowerCase().includes('tubulopathy') || feat.toLowerCase().includes('fanconi') || feat.toLowerCase().includes('glucosuria') ? '#1565c0' :
              feat.toLowerCase().includes('lactic') || feat.toLowerCase().includes('died') ? '#b71c1c' :
              feat.toLowerCase().includes('no ') || feat.toLowerCase().includes('normal') ? '#2e7d32' :
              feat.toLowerCase().includes('alive') ? '#2e7d32' :
              feat.toLowerCase().includes('leigh') || feat.toLowerCase().includes('cardinal') ? '#004d40' :
              COLOR
            }
          />
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Spinner />;
  const sections = [
    { key: 'pharmacology',       label: 'Pharmacology & Molecular Biology' },
    { key: 'gene_concepts',      label: 'Gene & Genotype–Phenotype Concepts' },
    { key: 'disease_concepts',   label: 'Disease Concepts & DDx' },
    { key: 'prescribing_safety', label: 'Prescribing Safety (Extended)' },
  ];
  return (
    <div>
      {sections.map(sec => (
        data[sec.key] && (
          <SectionCard key={sec.key} title={sec.label}>
            {data[sec.key].map(d => (
              <div key={d.term} className="mb-4">
                <div className="fw-semibold small mb-1" style={{ color: COLOR }}>{d.term}</div>
                <p className="small text-muted mb-0" style={{ whiteSpace: 'pre-wrap' }}>{d.definition}</p>
              </div>
            ))}
          </SectionCard>
        )
      ))}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function COX10Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/cox10/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1) fetch(`${API}/api/cox10/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 2) fetch(`${API}/api/cox10/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 3) fetch(`${API}/api/cox10/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="d-flex align-items-center mb-3 gap-3">
        <div style={{
          width: 48, height: 48, borderRadius: '50%',
          background: COLOR, display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: '#fff', fontWeight: 'bold', fontSize: 22, flexShrink: 0,
        }}>🫘</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            COX10 — Leigh Syndrome + Renal Tubulopathy (Complex IV / COX Deficiency)
          </h4>
          <div className="text-muted small">
            COX10-441aa · 17p12 · AR · OMIM *602125 ·
            Protoheme IX farnesyltransferase — heme a/o synthesis Step 1 for COX1 (MT-CO1) ·
            Leigh MRI ~88% CARDINAL · Renal Tubulopathy/Fanconi ~65% DISTINGUISHING ·
            NO Hepatopathy (DDx SCO1) · NO HCM (DDx SCO2) · NO Iron (DDx GRACILE) ·
            VPA / Metformin / Linezolid ABSOLUTE CI · LEV preferred AED
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <TreatmentsTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
