'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#880e4f';   // deep magenta — HCM/cardiac + COX15/mitochondrial
const LIGHT = '#fce4ec';

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

      <SectionCard title="Mechanism — COX15 / Heme-a Biosynthesis Step 2 / Complex IV COX1 Assembly">
        <p className="small text-muted mb-0">{data.mechanism}</p>
      </SectionCard>

      <SectionCard title="HCM (Hypertrophic Cardiomyopathy) — Distinguishing COX15 from COX10/SURF1 (75-80% vs <5% / ~10%)" borderColor="#ad1457">
        <p className="small text-muted mb-0" style={{ whiteSpace: 'pre-line' }}>{data.hcm_note}</p>
      </SectionCard>

      <SectionCard title={`KPIs — ${data.cohort}`}>
        <div className="row g-2">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Feature Frequencies (40-patient cohort, seed-601)">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar
            key={feat}
            label={feat}
            value={pct}
            color={
              feat.toLowerCase().includes('hcm') || feat.toLowerCase().includes('cardiomyopathy') ? '#ad1457' :
              feat.toLowerCase().includes('leigh') || feat.toLowerCase().includes('cardinal') ? '#6a1b9a' :
              feat.toLowerCase().includes('lactic') || feat.toLowerCase().includes('died') ? '#b71c1c' :
              feat.toLowerCase().includes('no ') || feat.toLowerCase().includes('normal') ? '#2e7d32' :
              feat.toLowerCase().includes('alive') ? '#2e7d32' :
              feat.toLowerCase().includes('tubulopathy') || feat.toLowerCase().includes('rare') ? '#1565c0' :
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
      <SectionCard title="Feature Frequencies (40-patient COX15 cohort, seed-601)">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar
            key={feat}
            label={feat}
            value={pct}
            color={
              feat.toLowerCase().includes('hcm') || feat.toLowerCase().includes('cardiomyopathy') ? '#ad1457' :
              feat.toLowerCase().includes('lactic') || feat.toLowerCase().includes('died') ? '#b71c1c' :
              feat.toLowerCase().includes('no ') || feat.toLowerCase().includes('normal') ? '#2e7d32' :
              feat.toLowerCase().includes('alive') ? '#2e7d32' :
              feat.toLowerCase().includes('leigh') || feat.toLowerCase().includes('cardinal') ? '#6a1b9a' :
              feat.toLowerCase().includes('tubulopathy') ? '#1565c0' :
              COLOR
            }
          />
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients, seed-601)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Onset (mo)</th><th>Lactate</th>
                <th>COX%</th><th>HCM</th><th>Genotype</th><th>Features</th><th>Outcome</th>
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
                  <td style={{ color: p.has_hcm ? '#ad1457' : '#78909c', fontWeight: p.has_hcm ? 'bold' : 'normal' }}>
                    {p.has_hcm ? 'HCM ✓' : '—'}
                  </td>
                  <td className="text-muted" style={{ maxWidth: 150, wordBreak: 'break-word' }}>{p.geno}</td>
                  <td className="text-muted" style={{ maxWidth: 160, wordBreak: 'break-word' }}>{p.features}</td>
                  <td style={{
                    color: p.outcome.startsWith('Died') ? '#b71c1c' : '#2e7d32',
                    maxWidth: 130, wordBreak: 'break-word'
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
      <SectionCard title="COX15 vs COX10 vs SCO2 vs SURF1 — HCM + COX Deficiency DDx" borderColor="#ad1457">
        <Alert variant="info" text="All four diseases cause isolated Complex IV deficiency and Leigh-like MRI. KEY DDx is cardiac + renal: COX15 → HCM 75-80%, tubulopathy <15%; COX10 → tubulopathy 65%, HCM <5%; SCO2 → HCM 100% (most severe); SURF1 → HCM 10%, no tubulopathy." />
        <div className="row g-3 small mt-1">
          {[
            ['COX15 HCM (DISTINGUISHING)', `${feats['HCM — Hypertrophic Cardiomyopathy (DISTINGUISHING)'] ?? '~78'}%`, '#ad1457'],
            ['SCO2 HCM',                   '100% (most severe)',  '#c62828'],
            ['SURF1 HCM',                  '~10% (infrequent)',  '#78909c'],
            ['COX10 HCM',                  '<5% (KEY DDx)',       '#2e7d32'],
            ['COX15 Tubulopathy',          `${feats['Renal Tubulopathy (RARE — KEY DDx COX10 65%)'] ?? '~14'}% (RARE)`, '#1565c0'],
            ['COX10 Tubulopathy',          '~65% (KEY DDx)',      '#1565c0'],
            ['COX15 Hepatopathy',          '0% (KEY DDx SCO1)',   '#2e7d32'],
            ['COX15 Iron Overload',        '0% (KEY DDx GRACILE)','#2e7d32'],
          ].map(([k, v, c]) => (
            <div key={k} className="col-12 col-md-6">
              <div className="p-2 rounded" style={{ background: LIGHT, borderLeft: `4px solid ${c}` }}>
                <span className="fw-semibold">{k}:</span> <span style={{ color: c }}>{v}</span>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="HCM Management (Hypertrophic Cardiomyopathy in COX15)" borderColor="#ad1457">
        {[
          ['Propranolol (First-line HCM)', '0.5-2 mg/kg/day in 2-3 divided doses — reduces LVOT gradient + myocardial O2 demand'],
          ['Atenolol (Alternative)', 'Cardioselective beta-1 blocker; renally excreted; once-daily compliance advantage'],
          ['Echocardiogram monitoring', 'Every 3-6 months — track wall thickness Z-score, LVOT gradient, diastolic function'],
          ['ECG / Holter monitor', 'Annual — arrhythmia surveillance; life-threatening VT/VF risk in hypertrophied myocardium'],
          ['BNP/NT-proBNP', 'Cardiac failure surrogate marker; trend over time'],
          ['AVOID — Digoxin', 'ABSOLUTE CI — positive inotropy worsens LVOT obstruction → haemodynamic collapse'],
          ['AVOID — ACEi/ARBs (obstructive phase)', 'Reduce preload → worsen LVOT obstruction → acute decompensation'],
          ['AVOID — Nifedipine/DHP-CCBs', 'Vasodilation → LVOT worsening; verapamil requires expert cardiology input'],
          ['Cardiac transplant (refractory)', 'Life-extending for cardiac failure; does NOT cure encephalopathy — ethics consultation mandatory'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: '#ad1457' }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Mitochondrial Cofactor & Supportive Therapy">
        {[
          ['CoQ10 / Ubiquinol (Level C)', '300-600 mg/day adults; 10-30 mg/kg/day children; ubiquinol preferred'],
          ['Riboflavin B2 (Level C)', '100-400 mg/day — cofactor for Complex I (FMN) and Complex II (FAD)'],
          ['Thiamine B1 (Level C — MANDATORY empiric)', '100-300 mg/day — empiric in ALL Leigh until SLC19A3/BTD excluded (TREATABLE mimics)'],
          ['Biotin (Level C — MANDATORY empiric)', '5-20 mg/day — BTD is a CURABLE Leigh mimic; give empirically until enzyme assay'],
          ['Carnitine (Level C)', '50-100 mg/kg/day — secondary carnitine deficiency common in OXPHOS diseases'],
          ['IV Dextrose GIR 6-8', 'Continuous glucose infusion (6-8 mg/kg/min) during Leigh crisis — NEVER FAST'],
          ['NaHCO3', 'IV bicarbonate for lactic acidosis (pH <7.2 or base excess < −10)'],
          ['NIV / BiPAP', 'Non-invasive ventilation for central respiratory failure — AVOID intubation+propofol'],
          ['LEV (preferred AED)', 'First-line seizure control; cardiac-safe; renal excretion; IV formulation available'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: COLOR }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Feature Breakdown (for Treatment Planning)" borderColor="#6a1b9a">
        {Object.entries(feats).map(([feat, pct]) => (
          <Bar
            key={feat}
            label={feat}
            value={pct}
            color={
              feat.toLowerCase().includes('hcm') || feat.toLowerCase().includes('cardiomyopathy') ? '#ad1457' :
              feat.toLowerCase().includes('lactic') || feat.toLowerCase().includes('died') ? '#b71c1c' :
              feat.toLowerCase().includes('no ') || feat.toLowerCase().includes('normal') ? '#2e7d32' :
              feat.toLowerCase().includes('alive') ? '#2e7d32' :
              feat.toLowerCase().includes('leigh') || feat.toLowerCase().includes('cardinal') ? '#6a1b9a' :
              feat.toLowerCase().includes('tubulopathy') ? '#1565c0' :
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
export default function COX15Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/cox15/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1) fetch(`${API}/api/cox15/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 2) fetch(`${API}/api/cox15/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 3) fetch(`${API}/api/cox15/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="d-flex align-items-center mb-3 gap-3">
        <div style={{
          width: 48, height: 48, borderRadius: '50%',
          background: COLOR, display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: '#fff', fontWeight: 'bold', fontSize: 22, flexShrink: 0,
        }}>❤️</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            COX15 — Leigh Syndrome + Infantile HCM (Complex IV / COX Deficiency)
          </h4>
          <div className="text-muted small">
            COX15-412aa · 10q24.2 · AR · OMIM *603646 ·
            Heme-o oxidase / heme-a synthase Step 2 for COX1 (MT-CO1) ·
            HCM ~78% DISTINGUISHING · Leigh MRI ~82% · NO Tubulopathy (KEY DDx COX10) ·
            NO Hepatopathy (DDx SCO1) · NO Iron (DDx GRACILE) ·
            VPA / Metformin / Linezolid / Propofol / Digoxin ABSOLUTE CI · LEV preferred AED
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
