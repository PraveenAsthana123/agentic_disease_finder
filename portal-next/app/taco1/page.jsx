'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#00695c';   // teal — mRNA translation biology; distinct from cardiac red/magenta
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

      <SectionCard title="Mechanism — TACO1 / MT-CO1 mRNA Translational Activation / Complex IV COX Assembly">
        <p className="small text-muted mb-0">{data.mechanism}</p>
      </SectionCard>

      <SectionCard title="Dysarthria — CARDINAL DISTINGUISHING Feature of TACO1 vs Other COX-Deficiency Diseases (~85%)" borderColor="#00897b">
        <p className="small text-muted mb-0" style={{ whiteSpace: 'pre-line' }}>{data.dysarthria_note}</p>
      </SectionCard>

      <SectionCard title={`KPIs — ${data.cohort}`}>
        <div className="row g-2">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Feature Frequencies (40-patient cohort, seed-603)">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar
            key={feat}
            label={feat}
            value={pct}
            color={
              feat.toLowerCase().includes('dysarthria') || feat.toLowerCase().includes('speech') ? '#00897b' :
              feat.toLowerCase().includes('cardinal') || feat.toLowerCase().includes('cognitive') ? '#00695c' :
              feat.toLowerCase().includes('leigh') || feat.toLowerCase().includes('regression') ? '#6a1b9a' :
              feat.toLowerCase().includes('lactic') || feat.toLowerCase().includes('died') ? '#b71c1c' :
              feat.toLowerCase().includes('no ') || feat.toLowerCase().includes('normal') ? '#2e7d32' :
              feat.toLowerCase().includes('alive') ? '#2e7d32' :
              feat.toLowerCase().includes('hcm') || feat.toLowerCase().includes('cardiomyopathy') ? '#ad1457' :
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
              variant={ci.severity.startsWith('ABSOLUTE') ? 'danger' : ci.severity.startsWith('CONTRA') ? 'warning' : 'warning'}
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
      <SectionCard title="Feature Frequencies (40-patient TACO1 cohort, seed-603)">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar
            key={feat}
            label={feat}
            value={pct}
            color={
              feat.toLowerCase().includes('dysarthria') || feat.toLowerCase().includes('speech') ? '#00897b' :
              feat.toLowerCase().includes('cognitive') || feat.toLowerCase().includes('cardinal') ? '#00695c' :
              feat.toLowerCase().includes('lactic') || feat.toLowerCase().includes('died') ? '#b71c1c' :
              feat.toLowerCase().includes('no ') || feat.toLowerCase().includes('normal') ? '#2e7d32' :
              feat.toLowerCase().includes('alive') ? '#2e7d32' :
              feat.toLowerCase().includes('leigh') || feat.toLowerCase().includes('regression') ? '#6a1b9a' :
              feat.toLowerCase().includes('hcm') ? '#ad1457' :
              feat.toLowerCase().includes('tubulopathy') ? '#1565c0' :
              COLOR
            }
          />
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients, seed-603)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Onset (yr)</th><th>Lactate</th>
                <th>COX%</th><th>Dysarthria</th><th>Genotype</th><th>Features</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_yr}yr</td>
                  <td style={{ color: p.lactate >= 7 ? '#b71c1c' : p.lactate >= 4 ? '#e65100' : '#2e7d32' }}>
                    {p.lactate}
                  </td>
                  <td style={{ color: p.cox_pct < 20 ? '#b71c1c' : p.cox_pct < 28 ? '#e65100' : '#2e7d32' }}>{p.cox_pct}%</td>
                  <td style={{ color: p.has_dysarthria ? '#00897b' : '#78909c', fontWeight: p.has_dysarthria ? 'bold' : 'normal' }}>
                    {p.has_dysarthria ? 'Dysarthria ✓' : '—'}
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
      <SectionCard title="TACO1 vs SURF1 vs COX10 vs SCO2 vs COX15 — COX Deficiency DDx" borderColor="#00897b">
        <Alert variant="info" text="All five diseases cause isolated Complex IV deficiency and Leigh-like MRI. KEY DDx is onset age + dominant organ feature: TACO1 → CHILDHOOD onset + Dysarthria; SURF1 → infantile + respiratory; SCO2 → infantile + HCM 100%; COX15 → infantile + HCM 78%; COX10 → infantile + tubulopathy 65%." />
        <div className="row g-3 small mt-1">
          {[
            ['TACO1 Dysarthria (CARDINAL)', `${feats['Dysarthria — Speech Motor Disorder (CARDINAL DISTINGUISHING)'] ?? '~85'}%`, '#00897b'],
            ['TACO1 Onset', 'Childhood (3-8yr) — LATER', '#00695c'],
            ['SCO2 HCM', '100% (most severe)',  '#c62828'],
            ['COX15 HCM', '~78% (DISTINGUISHING)', '#ad1457'],
            ['COX10 Tubulopathy', '~65% (KEY DDx)', '#1565c0'],
            ['TACO1 HCM', `${feats['HCM (RARE — KEY DDx SCO2 100% / COX15 78%)'] ?? '<5'}% (RARE — DDx)`, '#2e7d32'],
            ['TACO1 Tubulopathy', `${feats['Renal Tubulopathy (RARE — KEY DDx COX10 65%)'] ?? '<8'}% (RARE — DDx)`, '#1565c0'],
            ['TACO1 Hepatopathy', '0% (KEY DDx SCO1)', '#2e7d32'],
          ].map(([k, v, c]) => (
            <div key={k} className="col-12 col-md-6">
              <div className="p-2 rounded" style={{ background: LIGHT, borderLeft: `4px solid ${c}` }}>
                <span className="fw-semibold">{k}:</span> <span style={{ color: c }}>{v}</span>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Dysarthria & Neurological Management (TACO1 — Childhood Onset)" borderColor="#00897b">
        {[
          ['Speech-Language Pathology (SLP)', 'Baseline + 6-12 monthly; intelligibility tracking; dysphagia screen mandatory'],
          ['AAC (Augmentative/Alternative Communication)', 'Introduce EARLY while speech intelligible; eye-gaze device for severe stage'],
          ['Physiotherapy (ataxia + spasticity)', 'Regular PT from diagnosis; gait aids (walker, AFO); fall prevention; maintain ambulation'],
          ['Baclofen (spasticity)', 'GABA-B agonist; safe in mito disease; first-line for spasticity management'],
          ['Speech therapy intensive', 'Oromotor exercises; compensatory swallowing strategies; modified texture diet'],
          ['PEG gastrostomy', 'When oral intake <50% or aspiration risk high; use sevoflurane NOT propofol for anaesthesia'],
          ['Neuropsychological assessment', 'At diagnosis + 2-yearly; IEP/504 plan; OT for adaptive skills'],
          ['SSRI (mood)', 'If depression develops — SSRIs preferred over TCAs (cardiac safety); mito-safe'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: '#00897b' }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Mitochondrial Cofactor & Supportive Therapy">
        {[
          ['CoQ10 / Ubiquinol (Level C)', '300-600 mg/day adults; 10-30 mg/kg/day children; ubiquinol preferred'],
          ['Riboflavin B2 (Level C)', '100-400 mg/day — FMN/FAD cofactor for Complex I and Complex II'],
          ['Thiamine B1 (Level C — MANDATORY empiric)', '100-300 mg/day — ALL Leigh until SLC19A3/BTD excluded (TREATABLE mimics)'],
          ['Biotin (Level C — MANDATORY empiric)', '5-20 mg/day — BTD is CURABLE Leigh mimic; give empirically until enzyme assay'],
          ['Succinate (Level C)', '2-6 g/day — anaplerotic; bypasses Complex I → enters at Complex II; useful in COX deficiency'],
          ['Carnitine (Level C)', '50-100 mg/kg/day — secondary carnitine deficiency common in OXPHOS diseases'],
          ['IV Dextrose GIR 6-8', 'During crisis — NEVER FAST; continuous glucose infusion 6-8 mg/kg/min'],
          ['NaHCO3', 'IV bicarbonate for lactic acidosis (pH <7.2 or BE < −10)'],
          ['LEV (preferred AED)', 'First-line seizures; renal excretion; no mito toxicity; IV formulation available'],
          ['Physiotherapy + Speech therapy', 'Long-term cornerstone of TACO1 management — dysarthria + ataxia'],
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
              feat.toLowerCase().includes('dysarthria') || feat.toLowerCase().includes('speech') ? '#00897b' :
              feat.toLowerCase().includes('cognitive') || feat.toLowerCase().includes('cardinal') ? '#00695c' :
              feat.toLowerCase().includes('lactic') || feat.toLowerCase().includes('died') ? '#b71c1c' :
              feat.toLowerCase().includes('no ') || feat.toLowerCase().includes('normal') ? '#2e7d32' :
              feat.toLowerCase().includes('alive') ? '#2e7d32' :
              feat.toLowerCase().includes('leigh') || feat.toLowerCase().includes('cardinal') ? '#6a1b9a' :
              feat.toLowerCase().includes('hcm') ? '#ad1457' :
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
export default function TACO1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/taco1/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1) fetch(`${API}/api/taco1/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 2) fetch(`${API}/api/taco1/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 3) fetch(`${API}/api/taco1/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="d-flex align-items-center mb-3 gap-3">
        <div style={{
          width: 48, height: 48, borderRadius: '50%',
          background: COLOR, display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: '#fff', fontWeight: 'bold', fontSize: 22, flexShrink: 0,
        }}>🧬</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            TACO1 — Leigh Syndrome Childhood-Onset (Complex IV / COX Deficiency)
          </h4>
          <div className="text-muted small">
            TACO1-343aa · 17q23.3 · AR · OMIM *612958 ·
            MT-CO1 mRNA translational activator (mitochondrial matrix) ·
            Dysarthria ~85% CARDINAL DISTINGUISHING · Childhood onset (3-8yr) MILDER ·
            NO HCM (KEY DDx SCO2/COX15) · NO Tubulopathy (KEY DDx COX10) ·
            NO Hepatopathy (DDx SCO1) · NO Iron (DDx GRACILE) ·
            VPA / Metformin / Linezolid / Chloramphenicol ABSOLUTE CI · LEV preferred AED
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
