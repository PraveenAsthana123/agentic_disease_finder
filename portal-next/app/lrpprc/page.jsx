'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#1a237e';   // deep indigo — LSFC French-Canadian founder disease
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

      <SectionCard title="Mechanism — LRPPRC / All mt-mRNA Stabilisation / Combined Complex I + IV Deficiency">
        <p className="small text-muted mb-0">{data.mechanism}</p>
      </SectionCard>

      <SectionCard title="Episodic Metabolic Crises — CARDINAL DISTINGUISHING Feature of LRPPRC/LSFC (~92%)" borderColor="#c62828">
        <p className="small text-muted mb-0" style={{ whiteSpace: 'pre-line' }}>{data.crisis_note}</p>
      </SectionCard>

      <SectionCard title={`KPIs — ${data.cohort}`}>
        <div className="row g-2">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Feature Frequencies (40-patient cohort, seed-605)">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar
            key={feat}
            label={feat}
            value={pct}
            color={
              feat.toLowerCase().includes('crisis') || feat.toLowerCase().includes('episodic') ? '#c62828' :
              feat.toLowerCase().includes('combined') || feat.toLowerCase().includes('i + iv') ? '#1565c0' :
              feat.toLowerCase().includes('leigh') || feat.toLowerCase().includes('regression') ? '#6a1b9a' :
              feat.toLowerCase().includes('lactic') || feat.toLowerCase().includes('died') ? '#b71c1c' :
              feat.toLowerCase().includes('no ') || feat.toLowerCase().includes('normal') ? '#2e7d32' :
              feat.toLowerCase().includes('alive') ? '#2e7d32' :
              feat.toLowerCase().includes('hepatopathy') || feat.toLowerCase().includes('liver') ? '#e65100' :
              feat.toLowerCase().includes('hcm') || feat.toLowerCase().includes('cardiomyopathy') ? '#ad1457' :
              feat.toLowerCase().includes('tubulopathy') ? '#1565c0' :
              feat.toLowerCase().includes('founder') || feat.toLowerCase().includes('french') ? COLOR :
              COLOR
            }
          />
        ))}
      </SectionCard>

      <SectionCard title="Absolute Contraindications & Drug Safety" borderColor="#c62828">
        {(data.contraindications || []).map(ci => (
          <div key={ci.drug} className="mb-3">
            <Alert
              variant={ci.severity.startsWith('ABSOLUTE') ? 'danger' : ci.severity.startsWith('CONTRA') || ci.severity.startsWith('DANGER') ? 'warning' : 'warning'}
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
      <SectionCard title="Feature Frequencies (40-patient LRPPRC/LSFC cohort, seed-605)">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar
            key={feat}
            label={feat}
            value={pct}
            color={
              feat.toLowerCase().includes('crisis') || feat.toLowerCase().includes('episodic') ? '#c62828' :
              feat.toLowerCase().includes('combined') ? '#1565c0' :
              feat.toLowerCase().includes('lactic') || feat.toLowerCase().includes('died') ? '#b71c1c' :
              feat.toLowerCase().includes('no ') || feat.toLowerCase().includes('normal') ? '#2e7d32' :
              feat.toLowerCase().includes('alive') ? '#2e7d32' :
              feat.toLowerCase().includes('hepatopathy') ? '#e65100' :
              feat.toLowerCase().includes('leigh') || feat.toLowerCase().includes('regression') ? '#6a1b9a' :
              feat.toLowerCase().includes('hcm') ? '#ad1457' :
              feat.toLowerCase().includes('tubulopathy') ? '#1565c0' :
              COLOR
            }
          />
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients, seed-605)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Onset</th><th>Lactate (base)</th>
                <th>Lactate (crisis)</th><th>CI%</th><th>CIV%</th><th>Crises</th><th>Genotype</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_yr}yr</td>
                  <td style={{ color: p.lactate_base >= 5 ? '#e65100' : '#2e7d32' }}>
                    {p.lactate_base}
                  </td>
                  <td style={{ color: p.lactate_crisis >= 12 ? '#b71c1c' : p.lactate_crisis >= 9 ? '#e65100' : '#e65100' }}>
                    {p.lactate_crisis}
                  </td>
                  <td style={{ color: p.coxI_pct < 30 ? '#b71c1c' : p.coxI_pct < 38 ? '#e65100' : '#2e7d32' }}>{p.coxI_pct}%</td>
                  <td style={{ color: p.coxIV_pct < 25 ? '#b71c1c' : p.coxIV_pct < 32 ? '#e65100' : '#2e7d32' }}>{p.coxIV_pct}%</td>
                  <td style={{ color: p.has_crises ? '#c62828' : '#78909c', fontWeight: p.has_crises ? 'bold' : 'normal' }}>
                    {p.has_crises ? 'Crises ✓' : '—'}
                  </td>
                  <td className="text-muted" style={{ maxWidth: 160, wordBreak: 'break-word' }}>{p.geno}</td>
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
      <SectionCard title="LRPPRC vs TACO1 vs SURF1 vs SCO1 vs SCO2 vs COX10 — COX Deficiency DDx" borderColor="#c62828">
        <Alert variant="danger" text="LRPPRC/LSFC is the ONLY COX-deficiency disease with COMBINED Complex I + Complex IV deficiency. All others (SURF1, SCO1, SCO2, COX10, COX15, TACO1) have ISOLATED Complex IV deficiency. LRPPRC also has EPISODIC CRISES (fever-triggered), distinguishing it from TACO1 (progressive without crises) and SURF1/SCO2 (continuous infantile decline)." />
        <div className="row g-3 small mt-1">
          {[
            ['LRPPRC Episodic Crises (CARDINAL)', `${feats['Episodic Metabolic Crises (CARDINAL DISTINGUISHING — fever-triggered)'] ?? '~92'}% — DISTINGUISHING`, '#c62828'],
            ['LRPPRC Combined CI+CIV', '100% — ONLY COX disease with combined defect', '#1565c0'],
            ['LRPPRC Hepatopathy', `${feats['Hepatopathy — Mild-Moderate (KEY DDx SCO1 is 100% Severe Neonatal)'] ?? '~45'}% mild-moderate (vs SCO1 100% severe neonatal)`, '#e65100'],
            ['SCO1 Hepatopathy', '100% SEVERE NEONATAL — cardinal DDx vs LRPPRC', '#c62828'],
            ['SCO2 HCM', '100% — KEY DDx (LRPPRC has NO dominant HCM)', '#ad1457'],
            ['TACO1 Dysarthria', '~85% CARDINAL — TACO1 has NO episodic crises', '#00897b'],
            ['SURF1 Respiratory', '~75% dominant — continuous infantile decline', '#6a1b9a'],
            ['COX10 Tubulopathy', '~65% KEY DDx (LRPPRC rare tubulopathy)', '#1565c0'],
          ].map(([k, v, c]) => (
            <div key={k} className="col-12 col-md-6">
              <div className="p-2 rounded" style={{ background: LIGHT, borderLeft: `4px solid ${c}` }}>
                <span className="fw-semibold">{k}:</span> <span style={{ color: c }}>{v}</span>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Crisis Management Protocol — Acute LSFC Metabolic Crisis" borderColor="#c62828">
        {[
          ['IV Dextrose GIR 6-8 STAT (FIRST-LINE)', 'Suppress gluconeogenesis + beta-oxidation immediately; maintain GIR 6-8 mg/kg/min; NEVER fast'],
          ['Aggressive Fever Control (crisis prevention)', 'Acetaminophen 15mg/kg q4-6h + ibuprofen 10mg/kg q6-8h + cooling; target T <38°C'],
          ['NaHCO3 IV (lactic acidosis)', 'For pH <7.2 or BE < −12; continuous lactate monitoring during crisis'],
          ['LEV IV (seizures)', 'Loading 20-40 mg/kg over 15 min; ABSOLUTE CI VPA; continuous IV glucose simultaneously'],
          ['ICU admission', 'Lactate >10 mmol/L or encephalopathy or respiratory compromise → ICU'],
          ['AVOID: Propofol, linezolid, VPA, metformin, fasting', 'All ABSOLUTELY CONTRAINDICATED during crisis and at all other times'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: '#c62828' }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Crisis Prevention — Fever + Fasting Protocol (Between-Crisis Management)">
        {[
          ['Continuous glucose maintenance', 'High-carbohydrate diet; never fast >2h; cornstarch supplement at bedtime if early morning NPO'],
          ['Fever: immediate IV glucose threshold', 'Any fever >38°C → IV dextrose GIR 6-8 within 1 hour; do not wait for clinical worsening'],
          ['Annual influenza vaccine', 'Flu is the most common crisis trigger; all LSFC patients must receive annual flu vaccine'],
          ['Metabolic emergency card', 'Carried at all times; ER instructions: IV glucose, LEV, NEVER VPA/propofol/linezolid/fasting'],
          ['Metabolic specialist coordination', '24h on-call access; local ER pre-educated on LSFC protocol; annual review'],
          ['RSV prophylaxis (infants)', 'Palivizumab for infants <2yr in RSV season — RSV frequently triggers LSFC crises'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: COLOR }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Mitochondrial Cofactor & Supportive Therapy">
        {[
          ['CoQ10 / Ubiquinol (Level C)', '300-600 mg/day adults; 10-30 mg/kg/day children; ubiquinol preferred for bioavailability'],
          ['Riboflavin B2 (Level C)', '100-400 mg/day — FMN/FAD for Complex I and Complex II (both relevant in combined CI+CIV)'],
          ['Thiamine B1 (Level C — MANDATORY empiric)', '100-300 mg/day — ALL Leigh until SLC19A3 excluded (TREATABLE mimic)'],
          ['Biotin (Level C — MANDATORY empiric)', '5-20 mg/day — BTD is CURABLE Leigh mimic; give empirically until enzyme assay'],
          ['Carnitine (Level C)', '50-100 mg/kg/day — secondary carnitine deficiency common; especially during crises'],
          ['IV Dextrose GIR 6-8', 'Crisis ONLY acute intervention; never fast; continuous enteral feeds between crises'],
          ['NaHCO3', 'Bicarbonate for lactic acidosis (pH <7.2); target pH >7.25'],
          ['LEV (preferred AED)', 'First-line seizures; renal excretion; safe in hepatopathy; IV formulation available'],
          ['Liver monitoring', 'LFTs every 3-6 months; hepatopathy present in ~45%; avoid hepatotoxic drugs'],
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
              feat.toLowerCase().includes('crisis') || feat.toLowerCase().includes('episodic') ? '#c62828' :
              feat.toLowerCase().includes('combined') ? '#1565c0' :
              feat.toLowerCase().includes('lactic') || feat.toLowerCase().includes('died') ? '#b71c1c' :
              feat.toLowerCase().includes('no ') || feat.toLowerCase().includes('normal') ? '#2e7d32' :
              feat.toLowerCase().includes('alive') ? '#2e7d32' :
              feat.toLowerCase().includes('hepatopathy') ? '#e65100' :
              feat.toLowerCase().includes('leigh') || feat.toLowerCase().includes('regression') ? '#6a1b9a' :
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
export default function LRPPRCPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/lrpprc/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1) fetch(`${API}/api/lrpprc/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 2) fetch(`${API}/api/lrpprc/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 3) fetch(`${API}/api/lrpprc/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
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
            LRPPRC — Leigh Syndrome French-Canadian Type LSFC (Combined Complex I + IV Deficiency)
          </h4>
          <div className="text-muted small">
            LRPPRC-1394aa · 2p21 · AR · OMIM *607544 ·
            PPR-domain mt-mRNA stabiliser (ALL 13 mt-mRNAs) · SLIRP co-factor ·
            Combined CI+CIV deficiency (DISTINGUISHING vs isolated CIV) ·
            Episodic metabolic crises ~92% CARDINAL · SLSJ founder p.Ala354Val ·
            NO dominant HCM (DDx SCO2/COX15) · NO severe neonatal hepatopathy (DDx SCO1) ·
            VPA / Metformin / Linezolid / Propofol ABSOLUTE CI · Fasting DANGEROUS · LEV preferred AED
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
