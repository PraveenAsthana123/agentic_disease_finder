'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#1b5e20';   // deep forest green — Complex I (ETC entry point)
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

function featureColor(feat) {
  const f = feat.toLowerCase();
  if (f.includes('olfactory')) return '#00796b';
  if (f.includes('no ') || f.includes('normal') || f.includes('alive')) return '#2e7d32';
  if (f.includes('leigh') || f.includes('regression')) return '#6a1b9a';
  if (f.includes('lactic') || f.includes('died') || f.includes('fatal')) return '#b71c1c';
  if (f.includes('resp')) return '#e65100';
  if (f.includes('hcm') || f.includes('cardiomyopathy')) return '#ad1457';
  if (f.includes('hepatopathy') || f.includes('liver')) return '#e65100';
  if (f.includes('tubulopathy')) return '#1565c0';
  if (f.includes('founder') || f.includes('dutch')) return COLOR;
  return COLOR;
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

      <SectionCard title="Olfactory Bulb MRI — DISTINGUISHING Feature of NDUFS4 / CI-Leigh (~52–65%)" borderColor="#00796b">
        <Alert variant="success" text="Olfactory bulb + olfactory cortex T2/FLAIR hyperintensity is a hallmark of NDUFS4 CI-Leigh (Kruse 2008 Cell Metab; confirmed in human cohorts). NOT seen in SURF1, SCO2, COX10, COX15, or LRPPRC disease — helps distinguish CI-Leigh from COX-Leigh at the time of MRI, before enzyme or genetic results. Dedicated coronal olfactory bulb sequences are required." />
      </SectionCard>

      <SectionCard title={`KPIs — 40-patient NDUFS4 cohort (seed-607)`}>
        <div className="row g-2">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Feature Frequencies (40-patient cohort, seed-607)">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar key={feat} label={feat} value={pct} color={featureColor(feat)} />
        ))}
      </SectionCard>

      <SectionCard title="Absolute Contraindications & Drug Safety" borderColor="#c62828">
        {(data.contraindications || []).map(ci => (
          <div key={ci.drug} className="mb-3">
            <Alert
              variant={ci.severity.startsWith('ABSOLUTE') ? 'danger' : 'warning'}
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
  const bio = data.biochemistry_summary || {};
  return (
    <div>
      <SectionCard title="Biochemical Fingerprint — Isolated CI Deficiency (CII, CIII, CIV Normal)" borderColor="#00796b">
        <Alert variant="success" text={`NDUFS4 is ISOLATED Complex I deficiency: CI ${bio.complex_I_mean_pct ?? '~12'}% mean (range 5–22%), CII ${bio.complex_II_mean_pct ?? '~100'}% NORMAL, CIV ${bio.complex_IV_mean_pct ?? '~95'}% NORMAL. ${bio.note ?? ''}`} />
      </SectionCard>

      <SectionCard title="Feature Frequencies (40-patient NDUFS4/CI-Leigh cohort, seed-607)">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar key={feat} label={feat} value={pct} color={featureColor(feat)} />
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients, seed-607)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Onset</th>
                <th>Lactate (mmol/L)</th><th>CI%</th><th>CII%</th><th>CIV%</th>
                <th>Olfactory MRI</th><th>Genotype</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_yr}yr</td>
                  <td style={{ color: p.lactate_mm >= 10 ? '#b71c1c' : p.lactate_mm >= 5 ? '#e65100' : '#2e7d32' }}>
                    {p.lactate_mm}
                  </td>
                  <td style={{ color: p.ci_pct < 10 ? '#b71c1c' : p.ci_pct < 18 ? '#e65100' : '#f57f17' }}>{p.ci_pct}%</td>
                  <td style={{ color: '#2e7d32' }}>{p.cii_pct}%</td>
                  <td style={{ color: '#2e7d32' }}>{p.civ_pct}%</td>
                  <td style={{ color: p.has_olfactory ? '#00796b' : '#78909c', fontWeight: p.has_olfactory ? 'bold' : 'normal' }}>
                    {p.has_olfactory ? 'Yes ✓' : '—'}
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
      <SectionCard title="NDUFS4 vs SURF1 vs LRPPRC vs SCO2 vs GRACILE vs POLG — CI-Leigh DDx" borderColor="#c62828">
        <Alert variant="danger" text="NDUFS4/CI-Leigh has ISOLATED Complex I deficiency (CII, CIII, CIV all NORMAL). SURF1/SCO2/COX10/COX15/TACO1 have ISOLATED CIV (COX) deficiency. LRPPRC has COMBINED CI+CIV deficiency. NDUFS4 is further distinguished by olfactory bulb MRI lesions (~52-65%), NOT seen in SURF1/SCO2/LRPPRC. NO HCM (vs SCO2 100%). NO iron overload (vs GRACILE 100%). NO hepatopathy (vs POLG/DGUOK common)." />
        <div className="row g-3 small mt-1">
          {[
            ['NDUFS4 Isolated CI deficiency', '5–20% CI; CII, CIII, CIV NORMAL — BIOCHEMICAL FINGERPRINT', '#1b5e20'],
            ['NDUFS4 Olfactory Bulb MRI', `${feats['Olfactory Bulb MRI Lesions (DISTINGUISHING)'] ?? '~52'}% — DISTINGUISHING (not seen in COX-Leigh)`, '#00796b'],
            ['SURF1/SCO2/COX-Leigh Isolated CIV', 'CIV 5–20%; CI NORMAL — opposite to NDUFS4', '#c62828'],
            ['LRPPRC Combined CI+CIV', '100% combined (vs NDUFS4 isolated CI)', '#1565c0'],
            ['SCO2 HCM', '100% CARDINAL — KEY DDx (NDUFS4 HCM <6%)', '#ad1457'],
            ['GRACILE Iron Overload', 'Ferritin >2000 + IUGR + cholestasis (NO iron in NDUFS4)', '#e65100'],
            ['POLG Hepatopathy + EPC', 'Hepatopathy 80% + epilepsia partialis continua (RARE in NDUFS4)', '#6a1b9a'],
            ['NDUFS4 NO Hepatopathy', `${feats['Hepatopathy (RARE)'] ?? '<10'}% (RARE) — KEY DDx POLG/DGUOK`, '#2e7d32'],
          ].map(([k, v, c]) => (
            <div key={k} className="col-12 col-md-6">
              <div className="p-2 rounded" style={{ background: LIGHT, borderLeft: `4px solid ${c}` }}>
                <span className="fw-semibold">{k}:</span> <span style={{ color: c }}>{v}</span>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Crisis Management Protocol — NDUFS4/CI-Leigh Acute Decompensation" borderColor="#c62828">
        {[
          ['IV Dextrose GIR 6-8 (STAT — first-line)', 'Maximise glucose substrate for residual CI; maintain GIR 6-8 mg/kg/min; NEVER fast in CI-Leigh'],
          ['Hold mitochondrial toxins immediately', 'Stop phenobarbital; check for inadvertent metformin; NEVER propofol, linezolid, VPA'],
          ['NaHCO3 IV (pH <7.20)', '0.5–1 mEq/kg IV over 1-2h; continuous lactate monitoring q2h; target pH >7.25'],
          ['IV Riboflavin + Thiamine (100 mg each)', 'Both IV-safe; thiamine MANDATORY for any acute encephalopathy (SLC19A3 mimic)'],
          ['IV Succinate (metabolic centre)', '0.5–1 g/kg/day — CII-mediated bypass of blocked CI; not universally available'],
          ['Seizures → LEV IV loading', '20–40 mg/kg over 15 min; simultaneous IV dextrose; ABSOLUTE CI VPA'],
          ['Respiratory → NIV/BiPAP', 'SpO2 <92% or RR >40 → BiPAP; intubation: sevoflurane, NOT propofol'],
          ['Emergency card carried at all times', 'ER instructions: IV dextrose; NEVER VPA, metformin, linezolid, propofol, KD'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: '#c62828' }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Mitochondrial Cofactor & Supportive Therapy">
        {[
          ['CoQ10 / Ubiquinol (Level C)', '300-600 mg/day adults; 10-30 mg/kg/day children; ubiquinol (reduced form) preferred'],
          ['Riboflavin B2 (Level C — CI-SPECIFIC)', '100-400 mg/day; FMN precursor → primary CI N-module electron acceptor (NDUFV1 site)'],
          ['Thiamine B1 (MANDATORY empiric)', '100-300 mg/day ALL Leigh until SLC19A3 / PDH confirmed excluded (TREATABLE mimics)'],
          ['Biotin (MANDATORY empiric)', '5-20 mg/day ALL Leigh until BTD (biotinidase deficiency) enzyme activity confirmed'],
          ['Succinate (Level C — CI bypass)', '2-8 g/day orally; IV at metabolic centres; bypasses CI entirely via CII → ubiquinol'],
          ['Carnitine (Level C)', '50-100 mg/kg/day; secondary deficiency common; repletes mitochondrial carnitine pool'],
          ['LEV (preferred AED)', 'First-line all seizure types; renal excretion; no CYP interaction; IV available for crisis'],
          ['NIV/BiPAP', 'For central apnoea from brainstem Leigh lesions; titrate against SpO2 and CO2'],
          ['Enteral feeds / NG', 'Continuous feeds to prevent fasting; high-carbohydrate; avoid fat-predominant diet (KD CI)'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: COLOR }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Feature Breakdown (Treatment Planning)" borderColor="#6a1b9a">
        {Object.entries(feats).map(([feat, pct]) => (
          <Bar key={feat} label={feat} value={pct} color={featureColor(feat)} />
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
export default function NDUFS4Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/ndufs4/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1) fetch(`${API}/api/ndufs4/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 2) fetch(`${API}/api/ndufs4/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 3) fetch(`${API}/api/ndufs4/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
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
            NDUFS4 — Leigh Syndrome Isolated Complex I Deficiency (CI-Leigh / N-Module)
          </h4>
          <div className="text-muted small">
            NDUFS4-175aa · 5q11.2-q13.3 · AR · OMIM *602694 ·
            N-module accessory subunit (NDU N-arm, stabilises N-module assembly) ·
            Isolated CI deficiency (CII, CIII, CIV NORMAL — BIOCHEMICAL FINGERPRINT) ·
            Olfactory bulb MRI lesions ~52–65% (DISTINGUISHING — NOT in COX-Leigh) ·
            Dutch founder c.462delA · NO HCM (DDx SCO2 100%) · NO hepatopathy (DDx POLG) ·
            Metformin ABSOLUTE CI (direct CI inhibitor) · VPA / Linezolid ABSOLUTE CI ·
            KD CONTRAINDICATED · Riboflavin + Succinate CI-specific therapy · LEV preferred AED
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
