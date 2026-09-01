'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#006064';   // dark teal — Fe-S cluster / IP module / N-module electron relay
const LIGHT = '#e0f7fa';

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
  const bg     = variant === 'danger'  ? '#ffebee' : variant === 'warning' ? '#fff8e1'
               : variant === 'success' ? '#e8f5e9' : LIGHT;
  const border = variant === 'danger'  ? '#c62828' : variant === 'warning' ? '#f57f17'
               : variant === 'success' ? '#2e7d32' : COLOR;
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
  if (f.includes('neuropathy') || f.includes('axonal'))  return '#e65100';
  if (f.includes('no ') || f.includes('normal') || f.includes('alive')) return '#2e7d32';
  if (f.includes('leigh') || f.includes('regression')) return '#6a1b9a';
  if (f.includes('lactic') || f.includes('died') || f.includes('fatal')) return '#b71c1c';
  if (f.includes('resp'))   return '#b71c1c';
  if (f.includes('hcm') || f.includes('cardiomyopathy')) return '#ad1457';
  if (f.includes('hepato') || f.includes('liver')) return '#e65100';
  if (f.includes('founder')) return COLOR;
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

      <SectionCard title="Peripheral Neuropathy (~50%) — DISTINGUISHING Feature of NDUFS1" borderColor="#e65100">
        <Alert variant="warning" text="Peripheral neuropathy (axonal or demyelinating, sensorimotor) occurs in ~50% of NDUFS1/CI-Leigh — the most important clinical distinguisher within the CI-Leigh series. NDUFS4 shows no significant neuropathy. NDUFV1 shows no significant neuropathy. NDUFS1 is the ONLY common CI-Leigh gene with this frequency of peripheral neuropathy. EMG/NCS recommended in all suspected cases." />
      </SectionCard>

      <SectionCard title="NDUFS1 = Fe-S Backbone of N-module — N1b, N4, N5 Electron Relay" borderColor="#006064">
        <Alert variant="info" text="NDUFS1 (75 kDa, IP1) is the LARGEST nuclear-encoded CI subunit. It binds THREE iron-sulfur clusters (N1b, N4, N5) that form the central electron relay between the FMN/N3 site (NDUFV1) and the ubiquinone-binding Q-module. Without NDUFS1, electron transfer from NDUFV1/FMN to ubiquinone is broken → isolated CI deficiency. Biochemical fingerprint: CI 5–20%, CII/CIII/CIV NORMAL." />
      </SectionCard>

      <SectionCard title={`KPIs — 40-patient NDUFS1 cohort (seed-611)`}>
        <div className="row g-2">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Feature Frequencies (40-patient cohort, seed-611)">
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
  const acts = data.complex_activities || {};
  return (
    <div>
      <SectionCard title="Biochemical Fingerprint — Isolated CI Deficiency (CII, CIII, CIV Normal)" borderColor="#2e7d32">
        <Alert variant="success" text={`NDUFS1 is ISOLATED Complex I deficiency: CI ${acts.CI_mean ?? '~12'}% mean (range ${acts.CI_range ?? '5–22%'}), CII ${acts.CII_mean ?? '~100'}% NORMAL, CIV ${acts.CIV_mean ?? '~96'}% NORMAL. Same biochemical fingerprint as NDUFS4 and NDUFV1; clinical distinguisher is peripheral neuropathy.`} />
      </SectionCard>

      <SectionCard title="Feature Frequencies (40-patient NDUFS1/CI-Leigh cohort, seed-611)">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar key={feat} label={feat} value={pct} color={featureColor(feat)} />
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients, seed-611)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Onset</th>
                <th>Lactate (mmol/L)</th><th>CI%</th><th>CII%</th><th>CIV%</th>
                <th>Neuropathy</th><th>Genotype</th><th>Outcome</th>
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
                  <td style={{ color: p.has_neuropathy ? '#e65100' : '#78909c', fontWeight: p.has_neuropathy ? 'bold' : 'normal' }}>
                    {p.has_neuropathy ? 'Yes ✓' : '—'}
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
      <SectionCard title="NDUFS1 vs NDUFS4 vs NDUFV1 vs SURF1 vs LRPPRC vs POLG vs GRACILE — CI-Leigh DDx" borderColor="#c62828">
        <Alert variant="danger" text="NDUFS1/CI-Leigh has ISOLATED Complex I deficiency (CII, CIII, CIV all NORMAL). Key DDx within CI-Leigh series: NDUFS1 shows peripheral neuropathy (~50%) — NDUFS4 and NDUFV1 do NOT. NDUFS4 shows olfactory bulb lesions (52–65%) — NDUFS1 does NOT. NDUFV1 shows leukodystrophy (40–50%) — NDUFS1 does NOT. SURF1/SCO2/COX10/COX15/TACO1 all have ISOLATED CIV deficiency. LRPPRC has COMBINED CI+CIV. POLG: hepatopathy + EPC. GRACILE: iron overload + IUGR." />
        <div className="row g-3 small mt-1">
          {[
            ['NDUFS1 Isolated CI deficiency',     '5–20% CI; CII, CIII, CIV NORMAL — BIOCHEMICAL FINGERPRINT', '#006064'],
            ['Peripheral Neuropathy (NDUFS1 ~50%)', 'DISTINGUISHING vs NDUFS4 (none) and NDUFV1 (none) — EMG/NCS in all cases', '#e65100'],
            ['NDUFS4 CI-Leigh Comparison',        'Olfactory bulb lesions 52–65% (NDUFS4) vs NEVER (NDUFS1) — key MRI DDx', '#0d47a1'],
            ['NDUFV1 CI-Leigh Comparison',        'Leukodystrophy 40–50% (NDUFV1) vs RARELY (NDUFS1) — white matter DDx', '#1a237e'],
            ['SURF1/SCO2/COX-Leigh Isolated CIV', 'CIV 5–20%; CI NORMAL — opposite biochemistry to NDUFS1', '#c62828'],
            ['SCO2 HCM',                          '100% CARDINAL — KEY DDx (NDUFS1 HCM ~12%, much less)', '#ad1457'],
            ['GRACILE Iron Overload',              'Ferritin >2000 + IUGR + cholestasis (NO iron in NDUFS1)', '#e65100'],
            ['POLG Hepatopathy + EPC',            'Hepatopathy 80% + EPC (RARE in NDUFS1); mtDNA depletion on biopsy', '#6a1b9a'],
          ].map(([k, v, c]) => (
            <div key={k} className="col-12 col-md-6">
              <div className="p-2 rounded" style={{ background: LIGHT, borderLeft: `4px solid ${c}` }}>
                <span className="fw-semibold">{k}:</span> <span style={{ color: c }}>{v}</span>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Crisis Management Protocol — NDUFS1/CI-Leigh Acute Decompensation" borderColor="#c62828">
        {[
          ['IV Dextrose GIR 6-8 (STAT — first-line)', 'Maximise glucose substrate for residual CI; GIR 6-8 mg/kg/min; NEVER fast in CI-Leigh'],
          ['Hold mitochondrial toxins immediately', 'Check for metformin, phenobarbital, linezolid, chloramphenicol, propofol, VPA — stop all'],
          ['NaHCO3 IV (pH <7.20)', '0.5–1 mEq/kg over 1-2h; continuous lactate monitoring q2h; target pH >7.25'],
          ['IV Riboflavin + Thiamine (100 mg each)', 'Riboflavin: FMN precursor (upstream of NDUFS1 at NDUFV1). Thiamine MANDATORY for any acute encephalopathy (SLC19A3/BTBGD treatable mimic)'],
          ['IV Succinate (if available)', '0.5–1 g/kg/day — CII-mediated CI bypass; bypasses NDUFS1 Fe-S relay entirely'],
          ['Seizures → LEV IV', 'LEV 20–40 mg/kg IV loading; ABSOLUTE CI VPA; benzodiazepines acceptable'],
          ['Respiratory → NIV/BiPAP', 'SpO2 <92% or RR >40 → BiPAP; intubation: sevoflurane, NOT propofol. CAUTION: NDUFS1 neuropathy → lower threshold for respiratory support'],
          ['Emergency card at all times', 'ER: IV dextrose; NEVER VPA, metformin, linezolid, chloramphenicol, propofol, KD'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: '#c62828' }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Mitochondrial Cofactor & Supportive Therapy">
        {[
          ['Riboflavin B2 (Level C)', '100–400 mg/day; FMN precursor — upstream of NDUFS1 (FMN binds NDUFV1/N3 which feeds into NDUFS1 Fe-S relay); less direct than in NDUFV1 deficiency but used empirically in all CI-Leigh'],
          ['CoQ10 / Ubiquinol (Level C)', '300–600 mg/day adults; 10–30 mg/kg/day children; ubiquinol preferred'],
          ['Thiamine B1 (MANDATORY empiric)', '100–300 mg/day ALL Leigh until SLC19A3 / PDH confirmed excluded (TREATABLE mimics)'],
          ['Biotin (MANDATORY empiric)', '5–20 mg/day ALL Leigh until BTD (biotinidase deficiency) enzyme activity confirmed'],
          ['Succinate (Level C — CI bypass)', '2–8 g/day orally; IV at metabolic centres; bypasses CI entirely via CII → ubiquinol'],
          ['Carnitine (Level C)', '50–100 mg/kg/day; secondary deficiency common; repletes mitochondrial carnitine pool'],
          ['LEV (preferred AED)', 'First-line ALL seizure types; renal excretion; no CYP; IV available; no mito toxicity'],
          ['NIV/BiPAP', 'For central and peripheral respiratory compromise; earlier threshold in NDUFS1 (neuropathy component)'],
          ['Physiotherapy / orthotics', 'Peripheral neuropathy management: AFO for foot drop; wrist splints; gait rehabilitation'],
          ['Gabapentin / pregabalin', 'Neuropathic pain from peripheral neuropathy component; watch for sedation'],
          ['Enteral feeds / NG', 'Continuous feeds; high-carbohydrate; NEVER fat-predominant (KD absolutely CI)'],
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
export default function NDUFS1Page() {
  const [tab, setTab]             = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError]         = useState(null);

  useEffect(() => {
    fetch(`${API}/api/ndufs1/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1) fetch(`${API}/api/ndufs1/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 2) fetch(`${API}/api/ndufs1/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 3) fetch(`${API}/api/ndufs1/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
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
            NDUFS1 — Leigh Syndrome Isolated Complex I Deficiency (CI-Leigh / 75 kDa IP1 Fe-S N-Module)
          </h4>
          <div className="text-muted small">
            NDUFS1-727aa · 2q33.3 · AR · OMIM *157655 ·
            N-module IP1 (75 kDa, Fe-S clusters N1b/N4/N5 — CENTRAL electron relay NDUFV1→N2→UQ) ·
            Isolated CI deficiency (CII, CIII, CIV NORMAL) ·
            Peripheral Neuropathy ~50% (DISTINGUISHING — not NDUFS4, not NDUFV1) ·
            NO Olfactory Bulb Lesions (KEY DDx NDUFS4) ·
            NO Leukodystrophy (KEY DDx NDUFV1) ·
            Metformin ABSOLUTE CI · VPA / Linezolid / Chloramphenicol ABSOLUTE CI ·
            KD CONTRAINDICATED · Succinate CI bypass · LEV preferred AED
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
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
