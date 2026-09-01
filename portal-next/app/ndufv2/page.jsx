'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#1a237e';   // dark indigo — N1b [2Fe-2S] 2nd relay / NDUFV2 N-module theme
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
  if (f.includes('no ') || f.includes('never') || f.includes('normal') || f.includes('alive')) return '#2e7d32';
  if (f.includes('leigh') || f.includes('regression')) return '#6a1b9a';
  if (f.includes('lactic') || f.includes('died') || f.includes('fatal')) return '#b71c1c';
  if (f.includes('resp'))   return '#b71c1c';
  if (f.includes('hcm') || f.includes('cardiomyopathy')) return '#ad1457';
  if (f.includes('hepato') || f.includes('liver')) return '#e65100';
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
            ['Gene',        data.gene],
            ['OMIM Gene',   data.omim_gene],
            ['OMIM Disease',data.omim_disease],
            ['Chromosome',  data.chromosome],
            ['Inheritance', data.inheritance],
            ['Onset',       data.onset],
          ].map(([k,v]) => (
            <div className="col-12 col-md-6" key={k}>
              <span className="fw-semibold">{k}:</span> {v}
            </div>
          ))}
          <div className="col-12">
            <span className="fw-semibold">Protein:</span> <span className="font-monospace small">{data.protein}</span>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="NDUFV2 = 24 kDa N-module Subunit — N1b [2Fe-2S] Fe-S Cluster (2nd Electron Relay Step)" borderColor="#283593">
        <Alert variant="info" text="NDUFV2 (24 kDa subunit) carries the sole [2Fe-2S] N1b cluster in the CI Fe-S relay chain — the ONLY [2Fe-2S] cluster; all others are [4Fe-4S]. N1b occupies the 2nd position in the electron relay, directly downstream of NDUFV1/FMN/N3 (the primary NADH acceptor). Loss of NDUFV2 blocks the second relay step: electrons from NDUFV1/N3 cannot propagate toward NDUFS7/N4 and subsequently N2/ubiquinone. CI activity: 5–20%, CII/CIII/CIV NORMAL." />
        <Alert variant="danger" text="HCM ~80% — DISTINCTIVE: NDUFV2 causes the highest rate of hypertrophic cardiomyopathy among CI Fe-S relay subunit diseases (NDUFS7 ~6%, NDUFS8 ~5%). The N1b block creates severe NADH/NAD+ imbalance in cardiomyocytes (highest CI-dependent OXPHOS demand) → compensatory hypertrophy → HCM + LVOT obstruction. Digoxin is ABSOLUTE CI. Propranolol is first-line cardiac therapy." />
        <Alert variant="success" text="NDUFV2-CI-Leigh does NOT cause peripheral neuropathy. This is a critical distinguishing feature from NDUFS1 (IP1 subunit, ~50% axonal/demyelinating neuropathy). NDUFV2 also does NOT cause olfactory bulb lesions (DDx NDUFS4 52–65%) or leukodystrophy (DDx NDUFV1 40–50%). Genetic panel required to distinguish from other CI-Leigh subunits." />
        <Alert variant="warning" text="vs SCO2 (COX assembly factor): SCO2 also produces HCM (~100%) but causes CIV deficiency — NOT CI deficiency. NDUFV2: HCM ~80% with ISOLATED CI deficiency (CII/CIII/CIV NORMAL). The biochemical fingerprint (CI vs CIV) is the key differentiator when HCM dominates the presentation." />
      </SectionCard>

      <SectionCard title={`KPIs — 40-patient NDUFV2 cohort (seed-621)`}>
        <div className="row g-2">
          {data.kpis?.map(k => <KPI key={k.label} label={k.label} value={k.value} color={k.color} />)}
        </div>
      </SectionCard>

      <SectionCard title="Feature Frequencies — NDUFV2/CI-Leigh 40-patient cohort">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar key={feat} label={feat} value={pct} color={featureColor(feat)} />
        ))}
      </SectionCard>

      <SectionCard title="Fe-S Relay Chain — NDUFV2 N1b Position in Complex I" borderColor="#1b5e20">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Subunit</th><th>Fe-S Cluster(s)</th><th>Module</th><th>Relay Position</th><th>Key DDx Feature</th></tr>
            </thead>
            <tbody>
              {[
                ['NDUFV1', 'N3 [4Fe-4S]',      'N-module (FMN)',          '1st (primary NADH)',        'Leukodystrophy 40–50% (DISTINGUISHING)'],
                ['NDUFV2', 'N1b [2Fe-2S]',      'N-module',                '2nd relay — THIS',          'HCM ~80% DISTINCTIVE; no neuropathy; no WM; no olfactory'],
                ['NDUFS7', 'N4 [4Fe-4S]',       'Q/N-module junction',     '3rd relay',                 'No neuropathy / no olfactory / no WM; HCM ~6%'],
                ['NDUFS8', 'N6a + N6b [4Fe-4S]','Q-module approach (TYKY)','4th + 5th relay',           'Dual Fe-S block; no neuropathy; no WM; HCM ~5%'],
                ['NDUFS1', 'N5 [4Fe-4S]',       'N-module peripheral',     '6th relay',                 'Peripheral neuropathy 50% (KEY DDx)'],
                ['NDUFS2', 'N2 [4Fe-4S]',       'Q-module (terminal)',     '7th relay → UQ',            'Terminal N2; HCM ~8%'],
                ['NDUFS4', '—',                  'N-module (accessory)',    'Assembly role',             'Olfactory bulb lesions 52–65% (pathognomonic)'],
              ].map(r => (
                <tr key={r[0]} style={{ background: r[0] === 'NDUFV2' ? LIGHT : undefined, fontWeight: r[0] === 'NDUFV2' ? 'bold' : undefined }}>
                  {r.map((c, i) => <td key={i}>{c}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="HCM ~80% — Comparative HCM Rates in CI Fe-S Relay Series" borderColor="#ad1457">
        <Alert variant="danger" text="NDUFV2 HCM ~80% is the highest among CI Fe-S relay subunit diseases. Compare: NDUFS7 ~6%, NDUFS8 ~5%, NDUFS2 ~8%, NDUFS3 ~10%, NDUFV1 ~8–10%. Only SCO2 (100% HCM, CIV deficiency) exceeds NDUFV2. Digoxin ABSOLUTE CI. Propranolol first-line." />
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Gene</th><th>HCM Rate</th><th>Respiratory Chain Defect</th><th>Key DDx</th></tr>
            </thead>
            <tbody>
              {[
                ['NDUFV2', '~80% — DISTINCTIVE', 'Isolated CI (CII/CIII/CIV normal)', 'THIS DISEASE — N1b [2Fe-2S] block'],
                ['SCO2',   '~100%',              'CIV deficiency (COX assembly)',      'CIV deficient — NOT CI: biochemistry differentiates'],
                ['NDUFS3', '~10%',               'Isolated CI',                        'BN-PAGE CI sub-assembly intermediates'],
                ['NDUFV1', '~8–10%',             'Isolated CI',                        'Leukodystrophy 40–50%'],
                ['NDUFS2', '~8%',                'Isolated CI',                        'Terminal N2 loss; genetics required'],
                ['NDUFS7', '~6%',                'Isolated CI',                        'Single N4 block; genetics required'],
                ['NDUFS8', '~5%',                'Isolated CI',                        'Dual N6a+N6b block; genetics required'],
              ].map(r => (
                <tr key={r[0]} style={{ background: r[0] === 'NDUFV2' ? LIGHT : undefined, fontWeight: r[0] === 'NDUFV2' ? 'bold' : undefined }}>
                  {r.map((c, i) => <td key={i}>{c}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Absolute Contraindications — NDUFV2/CI-Leigh" borderColor="#b71c1c">
        <Alert variant="danger" text="ABSOLUTE CI: Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + hepatotoxicity. In NDUFV2-CI-Leigh, CI is already at 5–20%; VPA collapses residual OXPHOS. Use LEV." />
        <Alert variant="danger" text="ABSOLUTE CI: Metformin — direct Complex I inhibitor at ND1/quinone-binding site. NDUFV2/N1b block already prevents electrons reaching N4/N2; metformin eliminates residual CI. Catastrophic." />
        <Alert variant="danger" text="ABSOLUTE CI: Digoxin — positive inotrope in HCM with LVOT obstruction → haemodynamic collapse. NDUFV2 HCM ~80%. Use propranolol (beta-blocker) first-line." />
        <Alert variant="danger" text="ABSOLUTE CI: Linezolid + Chloramphenicol — mt-ribosome 23S rRNA block → all 7 mtDNA-encoded ND subunits absent → CI P-module collapses → near-zero CI in already-deficient NDUFV2 patient." />
        <Alert variant="danger" text="CONTRAINDICATED: Ketogenic Diet — forces NADH overload via beta-oxidation; N1b-blocked CI cannot re-oxidise NADH → worsened lactic acidosis. Beneficial in GLUT1-DS and PDHD, not CI-Leigh." />
        <Alert variant="warning" text="AVOID: Propofol (PRIS risk — CIV inhibition creates 2nd ETC block; HCM adds haemodynamic vulnerability). Use sevoflurane or dexmedetomidine." />
        <Alert variant="warning" text="HIGH CAUTION: Phenobarbital — secondary CI inhibitor. Reduces residual CI further. Use LEV or CLB instead." />
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Features ────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <Spinner />;
  const { patients = [], feature_frequencies = {}, genotype_distribution = {}, complex_activities = {} } = data;
  return (
    <div>
      <SectionCard title="Complex Activity Summary — NDUFV2 Cohort">
        <div className="row g-3 small">
          <div className="col-md-3"><strong>CI Mean:</strong> {complex_activities.CI_mean}%</div>
          <div className="col-md-3"><strong>CI Range:</strong> {complex_activities.CI_range}</div>
          <div className="col-md-3"><strong>CII Mean:</strong> {complex_activities.CII_mean}% (NORMAL)</div>
          <div className="col-md-3"><strong>CIV Mean:</strong> {complex_activities.CIV_mean}% (NORMAL)</div>
        </div>
      </SectionCard>

      <SectionCard title="Feature Frequencies — Breakdown">
        {Object.entries(feature_frequencies).map(([feat, pct]) => (
          <Bar key={feat} label={feat} value={pct} color={featureColor(feat)} />
        ))}
      </SectionCard>

      <SectionCard title="Genotype Distribution">
        <div className="row g-2 small">
          {Object.entries(genotype_distribution).map(([g, n]) => (
            <div className="col-12 col-md-6" key={g}>
              <span className="badge me-2" style={{ background: COLOR }}>n={n}</span>{g}
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="40-Patient Cohort (seed-621)">
        <div className="table-responsive">
          <table className="table table-sm table-striped small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Onset (yr)</th><th>Lactate (mM)</th>
                <th>CI %</th><th>CII %</th><th>CIV %</th><th>Features</th><th>Treatments</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.id} style={{ color: p.outcome.startsWith('Died') ? '#b71c1c' : undefined }}>
                  <td>{p.id}</td><td>{p.sex}</td><td>{p.onset_yr}</td><td>{p.lactate_mm}</td>
                  <td><span className="badge bg-danger">{p.ci_pct}%</span></td>
                  <td><span className="badge bg-success">{p.cii_pct}%</span></td>
                  <td><span className="badge bg-success">{p.civ_pct}%</span></td>
                  <td className="small">{p.features}</td>
                  <td className="small">{p.treatments}</td>
                  <td className="small">{p.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Treatments & DDx ───────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="NDUFV2-CI-Leigh vs CI-Leigh Series & SCO2 — Critical DDx" borderColor="#1b5e20">
        <Alert variant="info" text="All CI-Leigh diseases share identical biochemistry: CI 5–20%, CII/CIII/CIV NORMAL. SCO2 causes HCM + CIV deficiency (not CI). The key NDUFV2 distinguisher is HCM ~80% WITH isolated CI deficiency — biochemistry separates NDUFV2 from SCO2 (CIV deficiency). NDUFV2 has NO peripheral neuropathy (DDx NDUFS1 50%), NO olfactory bulb lesions (DDx NDUFS4 52–65%), NO leukodystrophy (DDx NDUFV1 40–50%)." />
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Gene</th><th>Module / Function</th><th>Distinguishing Feature</th><th>Key DDx Marker</th></tr>
            </thead>
            <tbody>
              {[
                ['NDUFS4',  'N-module (accessory)',              'Olfactory bulb lesions 52–65% (pathognomonic)', 'Olfactory T2 MRI'],
                ['NDUFV1',  'N-module FMN/N3 (51 kDa)',         'Leukodystrophy / white matter T2 40–50%',      'White matter FLAIR'],
                ['NDUFS1',  'N-module IP1 (75 kDa) N5',         'Peripheral neuropathy ~50%',                    'EMG / NCS'],
                ['NDUFV2',  'N-module 24 kDa N1b [2Fe-2S] — THIS', 'HCM ~80% DISTINCTIVE + isolated CI (no CIV deficiency)', 'Echo + biochemistry + WES'],
                ['NDUFS7',  'Q/N-junction N4',                  'Single N4 block; HCM ~6%',                    'WES / genetic panel'],
                ['NDUFS8',  'Q-module TYKY N6a+N6b',            'Dual N6a+N6b block; HCM ~5%',                 'WES / genetic panel'],
                ['NDUFS2',  'Q-module N2 terminal (49 kDa)',    'Terminal N2 loss; HCM ~8%',                     'WES / genetic panel'],
                ['SCO2',    'COX assembly factor (22q13.33)',   'HCM ~100% but CIV deficiency — NOT CI',        'CIV biochemistry + genetics'],
              ].map(r => (
                <tr key={r[0]} style={{ background: r[0] === 'NDUFV2' ? LIGHT : undefined, fontWeight: r[0] === 'NDUFV2' ? 'bold' : undefined }}>
                  {r.map((c, i) => <td key={i}>{c}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Supportive Treatment Ladder — NDUFV2/CI-Leigh" borderColor="#2e7d32">
        {[
          ['IV Dextrose (GIR 6-8 mg/kg/min)',           'Crisis — FIRST LINE — never fast in CI-Leigh',                              'success'],
          ['Propranolol (1–2 mg/kg/day ÷ 3–4 doses)',   'HCM FIRST LINE — beta-blocker; reduces LVOT gradient + HR control',         'success'],
          ['Riboflavin B2 (100–400 mg/day)',             'CI-specific (FMN precursor — adjacent to NDUFV2 N1b upstream at NDUFV1 N3)','info'],
          ['Succinate (2–8 g/day oral / IV if avail)',   'CII bypass — electrons enter ubiquinol BYPASSING NDUFV2 N1b block',         'info'],
          ['CoQ10 / Ubiquinol (300–600 mg/day)',         'Electron carrier; Level C; ubiquinol preferred form',                       'info'],
          ['Thiamine B1 (100–300 mg/day)',               'MANDATORY empiric — exclude SLC19A3 + PDHC mimics before genetics',         'warning'],
          ['Biotin (5–10 mg/day)',                       'MANDATORY empiric — exclude BTD (biotinidase deficiency)',                   'warning'],
          ['NaHCO3 (IV, pH <7.20)',                     'Lactic acidosis correction; 0.5–1 mEq/kg; target pH >7.25',                 'warning'],
          ['LEV (levetiracetam)',                        'Preferred AED — renal excretion, zero mito toxicity',                       'success'],
          ['NIV / BiPAP',                               'Central respiratory compromise (SpO2 <92% or RR >40)',                      'info'],
          ['Carnitine (50–100 mg/kg/day)',               'Secondary carnitine deficiency; L-carnitine supplementation',               'info'],
          ['Echocardiography (q6–12 months)',            'HCM surveillance — mandatory in all NDUFV2 patients',                      'info'],
          ['Sevoflurane (anaesthesia)',                  'Safe volatile agent — NOT propofol (PRIS + HCM haemodynamic risk)',          'success'],
        ].map(([drug, mech, v]) => (
          <div key={drug} className="mb-2 p-2 rounded small" style={{
            background: v === 'success' ? '#e8f5e9' : v === 'warning' ? '#fff8e1' : LIGHT,
            borderLeft: `4px solid ${v === 'success' ? '#2e7d32' : v === 'warning' ? '#f57f17' : COLOR}`,
          }}>
            <strong>{drug}</strong> — {mech}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications Summary — NDUFV2/CI-Leigh" borderColor="#b71c1c">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Drug / Intervention</th><th>Severity</th><th>Mechanism Summary</th></tr>
            </thead>
            <tbody>
              {(data.contraindications || []).map(c => (
                <tr key={c.drug}>
                  <td><strong>{c.drug}</strong></td>
                  <td><span className={`badge ${c.severity.includes('ABSOLUTE') ? 'bg-danger' : 'bg-warning text-dark'}`}>
                    {c.severity.split(' ')[0]} {c.severity.split(' ')[1]}
                  </span></td>
                  <td className="small">{c.mechanism.split('\n')[0]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Spinner />;
  const sections = [
    ['Pharmacology & Therapeutics',  data.pharmacology   || []],
    ['Gene Concepts',                data.gene_concepts  || []],
    ['Disease Concepts',             data.disease_concepts || []],
    ['Prescribing Safety',           data.prescribing_safety || []],
  ];
  return (
    <div>
      {sections.map(([title, items]) => (
        <SectionCard key={title} title={title}>
          {items.map(item => (
            <div key={item.term} className="mb-3">
              <div className="fw-semibold small mb-1" style={{ color: COLOR }}>{item.term}</div>
              <pre className="small bg-light p-2 rounded" style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>
                {item.definition}
              </pre>
            </div>
          ))}
        </SectionCard>
      ))}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function NDUFV2Page() {
  const [tab,      setTab]      = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown,setBreakdown]= useState(null);
  const [defs,     setDefs]     = useState(null);
  const [err,      setErr]      = useState('');

  useEffect(() => {
    fetch(`${API}/api/ndufv2/overview`).then(r => r.json()).then(setOverview).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 && !breakdown)
      fetch(`${API}/api/ndufv2/breakdown`).then(r => r.json()).then(setBreakdown).catch(e => setErr(String(e)));
    if ((tab === 2) && !breakdown)
      fetch(`${API}/api/ndufv2/breakdown`).then(r => r.json()).then(setBreakdown).catch(e => setErr(String(e)));
    if (tab === 3 && !defs)
      fetch(`${API}/api/ndufv2/definitions`).then(r => r.json()).then(setDefs).catch(e => setErr(String(e)));
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          &#x1f9ec; NDUFV2 Leigh Syndrome — Isolated Complex I Deficiency
        </h4>
        <p className="text-muted small mb-0">
          NDUFV2 · 24 kDa N-module · N1b [2Fe-2S] Fe-S Cluster · 2nd Electron Relay Step ·
          18p11.22 · OMIM *600532 / #256000 · AR Biallelic · 40-patient cohort seed-621
          · HCM ~80% — DISTINCTIVE (highest in CI Fe-S relay series)
        </p>
      </div>

      {err && <div className="alert alert-danger small">{err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button className={`nav-link ${tab === i ? 'active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab    data={overview}   />}
      {tab === 1 && <PatientsTab    data={breakdown}  />}
      {tab === 2 && <TreatmentsTab  data={breakdown && { ...breakdown, contraindications: overview?.contraindications }} />}
      {tab === 3 && <DefinitionsTab data={defs}       />}
    </div>
  );
}
