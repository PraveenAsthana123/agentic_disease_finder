'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#4a148c';   // deep purple — N-module peripheral stabiliser, no Fe-S cluster
const LIGHT = '#f3e5f5';

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

      <SectionCard title="NDUFS5 = N-Module Peripheral Structural Subunit — Contacts NDUFS1/N5 — NO Fe-S Cluster" borderColor="#6a1b9a">
        <Alert variant="info" text="NDUFS5 (106 aa / ~10.8 kDa) is a small peripheral structural subunit of the N-module (hydrophilic arm) of Complex I. It contacts NDUFS1 (IP1/75 kDa), which carries the N5 [4Fe-4S] cluster in the Fe-S relay chain. NDUFS5 does NOT carry a Fe-S cluster itself — its role is structural and stabilising. Without NDUFS5, the fully assembled CI holocomplex cannot form, resulting in CI sub-assembly intermediates on BN-PAGE (similar to NDUFS3/NDUFS4 assembly failure). CI activity: 5–20%, CII/CIII/CIV NORMAL." />
        <Alert variant="success" text="NDUFS5-CI-Leigh does NOT cause peripheral neuropathy. This is a critical distinguishing feature from NDUFS1 (IP1 subunit, ~50% axonal/demyelinating neuropathy). Note: NDUFS5 contacts NDUFS1 structurally, but the absence of NDUFS5 does not directly impair the N5 Fe-S cluster of NDUFS1 — rather, it prevents proper holocomplex assembly." />
        <Alert variant="warning" text="BN-PAGE in NDUFS5: CI sub-assembly intermediates visible — an ASSEMBLY FAILURE pattern. This contrasts with NDUFS7/NDUFS8, which show cleaner absent/reduced CI (direct Fe-S relay block). The sub-assembly pattern is similar to NDUFS3 (Q-module scaffold) and NDUFS4 (N-module accessory) — all three are structural subunits without Fe-S clusters." />
      </SectionCard>

      <SectionCard title={`KPIs — 40-patient NDUFS5 cohort (seed-623)`}>
        <div className="row g-2">
          {data.kpis?.map(k => <KPI key={k.label} label={k.label} value={k.value} color={k.color} />)}
        </div>
      </SectionCard>

      <SectionCard title="Feature Frequencies — NDUFS5/CI-Leigh 40-patient cohort">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar key={feat} label={feat} value={pct} color={featureColor(feat)} />
        ))}
      </SectionCard>

      <SectionCard title="Fe-S Relay Chain — NDUFS5 Structural Position in Complex I N-Module" borderColor="#4a148c">
        <p className="small text-muted mb-2">
          NDUFS5 does not occupy a relay cluster step. It is a peripheral structural stabiliser that contacts NDUFS1 (N5 carrier).
          Its loss causes assembly failure — all relay steps fail because the holocomplex cannot form.
        </p>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Subunit</th><th>Fe-S Cluster(s)</th><th>Module</th><th>Relay / Structural Role</th><th>Key DDx Feature</th></tr>
            </thead>
            <tbody>
              {[
                ['NDUFV1', 'N3 [4Fe-4S]',      'N-module (FMN)',              '1st relay (primary NADH acceptor)',      'Leukodystrophy 40–50% (DISTINGUISHING)'],
                ['NDUFV2', 'N1b [2Fe-2S]',      'N-module',                    '2nd relay',                              'No major isolated DDx marker'],
                ['NDUFS7', 'N4 [4Fe-4S]',       'Q/N-module junction',         '3rd relay',                              'No neuropathy / no olfactory / no WM'],
                ['NDUFS8', 'N6a + N6b [4Fe-4S]','Q-module approach (TYKY)',    '4th + 5th relay',                        'Dual Fe-S block; cleaner absent CI (BN-PAGE)'],
                ['NDUFS5', '— (NO cluster)',     'N-module peripheral',         'Structural stabiliser — contacts N5',   'Assembly failure; sub-assembly intermediates (BN-PAGE)'],
                ['NDUFS1', 'N5 [4Fe-4S]',       'N-module (NDUFS5 contacts)',  '6th relay',                              'Peripheral neuropathy 50% (KEY DDx)'],
                ['NDUFS2', 'N2 [4Fe-4S]',       'Q-module (terminal)',         '7th relay → UQ',                        'Terminal N2; HCM ~8%'],
                ['NDUFS4', '—',                  'N-module (accessory)',        'Assembly role',                          'Olfactory bulb lesions 52–65% (pathognomonic)'],
              ].map(r => (
                <tr key={r[0]} style={{ background: r[0] === 'NDUFS5' ? LIGHT : undefined, fontWeight: r[0] === 'NDUFS5' ? 'bold' : undefined }}>
                  {r.map((c, i) => <td key={i}>{c}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="BN-PAGE Pattern Comparison — Assembly Failure vs Direct Fe-S Block" borderColor="#6a1b9a">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Subunit</th><th>Role Type</th><th>BN-PAGE Pattern</th><th>Mechanistic Basis</th></tr>
            </thead>
            <tbody>
              {[
                ['NDUFS5', 'Structural / N-module peripheral (NO Fe-S)', 'Sub-assembly intermediates — ASSEMBLY FAILURE', 'Holocomplex cannot form; NDUFS1 mispositioning'],
                ['NDUFS4', 'Accessory / N-module (NO Fe-S)',             'Sub-assembly intermediates — ASSEMBLY FAILURE', 'N-module assembly destabilised without accessory subunit'],
                ['NDUFS3', 'Scaffold / Q-module (NO Fe-S)',               'Sub-assembly intermediates — ASSEMBLY FAILURE', 'Q-module scaffold absent; CI sub-complexes accumulate'],
                ['NDUFS7', 'Fe-S relay / N4 cluster carrier',            'Absent/severely reduced CI — CLEAN LOSS',        'Direct N4 relay block; full CI can partially assemble then fail'],
                ['NDUFS8', 'Fe-S relay / N6a+N6b dual cluster (TYKY)',   'Absent/severely reduced CI — CLEAN LOSS',        'Dual N6a/N6b direct relay block; similar to NDUFS7'],
              ].map(r => (
                <tr key={r[0]} style={{ background: r[0] === 'NDUFS5' ? LIGHT : undefined, fontWeight: r[0] === 'NDUFS5' ? 'bold' : undefined }}>
                  {r.map((c, i) => <td key={i}>{c}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Alert variant="info" text="BN-PAGE sub-assembly intermediates point toward a STRUCTURAL subunit defect (NDUFS3/NDUFS4/NDUFS5 — no Fe-S clusters). Cleaner absent CI on BN-PAGE suggests a direct Fe-S relay block (NDUFS7/NDUFS8). Both patterns produce identical isolated CI deficiency biochemically — genetics is required for definitive diagnosis." />
      </SectionCard>

      <SectionCard title="Absolute Contraindications — NDUFS5/CI-Leigh" borderColor="#b71c1c">
        <Alert variant="danger" text="ABSOLUTE CI: Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + hepatotoxicity. In NDUFS5-CI-Leigh, CI is already at 5–20% due to assembly failure; VPA collapses residual OXPHOS. Use LEV." />
        <Alert variant="danger" text="ABSOLUTE CI: Metformin — direct Complex I inhibitor at ND1/quinone-binding site. NDUFS5 assembly failure already limits CI to 5–20%; metformin eliminates residual CI activity. Catastrophic." />
        <Alert variant="danger" text="ABSOLUTE CI: Linezolid + Chloramphenicol — mt-ribosome 23S rRNA block → all 7 mtDNA-encoded ND subunits absent → CI P-module collapses → near-zero CI in already assembly-deficient NDUFS5 patient." />
        <Alert variant="danger" text="CONTRAINDICATED: Ketogenic Diet — forces NADH overload via beta-oxidation; assembly-failed CI cannot re-oxidise NADH → worsened lactic acidosis. Beneficial in GLUT1-DS and PDHD, not CI-Leigh." />
        <Alert variant="warning" text="AVOID: Propofol (PRIS risk — CIV inhibition creates 2nd ETC block downstream of already assembly-impaired CI). Use sevoflurane or dexmedetomidine." />
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
      <SectionCard title="Complex Activity Summary — NDUFS5 Cohort">
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

      <SectionCard title="40-Patient Cohort (seed-623)">
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
      <SectionCard title="NDUFS5-CI-Leigh vs CI-Leigh Series — Critical DDx" borderColor="#1b5e20">
        <Alert variant="info" text="All CI-Leigh diseases share identical biochemistry: CI 5–20%, CII/CIII/CIV NORMAL. The only differentiators are clinical + MRI + BN-PAGE features. NDUFS5 has NO peripheral neuropathy (DDx NDUFS1 50%), NO olfactory bulb lesions (DDx NDUFS4 52–65%), NO leukodystrophy (DDx NDUFV1 40–50%). BN-PAGE sub-assembly intermediates point to structural assembly failure (similar to NDUFS3/NDUFS4) rather than direct Fe-S relay block (NDUFS7/NDUFS8)." />
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Gene</th><th>Module / Function</th><th>BN-PAGE</th><th>Distinguishing Feature</th><th>Key DDx Marker</th></tr>
            </thead>
            <tbody>
              {[
                ['NDUFS4',  'N-module (accessory, NO Fe-S)',          'Sub-assembly intermediates',  'Olfactory bulb lesions 52–65% (pathognomonic)',  'Olfactory T2 MRI'],
                ['NDUFV1',  'N-module FMN/N3 (51 kDa, Fe-S N3)',     'Variable',                    'Leukodystrophy / white matter T2 40–50%',        'White matter FLAIR'],
                ['NDUFS1',  'N-module IP1 (75 kDa) N5 Fe-S',         'Variable',                    'Peripheral neuropathy ~50%',                      'EMG / NCS'],
                ['NDUFS7',  'Q/N-junction N4 Fe-S',                   'Absent CI — clean loss',      'Single N4 block; no neuropathy/WM/olfactory',    'WES / genetic panel'],
                ['NDUFS8',  'Q-module TYKY N6a+N6b Fe-S',            'Absent CI — clean loss',      'Dual N6a+N6b block; no neuropathy/WM/olfactory', 'WES / genetic panel'],
                ['NDUFS5',  'N-module peripheral structural (NO Fe-S)','Sub-assembly intermediates', 'Assembly failure; no neuropathy/WM/olfactory',   'WES / genetic panel'],
                ['NDUFS2',  'Q-module N2 terminal Fe-S (49 kDa)',    'Variable',                    'Terminal N2 loss; HCM ~8%',                       'WES / genetic panel'],
                ['NDUFS3',  'Q-module scaffold (30 kDa QP-C, NO Fe-S)','Sub-assembly intermediates','Assembly failure; similar BN-PAGE to NDUFS5',    'BN-PAGE + genetics'],
              ].map(r => (
                <tr key={r[0]} style={{ background: r[0] === 'NDUFS5' ? LIGHT : undefined, fontWeight: r[0] === 'NDUFS5' ? 'bold' : undefined }}>
                  {r.map((c, i) => <td key={i}>{c}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Supportive Treatment Ladder — NDUFS5/CI-Leigh" borderColor="#2e7d32">
        {[
          ['IV Dextrose (GIR 6-8 mg/kg/min)',          'Crisis — FIRST LINE — never fast in CI-Leigh',                              'success'],
          ['Riboflavin B2 (100–400 mg/day)',            'CI-specific (FMN precursor — N-module upstream of NDUFS5 assembly block)',  'info'],
          ['Succinate (2–8 g/day oral / IV if avail)', 'CII bypass — electrons enter ubiquinol BYPASSING NDUFS5 assembly failure',  'info'],
          ['CoQ10 / Ubiquinol (300–600 mg/day)',        'Electron carrier; Level C; ubiquinol preferred form',                       'info'],
          ['Thiamine B1 (100–300 mg/day)',              'MANDATORY empiric — exclude SLC19A3 + PDHC mimics before genetics',         'warning'],
          ['Biotin (5–10 mg/day)',                      'MANDATORY empiric — exclude BTD (biotinidase deficiency)',                   'warning'],
          ['NaHCO3 (IV, pH <7.20)',                    'Lactic acidosis correction; 0.5–1 mEq/kg; target pH >7.25',                 'warning'],
          ['LEV (levetiracetam)',                       'Preferred AED — renal excretion, zero mito toxicity',                       'success'],
          ['NIV / BiPAP',                              'Central respiratory compromise (SpO2 <92% or RR >40)',                      'info'],
          ['Carnitine (50–100 mg/kg/day)',              'Secondary carnitine deficiency; L-carnitine supplementation',                'info'],
          ['Sevoflurane (anaesthesia)',                 'Safe volatile agent — NOT propofol (PRIS risk)',                             'success'],
        ].map(([drug, mech, v]) => (
          <div key={drug} className="mb-2 p-2 rounded small" style={{
            background: v === 'success' ? '#e8f5e9' : v === 'warning' ? '#fff8e1' : LIGHT,
            borderLeft: `4px solid ${v === 'success' ? '#2e7d32' : v === 'warning' ? '#f57f17' : COLOR}`,
          }}>
            <strong>{drug}</strong> — {mech}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications Summary — NDUFS5/CI-Leigh" borderColor="#b71c1c">
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
export default function NDUFS5Page() {
  const [tab,      setTab]      = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown,setBreakdown]= useState(null);
  const [defs,     setDefs]     = useState(null);
  const [err,      setErr]      = useState('');

  useEffect(() => {
    fetch(`${API}/api/ndufs5/overview`).then(r => r.json()).then(setOverview).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 && !breakdown)
      fetch(`${API}/api/ndufs5/breakdown`).then(r => r.json()).then(setBreakdown).catch(e => setErr(String(e)));
    if ((tab === 2) && !breakdown)
      fetch(`${API}/api/ndufs5/breakdown`).then(r => r.json()).then(setBreakdown).catch(e => setErr(String(e)));
    if (tab === 3 && !defs)
      fetch(`${API}/api/ndufs5/definitions`).then(r => r.json()).then(setDefs).catch(e => setErr(String(e)));
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          &#x1f9ec; NDUFS5 Leigh Syndrome — Isolated Complex I Deficiency
        </h4>
        <p className="text-muted small mb-0">
          NDUFS5 · N-Module Peripheral Structural Subunit · Contacts NDUFS1/N5 [4Fe-4S] · NO Fe-S Cluster ·
          1p34.3 · OMIM *603847 / #256000 · AR Biallelic · 40-patient cohort seed-623
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
