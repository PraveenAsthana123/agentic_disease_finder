'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#0d47a1';   // deep blue — N4 Fe-S Q/N-module junction theme
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

      <SectionCard title="NDUFS7 = Q-Module/N-Module Junction — N4 [4Fe-4S] Electron Relay Bridge" borderColor="#0d47a1">
        <Alert variant="info" text="NDUFS7 (20 kDa PSST-related subunit) carries the N4 [4Fe-4S] Fe-S cluster at the Q-module/N-module junction of Complex I. N4 is the 4th electron relay bridge in the chain N3(NDUFV1) → N1b(NDUFS1) → N4(NDUFS7) → N5(NDUFS1) → N6a/N6b(NDUFS8) → N2(NDUFS2) → ubiquinone. Loss of N4 creates a direct electron transfer block upstream of N2 — CI at 5–20%, CII/CIII/CIV NORMAL. Unlike NDUFS3 (assembly scaffold failure with BN-PAGE intermediates), NDUFS7/N4 loss shows a cleaner BN-PAGE pattern (absent CI with fewer sub-assembly bands)." />
      </SectionCard>

      <SectionCard title="NO Peripheral Neuropathy — KEY DDx vs NDUFS1 (~50% neuropathy)" borderColor="#2e7d32">
        <Alert variant="success" text="NDUFS7-CI-Leigh does NOT cause peripheral neuropathy. This is a critical distinguishing feature from NDUFS1 (IP1 subunit, ~50% axonal/demyelinating neuropathy). Absence of neuropathy + isolated CI + no olfactory lesions + no leukodystrophy: genetic panel required to distinguish NDUFS2/NDUFS3/NDUFS7/NDUFS8 (all within Q-module or its junction)." />
      </SectionCard>

      <SectionCard title={`KPIs — 40-patient NDUFS7 cohort (seed-617)`}>
        <div className="row g-2">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Feature Frequencies (40-patient cohort, seed-617)">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar key={feat} label={feat} value={pct} color={featureColor(feat)} />
        ))}
      </SectionCard>

      <SectionCard title="Drug Contraindications" borderColor="#b71c1c">
        {(data.contraindications || []).map(c => (
          <div key={c.drug} className="mb-3 p-2 rounded" style={{ background: '#ffebee', borderLeft: '4px solid #c62828' }}>
            <div className="fw-bold small" style={{ color: '#b71c1c' }}>{c.drug} — {c.severity}</div>
            <div className="small text-muted mt-1" style={{ whiteSpace: 'pre-line' }}>{c.mechanism}</div>
          </div>
        ))}
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
      <SectionCard title="Complex Enzyme Activities (40-patient cohort)">
        <div className="row text-center small">
          {[
            ['CI Mean', `${complex_activities.CI_mean}%`, '#c62828'],
            ['CI Range', complex_activities.CI_range, '#b71c1c'],
            ['CII Mean', `${complex_activities.CII_mean}%`, '#2e7d32'],
            ['CIV Mean', `${complex_activities.CIV_mean}%`, '#2e7d32'],
          ].map(([l, v, c]) => (
            <div key={l} className="col-6 col-md-3 mb-2">
              <div className="fw-bold" style={{ color: c }}>{v}</div>
              <div className="text-muted">{l}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Genotype Distribution">
        {Object.entries(genotype_distribution).map(([g, n]) => (
          <div key={g} className="d-flex justify-content-between small border-bottom py-1">
            <span className="text-truncate" style={{ maxWidth: '80%' }}>{g}</span>
            <span className="fw-bold" style={{ color: COLOR }}>{n}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Feature Frequencies (breakdown)">
        {Object.entries(feature_frequencies).map(([feat, pct]) => (
          <Bar key={feat} label={feat} value={pct} color={featureColor(feat)} />
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients, seed-617)">
        <div style={{ overflowX: 'auto' }}>
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                {['ID', 'Sex', 'Onset (yr)', 'Lactate (mmol/L)', 'CI%', 'CII%', 'CIV%', 'Leigh MRI', 'Outcome'].map(h => (
                  <th key={h}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_yr}</td>
                  <td style={{ color: '#b71c1c' }}>{p.lactate_mm}</td>
                  <td style={{ color: '#c62828', fontWeight: 'bold' }}>{p.ci_pct}%</td>
                  <td style={{ color: '#2e7d32' }}>{p.cii_pct}%</td>
                  <td style={{ color: '#2e7d32' }}>{p.civ_pct}%</td>
                  <td>{p.has_leigh_mri ? '✓' : '—'}</td>
                  <td className="text-muted">{p.outcome.slice(0, 60)}</td>
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
function TreatmentsTab({ overview, breakdown }) {
  if (!overview || !breakdown) return <Spinner />;
  const { patients = [] } = breakdown;
  return (
    <div>
      <SectionCard title="CI-Leigh Differential Diagnosis (NDUFS7 vs Series)" borderColor="#6a1b9a">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead>
              <tr>
                <th>Gene</th><th>Module / Role</th><th>Distinguishing Feature</th><th>BN-PAGE / MRI Extra</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['NDUFS4',  'N-module (accessory)',               'Olfactory bulb lesions 52–65% (pathognomonic)',      'Olfactory T2 MRI'],
                ['NDUFV1',  'N-module (FMN core)',                'Leukodystrophy / white matter T2 40–50%',            'White matter MRI'],
                ['NDUFS1',  'N-module (IP1/75kDa, N1b/N4/N5)',   'Peripheral neuropathy ~50% (axonal/demyelin.)',       'None specific'],
                ['NDUFS2',  'Q-module (N2/PSST/49kDa)',          'NO neuropathy/olfactory/leukodystrophy; N2 terminal', 'Absent CI, few intermediates'],
                ['NDUFS3',  'Q-module (QP-C/30kDa scaffold)',    'Assembly failure — CI sub-assemblies on BN-PAGE',     'CI intermediates visible'],
                ['NDUFS7',  'Q/N junction (20kDa PSST, N4)',     'NO neuropathy/olfactory/leukodystrophy; N4 block',    'Absent CI, clean BN-PAGE'],
              ].map(([g, m, d, mr]) => (
                <tr key={g} style={{ background: g === 'NDUFS7' ? LIGHT : '' }}>
                  <td className="fw-bold" style={{ color: g === 'NDUFS7' ? COLOR : '#333' }}>{g}</td>
                  <td className="text-muted">{m}</td>
                  <td>{d}</td>
                  <td className="text-muted">{mr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Contraindications Summary" borderColor="#b71c1c">
        {(overview.contraindications || []).map(c => (
          <div key={c.drug} className="mb-2 p-2 rounded small" style={{ background: '#ffebee', borderLeft: '4px solid #c62828' }}>
            <div className="fw-bold" style={{ color: '#b71c1c' }}>{c.drug} — {c.severity}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Treatment Samples (per patient)">
        <div className="small">
          {patients.slice(0, 10).map(p => (
            <div key={p.id} className="border-bottom py-1">
              <span className="fw-semibold" style={{ color: COLOR }}>{p.id}</span>
              <span className="text-muted ms-2">{p.treatments}</span>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Definitions ───────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Spinner />;
  const sections = [
    { title: 'Pharmacology', items: data.pharmacology || [] },
    { title: 'Gene & Molecular Biology', items: data.gene_concepts || [] },
    { title: 'Disease Biology', items: data.disease_concepts || [] },
    { title: 'Prescribing Safety', items: data.prescribing_safety || [] },
  ];
  return (
    <div>
      {sections.map(({ title, items }) => items.length > 0 && (
        <SectionCard key={title} title={title}>
          {items.map(item => (
            <div key={item.term} className="mb-3">
              <div className="fw-bold small" style={{ color: COLOR }}>{item.term}</div>
              <div className="small text-muted mt-1" style={{ whiteSpace: 'pre-line' }}>{item.definition}</div>
            </div>
          ))}
        </SectionCard>
      ))}
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function NDUFS7Page() {
  const [tab, setTab]             = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ndufs7/overview`).then(r => r.json()),
      fetch(`${API}/api/ndufs7/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ndufs7/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov); setBreakdown(bk); setDefs(df);
    }).catch(e => setError(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 NDUFS7 Leigh Syndrome — Isolated Complex I Deficiency
        </h4>
        <div className="text-muted small">
          CI-Leigh / Q-Module/N-Module Junction N4 [4Fe-4S] Fe-S Cluster · 19p13.3 · OMIM *601825 / #618224 · AR Biallelic · 40-patient cohort seed-617
        </div>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR, fontWeight: 'bold' } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <TreatmentsTab overview={overview} breakdown={breakdown} />}
      {tab === 3 && <DefinitionsTab data={defs} />}
    </div>
  );
}
