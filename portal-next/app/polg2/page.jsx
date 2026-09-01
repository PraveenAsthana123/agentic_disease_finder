'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & PEO', 'Systemic Features', 'Treatments', 'Definitions'];
const COLOR = '#1565c0';   // deep blue — POLG2/PEOA4 (adult-onset, milder mitochondrial, ocular primary)
const LIGHT = '#e3f2fd';

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
        <div className="mt-3 p-2 rounded small" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          <strong>Mechanism:</strong> {data.mechanism}
        </div>
        <div className="mt-2 p-2 rounded small" style={{ background: '#e8eaf6', borderLeft: `4px solid #3949ab` }}>
          <strong>mtDNA Pattern:</strong> {data.mtdna_pattern}
        </div>
      </SectionCard>

      <SectionCard title="Clinical Feature Prevalence (40-Patient Cohort)">
        <div className="row g-3 mb-3">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
        {(data.feature_bars || []).map(b => (
          <Bar key={b.label} label={b.label} value={b.pct} />
        ))}
      </SectionCard>

      <SectionCard title="Key Contraindications">
        {(data.contraindications || []).map(c => (
          <Alert
            key={c.drug}
            variant={c.severity === 'ABSOLUTE' ? 'danger' : c.severity === 'CONTRAINDICATED' ? 'warning' : 'warning'}
            text={`⛔ ${c.drug} [${c.severity}]: ${c.reason}`}
          />
        ))}
      </SectionCard>

      <SectionCard title="Critical DDx Highlights">
        {(data.ddx_highlights || []).map((d, i) => (
          <Alert key={i} variant="info" text={d} />
        ))}
      </SectionCard>

      <SectionCard title="Key Investigations">
        <ul className="small mb-0">
          {(data.key_labs || []).map((l, i) => <li key={i}>{l}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="Key References">
        {(data.references || []).map((r, i) => (
          <div key={i} className="mb-2 small">
            <span className="fw-semibold">{r.author} ({r.year})</span> — <em>{r.journal}</em>: {r.title}.
            {r.note && <span className="text-muted ms-1">[{r.note}]</span>}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & PEO ──────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <Spinner />;
  const s = data.summary || {};
  return (
    <div>
      <SectionCard title="Cohort Summary">
        <div className="row g-2 small">
          {[
            ['N Patients', s.n_patients],
            ['Mean Age of Onset', `${s.avg_onset_years} years`],
            ['Mean Diagnosis Delay', `${s.avg_dx_delay_years} years`],
            ['Mean Deletion Load (muscle)', `${s.avg_deletion_load_pct}% of fibres`],
            ['PEO (Universal)', `${s.peo_pct}%`],
            ['Bilateral Ptosis', `${s.bilateral_ptosis_pct}%`],
          ].map(([k, v]) => (
            <div key={k} className="col-6 col-md-4">
              <span className="fw-semibold">{k}:</span> <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      <div className="row g-3">
        <div className="col-md-6">
          <SectionCard title="Variant / Etiology Distribution">
            {(data.etiology_distribution || []).map(e => (
              <Bar key={e.label} label={e.label.replace(/-/g, ' ')} value={e.pct} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Ophthalmoplegia Patterns">
            {(data.ophthalmoplegia_patterns || []).map(o => (
              <Bar key={o.label} label={o.label.replace(/-/g, ' ')} value={o.pct} color="#5c6bc0" />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Initial Misdiagnosis Distribution">
        {(data.misdiagnosis_distribution || []).map(m => (
          <Bar key={m.label} label={m.label.replace(/-/g, ' ')} value={m.pct} color="#e53935" />
        ))}
        <Alert variant="warning" text="⚠ Mean diagnostic delay 5+ years: POLG2/PEOA4 commonly misdiagnosed as myasthenia gravis, oculopharyngeal muscular dystrophy, or idiopathic CPEO." />
      </SectionCard>

      <SectionCard title="Per-Patient Table">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>ID</th><th>Onset (yr)</th><th>Oph Pattern</th><th>Myopathy</th>
                <th>SNHL</th><th>Ataxia</th><th>Seiz</th><th>CK (×ULN)</th>
                <th>Lactate</th><th>Del Load %</th><th>DxDelay</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.age_onset}</td>
                  <td className="small">{(p.oph_pattern || '').replace(/-/g, ' ')}</td>
                  <td>{p.myopathy ? '✓' : '–'}</td>
                  <td>{p.snhl ? `✓ ${p.pta_db}dB` : '–'}</td>
                  <td>{p.ataxia ? '✓' : '–'}</td>
                  <td>{p.seizures ? '✓' : '–'}</td>
                  <td>{p.ck_x_uln}</td>
                  <td>{p.lactate} mmol/L</td>
                  <td>{p.deletion_load_pct}%</td>
                  <td>{p.dx_delay_yr}yr</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Systemic Features ──────────────────────────────────────────────
function SystemicTab({ data }) {
  if (!data) return <Spinner />;
  const s = data.summary || {};
  const features = [
    { label: 'PEO (ptosis + ophthalmoplegia)', pct: s.peo_pct, note: 'Universal — cardinal feature; often isolated for years' },
    { label: 'Bilateral Ptosis', pct: s.bilateral_ptosis_pct, note: 'Symmetric bilateral ptosis; jaw/forehead compensation common' },
    { label: 'Proximal Myopathy', pct: s.myopathy_pct, note: 'Hip-girdle > shoulder-girdle; exercise intolerance; CK mildly elevated' },
    { label: 'Sensorineural Hearing Loss (SNHL)', pct: s.snhl_pct, note: 'Adult-onset progressive; bilateral; high-frequency initially; cochlear implant if severe' },
    { label: 'Ataxia', pct: s.ataxia_pct, note: 'Cerebellar gait ataxia ± limb ataxia; SARA scale for monitoring' },
    { label: 'Depression / Mood Disorder', pct: s.depression_pct, note: 'Clinically significant depression; may precede neurological diagnosis; SSRI safe' },
    { label: 'Sensory Neuropathy', pct: s.neuropathy_pct, note: 'Predominantly axonal sensory; distal paresthesias; NCS/EMG confirms' },
    { label: 'Dysphagia', pct: s.dysphagia_pct, note: 'Pharyngeal weakness ± oesophageal dysmotility; speech therapy; modified diet' },
    { label: 'Parkinsonism', pct: s.parkinsonism_pct, note: 'Bradykinesia ± rigidity; responds partially to levodopa; consider DaT-SPECT' },
    { label: 'Diplopia (uncommon)', pct: s.diplopia_pct, note: 'Uncommon due to symmetry; if present → asymmetric involvement or MG DDx' },
    { label: 'Seizures (uncommon)', pct: s.seizures_pct, note: 'Atypical for PEOA4; if present, investigate for compound POLG1 mutation' },
  ];

  return (
    <div>
      <SectionCard title="Systemic Feature Prevalence">
        <Alert variant="success" text="✅ POLG2/PEOA4 is the mildest mtDNA instability disease: most patients remain ambulatory and cognitively intact for decades. Systemic features develop gradually over years to decades." />
        {features.map(f => (
          <div key={f.label} className="mb-3">
            <Bar label={f.label} value={f.pct} />
            <div className="small text-muted ms-2">{f.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="DDx Matrix — Key Negative Distinctions">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>Feature</th><th>POLG2/PEOA4</th><th>POLG1/Alpers</th><th>SLC25A4/PEOA2</th><th>TWNK/PEOA3</th><th>KSS</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['PEO', '100% (cardinal)', '40-60%', '100%', '100%', '100%'],
                ['Epilepsy', '12% (uncommon)', '100% (often refractory)', 'Rare', '20%', '≤10%'],
                ['Hepatopathy', 'NO (key DDx)', '80% (Alpers liver failure)', 'NO', '50% biallelic', 'NO'],
                ['Cardiomyopathy', 'Rare', 'Rare', 'HCM 100% (MDDS2)', 'Rare', 'Conduction defects'],
                ['Retinopathy', 'NO', 'NO', 'NO', 'NO', 'YES (KSS hallmark)'],
                ['Onset', 'Adult (20-40yr)', 'Paediatric/adult', 'Infantile (MDDS2)', 'Adult', '<20yr'],
                ['Inheritance', 'AD', 'AR (severe), AD (mild PEO)', 'AR (MDDS2), AD (PEO2)', 'AD (PEO3)', 'Sporadic'],
                ['mtDNA', 'Multiple deletions', 'Depletion OR deletions', 'Depletion (MDDS2)', 'Multiple deletions', 'Single large deletion'],
              ].map(row => (
                <tr key={row[0]}>
                  {row.map((cell, i) => <td key={i} className={i === 0 ? 'fw-semibold' : ''}>{cell}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Monitoring Schedule (Annual unless noted)">
        <ul className="small mb-0">
          <li><strong>Ophthalmology:</strong> annual Hess chart + visual acuity + Bell's phenomenon + corneal sensation; pre-surgical assessment if ptosis correction planned</li>
          <li><strong>Audiology:</strong> annual pure-tone audiometry + speech discrimination; hearing aid fitting at moderate loss</li>
          <li><strong>Neurology:</strong> 6-monthly neurological exam; SARA ataxia scale; hand-grip strength; gait assessment</li>
          <li><strong>EMG/NCS:</strong> every 2-3 years if neuropathy suspected/present</li>
          <li><strong>Cardiology:</strong> ECG annually (arrhythmia screening); echo if symptoms; rare cardiac involvement in PEOA4</li>
          <li><strong>Psychiatry/Psychology:</strong> PHQ-9 at each visit (depression common); SSRI if indicated</li>
          <li><strong>Exercise tolerance:</strong> 6-min walk test + serum lactate at annual review</li>
          <li><strong>Family cascade testing:</strong> offer predictive genetic testing to first-degree adult relatives (AD, 50% risk)</li>
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Treatments ──────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <Alert variant="danger" text="⛔ VPA ABSOLUTE CI (accelerates mtDNA deletions + liver failure risk) · ⛔ KD CONTRAINDICATED (OXPHOS-dependent beta-oxidation) · ⛔ Propofol AVOID — PRIS · ✅ LEV preferred AED (renal excretion, no mitochondrial toxicity)" />

      {(data.treatments || []).map(t => (
        <SectionCard key={t.name} title={`${t.name} [${t.tier} — ${t.evidence}]`}>
          <div className="row g-2 small">
            <div className="col-12"><span className="fw-semibold">Mechanism:</span> {t.mechanism}</div>
            <div className="col-12 col-md-6"><span className="fw-semibold">Dose:</span> {t.dose}</div>
            <div className="col-12 col-md-6"><span className="fw-semibold">Monitoring:</span> {t.monitoring}</div>
            {t.caution && (
              <div className="col-12">
                <div className="p-2 rounded" style={{ background: '#fff8e1', borderLeft: '3px solid #f57f17' }}>
                  ⚠ {t.caution}
                </div>
              </div>
            )}
          </div>
        </SectionCard>
      ))}
    </div>
  );
}

// ── Tab 5: Definitions ──────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Spinner />;
  const Section = ({ title, items }) => (
    <SectionCard title={title}>
      {(items || []).map(d => (
        <div key={d.term || d.threshold} className="mb-3">
          <div className="fw-semibold small" style={{ color: COLOR }}>{d.term || d.threshold}</div>
          <div className="small text-muted">{d.definition || d.action}</div>
        </div>
      ))}
    </SectionCard>
  );
  return (
    <div>
      <Section title="Gene Biology" items={data.gene_biology} />
      <Section title="Disease Concepts" items={data.disease_concepts} />
      <Section title="Diagnostic Concepts" items={data.diagnostic_concepts} />
      <Section title="Pharmacology" items={data.pharmacology} />
      <Section title="Clinical Thresholds & Actions" items={data.thresholds} />
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────
export default function Polg2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/polg2/overview`).then(r => r.json()),
      fetch(`${API}/api/polg2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/polg2/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefs(df); })
      .catch(e => setError(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          POLG2 — Progressive External Ophthalmoplegia, AD (PEOA4)
        </h4>
        <p className="text-muted small mb-2">
          DNA Polymerase Gamma 2 (Accessory Subunit / p55) · 485 aa · 17q24.1 ·
          AD dominant-negative · mtDNA multiple deletions · OMIM Gene *604983 · Disease #610131 ·
          40-patient cohort (seed-571)
        </p>
        <div className="mt-2 p-2 rounded small fw-bold" style={{ background: LIGHT, border: `1px solid ${COLOR}`, color: COLOR }}>
          ⛔ VPA ABSOLUTE CI · ⛔ KD CONTRAINDICATED · ⛔ Propofol AVOID (PRIS) ·
          ✅ LEV preferred AED · ✅ CoQ10 Level C · ✅ Adult-onset (mildest mtDNA instability) ·
          🔵 mtDNA Multiple Deletions (NOT Depletion) · 🔵 Normal mtDNA Copy Number
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR, fontWeight: 600 } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SystemicTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={defs} />}
    </div>
  );
}
