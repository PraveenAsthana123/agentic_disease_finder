'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Ataxia', 'Systemic Features', 'Treatments', 'Definitions'];
const COLOR = '#004d40';   // deep teal — RNASEH1/AR-CPEO (AR, ataxia-primary, R-loop disease)
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
        <div className="mt-3 p-2 rounded small" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          <strong>Mechanism:</strong> {data.mechanism}
        </div>
        <div className="mt-2 p-2 rounded small" style={{ background: '#b2dfdb', borderLeft: `4px solid #00796b` }}>
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
            variant={c.severity === 'ABSOLUTE' ? 'danger' : 'warning'}
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

// ── Tab 2: Patients & Ataxia ──────────────────────────────────────────────────
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
            ['Mean SARA Score (ataxia pts)', `${s.avg_sara_score}/40`],
            ['Cerebellar Ataxia', `${s.ataxia_pct}% (primary feature)`],
            ['PEO', `${s.peo_pct}%`],
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
              <Bar key={o.label} label={o.label.replace(/-/g, ' ')} value={o.pct} color="#26a69a" />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Initial Misdiagnosis Distribution">
        {(data.misdiagnosis_distribution || []).map(m => (
          <Bar key={m.label} label={m.label.replace(/-/g, ' ')} value={m.pct} color="#e53935" />
        ))}
        <Alert variant="warning" text="⚠ Mean diagnostic delay ~9 years: RNASEH1-ARCO-PEO is most commonly misdiagnosed as spinocerebellar ataxia (SCA) or ILOCA. Muscle biopsy (COX-negative fibers) + long-range PCR (multiple deletions) + RNASEH1 biallelic panel testing is the definitive diagnostic pathway." />
      </SectionCard>

      <SectionCard title="Per-Patient Table">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>ID</th><th>Onset (yr)</th><th>Ataxia</th><th>SARA</th><th>PEO Pattern</th>
                <th>Neuropathy</th><th>Cognitive</th><th>Seiz</th><th>CK (×ULN)</th>
                <th>Lactate</th><th>Del %</th><th>DxDelay</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.age_onset}</td>
                  <td>{p.ataxia ? '✓' : '–'}</td>
                  <td>{p.sara > 0 ? p.sara : '–'}</td>
                  <td className="small">{(p.oph_pattern || '').replace(/-/g, ' ')}</td>
                  <td>{p.neuropathy ? '✓' : '–'}</td>
                  <td>{p.cognitive ? '✓' : '–'}</td>
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
    { label: 'Cerebellar Ataxia (primary feature)', pct: s.ataxia_pct, note: 'PRIMARY feature in RNASEH1 — gait ataxia + limb ataxia (SARA scale); Purkinje cell vulnerability to mtDNA deletion accumulation; distinguish from SCA (dominant, no COX-negative fibers, no mtDNA deletions)' },
    { label: 'PEO (ptosis + ophthalmoplegia)', pct: s.peo_pct, note: 'Secondary feature — typically develops after ataxia; bilateral ptosis + progressive ophthalmoplegia; Hess chart + Bell\'s phenomenon assessment; ptosis surgery if visual field obstructed' },
    { label: 'Sensory Neuropathy (axonal)', pct: s.neuropathy_pct, note: 'Distal paresthesias; vibration and proprioception loss preferentially; NCS/EMG confirms axonal sensory pattern; more prominent than in POLG2/PEOA4 (~25%) — reflects post-mitotic dorsal root ganglion neuron vulnerability' },
    { label: 'Proximal Myopathy', pct: s.myopathy_pct, note: 'Hip-girdle weakness; exercise intolerance; CK mildly elevated (<5× ULN); COX-negative fibers on biopsy; physiotherapy with aerobic training (avoid extreme exertion)' },
    { label: 'Cognitive Decline (frontal-subcortical)', pct: s.cognitive_pct, note: 'MORE COMMON in RNASEH1 (~40%) than POLG2 (~15%) or DNA2 (~20%); cerebellar cognitive-affective syndrome (CCAS); executive dysfunction + visuospatial deficits; neuropsychology assessment for monitoring' },
    { label: 'Sensorineural Hearing Loss (SNHL)', pct: s.snhl_pct, note: 'Progressive adult-onset SNHL; audiogram annually; hearing aids if significant; cochlear implant if severe — note: cochlear implant anaesthesia requires avoidance of propofol (PRIS risk)' },
    { label: 'Dysphagia', pct: s.dysphagia_pct, note: 'Pharyngeal weakness; SLT assessment; modified diet; videofluoroscopy if clinically impaired; PEG if severe; aspiration pneumonia risk — particularly with cerebellar dysarthria where swallow is also impaired' },
    { label: 'Seizures (uncommon)', pct: s.seizures_pct, note: 'Rare in RNASEH1-ARCO-PEO (<8%); if present, investigate for co-existing POLG1 (Alpers — epilepsy 100%); LEV preferred; VPA ABSOLUTE CI; RNASEH1 does NOT typically cause epilepsy' },
  ];

  return (
    <div>
      <SectionCard title="Systemic Feature Prevalence">
        <Alert variant="success" text="✅ RNASEH1-ARCO-PEO: AR disease; cerebellar ataxia is PRIMARY (vs PEO-primary in POLG2/DNA2). Key comparison: RNASEH1 has MORE ataxia (~85%) than DNA2/PEOA5 (~55%) and more than POLG2/PEOA4 (~35%). AR inheritance (not AD like POLG2/DNA2)." />
        {features.map(f => (
          <div key={f.label} className="mb-3">
            <Bar label={f.label} value={f.pct} />
            <div className="small text-muted ms-2">{f.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="DDx Matrix — RNASEH1 vs Other PEO-mtDNA Diseases">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>Feature</th><th>RNASEH1 (AR)</th><th>DNA2/PEOA5 (AD)</th><th>POLG2/PEOA4 (AD)</th><th>POLG1/Alpers (AR)</th><th>TWNK (AR)</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['Inheritance', 'AR (biallelic)', 'AD (dominant-neg FeS)', 'AD (dominant-neg p55)', 'AR (severe)', 'AR (hepatocerebral)'],
                ['Ataxia', '~85% (PRIMARY)', '~55% (prominent)', '~35% (secondary)', '30%', '20%'],
                ['PEO', '~70% (secondary)', '100% (cardinal)', '100% (cardinal)', '40–60%', '30%'],
                ['Sensory Neuropathy', '~60% (prominent)', '~40%', '~25%', '50%', 'Rare'],
                ['Cognitive Decline', '~40% (notable)', '~20%', '~15%', 'Common', '30%'],
                ['Hepatopathy', 'NO (key DDx)', 'NO', 'NO', '80% (liver failure)', '50% biallelic'],
                ['Epilepsy', '<8% (rare)', '<10% (rare)', '~12% (uncommon)', '100% (refractory)', '20%'],
                ['Onset (mean)', '~38 yr (range 18–62)', '~40 yr (range 25–65)', '~30 yr (mildest)', 'Paediatric/adult', 'Paediatric'],
                ['mtDNA', 'Multiple deletions', 'Multiple deletions', 'Multiple deletions', 'Depletion (± deletions)', 'Depletion'],
                ['Dx Delay', '~9 yr (longest)', '~7 yr', '~6 yr', 'Months (severe)', 'Years'],
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
          <li><strong>Neurology — Ataxia (6-monthly):</strong> SARA scale; Timed Up and Go (TUG); gait assessment; fall frequency; balance physiotherapy programme review</li>
          <li><strong>Neurology — General (annual):</strong> neurological exam; NCS/EMG every 2-3 years if neuropathy present; MoCA/neuropsychology if cognitive concern</li>
          <li><strong>Ophthalmology (annual):</strong> Hess chart + visual acuity + Bell's phenomenon + corneal sensation; ptosis surgery planning if visual field obstructed</li>
          <li><strong>Audiometry (annual):</strong> pure-tone audiogram; hearing aid referral if significant loss; cochlear implant MDT if severe</li>
          <li><strong>Speech & Language Therapy:</strong> annual swallowing + speech assessment; videofluoroscopy if dysphagia suspected; AAC if dysarthria severe</li>
          <li><strong>Physiotherapy:</strong> aerobic exercise programme (30 min/day, 5×/week); cerebellar balance training 3×/week; 6MWT; 6-monthly SARA-guided update</li>
          <li><strong>Cardiology (annual):</strong> ECG (rare cardiac involvement in RNASEH1); echo if symptoms</li>
          <li><strong>Family cascade testing:</strong> AR disease — biallelic gene panel for siblings (25% affected risk) when proband identified; predictive carrier testing for parents</li>
          <li><strong>Anaesthesia alert card:</strong> document RNASEH1-ARCO diagnosis, propofol AVOID (PRIS), VPA ABSOLUTE CI — carry at all times</li>
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
      <Alert variant="danger" text="⛔ VPA ABSOLUTE CI (accelerates mtDNA deletions + liver failure risk) · ⛔ KD CONTRAINDICATED (OXPHOS-dependent beta-oxidation) · ⛔ Propofol AVOID — PRIS · ✅ LEV preferred AED (renal excretion, no mitochondrial toxicity) · ✅ Cerebellar Rehabilitation Level B" />

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
        <div key={d.term} className="mb-3">
          <div className="fw-semibold small" style={{ color: COLOR }}>{d.term}</div>
          <div className="small text-muted">{d.definition}</div>
        </div>
      ))}
    </SectionCard>
  );
  return (
    <div>
      <Section title="Gene Biology" items={data.gene_biology} />
      <Section title="Disease Concepts" items={data.disease_concepts} />
      <Section title="Prescribing Safety" items={data.prescribing_safety} />
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────
export default function Rnaseh1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/rnaseh1/overview`).then(r => r.json()),
      fetch(`${API}/api/rnaseh1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/rnaseh1/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefs(df); })
      .catch(e => setError(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          RNASEH1 — AR Cerebellar Ataxia + PEO with mtDNA Multiple Deletions (ARCO-PEO)
        </h4>
        <p className="text-muted small mb-2">
          Ribonuclease H1 · 261 aa · 2p25.1 ·
          AR biallelic LOF (R-loop disease) · mtDNA multiple deletions · OMIM Gene *604122 ·
          40-patient cohort (seed-575)
        </p>
        <div className="mt-2 p-2 rounded small fw-bold" style={{ background: LIGHT, border: `1px solid ${COLOR}`, color: COLOR }}>
          ⛔ VPA ABSOLUTE CI · ⛔ KD CONTRAINDICATED · ⛔ Propofol AVOID (PRIS) ·
          ✅ LEV preferred AED · ✅ CoQ10 Level C · ✅ Cerebellar Rehabilitation Level B ·
          🟢 AR (both alleles — DDx from AD PEOA4/PEOA5) ·
          🔵 Ataxia PRIMARY (~85% — most ataxia-prominent in mtDNA deletion spectrum) ·
          🔵 mtDNA Multiple Deletions (NOT Depletion)
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
