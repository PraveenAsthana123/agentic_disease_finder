'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & PEO', 'Systemic Features', 'Treatments', 'Definitions'];
const COLOR = '#1b5e20';   // deep forest green — SLC25A4/ANT1 PEOA1 (ATP energy transporter; exercise-intolerance dominant)
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
        <div className="mt-2 p-2 rounded small" style={{ background: '#c8e6c9', borderLeft: `4px solid #388e3c` }}>
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
            ['Mean Deletion Load (muscle)', `${s.avg_deletion_load_pct}% of fibres COX-negative`],
            ['PEO (cardinal)', `${s.peo_pct}%`],
            ['Exercise Intolerance', `${s.exercise_intol_pct}%`],
            ['Proximal Myopathy', `${s.myopathy_pct}%`],
            ['Cardiomyopathy', `${s.cardiomyopathy_pct}% (RARE — key DDx from AR MDDS2 HCM 100%)`],
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
              <Bar key={o.label} label={o.label.replace(/-/g, ' ')} value={o.pct} color="#43a047" />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Initial Misdiagnosis Distribution">
        {(data.misdiagnosis_distribution || []).map(m => (
          <Bar key={m.label} label={m.label.replace(/-/g, ' ')} value={m.pct} color="#e53935" />
        ))}
        <Alert variant="warning" text="⚠ Commonly misdiagnosed as Myasthenia Gravis (seronegative) or CPEO-unclassified. Muscle biopsy (COX-negative fibres) + long-range PCR (multiple deletions, not depletion) + SLC25A4 heterozygous panel is the diagnostic pathway. AD inheritance = family history of PEO, exercise intolerance." />
      </SectionCard>

      <SectionCard title="Per-Patient Table">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>ID</th><th>Onset (yr)</th><th>PEO Pattern</th>
                <th>Exercise</th><th>Myopathy</th><th>SNHL</th><th>Ataxia</th>
                <th>Cardio</th><th>CK (×ULN)</th><th>Lactate</th>
                <th>Del %</th><th>DxDelay</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.age_onset}</td>
                  <td className="small">{(p.oph_pattern || '').replace(/-/g, ' ')}</td>
                  <td>{p.exercise_intol ? '✓' : '–'}</td>
                  <td>{p.myopathy ? '✓' : '–'}</td>
                  <td>{p.snhl ? '✓' : '–'}</td>
                  <td>{p.ataxia ? '✓' : '–'}</td>
                  <td>{p.cardiomyopathy ? '⚠' : '–'}</td>
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
    { label: 'PEO (bilateral ptosis + ophthalmoplegia — CARDINAL 100%)', pct: s.peo_pct, note: 'CARDINAL feature in PEOA1 — 100% bilateral ptosis + progressive external ophthalmoplegia; Hess chart + Bell\'s phenomenon assessment mandatory; ptosis surgery if visual field obstructed; ophthalmology follow-up 6-monthly' },
    { label: 'Exercise Intolerance (hallmark)', pct: s.exercise_intol_pct, note: 'HALLMARK of ANT1 disease: reduced lactate threshold, disproportionate lactic acidosis with moderate exercise, exercise-induced myalgia + fatigue + CK rise; aerobic training improves mitochondrial biogenesis and exercise capacity' },
    { label: 'Proximal Myopathy', pct: s.myopathy_pct, note: 'Hip-girdle > shoulder-girdle weakness; stairs + rising from chair most affected; CK normal to mildly elevated (<5× ULN); physiotherapy with aerobic training (avoid HIIT); ragged-red fibres + COX-negative fibres on biopsy' },
    { label: 'Sensorineural Hearing Loss (SNHL)', pct: s.snhl_pct, note: 'Progressive adult-onset SNHL; audiogram annually; hearing aids if significant; cochlear implant if severe — NOTE: avoid propofol for cochlear implant surgery (PRIS risk)' },
    { label: 'Ataxia (cerebellar, mild)', pct: s.ataxia_pct, note: 'Mild cerebellar ataxia (~20%) — MUCH LESS than RNASEH1 (~85%) or DNA2 (~55%); PEOA1 is PEO-predominant not ataxia-predominant; if prominent ataxia, consider RNASEH1 or DNA2 in DDx' },
    { label: 'Depression / Mood Disorder', pct: s.depression_pct, note: 'Reactive + possibly neurobiological; fatigue + disability contribute; SSRI safe in mitochondrial disease; avoid tricyclics (CYP2D6 interactions with mitochondrial drugs)' },
    { label: 'Dysphagia', pct: s.dysphagia_pct, note: 'Pharyngeal/oesophageal weakness from myopathy; SLT assessment; texture modification; videofluoroscopy; PEG if severe; aspiration pneumonia risk' },
    { label: 'Cardiomyopathy (RARE)', pct: s.cardiomyopathy_pct, note: 'RARE in PEOA1 (<10%) — CRITICAL DDx from AR MDDS2 (HCM 100% at presentation); if cardiomyopathy present in PEOA1, ECG + ECHO monitoring; ICD if arrhythmia; VPA/KD CI in either form' },
  ];

  return (
    <div>
      <SectionCard title="Systemic Feature Prevalence">
        <Alert variant="success" text="✅ SLC25A4-PEOA1: AD disease; PEO is PRIMARY (cardinal 100%); exercise intolerance HALLMARK. Key comparison: NO HCM (contrast AR MDDS2 HCM 100%); NO hepatopathy (contrast POLG/DGUOK/TWNK-AR); ataxia MILD (~20%, unlike RNASEH1 ~85%)." />
        {features.map(f => (
          <div key={f.label} className="mb-3">
            <Bar label={f.label} value={f.pct} />
            <div className="small text-muted ms-2">{f.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="DDx Matrix — PEOA1 vs Other PEO + mtDNA Deletion Diseases">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>Feature</th><th>PEOA1 ANT1 (AD)</th><th>POLG2/PEOA4 (AD)</th><th>DNA2/PEOA5 (AD)</th><th>RNASEH1 (AR)</th><th>SLC25A4 MDDS2 (AR)</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['Inheritance', 'AD heterozygous', 'AD dominant-neg p55', 'AD dominant-neg FeS', 'AR biallelic LOF', 'AR biallelic LOF'],
                ['PEO', '100% CARDINAL', '100% CARDINAL', '100% CARDINAL', '~70% (secondary)', 'N/A (infantile)'],
                ['Exercise Intolerance', '~85% HALLMARK', '~50%', '~50%', '~35%', 'N/A (infantile)'],
                ['Proximal Myopathy', '~80%', '~55%', '~68%', '~45%', '~50%'],
                ['Ataxia', '~20% (mild)', '~35%', '~55%', '~85% (PRIMARY)', 'N/A'],
                ['SNHL', '~35%', '~40%', '~25%', '~25%', '~10%'],
                ['Cardiomyopathy', '<10% (RARE)', 'Rare', 'Rare', 'Rare', 'HCM 100% (CARDINAL)'],
                ['Hepatopathy', 'NO', 'NO', 'NO', 'NO', 'Rare (MDDS2 mild)'],
                ['Epilepsy', '<8% (rare)', '~12%', '<10%', '<8%', 'N/A'],
                ['mtDNA', 'Multiple deletions', 'Multiple deletions', 'Multiple deletions', 'Multiple deletions', 'DEPLETION <20%'],
                ['Copy number', 'NORMAL', 'NORMAL', 'NORMAL', 'NORMAL', 'DEPLETED'],
                ['Onset (mean)', '~38 yr', '~30 yr (mildest)', '~40 yr', '~38 yr', 'Infantile (<2 yr)'],
                ['Key reference', 'Kaukonen 2000 Sci', 'Walter 2010 AnnNeur', 'Ronchi 2013 AnnNeur', 'Reyes 2015 NatGen', 'Echaniz-Laguna 2012'],
              ].map(row => (
                <tr key={row[0]}>
                  {row.map((cell, i) => <td key={i} className={i === 0 ? 'fw-semibold' : ''}>{cell}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Monitoring Schedule">
        <ul className="small mb-0">
          <li><strong>Ophthalmology (6-monthly):</strong> Hess chart + ocular motility + Bell's phenomenon + ptosis degree + corneal sensation; ptosis surgery MDT when visual field compromised</li>
          <li><strong>Exercise capacity (6-monthly):</strong> 6-minute walk test (6MWT); Borg RPE diary; CPET annually if exercise programme active; CK post-exercise (rhabdomyolysis surveillance)</li>
          <li><strong>Physiotherapy:</strong> aerobic exercise programme review 6-monthly; 30 min/day × 5/week programme; HIIT avoidance re-enforced</li>
          <li><strong>Cardiology (annual):</strong> ECG + ECHO (cardiomyopathy rare but monitor); Holter if palpitations; ICD discussion if dilated CM develops</li>
          <li><strong>Audiometry (annual):</strong> pure-tone audiogram; hearing aid fitting if significant loss; cochlear implant MDT if severe (avoid propofol for any procedure)</li>
          <li><strong>Neurology (annual):</strong> neurological exam; MoCA if cognitive concern; NCS/EMG if neuropathy suspected; brain MRI if ataxia progresses</li>
          <li><strong>Speech & Language:</strong> dysphagia screening annually; videofluoroscopy if symptoms; weight + nutritional status</li>
          <li><strong>Anaesthesia alert card:</strong> SLC25A4-PEOA1 diagnosis; propofol AVOID (PRIS); VPA ABSOLUTE CI; KD CONTRAINDICATED; carry at all times</li>
          <li><strong>Family cascade:</strong> AD disease — 50% offspring risk; predictive genetic testing for first-degree relatives; inform presymptomatic relatives before prescribing VPA</li>
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
      <Alert variant="danger" text="⛔ VPA ABSOLUTE CI (accelerates mtDNA deletions + Reye-like hepatotoxicity) · ⛔ KD CONTRAINDICATED (OXPHOS-dependent beta-oxidation) · ⛔ Propofol AVOID — PRIS · ✅ LEV preferred AED (renal excretion, no mito toxicity) · ✅ Aerobic Exercise Training Level B · ✅ CoQ10 Level C" />

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
export default function Slc25a4Peoa1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/slc25a4-peoa1/overview`).then(r => r.json()),
      fetch(`${API}/api/slc25a4-peoa1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/slc25a4-peoa1/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefs(df); })
      .catch(e => setError(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          SLC25A4 / ANT1 — Progressive External Ophthalmoplegia Autosomal Dominant 1 (PEOA1)
        </h4>
        <p className="text-muted small mb-2">
          Adenine Nucleotide Translocator 1 · 298 aa · 4q35.1 ·
          AD heterozygous dominant-negative (TM3/TM5 missense) · mtDNA multiple deletions ·
          OMIM Gene *103220 · OMIM Disease #157640 · 40-patient cohort (seed-577)
        </p>
        <div className="mt-2 p-2 rounded small fw-bold" style={{ background: LIGHT, border: `1px solid ${COLOR}`, color: COLOR }}>
          ⛔ VPA ABSOLUTE CI · ⛔ KD CONTRAINDICATED · ⛔ Propofol AVOID (PRIS) ·
          ✅ LEV preferred AED · ✅ CoQ10 Level C · ✅ Aerobic Exercise Training Level B ·
          🟢 PEO CARDINAL (100%) · 🏃 Exercise Intolerance HALLMARK (~85%) ·
          ❤️ NO HCM — KEY DDx from SLC25A4 AR MDDS2 (HCM 100%) ·
          🔵 mtDNA Multiple Deletions (NOT Depletion) · Normal copy number ·
          ⚡ Kaukonen 2000 Science — First ANT1 adPEO
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
