'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & PEO', 'Systemic Features', 'Treatments', 'Definitions'];
const COLOR = '#01579b';   // deep ocean blue — TWNK/Twinkle helicase (replication fork unwinding; blue for helicase ring)
const LIGHT = '#e1f5fe';

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
        <div className="mt-2 p-2 rounded small" style={{ background: '#b3e5fc', borderLeft: `4px solid #0288d1` }}>
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
            ['Ataxia (mild-mod)', `${s.ataxia_pct}%`],
            ['SNHL', `${s.snhl_pct}%`],
            ['Parkinsonism', `${s.parkinsonism_pct}% (partial L-DOPA)`],
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
              <Bar key={o.label} label={o.label.replace(/-/g, ' ')} value={o.pct} color="#0277bd" />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Initial Misdiagnosis Distribution">
        {(data.misdiagnosis_distribution || []).map(m => (
          <Bar key={m.label} label={m.label.replace(/-/g, ' ')} value={m.pct} color="#e53935" />
        ))}
        <Alert variant="warning" text="⚠ Commonly misdiagnosed as CPEO-unclassified or Myasthenia Gravis (seronegative). Muscle biopsy (COX-negative fibres) + long-range PCR (multiple deletions, NOT depletion) + TWNK heterozygous panel is the diagnostic pathway. AD inheritance = family history of PEO / exercise intolerance (often subclinical in relatives)." />
      </SectionCard>

      <SectionCard title="Per-Patient Table">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>ID</th><th>Onset (yr)</th><th>PEO Pattern</th>
                <th>Exercise</th><th>Myopathy</th><th>SNHL</th><th>Ataxia</th>
                <th>Parkinson</th><th>CK (×ULN)</th><th>Lactate</th>
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
                  <td>{p.parkinsonism ? '⚠' : '–'}</td>
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
  const features = data.systemic_features || [];

  return (
    <div>
      <SectionCard title="Systemic Feature Prevalence">
        <Alert variant="success" text="✅ TWNK-PEOA3: AD disease; PEO is CARDINAL (100%); exercise intolerance ~75%; ataxia ~35% (intermediate — more than SLC25A4-PEOA1 ~20%, less than RNASEH1 ~85%). Key: NO hepatopathy (DDx from AR MDDS7/IOSCA + POLG1/Alpers); NO HCM; multiple deletions (NOT depletion); normal copy number." />
        {features.map(f => (
          <div key={f.label} className="mb-3">
            <Bar label={f.label} value={f.pct} />
            <div className="small text-muted ms-2">{f.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="DDx Matrix — PEOA3 vs Other PEO + mtDNA Deletion Diseases">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>Feature</th>
                <th>TWNK PEOA3 (AD)</th>
                <th>SLC25A4 PEOA1 (AD)</th>
                <th>POLG2 PEOA4 (AD)</th>
                <th>DNA2 PEOA5 (AD)</th>
                <th>RNASEH1 ARCO (AR)</th>
                <th>TWNK MDDS7 (AR)</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['Inheritance', 'AD dom-neg linker', 'AD dom-neg TM3/TM5', 'AD dom-neg p55', 'AD dom-neg FeS', 'AR biallelic LOF', 'AR biallelic LOF'],
                ['PEO', '100% CARDINAL', '100% CARDINAL', '100% CARDINAL', '100% CARDINAL', '~70% (secondary)', 'N/A (infantile)'],
                ['Exercise Intolerance', '~75%', '~85% HALLMARK', '~50%', '~50%', '~35%', 'N/A'],
                ['Proximal Myopathy', '~65%', '~80%', '~55%', '~68%', '~45%', '~50%'],
                ['Ataxia', '~35% (intermediate)', '~20% (mild)', '~35%', '~55%', '~85% PRIMARY', 'N/A (infantile ataxia)'],
                ['SNHL', '~30%', '~35%', '~40%', '~25%', '~25%', 'Severe (IOSCA)'],
                ['Hepatopathy', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES 75% hepatocerebral'],
                ['Cardiomyopathy', '<5% (RARE)', '<10% (RARE)', 'Rare', 'Rare', 'Rare', 'None typically'],
                ['HCM', 'NO', 'NO (DDx MDDS2)', 'NO', 'NO', 'NO', 'NO'],
                ['Epilepsy', '<8% (rare)', '<8% (rare)', '~12%', '<10%', '<8%', 'N/A'],
                ['Parkinsonism', '~12%', 'Rare', '~18%', '~10%', 'Rare', 'N/A'],
                ['mtDNA', 'Multiple deletions', 'Multiple deletions', 'Multiple deletions', 'Multiple deletions', 'Multiple deletions', 'DEPLETION (<30%)'],
                ['Copy Number', 'NORMAL', 'NORMAL', 'NORMAL', 'NORMAL', 'NORMAL', 'DEPLETED'],
                ['Onset (mean)', '~35 yr', '~38 yr', '~30 yr (mildest)', '~40 yr', '~38 yr', 'Infantile (<2 yr)'],
                ['Key mechanism', 'TWNK hexamer linker', 'ANT1 homodimer TM', 'POLG2 p55 dimer', 'DNA2 FeS helicase', 'RNaseH1 R-loop', 'TWNK hexamer LOF'],
                ['Key reference', 'Spelbrink 2001 NatGen', 'Kaukonen 2000 Sci', 'Walter 2010 AnnNeur', 'Ronchi 2013 AnnNeur', 'Reyes 2015 NatGen', 'Spelbrink 2001 NatGen'],
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
          <li><strong>Ophthalmology (6-monthly):</strong> Hess chart + ocular motility + Bell's phenomenon + ptosis degree + corneal sensation; ptosis surgery MDT when visual field compromised; propofol-free anaesthesia protocol</li>
          <li><strong>Exercise capacity (6-monthly):</strong> 6-minute walk test (6MWT); Borg RPE diary; CPET annually if exercise programme active; CK post-exercise (rhabdomyolysis surveillance)</li>
          <li><strong>Physiotherapy:</strong> aerobic exercise programme review 6-monthly; 30 min/day × 5/week; HIIT avoidance re-enforced; hydrotherapy option if balance affected by ataxia</li>
          <li><strong>Neurology (annual):</strong> neurological exam; SARA scale if ataxia; MoCA if cognitive concern; NCS/EMG if neuropathy suspected; brain MRI if ataxia progresses</li>
          <li><strong>Cardiology (annual):</strong> ECG + ECHO (cardiomyopathy rare but monitor); Holter if palpitations; ICD discussion if dilated CM develops</li>
          <li><strong>Audiometry (annual):</strong> pure-tone audiogram; hearing aid fitting if significant loss; cochlear implant MDT if severe (propofol-free anaesthesia)</li>
          <li><strong>Speech & Language:</strong> dysphagia screening annually; videofluoroscopy if symptoms; weight + nutritional status</li>
          <li><strong>Anaesthesia alert card:</strong> TWNK PEOA3 diagnosis; propofol AVOID (PRIS); VPA ABSOLUTE CI; KD CONTRAINDICATED; carry at all times</li>
          <li><strong>Family cascade:</strong> AD disease — 50% offspring risk; predictive genetic testing for first-degree relatives; inform presymptomatic relatives before prescribing VPA</li>
          <li><strong>Parkinsonian features:</strong> MDS-UPDRS if parkinsonism; DaT-SPECT if uncertain; L-DOPA trial if functional impact; clozapine for dopaminergic psychosis (not typical antipsychotics)</li>
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
export default function TwnkPeoa3Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/twnk-peoa3/overview`).then(r => r.json()),
      fetch(`${API}/api/twnk-peoa3/breakdown`).then(r => r.json()),
      fetch(`${API}/api/twnk-peoa3/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefs(df); })
      .catch(e => setError(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          TWNK / Twinkle Helicase — Progressive External Ophthalmoplegia Autosomal Dominant 3 (PEOA3)
        </h4>
        <p className="text-muted small mb-2">
          Twinkle Helicase (C10orf2) · 684 aa · 10q24.31 ·
          AD heterozygous dominant-negative (linker domain p.R303W / p.A318T) · mtDNA multiple deletions ·
          OMIM Gene *606075 · OMIM Disease #609286 · 40-patient cohort (seed-579)
        </p>
        <div className="mt-2 p-2 rounded small fw-bold" style={{ background: LIGHT, border: `1px solid ${COLOR}`, color: COLOR }}>
          ⛔ VPA ABSOLUTE CI · ⛔ KD CONTRAINDICATED · ⛔ Propofol AVOID (PRIS) ·
          ✅ LEV preferred AED · ✅ CoQ10 Level C · ✅ Aerobic Exercise Training Level B ·
          🔵 PEO CARDINAL (100%) · 🏃 Exercise Intolerance ~75% ·
          🧠 Ataxia ~35% (intermediate — LESS than RNASEH1 85%, MORE than SLC25A4-PEOA1 20%) ·
          🔵 mtDNA Multiple Deletions (NOT Depletion) · Normal copy number ·
          ⚡ TWNK-PEOA3 vs MDDS7/IOSCA — SAME GENE (10q24.31) OPPOSITE PHENOTYPE (Allelic Paradox) ·
          ❌ NO Hepatopathy · ❌ NO HCM · Spelbrink 2001 NatGenet
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
