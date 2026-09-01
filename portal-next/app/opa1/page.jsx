'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Vision', 'Systemic Features', 'Treatments', 'Definitions'];
const COLOR = '#1b5e20';   // deep forest green — OPA1/ADOA (optic nerve + mitochondrial dynamics; green for vision/eye)
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
          <strong>mtDNA Pattern (OPA1-Plus):</strong> {data.mtdna_pattern}
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

      <SectionCard title="Key Contraindications — The OPA1 Danger Seven">
        {(data.contraindications || []).map(c => (
          <Alert
            key={c.drug}
            variant={c.severity.startsWith('ABSOLUTE') ? 'danger' : 'warning'}
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

// ── Tab 2: Patients & Vision ──────────────────────────────────────────────
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
            ['Optic Atrophy (cardinal)', `${s.optic_atrophy_pct}%`],
            ['OPA1-Plus (multisystem)', `${s.opa1_plus_pct}%`],
            ['SNHL', `${s.snhl_pct}%`],
            ['Ataxia', `${s.ataxia_pct}%`],
            ['Myopathy (COX-neg fibres)', `${s.myopathy_pct}%`],
            ['Neuropathy', `${s.neuropathy_pct}%`],
            ['PEO (secondary)', `${s.peo_pct}%`],
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
          <SectionCard title="Disease Tier Distribution">
            {(data.tier_distribution || []).map(t => (
              <Bar key={t.label} label={t.label.replace(/-/g, ' ')} value={t.pct} color="#2e7d32" />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Initial Misdiagnosis Distribution">
        {(data.misdiagnosis_distribution || []).map(m => (
          <Bar key={m.label} label={m.label.replace(/-/g, ' ')} value={m.pct} color="#e53935" />
        ))}
        <Alert variant="warning" text="⚠ Commonly misdiagnosed as Normal-Tension Glaucoma (NTG) due to bilateral disc pallor with normal IOP. Key DDx: OPA1 — childhood onset, central scotoma, tritanopia, AD family history, normal IOP throughout, OPA1 panel positive. Misdiagnosis leads to unnecessary glaucoma medications and missed contraindication counselling (ethambutol, linezolid, tobacco)." />
      </SectionCard>

      <SectionCard title="Per-Patient Table">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>ID</th><th>Tier</th><th>Onset (yr)</th>
                <th>SNHL</th><th>Ataxia</th><th>Myopathy</th><th>Neuropathy</th>
                <th>PEO</th><th>Parkinson</th>
                <th>VA-R</th><th>VA-L</th><th>CK×ULN</th><th>DxDelay</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>
                    <span className={`badge ${p.tier === 'OPA1-Plus' ? 'bg-warning text-dark' : 'bg-success'}`}>
                      {p.tier}
                    </span>
                  </td>
                  <td>{p.age_onset}</td>
                  <td>{p.snhl ? '✓' : '–'}</td>
                  <td>{p.ataxia ? '✓' : '–'}</td>
                  <td>{p.myopathy ? '✓' : '–'}</td>
                  <td>{p.neuropathy ? '✓' : '–'}</td>
                  <td>{p.peo ? '✓' : '–'}</td>
                  <td>{p.parkinsonism ? '⚠' : '–'}</td>
                  <td>{p.va_right}</td>
                  <td>{p.va_left}</td>
                  <td>{p.ck_x_uln}×</td>
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
  const features = data.systemic_features || [];

  return (
    <div>
      <SectionCard title="Systemic Feature Prevalence">
        <Alert variant="success" text="✅ OPA1 ADOA: AD nuclear gene (NOT maternal inheritance like LHON); optic atrophy CARDINAL (100%); childhood onset; tritanopia (blue-yellow dyschromatopsia — DDx from LHON red-green); OPA1-Plus ~20% (GTPase missense = multisystem + mtDNA multiple deletions in muscle = same pattern as adPEO series). Key contraindications: ethambutol ABSOLUTE CI, linezolid ABSOLUTE CI, tobacco ABSOLUTE environmental CI." />
        {features.map(f => (
          <div key={f.label} className="mb-3">
            <Bar label={f.label} value={f.pct} />
            <div className="small text-muted ms-2">{f.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="DDx Matrix — OPA1 ADOA vs Other Hereditary Optic Neuropathies + OPA1-Plus vs adPEO Series">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>Feature</th>
                <th>OPA1 ADOA (pure)</th>
                <th>OPA1-Plus</th>
                <th>LHON (mt m.11778G>A)</th>
                <th>Wolfram (WFS1)</th>
                <th>TWNK PEOA3 (adPEO)</th>
                <th>TWNK MDDS7 (AR)</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['Inheritance', 'AD nuclear', 'AD nuclear', 'Mitochondrial (maternal)', 'AR nuclear', 'AD nuclear', 'AR nuclear'],
                ['Primary Feature', 'Optic atrophy', 'Optic atrophy + multisystem', 'Acute optic neuropathy', 'Optic atrophy + DM', 'PEO (cardinal)', 'IOSCA + hepatopathy'],
                ['Onset', 'Mean 7 yr (childhood)', 'Mean 6 yr (childhood)', '15–35 yr (subacute adult)', '~6 yr (optic)', 'Mean 35 yr (adult)', 'Infantile (<2 yr)'],
                ['Vision Loss Rate', 'Slow progressive', 'Slow progressive', 'Acute (weeks)', 'Progressive', 'N/A (not visual)', 'N/A (not visual)'],
                ['Colour Vision', 'Tritanopia (blue-yellow)', 'Tritanopia', 'Red-green + non-specific', 'Generalised', 'N/A', 'N/A'],
                ['Disc Appearance', 'Temporal pallor → full atrophy', 'Temporal pallor', 'Telangiectatic micro-angiopathy → atrophy', 'Temporal pallor', 'Normal (PEO focus)', 'Normal'],
                ['mtDNA Pattern', 'Normal (nuclear disease)', 'Multiple deletions (OPA1-Plus = adPEO)', 'mtDNA point mutation (m.11778A)', 'Normal (nuclear)', 'Multiple deletions', 'Depletion (<30%)'],
                ['SNHL', '5% (pure ADOA)', '~55% (OPA1-Plus)', 'Rare', 'Common (WFS2)', '~30%', '~100% (IOSCA)'],
                ['Hepatopathy', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES 75%'],
                ['PEO', 'NO', '~25% (secondary)', 'NO', 'NO', '100% CARDINAL', 'N/A (infantile)'],
                ['Key CI drugs', 'Ethambutol, Linezolid, Tobacco', 'As pure + VPA caution', 'Ethambutol, Linezolid, Tobacco', 'Linezolid caution', 'VPA, KD, Propofol', 'VPA, KD, Propofol'],
                ['Key reference', 'Delettre 2000 NatGen', 'Amati-Bonneau 2008 Brain', 'Wallace 1988 Science', 'Strom 1998 NatGen', 'Spelbrink 2001 NatGen', 'Nikali 2005 Neurology'],
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
          <li><strong>Ophthalmology (6-monthly):</strong> BCVA (Snellen both eyes) + HVF (central/centrocaecal scotoma progression) + OCT (RNFL temporal sector + macular GCL) + fundus photography (disc pallor documentation)</li>
          <li><strong>Colour vision (annual):</strong> FM-100 hue or Ishihara; document tritanopia axis; significant worsening = consider idebenone trial</li>
          <li><strong>VEP (annual):</strong> pattern-reversal P100 latency + amplitude; trend over time for progression rate</li>
          <li><strong>Audiogram (annual):</strong> pure-tone audiometry bilateral; SNHL in OPA1-Plus; cochlear implant MDT if severe</li>
          <li><strong>Neurology (annual for OPA1-Plus):</strong> SARA scale (ataxia), MoCA (cognitive), NCS/EMG (neuropathy), MDS-UPDRS (parkinsonism); brain MRI if cerebellar symptoms</li>
          <li><strong>CK + muscle (OPA1-Plus screen):</strong> CK annually; if elevated → muscle biopsy + long-range PCR (mtDNA multiple deletions = OPA1-Plus confirmation); reclassify tier</li>
          <li><strong>Driving assessment:</strong> DVLA/DVSA notification mandatory if BCVA &lt; 6/12 (UK legal standard); low vision specialist for driving cessation support</li>
          <li><strong>Genetics (family cascade):</strong> AD 50% offspring risk; first-degree relatives tested before any ethambutol/linezolid prescription; pre-symptomatic management (smoking cessation counselling pre-diagnosis)</li>
          <li><strong>Psychological support:</strong> vision loss depression comorbidity high; early referral to low vision + clinical psychology; RNIB/visual impairment charity registration</li>
          <li><strong>Anaesthesia alert card:</strong> OPA1 diagnosis; ethambutol ABSOLUTE CI; linezolid ABSOLUTE CI; tobacco/alcohol AVOID; carry at all times (for TB or MRSA treatment emergencies)</li>
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
      <Alert variant="danger" text="⛔ Ethambutol ABSOLUTE CI (Complex IV inhibition amplified by OPA1 LOF → irreversible vision loss) · ⛔ Linezolid ABSOLUTE CI (mitochondrial ribosome inhibition → acute DION) · ⛔ Tobacco ABSOLUTE CI (cyanide + acrolein → amplified OXPHOS failure in RGCs) · ⚠ VPA CAUTION in OPA1-Plus myopathy · ✅ Idebenone Level C (ADOA) · ✅ Tobacco cessation = single most important intervention · ✅ LEV preferred AED" />

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
export default function OPA1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/opa1/overview`).then(r => r.json()),
      fetch(`${API}/api/opa1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/opa1/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefs(df); })
      .catch(e => setError(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          OPA1 — Autosomal Dominant Optic Atrophy (ADOA / Kjer Disease / OPA1-Plus)
        </h4>
        <p className="text-muted small mb-2">
          OPA1 Mitochondrial Dynamin-Like GTPase · 960 aa · 3q29 ·
          AD haploinsufficiency (ADOA pure) + dominant-negative GTPase missense (OPA1-Plus) ·
          OMIM Gene *605290 · OMIM Disease #165500 (ADOA) / #125250 (OPA1-Plus) ·
          40-patient cohort (seed-581)
        </p>
        <div className="mt-2 p-2 rounded small fw-bold" style={{ background: LIGHT, border: `1px solid ${COLOR}`, color: COLOR }}>
          ⛔ Ethambutol ABSOLUTE CI · ⛔ Linezolid ABSOLUTE CI · ⛔ Tobacco ABSOLUTE CI ·
          ⚠ VPA CAUTION (OPA1-Plus myopathy) · ✅ Idebenone Level C · ✅ LEV preferred AED ·
          👁 Optic Atrophy CARDINAL (100%) · 🎨 Tritanopia (blue-yellow — DDx from LHON red-green) ·
          🧬 OPA1-Plus ~20% (GTPase missense = mtDNA deletions = adPEO-spectrum linkage) ·
          ❌ NOT maternal inheritance (nuclear AD — DDx from LHON mtDNA) ·
          Delettre 2000 NatGenet + Alexander 2000 NatGenet
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
