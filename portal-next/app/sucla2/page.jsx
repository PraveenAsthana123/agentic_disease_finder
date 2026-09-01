'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Variants', 'MRI & SNHL', 'Treatments', 'Definitions'];
const COLOR = '#6d4c41';   // deep brown — SUCLA2/MDDS10 (encephalomyopathic; TCA-cycle; Faroe founder)
const LIGHT = '#efebe9';

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

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview...</div>;
  const cis = data.key_contraindications || [];
  const ddx = data.pathognomonic_ddx || {};
  const cs = data.cohort_summary || {};

  return (
    <div>
      {/* Critical VPA Warning Banner */}
      <div className="mb-3 p-3 rounded fw-bold text-center" style={{ background: '#b71c1c', color: 'white', fontSize: '1.05rem' }}>
        ⛔ VPA = ABSOLUTE CONTRAINDICATION IN SUCLA2 MDDS10 — mtDNA DEPLETION (POLG INHIBITION + CoA SEQUESTRATION + HEPATOTOXIC EPOXIDE)
      </div>
      <div className="mb-3 p-2 rounded fw-semibold text-center" style={{ background: '#e65100', color: 'white', fontSize: '0.95rem' }}>
        🚫 KETOGENIC DIET = CONTRAINDICATED — Forces OXPHOS-Dependent Fat Oxidation That Fails in mtDNA Depletion
      </div>

      {/* Key distinguishing banner */}
      <div className="mb-4 p-2 rounded fw-semibold text-center" style={{ background: '#1b5e20', color: 'white', fontSize: '0.9rem' }}>
        ✅ NO HEPATOPATHY — Liver NORMAL (DDx DGUOK/MPV17/TWNK/POLG) · MMA is MILD (DDx MUT/MMACHC) · NO Homocystinuria (DDx MMACHC/cblC)
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Hypotonia" value="100%" color={COLOR} />
        <KPI label="Mild MMA" value="100%" color="#e65100" />
        <KPI label="Lactic Acidosis" value="~85%" color="#c62828" />
        <KPI label="Leigh-like MRI" value="~80%" color="#6a1b9a" />
        <KPI label="SNHL" value="~75%" color="#1565c0" />
        <KPI label="Dystonia" value="~70%" color={COLOR} />
      </div>

      {/* Identity */}
      <SectionCard title="🧬 Disease Identity">
        <div className="row g-2 small">
          <div className="col-md-6"><strong>Disease:</strong> {data.disease}</div>
          <div className="col-md-6"><strong>Gene:</strong> {data.gene} — {data.protein} ({data.protein_size_aa} aa)</div>
          <div className="col-md-4"><strong>Locus:</strong> {data.locus}</div>
          <div className="col-md-4"><strong>OMIM Gene:</strong> {data.omim_gene} &nbsp; <strong>Disease:</strong> {data.omim_disease}</div>
          <div className="col-md-4"><strong>Inheritance:</strong> {data.inheritance}</div>
          <div className="col-12"><strong>Mechanism:</strong> <span className="text-muted">{data.mechanism}</span></div>
          <div className="col-12"><strong>First described:</strong> {data.first_description}</div>
        </div>
      </SectionCard>

      {/* Faroe Islands Founder */}
      <div className="mb-4 p-3 rounded" style={{ background: '#e3f2fd', border: '2px solid #1565c0' }}>
        <div className="fw-bold mb-1" style={{ color: '#1565c0' }}>🧬 Faroe Islands Founder — p.Asp333Gly</div>
        <div className="small">{data.faroe_founder}</div>
        <div className="small mt-1 text-muted">Highest MDDS prevalence per capita of any population worldwide. Carrier screening recommended in Faroese communities.</div>
      </div>

      {/* Pathognomonic DDx */}
      <SectionCard title="🔑 Key Diagnostic Pearls — SUCLA2 DDx" borderColor="#1565c0">
        <div className="row g-3">
          {Object.entries(ddx).map(([key, val], i) => (
            <div key={i} className="col-md-6">
              <div className="p-3 rounded h-100" style={{ background: '#e8f5e9', border: '2px solid #2e7d32' }}>
                <div className="fw-bold small mb-1" style={{ color: '#2e7d32' }}>
                  {key === 'methylmalonic_aciduria_mild' ? '🔬 Mild MMA (DDx MUT/MMACHC)' :
                   key === 'no_hepatopathy' ? '🏥 No Hepatopathy (DDx DGUOK/MPV17/TWNK)' :
                   key === 'no_homocystinuria' ? '🧪 No Homocystinuria (DDx MMACHC/cblC)' :
                   '🦻 SNHL 75% — Cochlear Implants Effective'}
                </div>
                <div className="small text-muted">{val}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Contraindications */}
      <SectionCard title="⛔ Contraindications" borderColor="#c62828">
        {cis.map((ci, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: i < 2 ? '#ffebee' : '#fff8e1', border: `1px solid ${i < 2 ? '#c62828' : '#f57f17'}` }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold small">{ci.drug}</span>
              <span className="badge" style={{ background: i < 2 ? '#c62828' : '#e65100', fontSize: '0.65rem' }}>{ci.level}</span>
            </div>
            <div className="text-muted small">{ci.reason}</div>
          </div>
        ))}
      </SectionCard>

      {/* Cohort Summary */}
      <SectionCard title="👥 Cohort Summary (n=40, seed-557)">
        <div className="row g-2 small">
          <div className="col-md-4"><strong>Patients:</strong> {cs.n}</div>
          <div className="col-md-4"><strong>Median onset:</strong> {cs.median_onset_months} months</div>
          <div className="col-md-4"><strong>Median diagnosis:</strong> {cs.median_diagnosis_months} months</div>
          <div className="col-12 mt-2"><strong>Top presenting features:</strong></div>
          {(cs.top_presenting_features || []).map((f, i) => (
            <div key={i} className="col-md-6">
              <span className="badge me-1" style={{ background: COLOR }}>{f}</span>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const phenotypes = data.phenotype_distribution || [];
  const genotypes = data.genotype_breakdown || [];
  const features = data.feature_prevalence || [];

  return (
    <div>
      <SectionCard title="👥 Cohort Phenotype Distribution (n=40, seed-557)">
        <div className="row g-3 mb-3">
          {phenotypes.map((g, i) => (
            <div key={i} className="col-md-6 mb-3">
              <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${[COLOR, '#1565c0', '#c62828'][i] || COLOR}` }}>
                <div className="card-body text-center">
                  <div className="fw-bold fs-3" style={{ color: [COLOR, '#1565c0', '#c62828'][i] || COLOR }}>{g.n}</div>
                  <div className="fw-semibold small">{g.name}</div>
                  <div className="text-muted small">{g.pct}% of cohort</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🧬 Genotype Breakdown">
        {genotypes.map((v, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff', border: '1px solid #d7ccc8' }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold small" style={{ color: COLOR }}>{v.genotype}</span>
              <span className="badge" style={{ background: COLOR }}>{Math.round(v.fraction * 40)} patients ({Math.round(v.fraction * 100)}%)</span>
            </div>
            <div className="small text-muted mb-1"><strong>Example:</strong> {v.example}</div>
            <div className="small text-muted">{v.phenotype}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📊 Clinical Feature Prevalence">
        {features.map((f, i) => (
          <div key={i} className="mb-2">
            <Bar
              label={f.feature}
              value={f.pct}
              color={f.pct === 0 ? '#2e7d32' : f.pct === 100 ? '#c62828' : COLOR}
            />
            <div className="text-muted small mb-2" style={{ marginLeft: 4 }}>{f.note}</div>
          </div>
        ))}
      </SectionCard>

      {/* Sample Patients */}
      {data.patients_sample && (
        <SectionCard title="🔍 Sample Patients (first 8)">
          <div className="table-responsive">
            <table className="table table-sm small">
              <thead><tr style={{ background: LIGHT }}>
                <th>#</th><th>Sex</th><th>Ethnicity</th><th>Onset (mo)</th>
                <th>Diag (mo)</th><th>MMA (µmol/mmol)</th><th>Lactate</th>
                <th>SNHL</th><th>Leigh MRI</th><th>Dystonia</th>
              </tr></thead>
              <tbody>
                {data.patients_sample.map((p, i) => (
                  <tr key={i}>
                    <td>{p.pid}</td>
                    <td>{p.sex}</td>
                    <td>{p.ethnicity}</td>
                    <td>{p.onset_months}</td>
                    <td>{p.diagnosis_months}</td>
                    <td style={{ color: '#e65100' }}>{p.mma_urine_umol_per_mmol_creat}</td>
                    <td style={{ color: p.lactic_acidosis ? '#c62828' : '#2e7d32' }}>
                      {p.peak_lactate_mmol} mmol/L
                    </td>
                    <td style={{ color: p.snhl ? '#1565c0' : '#2e7d32' }}>{p.snhl ? 'Yes' : 'No'}</td>
                    <td style={{ color: p.leigh_mri ? '#6a1b9a' : '#2e7d32' }}>{p.leigh_mri ? 'Yes' : 'No'}</td>
                    <td style={{ color: p.dystonia ? COLOR : '#2e7d32' }}>{p.dystonia ? 'Yes' : 'No'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}
    </div>
  );
}

function MriSnhlTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const mma = data.mma_ddx_table || {};
  const timeline = data.disease_timeline || [];

  return (
    <div>
      {/* MMA DDx table */}
      <SectionCard title="🔬 Methylmalonic Aciduria DDx — SUCLA2 vs MUT vs MMACHC" borderColor="#e65100">
        <Alert variant="warning" text="⚠️ KEY: SUCLA2 MMA is MILD — a metabolic bystander of TCA blockade. NOT primary organic aciduria. No ketoacidosis. DDx from MUT (severe) and MMACHC (moderate + Hcy)." />
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr style={{ background: '#fff3e0' }}>
              <th>Parameter</th><th>SUCLA2 MDDS10</th><th>MUT Methylmalonic Aciduria</th><th>MMACHC / cblC</th>
            </tr></thead>
            <tbody>
              <tr>
                <td><strong>Urine MMA</strong></td>
                <td style={{ color: '#e65100' }}>{mma.sucla2_mma_umol_mmol_creat_typical}</td>
                <td style={{ color: '#c62828' }}>{mma.mut_mma_umol_mmol_creat_typical}</td>
                <td style={{ color: '#e65100' }}>{mma.mmachc_mma_umol_mmol_creat_typical}</td>
              </tr>
              <tr>
                <td><strong>Plasma Hcy</strong></td>
                <td style={{ color: '#2e7d32' }}>{mma.sucla2_plasma_hcy}</td>
                <td style={{ color: '#2e7d32' }}>Normal</td>
                <td style={{ color: '#c62828' }}>{mma.mmachc_plasma_hcy}</td>
              </tr>
              <tr>
                <td><strong>Ketoacidosis risk</strong></td>
                <td style={{ color: '#2e7d32' }}>{mma.sucla2_ketoacidosis_risk}</td>
                <td style={{ color: '#c62828' }}>{mma.mut_ketoacidosis_risk}</td>
                <td>MODERATE</td>
              </tr>
              <tr>
                <td><strong>Hepatopathy</strong></td>
                <td style={{ color: '#2e7d32' }}>ABSENT</td>
                <td style={{ color: '#c62828' }}>Present (50%)</td>
                <td style={{ color: '#e65100' }}>Variable</td>
              </tr>
              <tr>
                <td><strong>mtDNA depletion</strong></td>
                <td style={{ color: '#c62828' }}>Yes (muscle/brain)</td>
                <td style={{ color: '#2e7d32' }}>No</td>
                <td style={{ color: '#2e7d32' }}>No</td>
              </tr>
              <tr>
                <td><strong>VPA CI</strong></td>
                <td style={{ color: '#c62828' }}>ABSOLUTE</td>
                <td style={{ color: '#f57f17' }}>Relative</td>
                <td style={{ color: '#f57f17' }}>Relative</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="small text-muted mt-2"><strong>Note:</strong> {mma.note}</div>
      </SectionCard>

      {/* Leigh MRI */}
      <SectionCard title="🧠 Leigh-Syndrome MRI — Basal Ganglia Involvement" borderColor="#6a1b9a">
        <Alert variant="info" text="Leigh-like MRI (80%): bilateral symmetric T2/FLAIR hyperintensity in putamen, caudate, dorsal midbrain, periaqueductal grey ± pontine tegmentum." />
        <Alert variant="warning" text="Leigh syndrome is NOT a single disease — it is the MRI/pathological endpoint of >75 metabolic disorders. If MRI shows Leigh + urine MMA elevated (mild) → order SUCLA2 gene panel." />
        <div className="row g-3">
          {[
            { region: 'Putamen', pct: 80, color: '#6a1b9a' },
            { region: 'Caudate', pct: 70, color: '#6a1b9a' },
            { region: 'Dorsal Midbrain', pct: 55, color: '#6a1b9a' },
            { region: 'Periaqueductal Grey', pct: 45, color: '#6a1b9a' },
            { region: 'Pontine Tegmentum', pct: 35, color: '#6a1b9a' },
            { region: 'Brainstem (other)', pct: 25, color: '#6a1b9a' },
          ].map((r, i) => (
            <div key={i} className="col-md-6">
              <Bar label={r.region} value={r.pct} color={r.color} />
            </div>
          ))}
        </div>
      </SectionCard>

      {/* SNHL Management */}
      <SectionCard title="🦻 Sensorineural Hearing Loss — Management Protocol" borderColor="#1565c0">
        <Alert variant="success" text="SNHL 75% — cochlear implants are EFFECTIVE in SUCLA2. Early implantation enables speech and language development. Do NOT delay." />
        <div className="row g-3 small">
          {[
            { step: 'At diagnosis', action: 'ABR (auditory brainstem response) + OAE (otoacoustic emissions)' },
            { step: 'Every 6 months', action: 'Repeat ABR/audiogram — SNHL is progressive' },
            { step: 'SNHL ≥40 dB bilateral', action: 'Cochlear implant evaluation immediately' },
            { step: 'Rapid deterioration', action: 'Expedited implant regardless of dB level' },
            { step: 'Pre-implant anaesthesia', action: 'AVOID propofol (PRIS) — use sevoflurane + ketamine' },
            { step: 'Post-implant', action: 'Speech therapy; auditory verbal therapy; school support' },
          ].map((s, i) => (
            <div key={i} className="col-md-6">
              <div className="p-2 rounded mb-2" style={{ background: i % 2 === 0 ? '#e3f2fd' : LIGHT, border: '1px solid #90caf9' }}>
                <div className="fw-bold" style={{ color: '#1565c0' }}>{s.step}</div>
                <div className="text-muted">{s.action}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Disease Timeline */}
      <SectionCard title="📅 Disease Natural History Timeline">
        {timeline.map((t, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff', border: '1px solid #d7ccc8' }}>
            <div className="fw-bold small mb-1" style={{ color: COLOR }}>{t.phase}</div>
            <div className="small text-muted">{t.events}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const treatments = data.treatments || [];

  return (
    <div>
      <Alert variant="danger" text="⛔ VPA = ABSOLUTE CONTRAINDICATION in SUCLA2 MDDS10. Document allergy-equivalent in ALL records. No safe dose in any mtDNA depletion syndrome." />
      <Alert variant="warning" text="🚫 Ketogenic Diet = CONTRAINDICATED — OXPHOS fails in mtDNA depletion; KD forces fat oxidation → metabolic crisis." />
      <Alert variant="warning" text="🚫 Propofol = AVOID — PRIS risk. Anaesthesia: ketamine + sevoflurane." />
      <Alert variant="success" text="🦻 Cochlear implants effective for SNHL (75%) — begin ABR at diagnosis, implant early at ≥40 dB SNHL." />
      <Alert variant="info" text="🚫 FASTING = FORBIDDEN at all ages. Emergency letter for family. IV dextrose GIR 6-8 during nil-by-mouth." />

      {treatments.map((t, i) => {
        const isAbsCI = t.level?.includes('ABSOLUTE') || t.level?.includes('CONTRAINDICATED') || t.level?.includes('AVOID');
        const isTopA = t.level?.startsWith('A');
        const bg = isAbsCI ? '#ffebee' : isTopA ? LIGHT : '#fff';
        const border = isAbsCI ? '#c62828' : isTopA ? COLOR : '#ddd';
        return (
          <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${border}`, background: bg }}>
            <div className="card-body">
              <div className="d-flex justify-content-between align-items-start mb-1">
                <span className="fw-bold small">{t.tx}</span>
                <span className="badge" style={{ background: isAbsCI ? '#c62828' : COLOR, fontSize: '0.65rem' }}>
                  {t.level?.split('—')[0].trim()}
                </span>
              </div>
              <div className="small text-muted">{t.note}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const terms = data.terms || [];
  return (
    <div>
      {terms.map((d, i) => (
        <div key={i} className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${COLOR}` }}>
          <div className="card-body">
            <h6 className="fw-bold mb-2" style={{ color: COLOR }}>{d.term}</h6>
            <p className="small text-muted mb-0">{d.definition}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function SUCLA2Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);

  const fetchData = async (endpoint, setter, key) => {
    if (loading[key]) return;
    setLoading(l => ({ ...l, [key]: true }));
    try {
      const res = await fetch(`${API}${endpoint}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setter(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(l => ({ ...l, [key]: false }));
    }
  };

  useEffect(() => {
    fetchData('/api/sucla2/overview', setOverview, 'overview');
  }, []);

  useEffect(() => {
    if (activeTab === 1 || activeTab === 2 || activeTab === 3) {
      if (!breakdown) fetchData('/api/sucla2/breakdown', setBreakdown, 'breakdown');
    }
    if (activeTab === 4) {
      if (!definitions) fetchData('/api/sucla2/definitions', setDefinitions, 'definitions');
    }
  }, [activeTab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧠 SUCLA2 Encephalomyopathic mtDNA Depletion Syndrome (MDDS10)
        </h4>
        <div className="text-muted small">
          Mitochondrial DNA Depletion Syndrome 10 (MDDS10) ·
          SUCLA2 Succinyl-CoA Synthetase Beta Subunit · 463 aa · 13q14.2 ·
          OMIM Gene 603921 · Disease 615084 · AR
        </div>
        <div className="mt-1 small fw-semibold" style={{ color: '#c62828' }}>
          ⛔ VPA ABSOLUTE CI · 🚫 KD CONTRAINDICATED · ✅ NO HEPATOPATHY (DDx DGUOK/MPV17/TWNK) ·
          🔬 MILD MMA (DDx MUT/MMACHC) · 🦻 SNHL 75% Cochlear Implants Effective · 🧠 Leigh MRI 80%
        </div>
      </div>

      {error && <div className="alert alert-danger small">Error: {error}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${activeTab === i ? ' active fw-semibold' : ''}`}
              onClick={() => setActiveTab(i)}
              style={activeTab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {activeTab === 0 && <OverviewTab data={overview} />}
      {activeTab === 1 && <PatientsTab data={breakdown} />}
      {activeTab === 2 && <MriSnhlTab data={breakdown} />}
      {activeTab === 3 && <TreatmentsTab data={breakdown} />}
      {activeTab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
