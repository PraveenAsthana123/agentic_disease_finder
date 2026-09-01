'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Variants', 'MRI & Renal', 'Treatments', 'Definitions'];
const COLOR = '#37474f';   // blue-grey 800 — RRM2B/MDDS8A (dNTP/replication integrity)
const LIGHT = '#eceff1';

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
        ⛔ VPA = ABSOLUTE CONTRAINDICATION IN RRM2B MDDS8A — mtDNA DEPLETION (POLG INHIBITION + CoA SEQUESTRATION + HEPATOTOXIC EPOXIDE)
      </div>
      <div className="mb-3 p-2 rounded fw-semibold text-center" style={{ background: '#e65100', color: 'white', fontSize: '0.95rem' }}>
        🚫 KETOGENIC DIET = CONTRAINDICATED — Forces OXPHOS-Dependent Fat Oxidation That Fails in mtDNA Depletion
      </div>

      {/* Key distinguishing banner */}
      <div className="mb-4 p-2 rounded fw-semibold text-center" style={{ background: '#1b5e20', color: 'white', fontSize: '0.9rem' }}>
        ✅ NO HEPATOPATHY — Liver NORMAL (DDx DGUOK/MPV17/TWNK/POLG) · NO MMA (DDx SUCLA2) · FANCONI SYNDROME ~50% DISTINCTIVE DDx · CK MILD (DDx TK2)
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Hypotonia" value="100%" color={COLOR} />
        <KPI label="Lactic Acidosis" value="~90%" color="#c62828" />
        <KPI label="Respiratory Failure" value="~65%" color="#c62828" />
        <KPI label="Fanconi Syndrome" value="~52%" color="#1565c0" />
        <KPI label="CK Mild" value="~60%" color={COLOR} />
        <KPI label="SNHL" value="~35%" color="#6a1b9a" />
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
          {data.iberian_founder && (
            <div className="col-12"><strong>Iberian Founder:</strong> <span className="text-muted">{data.iberian_founder}</span></div>
          )}
        </div>
      </SectionCard>

      {/* Fanconi Syndrome Highlight — DISTINCTIVE DDx feature */}
      <div className="mb-4 p-3 rounded" style={{ background: '#e3f2fd', border: '2px solid #1565c0' }}>
        <div className="fw-bold mb-1" style={{ color: '#1565c0' }}>🔬 Fanconi Syndrome — DISTINCTIVE DDx Feature (~50%)</div>
        <div className="small">Proximal renal tubular dysfunction in ~50% of RRM2B MDDS8A patients — NOT seen in TK2, SUCLA2, or other encephalomyopathic MDDS. Presents as phosphaturia, glucosuria (normoglycemia), aminoaciduria, and low-molecular-weight proteinuria. Renal tubular acidosis may complicate metabolic management. Mandates urine amino acid panel and urine glucose at diagnosis and annually.</div>
        <div className="small mt-1 text-muted">Fanconi syndrome in a child with Leigh-like MRI and mtDNA depletion — order RRM2B panel first. Absence of MMA excludes SUCLA2. Absence of elevated CK (or only mild) excludes TK2.</div>
      </div>

      {/* Pathognomonic DDx */}
      <SectionCard title="🔑 Key Diagnostic Pearls — RRM2B DDx" borderColor="#1565c0">
        <div className="row g-3">
          {Object.entries(ddx).map(([key, val], i) => (
            <div key={i} className="col-md-6">
              <div className="p-3 rounded h-100" style={{ background: '#e8f5e9', border: '2px solid #2e7d32' }}>
                <div className="fw-bold small mb-1" style={{ color: '#2e7d32' }}>
                  {key === 'fanconi_syndrome' ? '🔬 Fanconi Syndrome ~50% (DDx TK2/SUCLA2 absent)' :
                   key === 'no_hepatopathy' ? '🏥 No Hepatopathy (DDx DGUOK/MPV17/TWNK/POLG)' :
                   key === 'no_methylmalonic_aciduria' ? '🧪 No MMA (DDx SUCLA2 mild MMA)' :
                   key === 'ck_mild_not_tk2' ? '💪 CK Mild Only (DDx TK2 high CK 90%)' :
                   key.replace(/_/g, ' ')}
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
      <SectionCard title="👥 Cohort Summary (n=40)">
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
      <SectionCard title="👥 Cohort Phenotype Distribution (n=40)">
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
          <div key={i} className="mb-3 p-3 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff', border: '1px solid #cfd8dc' }}>
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
                <th>Diag (mo)</th><th>Fanconi</th><th>Resp. Failure</th>
                <th>SNHL</th><th>Leigh MRI</th><th>Lactate (mmol/L)</th>
              </tr></thead>
              <tbody>
                {data.patients_sample.map((p, i) => (
                  <tr key={i}>
                    <td>{p.pid}</td>
                    <td>{p.sex}</td>
                    <td>{p.ethnicity}</td>
                    <td>{p.onset_months}</td>
                    <td>{p.diagnosis_months}</td>
                    <td style={{ color: p.fanconi ? '#1565c0' : '#2e7d32' }}>{p.fanconi ? 'Yes' : 'No'}</td>
                    <td style={{ color: p.respiratory_failure ? '#c62828' : '#2e7d32' }}>{p.respiratory_failure ? 'Yes' : 'No'}</td>
                    <td style={{ color: p.snhl ? '#6a1b9a' : '#2e7d32' }}>{p.snhl ? 'Yes' : 'No'}</td>
                    <td style={{ color: p.leigh_mri ? '#6a1b9a' : '#2e7d32' }}>{p.leigh_mri ? 'Yes' : 'No'}</td>
                    <td style={{ color: '#c62828' }}>{p.peak_lactate_mmol}</td>
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

function MriRenalTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const mma = data.mma_ddx_table || {};
  const timeline = data.disease_timeline || [];
  const fanconi = data.fanconi_features || [];

  return (
    <div>
      {/* Fanconi Syndrome Section */}
      <SectionCard title="🔬 Fanconi Syndrome — Proximal Renal Tubular Dysfunction (~50%)" borderColor="#1565c0">
        <Alert variant="info" text="DISTINCTIVE DDx: Fanconi syndrome occurs in ~50% of RRM2B MDDS8A — NOT seen in TK2, SUCLA2, DGUOK, MPV17, or TWNK. Tubular dysfunction from mitochondrial energy failure in proximal tubule cells." />
        <Alert variant="warning" text="⚠️ Fanconi + Leigh MRI + no MMA + no hepatopathy + mild CK → ORDER RRM2B PANEL IMMEDIATELY" />
        <div className="row g-3 mt-1">
          {fanconi.map((f, i) => (
            <div key={i} className="col-md-6">
              <div className="p-3 rounded h-100" style={{ background: i % 2 === 0 ? '#e3f2fd' : LIGHT, border: '1px solid #90caf9' }}>
                <div className="fw-bold small mb-1" style={{ color: '#1565c0' }}>{f.feature}</div>
                <div className="small text-muted">{f.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* MMA DDx table */}
      <SectionCard title="🔬 Metabolic DDx Table — RRM2B vs SUCLA2 vs TK2" borderColor="#e65100">
        <Alert variant="success" text="✅ KEY: RRM2B has NO MMA (DDx SUCLA2 mild MMA), NO hepatopathy (DDx DGUOK/MPV17/TWNK), and only mild CK elevation (DDx TK2 high CK). Fanconi syndrome is the DISTINCTIVE RRM2B marker." />
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr style={{ background: '#fff3e0' }}>
              <th>Parameter</th>
              <th>{mma.col_rrm2b || 'RRM2B MDDS8A'}</th>
              <th>{mma.col_sucla2 || 'SUCLA2 MDDS10'}</th>
              <th>{mma.col_tk2 || 'TK2 MDDS4A'}</th>
            </tr></thead>
            <tbody>
              {(mma.rows || []).map((row, i) => (
                <tr key={i}>
                  <td><strong>{row.parameter}</strong></td>
                  <td style={{ color: row.rrm2b_color || '#37474f' }}>{row.rrm2b}</td>
                  <td style={{ color: row.sucla2_color || '#6d4c41' }}>{row.sucla2}</td>
                  <td style={{ color: row.tk2_color || '#4e342e' }}>{row.tk2}</td>
                </tr>
              ))}
              {/* Fallback static rows if backend rows not provided */}
              {(!mma.rows || mma.rows.length === 0) && (
                <>
                  <tr>
                    <td><strong>Urine MMA</strong></td>
                    <td style={{ color: '#2e7d32' }}>ABSENT / Normal</td>
                    <td style={{ color: '#e65100' }}>Mild elevated (50–500)</td>
                    <td style={{ color: '#2e7d32' }}>ABSENT / Normal</td>
                  </tr>
                  <tr>
                    <td><strong>Fanconi Syndrome</strong></td>
                    <td style={{ color: '#1565c0' }}>~50% PRESENT</td>
                    <td style={{ color: '#2e7d32' }}>ABSENT</td>
                    <td style={{ color: '#2e7d32' }}>ABSENT</td>
                  </tr>
                  <tr>
                    <td><strong>CK elevation</strong></td>
                    <td style={{ color: '#e65100' }}>Mild (~60%)</td>
                    <td style={{ color: '#2e7d32' }}>Normal/Mild</td>
                    <td style={{ color: '#c62828' }}>HIGH 90% (myopathic)</td>
                  </tr>
                  <tr>
                    <td><strong>Hepatopathy</strong></td>
                    <td style={{ color: '#2e7d32' }}>ABSENT</td>
                    <td style={{ color: '#2e7d32' }}>ABSENT</td>
                    <td style={{ color: '#2e7d32' }}>ABSENT</td>
                  </tr>
                  <tr>
                    <td><strong>Respiratory failure</strong></td>
                    <td style={{ color: '#e65100' }}>~65%</td>
                    <td style={{ color: '#e65100' }}>~60%</td>
                    <td style={{ color: '#c62828' }}>~85% (leading cause of death)</td>
                  </tr>
                  <tr>
                    <td><strong>mtDNA depletion</strong></td>
                    <td style={{ color: '#c62828' }}>Yes (muscle/brain)</td>
                    <td style={{ color: '#c62828' }}>Yes (muscle/brain)</td>
                    <td style={{ color: '#c62828' }}>Yes (muscle)</td>
                  </tr>
                  <tr>
                    <td><strong>VPA CI</strong></td>
                    <td style={{ color: '#c62828' }}>ABSOLUTE</td>
                    <td style={{ color: '#c62828' }}>ABSOLUTE</td>
                    <td style={{ color: '#c62828' }}>ABSOLUTE</td>
                  </tr>
                </>
              )}
            </tbody>
          </table>
        </div>
        {mma.note && <div className="small text-muted mt-2"><strong>Note:</strong> {mma.note}</div>}
      </SectionCard>

      {/* Leigh MRI */}
      <SectionCard title="🧠 Leigh-Syndrome MRI — Basal Ganglia Involvement" borderColor="#6a1b9a">
        <Alert variant="info" text="Leigh-like MRI: bilateral symmetric T2/FLAIR hyperintensity in putamen, caudate, dorsal midbrain, periaqueductal grey ± pontine tegmentum. Present in the majority of RRM2B encephalomyopathic cases." />
        <Alert variant="warning" text="Leigh syndrome is NOT a single disease — it is the MRI/pathological endpoint of >75 metabolic disorders. If MRI shows Leigh + Fanconi + no MMA + no hepatopathy → order RRM2B gene panel." />
        <div className="row g-3">
          {[
            { region: 'Putamen', pct: 75, color: '#6a1b9a' },
            { region: 'Caudate', pct: 65, color: '#6a1b9a' },
            { region: 'Dorsal Midbrain', pct: 50, color: '#6a1b9a' },
            { region: 'Periaqueductal Grey', pct: 40, color: '#6a1b9a' },
            { region: 'Pontine Tegmentum', pct: 30, color: '#6a1b9a' },
            { region: 'Brainstem (other)', pct: 20, color: '#6a1b9a' },
          ].map((r, i) => (
            <div key={i} className="col-md-6">
              <Bar label={r.region} value={r.pct} color={r.color} />
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Disease Timeline */}
      <SectionCard title="📅 Disease Natural History Timeline">
        {timeline.map((t, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff', border: '1px solid #cfd8dc' }}>
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
      <Alert variant="danger" text="⛔ VPA = ABSOLUTE CONTRAINDICATION in RRM2B MDDS8A. Document allergy-equivalent in ALL records. No safe dose in any mtDNA depletion syndrome." />
      <Alert variant="warning" text="🚫 Ketogenic Diet = CONTRAINDICATED — OXPHOS fails in mtDNA depletion; KD forces fat oxidation → metabolic crisis." />
      <Alert variant="warning" text="🚫 Propofol = AVOID — PRIS risk. Anaesthesia: ketamine + sevoflurane." />
      <Alert variant="info" text="🔬 Fanconi Syndrome Management — monitor urine phosphate, amino acids, glucose; replace phosphate; watch for renal tubular acidosis." />
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

export default function RRM2BPage() {
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
    fetchData('/api/rrm2b/overview', setOverview, 'overview');
  }, []);

  useEffect(() => {
    if (activeTab === 1 || activeTab === 2 || activeTab === 3) {
      if (!breakdown) fetchData('/api/rrm2b/breakdown', setBreakdown, 'breakdown');
    }
    if (activeTab === 4) {
      if (!definitions) fetchData('/api/rrm2b/definitions', setDefinitions, 'definitions');
    }
  }, [activeTab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧠 RRM2B Encephalomyopathic mtDNA Depletion Syndrome (MDDS8A)
        </h4>
        <div className="text-muted small">
          Mitochondrial DNA Depletion Syndrome 8A (MDDS8A) ·
          RRM2B Ribonucleoside-Diphosphate Reductase Subunit M2 B · 351 aa · 8q22.3 ·
          OMIM Gene 604712 · Disease 612075 · AR
        </div>
        <div className="mt-1 small fw-semibold" style={{ color: '#c62828' }}>
          ⛔ VPA ABSOLUTE CI · 🚫 KD CONTRAINDICATED · ✅ NO HEPATOPATHY (DDx DGUOK/MPV17/TWNK) ·
          🔬 NO MMA (DDx SUCLA2) · 🏥 FANCONI ~50% DISTINCTIVE DDx · 💪 CK MILD (DDx TK2)
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
      {activeTab === 2 && <MriRenalTab data={breakdown} />}
      {activeTab === 3 && <TreatmentsTab data={breakdown} />}
      {activeTab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
