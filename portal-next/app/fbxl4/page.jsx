'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Variants', 'OXPHOS Profile', 'Treatments', 'Definitions'];
const COLOR = '#00695c';   // deep teal — FBXL4/MDDS13 (mitophagy/ubiquitin; no MMA; no C4-DC)
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

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview...</div>;
  const cis = data.key_contraindications || [];
  const features = data.key_features || {};

  return (
    <div>
      {/* Critical VPA Warning Banner */}
      <div className="mb-3 p-3 rounded fw-bold text-center" style={{ background: '#b71c1c', color: 'white', fontSize: '1.05rem' }}>
        ⛔ VPA = ABSOLUTE CONTRAINDICATION IN FBXL4 MDDS13 — mtDNA DEPLETION + CoA SEQUESTRATION (POLG INHIBITION — ALL MDDS RULE)
      </div>
      <div className="mb-3 p-2 rounded fw-semibold text-center" style={{ background: '#e65100', color: 'white', fontSize: '0.95rem' }}>
        🚫 KETOGENIC DIET = CONTRAINDICATED — Forces OXPHOS-Dependent Fat Oxidation That Fails in Pan-OXPHOS Deficiency
      </div>

      {/* Key distinguishing banner */}
      <div className="mb-4 p-2 rounded fw-semibold text-center" style={{ background: '#004d40', color: 'white', fontSize: '0.9rem' }}>
        ✅ MMA NORMAL — KEY DDx from SUCLA2 (mild MMA) + SUCLG1 (severe MMA) ·
        ✅ C4-DC NORMAL — KEY DDx from SCS-axis MDDS ·
        ⚠ MULTI-COMPLEX OXPHOS DEFICIENCY — Complexes I+III+IV+V all depressed · mtDNA muscle &lt;20%
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Hypotonia" value="100%" color={COLOR} />
        <KPI label="Lactic Acidosis" value="100%" color="#c62828" />
        <KPI label="MMA" value="NORMAL" color="#2e7d32" />
        <KPI label="C4-DC" value="NORMAL" color="#2e7d32" />
        <KPI label="mtDNA (muscle)" value="<20%" color="#b71c1c" />
        <KPI label="Leigh MRI" value="~65%" color="#006064" />
      </div>

      {/* No-MMA no-C4DC diagnostic banner */}
      <div className="mb-4 p-3 rounded" style={{ background: '#e0f2f1', border: '2px solid #00695c' }}>
        <div className="fw-bold mb-1" style={{ color: '#00695c' }}>🔬 FBXL4 DDx Key — Normal MMA, Normal C4-DC, Severe OXPHOS</div>
        <div className="small">{data.no_mma_no_c4dc}</div>
      </div>

      {/* Identity */}
      <SectionCard title="🧬 Disease Identity">
        <div className="row g-2 small">
          <div className="col-md-6"><strong>Disease:</strong> {data.disease}</div>
          <div className="col-md-6"><strong>Gene:</strong> {data.gene} — {data.protein} ({data.protein_size_aa} aa precursor / {data.mature_protein_aa} aa mature)</div>
          <div className="col-md-4"><strong>Locus:</strong> {data.locus}</div>
          <div className="col-md-4"><strong>OMIM Gene:</strong> {data.omim_gene} &nbsp; <strong>Disease:</strong> {data.omim_disease}</div>
          <div className="col-md-4"><strong>Inheritance:</strong> {data.inheritance}</div>
          <div className="col-12"><strong>Mechanism:</strong> <span className="text-muted">{data.mechanism}</span></div>
          <div className="col-12"><strong>First described:</strong> {data.first_author} {data.first_publication_year} {data.first_journal}</div>
        </div>
      </SectionCard>

      {/* Key features */}
      <SectionCard title="📊 Key Clinical Features">
        <div className="row g-3">
          {Object.entries(features).map(([key, val], i) => (
            <div key={i} className="col-md-6">
              <div className="p-2 rounded" style={{ background: LIGHT, border: `1px solid ${COLOR}` }}>
                <div className="d-flex justify-content-between small mb-1">
                  <span className="fw-bold" style={{ color: COLOR }}>
                    {key === 'multi_complex_oxphos_deficiency' ? '⚡ Multi-Complex OXPHOS Deficiency' :
                     key === 'leigh_mri' ? '🧠 Leigh MRI' :
                     key === 'mtdna_copy_depletion' ? '🔬 mtDNA Copy Depleted (muscle <20%)' :
                     key === 'lactic_acidosis' ? '⚠ Lactic Acidosis (SEVERE)' :
                     key === 'psychomotor_regression' ? '📉 Psychomotor Regression' :
                     key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  </span>
                  <span className="fw-bold" style={{ color: val.pct === 100 ? '#c62828' : COLOR }}>{val.pct}%</span>
                </div>
                <div className="small text-muted">{val.note}</div>
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
            <div className="text-muted small">{ci.mechanism}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading patients data...</div>;
  const patients = data.patients_sample || [];
  const genotypes = data.genotype_breakdown || [];
  const phenotypes = data.phenotype_distribution || [];
  const fp = data.feature_prevalence || [];

  return (
    <div>
      {/* Phenotype distribution */}
      <SectionCard title="📊 Phenotype Distribution (n=40, seed-563)">
        {phenotypes.map((p, i) => (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span>{p.name}</span><span className="text-muted">{p.n} patients ({p.pct}%)</span>
            </div>
            <div className="progress" style={{ height: 14 }}>
              <div className="progress-bar" style={{ width: `${p.pct}%`, backgroundColor: COLOR }} />
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Feature prevalence */}
      <SectionCard title="📈 Feature Prevalence" borderColor="#006064">
        {fp.map((f, i) => (
          <div key={i} className="mb-3">
            <Bar label={f.feature} value={f.pct} color={COLOR} />
            <div className="small text-muted ms-1">{f.note}</div>
          </div>
        ))}
      </SectionCard>

      {/* Genotype breakdown */}
      <SectionCard title="🧬 Genotype–Phenotype Classes" borderColor="#1565c0">
        {genotypes.map((g, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: LIGHT, border: `1px solid ${COLOR}` }}>
            <div className="d-flex justify-content-between small mb-1">
              <span className="fw-bold">{g.variant_class}</span>
              <span className="badge" style={{ background: COLOR }}>{g.n} ({g.pct}%)</span>
            </div>
            <div className="small text-muted">{g.note}</div>
          </div>
        ))}
      </SectionCard>

      {/* Patient sample table */}
      <SectionCard title="👥 Patient Sample (first 8 of cohort)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr style={{ background: LIGHT }}>
                <th>PID</th><th>Sex</th><th>Ethnicity</th><th>Genotype</th>
                <th>Onset (mo)</th><th>Lactate</th><th>pH</th>
                <th>Leigh MRI</th><th>Seizures</th><th>mtDNA %</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i}>
                  <td>{p.pid}</td>
                  <td>{p.sex}</td>
                  <td>{p.ethnicity}</td>
                  <td><span title={p.variant_description} className="text-truncate d-inline-block" style={{ maxWidth: 140 }}>{p.genotype}</span></td>
                  <td>{p.onset_months}</td>
                  <td className="fw-bold" style={{ color: '#c62828' }}>{p.peak_lactate_mmol}</td>
                  <td className={p.blood_ph_nadir < 7.1 ? 'fw-bold text-danger' : ''}>{p.blood_ph_nadir}</td>
                  <td>{p.leigh_mri ? '✅' : '—'}</td>
                  <td>{p.seizures ? '✅' : '—'}</td>
                  <td className="fw-bold" style={{ color: '#c62828' }}>{p.mtdna_copy_pct_muscle}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="small text-muted">mtDNA % = muscle mtDNA copy number as % of age-matched controls (normal ~100%). MMA and C4-DC are NORMAL in all FBXL4 patients.</div>
      </SectionCard>
    </div>
  );
}

function OXPHOSTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading OXPHOS data...</div>;
  const oxphos = data.oxphos_profile || {};
  const mma = data.mma_c4dc_summary || {};
  const timeline = data.disease_timeline || [];

  const complexes = [
    { key: 'complex_I_NADH_dehydrogenase', label: 'Complex I (NADH Dehydrogenase)', encoded: 'mtDNA (ND1-6/4L)', color: '#c62828' },
    { key: 'complex_II_succinate_dehydrogenase', label: 'Complex II (Succinate Dehydrogenase)', encoded: 'nDNA (all subunits)', color: '#2e7d32' },
    { key: 'complex_III_cytochrome_bc1', label: 'Complex III (Cytochrome bc1)', encoded: 'mtDNA (Cytochrome b)', color: '#c62828' },
    { key: 'complex_IV_cytochrome_c_oxidase', label: 'Complex IV (Cytochrome c Oxidase)', encoded: 'mtDNA (COX I/II/III)', color: '#b71c1c' },
    { key: 'complex_V_ATP_synthase', label: 'Complex V (ATP Synthase)', encoded: 'mtDNA (ATP6/ATP8)', color: '#e65100' },
  ];

  return (
    <div>
      {/* OXPHOS complex table */}
      <SectionCard title="⚡ ETC Enzyme Analysis — Pan-OXPHOS Deficiency (FBXL4 Signature)" borderColor="#c62828">
        <Alert variant="danger" text="⚠ Pan-OXPHOS deficiency (CI+CIII+CIV+CV all reduced) is the biochemical signature of mtDNA template insufficiency. In FBXL4, all mtDNA-encoded complexes are impaired. Complex II (nDNA-encoded only) is relatively preserved — the CII/CS ratio is the reference standard." />
        {complexes.map((c, i) => {
          const d = oxphos[c.key] || {};
          return (
            <div key={i} className="mb-3 p-3 rounded" style={{ background: i === 1 ? '#e8f5e9' : '#ffebee', border: `1px solid ${c.color}` }}>
              <div className="d-flex justify-content-between align-items-start mb-1">
                <span className="fw-bold small">{c.label}</span>
                <span className="badge" style={{ background: c.color, fontSize: '0.65rem' }}>{c.encoded}</span>
              </div>
              <div className="small">
                <strong>Typical % control:</strong> {d.typical_pct_control || d.typical_result || '—'}
              </div>
              <div className="small text-muted">{d.note}</div>
            </div>
          );
        })}
        <div className="mt-2 p-2 rounded small fw-semibold" style={{ background: LIGHT, border: `2px solid ${COLOR}` }}>
          💡 {oxphos.pattern_interpretation}
        </div>
      </SectionCard>

      {/* MMA / C4-DC comparison */}
      <SectionCard title="🔬 MMA & C4-DC — FBXL4 vs SCS-Axis MDDS" borderColor="#1565c0">
        <div className="mb-3 p-3 rounded" style={{ background: '#004d40', color: 'white' }}>
          <div className="fw-bold mb-1">FBXL4 MMA: {mma.fbxl4_mma}</div>
          <div className="fw-bold">FBXL4 C4-DC: {mma.fbxl4_c4dc}</div>
        </div>
        <div className="row g-3 small">
          <div className="col-md-6">
            <div className="p-2 rounded" style={{ background: '#e0f2f1', border: `1px solid ${COLOR}` }}>
              <div className="fw-bold mb-1" style={{ color: COLOR }}>FBXL4 (MDDS13)</div>
              <div><strong>MMA:</strong> {mma.fbxl4_mma}</div>
              <div><strong>C4-DC:</strong> {mma.fbxl4_c4dc}</div>
              <div><strong>Fanconi:</strong> {mma.fbxl4_fanconi}</div>
              <div><strong>Nystagmus:</strong> {mma.fbxl4_nystagmus}</div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-2 rounded" style={{ background: '#fff3e0', border: '1px solid #e65100' }}>
              <div className="fw-bold mb-1" style={{ color: '#e65100' }}>SUCLA2 (MDDS10)</div>
              <div><strong>MMA:</strong> {mma.sucla2_mma}</div>
              <div><strong>C4-DC:</strong> {mma.sucla2_c4dc}</div>
              <div><strong>SNHL:</strong> 75%</div>
              <div><strong>Hepatopathy:</strong> Absent</div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-2 rounded" style={{ background: '#ffebee', border: '1px solid #c62828' }}>
              <div className="fw-bold mb-1" style={{ color: '#c62828' }}>SUCLG1 (MDDS9)</div>
              <div><strong>MMA:</strong> {mma.suclg1_mma}</div>
              <div><strong>C4-DC:</strong> {mma.suclg1_c4dc}</div>
              <div><strong>Hepatopathy:</strong> 70%</div>
              <div><strong>Severity:</strong> More severe than SUCLA2</div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-2 rounded" style={{ background: '#e8eaf6', border: '1px solid #3949ab' }}>
              <div className="fw-bold mb-1" style={{ color: '#3949ab' }}>RRM2B (MDDS8A)</div>
              <div><strong>MMA:</strong> Normal</div>
              <div><strong>C4-DC:</strong> Normal</div>
              <div><strong>Fanconi:</strong> 52% (KEY DDx FBXL4)</div>
              <div><strong>PEO-AD:</strong> Adults (FBXL4 paediatric)</div>
            </div>
          </div>
        </div>
        <div className="mt-3 p-2 rounded small fw-semibold" style={{ background: LIGHT, border: `2px solid ${COLOR}` }}>
          💡 {mma.key_diagnostic_clue}
        </div>
      </SectionCard>

      {/* Disease timeline */}
      <SectionCard title="⏱ Disease Timeline" borderColor="#006064">
        {timeline.map((t, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
            <div className="fw-bold small mb-1" style={{ color: COLOR }}>{t.phase}</div>
            <div className="small text-muted">{t.events}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading treatments data...</div>;
  const txs = data.treatments || [];

  return (
    <div>
      <Alert variant="danger" text="⛔ VPA ABSOLUTE CI — mtDNA depletion + POLG inhibition + CoA sequestration are synergistically lethal. NEVER prescribe valproic acid, valproate, divalproex, or any VPA preparation in FBXL4 disease." />
      <Alert variant="warning" text="🚫 KD CONTRAINDICATED — Pan-OXPHOS deficiency means cells cannot sustain fat oxidation via ETC. High-carbohydrate continuous feeds mandatory." />
      <Alert variant="warning" text="⚠ PROPOFOL AVOID — PRIS risk universally elevated in mitochondrial disease. Complex I+II inhibition by propofol is lethal in pre-existing OXPHOS failure." />
      <Alert variant="warning" text="⚠ FOR INFANTILE SPASMS — Use ACTH first-line, VGB second-line. NEVER VPA even for infantile spasms in FBXL4." />
      {txs.map((tx, i) => (
        <div key={i} className="mb-3 p-3 rounded shadow-sm" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
          <div className="d-flex justify-content-between align-items-start mb-1">
            <span className="fw-bold small">{tx.tx}</span>
            <span className="badge" style={{
              background: tx.level.startsWith('A') ? '#2e7d32' :
                          tx.level.startsWith('B') ? '#1565c0' : '#757575',
              fontSize: '0.65rem'
            }}>{tx.level}</span>
          </div>
          <div className="text-muted small">{tx.note}</div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const terms = data.terms || [];

  return (
    <div>
      {terms.map((t, i) => (
        <div key={i} className="mb-4 p-3 rounded shadow-sm" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
          <div className="fw-bold mb-2" style={{ color: COLOR }}>{t.term}</div>
          <div className="small text-muted">{t.definition}</div>
        </div>
      ))}
    </div>
  );
}

export default function FBXL4Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/fbxl4/overview`).then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/fbxl4/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/fbxl4/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          &#x26d4; FBXL4 Encephalomyopathic mtDNA Depletion Syndrome 13 (MDDS13)
        </h4>
        <div className="small text-muted">
          FBXL4 · 891 aa · 6q16.1-q16.2 · OMIM Gene 605654 · Disease OMIM 615471 · AR · Seed-563 · n=40
        </div>
        <div className="small mt-1 p-2 rounded fw-semibold" style={{ background: '#e0f2f1', color: '#004d40' }}>
          F-box E3 ubiquitin adaptor (mitochondrial matrix) — loss → excessive mitophagy → mtDNA depletion
          (muscle &lt;20%) → pan-OXPHOS deficiency (CI+III+IV+V) — NO MMA · NO C4-DC · NO Fanconi · NO Nystagmus
        </div>
      </div>

      {error && <div className="alert alert-danger small">API error: {error}</div>}

      {/* Tab navigation */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((tab, i) => (
          <li className="nav-item" key={i}>
            <button
              className={`nav-link ${activeTab === i ? 'active fw-bold' : ''}`}
              style={activeTab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setActiveTab(i)}
            >
              {tab}
            </button>
          </li>
        ))}
      </ul>

      {activeTab === 0 && <OverviewTab data={overview} />}
      {activeTab === 1 && <PatientsTab data={breakdown} />}
      {activeTab === 2 && <OXPHOSTab data={breakdown} />}
      {activeTab === 3 && <TreatmentsTab data={breakdown} />}
      {activeTab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
