'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Variants', 'MMA & Hepatopathy', 'Treatments', 'Definitions'];
const COLOR = '#7b1fa2';   // violet/purple — SUCLG1/MDDS9 (shared alpha subunit; severe MMA + hepatopathy)
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
        ⛔ VPA = ABSOLUTE CONTRAINDICATION IN SUCLG1 MDDS9 — mtDNA DEPLETION + CoA SEQUESTRATION + HEPATOPATHY (SYNERGISTICALLY LETHAL)
      </div>
      <div className="mb-3 p-2 rounded fw-semibold text-center" style={{ background: '#e65100', color: 'white', fontSize: '0.95rem' }}>
        🚫 KETOGENIC DIET = CONTRAINDICATED — Forces OXPHOS-Dependent Fat Oxidation That Fails in mtDNA Depletion
      </div>

      {/* Key distinguishing banner */}
      <div className="mb-4 p-2 rounded fw-semibold text-center" style={{ background: '#4a148c', color: 'white', fontSize: '0.9rem' }}>
        ⚠ SEVERE MMA (&gt;500 µmol/mmol creat) — KEY DDx from SUCLA2 (MILD MMA) · HEPATOPATHY 70% — KEY DDx from SUCLA2 (NO hepatopathy) · C4-DC (succinylcarnitine) ELEVATED — SCS axis marker
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Hypotonia" value="100%" color={COLOR} />
        <KPI label="Severe MMA" value="100%" color="#c62828" />
        <KPI label="Lactic Acidosis" value="100%" color="#b71c1c" />
        <KPI label="Hepatopathy" value="~70%" color="#e65100" />
        <KPI label="Leigh-like MRI" value="~60%" color="#6a1b9a" />
        <KPI label="Seizures" value="~65%" color="#1565c0" />
      </div>

      {/* Severity vs SUCLA2 banner */}
      <div className="mb-4 p-3 rounded" style={{ background: '#f3e5f5', border: '2px solid #7b1fa2' }}>
        <div className="fw-bold mb-1" style={{ color: '#7b1fa2' }}>🔬 SUCLG1 vs SUCLA2 — Why SUCLG1 is Typically MORE SEVERE</div>
        <div className="small">{data.severity_vs_sucla2}</div>
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
                    {key === 'methylmalonic_aciduria_severe' ? '🔬 MMA (SEVERE)' :
                     key === 'hepatopathy' ? '🏥 Hepatopathy (⚠ KEY DDx SUCLA2)' :
                     key === 'c4dc_succinylcarnitine' ? '🧪 C4-DC Succinylcarnitine' :
                     key === 'fasting_hypoglycaemia' ? '⚡ Fasting Hypoglycaemia (SCS-G)' :
                     key === 'leigh_mri' ? '🧠 Leigh MRI' :
                     key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  </span>
                  <span className="fw-bold" style={{ color: '#c62828' }}>{val.pct}%</span>
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
      <SectionCard title="📊 Phenotype Distribution (n=40, seed-561)">
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
      <SectionCard title="📈 Feature Prevalence" borderColor="#6a1b9a">
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
                <th>Onset (mo)</th><th>MMA (µmol/mmol)</th><th>Lactate</th>
                <th>Hepatopathy</th><th>Leigh MRI</th><th>SNHL</th>
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
                  <td className="fw-bold" style={{ color: '#c62828' }}>{p.mma_urine_umol_per_mmol_creat}</td>
                  <td>{p.peak_lactate_mmol}</td>
                  <td>{p.hepatopathy ? '✅' : '—'}</td>
                  <td>{p.leigh_mri ? '✅' : '—'}</td>
                  <td>{p.snhl ? '✅' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="small text-muted">MMA values shown as urine organic acids (µmol/mmol creatinine). Values &gt;500 = SEVERE (DDx SUCLA2 mild 10-100).</div>
      </SectionCard>
    </div>
  );
}

function MMAHepatopathyTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading MMA & hepatopathy data...</div>;
  const mma = data.mma_severity_comparison || {};
  const timeline = data.disease_timeline || [];

  return (
    <div>
      {/* MMA severity comparison */}
      <SectionCard title="🔬 MMA Severity Comparison — SUCLG1 vs SUCLA2 vs MUT vs MMACHC" borderColor="#c62828">
        <div className="mb-3 p-3 rounded" style={{ background: '#b71c1c', color: 'white' }}>
          <div className="fw-bold mb-1">SUCLG1 MMA: {mma.suclg1_mma_umol_mmol_creat_typical}</div>
          <div className="fw-bold">SUCLA2 MMA: {mma.sucla2_mma_umol_mmol_creat_typical}</div>
        </div>
        <div className="row g-3 small">
          <div className="col-md-6">
            <div className="p-2 rounded" style={{ background: '#ffebee', border: '1px solid #c62828' }}>
              <div className="fw-bold mb-1" style={{ color: '#c62828' }}>SUCLG1 (MDDS9)</div>
              <div><strong>MMA:</strong> {mma.suclg1_mma_umol_mmol_creat_typical}</div>
              <div><strong>Ketoacidosis:</strong> {mma.suclg1_ketoacidosis_risk}</div>
              <div><strong>Hepatopathy:</strong> {mma.suclg1_hepatopathy}</div>
              <div><strong>Homocysteine:</strong> {mma.suclg1_plasma_hcy}</div>
              <div><strong>C4-DC:</strong> {mma.c4dc_succinylcarnitine}</div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-2 rounded" style={{ background: '#e8f5e9', border: '1px solid #2e7d32' }}>
              <div className="fw-bold mb-1" style={{ color: '#2e7d32' }}>SUCLA2 (MDDS10)</div>
              <div><strong>MMA:</strong> {mma.sucla2_mma_umol_mmol_creat_typical}</div>
              <div><strong>Ketoacidosis:</strong> {mma.sucla2_ketoacidosis_risk}</div>
              <div><strong>Hepatopathy:</strong> {mma.sucla2_hepatopathy}</div>
              <div><strong>C4-DC:</strong> {mma.c4dc_succinylcarnitine}</div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-2 rounded" style={{ background: '#fff8e1', border: '1px solid #f57f17' }}>
              <div className="fw-bold mb-1" style={{ color: '#f57f17' }}>MUT (Methylmalonyl-CoA Mutase)</div>
              <div><strong>MMA:</strong> {mma.mut_mma_umol_mmol_creat_typical}</div>
              <div><strong>C4-DC:</strong> NOT elevated (no SCS defect)</div>
              <div><strong>mtDNA depletion:</strong> NO</div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-2 rounded" style={{ background: '#e8eaf6', border: '1px solid #3949ab' }}>
              <div className="fw-bold mb-1" style={{ color: '#3949ab' }}>MMACHC / cblC</div>
              <div><strong>MMA:</strong> {mma.mmachc_mma_umol_mmol_creat_typical}</div>
              <div><strong>Homocystinuria:</strong> PRESENT (DDx SUCLG1 no Hcy)</div>
              <div><strong>Responds to:</strong> Hydroxocobalamin (SUCLG1 does not)</div>
            </div>
          </div>
        </div>
        <div className="mt-3 p-2 rounded small fw-semibold" style={{ background: '#f3e5f5', border: '2px solid #7b1fa2' }}>
          💡 {mma.note}
        </div>
      </SectionCard>

      {/* PEPCK and hepatopathy */}
      <SectionCard title="🏥 Hepatopathy Mechanism — SCS-G → PEPCK → Gluconeogenesis Failure" borderColor="#e65100">
        <Alert variant="warning" text="⚠ SUCLG1 hepatopathy is unique among MDDS — caused by SCS-G (GTP-forming SCS) loss → hepatic GTP ↓ → PEPCK impaired → gluconeogenesis fails → fasting hypoglycaemia. SUCLA2 does NOT cause hepatopathy because SCS-G (SUCLG1+SUCLG2) is intact." />
        <div className="row g-3 small">
          <div className="col-md-6">
            <div className="fw-bold mb-1" style={{ color: '#e65100' }}>SCS-G Pathway (Liver)</div>
            <div className="p-2 rounded" style={{ background: '#fff3e0' }}>
              SUCLG1 + SUCLG2 = SCS-G (GTP-forming)<br/>
              → Succinyl-CoA + GDP + Pi → Succinate + CoA + GTP<br/>
              → GTP feeds PEPCK<br/>
              → PEPCK: OAA + GTP → PEP + CO2 + GDP<br/>
              → PEP → gluconeogenesis → glucose output
            </div>
          </div>
          <div className="col-md-6">
            <div className="fw-bold mb-1" style={{ color: '#c62828' }}>In SUCLG1 Disease:</div>
            <div className="p-2 rounded" style={{ background: '#ffebee' }}>
              SUCLG1 absent → SCS-G ablated<br/>
              → Hepatic GTP production ↓↓<br/>
              → PEPCK activity ↓ (GTP-dependent)<br/>
              → Gluconeogenesis impaired<br/>
              → FASTING HYPOGLYCAEMIA (55%)<br/>
              → Hepatocellular injury + elevated LFTs (70%)
            </div>
          </div>
        </div>
        <Alert variant="danger" text="CLINICAL ACTION: Never fast a SUCLG1 patient. Continuous enteral feeds (NG/PEG). Emergency IV dextrose GIR 8-10 mg/kg/min for any illness. Monitor blood glucose 2-4 hourly during crisis." />
      </SectionCard>

      {/* Disease timeline */}
      <SectionCard title="⏱ Disease Timeline" borderColor="#6a1b9a">
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
      <Alert variant="danger" text="⛔ VPA ABSOLUTE CI — mtDNA depletion + CoA sequestration + hepatotoxicity are synergistically lethal. NEVER prescribe valproic acid, valproate, divalproex, or any VPA preparation in SUCLG1 disease." />
      <Alert variant="warning" text="🚫 KD CONTRAINDICATED — Ketogenic diet forces OXPHOS-dependent fat oxidation that fails in mtDNA depletion. Avoid in all MDDS." />
      <Alert variant="warning" text="⚠ PROPOFOL AVOID — PRIS risk universally elevated in mitochondrial disease. Use sevoflurane or ketamine (low dose) as alternatives." />
      <Alert variant="warning" text="⚠ NEVER FAST — SCS-G loss impairs gluconeogenesis. Continuous enteral feeds mandatory. Emergency IV dextrose for any illness." />
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

export default function SUCLG1Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/suclg1/overview`).then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/suclg1/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/suclg1/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          &#x26d4; SUCLG1 Encephalomyopathic mtDNA Depletion Syndrome 9 (MDDS9)
        </h4>
        <div className="small text-muted">
          SUCLG1 · 394 aa · 2p11.2 · OMIM Gene 611224 · Disease OMIM 612235 · AR · Seed-561 · n=40
        </div>
        <div className="small mt-1 p-2 rounded fw-semibold" style={{ background: '#f3e5f5', color: '#4a148c' }}>
          SHARED alpha subunit of BOTH SCS-A (with SUCLA2) and SCS-G (with SUCLG2) —
          loss abolishes BOTH isoforms → SEVERE MMA + HEPATOPATHY (vs SUCLA2: mild MMA + no hepatopathy)
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
      {activeTab === 2 && <MMAHepatopathyTab data={breakdown} />}
      {activeTab === 3 && <TreatmentsTab data={breakdown} />}
      {activeTab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
