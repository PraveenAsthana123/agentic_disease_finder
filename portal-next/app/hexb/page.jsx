'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#3e2723';   // dark-warm-brown — HEXB / Sandhoff / GM2 Gangliosidosis Type 2 / Visceral Storage
const ACCENT2 = '#b71c1c';   // dark-red — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep-orange — high-risk warnings / urgent
const ACCENT4 = '#2e7d32';   // deep-green — safe treatments / monitoring
const ACCENT5 = '#4527a0';   // deep-violet — molecular biology / enzyme triad
const ACCENT6 = '#01579b';   // dark-cerulean — gene therapy / visceral management / research

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

function PctBar({ label, pct, color = ACCENT }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

// ── Overview tab ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading overview…</div>;
  const d = data;
  return (
    <div>
      <Alert text="⚠ CBZ / OXC / PHT — ABSOLUTE CI in Sandhoff PME (myoclonic worsening). Fosphenytoin — ABSOLUTE CI; IV LEV replaces in SE." variant="danger" />
      <Alert text="⚠ VGB — HIGH RISK Type 1 Infantile (cherry-red spot 90%; GM2 retinal storage + VGB retinopathy = catastrophic blindness). ACTH preferred for IS." variant="warning" />
      <Alert text="⚠ Sandhoff systemic: Hepatosplenomegaly ~70% Type 1 + bone marrow foam cells. AJ Tay-Sachs carrier screens do NOT detect Sandhoff (different gene, no AJ founder)." variant="info" />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Mean Onset (y)" value={d.mean_onset_years} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="Myoclonus %" value={`${d.myoclonus_pct}%`} color={ACCENT2} />
        <KPI label="Drug Resistant" value={`${d.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Dx Delay (y)" value={d.mean_diagnosis_delay_years} color={ACCENT3} />
        <KPI label="Cherry-Red %" value={`${d.cherry_red_spot_pct}%`} color={ACCENT3} />
        <KPI label="Hepatospleno %" value={`${d.hepatosplenomegaly_pct}%`} color={ACCENT3} />
        <KPI label="Marrow Foam %" value={`${d.bone_marrow_foam_cells_pct}%`} color={ACCENT5} />
        <KPI label="Type 2 Juv %" value={`${d.type2_juvenile_pct}%`} color={ACCENT4} />
        <KPI label="On VPA %" value={`${d.on_vpa_pct}%`} color={ACCENT4} />
        <KPI label="On LEV %" value={`${d.on_lev_pct}%`} color={ACCENT4} />
      </div>

      <SectionCard title="Disease Summary" borderColor={ACCENT}>
        <p className="small mb-0">{d.disease}</p>
      </SectionCard>

      <SectionCard title="Gene & Protein (HEXB — 5q13.3)" borderColor={ACCENT5}>
        <p className="small mb-1"><strong>Gene:</strong> {d.gene}</p>
        <p className="small mb-1"><strong>OMIM:</strong> {d.omim}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-0"><strong>Protein:</strong> {d.protein}</p>
      </SectionCard>

      <SectionCard title="Pathomechanism — Dual CNS + Systemic Accumulation" borderColor={ACCENT5}>
        <p className="small mb-0">{d.mechanism}</p>
      </SectionCard>

      <SectionCard title="⚠ Sandhoff Systemic Involvement (DISTINCTIVE vs Tay-Sachs)" borderColor={ACCENT3}>
        <p className="small mb-0">{d.systemic_involvement_note}</p>
      </SectionCard>

      <SectionCard title="HEXB vs HEXA vs GM2A — Critical Diagnostic Triad" borderColor={ACCENT2}>
        <p className="small mb-0">{d.hexb_hexa_gm2a_differential_note}</p>
      </SectionCard>

      <SectionCard title="⚠ No AJ Founder Mutation — Tay-Sachs Screen Does NOT Detect Sandhoff" borderColor={ACCENT3}>
        <p className="small mb-0">{d.no_aj_founder_note}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Seizure & Therapy Profile" borderColor={ACCENT4}>
            <PctBar label="Infantile Spasms (Type 1)" pct={d.infantile_spasms_pct} color={ACCENT2} />
            <PctBar label="Myoclonus" pct={d.myoclonus_pct} color={ACCENT2} />
            <PctBar label="Dystonia" pct={d.dystonia_pct} color={ACCENT3} />
            <PctBar label="Drug Resistant" pct={d.drug_resistant_pct} color={ACCENT2} />
            <PctBar label="On ACTH (IS Type 1)" pct={d.on_acth_pct} color={ACCENT4} />
            <PctBar label="On Piracetam" pct={d.on_piracetam_pct} color={ACCENT4} />
            <PctBar label="On Clonazepam" pct={d.on_clonazepam_pct} color={ACCENT4} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Visceral & Clinical Features" borderColor={ACCENT5}>
            <PctBar label="Cherry-Red Spot" pct={d.cherry_red_spot_pct} color={ACCENT3} />
            <PctBar label="Hepatosplenomegaly" pct={d.hepatosplenomegaly_pct} color={ACCENT3} />
            <PctBar label="Bone Marrow Foam Cells" pct={d.bone_marrow_foam_cells_pct} color={ACCENT3} />
            <PctBar label="Type 1 Infantile" pct={d.type1_infantile_pct} color={ACCENT2} />
            <PctBar label="Type 2 Juvenile" pct={d.type2_juvenile_pct} color={ACCENT} />
            <PctBar label="Type 3 Adult" pct={d.type3_adult_pct} color={ACCENT4} />
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Discovery History" borderColor={ACCENT6}>
        <p className="small mb-0">{d.discovery}</p>
      </SectionCard>

      <SectionCard title="Unique Features of Sandhoff / HEXB" borderColor={ACCENT}>
        <p className="small mb-0">{d.unique_feature}</p>
      </SectionCard>

      {d.key_pharmacological_distinctions && (
        <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
          {Object.entries(d.key_pharmacological_distinctions).map(([k, v]) => (
            <div key={k} className="mb-2 pb-2 border-bottom">
              <div className="small fw-semibold" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</div>
              <div className="small text-muted">{v}</div>
            </div>
          ))}
        </SectionCard>
      )}
    </div>
  );
}

// ── Patients & Etiology tab ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { etiologies } = data;
  return (
    <div>
      <Alert text="⚠ No AJ founder mutation in Sandhoff — Tay-Sachs AJ carrier screens do NOT detect Sandhoff. HEXB WES + CNV/MLPA (Lebanese exonic deletion) required." variant="info" />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Sandhoff Etiology Classes — 6 Classes (40 Patients)</h6>
      {etiologies?.map((e, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-2">
              <h6 className="fw-bold mb-0" style={{ color: ACCENT }}>{e.class}</h6>
              <span className="badge" style={{ backgroundColor: ACCENT, color: '#fff', fontSize: 13 }}>{e.pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT }} />
            </div>
            <p className="small mb-1">{e.description}</p>
            <div className="row small text-muted">
              <div className="col-md-6"><strong>Typical onset:</strong> {e.typical_onset}</div>
              <div className="col-md-6"><strong>Genotype notes:</strong> {e.genotype_notes}</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Seizures & Triggers tab ───────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types, triggers } = data;
  return (
    <div>
      <Alert text="⚠ CBZ / OXC / PHT ABSOLUTE CI in Sandhoff PME — Type 2 GTCS misidentified as GGE/JME → CBZ → myoclonic storm. VPA + LEV safe backbone." variant="danger" />
      <Alert text="⚠ Fosphenytoin ABSOLUTE CI in SE — replace with IV LEV 60 mg/kg. Hepatomegaly — LFT monitoring if IV VPA used." variant="danger" />

      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Seizure Types</h6>
      {seizure_types?.map((s, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <h6 className="fw-bold mb-0" style={{ color: ACCENT2 }}>{s.type}</h6>
              <span className="badge bg-danger">{s.prevalence_pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 6 }}>
              <div className="progress-bar bg-danger" style={{ width: `${s.prevalence_pct}%` }} />
            </div>
            <div className="small mb-1"><strong>EEG:</strong> {s.eeg_pattern}</div>
            <div className="small mb-1"><strong>Semiology:</strong> {s.semiology}</div>
            <div className="small text-muted"><strong>Tips:</strong> {s.clinical_tips}</div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT3 }}>Seizure Triggers</h6>
      {triggers?.map((t, i) => (
        <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT3}` }}>
          <div className="card-body py-2">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="small fw-bold" style={{ color: ACCENT3 }}>{t.trigger}</span>
              <span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.prevalence_pct}%</span>
            </div>
            <div className="small mb-1"><strong>Mechanism:</strong> {t.mechanism}</div>
            <div className="small text-muted"><strong>Management:</strong> {t.management}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Treatments tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatments, contraindications, lifecycle_stages } = data;
  return (
    <div>
      <Alert text="⚠ CBZ / OXC / PHT / Fosphenytoin — ABSOLUTE CI. VGB — HIGH RISK Type 1 (cherry-red 90%). Hepatomegaly: enhanced LFT monitoring on VPA (3-monthly)." variant="danger" />
      <Alert text="ℹ VPA SAFE in Sandhoff (lysosomal, not mitochondrial). Hepatomegaly (Gb4) does NOT contraindicate VPA — POLG1/MERRF exclusion mandatory." variant="info" />

      <h6 className="fw-bold mb-3" style={{ color: ACCENT4 }}>Treatments (8)</h6>
      {treatments?.map((t, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <h6 className="fw-bold mb-0" style={{ color: ACCENT4 }}>{t.drug}</h6>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.level}</span>
            </div>
            <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>
            <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
            {t.hexb_note && (
              <div className="small p-2 rounded mt-1" style={{ backgroundColor: '#fff8e1', color: '#5d4037' }}>
                <strong>HEXB Note:</strong> {t.hexb_note}
              </div>
            )}
          </div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT2 }}>Contraindications</h6>
      {contraindications?.map((c, i) => (
        <div key={i} className="card mb-2 shadow-sm border-danger">
          <div className="card-body py-2">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="small fw-bold" style={{ color: ACCENT2 }}>{c.drug}</span>
              <span className={`badge ${c.severity === 'ABSOLUTE CI' ? 'bg-danger' : c.severity === 'HIGH RISK' ? 'bg-warning text-dark' : 'bg-secondary'}`}>{c.severity}</span>
            </div>
            <div className="small mb-1">{c.reason}</div>
            <div className="small text-muted"><strong>Note:</strong> {c.note}</div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT6 }}>Disease Lifecycle Stages</h6>
      {lifecycle_stages?.map((s, i) => (
        <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT6}` }}>
          <div className="card-body py-2">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="small fw-bold" style={{ color: ACCENT6 }}>{s.stage}</span>
              <span className="badge" style={{ backgroundColor: ACCENT6 }}>{s.age_range}</span>
            </div>
            <p className="small mb-1">{s.description}</p>
            <ul className="mb-0 ps-3">
              {s.priorities?.map((p, j) => <li key={j} className="small">{p}</li>)}
            </ul>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Definitions tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="Disease & Gene" borderColor={ACCENT}>
        <div className="small mb-1"><strong>Disease:</strong> {data.disease_name}</div>
        <div className="small mb-1"><strong>Gene (full):</strong> {data.gene_full}</div>
        <div className="small mb-1"><strong>OMIM Gene:</strong> {data.omim_gene}</div>
        <div className="small mb-1"><strong>OMIM Disease:</strong> {data.omim_disease}</div>
        <div className="small mb-1"><strong>Inheritance:</strong> {data.inheritance_mode}</div>
        <div className="small mb-0"><strong>Onset age:</strong> {data.onset_age}</div>
      </SectionCard>

      <SectionCard title="Protein Structure" borderColor={ACCENT5}>
        <p className="small mb-0">{data.protein_full}</p>
      </SectionCard>

      <SectionCard title="GM2 Gangliosidosis Diagnostic Triad (HEXB/HEXA/GM2A)" borderColor={ACCENT2}>
        <p className="small mb-0">{data.multienzyme_triad_note}</p>
      </SectionCard>

      {data.sandhoff_vs_taysachs_key_differences && (
        <SectionCard title="Sandhoff vs Tay-Sachs — Key Differences" borderColor={ACCENT3}>
          <div className="table-responsive">
            <table className="table table-sm small mb-0">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th style={{ color: ACCENT }}>Sandhoff (HEXB)</th>
                  <th style={{ color: '#1a237e' }}>Tay-Sachs (HEXA)</th>
                </tr>
              </thead>
              <tbody>
                {data.sandhoff_vs_taysachs_key_differences.map((r, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{r.feature}</td>
                    <td>{r.sandhoff}</td>
                    <td>{r.taysachs}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      <h6 className="fw-bold mb-3 mt-3" style={{ color: ACCENT }}>Key Concepts (15)</h6>
      {data.concepts?.map((c, i) => (
        <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `3px solid ${ACCENT}` }}>
          <div className="card-body py-2">
            <div className="small fw-bold mb-1" style={{ color: ACCENT }}>{c.name}</div>
            <div className="small text-muted">{c.definition}</div>
          </div>
        </div>
      ))}

      <SectionCard title="Thresholds" borderColor={ACCENT3}>
        {data.thresholds?.map((t, i) => (
          <div key={i} className="mb-2 pb-2 border-bottom">
            <div className="small fw-semibold" style={{ color: ACCENT3 }}>{t.parameter}</div>
            <div className="small"><strong>Value:</strong> {t.value}</div>
            <div className="small text-muted"><strong>Action:</strong> {t.action}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Standards" borderColor={ACCENT6}>
        <ol className="mb-0 ps-3">
          {data.standards?.map((s, i) => <li key={i} className="small mb-1">{s}</li>)}
        </ol>
      </SectionCard>

      <SectionCard title="References" borderColor={ACCENT}>
        <ol className="mb-0 ps-3">
          {data.references?.map((r, i) => <li key={i} className="small mb-1">{r}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function HEXBPage() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hexb/overview`).then(r => r.json()),
      fetch(`${API}/api/hexb/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hexb/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error) return <div className="container py-4"><div className="alert alert-danger">Error: {error}</div></div>;

  return (
    <div className="container-fluid py-3">
      <div className="mb-3 p-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #5d4037 100%)` }}>
        <h4 className="mb-1 fw-bold">&#x1f7eb; HEXB Epilepsy — Sandhoff Disease</h4>
        <div className="small opacity-90">
          GM2 Gangliosidosis Type 2 · HEXB (5q13.3) · β-Hexosaminidase B β-subunit Deficiency ·
          ALL Hex Forms Deficient (Hex A + Hex B + Hex S Low = PATHOGNOMONIC) ·
          Hepatosplenomegaly + Bone Marrow Foam Cells (DISTINCTIVE vs Tay-Sachs) ·
          No AJ Founder (Tay-Sachs Screen Does NOT Detect Sandhoff) ·
          CBZ/OXC/PHT ABSOLUTE CI · VPA SAFE (Enhanced LFT Monitoring Hepatomegaly) ·
          ACTH Level A IS · VGB HIGH RISK Type 1 · IV LEV Replaces Fosphenytoin ·
          AAV9-HEXA/HEXB Bicistronic Gene Therapy Phase I/II 2024 · AR Biallelic LOF · OMIM #268800
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${activeTab === i ? 'active fw-bold' : ''}`}
              style={activeTab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setActiveTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      <div>
        {activeTab === 0 && <OverviewTab data={overview} />}
        {activeTab === 1 && <PatientsTab data={breakdown} />}
        {activeTab === 2 && <SeizuresTab data={breakdown} />}
        {activeTab === 3 && <TreatmentsTab data={breakdown} />}
        {activeTab === 4 && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
