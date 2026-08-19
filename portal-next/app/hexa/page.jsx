'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep-midnight-blue — HEXA / Tay-Sachs / GM2 Gangliosidosis
const ACCENT2 = '#b71c1c';   // dark-red — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep-orange — high-risk warnings / urgent
const ACCENT4 = '#2e7d32';   // deep-green — safe treatments / monitoring
const ACCENT5 = '#4a0080';   // deep-violet — molecular biology / enzyme triad
const ACCENT6 = '#0277bd';   // dark-cerulean — gene therapy / adult type 3 / research

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
    <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header fw-bold" style={{ backgroundColor: '#e8eaf6', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

export default function HEXAPage() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/hexa/overview`).then(r => r.json()).then(setOv).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 && !bk) fetch(`${API}/api/hexa/breakdown`).then(r => r.json()).then(setBk).catch(e => setErr(String(e)));
    if (tab === 4 && !df) fetch(`${API}/api/hexa/definitions`).then(r => r.json()).then(setDf).catch(e => setErr(String(e)));
    if ((tab === 2 || tab === 3) && !bk) fetch(`${API}/api/hexa/breakdown`).then(r => r.json()).then(setBk).catch(e => setErr(String(e)));
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div className="rounded p-3 mb-3 text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #000051 100%)` }}>
        <h4 className="mb-1 fw-bold">🔵 HEXA Epilepsy — Tay-Sachs Disease (GM2 Gangliosidosis Type 1)</h4>
        <div style={{ fontSize: 13 }}>
          <strong>HEXA (15q23)</strong> — β-Hexosaminidase A α-subunit · GM2 Ganglioside Hydrolysis · AR Biallelic LOF
          <span className="ms-3 badge bg-warning text-dark">Type 2 Juvenile Onset 2–10y (mean 5y)</span>
          <span className="ms-2 badge" style={{ backgroundColor: ACCENT6 }}>🧬 AAV9-HEXA/HEXB Bicistronic Gene Therapy Phase I/II 2024</span>
          <span className="ms-2 badge bg-danger">CBZ/OXC/PHT ABSOLUTE CI — PME Myoclonus Trap</span>
        </div>
      </div>

      {/* ENZYME TRIAD ALERT */}
      <div className="alert py-2 mb-2" style={{ backgroundColor: '#ede7f6', borderLeft: `5px solid ${ACCENT5}`, fontSize: 13 }}>
        <strong>🔬 GM2 GANGLIOSIDOSIS TRIAD — HEXA vs HEXB vs GM2A (AB variant):</strong>{' '}
        <strong>Tay-Sachs (HEXA LOF):</strong> Hex A LOW + Hex B <u>ELEVATED</u> (ββ retained — PATHOGNOMONIC).{' '}
        <strong>Sandhoff (HEXB LOF):</strong> Hex A LOW + Hex B LOW + Hex S LOW (all deficient; more systemic).{' '}
        <strong>AB Variant (GM2A LOF):</strong> Hex A NORMAL in vitro; GM2AP protein deficient → confirm with GM2AP assay.{' '}
        <strong>RULE: Measure Hex A AND Hex B simultaneously in LEUKOCYTES — not DBS alone</strong>{' '}
        (pseudodeficiency p.Arg247Trp + OCP interference invalidate DBS).
      </div>

      {/* PSEUDODEFICIENCY ALERT */}
      <div className="alert py-2 mb-2" style={{ backgroundColor: '#e8f5e9', borderLeft: `5px solid ${ACCENT4}`, fontSize: 13 }}>
        <strong>⚠️ PSEUDODEFICIENCY TRAP — CONFIRM ALL DBS LOW HEX A WITH LEUKOCYTE ASSAY:</strong>{' '}
        p.Arg247Trp and p.Arg249Trp → DBS Hex A appears LOW but Leukocyte Hex A NORMAL (not disease alleles).{' '}
        5% of apparent 'low Hex A' on DBS = pseudodeficiency.{' '}
        Females on OCP → serum Hex A falsely low — use leukocyte assay.{' '}
        <strong>NEVER counsel genetically based on DBS alone.</strong>{' '}
        Fundoscopy before infantile spasm treatment: cherry-red present → <strong>ACTH</strong> (not VGB).
      </div>

      {/* CBZ ABSOLUTE CI ALERT */}
      <div className="alert alert-danger py-2 mb-3" style={{ fontSize: 13 }}>
        <strong>🚫 CBZ/OXC/PHT ABSOLUTE CI — FOSPHENYTOIN ABSOLUTE CI — TGB/GBP/PGB HIGH RISK IN HEXA/TAY-SACHS:</strong>{' '}
        Type 2 juvenile GTCS misidentified as GGE/JME → CBZ → acute myoclonic storm (mean Dx delay 2.9y).{' '}
        Fosphenytoin ABSOLUTE CI in SE → replace with <strong>IV LEV 60 mg/kg</strong>.{' '}
        <strong>Safe backbone: ACTH (IS) + VPA + LEV + Piracetam (action myoclonus).</strong>{' '}
        <strong>POLG1/MERRF EXCLUSION MANDATORY before VPA</strong> (POLG1 Alpers mimics Type 2 PME — fatal VPA hepatotoxicity).
      </div>

      {err && <Alert text={`API error: ${err}`} variant="danger" />}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button className={`nav-link${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && ov && (
        <div>
          <div className="row mb-3">
            <KPI label="Cohort" value={`${ov.cohort_size}pts`} color={ACCENT} />
            <KPI label="Mean Onset" value={`${ov.mean_onset_years}y`} color={ACCENT3} />
            <KPI label="Seizures" value={`${ov.seizure_pct}%`} color={ACCENT2} />
            <KPI label="Action Myoclonus" value={`${ov.myoclonus_pct}%`} color={ACCENT2} />
            <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
            <KPI label="Dx Delay" value={`${ov.mean_diagnosis_delay_years}y`} color={ACCENT3} />
          </div>

          <SectionCard title="🧬 Gene, Protein & Mechanism" borderColor={ACCENT}>
            <div className="mb-2"><strong>Gene:</strong> <code>{ov.gene}</code></div>
            <div className="mb-2"><strong>Protein:</strong> {ov.protein}</div>
            <div className="mb-2"><strong>Inheritance:</strong> {ov.inheritance}</div>
            <div className="mb-2"><strong>OMIM:</strong> {ov.omim}</div>
            <div className="mb-2"><strong>Mechanism:</strong> {ov.mechanism}</div>
            <div className="mb-2"><strong>Disease Summary:</strong> {ov.disease}</div>
          </SectionCard>

          <SectionCard title="🔬 GM2 Triad — HEXA vs HEXB vs GM2A (AB variant): Hex B Elevated = Tay-Sachs PATHOGNOMONIC" borderColor={ACCENT5}>
            <p style={{ fontSize: 13 }}>{ov.hexa_hexb_gm2a_differential_note}</p>
          </SectionCard>

          <SectionCard title="⚠️ Pseudodeficiency Trap — DBS vs Leukocyte Assay" borderColor={ACCENT4}>
            <p style={{ fontSize: 13 }}>{ov.pseudodeficiency_trap_note}</p>
          </SectionCard>

          <SectionCard title="📊 Type 2 Juvenile PME vs Type 3 Adult — Epilepsy Treatment Pivot" borderColor={ACCENT}>
            <p style={{ fontSize: 13 }}>{ov.type2_type3_differential_note}</p>
          </SectionCard>

          <SectionCard title="💊 Key Pharmacological Distinctions" borderColor={ACCENT2}>
            {ov.key_pharmacological_distinctions && Object.entries(ov.key_pharmacological_distinctions).map(([k, v]) => (
              <div key={k} className="mb-3 p-2 rounded" style={{ backgroundColor: '#fafafa', border: '1px solid #ddd' }}>
                <div className="fw-bold small mb-1" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ fontSize: 13 }}>{v}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="📅 Discovery, Carrier Screening & Gene Therapy" borderColor={ACCENT}>
            <p style={{ fontSize: 13 }}>{ov.discovery}</p>
            <p style={{ fontSize: 13 }}><strong>Unique Feature:</strong> {ov.unique_feature}</p>
          </SectionCard>

          <div className="row">
            <div className="col-md-6">
              <SectionCard title="📊 Neurological Profile" borderColor={ACCENT4}>
                <PctBar label="Seizures (any type)" pct={ov.seizure_pct} color={ACCENT2} />
                <PctBar label="Action Myoclonus (Type 2)" pct={ov.myoclonus_pct} color={ACCENT2} />
                <PctBar label="Infantile Spasms (Type 1)" pct={ov.infantile_spasms_pct} color={ACCENT3} />
                <PctBar label="Dystonia" pct={ov.dystonia_pct} color={ACCENT5} />
                <PctBar label="Drug-Resistant Epilepsy" pct={ov.drug_resistant_pct} color={ACCENT2} />
                <PctBar label="Cherry-Red Spot (all types)" pct={ov.cherry_red_spot_pct} color={ACCENT3} />
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="📊 Population & Treatment Profile" borderColor={ACCENT4}>
                <PctBar label="Type 2 Juvenile (primary PME)" pct={ov.type2_juvenile_pct} color={ACCENT} />
                <PctBar label="Ashkenazi Jewish Founder" pct={ov.ashkenazi_jewish_pct} color={ACCENT5} />
                <PctBar label="French-Canadian IVS7+1G>A" pct={ov.french_canadian_pct} color={ACCENT5} />
                <PctBar label="On VPA Backbone" pct={ov.on_vpa_pct} color={ACCENT4} />
                <PctBar label="On LEV" pct={ov.on_lev_pct} color={ACCENT4} />
                <PctBar label="ACTH (Infantile Spasms)" pct={ov.on_acth_pct} color={ACCENT3} />
                <PctBar label="On Piracetam (Action Myoclonus)" pct={ov.on_piracetam_pct} color={ACCENT6} />
                <PctBar label="On Clonazepam (Nocturnal)" pct={ov.on_clonazepam_pct} color={ACCENT6} />
              </SectionCard>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 1: PATIENTS & ETIOLOGY ── */}
      {tab === 1 && bk && (
        <div>
          <SectionCard title="🧬 Etiological Classes (n=40)" borderColor={ACCENT}>
            {bk.etiologies?.map((e, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: '1px solid #ddd' }}>
                <div className="d-flex justify-content-between align-items-center mb-1">
                  <strong style={{ color: ACCENT }}>{e.class}</strong>
                  <span className="badge" style={{ backgroundColor: ACCENT }}>{e.pct}%</span>
                </div>
                <div className="progress mb-1" style={{ height: 6 }}>
                  <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT }} />
                </div>
                <div style={{ fontSize: 12 }} className="text-muted">{e.description}</div>
                <div style={{ fontSize: 12 }}>
                  <strong>Typical onset:</strong> {e.typical_onset} |{' '}
                  <strong>Genotype note:</strong> {e.genotype_notes}
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
      )}

      {/* ── TAB 2: SEIZURES & TRIGGERS ── */}
      {tab === 2 && bk && (
        <div>
          <SectionCard title="⚡ Seizure Types" borderColor={ACCENT2}>
            {bk.seizure_types?.map((s, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: '1px solid #ddd' }}>
                <div className="d-flex justify-content-between mb-1">
                  <strong style={{ color: ACCENT2 }}>{s.type}</strong>
                  <span className="badge bg-danger">{s.prevalence_pct}%</span>
                </div>
                <div className="progress mb-1" style={{ height: 6 }}>
                  <div className="progress-bar bg-danger" style={{ width: `${s.prevalence_pct}%` }} />
                </div>
                <div style={{ fontSize: 12 }} className="mb-1"><strong>EEG:</strong> {s.eeg_pattern}</div>
                <div style={{ fontSize: 12 }} className="mb-1"><strong>Semiology:</strong> {s.semiology}</div>
                <div style={{ fontSize: 12, color: ACCENT3 }}><strong>Clinical tip:</strong> {s.clinical_tips}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="🌡️ Seizure Triggers" borderColor={ACCENT3}>
            {bk.triggers?.map((t, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ border: '1px solid #ffd54f', backgroundColor: '#fffde7' }}>
                <div className="d-flex justify-content-between mb-1">
                  <strong style={{ fontSize: 13 }}>{t.trigger}</strong>
                  <span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.prevalence_pct}%</span>
                </div>
                <div style={{ fontSize: 12 }} className="mb-1 text-muted">{t.mechanism}</div>
                <div style={{ fontSize: 12, color: ACCENT4 }}><strong>Management:</strong> {t.management}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      )}

      {/* ── TAB 3: TREATMENTS ── */}
      {tab === 3 && bk && (
        <div>
          <SectionCard title="💊 Treatments" borderColor={ACCENT4}>
            {bk.treatments?.map((t, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: '1px solid #c8e6c9' }}>
                <div className="d-flex justify-content-between mb-1">
                  <strong style={{ color: ACCENT4 }}>{t.drug}</strong>
                  <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.level}</span>
                </div>
                <div style={{ fontSize: 12 }} className="mb-1"><strong>Dose:</strong> {t.dose}</div>
                <div style={{ fontSize: 12 }} className="mb-1"><strong>MOA:</strong> {t.moa}</div>
                <div style={{ fontSize: 12 }} className="mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
                <div style={{ fontSize: 12 }} className="mb-1 text-muted"><strong>Monitoring:</strong> {t.monitoring}</div>
                <div style={{ fontSize: 12, color: ACCENT }}><strong>HEXA note:</strong> {t.hexa_note}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="🚫 Contraindications" borderColor={ACCENT2}>
            {bk.contraindications?.map((ci, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{
                border: `2px solid ${ci.severity === 'ABSOLUTE CI' ? ACCENT2 : ACCENT3}`,
                backgroundColor: ci.severity === 'ABSOLUTE CI' ? '#ffebee' : '#fff8e1'
              }}>
                <div className="d-flex justify-content-between mb-1">
                  <strong style={{ color: ci.severity === 'ABSOLUTE CI' ? ACCENT2 : ACCENT3 }}>{ci.drug}</strong>
                  <span className={`badge ${ci.severity === 'ABSOLUTE CI' ? 'bg-danger' : 'bg-warning text-dark'}`}>{ci.severity}</span>
                </div>
                <div style={{ fontSize: 12 }} className="mb-1">{ci.reason}</div>
                <div style={{ fontSize: 12, color: '#555' }}><em>{ci.note}</em></div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="🔬 Monitoring" borderColor={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead style={{ backgroundColor: ACCENT, color: 'white' }}>
                  <tr><th>Item</th><th>Frequency</th><th>Note</th></tr>
                </thead>
                <tbody>
                  {bk.monitoring?.map((m, i) => (
                    <tr key={i}>
                      <td><strong>{m.item}</strong></td>
                      <td>{m.frequency}</td>
                      <td>{m.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </div>
      )}

      {/* ── TAB 4: DEFINITIONS ── */}
      {tab === 4 && df && (
        <div>
          <SectionCard title="📖 Disease Definition" borderColor={ACCENT}>
            <div className="row g-2 mb-3" style={{ fontSize: 13 }}>
              {[
                ['Disease', df.disease_name],
                ['Gene', df.gene_full],
                ['OMIM Gene', df.omim_gene],
                ['OMIM Disease', df.omim_disease],
                ['Protein', df.protein_full],
                ['Inheritance', df.inheritance_mode],
                ['Onset Age', df.onset_age],
              ].map(([label, val]) => (
                <div key={label} className="col-12">
                  <span className="fw-bold" style={{ color: ACCENT }}>{label}: </span>{val}
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="🔬 GM2 Triad — HEXA / HEXB / GM2A Enzyme Biochemistry" borderColor={ACCENT5}>
            <p style={{ fontSize: 13 }}>{df.multienzyme_triad_note}</p>
          </SectionCard>

          <SectionCard title="🍒 Cherry-Red Spot Differential Diagnosis" borderColor={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead style={{ backgroundColor: ACCENT, color: 'white' }}>
                  <tr><th>Disease</th><th>Distinguishing Feature</th></tr>
                </thead>
                <tbody>
                  {df.cherry_red_differentials?.map((d, i) => (
                    <tr key={i}>
                      <td><strong>{d.disease}</strong></td>
                      <td>{d.distinguishing}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="💡 Key Concepts (15)" borderColor={ACCENT5}>
            {df.concepts?.map((c, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: '1px solid #e0e0e0' }}>
                <div className="fw-bold small mb-1" style={{ color: ACCENT5 }}>{c.name}</div>
                <div style={{ fontSize: 13 }}>{c.definition}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="📏 Thresholds (12)" borderColor={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead style={{ backgroundColor: ACCENT3, color: 'white' }}>
                  <tr><th>Parameter</th><th>Value</th><th>Action</th></tr>
                </thead>
                <tbody>
                  {df.thresholds?.map((t, i) => (
                    <tr key={i}>
                      <td><strong>{t.parameter}</strong></td>
                      <td>{t.value}</td>
                      <td>{t.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="📚 Standards (12)" borderColor={ACCENT4}>
            <ol style={{ fontSize: 13 }}>
              {df.standards?.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
            </ol>
          </SectionCard>

          <SectionCard title="📝 References (6)" borderColor={ACCENT}>
            <ol style={{ fontSize: 13 }}>
              {df.references?.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
            </ol>
          </SectionCard>

          <SectionCard title="🗓️ Lifecycle Stages (6)" borderColor={ACCENT6}>
            {bk?.lifecycle_stages?.map((s, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT6}`, backgroundColor: '#e3f2fd' }}>
                <div className="fw-bold mb-1" style={{ color: ACCENT6 }}>{s.stage}</div>
                <div className="text-muted small mb-1">{s.age_range}</div>
                <div style={{ fontSize: 13 }} className="mb-1">{s.description}</div>
                <ul className="mb-0" style={{ fontSize: 12 }}>
                  {s.priorities?.map((p, j) => <li key={j}>{p}</li>)}
                </ul>
              </div>
            ))}
          </SectionCard>
        </div>
      )}

      {!ov && !err && (
        <div className="text-center py-5">
          <div className="spinner-border" style={{ color: ACCENT }} />
          <div className="mt-2 text-muted">Loading HEXA / Tay-Sachs Disease Epilepsy data…</div>
        </div>
      )}
    </div>
  );
}
