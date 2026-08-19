'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#880033';   // dark-cherry-spot-maroon — NEU1 / Sialidosis Type I / Cherry-Red Spot reference
const ACCENT2 = '#b71c1c';   // dark-red — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep-orange — high-risk warnings / urgent
const ACCENT4 = '#1b5e20';   // deep-green — safe treatments / monitoring
const ACCENT5 = '#6a1b9a';   // deep-purple — multienzyme complex / molecular biology
const ACCENT6 = '#0d47a1';   // deep-blue — normal cognition note / research

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#fce4ec', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

export default function NEU1Page() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/neu1/overview`).then(r => r.json()).then(setOv).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 && !bk) fetch(`${API}/api/neu1/breakdown`).then(r => r.json()).then(setBk).catch(e => setErr(String(e)));
    if (tab === 4 && !df) fetch(`${API}/api/neu1/definitions`).then(r => r.json()).then(setDf).catch(e => setErr(String(e)));
    if ((tab === 2 || tab === 3) && !bk) fetch(`${API}/api/neu1/breakdown`).then(r => r.json()).then(setBk).catch(e => setErr(String(e)));
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div className="rounded p-3 mb-3 text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #5c0022 100%)` }}>
        <h4 className="mb-1 fw-bold">🍒 NEU1 Epilepsy — Sialidosis Type I (Cherry-Red-Spot-Myoclonus Syndrome)</h4>
        <div style={{ fontSize: 13 }}>
          <strong>NEU1 (6p21.33)</strong> — α-Neuraminidase 1 · Lysosomal Sialidase · AR Biallelic LOF
          <span className="ms-3 badge bg-warning text-dark">Adolescent Onset 8–25y (mean 14.2y)</span>
          <span className="ms-2 badge" style={{ backgroundColor: ACCENT6 }}>🧠 Normal Cognition — Distinguishes from ALL NCL</span>
          <span className="ms-2 badge bg-danger">CBZ/OXC/PHT ABSOLUTE CI — PME Myoclonus Trap</span>
        </div>
      </div>

      {/* CHERRY-RED SPOT ALERT */}
      <div className="alert py-2 mb-2" style={{ backgroundColor: '#fce4ec', borderLeft: `5px solid ${ACCENT}`, fontSize: 13 }}>
        <strong>🍒 PATHOGNOMONIC COMBINATION — CHERRY-RED SPOT + PME + NORMAL COGNITION + ADOLESCENT ONSET = SIALIDOSIS TYPE I:</strong>{' '}
        Cherry-red macular spot (90-95%) + action myoclonus (98%) + GTCS + NORMAL INTELLIGENCE in adolescent/young adult.{' '}
        <strong>Fundoscopy is the FASTEST diagnostic clue</strong> (minutes in any eye department).{' '}
        <strong>Visual acuity PRESERVED</strong> (unlike retinal NCL where progressive blindness occurs).
      </div>

      {/* NORMAL COGNITION ALERT */}
      <div className="alert py-2 mb-2" style={{ backgroundColor: '#e8f5e9', borderLeft: `5px solid ${ACCENT4}`, fontSize: 13 }}>
        <strong>🧠 NORMAL COGNITION — KEY DISTINCTION FROM ALL NCL TYPES AND SIALIDOSIS TYPE II:</strong>{' '}
        Sialidosis Type I PRESERVES cognitive function (unlike ALL NCL types which cause universal progressive cognitive decline).{' '}
        Adolescent PME + <strong>NORMAL SCHOOL PERFORMANCE</strong> → Sialidosis Type I first differential.{' '}
        Enables employment, driving, independent living, relationships — major quality-of-life distinction.
      </div>

      {/* CBZ/FOSPHENYTOIN ABSOLUTE CI ALERT */}
      <div className="alert alert-danger py-2 mb-3" style={{ fontSize: 13 }}>
        <strong>🚫 CBZ/OXC/PHT ABSOLUTE CI — FOSPHENYTOIN ABSOLUTE CI — TGB ABSOLUTE CI IN SIALIDOSIS PME:</strong>{' '}
        Na-channel blockers worsen cortical PME myoclonus. Adolescent GTCS frequently misidentified as JME → CBZ → acute myoclonic worsening.
        Fosphenytoin ABSOLUTE CI in SE (IV Na-channel). TGB ABSOLUTE CI (NCSE risk). VGB HIGH RISK (cherry-red macular — not retinal NCL ABSOLUTE CI).{' '}
        <strong>Safe PME backbone: VPA + LEV + Piracetam.</strong>{' '}
        <strong>POLG1/MERRF EXCLUSION MANDATORY before VPA</strong> (MERRF = VPA absolute CI, phenocopy with cherry-red + PME).
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
            <KPI label="Cherry-Red Spot" value={`${ov.cherry_red_spot_pct}%`} color={ACCENT} />
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

          <SectionCard title="🍒 Cherry-Red Macular Spot — Mechanism & Clinical Significance" borderColor={ACCENT}>
            <p style={{ fontSize: 13 }}>{ov.cherry_red_spot_note}</p>
          </SectionCard>

          <SectionCard title="🔬 NEU1-CTSA-GLB1 Multienzyme Complex — Galactosialidosis Differential" borderColor={ACCENT5}>
            <p style={{ fontSize: 13 }}>{ov.multienzyme_complex_note}</p>
          </SectionCard>

          <SectionCard title="🧠 Normal Cognition — Key Distinction from NCL" borderColor={ACCENT4}>
            <p style={{ fontSize: 13 }}>{ov.normal_cognition_note}</p>
          </SectionCard>

          <SectionCard title="💊 Key Pharmacological Distinctions" borderColor={ACCENT2}>
            {ov.key_pharmacological_distinctions && Object.entries(ov.key_pharmacological_distinctions).map(([k, v]) => (
              <div key={k} className="mb-3 p-2 rounded" style={{ backgroundColor: '#fafafa', border: '1px solid #ddd' }}>
                <div className="fw-bold small mb-1" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ fontSize: 13 }}>{v}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="📅 Discovery" borderColor={ACCENT}>
            <p style={{ fontSize: 13 }}>{ov.discovery}</p>
            <p style={{ fontSize: 13 }}><strong>Unique Feature:</strong> {ov.unique_feature}</p>
          </SectionCard>

          <div className="row">
            <div className="col-md-6">
              <SectionCard title="📊 Clinical Stats" borderColor={ACCENT4}>
                <PctBar label="Cherry-Red Macular Spot" pct={ov.cherry_red_spot_pct} color={ACCENT} />
                <PctBar label="Action Myoclonus (98%)" pct={ov.myoclonus_pct} color={ACCENT2} />
                <PctBar label="Stimulus-Sensitive Myoclonus" pct={ov.stimulus_sensitive_pct} color={ACCENT2} />
                <PctBar label="Cerebellar Ataxia" pct={ov.cerebellar_ataxia_pct} color={ACCENT3} />
                <PctBar label="GTCS" pct={ov.gtcs_pct} color={ACCENT3} />
                <PctBar label="Drug-Resistant PME" pct={ov.drug_resistant_pct} color={ACCENT2} />
                <PctBar label="Photosensitivity" pct={ov.photosensitivity_pct} color={ACCENT5} />
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="📊 Cohort Profile" borderColor={ACCENT4}>
                <PctBar label="Visual Acuity Preserved" pct={ov.visual_acuity_preserved_pct} color={ACCENT4} />
                <PctBar label="Cognitive Impairment (low — normal IQ)" pct={ov.cognitive_impairment_pct} color={ACCENT6} />
                <PctBar label="On VPA" pct={ov.on_vpa_pct} color={ACCENT4} />
                <PctBar label="On LEV" pct={ov.on_lev_pct} color={ACCENT4} />
                <PctBar label="On Piracetam" pct={ov.on_piracetam_pct} color={ACCENT4} />
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
                <div style={{ fontSize: 12, color: ACCENT }}><strong>NEU1 note:</strong> {t.neu1_note}</div>
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

          <SectionCard title="🔬 NEU1-CTSA-GLB1 Multienzyme Complex" borderColor={ACCENT5}>
            <p style={{ fontSize: 13 }}>{df.multienzyme_complex}</p>
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
              <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT6}`, backgroundColor: '#f8f9ff' }}>
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
          <div className="mt-2 text-muted">Loading NEU1 / Sialidosis Type I Epilepsy data…</div>
        </div>
      )}
    </div>
  );
}
