'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#2e5c8a';   // deep steel blue — RORB / thalamocortical / nuclear receptor
const ACCENT2 = '#8a1c1c';   // deep crimson — contraindications / CBZ-CI / danger
const ACCENT3 = '#1a5a1a';   // deep forest green — seizure freedom / good prognosis
const ACCENT4 = '#7a4f00';   // amber-brown — caution / VPA-monitoring / VPPP

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#eaf2f7', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: 11 }}>
      {text}
    </span>
  );
}

export default function RORBPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/rorb/overview`).then(r => r.json()),
      fetch(`${API}/api/rorb/breakdown`).then(r => r.json()),
      fetch(`${API}/api/rorb/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading RORB data…</p></div>;
  if (error) return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  const ov = overview;
  const bk = breakdown;
  const df = definitions;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="row mb-3">
        <div className="col">
          <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
            🧬 RORB Epilepsy
          </h2>
          <p className="text-muted mb-1" style={{ fontSize: 14 }}>
            <strong>RORB (9q21.13)</strong> · RORβ Orphan Nuclear Receptor ·
            Thalamocortical Transcription Factor · GGE / MAE / DEE Spectrum ·
            OMIM #614142 / #619696
          </p>
          <div className="d-flex flex-wrap gap-2">
            <Badge text="GGE-Absence" color={ACCENT} />
            <Badge text="JME-like Myoclonic" color={ACCENT} />
            <Badge text="MAE Drop-Attacks" color="#8b3a00" />
            <Badge text="DEE-RORB" color="#6a0000" />
            <Badge text="AD 80% Penetrance" color="#2e7d5e" />
            <Badge text="VPA Level-A" color={ACCENT3} />
            <Badge text="ETH Level-A Absence" color={ACCENT3} />
            <Badge text="CBZ ABSOLUTE CI" color={ACCENT2} />
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <div className="row mb-3">
        <div className="col">
          {(ov.critical_alerts || []).map((a, i) => (
            <Alert key={i} text={`⚠️ ${a}`} variant={i === 0 || i === 1 ? 'danger' : i === 2 || i === 3 ? 'warning' : 'info'} />
          ))}
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ─────────────────────────────────────────────────── */}
      {tab === 0 && (
        <>
          <div className="row">
            <KPI label="Patients" value={ov.n_patients} color={ACCENT} />
            <KPI label="Seizure-Free" value={`${ov.seizure_free_pct}%`} color={ACCENT3} />
            <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
            <KPI label="Drop Attacks" value={`${ov.drop_attacks_pct}%`} color="#8b3a00" />
            <KPI label="PPR Positive" value={`${ov.ppr_positive_pct}%`} color="#7a1c7a"} />
            <KPI label="KD Therapy" value={`${ov.on_kd_pct}%`} color="#1a5c6e"} />
          </div>
          <div className="row">
            <KPI label="POLG Done" value={`${ov.polg_done_pct}%`} color={ACCENT4} />
            <KPI label="VPPP Enrolled" value={`${ov.vppp_enrolled_pct}%`} color={ACCENT4} />
            <KPI label="MRI Normal" value={`${ov.mri_normal_pct}%`} color="#2e6e4e"} />
            <KPI label="On VPA" value={`${ov.on_vpa_pct !== undefined ? ov.on_vpa_pct : '—'}%`} color={ACCENT} />
            <KPI label="Penetrance" value="~80%" color={ACCENT} />
            <KPI label="De Novo Rate" value="~60%" color="#4e4e8a"} />
          </div>

          <div className="row mt-2">
            <div className="col-md-6">
              <SectionCard title="Gene Biology — RORB / RORβ (9q21.13)" borderColor={ACCENT}>
                <p style={{ fontSize: 13 }}>
                  <strong>RORB</strong> encodes RORβ (Retinoid acid Receptor-related Orphan Receptor Beta) —
                  an <em>orphan nuclear receptor</em> transcription factor in the NR1 subfamily.
                  RORβ acts as the master regulator of <strong>thalamic relay neuron specification</strong> and
                  <strong> cortical layer IV neuron identity</strong> during fetal development (weeks 15–28).
                </p>
                <ul style={{ fontSize: 13 }}>
                  <li><strong>DBD (2 zinc fingers):</strong> binds RORE (RGGTCA motifs) → controls &gt;1,000 target genes</li>
                  <li><strong>LBD:</strong> orphan pocket; partial agonist cholesterol sulphate; no pharmacological ligand</li>
                  <li><strong>LOF → thalamocortical hyperexcitability:</strong> reduced PV+ interneurons + abnormal relay neurons → 3 Hz SWD</li>
                  <li><strong>Circadian integration:</strong> RORβ drives Bmal1 (core clock) → LOF disrupts sleep-wake seizure gating</li>
                </ul>
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="Etiology Distribution" borderColor={ACCENT}>
                {(bk.etiology_distribution || []).map((e, i) => (
                  <PctBar key={i} label={`${e.etiology} (n=${e.n})`} pct={e.pct}
                    color={i === 3 ? ACCENT2 : i === 2 ? '#8b3a00' : ACCENT} />
                ))}
              </SectionCard>
            </div>
          </div>

          <div className="row">
            <div className="col-md-6">
              <SectionCard title="Seizure Type Prevalence" borderColor="#5a2d82">
                {(bk.seizure_types || []).map((s, i) => (
                  <PctBar key={i} label={s.type} pct={s.prevalence_pct}
                    color={s.prevalence_pct > 70 ? ACCENT : s.prevalence_pct > 45 ? '#8b3a00' : '#5a2d82'} />
                ))}
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="Top Triggers" borderColor="#7a4f00">
                {(bk.triggers || []).map((t, i) => (
                  <PctBar key={i} label={t.trigger} pct={t.prevalence_pct}
                    color={t.prevalence_pct > 80 ? ACCENT2 : t.prevalence_pct > 60 ? ACCENT4 : ACCENT} />
                ))}
              </SectionCard>
            </div>
          </div>

          <SectionCard title="Key Standards" borderColor="#2e5c5c">
            <div className="row">
              {(ov.standards || []).map((s, i) => (
                <div key={i} className="col-md-6 mb-1">
                  <span className="badge me-2" style={{ backgroundColor: '#2e5c5c' }}>{i + 1}</span>
                  <span style={{ fontSize: 12 }}>{s}</span>
                </div>
              ))}
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Tab 1: Patients & Etiology ──────────────────────────────────────── */}
      {tab === 1 && (
        <>
          <SectionCard title="Etiology Catalog — RORB Phenotype Classes" borderColor={ACCENT}>
            {(bk.etiology_catalog || []).map((e, i) => (
              <div key={i} className="mb-4 pb-3 border-bottom">
                <div className="d-flex justify-content-between align-items-start mb-1">
                  <h6 className="fw-bold mb-0" style={{ color: i === 3 ? ACCENT2 : i === 2 ? '#8b3a00' : ACCENT }}>
                    {e.etiology}
                  </h6>
                  <span className="badge" style={{ backgroundColor: ACCENT }}>{e.pct}% · n={e.n}</span>
                </div>
                <p style={{ fontSize: 12, marginBottom: 4 }}><strong>Mechanism:</strong> {e.mechanism}</p>
                <p style={{ fontSize: 12, marginBottom: 4 }}><strong>EEG:</strong> {e.eeg_correlate}</p>
                <p style={{ fontSize: 12, marginBottom: 4 }}><strong>Semiology:</strong> {e.semiology}</p>
                <p style={{ fontSize: 12, marginBottom: 4 }}><strong>Treatment:</strong> {e.treatment}</p>
                <p style={{ fontSize: 12, marginBottom: 0 }}><strong>Prognosis:</strong> {e.prognosis}</p>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Patient Sample (15 of 40)" borderColor={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead style={{ backgroundColor: '#eaf2f7' }}>
                  <tr>
                    <th>ID</th><th>Age</th><th>Sex</th><th>Onset</th><th>Etiology</th>
                    <th>Variant</th><th>Seizures</th><th>AEDs</th><th>Seizure-Free</th>
                    <th>Drop</th><th>PPR</th><th>Cognitive</th><th>POLG</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.patients_sample || []).map((p, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{p.id}</td>
                      <td>{p.age}y</td>
                      <td>{p.sex}</td>
                      <td>{p.onset_age}y</td>
                      <td style={{ maxWidth: 160, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {p.etiology}
                      </td>
                      <td style={{ maxWidth: 160, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {p.variant}
                      </td>
                      <td>{(p.seizure_types || []).join(', ')}</td>
                      <td>{p.aed_count}</td>
                      <td>
                        <span className={`badge ${p.seizure_free ? 'bg-success' : 'bg-secondary'}`}>
                          {p.seizure_free ? 'Yes' : 'No'}
                        </span>
                      </td>
                      <td>
                        {p.drop_attacks
                          ? <span className="badge bg-danger">⚠️ Yes</span>
                          : <span className="badge bg-light text-dark">No</span>}
                      </td>
                      <td>
                        {p.ppr_positive
                          ? <span className="badge bg-warning text-dark">PPR+</span>
                          : '—'}
                      </td>
                      <td>
                        <span className={`badge ${p.cognitive === 'Normal' ? 'bg-success' : p.cognitive === 'Borderline' ? 'bg-warning text-dark' : 'bg-danger'}`}>
                          {p.cognitive}
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${p.polg_done ? 'bg-success' : 'bg-danger'}`}>
                          {p.polg_done ? '✓' : '✗'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Tab 2: Seizures & Triggers ──────────────────────────────────────── */}
      {tab === 2 && (
        <>
          <SectionCard title="Seizure Types — RORB-GGE Spectrum" borderColor="#5a2d82">
            {(bk.seizure_detail || []).map((s, i) => (
              <div key={i} className="mb-4 pb-3 border-bottom">
                <div className="d-flex justify-content-between mb-1">
                  <h6 className="fw-bold mb-0" style={{ color: '#5a2d82' }}>{s.type}</h6>
                  <span className="badge" style={{ backgroundColor: '#5a2d82' }}>{s.prevalence_pct}%</span>
                </div>
                <div className="progress mb-2" style={{ height: 8 }}>
                  <div className="progress-bar" style={{ width: `${s.prevalence_pct}%`, backgroundColor: '#5a2d82' }} />
                </div>
                <p style={{ fontSize: 12, marginBottom: 3 }}><strong>Duration:</strong> {s.duration}</p>
                <p style={{ fontSize: 12, marginBottom: 3 }}><strong>EEG ictal:</strong> {s.eeg_ictal}</p>
                <p style={{ fontSize: 12, marginBottom: 3 }}><strong>Semiology:</strong> {s.semiology}</p>
                <div className="alert alert-info py-1 mb-0" style={{ fontSize: 12 }}>
                  💡 <strong>Clinical tip:</strong> {s.clinical_tip}
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Seizure Triggers — RORB-GGE" borderColor={ACCENT4}>
            {(bk.trigger_detail || []).map((t, i) => (
              <div key={i} className="mb-3 pb-3 border-bottom">
                <div className="d-flex justify-content-between mb-1">
                  <h6 className="fw-bold mb-0" style={{ color: ACCENT4 }}>{t.trigger}</h6>
                  <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.prevalence_pct}%</span>
                </div>
                <div className="progress mb-2" style={{ height: 8 }}>
                  <div className="progress-bar" style={{ width: `${t.prevalence_pct}%`, backgroundColor: ACCENT4 }} />
                </div>
                <p style={{ fontSize: 12, marginBottom: 3 }}><strong>Mechanism:</strong> {t.mechanism}</p>
                <p style={{ fontSize: 12, marginBottom: 0 }}><strong>Management:</strong> {t.management}</p>
              </div>
            ))}
          </SectionCard>
        </>
      )}

      {/* ── Tab 3: Treatments ───────────────────────────────────────────────── */}
      {tab === 3 && (
        <>
          <div className="alert alert-danger py-2 mb-3" style={{ fontSize: 13 }}>
            🚫 <strong>ABSOLUTE CONTRAINDICATIONS in ALL RORB-GGE:</strong>{' '}
            CBZ / OXC / PHT (Na-channel blockers → GGE aggravation → absence status) ·
            Tiagabine (NCSE) — see Definitions tab for full CI detail.
          </div>
          {(bk.treatment_detail || []).map((tx, i) => (
            <SectionCard key={i} title={`${tx.drug} — ${tx.evidence}`}
              borderColor={i === 0 || i === 1 ? ACCENT3 : i === 3 ? ACCENT4 : ACCENT}>
              <div className="row">
                <div className="col-md-6">
                  <p style={{ fontSize: 12, marginBottom: 4 }}><strong>Dose:</strong> {tx.dose}</p>
                  <p style={{ fontSize: 12, marginBottom: 4 }}><strong>MOA:</strong> {tx.moa}</p>
                  <p style={{ fontSize: 12, marginBottom: 4 }}><strong>Efficacy:</strong> {tx.efficacy}</p>
                </div>
                <div className="col-md-6">
                  <p style={{ fontSize: 12, marginBottom: 4 }}><strong>Safety:</strong> {tx.safety}</p>
                  <p style={{ fontSize: 12, marginBottom: 4 }}><strong>Monitoring:</strong> {tx.monitoring}</p>
                  <div className="alert alert-primary py-1 mb-0" style={{ fontSize: 12 }}>
                    🧬 <strong>RORB-specific:</strong> {tx.rorb_specific}
                  </div>
                </div>
              </div>
            </SectionCard>
          ))}

          <SectionCard title="Contraindications — RORB-GGE" borderColor={ACCENT2}>
            {(bk.contraindication_detail || []).map((ci, i) => (
              <div key={i} className="mb-3 pb-3 border-bottom">
                <div className="d-flex justify-content-between mb-1">
                  <h6 className="fw-bold mb-0" style={{ color: ACCENT2 }}>{ci.drug}</h6>
                  <span className="badge bg-danger">{ci.level.split(' — ')[0]}</span>
                </div>
                <p style={{ fontSize: 12, marginBottom: 3 }}><strong>Level:</strong> {ci.level}</p>
                <p style={{ fontSize: 12, marginBottom: 3 }}><strong>Mechanism:</strong> {ci.mechanism}</p>
                <p style={{ fontSize: 12, marginBottom: 3 }}><strong>Consequence:</strong> {ci.consequence}</p>
                <p style={{ fontSize: 12, marginBottom: 0 }}><strong>Alternative:</strong> {ci.alternative}</p>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Monitoring Schedule" borderColor={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead style={{ backgroundColor: '#fef9ec' }}>
                  <tr><th>Monitoring Item</th><th>Frequency</th><th>Rationale</th></tr>
                </thead>
                <tbody>
                  {(bk.monitoring || []).map((m, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{m.item}</td>
                      <td>{m.frequency}</td>
                      <td>{m.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Tab 4: Definitions ──────────────────────────────────────────────── */}
      {tab === 4 && (
        <>
          <div className="row">
            <div className="col-md-6">
              <SectionCard title="Lifecycle Windows — RORB-GGE" borderColor={ACCENT}>
                {(bk.lifecycle || []).map((lc, i) => (
                  <div key={i} className="mb-3 pb-2 border-bottom">
                    <div className="d-flex justify-content-between mb-1">
                      <strong style={{ color: ACCENT, fontSize: 13 }}>{lc.window}</strong>
                      <span className="badge" style={{ backgroundColor: ACCENT }}>{lc.age_range}</span>
                    </div>
                    <p style={{ fontSize: 12, marginBottom: 0 }}>{lc.clinical_notes}</p>
                  </div>
                ))}
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="Key Concepts (15)" borderColor="#2e5c5c">
                {(df.concepts || []).map((c, i) => (
                  <div key={i} className="mb-2 pb-2 border-bottom">
                    <strong style={{ color: '#2e5c5c', fontSize: 13 }}>{c.term}</strong>
                    <p style={{ fontSize: 12, marginBottom: 0 }}>{c.definition}</p>
                  </div>
                ))}
              </SectionCard>
            </div>
          </div>

          <SectionCard title="Clinical Thresholds" borderColor={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm" style={{ fontSize: 12 }}>
                <thead style={{ backgroundColor: '#fef9ec' }}>
                  <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
                </thead>
                <tbody>
                  {(df.thresholds || []).map((t, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{t.parameter}</td>
                      <td>{t.value}</td>
                      <td>{t.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="References" borderColor="#4a4a6e">
            {(df.references || []).map((r, i) => (
              <p key={i} style={{ fontSize: 12, marginBottom: 6 }}>
                <span className="badge me-2" style={{ backgroundColor: '#4a4a6e' }}>{i + 1}</span>
                {r}
              </p>
            ))}
          </SectionCard>
        </>
      )}
    </div>
  );
}
