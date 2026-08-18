'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4527a0';   // deep indigo-violet — GABRB1 / limbic-hippocampal β1
const ACCENT2 = '#b71c1c';   // dark red — CI / danger / NCSE / BDZ warning
const ACCENT3 = '#1b5e20';   // dark green — KD / ACTH / precision
const ACCENT4 = '#e65100';   // deep orange — triggers / alerts / thresholds

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#f3e5f5', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function TabBtn({ label, active, onClick }) {
  return (
    <button
      className={`btn btn-sm me-1 mb-1 ${active ? 'text-white' : 'btn-outline-secondary'}`}
      style={active ? { backgroundColor: ACCENT, borderColor: ACCENT } : {}}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

export default function Gabrb1Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState('');

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/gabrb1/overview`).then(r => r.json()),
      fetch(`${API}/api/gabrb1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/gabrb1/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefs(df); setLoading(false); })
      .catch(e => { setError(String(e)); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error)   return <div className="p-4 alert alert-danger">{error}</div>;

  const ov = overview;
  const bk = breakdown;
  const df = defs;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="alert py-2 mb-3" style={{ fontSize: 14, backgroundColor: '#ede7f6', borderLeft: `5px solid ${ACCENT}`, color: '#1a0050' }}>
        <strong>🧬 GABRB1 Epilepsy</strong> — DEE / GEFS+ / GABA-A β1 (Beta-1) Subunit / Limbic-Hippocampal /
        Perampanel AMPA Precision · <code style={{ fontSize: 12 }}>4p12 · OMIM *137192</code>
        <span className="ms-2 badge" style={{ backgroundColor: ACCENT, fontSize: 11 }}>β1 = HIPPOCAMPAL TONIC INHIBITION</span>
      </div>

      {/* Key pharmacology warnings */}
      <div className="row g-2 mb-3">
        <div className="col-12 col-md-6">
          <Alert variant="danger" text="⛔ TGB (Tiagabine) ABSOLUTE CI — NCSE in β1-deficient limbic/hippocampal networks" />
        </div>
        <div className="col-12 col-md-6">
          <Alert variant="danger" text="⛔ VPA without POLG1 screening ABSOLUTE CI — fatal hepatic failure if POLG mutation" />
        </div>
        <div className="col-12 col-md-6">
          <Alert variant="warning" text="⚠️ HLA-B*15:02 mandatory before CBZ/OXC (SE Asian) — SJS/TEN CPIC Level A" />
        </div>
        <div className="col-12 col-md-6">
          <Alert variant="warning" text="⚠️ LTG HIGH RISK if myoclonic DEE component — Na-channel worsens myoclonic/atonic" />
        </div>
      </div>

      {/* Tabs */}
      <div className="mb-3">
        {TABS.map((t, i) => <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />)}
      </div>

      {/* ── TAB 0: OVERVIEW ─────────────────────────────────────── */}
      {tab === 0 && ov && (
        <>
          <div className="row g-2 mb-3">
            <KPI label="Total Patients"         value={ov.kpis.total_patients}                 color={ACCENT} />
            <KPI label="Seizure-Free ≥12m"      value={`${ov.kpis.seizure_free_pct}%`}         color={ACCENT3} />
            <KPI label="Drug-Resistant"          value={`${ov.kpis.drug_resistant_pct}%`}       color={ACCENT2} />
            <KPI label="West Syndrome History"   value={`${ov.kpis.west_syndrome_history_pct}%`} color={ACCENT4} />
            <KPI label="Temporal Lobe Features"  value={`${ov.kpis.temporal_lobe_features_pct}%`} color={ACCENT} />
            <KPI label="On Perampanel"           value={`${ov.kpis.on_perampanel_pct}%`}        color="#6a1b9a" />
            <KPI label="On VPA"                  value={`${ov.kpis.on_vpa_pct}%`}               color={ACCENT4} />
            <KPI label="On KD"                   value={`${ov.kpis.on_kd_pct}%`}                color={ACCENT3} />
            <KPI label="POLG Done"               value={`${ov.kpis.polg_done_pct}%`}            color={ACCENT} />
            {ov.kpis.vpa_without_polg > 0 && (
              <KPI label="VPA Without POLG ⚠️"  value={ov.kpis.vpa_without_polg}               color={ACCENT2} />
            )}
          </div>

          <SectionCard title="Gene & Protein Context">
            <div className="row">
              <div className="col-md-6">
                <table className="table table-sm table-bordered mb-0" style={{ fontSize: 13 }}>
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>{ov.gene} ({ov.locus})</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>{ov.protein}</td></tr>
                    <tr><td className="fw-bold">OMIM</td><td>{ov.omim}</td></tr>
                    <tr><td className="fw-bold">Syndrome</td><td>{ov.syndrome}</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>{ov.inheritance}</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="col-md-6">
                <div className="p-2 rounded" style={{ backgroundColor: '#f3e5f5', fontSize: 13 }}>
                  <strong>β1 Key Distinctions vs GABRB2/3:</strong>
                  <ul className="mb-0 mt-1 ps-3">
                    <li><strong>GABRB1 (4p12)</strong>: Limbic/hippocampal β isoform → α5β1γ2 extrasynaptic tonic inhibition → temporal lobe + DEE spectrum</li>
                    <li><strong>GABRB2 (5q34)</strong>: Cortical/thalamic isoform → thalamo-cortical 3 Hz oscillations → CAE/JME/GEFS+</li>
                    <li><strong>GABRB3 (15q12)</strong>: Fetal/neonatal dominant → West syndrome → LGS; BDZ severely reduced (phasic assembly)</li>
                    <li><strong>GABRB1</strong>: BDZ PARTIALLY reduced (extrasynaptic α5β1 less BDZ-sensitive). Perampanel AMPA rationale unique.</li>
                  </ul>
                </div>
              </div>
            </div>
          </SectionCard>

          <SectionCard title="Key Pharmacology — GABRB1-Specific">
            <ul className="mb-0" style={{ fontSize: 13 }}>
              {ov.key_pharmacology.map((p, i) => <li key={i} className="mb-1">{p}</li>)}
            </ul>
          </SectionCard>

          <div className="row">
            <div className="col-md-6">
              <SectionCard title="Etiology Distribution">
                {ov.etiology_distribution.map((e, i) => (
                  <div key={i} className="mb-2">
                    <PctBar label={e.class.replace(/-/g, ' ')} pct={e.pct} color={ACCENT} />
                    <div style={{ fontSize: 12, color: '#555' }}>{e.mechanism}</div>
                  </div>
                ))}
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="Seizure Types" borderColor={ACCENT4}>
                {ov.seizure_types_summary.map((s, i) => (
                  <PctBar key={i} label={s.type} pct={s.prevalence_pct} color={ACCENT4} />
                ))}
              </SectionCard>
              <SectionCard title="Triggers" borderColor={ACCENT2}>
                {ov.triggers_summary.map((t, i) => (
                  <PctBar key={i} label={t.trigger} pct={t.prevalence_pct} color={ACCENT2} />
                ))}
              </SectionCard>
            </div>
          </div>

          <SectionCard title="Contraindications Summary" borderColor={ACCENT2}>
            <div className="row">
              {ov.contraindications_summary.map((c, i) => (
                <div key={i} className="col-12 col-md-6 mb-2">
                  <span className={`badge me-1 ${c.level.includes('ABSOLUTE') ? 'bg-danger' : c.level.includes('HIGH') ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                    {c.level}
                  </span>
                  <span style={{ fontSize: 13 }}>{c.drug}</span>
                </div>
              ))}
            </div>
          </SectionCard>
        </>
      )}

      {/* ── TAB 1: PATIENTS & ETIOLOGY ──────────────────────────── */}
      {tab === 1 && bk && (
        <>
          <SectionCard title="Summary Statistics">
            <div className="row text-center">
              {[
                ['Patients', bk.summary.n, ACCENT],
                ['Seizure-Free', `${bk.summary.seizure_free_pct}%`, ACCENT3],
                ['Drug-Resistant', `${bk.summary.drug_resistant_pct}%`, ACCENT2],
                ['On VPA', `${bk.summary.on_vpa_pct}%`, ACCENT4],
                ['On KD', `${bk.summary.on_kd_pct}%`, ACCENT3],
                ['POLG Done', `${bk.summary.polg_done_pct}%`, ACCENT],
              ].map(([l, v, c], i) => (
                <div key={i} className="col-6 col-md-2 mb-2">
                  <div className="fw-bold fs-5" style={{ color: c }}>{v}</div>
                  <div className="text-muted small">{l}</div>
                </div>
              ))}
            </div>
            {bk.summary.vpa_without_polg > 0 && (
              <Alert variant="danger" text={`⛔ ${bk.summary.vpa_without_polg} patient(s) on VPA without POLG1 screening — immediate action required`} />
            )}
          </SectionCard>

          <SectionCard title="Etiology Classes (Full Detail)">
            {bk.etiology_distribution.map((e, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT}30`, backgroundColor: '#faf7ff' }}>
                <div className="d-flex justify-content-between align-items-center mb-1">
                  <span className="fw-bold" style={{ fontSize: 14, color: ACCENT }}>{e.class.replace(/-/g, ' ')}</span>
                  <span className="badge" style={{ backgroundColor: ACCENT }}>{e.n} patients ({e.pct}%)</span>
                </div>
                <PctBar label="" pct={e.pct} color={ACCENT} />
                <div style={{ fontSize: 12, color: '#444' }}><strong>Mechanism:</strong> {e.mechanism}</div>
                <div style={{ fontSize: 12, color: '#666', fontStyle: 'italic' }}>{e.note}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Patient Sample (first 15)" borderColor={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>ID</th><th>Onset (m)</th><th>Etiology</th><th>Control</th>
                    <th>VPA</th><th>KD</th><th>POLG</th><th>LEV</th><th>PER</th><th>West Hx</th>
                  </tr>
                </thead>
                <tbody>
                  {bk.patients_sample.map((p, i) => (
                    <tr key={i} style={p.vpa_without_polg ? { backgroundColor: '#fff3f3' } : {}}>
                      <td>{p.id}</td>
                      <td>{p.age_onset_months}m</td>
                      <td style={{ maxWidth: 200 }}><small>{p.etiology.replace(/-/g, ' ')}</small></td>
                      <td>
                        <span className={`badge ${p.seizure_control === 'Seizure-free ≥12m' ? 'bg-success' : p.seizure_control === 'Drug-resistant' ? 'bg-danger' : 'bg-warning text-dark'}`} style={{ fontSize: 10 }}>
                          {p.seizure_control}
                        </span>
                      </td>
                      <td>{p.on_vpa ? '✅' : '—'}</td>
                      <td>{p.on_kd ? '🥑' : '—'}</td>
                      <td>{p.polg_done ? '✅' : <span className="text-danger">⚠️</span>}</td>
                      <td>{p.on_lev ? '✅' : '—'}</td>
                      <td>{p.on_perampanel ? '🟣' : '—'}</td>
                      <td>{p.west_syndrome_history ? '🔵' : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="Lifecycle Windows" borderColor={ACCENT3}>
            {bk.lifecycle.map((l, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ backgroundColor: '#f1f8e9', border: '1px solid #c5e1a5' }}>
                <div className="fw-bold" style={{ fontSize: 13, color: ACCENT3 }}>{l.stage}</div>
                <div style={{ fontSize: 12, color: '#444' }}>{l.focus}</div>
              </div>
            ))}
          </SectionCard>
        </>
      )}

      {/* ── TAB 2: SEIZURES & TRIGGERS ──────────────────────────── */}
      {tab === 2 && bk && (
        <>
          <SectionCard title="Seizure Types (full detail)" borderColor={ACCENT4}>
            {bk.seizure_detail.map((s, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT4}40`, backgroundColor: '#fff8f0' }}>
                <div className="d-flex justify-content-between align-items-center mb-1">
                  <span className="fw-bold" style={{ fontSize: 14, color: ACCENT4 }}>{s.type}</span>
                  <span className="badge" style={{ backgroundColor: ACCENT4 }}>{s.prevalence_pct}%</span>
                </div>
                <PctBar label="" pct={s.prevalence_pct} color={ACCENT4} />
                <div className="row mt-1" style={{ fontSize: 12 }}>
                  <div className="col-md-4"><strong>EEG:</strong> {s.eeg}</div>
                  <div className="col-md-4"><strong>Semiology:</strong> {s.semiology}</div>
                  <div className="col-md-4"><strong>💡 Clinical tip:</strong> <em>{s.tip}</em></div>
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Triggers (full detail)" borderColor={ACCENT2}>
            {bk.trigger_detail.map((t, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT2}40`, backgroundColor: '#fff5f5' }}>
                <div className="d-flex justify-content-between align-items-center mb-1">
                  <span className="fw-bold" style={{ fontSize: 14, color: ACCENT2 }}>{t.trigger}</span>
                  <span className="badge bg-danger">{t.prevalence_pct}%</span>
                </div>
                <PctBar label="" pct={t.prevalence_pct} color={ACCENT2} />
                <div className="row mt-1" style={{ fontSize: 12 }}>
                  <div className="col-md-6"><strong>Mechanism:</strong> {t.mechanism}</div>
                  <div className="col-md-6"><strong>Management:</strong> {t.management}</div>
                </div>
              </div>
            ))}
          </SectionCard>
        </>
      )}

      {/* ── TAB 3: TREATMENTS ───────────────────────────────────── */}
      {tab === 3 && bk && (
        <>
          <SectionCard title="Treatments (GABRB1-specific dosing + mechanism)" borderColor={ACCENT3}>
            {bk.treatment_detail.map((t, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT3}40`, backgroundColor: '#f1f8e9' }}>
                <div className="d-flex justify-content-between align-items-start">
                  <div>
                    <span className="fw-bold" style={{ color: ACCENT3, fontSize: 14 }}>{t.name}</span>
                    <span className={`badge ms-2 ${t.level.includes('A') && !t.level.includes('B') && !t.level.includes('C') ? 'bg-success' : t.level.includes('B') ? 'bg-primary' : 'bg-secondary'}`} style={{ fontSize: 11 }}>
                      {t.level}
                    </span>
                  </div>
                </div>
                <div className="row mt-1" style={{ fontSize: 12 }}>
                  <div className="col-md-3"><strong>Indication:</strong> {t.indication}</div>
                  <div className="col-md-3"><strong>Dose:</strong> {t.dose}</div>
                  <div className="col-md-3"><strong>MOA:</strong> {t.moa}</div>
                  <div className="col-md-3"><strong>Efficacy:</strong> {t.efficacy}</div>
                </div>
                <div className="row mt-1" style={{ fontSize: 12 }}>
                  <div className="col-md-6">
                    <span className="badge bg-warning text-dark me-1">Monitoring</span>
                    {t.monitoring}
                  </div>
                  <div className="col-md-6">
                    <span className="badge me-1" style={{ backgroundColor: ACCENT }}>GABRB1-specific</span>
                    {t.gabrb1_note}
                  </div>
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Contraindications (full detail)" borderColor={ACCENT2}>
            {bk.contraindication_detail.map((c, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT2}40`, backgroundColor: '#fff5f5' }}>
                <div className="d-flex align-items-center mb-1">
                  <span className={`badge me-2 ${c.level.includes('ABSOLUTE') ? 'bg-danger' : c.level.includes('HIGH') ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                    {c.level}
                  </span>
                  <span className="fw-bold" style={{ color: ACCENT2, fontSize: 13 }}>{c.drug}</span>
                </div>
                <div style={{ fontSize: 12 }}><strong>Reason:</strong> {c.reason}</div>
                <div style={{ fontSize: 12, color: '#1b5e20' }}><strong>Alternative:</strong> {c.alternative}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Monitoring Schedule" borderColor={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 13 }}>
                <thead>
                  <tr><th>Monitoring Item</th><th>Frequency</th></tr>
                </thead>
                <tbody>
                  {bk.monitoring.map((m, i) => (
                    <tr key={i}>
                      <td>{m.item}</td>
                      <td className="text-muted">{m.frequency}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── TAB 4: DEFINITIONS ──────────────────────────────────── */}
      {tab === 4 && df && (
        <>
          <SectionCard title="Key Concepts (15)">
            {df.concepts.map((c, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ backgroundColor: '#f3e5f5', border: '1px solid #ce93d8' }}>
                <div className="fw-bold" style={{ color: ACCENT, fontSize: 13 }}>{c.term}</div>
                <div style={{ fontSize: 12, color: '#444' }}>{c.definition}</div>
              </div>
            ))}
          </SectionCard>

          <div className="row">
            <div className="col-md-6">
              <SectionCard title="Thresholds" borderColor={ACCENT4}>
                <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
                  <thead><tr><th>Parameter</th><th>Target</th></tr></thead>
                  <tbody>
                    {df.thresholds.map((t, i) => (
                      <tr key={i}><td>{t.parameter}</td><td>{t.target}</td></tr>
                    ))}
                  </tbody>
                </table>
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="Standards" borderColor={ACCENT3}>
                {df.standards.map((s, i) => (
                  <div key={i} className="mb-1" style={{ fontSize: 12 }}>
                    <span className="fw-bold" style={{ color: ACCENT3 }}>{s.standard}:</span>{' '}{s.relevance}
                  </div>
                ))}
              </SectionCard>
            </div>
          </div>

          <SectionCard title="References" borderColor={ACCENT}>
            {df.references.map((r, i) => (
              <div key={i} className="mb-1" style={{ fontSize: 12 }}>
                <span className="badge me-1" style={{ backgroundColor: ACCENT, fontSize: 10 }}>{r.ref}</span>
                {r.citation}
              </div>
            ))}
          </SectionCard>
        </>
      )}
    </div>
  );
}
