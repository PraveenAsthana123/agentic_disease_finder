'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4a2080';   // deep purple — SPTAN1 / spectrin / AIS cytoskeleton
const ACCENT2 = '#8a1c1c';   // deep crimson — contraindications / danger
const ACCENT3 = '#1a5a1a';   // forest green — ACTH response / monitoring
const ACCENT4 = '#7a4f00';   // amber — caution / VGB VFD / KD

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#f0eaf7', color: borderColor }}>
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

export default function SPTAN1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/sptan1/overview`).then(r => r.json()),
      fetch(`${API}/api/sptan1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/sptan1/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading SPTAN1 data…</p></div>;
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
            🧬 SPTAN1 Epilepsy
          </h2>
          <p className="text-muted mb-1" style={{ fontSize: 14 }}>
            <strong>SPTAN1 (9q34.11)</strong> · Alpha-II Spectrin · Axon Initial Segment Cytoskeleton ·
            {' '}<strong>DEE5</strong> (OMIM #613477) · West Syndrome → LGS · AD de novo &gt;95%
          </p>
          <p className="text-muted mb-0" style={{ fontSize: 13 }}>
            {ov?.n_patients}-patient cohort · AIS collapse → IS + hypsarrhythmia · Drug-resistant in {ov?.drug_resistant_pct}% ·
            {' '}ACTH first-line · VGB SHARE REMS · KD for drug-resistant LGS
          </p>
        </div>
      </div>

      {/* Critical Alerts */}
      {ov?.critical_alerts?.map((a, i) => (
        <Alert key={i} text={`⚠ ${a}`} variant={i < 2 ? 'danger' : 'warning'} />
      ))}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderColor: `${ACCENT} ${ACCENT} #fff` } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && (
        <div>
          <SectionCard title="Gene &amp; Syndrome Summary" borderColor={ACCENT}>
            <div className="row g-2" style={{ fontSize: 13 }}>
              {[
                ['Gene', ov?.gene],
                ['Locus', ov?.locus],
                ['Protein', ov?.protein],
                ['Inheritance', ov?.inheritance],
                ['Syndrome', ov?.syndrome],
                ['Phenotype', ov?.phenotype_spectrum],
                ['First-line AED', ov?.first_line_aed],
                ['Contraindicated', ov?.ci_aed],
              ].map(([k, v]) => (
                <div key={k} className="col-12 col-md-6">
                  <span className="fw-semibold">{k}:</span>{' '}
                  <span className={k === 'Contraindicated' ? 'text-danger fw-bold' : ''}>{v}</span>
                </div>
              ))}
            </div>
          </SectionCard>

          <div className="row mb-3">
            <KPI label="Patients" value={ov?.n_patients} color={ACCENT} />
            <KPI label="ACTH Response" value={`${ov?.acth_response_pct}%`} color={ACCENT3} />
            <KPI label="Drug-Resistant" value={`${ov?.drug_resistant_pct}%`} color={ACCENT2} />
            <KPI label="LGS Transition" value={`${ov?.lgs_transition_pct}%`} color={ACCENT2} />
            <KPI label="MRI Hypomyelin" value={`${ov?.mri_hypomyelination_pct}%`} color={ACCENT4} />
            <KPI label="On KD" value={`${ov?.on_kd_pct}%`} color={ACCENT3} />
          </div>
          <div className="row mb-3">
            <KPI label="POLG Screened" value={`${ov?.polg_done_pct}%`} color={ACCENT3} />
            <KPI label="VPPP Enrolled" value={`${ov?.vppp_enrolled_pct}%`} color={ACCENT3} />
            <KPI label="Drop Attacks" value={`${ov?.drop_attacks_pct}%`} color={ACCENT2} />
          </div>

          <SectionCard title="Etiology Distribution" borderColor={ACCENT}>
            {ov?.etiology_counts && Object.entries(ov.etiology_counts).map(([e, c]) => (
              <PctBar key={e} label={e} pct={Math.round(100 * c / ov.n_patients)} color={ACCENT} />
            ))}
          </SectionCard>

          <SectionCard title="Key Thresholds" borderColor={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
                <thead><tr><th>Parameter</th><th>Value</th><th>Note</th></tr></thead>
                <tbody>
                  {ov?.thresholds?.map((t, i) => (
                    <tr key={i}><td>{t.parameter}</td><td className="fw-semibold">{t.value}</td><td className="text-muted">{t.note}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="Standards &amp; References" borderColor={ACCENT}>
            <div className="row">
              <div className="col-12 col-md-6">
                <strong style={{ fontSize: 12 }}>Standards</strong>
                {ov?.standards?.map((s, i) => (
                  <div key={i} style={{ fontSize: 12 }}><Badge text={s.standard} color={ACCENT} /> {s.topic}</div>
                ))}
              </div>
              <div className="col-12 col-md-6">
                <strong style={{ fontSize: 12 }}>References</strong>
                {ov?.references?.map((r, i) => (
                  <div key={i} style={{ fontSize: 12 }}><Badge text={r.ref} color={ACCENT4} /> {r.citation}</div>
                ))}
              </div>
            </div>
          </SectionCard>
        </div>
      )}

      {/* ── Tab 1: Patients & Etiology ── */}
      {tab === 1 && (
        <div>
          <SectionCard title="Etiology Catalog — 5 Classes" borderColor={ACCENT}>
            {bk?.etiology_catalog?.map((ec, i) => (
              <div key={i} className="mb-4 p-3 rounded" style={{ background: '#f8f4ff', borderLeft: `3px solid ${ACCENT}` }}>
                <div className="fw-bold mb-1" style={{ color: ACCENT }}>{ec.etiology} ({ec.pct}%, n={ec.n})</div>
                <div style={{ fontSize: 12 }}>
                  <strong>Mechanism:</strong> {ec.mechanism}
                </div>
                <div style={{ fontSize: 12 }}>
                  <strong>EEG:</strong> {ec.eeg_correlate}
                </div>
                <div style={{ fontSize: 12 }}>
                  <strong>Semiology:</strong> {ec.semiology}
                </div>
                <div style={{ fontSize: 12 }}>
                  <strong>Treatment:</strong> {ec.treatment}
                </div>
                <div style={{ fontSize: 12 }}>
                  <strong>Prognosis:</strong> <span className={ec.prognosis.includes('Poor') ? 'text-danger' : 'text-success'}>{ec.prognosis}</span>
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Patient Sample (15 of 40)" borderColor={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 11 }}>
                <thead className="table-light">
                  <tr>
                    <th>ID</th><th>Age</th><th>Sex</th><th>Onset(mo)</th>
                    <th>Etiology</th><th>ACTH</th><th>LGS</th>
                    <th>KD</th><th>DR</th><th>Cognition</th>
                  </tr>
                </thead>
                <tbody>
                  {bk?.patients_sample?.map(p => (
                    <tr key={p.id}>
                      <td className="fw-semibold">{p.id}</td>
                      <td>{p.age}y</td>
                      <td>{p.sex[0]}</td>
                      <td>{p.onset_months}m</td>
                      <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={p.etiology}>{p.etiology.replace('SPTAN1-de-novo-','').replace('phenocopy-DEE-','copy-')}</td>
                      <td>{p.acth_response ? <span className="text-success fw-bold">✓</span> : <span className="text-danger">✗</span>}</td>
                      <td>{p.lgs_transition ? <span className="text-danger fw-bold">LGS</span> : '—'}</td>
                      <td>{p.on_kd ? <span className="text-success">✓</span> : '—'}</td>
                      <td>{p.drug_resistant ? <span className="text-danger fw-bold">DR</span> : '—'}</td>
                      <td style={{ fontSize: 10 }}>{p.cognitive?.split(' (')[0]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="Cohort Metrics" borderColor={ACCENT}>
            <div className="row">
              <div className="col-12 col-md-6">
                {[
                  ['ACTH Response', bk?.summary?.acth_response_pct],
                  ['Drug-Resistant', bk?.summary?.drug_resistant_pct],
                  ['LGS Transition', bk?.summary?.lgs_transition_pct],
                  ['Drop Attacks', bk?.summary?.drop_attacks_pct],
                ].map(([l, v]) => <PctBar key={l} label={l} pct={v}
                  color={l === 'ACTH Response' ? ACCENT3 : ACCENT2} />)}
              </div>
              <div className="col-12 col-md-6">
                {[
                  ['MRI Hypomyelination', bk?.summary?.mri_hypomyelination_pct],
                  ['On Ketogenic Diet', bk?.summary?.on_kd_pct],
                  ['POLG Screened', bk?.summary?.polg_done_pct],
                  ['VPPP Enrolled', bk?.summary?.vppp_enrolled_pct],
                ].map(([l, v]) => <PctBar key={l} label={l} pct={v}
                  color={l === 'MRI Hypomyelination' ? ACCENT4 : ACCENT3} />)}
              </div>
            </div>
          </SectionCard>
        </div>
      )}

      {/* ── Tab 2: Seizures & Triggers ── */}
      {tab === 2 && (
        <div>
          <SectionCard title="Seizure Type Prevalence" borderColor={ACCENT}>
            {bk?.seizure_types?.map((s, i) => (
              <PctBar key={i} label={s.type} pct={s.prevalence_pct} color={ACCENT} />
            ))}
          </SectionCard>

          <SectionCard title="Seizure Types — Detail" borderColor={ACCENT}>
            {bk?.seizure_detail?.map((s, i) => (
              <div key={i} className="mb-4 p-3 rounded" style={{ background: '#f8f4ff', borderLeft: `3px solid ${ACCENT}` }}>
                <div className="fw-bold mb-1" style={{ color: ACCENT }}>{s.type} — {s.prevalence_pct}%</div>
                <div style={{ fontSize: 12 }}><strong>EEG:</strong> {s.eeg}</div>
                <div style={{ fontSize: 12 }}><strong>Semiology:</strong> {s.semiology}</div>
                <div style={{ fontSize: 12 }}><strong>EEG tip:</strong> <em>{s.eeg_tip}</em></div>
                <div style={{ fontSize: 12 }}><strong>Clinical tip:</strong> <em>{s.clinical_tip}</em></div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Trigger Prevalence" borderColor={ACCENT4}>
            {bk?.triggers?.map((t, i) => (
              <PctBar key={i} label={t.trigger} pct={t.prevalence_pct} color={ACCENT4} />
            ))}
          </SectionCard>

          <SectionCard title="Trigger Management" borderColor={ACCENT4}>
            {bk?.trigger_detail?.map((t, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ background: '#fdf8ef', borderLeft: `3px solid ${ACCENT4}` }}>
                <div className="fw-bold" style={{ fontSize: 13, color: ACCENT4 }}>{t.trigger} — {t.prevalence_pct}%</div>
                <div style={{ fontSize: 12 }}><strong>Mechanism:</strong> {t.mechanism}</div>
                <div style={{ fontSize: 12 }}><strong>Management:</strong> {t.management}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Lifecycle Windows" borderColor={ACCENT}>
            {bk?.lifecycle?.map((lw, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: '#f8f4ff', borderLeft: `3px solid ${ACCENT}` }}>
                <div className="fw-bold" style={{ fontSize: 13, color: ACCENT }}>{lw.window}</div>
                <div style={{ fontSize: 12 }}><strong>Events:</strong> {lw.key_events}</div>
                <div style={{ fontSize: 12 }}><strong>Action:</strong> {lw.action}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      )}

      {/* ── Tab 3: Treatments ── */}
      {tab === 3 && (
        <div>
          <SectionCard title="Contraindications" borderColor={ACCENT2}>
            {bk?.contraindication_detail?.map((c, i) => (
              <div key={i} className="mb-3 p-3 rounded" style={{ background: '#fff5f5', borderLeft: `4px solid ${ACCENT2}` }}>
                <div className="fw-bold text-danger mb-1">{c.drug} — {c.risk}</div>
                <div style={{ fontSize: 12 }}><strong>Mechanism:</strong> {c.mechanism}</div>
                <div style={{ fontSize: 12 }}><strong>Evidence:</strong> {c.evidence}</div>
                <div style={{ fontSize: 12 }}><strong>Action:</strong> <span className="fw-semibold">{c.action}</span></div>
                {c.exception && <div style={{ fontSize: 12 }}><strong>Exception:</strong> {c.exception}</div>}
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Treatment Protocols" borderColor={ACCENT}>
            {bk?.treatment_detail?.map((t, i) => (
              <div key={i} className="mb-4 p-3 rounded" style={{ background: '#f8f4ff', borderLeft: `3px solid ${ACCENT}` }}>
                <div className="fw-bold mb-1" style={{ color: ACCENT }}>{t.drug} <Badge text={t.level} color={ACCENT} /></div>
                <div style={{ fontSize: 12 }}><strong>Dose:</strong> {t.dose}</div>
                <div style={{ fontSize: 12 }}><strong>MOA:</strong> {t.moa}</div>
                <div style={{ fontSize: 12 }}><strong>Efficacy:</strong> {t.efficacy}</div>
                <div style={{ fontSize: 12 }}><strong>Safety:</strong> {t.safety}</div>
                <div style={{ fontSize: 12 }}><strong>Monitoring:</strong> {t.monitoring}</div>
                <div style={{ fontSize: 12, color: ACCENT4 }}><strong>SPTAN1 note:</strong> <em>{t.sptan1_note}</em></div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Monitoring Schedule" borderColor={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
                <thead><tr><th>Item</th><th>Frequency</th><th>Rationale</th></tr></thead>
                <tbody>
                  {bk?.monitoring?.map((m, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{m.item}</td>
                      <td>{m.frequency}</td>
                      <td className="text-muted">{m.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </div>
      )}

      {/* ── Tab 4: Definitions ── */}
      {tab === 4 && (
        <div>
          <SectionCard title="Key Concepts (15)" borderColor={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
                <thead><tr><th>Term</th><th>Definition</th></tr></thead>
                <tbody>
                  {df?.concepts?.map((c, i) => (
                    <tr key={i}>
                      <td className="fw-semibold text-nowrap">{c.term}</td>
                      <td>{c.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="Contraindications Reference" borderColor={ACCENT2}>
            {df?.contraindications_full?.map((c, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: '#fff5f5', borderLeft: `3px solid ${ACCENT2}` }}>
                <div className="fw-bold text-danger small">{c.drug} — {c.risk}</div>
                <div style={{ fontSize: 11 }}>{c.action}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Clinical Thresholds" borderColor={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm" style={{ fontSize: 12 }}>
                <thead><tr><th>Parameter</th><th>Value</th><th>Note</th></tr></thead>
                <tbody>
                  {df?.thresholds?.map((t, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{t.parameter}</td>
                      <td><Badge text={t.value} color={ACCENT4} /></td>
                      <td className="text-muted">{t.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="Evidence Standards" borderColor={ACCENT}>
            {df?.standards?.map((s, i) => (
              <div key={i} className="mb-1" style={{ fontSize: 12 }}>
                <Badge text={s.standard} color={ACCENT} /> {s.topic}
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Key References" borderColor={ACCENT4}>
            {df?.references?.map((r, i) => (
              <div key={i} className="mb-1" style={{ fontSize: 12 }}>
                <Badge text={r.ref} color={ACCENT4} /> {r.citation}
              </div>
            ))}
          </SectionCard>
        </div>
      )}
    </div>
  );
}
