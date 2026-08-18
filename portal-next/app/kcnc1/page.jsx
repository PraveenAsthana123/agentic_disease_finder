'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4527a0';   // deep violet — Kv3.1 Shaw subfamily / PV+ interneuron fast-spiking identity
const ACCENT2 = '#b71c1c';   // dark red — CI / danger / absolute contraindications
const ACCENT3 = '#e65100';   // deep orange — alerts / photosensitivity / triggers
const ACCENT4 = '#1b5e20';   // deep green — safe / POLG-safe / seizure freedom
const ACCENT5 = '#004d40';   // dark teal — progressive / ataxia / monitoring

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

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header py-2" style={{ backgroundColor: '#f8f9fa', borderBottom: `1px solid ${borderColor}20` }}>
        <strong style={{ color: borderColor }}>{title}</strong>
      </div>
      <div className="card-body py-3">{children}</div>
    </div>
  );
}

function AlertBox({ color, children }) {
  return (
    <div className="p-3 rounded mb-3" style={{ backgroundColor: color + '15', border: `1px solid ${color}40` }}>
      <span style={{ color }}>{children}</span>
    </div>
  );
}

// ── Tab: Overview ───────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="KCNC1 Gene Overview — Kv3.1 Shaw K⁺ / PV+ Fast-Spiking Interneuron / 21q22.13" borderColor={ACCENT}>
        <div className="row g-2 mb-3">
          <div className="col-md-6"><strong>Gene / Locus:</strong> {data.gene} / {data.locus}</div>
          <div className="col-md-6"><strong>OMIM:</strong> {data.omim_gene} (gene) · {data.omim_disease}</div>
          <div className="col-md-6"><strong>Inheritance:</strong> {data.inheritance}</div>
          <div className="col-md-6"><strong>Precision Therapy:</strong> {data.precision_therapy}</div>
          <div className="col-12 mt-2"><strong>Protein:</strong> {data.protein}</div>
          <div className="col-12 mt-2"><strong>Mechanism:</strong> <span className="text-muted small">{data.mechanism}</span></div>
        </div>
      </SectionCard>

      <AlertBox color={ACCENT2}>
        ⚡ CRITICAL AHA: {data.key_aha}
      </AlertBox>

      <SectionCard title={`Cohort KPIs — ${data.n_patients} KCNC1 / EPM7 Patients`} borderColor={ACCENT}>
        <div className="row g-2">
          <KPI label="LOF Subtype" value={`${data.lof_pct}%`} color={ACCENT} />
          <KPI label="R320H Founder" value={`${data.r320h_pct}%`} color={ACCENT} />
          <KPI label="GTCS-Free" value={`${data.seizure_free_pct}%`} color={ACCENT4} />
          <KPI label="Drug Resistant" value={`${data.drug_resistant_pct}%`} color={ACCENT2} />
          <KPI label="Photosensitive" value={`${data.photosensitive_pct}%`} color={ACCENT3} />
          <KPI label="Cerebellar Ataxia" value={`${data.cerebellar_ataxia_pct}%`} color={ACCENT5} />
          <KPI label="Giant SEPs" value={`${data.giant_sep_pct}%`} color={ACCENT} />
          <KPI label="On LEV" value={`${data.on_lev_pct}%`} color={ACCENT4} />
          <KPI label="On VPA" value={`${data.on_vpa_pct}%`} color={ACCENT} />
          <KPI label="On Piracetam" value={`${data.on_piracetam_pct}%`} color={ACCENT} />
          <KPI label="POLG Done" value={`${data.polg_done_pct}%`} color={ACCENT4} />
          <KPI label="Avg Onset" value={`${data.avg_onset_years}Y`} color={ACCENT} />
        </div>
      </SectionCard>

      <SectionCard title="⚠ Safety Alerts — EPM7 / KCNC1" borderColor={ACCENT2}>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT2 }}>ABSOLUTE CI</span>
          <strong>CBZ / OXC / PHT / LTG:</strong> {data.cbz_lmg_alert}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT2 }}>ABSOLUTE CI</span>
          <strong>VGB:</strong> {data.vgb_alert}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT2 }}>ABSOLUTE CI</span>
          <strong>Tiagabine:</strong> {data.tiagabine_alert}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT2 }}>MANDATORY</span>
          <strong>POLG1 before VPA:</strong> {data.polg_alert}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT3 }}>MANDATORY</span>
          <strong>VPPP females ≥12y on VPA:</strong> {data.vppp_alert}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT3 }}>HIGH PRIORITY</span>
          <strong>Photosensitivity (75%):</strong> {data.photosensitivity_alert}
        </div>
      </SectionCard>

      <SectionCard title="Diagnostic Mandatories — Giant SEPs + Jerk-Locked Back-Averaging" borderColor={ACCENT5}>
        <div className="row g-2">
          <div className="col-md-6 small">
            <strong>Giant SEPs:</strong> N20-P25 &gt;4 µV (normal &lt;2 µV) — present in &gt;80% EPM7.
            MANDATORY at diagnosis. Hallmark of cortical myoclonus. Track every 2 years (amplitude increase = disease progression).
          </div>
          <div className="col-md-6 small">
            <strong>Jerk-Locked Back-Averaging (EEG-EMG):</strong> Cortical spike 15-25 ms before EMG jerk
            confirms cortical myoclonus origin. Essential for EPM7 diagnosis and distinguishing
            from subcortical / spinal myoclonus (no giant SEPs, no cortical correlate).
          </div>
          <div className="col-md-6 small">
            <strong>IPS (Photic Stimulation):</strong> PPR in ~75% EPM7 — one of highest rates in genetic epilepsy.
            Test at diagnosis and annually. LEV reduces PPR threshold in ~60%.
          </div>
          <div className="col-md-6 small">
            <strong>SARA / ICARS (Ataxia Scale):</strong> Quantify cerebellar ataxia at baseline and annually.
            SARA 0-40. Progressive — track rate of cerebellar deterioration.
          </div>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Patients & Etiology ─────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { summary, etiologies, patients } = data;
  return (
    <div>
      <SectionCard title="Cohort Summary" borderColor={ACCENT}>
        <div className="row g-2">
          <KPI label="Total" value={summary?.total ?? 40} color={ACCENT} />
          <KPI label="LOF" value={`${summary?.lof_pct ?? 98}%`} color={ACCENT} />
          <KPI label="R320H" value={`${summary?.r320h_founder_pct ?? 60}%`} color={ACCENT} />
          <KPI label="On LEV" value={`${summary?.on_lev_pct ?? 92}%`} color={ACCENT4} />
          <KPI label="Giant SEPs" value={`${summary?.giant_sep_pct ?? 82}%`} color={ACCENT} />
          <KPI label="Photosensitive" value={`${summary?.photosensitive_pct ?? 75}%`} color={ACCENT3} />
          <KPI label="Cerebellar Ataxia" value={`${summary?.cerebellar_ataxia_pct ?? 88}%`} color={ACCENT5} />
          <KPI label="Drug Resistant" value={`${summary?.drug_resistant_pct ?? 42}%`} color={ACCENT2} />
        </div>
        <AlertBox color={ACCENT2}>
          ⛔ Key CI: {summary?.key_ci ?? 'CBZ/OXC/LTG — NaV1.1 interneuron blockade → myoclonus catastrophic worsening'}
        </AlertBox>
      </SectionCard>

      <SectionCard title="Etiology Catalog — KCNC1 Variant Classes" borderColor={ACCENT}>
        {(etiologies || []).map((e, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ backgroundColor: '#f8f9fa', border: `1px solid ${ACCENT}20` }}>
            <div className="d-flex justify-content-between align-items-start mb-2">
              <strong style={{ color: ACCENT }}>{e.etiology}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{e.pct}% (n={e.n})</span>
            </div>
            <div className="small text-muted mb-2"><strong>Mechanism:</strong> {e.mechanism}</div>
            <div className="small mb-1"><strong>EEG:</strong> {e.eeg_correlate}</div>
            <div className="small mb-1"><strong>MRI:</strong> {e.mri_finding}</div>
            <div className="small"><strong>Semiology:</strong> {e.semiology}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Sample (first 15)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead style={{ backgroundColor: ACCENT + '15' }}>
              <tr>
                <th>ID</th><th>Etiology</th><th>Onset (Y)</th>
                <th>LEV</th><th>LEV Resp</th><th>VPA</th><th>CLB</th><th>Piracetam</th>
                <th>POLG</th><th>Giant SEP</th><th>Photo</th><th>Ataxia</th><th>DRE</th>
              </tr>
            </thead>
            <tbody>
              {(patients || []).slice(0, 15).map((p, i) => (
                <tr key={i}>
                  <td className="small">{p.id}</td>
                  <td className="small" style={{ maxWidth: 140 }}>{p.category}</td>
                  <td className="small">{p.onset_years}</td>
                  <td><span style={{ color: p.on_lev ? ACCENT4 : '#999' }}>{p.on_lev ? '✓' : '–'}</span></td>
                  <td><span style={{ color: p.lev_responder ? ACCENT4 : ACCENT2 }}>{p.lev_responder ? '✓' : '✗'}</span></td>
                  <td><span style={{ color: p.on_vpa ? ACCENT : '#999' }}>{p.on_vpa ? '✓' : '–'}</span></td>
                  <td><span style={{ color: p.on_clb ? ACCENT : '#999' }}>{p.on_clb ? '✓' : '–'}</span></td>
                  <td><span style={{ color: p.on_piracetam ? ACCENT : '#999' }}>{p.on_piracetam ? '✓' : '–'}</span></td>
                  <td><span style={{ color: p.polg_done ? ACCENT4 : ACCENT2 }}>{p.polg_done ? 'Y' : '?'}</span></td>
                  <td><span style={{ color: p.giant_sep ? ACCENT : '#999' }}>{p.giant_sep ? '✓' : '–'}</span></td>
                  <td><span style={{ color: p.photosensitive ? ACCENT3 : '#999' }}>{p.photosensitive ? '⚡' : '–'}</span></td>
                  <td><span style={{ color: p.cerebellar_ataxia ? ACCENT5 : '#999' }}>{p.cerebellar_ataxia ? '✓' : '–'}</span></td>
                  <td><span style={{ color: p.drug_resistant ? ACCENT2 : ACCENT4 }}>{p.drug_resistant ? '✓' : '–'}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Seizure Types & Triggers ────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types, triggers } = data;
  return (
    <div>
      <SectionCard title="Seizure / Symptom Type Distribution" borderColor={ACCENT}>
        {(seizure_types || []).map((s, i) => (
          <PctBar key={i} label={s.type} pct={s.prevalence_pct} color={ACCENT} />
        ))}
      </SectionCard>

      <SectionCard title="Seizure Types — EEG + Semiology + Clinical Tips" borderColor={ACCENT}>
        {(seizure_types || []).map((s, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ backgroundColor: '#f8f9fa', border: `1px solid ${ACCENT}20` }}>
            <div className="d-flex justify-content-between mb-2">
              <strong style={{ color: ACCENT }}>{s.type}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{s.prevalence_pct}%</span>
            </div>
            <div className="small mb-1"><strong>EEG:</strong> {s.eeg}</div>
            <div className="small mb-1"><strong>Semiology:</strong> {s.semiology}</div>
            <div className="small p-2 rounded" style={{ backgroundColor: ACCENT + '10', color: ACCENT }}>
              <strong>💡 Clinical Tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Trigger Distribution" borderColor={ACCENT3}>
        {(triggers || []).map((t, i) => (
          <PctBar key={i} label={t.trigger} pct={t.prevalence_pct} color={ACCENT3} />
        ))}
      </SectionCard>

      <SectionCard title="Trigger Detail — Mechanism + Management" borderColor={ACCENT3}>
        {(triggers || []).map((t, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ backgroundColor: '#fff8f0', border: `1px solid ${ACCENT3}30` }}>
            <div className="d-flex justify-content-between mb-2">
              <strong style={{ color: ACCENT3 }}>{t.trigger}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.prevalence_pct}%</span>
            </div>
            <div className="small mb-1"><strong>Mechanism:</strong> {t.mechanism}</div>
            <div className="small p-2 rounded" style={{ backgroundColor: ACCENT3 + '10' }}>
              <strong>Management:</strong> {t.management}
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Treatments ──────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatments, contraindications, monitoring, lifecycle } = data;
  return (
    <div>
      <SectionCard title="Treatment Protocols — EPM7 / KCNC1-Specific" borderColor={ACCENT4}>
        {(treatments || []).map((t, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ backgroundColor: '#f0fdf4', border: `1px solid ${ACCENT4}30` }}>
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ color: ACCENT4 }}>{t.drug}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.evidence}</span>
            </div>
            <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>
            <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small mb-1"><strong>Safety:</strong> {t.safety}</div>
            {t.kcnc1_note && (
              <div className="small p-2 rounded" style={{ backgroundColor: ACCENT + '10', color: ACCENT }}>
                <strong>🧬 KCNC1 Note:</strong> {t.kcnc1_note}
              </div>
            )}
            {(t.monitoring || []).length > 0 && (
              <div className="small mt-2">
                <strong>Monitoring:</strong>{' '}
                {t.monitoring.map((m, j) => (
                  <span key={j} className="badge me-1" style={{ backgroundColor: ACCENT + '25', color: ACCENT }}>{m}</span>
                ))}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications — EPM7 / KCNC1" borderColor={ACCENT2}>
        {(contraindications || []).map((c, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ backgroundColor: '#fff0f0', border: `1px solid ${ACCENT2}30` }}>
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ color: ACCENT2 }}>⛔ {c.drug}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT2 }}>{c.severity}</span>
            </div>
            <div className="small mb-1">{c.mechanism}</div>
            {c.alternative && (
              <div className="small p-2 rounded" style={{ backgroundColor: ACCENT4 + '10', color: ACCENT4 }}>
                <strong>Alternative:</strong> {c.alternative}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Protocol" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead style={{ backgroundColor: ACCENT + '15' }}>
              <tr><th>Monitoring Item</th><th>Frequency</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{m.item}</td>
                  <td className="small">{m.frequency}</td>
                  <td className="small text-muted">{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Disease Lifecycle — EPM7 / KCNC1 Progressive Stages" borderColor={ACCENT5}>
        {(lifecycle || []).map((l, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ backgroundColor: '#e0f2f1', border: `1px solid ${ACCENT5}20` }}>
            <strong style={{ color: ACCENT5 }}>{l.stage}</strong>
            <div className="small mt-1 text-muted">{l.description}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { concepts, thresholds, standards, references } = data;
  return (
    <div>
      <SectionCard title="Key Concepts (15)" borderColor={ACCENT}>
        {(concepts || []).map((c, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ backgroundColor: '#f3f0ff', border: `1px solid ${ACCENT}20` }}>
            <strong style={{ color: ACCENT }}>{c.term}</strong>
            <div className="small text-muted mt-1">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead style={{ backgroundColor: ACCENT3 + '15' }}>
              <tr><th>Parameter</th><th>Normal</th><th>Action Threshold</th><th>Action</th></tr>
            </thead>
            <tbody>
              {(thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{t.parameter}</td>
                  <td className="small text-muted">{t.normal}</td>
                  <td className="small" style={{ color: ACCENT3 }}>{t.action_threshold}</td>
                  <td className="small">{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Clinical Standards" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead style={{ backgroundColor: ACCENT + '15' }}>
              <tr><th>Standard</th><th>Relevance</th></tr>
            </thead>
            <tbody>
              {(standards || []).map((s, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{s.standard}</td>
                  <td className="small text-muted">{s.relevance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="References" borderColor={ACCENT}>
        <ol className="small mb-0">
          {(references || []).map((r, i) => (
            <li key={i} className="mb-1">{r.citation}</li>
          ))}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function KCNC1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/kcnc1/overview`).then(r => r.json()),
      fetch(`${API}/api/kcnc1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/kcnc1/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(String(e)));
  }, []);

  if (error) return <div className="alert alert-danger m-4">{error}</div>;

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4 p-4 rounded text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT5} 100%)` }}>
        <h2 className="mb-1">⚡ KCNC1 Epilepsy — Progressive Myoclonic Epilepsy 7 (EPM7)</h2>
        <div className="opacity-90 small">
          Kv3.1 Shaw K⁺ Channel · PV+ Fast-Spiking Interneuron · R320H Founder Mutation · 21q22.13
          &nbsp;·&nbsp;OMIM #618323&nbsp;·&nbsp;Action Myoclonus + Cerebellar Ataxia + GTCS (Progressive)
          &nbsp;·&nbsp;<strong>CBZ / OXC / LTG / PHT ABSOLUTE CI</strong> (NaV1.1 interneuron → myoclonus worsening)
          &nbsp;·&nbsp;LEV Level A first-line&nbsp;·&nbsp;POLG1 mandatory before VPA
          &nbsp;·&nbsp;Giant SEPs mandatory at diagnosis&nbsp;·&nbsp;Photosensitivity 75%
        </div>
      </div>

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { borderBottomColor: ACCENT, color: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
