'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#0277bd';   // medium-dark blue — Kv7.5 M-current / KCNQ5 (distinct from KCNQ2/3 deep blue #1a5276)
const ACCENT2 = '#b71c1c';   // dark red — CI / danger / withdrawn
const ACCENT3 = '#e65100';   // deep orange — alerts / warnings
const ACCENT4 = '#1b5e20';   // deep green — seizure freedom / safe
const ACCENT5 = '#4a148c';   // purple — mechanisms / LOF paradox

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

// ── Tab: Overview ──────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="KCNQ5 Gene Overview — Kv7.5 M-current / Interneuron-Enriched / 6q14.1" borderColor={ACCENT}>
        <div className="row g-2 mb-3">
          <div className="col-md-6"><strong>Gene / Locus:</strong> {data.gene} / {data.locus}</div>
          <div className="col-md-6"><strong>Inheritance:</strong> {data.inheritance}</div>
          <div className="col-12 mt-2"><strong>Protein:</strong> {data.protein}</div>
          <div className="col-12 mt-2"><strong>Mechanism:</strong> <span className="text-muted small">{data.mechanism}</span></div>
        </div>
      </SectionCard>

      <AlertBox color={ACCENT2}>
        ⚡ CRITICAL AHA: {data.key_aha}
      </AlertBox>

      <SectionCard title={`Cohort KPIs — ${data.n_patients} KCNQ5 Patients`} borderColor={ACCENT}>
        <div className="row g-2">
          <KPI label="GOF Subtype" value={`${data.gof_pct}%`} color={ACCENT} />
          <KPI label="Seizure Free" value={`${data.seizure_free_pct}%`} color={ACCENT4} />
          <KPI label="Drug Resistant" value={`${data.drug_resistant_pct}%`} color={ACCENT2} />
          <KPI label="West Syndrome" value={`${data.west_pct}%`} color={ACCENT3} />
          <KPI label="On CBZ/OXC" value={`${data.on_cbz_oxc_pct}%`} color={ACCENT} />
          <KPI label="On KD" value={`${data.on_kd_pct}%`} color={ACCENT5} />
          <KPI label="On LEV" value={`${data.on_lev_pct}%`} color={ACCENT} />
          <KPI label="Paradoxical CBZ↑" value={`${data.paradoxical_cbz_pct}%`} color={ACCENT3} />
          <KPI label="POLG Done" value={`${data.polg_done_pct}%`} color={ACCENT4} />
          <KPI label="HLA Done" value={`${data.hla_done_pct}%`} color={ACCENT4} />
          <KPI label="Hyponatraemia" value={`${data.hyponatraemia_pct}%`} color={ACCENT3} />
          <KPI label="Avg Onset" value={`${data.avg_onset_months}M`} color={ACCENT} />
        </div>
      </SectionCard>

      <SectionCard title="⚠ Safety Alerts" borderColor={ACCENT2}>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT2 }}>ABSOLUTE CI</span>
          <strong>Tiagabine:</strong> {data.tiagabine_alert}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT3 }}>HIGH RISK / CAUTION</span>
          <strong>CBZ/OXC:</strong> {data.cbz_alert}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT2 }}>WITHDRAWN 2017</span>
          <strong>Retigabine:</strong> {data.retigabine_alert}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT2 }}>MANDATORY</span>
          <strong>POLG:</strong> {data.polg_alert}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT5 }}>LOF PARADOX</span>
          <strong>CBZ/OXC in LOF:</strong> {data.paradoxical_alert}
        </div>
      </SectionCard>

      <SectionCard title="Contraindications Summary" borderColor={ACCENT2}>
        {(data.contraindications_summary || []).map((ci, i) => (
          <div key={i} className="mb-1 small">
            <span className="me-2" style={{ color: ACCENT2 }}>⛔</span>{ci}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Patients & Etiology ──────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { summary, etiology_distribution, patients_sample } = data;
  return (
    <div>
      <SectionCard title="Cohort Summary" borderColor={ACCENT}>
        <div className="row g-2">
          <KPI label="Total" value={summary.n} color={ACCENT} />
          <KPI label="Seizure Free" value={`${summary.seizure_free_pct}%`} color={ACCENT4} />
          <KPI label="Drug Resistant" value={`${summary.drug_resistant_pct}%`} color={ACCENT2} />
          <KPI label="On CBZ/OXC" value={`${summary.on_cbz_oxc_pct}%`} color={ACCENT} />
          <KPI label="On KD" value={`${summary.on_kd_pct}%`} color={ACCENT5} />
          <KPI label="Paradoxical CBZ↑" value={`${summary.paradoxical_cbz_pct}%`} color={ACCENT3} />
          <KPI label="POLG Done" value={`${summary.polg_done_pct}%`} color={ACCENT4} />
          <KPI label="HLA Done" value={`${summary.hla_done_pct}%`} color={ACCENT4} />
          <KPI label="Hyponatraemia" value={`${summary.hyponatraemia_pct}%`} color={ACCENT3} />
        </div>
        {summary.vpa_without_polg > 0 && (
          <AlertBox color={ACCENT2}>
            ⚠ {summary.vpa_without_polg} patient(s) on VPA without POLG1 screening documented — urgent review required.
          </AlertBox>
        )}
      </SectionCard>

      <SectionCard title="Etiology Catalog — KCNQ5 Variant Classes" borderColor={ACCENT}>
        {(etiology_distribution || []).map((e, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ backgroundColor: '#f8f9fa', border: `1px solid ${ACCENT}20` }}>
            <div className="d-flex justify-content-between align-items-start mb-2">
              <strong style={{ color: ACCENT }}>{e.etiology}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{e.pct}% (n={e.n})</span>
            </div>
            <div className="small text-muted mb-1"><strong>Mechanism:</strong> {e.mechanism_short}…</div>
            <div className="small text-muted"><strong>EEG:</strong> {e.eeg_signature_short}…</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Sample (first 15)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead style={{ backgroundColor: ACCENT + '15' }}>
              <tr>
                <th>ID</th><th>Name</th><th>Age (M)</th><th>Onset (M)</th><th>Etiology</th>
                <th>Sz-Free</th><th>DRE</th><th>West</th><th>ID</th>
                <th>CBZ/OXC</th><th>Paradoxical</th><th>POLG</th><th>HLA</th>
              </tr>
            </thead>
            <tbody>
              {(patients_sample || []).map((p, i) => (
                <tr key={i}>
                  <td className="small">{p.id}</td>
                  <td className="small">{p.name}</td>
                  <td className="small">{p.age_months}</td>
                  <td className="small">{p.onset_months}</td>
                  <td className="small" style={{ maxWidth: 160 }}>{p.etiology}</td>
                  <td><span style={{ color: p.seizure_free ? ACCENT4 : ACCENT2 }}>{p.seizure_free ? '✓' : '✗'}</span></td>
                  <td><span style={{ color: p.drug_resistant ? ACCENT2 : ACCENT4 }}>{p.drug_resistant ? '✓' : '–'}</span></td>
                  <td><span style={{ color: p.west_syndrome ? ACCENT3 : '#999' }}>{p.west_syndrome ? '✓' : '–'}</span></td>
                  <td className="small">{p.id_severity}</td>
                  <td><span style={{ color: p.on_cbz_oxc ? ACCENT : '#999' }}>{p.on_cbz_oxc ? '✓' : '–'}</span></td>
                  <td><span style={{ color: p.paradoxical_cbz_worsening ? ACCENT3 : '#999' }}>{p.paradoxical_cbz_worsening ? '⚠' : '–'}</span></td>
                  <td><span style={{ color: p.polg_tested === 'Y' ? ACCENT4 : ACCENT2 }}>{p.polg_tested}</span></td>
                  <td><span style={{ color: p.hla_b1502_tested ? ACCENT4 : ACCENT2 }}>{p.hla_b1502_tested ? 'Y' : 'N'}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Seizure Types & Triggers ─────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types, seizure_detail, triggers, trigger_detail } = data;
  return (
    <div>
      <SectionCard title="Seizure Type Distribution" borderColor={ACCENT}>
        {(seizure_types || []).map((s, i) => (
          <PctBar key={i} label={s.type} pct={s.prevalence_pct} color={ACCENT} />
        ))}
      </SectionCard>

      <SectionCard title="Seizure Types — EEG + Semiology + Clinical Tips" borderColor={ACCENT}>
        {(seizure_detail || []).map((s, i) => (
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
        {(trigger_detail || []).map((t, i) => (
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

// ── Tab: Treatments ───────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatment_detail, contraindication_detail, monitoring, lifecycle } = data;
  return (
    <div>
      <SectionCard title="Treatment Protocols — KCNQ5-Specific" borderColor={ACCENT4}>
        {(treatment_detail || []).map((t, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ backgroundColor: '#f0fdf4', border: `1px solid ${ACCENT4}30` }}>
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ color: ACCENT4 }}>{t.drug}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.evidence}</span>
            </div>
            <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>
            <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small mb-1"><strong>Safety:</strong> {t.safety}</div>
            {t.kcnq5_note && (
              <div className="small p-2 rounded" style={{ backgroundColor: ACCENT + '10', color: ACCENT }}>
                <strong>🧬 KCNQ5 Note:</strong> {t.kcnq5_note}
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

      <SectionCard title="Contraindications" borderColor={ACCENT2}>
        {(contraindication_detail || []).map((c, i) => (
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
              <tr><th>Monitoring Item</th><th>Timing</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{m.item}</td>
                  <td className="small">{m.timing}</td>
                  <td className="small text-muted">{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Disease Lifecycle — KCNQ5 Epilepsy Stages" borderColor={ACCENT5}>
        {(lifecycle || []).map((l, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ backgroundColor: '#f5f0ff', border: `1px solid ${ACCENT5}20` }}>
            <strong style={{ color: ACCENT5 }}>{l.stage}</strong>
            <ul className="mt-2 mb-2 small">
              {(l.key_events || []).map((e, j) => <li key={j}>{e}</li>)}
            </ul>
            {l.aha && (
              <div className="small p-2 rounded" style={{ backgroundColor: ACCENT5 + '10', color: ACCENT5 }}>
                <strong>💡 AHA:</strong> {l.aha}
              </div>
            )}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { concepts, thresholds, standards, references } = data;
  return (
    <div>
      <SectionCard title="Key Concepts (15)" borderColor={ACCENT}>
        {(concepts || []).map((c, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ backgroundColor: '#f0f8ff', border: `1px solid ${ACCENT}20` }}>
            <strong style={{ color: ACCENT }}>{c.term}</strong>
            <div className="small text-muted mt-1">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds" borderColor={ACCENT3}>
        <div className="row g-2">
          {Object.entries(thresholds || {}).map(([k, v]) => (
            <div key={k} className="col-md-6 col-lg-4">
              <div className="p-2 rounded border small">
                <div className="text-muted">{k.replace(/_/g, ' ')}</div>
                <div className="fw-bold" style={{ color: ACCENT3 }}>{String(v)}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Clinical Standards" borderColor={ACCENT}>
        <ul className="small mb-0">
          {(standards || []).map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="References" borderColor={ACCENT}>
        <ol className="small mb-0">
          {(references || []).map((r, i) => <li key={i}>{r}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────
export default function KCNQ5Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/kcnq5/overview`).then(r => r.json()),
      fetch(`${API}/api/kcnq5/breakdown`).then(r => r.json()),
      fetch(`${API}/api/kcnq5/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(String(e)));
  }, []);

  if (error) return <div className="alert alert-danger m-4">{error}</div>;

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4 p-4 rounded text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT5} 100%)` }}>
        <h2 className="mb-1">⚡ KCNQ5 Epilepsy</h2>
        <div className="opacity-90 small">
          DEE / ID-Epilepsy / Kv7.5 M-Current / Interneuron-Enriched / GOF-LOF / 6q14.1
          &nbsp;·&nbsp;No BFNS (onset 6–24M)&nbsp;·&nbsp;LOF paradoxical CBZ/OXC worsening ~20%
          &nbsp;·&nbsp;POLG mandatory before VPA&nbsp;·&nbsp;HLA-B*15:02 mandatory before CBZ/OXC
          &nbsp;·&nbsp;Tiagabine ABSOLUTE CI&nbsp;·&nbsp;Retigabine WITHDRAWN 2017
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
