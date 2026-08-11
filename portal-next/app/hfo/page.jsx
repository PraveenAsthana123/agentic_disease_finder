'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'electrodes',  label: 'Electrode Map' },
  { id: 'patients',    label: 'Per Patient' },
  { id: 'preictal',    label: 'Pre-Ictal vs Interictal' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function SOZBadge({ likely }) {
  return (
    <span className={`badge bg-${likely ? 'danger' : 'secondary'}`}>
      {likely ? '⚠ SOZ Likely' : 'Normal'}
    </span>
  );
}

function Bar({ value, max, color }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="progress" style={{ height: 12 }}>
      <div className={`progress-bar bg-${color || 'primary'}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const topPats = data.per_patient_kpis || [];
  const region  = data.region_soz_counts || {};
  const methods = data.detection_methods || [];
  const aed     = data.aed_hfo_response || [];

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Patients Analysed"     value={data.n_patients}               color="info"    sub="41-patient cohort" />
        <KPI label="Total Ripple Events"   value={data.total_ripple_events?.toLocaleString()} color="warning" sub="30-min session × 19 ch" />
        <KPI label="Total Fast Ripple Events" value={data.total_fr_events?.toLocaleString()} color="danger" sub="250–500 Hz, pathological" />
        <KPI label="SOZ Patients"          value={`${data.soz_patients} (${data.soz_patients_pct}%)`} color="danger" sub="≥2 SOZ-flagged channels" />
      </div>
      <div className="row mb-4">
        <KPI label="Avg Ripple Rate (cohort)" value={`${data.avg_ripple_rate_cohort} ev/min`} color="primary" sub="mean across 19 electrodes" />
        <KPI label="Avg Fast Ripple Rate"     value={`${data.avg_fr_rate_cohort} ev/min`} color="danger" sub="per electrode, per patient" />
        <KPI label="High-FR Patients"         value={`${data.high_fr_patients} (${data.high_fr_pct}%)`} color="warning" sub="avg FR > 1.5 ev/min" />
        <KPI label="Analyses Processed"       value={data.n_analyses} color="secondary" sub="total EEG segments" />
      </div>

      {/* Region SOZ heatmap */}
      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">SOZ Channels by Brain Region (cohort)</div>
            <div className="card-body">
              {Object.entries(region).sort((a, b) => b[1] - a[1]).map(([reg, cnt]) => (
                <div key={reg} className="mb-2">
                  <div className="d-flex justify-content-between mb-1 small">
                    <span>{reg}</span><strong>{cnt} ch</strong>
                  </div>
                  <Bar value={cnt} max={Math.max(...Object.values(region))} color="danger" />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Detection method table */}
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Detection Method Comparison</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr>
                    <th>Method</th><th>Sens %</th><th>Spec %</th><th>Throughput</th>
                  </tr></thead>
                  <tbody>
                    {methods.map((m, i) => (
                      <tr key={i}>
                        <td className="small">{m.method}</td>
                        <td><span className="badge bg-success">{m.sensitivity}</span></td>
                        <td><span className="badge bg-info">{m.specificity}</span></td>
                        <td className="small text-muted">{m.throughput}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* AED HFO response */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">AED Effect on HFO Rate (published benchmarks)</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-striped mb-0">
              <thead><tr>
                <th>AED</th><th>Ripple Change %</th><th>Fast Ripple Change %</th><th>Studies</th>
              </tr></thead>
              <tbody>
                {aed.map((a, i) => (
                  <tr key={i}>
                    <td><strong>{a.aed}</strong></td>
                    <td className={a.ripple_change_pct < -25 ? 'text-success fw-bold' : 'text-warning'}>
                      {a.ripple_change_pct}%
                    </td>
                    <td className={a.fr_change_pct < -25 ? 'text-success fw-bold' : 'text-warning'}>
                      {a.fr_change_pct}%
                    </td>
                    <td className="text-muted">{a.n_studies}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Top patients by FR rate */}
      <div className="card shadow-sm">
        <div className="card-header fw-semibold">Top Patients by Fast Ripple Rate</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm mb-0">
              <thead><tr>
                <th>Patient</th><th>Epilepsy Type</th><th>Ripple (ev/min)</th>
                <th>Fast Ripple (ev/min)</th><th>SOZ Channels</th><th>Peak Electrode</th>
              </tr></thead>
              <tbody>
                {topPats.slice(0, 15).map((p, i) => (
                  <tr key={i}>
                    <td className="font-monospace small">{p.patient_id}</td>
                    <td className="small text-muted">{p.epilepsy_type}</td>
                    <td><span className="badge bg-warning text-dark">{p.avg_ripple_rate}</span></td>
                    <td><span className={`badge bg-${p.avg_fr_rate > 1.5 ? 'danger' : 'secondary'}`}>{p.avg_fr_rate}</span></td>
                    <td>{p.soz_channels}</td>
                    <td className="font-monospace">{p.top_electrode}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function ElectrodePanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const electrodes = data.electrode_averages || [];
  const rippleHist = data.ripple_histogram || [];
  const frHist     = data.fr_histogram || [];

  const maxTotal = Math.max(...electrodes.map(e => e.avg_total), 1);

  return (
    <div>
      <div className="row mb-4">
        {/* Electrode HFO rate table */}
        <div className="col-md-7">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Per-Electrode HFO Rates (cohort avg, events/min)</div>
            <div className="card-body p-0">
              <div className="table-responsive" style={{ maxHeight: 500, overflowY: 'auto' }}>
                <table className="table table-sm mb-0">
                  <thead className="table-dark sticky-top">
                    <tr><th>Electrode</th><th>Region</th><th>Ripple</th><th>Fast Ripple</th><th>Total</th><th>SOZ %</th><th>Rate</th></tr>
                  </thead>
                  <tbody>
                    {electrodes.map((e, i) => (
                      <tr key={i} className={e.avg_fr > 2.0 ? 'table-danger' : e.avg_fr > 1.0 ? 'table-warning' : ''}>
                        <td className="font-monospace fw-bold">{e.electrode}</td>
                        <td className="small text-muted">{e.region}</td>
                        <td>{e.avg_ripple}</td>
                        <td className={e.avg_fr > 2.0 ? 'text-danger fw-bold' : ''}>{e.avg_fr}</td>
                        <td><strong>{e.avg_total}</strong></td>
                        <td><span className={`badge bg-${e.soz_pct > 30 ? 'danger' : 'secondary'}`}>{e.soz_pct}%</span></td>
                        <td style={{ width: 120 }}><Bar value={e.avg_total} max={maxTotal} color={e.avg_fr > 2.0 ? 'danger' : 'warning'} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        {/* Histograms */}
        <div className="col-md-5">
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">Ripple Rate Distribution (events/min)</div>
            <div className="card-body">
              {rippleHist.map((b, i) => (
                <div key={i} className="mb-1">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>{b.range}</span><span>{b.count}</span>
                  </div>
                  <Bar value={b.count} max={Math.max(...rippleHist.map(x => x.count), 1)} color="warning" />
                </div>
              ))}
            </div>
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Fast Ripple Rate Distribution (events/min)</div>
            <div className="card-body">
              {frHist.map((b, i) => (
                <div key={i} className="mb-1">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>{b.range}</span><span>{b.count}</span>
                  </div>
                  <Bar value={b.count} max={Math.max(...frHist.map(x => x.count), 1)} color="danger" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function PatientPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const profiles = data.per_patient_profiles || [];
  const [sel, setSel] = useState(0);
  const p = profiles[sel] || {};
  const emap = p.electrode_map || [];
  const maxRate = Math.max(...emap.map(e => e.total_rate), 1);

  return (
    <div>
      <div className="row mb-3">
        <div className="col-md-3">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold small">Select Patient</div>
            <div className="list-group list-group-flush" style={{ maxHeight: 500, overflowY: 'auto' }}>
              {profiles.map((pr, i) => (
                <button
                  key={i}
                  className={`list-group-item list-group-item-action small ${i === sel ? 'active' : ''}`}
                  onClick={() => setSel(i)}
                >
                  <div className="d-flex justify-content-between">
                    <span className="font-monospace">{pr.patient_id}</span>
                    <SOZBadge likely={pr.n_soz >= 2} />
                  </div>
                  <div className="text-muted" style={{ fontSize: '0.7rem' }}>
                    FR: {pr.electrode_map?.[0]?.fr_rate ?? '—'} ev/min · SOZ: {pr.n_soz} ch
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-9">
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">
              Patient: <span className="font-monospace">{p.patient_id}</span>
              &nbsp;·&nbsp; Confidence: {(p.confidence * 100)?.toFixed(1)}%
              &nbsp;·&nbsp; Peak: {p.peak_electrode} ({p.peak_total_rate} ev/min)
            </div>
            <div className="card-body">
              {p.soz_channels?.length > 0 && (
                <div className="alert alert-danger py-2 mb-3">
                  <strong>⚠ Likely SOZ Channels:</strong> {p.soz_channels.join(', ')}
                </div>
              )}
              <div className="table-responsive" style={{ maxHeight: 380, overflowY: 'auto' }}>
                <table className="table table-sm mb-0">
                  <thead className="table-dark sticky-top">
                    <tr><th>Electrode</th><th>Region</th><th>Ripple</th><th>Fast Ripple</th><th>Total</th><th>SOZ</th><th>Bar</th></tr>
                  </thead>
                  <tbody>
                    {emap.map((e, i) => (
                      <tr key={i} className={e.soz_likely ? 'table-danger' : ''}>
                        <td className="font-monospace fw-bold">{e.electrode}</td>
                        <td className="small text-muted">{e.region}</td>
                        <td>{e.ripple_rate}</td>
                        <td className={e.fr_rate > 2.0 ? 'text-danger fw-bold' : ''}>{e.fr_rate}</td>
                        <td><strong>{e.total_rate}</strong></td>
                        <td><SOZBadge likely={e.soz_likely} /></td>
                        <td style={{ width: 120 }}><Bar value={e.total_rate} max={maxRate} color={e.soz_likely ? 'danger' : 'warning'} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function PreictalPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const rows = data.preictal_vs_interictal || [];
  const maxRipple = Math.max(...rows.flatMap(r => [r.interictal_ripple, r.preictal_ripple]), 1);
  const maxFR     = Math.max(...rows.flatMap(r => [r.interictal_fr,    r.preictal_fr]),    1);

  return (
    <div>
      <div className="alert alert-info mb-3">
        <strong>Pre-ictal HFO surge:</strong> HFO rates increase significantly in the 2–15 seconds
        before seizure onset, especially fast ripples (250–500 Hz). This pre-ictal surge enables
        closed-loop stimulation systems to abort seizures before clinical onset.
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Pre-Ictal vs Interictal HFO Rates per Patient</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm mb-0">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Interictal Ripple</th><th>Pre-Ictal Ripple</th><th>Ripple Ratio</th>
                  <th>Interictal FR</th><th>Pre-Ictal FR</th><th>FR Ratio</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>
                    <td className="font-monospace small">{r.patient_id}</td>
                    <td>{r.interictal_ripple}</td>
                    <td className="text-warning fw-bold">{r.preictal_ripple}</td>
                    <td><span className={`badge bg-${r.ripple_ratio > 2 ? 'warning text-dark' : 'secondary'}`}>×{r.ripple_ratio}</span></td>
                    <td>{r.interictal_fr}</td>
                    <td className="text-danger fw-bold">{r.preictal_fr}</td>
                    <td><span className={`badge bg-${r.fr_ratio > 2 ? 'danger' : 'secondary'}`}>×{r.fr_ratio}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Ripple Rate: Interictal vs Pre-Ictal (per patient)</div>
            <div className="card-body">
              {rows.map((r, i) => (
                <div key={i} className="mb-2">
                  <div className="small text-muted mb-1">{r.patient_id}</div>
                  <div className="d-flex align-items-center gap-2 mb-1">
                    <span style={{ width: 70 }} className="text-muted small">Interictal</span>
                    <div style={{ flex: 1 }}><Bar value={r.interictal_ripple} max={maxRipple} color="secondary" /></div>
                    <span className="small">{r.interictal_ripple}</span>
                  </div>
                  <div className="d-flex align-items-center gap-2">
                    <span style={{ width: 70 }} className="text-warning small">Pre-Ictal</span>
                    <div style={{ flex: 1 }}><Bar value={r.preictal_ripple} max={maxRipple} color="warning" /></div>
                    <span className="small">{r.preictal_ripple}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Fast Ripple Rate: Interictal vs Pre-Ictal</div>
            <div className="card-body">
              {rows.map((r, i) => (
                <div key={i} className="mb-2">
                  <div className="small text-muted mb-1">{r.patient_id}</div>
                  <div className="d-flex align-items-center gap-2 mb-1">
                    <span style={{ width: 70 }} className="text-muted small">Interictal</span>
                    <div style={{ flex: 1 }}><Bar value={r.interictal_fr} max={maxFR} color="secondary" /></div>
                    <span className="small">{r.interictal_fr}</span>
                  </div>
                  <div className="d-flex align-items-center gap-2">
                    <span style={{ width: 70 }} className="text-danger small">Pre-Ictal</span>
                    <div style={{ flex: 1 }}><Bar value={r.preictal_fr} max={maxFR} color="danger" /></div>
                    <span className="small">{r.preictal_fr}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const types    = data.hfo_types || [];
  const pipeline = data.detection_pipeline || [];
  const glossary = data.glossary || [];
  const refs     = data.references || [];
  const outcome  = data.surgical_outcome || {};

  return (
    <div>
      {/* HFO type cards */}
      <div className="row mb-4">
        {types.map((t, i) => (
          <div className="col-md-6 mb-3" key={i}>
            <div className={`card shadow-sm border-${i === 0 ? 'warning' : 'danger'} h-100`}>
              <div className={`card-header fw-bold text-${i === 0 ? 'warning' : 'danger'}`}>
                {t.type} ({t.frequency})
              </div>
              <div className="card-body small">
                <table className="table table-sm mb-0">
                  <tbody>
                    <tr><th>Duration</th><td>{t.duration}</td></tr>
                    <tr><th>Amplitude</th><td>{t.amplitude}</td></tr>
                    <tr><th>Origin</th><td>{t.origin}</td></tr>
                    <tr><th>Physiology</th><td>{t.physiology}</td></tr>
                    <tr><th>Clinical Use</th><td>{t.clinical}</td></tr>
                    <tr><th>Sensitivity</th><td>{t.sensitivity}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Surgical outcome */}
      <div className="card shadow-sm mb-4 border-success">
        <div className="card-header fw-semibold text-success">Surgical Outcome — HFO-Guided vs Standard Resection</div>
        <div className="card-body small">
          <p>{outcome.description}</p>
          <div className="row">
            <div className="col-md-4 text-center">
              <div className="h3 text-success fw-bold">{outcome.seizure_freedom_hfo_guided}</div>
              <div className="text-muted">Seizure Freedom — HFO-Guided</div>
            </div>
            <div className="col-md-4 text-center">
              <div className="h3 text-warning fw-bold">{outcome.seizure_freedom_standard}</div>
              <div className="text-muted">Seizure Freedom — Standard Mapping</div>
            </div>
            <div className="col-md-4 text-center text-muted">
              <div className="small mt-2">{outcome.reference}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Detection pipeline */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">HFO Detection Pipeline (6 steps)</div>
        <div className="card-body">
          {pipeline.map((s, i) => (
            <div key={i} className="d-flex mb-2">
              <div className="me-3">
                <span className="badge bg-primary rounded-circle">{s.step}</span>
              </div>
              <div>
                <strong>{s.name}</strong>
                <div className="text-muted small">{s.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Glossary */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Glossary</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {glossary.map((g, i) => (
                <tr key={i}><td><code>{g.term}</code></td><td className="small">{g.definition}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* References */}
      <div className="card shadow-sm">
        <div className="card-header fw-semibold">References</div>
        <div className="card-body">
          <ol className="mb-0 small">
            {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
          </ol>
        </div>
      </div>
    </div>
  );
}

export default function HFOPage() {
  const [tab, setTab]         = useState('overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hfo/overview`).then(r => r.json()).then(setOverview).catch(e => setOverview({ error: String(e) }));
    fetch(`${API}/api/hfo/breakdown`).then(r => r.json()).then(setBreakdown).catch(e => setBreakdown({ error: String(e) }));
  }, []);

  const [defs, setDefs] = useState(null);
  useEffect(() => {
    if (tab === 'definitions' && !defs) {
      fetch(`${API}/api/hfo/definitions`).then(r => r.json()).then(setDefs).catch(e => setDefs({ error: String(e) }));
    }
  }, [tab, defs]);

  return (
    <div className="container-fluid py-4">
      <div className="mb-4">
        <h2 className="fw-bold">⚡ High-Frequency Oscillations (HFO) Dashboard</h2>
        <p className="text-muted mb-0">
          Ripple (80–250 Hz) &amp; Fast Ripple (250–500 Hz) biomarker analytics for epileptic zone localization.
          Cohort: {overview?.n_patients ?? '…'} patients · {overview?.n_analyses ?? '…'} EEG analyses ·
          SOZ-flagged: {overview?.soz_patients ?? '…'} patients ({overview?.soz_patients_pct ?? '…'}%).
        </p>
      </div>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel    data={overview}   />}
      {tab === 'electrodes'  && <ElectrodePanel   data={breakdown}  />}
      {tab === 'patients'    && <PatientPanel      data={breakdown}  />}
      {tab === 'preictal'    && <PreictalPanel     data={breakdown}  />}
      {tab === 'definitions' && <DefinitionsPanel  data={defs}       />}
    </div>
  );
}
