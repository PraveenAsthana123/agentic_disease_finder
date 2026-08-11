'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'power',       label: 'Power Comparison' },
  { id: 'patients',    label: 'Per Patient' },
  { id: 'electrodes',  label: 'Electrode Spikes' },
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

function BudgetBadge({ ok }) {
  return <span className={`badge bg-${ok ? 'success' : 'danger'}`}>{ok ? '✓ Within Budget' : '✗ Exceeds Budget'}</span>;
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const lif = data.lif_neuron_config || {};
  const spikeStats = data.spike_rate_stats || {};
  const topPats = data.top_active_patients || [];

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Patients Analysed" value={data.total_patients_analyzed} color="info" sub="SNN inference run" />
        <KPI label="Total Analyses" value={data.total_analyses} color="primary" sub="EEG segments processed" />
        <KPI label="Inference Latency" value={`${data.snn_inference_latency_ms} ms`} color="success" sub="Loihi-2 SNN" />
        <KPI label="Mean Power" value={`${data.snn_mean_power_uw} µW`} color="warning" sub={<BudgetBadge ok={data.within_implant_budget} />} />
      </div>

      <div className="row mb-4">
        <KPI label="Mean Spike Rate" value={`${spikeStats.mean_hz} Hz`} color="secondary" sub={`${spikeStats.min_hz}–${spikeStats.max_hz} Hz range`} />
        <KPI label="Temporal Efficiency" value={`${(data.mean_temporal_coding_efficiency * 100).toFixed(1)}%`} color="info" sub="vs ANN rate coding" />
        <KPI label="Implant Budget" value={`${data.implant_power_budget_mw} mW`} color="success" sub="max allowable" />
        <KPI label="Montage" value="10-20" color="secondary" sub={data.montage || '19 electrodes'} />
      </div>

      {/* Efficiency interpretation */}
      <div className="alert alert-info mb-4">
        <strong>Efficiency: </strong>{data.efficiency_interpretation}
      </div>

      {/* LIF Config */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">LIF Neuron Configuration (Loihi-2)</div>
        <div className="card-body">
          <div className="row">
            {Object.entries(lif).map(([k, v]) => (
              <div key={k} className="col-6 col-md-4 mb-2">
                <span className="text-muted small">{k.replace(/_/g, ' ')}: </span>
                <strong>{v}</strong>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Top Active Patients */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">Top Active Patients by Spike Rate</div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Patient</th>
                <th>Label</th>
                <th>Analyses</th>
                <th>Spike Rate (Hz)</th>
                <th>Power (µW)</th>
                <th>Efficiency</th>
                <th>Refractory %</th>
              </tr>
            </thead>
            <tbody>
              {topPats.map(p => (
                <tr key={p.patient_id}>
                  <td><code>{p.patient_id}</code></td>
                  <td><span className={`badge bg-${p.predicted_label === 'Epilepsy' ? 'danger' : 'secondary'}`}>{p.predicted_label || '—'}</span></td>
                  <td>{p.n_analyses}</td>
                  <td><strong>{p.mean_spike_rate_hz?.toFixed(1)}</strong></td>
                  <td>{p.snn_power_uw?.toFixed(3)}</td>
                  <td>{(p.temporal_coding_efficiency * 100).toFixed(1)}%</td>
                  <td>{p.refractory_pct?.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="text-muted small">Chip: {data.neuromorphic_chip}</div>
    </div>
  );
}

function PowerPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const models = data.model_comparison_table || [];
  const maxPower = Math.max(...models.map(m => m.power_mw), 1);

  return (
    <div>
      <div className="card mb-4">
        <div className="card-header fw-semibold">Model Power &amp; Latency Comparison (EEG Seizure Detection)</div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Model</th>
                <th>Power (mW)</th>
                <th>Latency (ms)</th>
                <th>Accuracy (%)</th>
                <th>Memory (KB)</th>
                <th>Hardware</th>
                <th>Use Case</th>
              </tr>
            </thead>
            <tbody>
              {models.map(m => (
                <tr key={m.model} className={m.model.startsWith('SNN') ? 'table-success' : ''}>
                  <td><strong>{m.model}</strong></td>
                  <td>
                    <div className="d-flex align-items-center gap-2">
                      <div style={{ width: 80, background: '#eee', borderRadius: 4, height: 10 }}>
                        <div style={{ width: `${(m.power_mw / maxPower) * 100}%`, background: m.model.startsWith('SNN') ? '#198754' : '#0d6efd', height: '100%', borderRadius: 4 }} />
                      </div>
                      <span>{m.power_mw}</span>
                    </div>
                  </td>
                  <td>{m.latency_ms}</td>
                  <td>{m.accuracy_pct}</td>
                  <td>{m.memory_kb}</td>
                  <td><small>{m.hardware}</small></td>
                  <td><small className="text-muted">{m.suitable_for}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card-footer text-muted small">
          Highlighted row = SNN (Loihi-2). SNN achieves {(models[0] && models[1]) ? Math.round(models[1].power_mw / (models[0].power_mw || 0.5)) + '×' : '90×'} lower power than CNN at {Math.round((models[0]?.latency_ms / (models[1]?.latency_ms || 18)) * 100)}% of CNN latency.
        </div>
      </div>

      {/* Power bar chart */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">Relative Power Consumption</div>
        <div className="card-body">
          {models.map(m => (
            <div key={m.model} className="mb-3">
              <div className="d-flex justify-content-between mb-1">
                <small>{m.model}</small>
                <small><strong>{m.power_mw} mW</strong></small>
              </div>
              <div style={{ background: '#eee', borderRadius: 4, height: 20 }}>
                <div
                  style={{
                    width: `${Math.max((m.power_mw / maxPower) * 100, 0.5)}%`,
                    background: m.model.startsWith('SNN') ? '#198754' : '#0d6efd',
                    height: '100%',
                    borderRadius: 4,
                    minWidth: 4,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Accuracy vs Power scatter description */}
      <div className="alert alert-success">
        <strong>Clinical takeaway:</strong> SNN on Intel Loihi-2 uses ~0.5 mW at 88.5% accuracy with 2 ms latency — meeting all implantable device constraints (power &lt; 1 mW, latency &lt; 10 ms). Transformer achieves highest accuracy (93.4%) but at 680 mW — suitable only for clinical workstations.
      </div>
    </div>
  );
}

function PatientsPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const patients = data.patient_snn_metrics || [];
  const [sort, setSort] = useState('mean_spike_rate_hz');
  const [dir, setDir] = useState(-1);

  const sorted = [...patients].sort((a, b) => dir * ((a[sort] ?? 0) - (b[sort] ?? 0)));

  const toggleSort = col => {
    if (sort === col) setDir(d => -d);
    else { setSort(col); setDir(-1); }
  };

  const Th = ({ col, label }) => (
    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort(col)}>
      {label}{sort === col ? (dir === 1 ? ' ▲' : ' ▼') : ''}
    </th>
  );

  const maxRate = Math.max(...patients.map(p => p.mean_spike_rate_hz || 0), 1);

  return (
    <div>
      <div className="card mb-4">
        <div className="card-header fw-semibold">Per-Patient SNN Metrics ({patients.length} patients)</div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Patient</th>
                <th>Label</th>
                <Th col="mean_spike_rate_hz" label="Spike Rate (Hz)" />
                <Th col="membrane_potential_norm" label="V_mem (norm)" />
                <Th col="snn_power_uw" label="Power (µW)" />
                <Th col="temporal_coding_efficiency" label="TCE" />
                <Th col="refractory_pct" label="Refrac %" />
                <Th col="n_analyses" label="Analyses" />
              </tr>
            </thead>
            <tbody>
              {sorted.map(p => {
                const rateWidth = `${(p.mean_spike_rate_hz / maxRate) * 100}%`;
                return (
                  <tr key={p.patient_id}>
                    <td><code>{p.patient_id}</code><div className="text-muted" style={{ fontSize: '0.7rem' }}>{p.name !== p.patient_id ? p.name : ''}</div></td>
                    <td><span className={`badge bg-${p.predicted_label === 'Epilepsy' ? 'danger' : 'secondary'}`}>{p.predicted_label || '—'}</span></td>
                    <td>
                      <div style={{ background: '#eee', borderRadius: 3, height: 8, width: 80 }}>
                        <div style={{ width: rateWidth, background: '#0d6efd', height: '100%', borderRadius: 3 }} />
                      </div>
                      <small>{p.mean_spike_rate_hz?.toFixed(1)}</small>
                    </td>
                    <td>{p.membrane_potential_norm?.toFixed(3)}</td>
                    <td>{p.snn_power_uw?.toFixed(3)}</td>
                    <td>{(p.temporal_coding_efficiency * 100).toFixed(2)}%</td>
                    <td>{p.refractory_pct?.toFixed(1)}%</td>
                    <td>{p.n_analyses}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="card-footer text-muted small">Click column headers to sort. TCE = temporal coding efficiency vs ANN rate-coded baseline.</div>
      </div>
    </div>
  );
}

function ElectrodesPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const maps = (data.patient_electrode_maps || []).slice(0, 6);
  const [sel, setSel] = useState(0);
  const map = maps[sel];

  if (!map) return <div className="text-muted p-3">No electrode data available.</div>;

  const electrodes = map.electrode_spike_rates || [];
  const maxRate = Math.max(...electrodes.map(e => e.spike_rate_hz || 0), 1);

  return (
    <div>
      {/* Patient selector */}
      <div className="mb-3">
        <label className="form-label fw-semibold">Select Patient:</label>
        <select className="form-select" value={sel} onChange={e => setSel(+e.target.value)}>
          {maps.map((m, i) => (
            <option key={i} value={i}>{m.patient_id} — {m.seizure_type}</option>
          ))}
        </select>
      </div>

      <div className="row mb-3">
        <div className="col-md-8">
          <div className="card mb-4">
            <div className="card-header fw-semibold">
              Electrode Spike Rates — {map.patient_id}
              <span className="ms-2 badge bg-secondary">{map.seizure_type}</span>
            </div>
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Electrode</th>
                    <th>Region</th>
                    <th>Spike Rate (Hz)</th>
                    <th>Ictal?</th>
                  </tr>
                </thead>
                <tbody>
                  {electrodes.map(e => (
                    <tr key={e.electrode} className={e.is_ictal_region ? 'table-danger' : ''}>
                      <td><strong>{e.electrode}</strong></td>
                      <td><small>{e.region}</small></td>
                      <td>
                        <div style={{ background: '#eee', borderRadius: 3, height: 8, width: 100 }}>
                          <div style={{
                            width: `${(e.spike_rate_hz / maxRate) * 100}%`,
                            background: e.is_ictal_region ? '#dc3545' : '#0d6efd',
                            height: '100%', borderRadius: 3,
                          }} />
                        </div>
                        <small>{e.spike_rate_hz?.toFixed(1)}</small>
                      </td>
                      <td>{e.is_ictal_region ? <span className="badge bg-danger">Ictal</span> : <span className="badge bg-light text-dark">—</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card mb-3">
            <div className="card-header fw-semibold">Region Summary</div>
            <div className="card-body p-2">
              {['Frontal', 'Temporal', 'Central', 'Parietal', 'Occipital'].map(region => {
                const group = electrodes.filter(e => e.region && e.region.startsWith(region));
                if (!group.length) return null;
                const avg = group.reduce((s, e) => s + (e.spike_rate_hz || 0), 0) / group.length;
                const ictal = group.filter(e => e.is_ictal_region).length;
                return (
                  <div key={region} className="mb-2">
                    <div className="d-flex justify-content-between">
                      <small><strong>{region}</strong></small>
                      <small>{avg.toFixed(1)} Hz avg</small>
                    </div>
                    {ictal > 0 && <small className="text-danger">{ictal} ictal electrode{ictal > 1 ? 's' : ''}</small>}
                    <div style={{ background: '#eee', borderRadius: 3, height: 6 }}>
                      <div style={{ width: `${(avg / maxRate) * 100}%`, background: ictal ? '#dc3545' : '#0d6efd', height: '100%', borderRadius: 3 }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="alert alert-info p-2">
            <small><strong>Ictal electrodes</strong> (red) show elevated spike rates during seizure propagation. SNN detects these patterns in real-time at 2 ms latency.</small>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const defs = data.definitions || [];
  const [q, setQ] = useState('');
  const filtered = q ? defs.filter(d => d.term.toLowerCase().includes(q.toLowerCase()) || d.definition.toLowerCase().includes(q.toLowerCase())) : defs;

  return (
    <div>
      <div className="mb-3">
        <input className="form-control" placeholder="Search terms…" value={q} onChange={e => setQ(e.target.value)} />
      </div>
      <p className="text-muted small">{filtered.length} of {defs.length} terms shown</p>
      {filtered.map((d, i) => (
        <div key={i} className="card mb-2">
          <div className="card-body py-2">
            <div className="fw-semibold text-primary mb-1">{d.term}</div>
            <div className="small text-muted">{d.definition}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function SNNNeuromorphicPage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/snn-neuromorphic/overview`).then(r => r.json()).then(setOverview).catch(e => setOverview({ error: String(e) }));
    fetch(`${API}/api/snn-neuromorphic/breakdown`).then(r => r.json()).then(setBreakdown).catch(e => setBreakdown({ error: String(e) }));
    fetch(`${API}/api/snn-neuromorphic/definitions`).then(r => r.json()).then(setDefinitions).catch(e => setDefinitions({ error: String(e) }));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <div>
          <h4 className="mb-0">&#x26a1; SNN Neuromorphic Dashboard</h4>
          <small className="text-muted">Spiking Neural Network · Intel Loihi-2 · Implantable EEG Seizure Detection</small>
        </div>
        <span className="ms-auto badge bg-success">Live Data</span>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel data={overview} />}
      {tab === 'power'       && <PowerPanel data={overview} />}
      {tab === 'patients'    && <PatientsPanel data={overview} />}
      {tab === 'electrodes'  && <ElectrodesPanel data={breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel data={definitions} />}
    </div>
  );
}
