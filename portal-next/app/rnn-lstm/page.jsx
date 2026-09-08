'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',   label: 'Overview' },
  { id: 'sequences',  label: 'Sequence Inventory' },
  { id: 'band-power', label: 'Band Power' },
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

function ConfBar({ pct, color }) {
  const c = color || (pct >= 80 ? 'success' : pct >= 60 ? 'info' : pct >= 40 ? 'warning' : 'danger');
  return (
    <div className="progress" style={{ height: 14, borderRadius: 6 }}>
      <div className={`progress-bar bg-${c}`} style={{ width: `${Math.min(pct, 100)}%`, borderRadius: 6, transition: 'width 0.5s' }} />
    </div>
  );
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;

  const kpis = ov.kpis || [];
  const qualDist = ov.quality_distribution || [];
  const classDist = ov.classification_chart || [];
  const daily = ov.daily_activity || [];

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-4">
        {kpis.slice(0, 8).map((k, i) => (
          <KPI key={i} label={k.label} value={k.value} color={['primary','success','info','warning','secondary','dark','danger','primary'][i]} sub={k.sub} />
        ))}
      </div>

      <div className="row mb-4">
        {/* Signal Quality Distribution */}
        <div className="col-md-5 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white"><strong>Signal Quality Distribution</strong></div>
            <div className="card-body">
              {qualDist.map((q, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-semibold">{q.quality}</span>
                    <span className="small text-muted">{q.count}</span>
                  </div>
                  <div className="progress" style={{ height: 12, borderRadius: 6 }}>
                    <div
                      className="progress-bar"
                      style={{
                        width: `${Math.round(q.count / (ov.total_sequences || 133) * 100)}%`,
                        backgroundColor: q.color,
                        borderRadius: 6,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Classification Results */}
        <div className="col-md-7 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white"><strong>Classification Results by Disease</strong></div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-dark">
                  <tr><th>Predicted Label</th><th>Count</th><th>Mean Confidence</th><th>Bar</th></tr>
                </thead>
                <tbody>
                  {classDist.map((c, i) => {
                    const pct = Math.round(c.mean_confidence * 100);
                    return (
                      <tr key={i}>
                        <td>{c.predicted_label}</td>
                        <td className="fw-bold">{c.count}</td>
                        <td>{pct}%</td>
                        <td style={{ width: '30%' }}><ConfBar pct={pct} /></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Daily Pipeline Activity */}
      {daily.length > 0 && (
        <div className="card shadow-sm mb-4">
          <div className="card-header py-2 bg-dark text-white"><strong>Daily Pipeline Activity (last 14 days)</strong></div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-dark">
                <tr><th>Date</th><th>Analyses</th><th>Seizure Events</th></tr>
              </thead>
              <tbody>
                {daily.slice(0, 14).map((d, i) => (
                  <tr key={i}>
                    <td>{d.date}</td>
                    <td>{d.total_analyses}</td>
                    <td>{d.seizure_events ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Metadata summary */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-secondary text-white"><strong>Pipeline Summary</strong></div>
        <div className="card-body">
          <div className="row">
            <div className="col-md-6">
              <ul className="list-unstyled mb-0">
                <li><strong>Total Sequences:</strong> {ov.total_sequences}</li>
                <li><strong>Patients Analyzed:</strong> {ov.patients_analyzed}</li>
                <li><strong>RNN/LSTM Architectures:</strong> {ov.n_architectures}</li>
                <li><strong>Frequency Bands:</strong> {ov.total_freq_bands}</li>
              </ul>
            </div>
            <div className="col-md-6">
              <ul className="list-unstyled mb-0">
                <li><strong>Mean Confidence:</strong> {(ov.mean_confidence * 100).toFixed(1)}%</li>
                <li><strong>Mean Duration:</strong> {ov.mean_duration?.toFixed(0)}s</li>
                <li><strong>Sampling Rate:</strong> {ov.mean_sampling_rate} Hz</li>
                <li><strong>Seizure Events:</strong> {ov.seizure_events}</li>
              </ul>
            </div>
          </div>
          {ov.seq_length_label && (
            <div className="mt-2 p-2 bg-light rounded">
              <span className="badge bg-info text-dark me-2">Seq Length</span>
              <span className="small text-muted">{ov.seq_length_label}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SequencesPanel({ bk }) {
  const [filter, setFilter] = useState('');
  if (!bk) return <div className="text-muted p-3">Loading…</div>;

  const inv = bk.sequence_inventory || [];
  const filtered = filter
    ? inv.filter(s =>
        s.disease?.toLowerCase().includes(filter.toLowerCase()) ||
        s.predicted_label?.toLowerCase().includes(filter.toLowerCase()) ||
        s.patient_id?.toLowerCase().includes(filter.toLowerCase())
      )
    : inv;

  return (
    <div>
      <div className="mb-3">
        <input
          className="form-control form-control-sm"
          placeholder="Filter by disease, patient or label…"
          value={filter}
          onChange={e => setFilter(e.target.value)}
          style={{ maxWidth: 340 }}
        />
      </div>
      <div className="card shadow-sm">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Sequence Inventory</strong>
          <span className="badge bg-secondary ms-2">{filtered.length} / {inv.length}</span>
        </div>
        <div className="card-body p-0" style={{ maxHeight: 520, overflowY: 'auto' }}>
          <table className="table table-sm table-hover mb-0">
            <thead className="table-dark" style={{ position: 'sticky', top: 0 }}>
              <tr>
                <th>ID</th><th>Patient</th><th>Disease</th>
                <th>Predicted</th><th>Confidence</th><th>Quality</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((s, i) => {
                const pct = Math.round(s.confidence * 100);
                const qColor = { Excellent: 'success', Good: 'info', Fair: 'warning', Poor: 'danger' }[s.signal_quality] || 'secondary';
                return (
                  <tr key={i}>
                    <td className="text-muted small">{s.id}</td>
                    <td className="fw-semibold">{s.patient_id}</td>
                    <td className="text-capitalize">{s.disease?.replace(/_/g, ' ')}</td>
                    <td>{s.predicted_label}</td>
                    <td>
                      <div className="d-flex align-items-center gap-1">
                        <span className="small">{pct}%</span>
                        <div className="flex-grow-1"><ConfBar pct={pct} /></div>
                      </div>
                    </td>
                    <td><span className={`badge bg-${qColor}`}>{s.signal_quality}</span></td>
                    <td className="text-muted small">{s.created_at?.slice(0, 10)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function BandPowerPanel({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;
  const bands = ov.band_power_chart || [];
  const maxPower = Math.max(...bands.map(b => b.mean_power), 0.001);

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>EEG Frequency Band Power</strong>
          <span className="badge bg-secondary ms-2">Mean power across {ov.total_sequences} sequences</span>
        </div>
        <div className="card-body">
          {bands.map((b, i) => {
            const pct = Math.round((b.mean_power / maxPower) * 100);
            return (
              <div key={i} className="mb-4">
                <div className="d-flex justify-content-between mb-1">
                  <span className="fw-semibold">{b.band} <span className="text-muted small">({b.range})</span></span>
                  <span className="text-muted small">μ = {b.mean_power.toFixed(4)} · n={b.n}</span>
                </div>
                <div className="progress" style={{ height: 22, borderRadius: 8 }}>
                  <div
                    className="progress-bar"
                    style={{ width: `${pct}%`, backgroundColor: b.color, borderRadius: 8, transition: 'width 0.6s' }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="row">
        {bands.map((b, i) => (
          <div key={i} className="col-6 col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-body text-center">
                <div className="h5 fw-bold mb-1" style={{ color: b.color }}>{b.band}</div>
                <div className="text-muted small mb-2">{b.range}</div>
                <div className="h4 fw-bold">{b.mean_power.toFixed(4)}</div>
                <div className="text-muted" style={{ fontSize: '0.7rem' }}>Mean Power</div>
                <div className="text-muted" style={{ fontSize: '0.7rem' }}>n = {b.n} sequences</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DefinitionsPanel({ df }) {
  if (!df) return <div className="text-muted p-3">Loading…</div>;
  const concepts = df.concepts || [];
  const metrics = df.quality_metrics || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white"><strong>Core RNN / LSTM Concepts</strong></div>
        <div className="card-body p-0">
          {concepts.map((c, i) => (
            <div key={i} className={`p-3 ${i % 2 === 0 ? 'bg-light' : ''}`}>
              <div className="fw-semibold mb-1">{c.term}</div>
              <div className="text-muted small">{c.definition}</div>
            </div>
          ))}
        </div>
      </div>

      {metrics.length > 0 && (
        <div className="card shadow-sm mb-4">
          <div className="card-header py-2 bg-secondary text-white"><strong>Quality Metrics &amp; Targets</strong></div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-dark">
                <tr><th>Metric</th><th>Target</th><th>Description</th></tr>
              </thead>
              <tbody>
                {metrics.map((m, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{m.metric}</td>
                    <td><code>{m.target}</code></td>
                    <td className="text-muted small">{m.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="card shadow-sm">
        <div className="card-header py-2 bg-info text-dark"><strong>Architecture Notes</strong></div>
        <div className="card-body small text-muted">
          <ul className="mb-0">
            <li><strong>Vanilla RNN</strong> — fast but limited temporal memory; suitable for &lt;50 time steps</li>
            <li><strong>LSTM</strong> — gated cell state enables 1,000+ time-step dependencies; standard for EEG temporal classification</li>
            <li><strong>GRU</strong> — simplified LSTM variant with ~33% fewer parameters; comparable accuracy on most EEG benchmarks</li>
            <li><strong>Bidirectional LSTM</strong> — processes forward + backward simultaneously; best for offline analysis where full sequence is available</li>
            <li><strong>Attention LSTM</strong> — adds soft alignment over hidden states; produces interpretable temporal saliency for clinical review</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default function RnnLstmPage() {
  const [tab, setTab] = useState('overview');
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/rnn-lstm/overview`).then(r => r.json()),
      fetch(`${API}/api/rnn-lstm/breakdown`).then(r => r.json()),
      fetch(`${API}/api/rnn-lstm/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3">
        <div>
          <h4 className="mb-0 fw-bold">RNN / LSTM — Temporal EEG Classifier</h4>
          <div className="text-muted small">
            Recurrent Neural Network · Long Short-Term Memory · Gated Recurrent Unit · Bidirectional · Attention
          </div>
        </div>
        <div className="ms-auto d-flex gap-2">
          <span className="badge bg-primary">RNN</span>
          <span className="badge bg-info text-dark">LSTM</span>
          <span className="badge bg-secondary">GRU</span>
          <span className="badge bg-dark">Bidirectional</span>
        </div>
      </div>

      {err && <div className="alert alert-danger">Error: {err}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel ov={ov} />}
      {tab === 'sequences'   && <SequencesPanel bk={bk} />}
      {tab === 'band-power'  && <BandPowerPanel ov={ov} />}
      {tab === 'definitions' && <DefinitionsPanel df={df} />}
    </div>
  );
}
