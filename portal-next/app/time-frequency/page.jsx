'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Method Breakdown' },
  { id: 'definitions', label: 'Definitions' },
];

const FAMILY_COLORS = {
  fourier: 'primary',
  wavelet: 'success',
  quadratic: 'warning',
};

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

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (!data.available) return <div className="alert alert-warning">{data.note || 'No data available'}</div>;

  const kpis = data.kpis || [];
  const families = data.family_distribution || [];
  const pipeline = data.pipeline_stages || [];
  const bands = data.frequency_bands || [];
  const methods = data.methods_table || [];
  const summary = data.summary || {};

  return (
    <div>
      <div className="row mb-3">
        {kpis.map(k => (
          <KPI key={k.label} label={k.label} value={k.value} color={k.color} sub={k.sub} />
        ))}
      </div>

      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Method Family Distribution</div>
            <div className="card-body">
              {families.map(f => (
                <div key={f.name} className="d-flex align-items-center mb-2">
                  <span className={`badge bg-${FAMILY_COLORS[f.name.toLowerCase()] || 'secondary'} me-2`} style={{ width: 80 }}>
                    {f.name}
                  </span>
                  <div className="progress flex-grow-1" style={{ height: 18 }}>
                    <div
                      className={`progress-bar bg-${FAMILY_COLORS[f.name.toLowerCase()] || 'secondary'}`}
                      style={{ width: `${(f.value / (summary.methods_total || 6)) * 100}%` }}
                    >
                      {f.value}
                    </div>
                  </div>
                </div>
              ))}
              <div className="text-muted small mt-2">
                Fourier (fixed window) · Wavelet (adaptive) · Quadratic (cross-terms)
              </div>
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">EEG Frequency Bands</div>
            <div className="card-body p-2">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Band</th><th>Range (Hz)</th><th>Seizure Relevance</th></tr>
                </thead>
                <tbody>
                  {bands.map(b => (
                    <tr key={b.band}>
                      <td>
                        <span className="badge me-1" style={{ backgroundColor: b.color, fontSize: '0.7rem' }}>&nbsp;</span>
                        <strong>{b.band}</strong>
                      </td>
                      <td className="font-monospace small">{b.range_hz}</td>
                      <td className="small text-muted">{b.seizure_relevance}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="card mb-4 shadow-sm">
        <div className="card-header fw-semibold">EEG → TFR Pipeline</div>
        <div className="card-body">
          <div className="d-flex flex-wrap gap-2 align-items-center">
            {pipeline.map((stage, i) => (
              <div key={stage.step} className="d-flex align-items-center gap-2">
                <div className="card border-primary text-center px-2 py-1" style={{ minWidth: 120 }}>
                  <div className="fw-bold small text-primary">Step {stage.step}</div>
                  <div className="small">{stage.name}</div>
                  {stage.format && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{stage.format}</div>}
                  {stage.range_hz && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{stage.range_hz} Hz</div>}
                  {stage.methods && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{stage.methods.join(' · ')}</div>}
                  {stage.output && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{stage.output}</div>}
                  {stage.formats && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{stage.formats.join(' · ')}</div>}
                  {stage.models && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{stage.models.join(' · ')}</div>}
                </div>
                {i < pipeline.length - 1 && <span className="text-muted fs-5">→</span>}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold">Methods at a Glance</div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Method</th>
                <th>Family</th>
                <th>Resolution</th>
                <th>Output</th>
                <th>Pipeline Stage</th>
              </tr>
            </thead>
            <tbody>
              {methods.map(m => (
                <tr key={m.name}>
                  <td className="fw-semibold small">{m.name}</td>
                  <td>
                    <span className={`badge bg-${FAMILY_COLORS[m.family.toLowerCase()] || 'secondary'}`}>
                      {m.family}
                    </span>
                  </td>
                  <td className="small">{m.resolution}</td>
                  <td className="small text-muted">{m.output}</td>
                  <td className="small">{m.pipeline_stage}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (!data.available) return <div className="alert alert-warning">{data.note || 'No data'}</div>;

  const cards = data.method_cards || [];
  const matrix = data.resolution_matrix || [];
  const useCases = data.use_case_map || [];

  return (
    <div>
      <div className="row mb-4">
        {cards.map(m => (
          <div key={m.id} className="col-md-6 col-lg-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className={`card-header fw-semibold bg-${FAMILY_COLORS[m.family.toLowerCase()] || 'secondary'} text-white`}>
                {m.name}
              </div>
              <div className="card-body small">
                <div className="mb-1">
                  <span className="fw-semibold">Output:</span> {m.output}
                </div>
                <div className="mb-1">
                  <span className="fw-semibold">Stage:</span> {m.pipeline_stage}
                </div>
                <div className="mb-1">
                  <span className="fw-semibold">Time res:</span> {m.time_resolution} &nbsp;|&nbsp;
                  <span className="fw-semibold">Freq res:</span> {m.freq_resolution}
                </div>
                {m.params && Object.keys(m.params).length > 0 && (
                  <div className="mb-1">
                    <span className="fw-semibold">Params:</span>{' '}
                    {Object.entries(m.params).map(([k, v]) => `${k}=${v}`).join(', ')}
                  </div>
                )}
                {m.used_for.length > 0 && (
                  <div className="mb-1">
                    <span className="fw-semibold">Used for:</span>
                    <ul className="mb-0 ps-3">
                      {m.used_for.map(u => <li key={u}>{u}</li>)}
                    </ul>
                  </div>
                )}
                {m.notes && (
                  <div className="text-muted mt-2" style={{ fontSize: '0.75rem', fontStyle: 'italic' }}>
                    {m.notes}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Resolution × Complexity Matrix</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr><th>Method</th><th>Time Res</th><th>Freq Res</th><th>Complexity</th></tr>
                </thead>
                <tbody>
                  {matrix.map(m => (
                    <tr key={m.method}>
                      <td className="fw-semibold small">{m.method}</td>
                      <td className="small">{m.time_res}</td>
                      <td className="small">{m.freq_res}</td>
                      <td className="font-monospace small">{m.complexity}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Use-Case → Method Mapping</div>
            <div className="card-body p-0" style={{ maxHeight: 300, overflowY: 'auto' }}>
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Use Case</th><th>Methods</th></tr>
                </thead>
                <tbody>
                  {useCases.map(u => (
                    <tr key={u.use_case}>
                      <td className="small">{u.use_case}</td>
                      <td className="small">{u.methods.join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (!data.available) return <div className="alert alert-warning">{data.note || 'No data'}</div>;

  const defs = data.definitions || [];
  const glossary = data.method_glossary || [];
  const refs = data.references || [];
  const up = data.uncertainty_principle || {};

  return (
    <div>
      {up.statement && (
        <div className="alert alert-info mb-4">
          <strong>Heisenberg Uncertainty Principle (TFR):</strong>{' '}
          <span className="font-monospace">{up.statement}</span>
          <div className="small mt-1">{up.implication}</div>
        </div>
      )}

      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Method Glossary</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr><th>Abbrev</th><th>Name</th><th>Family</th></tr>
                </thead>
                <tbody>
                  {glossary.map(g => (
                    <tr key={g.abbrev}>
                      <td><span className="badge bg-secondary font-monospace">{g.abbrev}</span></td>
                      <td className="small fw-semibold">{g.term}</td>
                      <td>
                        <span className={`badge bg-${FAMILY_COLORS[g.family.toLowerCase()] || 'secondary'}`} style={{ fontSize: '0.65rem' }}>
                          {g.family}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">References</div>
            <div className="card-body">
              {refs.map((r, i) => (
                <div key={i} className="d-flex align-items-start mb-2">
                  <span className={`badge me-2 mt-1 bg-${r.type === 'paper' ? 'info' : r.type === 'textbook' ? 'success' : 'secondary'}`} style={{ fontSize: '0.65rem', minWidth: 55 }}>
                    {r.type}
                  </span>
                  <span className="small">{r.title}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold">Term Definitions</div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr><th style={{ width: '20%' }}>Term</th><th>Definition</th></tr>
            </thead>
            <tbody>
              {defs.map(d => (
                <tr key={d.term}>
                  <td className="fw-semibold small">{d.term}</td>
                  <td className="small text-muted">{d.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function TimeFrequencyPage() {
  const [tab, setTab] = useState('overview');
  const [panels, setPanels] = useState({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (panels[tab]) return;
    setLoading(true);
    fetch(`${API}/api/time-frequency/${tab}`)
      .then(r => r.json())
      .then(d => setPanels(p => ({ ...p, [tab]: d })))
      .catch(() => setPanels(p => ({ ...p, [tab]: { available: false, note: 'API unavailable' } })))
      .finally(() => setLoading(false));
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <h4 className="mb-0">📊 Time-Frequency Representations</h4>
        <span className="badge bg-success">Built</span>
        <span className="badge bg-secondary">STFT · Wavelet · Spectrogram</span>
      </div>
      <p className="text-muted small mb-3">
        Multi-resolution EEG signal analysis: Short-Time Fourier Transform, Continuous &amp; Discrete
        Wavelet Transforms, power spectrograms, mel spectrograms, and Wigner-Ville distributions —
        methods used as CNN/LSTM inputs for seizure detection and localisation.
      </p>

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

      {loading && (
        <div className="text-center py-4">
          <div className="spinner-border spinner-border-sm text-primary me-2" />
          Loading…
        </div>
      )}

      {!loading && tab === 'overview' && <OverviewPanel data={panels.overview} />}
      {!loading && tab === 'breakdown' && <BreakdownPanel data={panels.breakdown} />}
      {!loading && tab === 'definitions' && <DefinitionsPanel data={panels.definitions} />}
    </div>
  );
}
