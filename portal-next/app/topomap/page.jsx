'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Band Power Map' },
  { id: 'electrodes', label: 'Electrode Layout' },
  { id: 'asymmetry', label: 'Alpha Asymmetry' },
  { id: 'definitions', label: 'Definitions' },
];

const BAND_COLORS = {
  delta: '#6366f1',
  theta: '#8b5cf6',
  alpha: '#06b6d4',
  beta: '#10b981',
  gamma: '#f59e0b',
};

const BAND_KEYS = ['delta', 'theta', 'alpha', 'beta', 'gamma'];

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

/* Minimalist SVG head + electrode scatter */
function TopoHead({ electrodes, band }) {
  if (!electrodes || electrodes.length === 0) return null;

  const W = 320, H = 320, CX = 160, CY = 160, R = 130;

  // Normalize band power to [0,1] for color mapping
  const vals = electrodes.map(e => e[band] ?? 0);
  const minV = Math.min(...vals);
  const maxV = Math.max(...vals);
  const norm = v => maxV === minV ? 0.5 : (v - minV) / (maxV - minV);

  // Color scale: blue (low) → yellow → red (high)
  function powerColor(n) {
    // 0=blue, 0.5=yellow, 1=red
    const r = Math.round(Math.min(255, n * 2 * 255));
    const g = Math.round(Math.min(255, (1 - Math.abs(n - 0.5) * 2) * 255));
    const b = Math.round(Math.max(0, (1 - n * 2) * 255));
    return `rgb(${r},${g},${b})`;
  }

  // Project EEG (x,y) in [-1,1] range to SVG coords
  const px = x => CX + x * R;
  const py = y => CY - y * R; // y is inverted (nose at top)

  // Deduplicate channels by name (take first occurrence)
  const seen = new Set();
  const unique = electrodes.filter(e => {
    if (seen.has(e.channel)) return false;
    seen.add(e.channel);
    return true;
  });

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: '100%' }}>
      {/* Head circle */}
      <circle cx={CX} cy={CY} r={R} fill="#f8f9fa" stroke="#aaa" strokeWidth={2} />
      {/* Nose */}
      <polygon points={`${CX - 10},${CY - R + 10} ${CX},${CY - R - 14} ${CX + 10},${CY - R + 10}`}
        fill="#f8f9fa" stroke="#aaa" strokeWidth={2} />
      {/* Left ear */}
      <ellipse cx={CX - R - 8} cy={CY} rx={8} ry={14} fill="#f8f9fa" stroke="#aaa" strokeWidth={2} />
      {/* Right ear */}
      <ellipse cx={CX + R + 8} cy={CY} rx={8} ry={14} fill="#f8f9fa" stroke="#aaa" strokeWidth={2} />
      {/* Electrodes */}
      {unique.map(e => {
        const n = norm(e[band] ?? 0);
        const col = powerColor(n);
        return (
          <g key={e.channel}>
            <circle cx={px(e.x)} cy={py(e.y)} r={14} fill={col} opacity={0.82} stroke="#333" strokeWidth={1} />
            <text x={px(e.x)} y={py(e.y) + 4} textAnchor="middle" fontSize={9} fontWeight="bold" fill="#333">
              {e.channel}
            </text>
          </g>
        );
      })}
      {/* Color bar legend */}
      <defs>
        <linearGradient id="cb" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="rgb(0,0,255)" />
          <stop offset="50%" stopColor="rgb(255,255,0)" />
          <stop offset="100%" stopColor="rgb(255,0,0)" />
        </linearGradient>
      </defs>
      <rect x={CX - 60} y={H - 22} width={120} height={10} fill="url(#cb)" rx={3} />
      <text x={CX - 62} y={H - 10} fontSize={9} fill="#666">Low</text>
      <text x={CX + 40} y={H - 10} fontSize={9} fill="#666">High</text>
    </svg>
  );
}

function OverviewPanel({ data }) {
  const [selBand, setSelBand] = useState('delta');
  if (!data) return <div className="text-muted">Loading…</div>;
  if (!data.available) return <div className="alert alert-warning">No EEG data loaded. Upload an EDF/CHB-MIT file to see the topomap.</div>;

  const electrodes = data.electrodes || [];
  const bandAvgs = {};
  BAND_KEYS.forEach(b => {
    const vals = electrodes.map(e => e[b] ?? 0).filter(v => v > 0);
    bandAvgs[b] = vals.length ? (vals.reduce((a, c) => a + c, 0) / vals.length).toFixed(4) : '—';
  });

  const dominantBand = BAND_KEYS.reduce((best, b) => {
    const avg = parseFloat(bandAvgs[b]) || 0;
    const bestAvg = parseFloat(bandAvgs[best]) || 0;
    return avg > bestAvg ? b : best;
  }, 'delta');

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Channels Mapped" value={data.n_channels_mapped} color="primary" sub={`of ${data.n_channels_total} total`} />
        <KPI label="Dominant Band" value={dominantBand.toUpperCase()} color="success" sub="highest mean power" />
        <KPI label="Sampling Rate" value={`${data.sfreq} Hz`} color="info" sub="signal frequency" />
        <KPI label="Epoch Length" value={`${data.duration_seconds}s`} color="warning" sub="analysis window" />
      </div>

      <div className="row mb-4">
        <div className="col-md-5 mb-3 d-flex flex-column align-items-center">
          <div className="card shadow-sm w-100 h-100">
            <div className="card-header fw-semibold d-flex align-items-center gap-2">
              🧠 10-20 Topomap
              <div className="ms-auto d-flex gap-1 flex-wrap">
                {BAND_KEYS.map(b => (
                  <button
                    key={b}
                    onClick={() => setSelBand(b)}
                    className={`btn btn-sm ${selBand === b ? 'btn-dark' : 'btn-outline-secondary'}`}
                    style={{ fontSize: '0.7rem', padding: '2px 7px' }}
                  >
                    {b.charAt(0).toUpperCase() + b.slice(1)}
                  </button>
                ))}
              </div>
            </div>
            <div className="card-body d-flex flex-column align-items-center">
              <TopoHead electrodes={electrodes} band={selBand} />
              <div className="text-muted small mt-2">
                International 10-20 system · relative band power · colour: low (blue) → high (red)
              </div>
            </div>
          </div>
        </div>

        <div className="col-md-7 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Band Power per Electrode</div>
            <div className="card-body p-2" style={{ overflowY: 'auto', maxHeight: 380 }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light sticky-top">
                  <tr>
                    <th>Channel</th>
                    {BAND_KEYS.map(b => (
                      <th key={b} style={{ color: BAND_COLORS[b] }}>{b.charAt(0).toUpperCase() + b.slice(1)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {electrodes.map((e, i) => (
                    <tr key={i}>
                      <td><strong>{e.channel}</strong><br /><span className="text-muted" style={{ fontSize: '0.68rem' }}>{e.original_name}</span></td>
                      {BAND_KEYS.map(b => (
                        <td key={b} className="font-monospace small"
                          style={{ background: `rgba(${b === 'delta' ? '99,102,241' : b === 'theta' ? '139,92,246' : b === 'alpha' ? '6,182,212' : b === 'beta' ? '16,185,129' : '245,158,11'},${Math.min(0.6, (e[b] || 0) * 2)})` }}>
                          {(e[b] ?? 0).toFixed(4)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
                <tfoot className="table-secondary fw-semibold">
                  <tr>
                    <td>Mean</td>
                    {BAND_KEYS.map(b => <td key={b} className="font-monospace small">{bandAvgs[b]}</td>)}
                  </tr>
                </tfoot>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ElectrodesPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;

  const positions = data.positions || [];

  return (
    <div>
      <div className="row mb-3">
        <KPI label="System" value={data.system} color="primary" sub="electrode placement standard" />
        <KPI label="Positions" value={data.n_positions} color="success" sub="mapped electrodes" />
      </div>

      <div className="row">
        <div className="col-md-5 mb-3 d-flex align-items-center justify-content-center">
          <div className="card shadow-sm w-100">
            <div className="card-header fw-semibold">🧠 Electrode Locations (10-20)</div>
            <div className="card-body d-flex justify-content-center">
              {/* Render layout-only topomap with uniform colour */}
              <svg width={300} height={300} viewBox="0 0 300 300">
                <circle cx={150} cy={150} r={120} fill="#f0f4ff" stroke="#aaa" strokeWidth={2} />
                <polygon points="140,32 150,16 160,32" fill="#f0f4ff" stroke="#aaa" strokeWidth={2} />
                <ellipse cx={22} cy={150} rx={8} ry={14} fill="#f0f4ff" stroke="#aaa" strokeWidth={2} />
                <ellipse cx={278} cy={150} rx={8} ry={14} fill="#f0f4ff" stroke="#aaa" strokeWidth={2} />
                {positions.map((e, i) => {
                  const x = 150 + e.x * 120;
                  const y = 150 - e.y * 120;
                  return (
                    <g key={i}>
                      <circle cx={x} cy={y} r={13} fill="#6366f1" opacity={0.75} stroke="#4338ca" strokeWidth={1.5} />
                      <text x={x} y={y + 4} textAnchor="middle" fontSize={8} fontWeight="bold" fill="white">
                        {e.channel}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>
        </div>

        <div className="col-md-7 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Electrode Coordinates</div>
            <div className="card-body p-2" style={{ overflowY: 'auto', maxHeight: 340 }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr><th>#</th><th>Channel</th><th>X</th><th>Y</th><th>Region</th></tr>
                </thead>
                <tbody>
                  {positions.map((e, i) => {
                    const region = e.y > 0.5 ? 'Frontal' : e.y < -0.5 ? 'Occipital' : e.x < -0.6 ? 'Left Temporal' : e.x > 0.6 ? 'Right Temporal' : 'Central';
                    return (
                      <tr key={i}>
                        <td className="text-muted small">{i + 1}</td>
                        <td><strong>{e.channel}</strong></td>
                        <td className="font-monospace small">{e.x?.toFixed(2)}</td>
                        <td className="font-monospace small">{e.y?.toFixed(2)}</td>
                        <td><span className="badge bg-secondary">{region}</span></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function AsymmetryPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (!data.available) return <div className="alert alert-warning">No EEG data loaded.</div>;

  const pairs = data.pairs || [];

  const badgeColor = (asym) => {
    if (asym > 0.5) return 'success';
    if (asym < -0.5) return 'danger';
    return 'secondary';
  };

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Metric" value="Alpha Asymmetry" color="info" sub={data.metric || ''} />
        <KPI label="Formula" value="ln(R) − ln(L)" color="primary" sub="Davidson (1998)" />
        <KPI label="Pairs Analysed" value={pairs.length} color="success" sub="frontal/parietal/central" />
        <KPI label="Frontal Verdict" value={pairs[0]?.interpretation?.split(' ')[0] === 'right' ? 'R>L' : 'L>R'} color={pairs[0]?.asymmetry > 0 ? 'success' : 'danger'} sub={pairs[0]?.interpretation || ''} />
      </div>

      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Alpha Asymmetry by Region</div>
            <div className="card-body">
              {pairs.map(p => (
                <div key={p.region} className="mb-3">
                  <div className="d-flex justify-content-between align-items-center mb-1">
                    <strong className="text-capitalize">{p.region}</strong>
                    <span className={`badge bg-${badgeColor(p.asymmetry)}`}>{p.asymmetry?.toFixed(4)}</span>
                  </div>
                  <div className="d-flex gap-2 small text-muted mb-1">
                    <span>L ({p.left}): {p.left_alpha?.toFixed(4)}</span>
                    <span>|</span>
                    <span>R ({p.right}): {p.right_alpha?.toFixed(4)}</span>
                  </div>
                  {/* Bar: centre=0, left=negative, right=positive */}
                  <div style={{ position: 'relative', height: 18, background: '#e9ecef', borderRadius: 4 }}>
                    <div
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: p.asymmetry >= 0 ? '50%' : `${50 + (p.asymmetry / 2) * 100}%`,
                        width: `${Math.abs(p.asymmetry) / 2 * 100}%`,
                        height: '100%',
                        background: p.asymmetry >= 0 ? '#10b981' : '#ef4444',
                        borderRadius: 4,
                        maxWidth: '50%',
                      }}
                    />
                    <div style={{ position: 'absolute', left: '50%', top: 0, width: 1, height: '100%', background: '#aaa' }} />
                  </div>
                  <div className="text-muted mt-1" style={{ fontSize: '0.72rem' }}>{p.interpretation}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Clinical Context</div>
            <div className="card-body small">
              <p><strong>Frontal alpha asymmetry</strong> (Davidson, 1998) reflects hemispheric approach/withdrawal motivation:</p>
              <ul>
                <li><span className="text-success fw-semibold">Positive (R&gt;L)</span>: greater left frontal activity → approach motivation</li>
                <li><span className="text-danger fw-semibold">Negative (L&gt;R)</span>: greater right frontal activity → withdrawal / depression</li>
              </ul>
              <p className="mb-1"><strong>Epilepsy relevance:</strong></p>
              <ul>
                <li>Persistent asymmetry may indicate unilateral cortical dysfunction</li>
                <li>Lateralised alpha suppression correlates with focal seizure onset zones</li>
                <li>Used in pre-surgical workup for temporal lobe epilepsy localisation</li>
              </ul>
              <div className="alert alert-info p-2 mb-0 small">
                Formula: <code>ln(right_alpha_power) − ln(left_alpha_power)</code><br />
                Reference: Davidson (1998). Affective style and affective disorders.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;

  const bands = data.bands || [];
  const tools = data.tools || [];
  const asymDef = data.asymmetry || {};
  const topoMap = data.topographic_mapping || {};

  return (
    <div>
      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">EEG Frequency Bands</div>
            <div className="card-body p-2">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Band</th><th>Range</th><th>Role</th><th>Clinical Use</th></tr>
                </thead>
                <tbody>
                  {bands.map(b => (
                    <tr key={b.name}>
                      <td><span className="badge me-1" style={{ background: BAND_COLORS[b.name.toLowerCase()] }}>&nbsp;</span><strong>{b.name}</strong></td>
                      <td className="font-monospace small">{b.range}</td>
                      <td className="small">{b.role}</td>
                      <td className="small text-muted">{b.clinical}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Topographic Mapping</div>
            <div className="card-body small">
              <p><strong>System:</strong> {topoMap.system || 'International 10-20'}</p>
              <p><strong>Reference:</strong> {topoMap.reference}</p>
              <p><strong>Method:</strong> {topoMap.method}</p>
              <hr />
              <p><strong>Alpha Asymmetry:</strong> {asymDef.description}</p>
              <ul>
                <li><strong>Positive:</strong> {asymDef.positive}</li>
                <li><strong>Negative:</strong> {asymDef.negative}</li>
                <li><strong>Clinical:</strong> {asymDef.clinical}</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-semibold">Libraries & Tools</div>
        <div className="card-body p-2">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr><th>Library</th><th>Version</th><th>Role</th><th>Reference</th></tr>
            </thead>
            <tbody>
              {tools.map(t => (
                <tr key={t.name}>
                  <td><strong>{t.name}</strong></td>
                  <td className="small text-muted">{t.version || '—'}</td>
                  <td className="small">{t.role}</td>
                  <td className="small text-muted" style={{ fontSize: '0.7rem' }}>{t.reference}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function TopomapPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [electrodes, setElectrodes] = useState(null);
  const [asymmetry, setAsymmetry] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/topomap/overview`).then(r => r.json()),
      fetch(`${API}/api/topomap/electrodes`).then(r => r.json()),
      fetch(`${API}/api/topomap/asymmetry`).then(r => r.json()),
      fetch(`${API}/api/topomap/definitions`).then(r => r.json()),
    ])
      .then(([ov, el, asym, defs]) => {
        setOverview(ov);
        setElectrodes(el);
        setAsymmetry(asym);
        setDefinitions(defs);
      })
      .catch(e => setError(e.message));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <h4 className="mb-0 fw-bold">🧠 10-20 EEG Topomap</h4>
        <span className="badge bg-primary">International 10-20</span>
        <span className="badge bg-secondary">{overview?.n_channels_mapped ?? '…'} channels</span>
        <span className="badge bg-success">P0 Clinical</span>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${activeTab === t.id ? 'active' : ''}`}
              onClick={() => setActiveTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {activeTab === 'overview' && <OverviewPanel data={overview} />}
      {activeTab === 'electrodes' && <ElectrodesPanel data={electrodes} />}
      {activeTab === 'asymmetry' && <AsymmetryPanel data={asymmetry} />}
      {activeTab === 'definitions' && <DefinitionsPanel data={definitions} />}
    </div>
  );
}
