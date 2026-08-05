'use client';
import { useEffect, useState, useCallback } from 'react';

const API = 'http://localhost:8010';
const BAND_COLORS = { delta:'#6366f1', theta:'#8b5cf6', alpha:'#06b6d4', beta:'#10b981', gamma:'#f59e0b' };

function TraceRow({ ch, uv, height=28, gain=0.12, color='#1a4d8f', isBad=false }) {
  if (!uv || uv.length === 0) return null;
  const mid = height / 2;
  const pts = uv.map((v, x) => `${(x / (uv.length - 1)) * 100}%,${mid - v * gain}`).join(' ');
  return (
    <div className="d-flex align-items-center border-bottom py-1">
      <span className="small" style={{ width: 96, color: isBad ? '#dc2626' : '#6b7280', fontFamily: 'monospace', fontSize: 11 }}>
        {ch}{isBad ? ' ⚠' : ''}
      </span>
      <svg width="100%" height={height} style={{ overflow: 'visible' }}>
        <polyline fill="none" stroke={isBad ? '#fca5a5' : color} strokeWidth="1"
          points={uv.map((v, x) => `${(x / (uv.length - 1)) * 640},${mid - v * gain}`).join(' ')} />
      </svg>
    </div>
  );
}

export default function EegViewer() {
  const [presets, setPresets] = useState([]);
  const [recordings, setRecordings] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [start, setStart] = useState(0);
  const [windowSec, setWindowSec] = useState(10);
  const [traces, setTraces] = useState([]);
  const [meta, setMeta] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [badChannels, setBadChannels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState('viewer');
  const [gainScale, setGainScale] = useState(0.12);

  useEffect(() => {
    fetch(`${API}/api/eeg-viz/presets`).then(r => r.json()).then(d => {
      setPresets(d.presets || []);
      const def = d.default || '';
      setSelectedFile(def);
    }).catch(() => {});
    fetch(`${API}/api/eeg-viz/recordings`).then(r => r.json()).then(d => {
      setRecordings((d.recordings || []).slice(0, 60));
    }).catch(() => {});
  }, []);

  const loadTraces = useCallback((file, t, sec) => {
    if (!file) return;
    setLoading(true);
    const url = `${API}/api/eeg-viz/traces?file=${encodeURIComponent(file)}&start=${t}&seconds=${sec}`;
    fetch(url).then(r => r.json()).then(d => {
      setTraces(d.traces || []);
      setMeta({ sfreq: d.sfreq, duration_s: d.duration_s, n_channels: d.n_channels, n_points: d.n_points, source: d.source });
    }).catch(() => setTraces([])).finally(() => setLoading(false));
  }, []);

  const loadAnalysis = useCallback((file) => {
    if (!file) return;
    fetch(`${API}/api/eeg-viz?file=${encodeURIComponent(file)}`).then(r => r.json()).then(d => {
      setAnalysis(d);
    }).catch(() => {});
    fetch(`${API}/api/eeg-viz/bad-channels?file=${encodeURIComponent(file)}`).then(r => r.json()).then(d => {
      setBadChannels((d.bad_channels || []).map(c => c.channel || c));
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (selectedFile) {
      loadTraces(selectedFile, start, windowSec);
      loadAnalysis(selectedFile);
    }
  }, [selectedFile]);

  const handleFileChange = (f) => { setSelectedFile(f); setStart(0); };
  const step = (dir) => {
    const next = Math.max(0, start + dir * windowSec);
    setStart(next);
    loadTraces(selectedFile, next, windowSec);
  };

  const spikes = analysis?.spikes || {};
  const bp = analysis?.band_power || [];
  const lat = analysis?.lateralization || {};
  const totalDur = meta?.duration_s || 0;

  return (
    <div>
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <h4 className="mb-0 fw-bold">EEG Signal Viewer</h4>
        {meta && (
          <span className="badge bg-secondary">{meta.n_channels} ch · {meta.sfreq} Hz · {totalDur.toFixed(0)}s total</span>
        )}
      </div>

      {/* File selector */}
      <div className="card shadow-sm mb-3">
        <div className="card-body py-2">
          <div className="row g-2 align-items-center">
            <div className="col-md-5">
              <label className="form-label small mb-1 fw-semibold">Preset Recording</label>
              <select className="form-select form-select-sm" value={selectedFile}
                onChange={e => handleFileChange(e.target.value)}>
                <option value="">— select preset —</option>
                {presets.map(p => <option key={p.key} value={p.file}>{p.label}</option>)}
              </select>
            </div>
            <div className="col-md-5">
              <label className="form-label small mb-1 fw-semibold">Or pick from {recordings.length} recordings</label>
              <select className="form-select form-select-sm" value={selectedFile}
                onChange={e => handleFileChange(e.target.value)}>
                <option value="">— all recordings —</option>
                {recordings.map((r, i) => (
                  <option key={i} value={r.file}>{r.file.split('/').pop()} ({r.group})</option>
                ))}
              </select>
            </div>
            <div className="col-md-2">
              <label className="form-label small mb-1 fw-semibold">Window (s)</label>
              <select className="form-select form-select-sm" value={windowSec}
                onChange={e => { const v = Number(e.target.value); setWindowSec(v); loadTraces(selectedFile, start, v); }}>
                {[5, 10, 20, 30].map(s => <option key={s} value={s}>{s}s</option>)}
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {['viewer', 'band-power', 'spikes', 'lateralization', 'bad-channels'].map(t2 => (
          <li key={t2} className="nav-item">
            <button className={`nav-link${tab === t2 ? ' active' : ''}`} onClick={() => setTab(t2)}>
              {t2 === 'viewer' ? 'Waveforms' :
               t2 === 'band-power' ? 'Band Power' :
               t2 === 'spikes' ? `Spikes${spikes.total_spikes ? ` (${spikes.total_spikes})` : ''}` :
               t2 === 'lateralization' ? 'Lateralization' : 'Channel QC'}
            </button>
          </li>
        ))}
      </ul>

      {/* VIEWER TAB */}
      {tab === 'viewer' && (
        <div className="card shadow-sm">
          <div className="card-header d-flex align-items-center gap-3 py-2">
            <button className="btn btn-outline-secondary btn-sm" onClick={() => step(-1)} disabled={start === 0}>◀ Prev</button>
            <span className="small text-muted">{start.toFixed(0)}s – {(start + windowSec).toFixed(0)}s
              {totalDur > 0 ? ` / ${totalDur.toFixed(0)}s total` : ''}</span>
            <button className="btn btn-outline-secondary btn-sm"
              onClick={() => step(1)} disabled={totalDur > 0 && start + windowSec >= totalDur}>Next ▶</button>
            <div className="ms-auto d-flex align-items-center gap-2">
              <label className="small mb-0">Gain:</label>
              <input type="range" min="0.01" max="0.5" step="0.01" value={gainScale}
                onChange={e => setGainScale(Number(e.target.value))} style={{ width: 80 }} />
              <span className="small text-muted">{(gainScale * 100 / 0.12).toFixed(0)}%</span>
            </div>
            {loading && <span className="spinner-border spinner-border-sm ms-2" />}
          </div>
          <div className="card-body p-2">
            {traces.length === 0 && !loading && (
              <p className="text-muted small text-center py-4">Select a recording to load real EEG traces.</p>
            )}
            {traces.map((tr, i) => (
              <TraceRow key={tr.channel || i} ch={tr.channel} uv={tr.uv}
                isBad={badChannels.includes(tr.channel)}
                gain={gainScale}
                color={i % 2 === 0 ? '#1a4d8f' : '#2563eb'} />
            ))}
          </div>
          {traces.length > 0 && (
            <div className="card-footer py-1 d-flex gap-4">
              <span className="small text-muted">Source: {meta?.source}</span>
              <span className="small text-muted">{traces.length} channels · {meta?.n_points} points/ch</span>
              {badChannels.length > 0 && <span className="small text-danger">⚠ {badChannels.length} bad ch</span>}
            </div>
          )}
        </div>
      )}

      {/* BAND POWER TAB */}
      {tab === 'band-power' && (
        <div className="card shadow-sm">
          <div className="card-header py-2"><strong>Spectral Band Power</strong></div>
          <div className="card-body">
            {bp.length === 0 ? <p className="text-muted">No data — select a recording.</p> : (
              <div>
                {bp.map(b => (
                  <div key={b.band} className="mb-3">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className="fw-semibold text-capitalize">{b.band}</span>
                      <span className="text-muted">{(b.rel_power * 100).toFixed(1)}%</span>
                    </div>
                    <div className="progress" style={{ height: 20 }}>
                      <div className="progress-bar" style={{
                        width: `${b.rel_power * 100}%`,
                        backgroundColor: BAND_COLORS[b.band] || '#6b7280'
                      }}>{(b.rel_power * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                ))}
                <p className="text-muted small mt-3">Relative band power over 30s analysis window. Delta (&lt;4Hz) · Theta (4–8Hz) · Alpha (8–13Hz) · Beta (13–30Hz) · Gamma (&gt;30Hz).</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* SPIKES TAB */}
      {tab === 'spikes' && (
        <div className="card shadow-sm">
          <div className="card-header py-2"><strong>Interictal Spike Screening</strong></div>
          <div className="card-body">
            {!spikes.available ? <p className="text-muted">No data — select a recording.</p> : (
              <div>
                <div className="row g-3 mb-4">
                  <div className="col-md-3">
                    <div className="card bg-danger text-white text-center p-2">
                      <div className="fs-3 fw-bold">{spikes.total_spikes}</div>
                      <div className="small">Total Spikes</div>
                    </div>
                  </div>
                  <div className="col-md-3">
                    <div className="card bg-warning text-dark text-center p-2">
                      <div className="fs-3 fw-bold">{spikes.rate_per_min?.toFixed(0)}</div>
                      <div className="small">Rate / min</div>
                    </div>
                  </div>
                </div>
                <table className="table table-sm table-hover">
                  <thead><tr><th>Channel</th><th>Spikes</th><th>Rate / min</th></tr></thead>
                  <tbody>
                    {(spikes.top_channels || []).map(c => (
                      <tr key={c.channel}>
                        <td className="font-monospace">{c.channel}</td>
                        <td>{c.spikes}</td>
                        <td>{c.rate_per_min?.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="alert alert-warning small">{spikes.method} · {spikes.note}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* LATERALIZATION TAB */}
      {tab === 'lateralization' && (
        <div className="card shadow-sm">
          <div className="card-header py-2"><strong>Hemispheric Lateralization</strong></div>
          <div className="card-body">
            {!lat.available ? <p className="text-muted">No data — select a recording.</p> : (
              <div>
                <div className="row g-3 mb-4">
                  <div className="col-md-4">
                    <div className="card border-primary text-center p-2">
                      <div className="fs-4 fw-bold text-primary">{lat.focus}</div>
                      <div className="small text-muted">Dominant Focus</div>
                    </div>
                  </div>
                  <div className="col-md-4">
                    <div className="card border-secondary text-center p-2">
                      <div className="fs-4 fw-bold">{lat.overall_index?.toFixed(3)}</div>
                      <div className="small text-muted">Asymmetry Index</div>
                    </div>
                  </div>
                  <div className="col-md-4">
                    <div className="card border-secondary text-center p-2">
                      <div className="fs-5 fw-bold">{lat.n_left}L / {lat.n_right}R</div>
                      <div className="small text-muted">Channels</div>
                    </div>
                  </div>
                </div>
                <table className="table table-sm table-hover">
                  <thead><tr><th>Band</th><th>Asymmetry Index</th><th>Lateralization</th></tr></thead>
                  <tbody>
                    {(lat.by_band || []).map(b => (
                      <tr key={b.band}>
                        <td className="text-capitalize">{b.band}</td>
                        <td>{b.asymmetry_index?.toFixed(3)}</td>
                        <td><span className={`badge ${b.lateralization?.includes('Left') ? 'bg-primary' : 'bg-success'}`}>{b.lateralization}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="alert alert-info small">{lat.basis} · {lat.note}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* BAD CHANNELS TAB */}
      {tab === 'bad-channels' && (
        <div className="card shadow-sm">
          <div className="card-header py-2"><strong>Channel Quality Control</strong></div>
          <div className="card-body">
            {!analysis ? <p className="text-muted">No data — select a recording.</p> : (
              <div>
                <div className="mb-3">
                  <span className={`badge me-2 ${badChannels.length === 0 ? 'bg-success' : 'bg-danger'}`}>
                    {badChannels.length === 0 ? 'All channels good' : `${badChannels.length} bad channel(s)`}
                  </span>
                </div>
                {(() => {
                  const chans = analysis.channels || [];
                  return (
                    <table className="table table-sm table-hover">
                      <thead><tr><th>Channel</th><th>Std (µV)</th><th>P2P (µV)</th><th>Flat ratio</th><th>Line noise</th><th>Status</th></tr></thead>
                      <tbody>
                        {chans.map(c => (
                          <tr key={c.channel}>
                            <td className="font-monospace small">{c.channel}</td>
                            <td>{c.std_uv?.toFixed(1)}</td>
                            <td>{c.p2p_uv?.toFixed(1)}</td>
                            <td>{c.flat_ratio?.toFixed(3)}</td>
                            <td>{c.line_noise_rel?.toFixed(3)}</td>
                            <td><span className={`badge ${c.verdict === 'good' ? 'bg-success' : 'bg-danger'}`}>{c.verdict}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  );
                })()}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
