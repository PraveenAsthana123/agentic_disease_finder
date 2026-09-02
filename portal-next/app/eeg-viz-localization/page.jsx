'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const REGION_COLORS = {
  'fronto-temporal': '#6366f1',
  frontal:           '#f59e0b',
  temporal:          '#22c55e',
  parietal:          '#ef4444',
  central:           '#a855f7',
  occipital:         '#14b8a6',
};
const HEMI_BADGE = {
  right:     'danger',
  left:      'primary',
  bilateral: 'warning',
  midline:   'secondary',
};

function KPI({ label, value, color, icon, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className={`card border-${color || 'primary'} border-2 h-100`}>
        <div className="card-body text-center py-2 px-2">
          <div style={{ fontSize: '1.4rem' }}>{icon}</div>
          <div className={`fw-bold fs-5 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function PowerBar({ ch, max }) {
  const pct = max > 0 ? (ch.ictal_increase_x / max) * 100 : 0;
  const col = REGION_COLORS[ch.region] || '#6b7280';
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span className="fw-semibold">{ch.channel}</span>
        <span className="text-muted">{ch.ictal_increase_x}× — {ch.region} ({ch.hemisphere})</span>
      </div>
      <div className="progress" style={{ height: 14 }}>
        <div
          className="progress-bar"
          role="progressbar"
          style={{ width: `${pct}%`, backgroundColor: col }}
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
}

function RegionSummary({ channels }) {
  const regionMap = {};
  channels.forEach(ch => {
    if (!regionMap[ch.region]) regionMap[ch.region] = { count: 0, maxX: 0 };
    regionMap[ch.region].count += 1;
    if (ch.ictal_increase_x > regionMap[ch.region].maxX) {
      regionMap[ch.region].maxX = ch.ictal_increase_x;
    }
  });
  const entries = Object.entries(regionMap).sort((a, b) => b[1].maxX - a[1].maxX);
  return (
    <div className="row g-2">
      {entries.map(([region, { count, maxX }]) => (
        <div key={region} className="col-6 col-md-4">
          <div className="card border-0 shadow-sm">
            <div className="card-body py-2 px-3 d-flex align-items-center gap-2">
              <div
                className="rounded-circle flex-shrink-0"
                style={{ width: 12, height: 12, backgroundColor: REGION_COLORS[region] || '#6b7280' }}
              />
              <div>
                <div className="fw-semibold small text-capitalize">{region}</div>
                <div className="text-muted" style={{ fontSize: '0.7rem' }}>
                  {count} ch · peak {maxX}×
                </div>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function EegVizLocalizationPage() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState('overview');
  const [hemiFilter, setHemiFilter] = useState('all');

  useEffect(() => {
    fetch(`${API}/api/eeg-viz/localization`)
      .then(r => r.json())
      .then(setD)
      .catch(e => setErr(e.message));
  }, []);

  if (err) return (
    <div className="container-fluid py-3">
      <div className="alert alert-danger">Error loading localization data: {err}</div>
    </div>
  );
  if (!d) return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" />
      <div className="mt-2 text-muted small">Loading localization data…</div>
    </div>
  );

  const all = d.all_channels_ranked || [];
  const top = d.top_focus_channels || [];
  const focus = d.localized_focus || {};
  const maxX = all.length > 0 ? all[0].ictal_increase_x : 1;
  const seizureWindow = d.seizure_window || {};

  const filtered = hemiFilter === 'all' ? all : all.filter(c => c.hemisphere === hemiFilter);

  const TABS = [
    { id: 'overview', label: '🧠 Overview' },
    { id: 'channels', label: '📡 All Channels' },
    { id: 'region', label: '🗺️ By Region' },
    { id: 'info', label: 'ℹ️ Method & Notes' },
  ];

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1100 }}>
      {/* Header */}
      <div className="d-flex align-items-center gap-2 mb-1">
        <Link href="/eeg-viz" className="btn btn-sm btn-outline-secondary">← EEG Viz</Link>
        <h4 className="mb-0 fw-bold">🧠 Seizure Focus Localization</h4>
        <span className="badge bg-primary ms-1">{d.file}</span>
        <span className="badge bg-secondary ms-1">{d.sfreq} Hz</span>
      </div>
      <p className="text-muted small mb-3">
        Scalp-EEG focal abnormality region · ictal/interictal broadband power ratio · CHB-MIT annotated recording
      </p>

      {/* Focus Alert */}
      <div className="alert alert-info d-flex align-items-center gap-3 py-2 mb-3">
        <span style={{ fontSize: '1.6rem' }}>🎯</span>
        <div>
          <div className="fw-bold">
            Localized Focus: <span className="text-capitalize">{focus.summary || '—'}</span>
          </div>
          <div className="small text-muted">
            Peak channel increase {focus.peak_increase_x}× · Seizure window {seizureWindow.start_s}–{seizureWindow.end_s} s
          </div>
        </div>
        <div className="ms-auto">
          <span className={`badge bg-${HEMI_BADGE[focus.hemisphere] || 'secondary'} text-capitalize`}>
            {focus.hemisphere} hemisphere
          </span>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI
          icon="📡" label="Total Channels" value={all.length}
          color="primary" sub="ranked by ictal increase"
        />
        <KPI
          icon="🎯" label="Top Focus Channels" value={top.length}
          color="success" sub="highest ictal power"
        />
        <KPI
          icon="⚡" label="Peak Ictal Increase" value={`${maxX}×`}
          color="danger" sub={top[0]?.channel || ''}
        />
        <KPI
          icon="🗺️" label="Focus Region" value={focus.region ? focus.region.replace('-', ' ') : '—'}
          color="warning" sub={`${focus.hemisphere || ''} hemisphere`}
        />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active fw-semibold' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div className="row g-3">
          <div className="col-md-7">
            <div className="card shadow-sm">
              <div className="card-header py-2">
                <strong>🔝 Top Focus Channels — Ictal Power Increase</strong>
              </div>
              <div className="card-body">
                {top.map(ch => (
                  <PowerBar key={ch.channel} ch={ch} max={maxX} />
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-5">
            <div className="card shadow-sm mb-3">
              <div className="card-header py-2">
                <strong>🎯 Focus Summary</strong>
              </div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <tbody>
                    <tr><th className="text-muted fw-normal">Region</th><td className="fw-semibold text-capitalize">{focus.region}</td></tr>
                    <tr><th className="text-muted fw-normal">Hemisphere</th>
                      <td>
                        <span className={`badge bg-${HEMI_BADGE[focus.hemisphere] || 'secondary'} text-capitalize`}>
                          {focus.hemisphere}
                        </span>
                      </td>
                    </tr>
                    <tr><th className="text-muted fw-normal">Peak channel</th><td className="font-monospace small">{top[0]?.channel}</td></tr>
                    <tr><th className="text-muted fw-normal">Peak ictal ×</th><td className="fw-bold text-danger">{focus.peak_increase_x}×</td></tr>
                    <tr><th className="text-muted fw-normal">Seizure start</th><td>{seizureWindow.start_s} s</td></tr>
                    <tr><th className="text-muted fw-normal">Seizure end</th><td>{seizureWindow.end_s} s</td></tr>
                    <tr><th className="text-muted fw-normal">Duration</th><td>{seizureWindow.end_s - seizureWindow.start_s} s</td></tr>
                    <tr><th className="text-muted fw-normal">EDF file</th><td className="font-monospace small">{d.file}</td></tr>
                    <tr><th className="text-muted fw-normal">Sample rate</th><td>{d.sfreq} Hz</td></tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm border-warning">
              <div className="card-body py-2 px-3">
                <div className="d-flex align-items-start gap-2">
                  <span>⚠️</span>
                  <div className="small text-muted">
                    <strong className="text-dark">Clinical Disclaimer</strong><br />
                    Scalp-EEG power localization is a <em>screening tool</em>, not intracranial
                    seizure-onset-zone mapping. Surgical planning requires SEEG/ECoG and
                    multidisciplinary epilepsy team review.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* All channels tab */}
      {tab === 'channels' && (
        <div className="card shadow-sm">
          <div className="card-header py-2 d-flex align-items-center gap-3">
            <strong>📡 All {all.length} Channels — Ranked by Ictal Power Increase</strong>
            <div className="ms-auto d-flex align-items-center gap-2">
              <label className="form-label mb-0 small">Hemisphere:</label>
              <select
                className="form-select form-select-sm"
                style={{ width: 120 }}
                value={hemiFilter}
                onChange={e => setHemiFilter(e.target.value)}
              >
                <option value="all">All</option>
                <option value="right">Right</option>
                <option value="left">Left</option>
                <option value="bilateral">Bilateral</option>
                <option value="midline">Midline</option>
              </select>
            </div>
          </div>
          <div className="card-body">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>#</th>
                    <th>Channel</th>
                    <th>Ictal Increase</th>
                    <th>Region</th>
                    <th>Hemisphere</th>
                    <th>Power Bar</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((ch, i) => (
                    <tr key={ch.channel}>
                      <td className="text-muted small">{i + 1}</td>
                      <td className="font-monospace small fw-semibold">{ch.channel}</td>
                      <td className="fw-bold" style={{ color: REGION_COLORS[ch.region] || '#374151' }}>
                        {ch.ictal_increase_x}×
                      </td>
                      <td>
                        <span
                          className="badge rounded-pill text-white text-capitalize small"
                          style={{ backgroundColor: REGION_COLORS[ch.region] || '#6b7280' }}
                        >
                          {ch.region}
                        </span>
                      </td>
                      <td>
                        <span className={`badge bg-${HEMI_BADGE[ch.hemisphere] || 'secondary'} text-capitalize small`}>
                          {ch.hemisphere}
                        </span>
                      </td>
                      <td style={{ width: 150 }}>
                        <div className="progress" style={{ height: 10 }}>
                          <div
                            className="progress-bar"
                            style={{
                              width: `${(ch.ictal_increase_x / maxX) * 100}%`,
                              backgroundColor: REGION_COLORS[ch.region] || '#6b7280',
                            }}
                          />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {filtered.length === 0 && (
              <div className="text-center text-muted py-3 small">No channels match the selected filter.</div>
            )}
          </div>
        </div>
      )}

      {/* Region tab */}
      {tab === 'region' && (
        <div>
          <div className="mb-3">
            <h6 className="fw-semibold mb-2">🗺️ Region Summary</h6>
            <RegionSummary channels={all} />
          </div>
          <div className="card shadow-sm">
            <div className="card-header py-2">
              <strong>Channels by Region</strong>
            </div>
            <div className="card-body">
              {Object.keys(REGION_COLORS).map(region => {
                const chs = all.filter(c => c.region === region);
                if (!chs.length) return null;
                return (
                  <div key={region} className="mb-4">
                    <div className="d-flex align-items-center gap-2 mb-2">
                      <div
                        className="rounded-circle"
                        style={{ width: 10, height: 10, backgroundColor: REGION_COLORS[region] }}
                      />
                      <span className="fw-semibold text-capitalize">{region}</span>
                      <span className="badge bg-light text-dark border">{chs.length} channels</span>
                    </div>
                    {chs.map(ch => <PowerBar key={ch.channel} ch={ch} max={maxX} />)}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Method & Notes tab */}
      {tab === 'info' && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header py-2">
                <strong>🔬 Method</strong>
              </div>
              <div className="card-body">
                <p className="small mb-2">{d.method}</p>
                <dl className="row small mb-0">
                  <dt className="col-5 text-muted">Frequency band</dt>
                  <dd className="col-7">1–30 Hz (broadband)</dd>
                  <dt className="col-5 text-muted">Metric</dt>
                  <dd className="col-7">Ictal ÷ Interictal RMS power</dd>
                  <dt className="col-5 text-muted">Focus criterion</dt>
                  <dd className="col-7">Top channels by power ratio</dd>
                  <dt className="col-5 text-muted">Data source</dt>
                  <dd className="col-7">{d.source}</dd>
                  <dt className="col-5 text-muted">EDF file</dt>
                  <dd className="col-7 font-monospace">{d.file}</dd>
                  <dt className="col-5 text-muted">Sample rate</dt>
                  <dd className="col-7">{d.sfreq} Hz</dd>
                </dl>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm border-warning">
              <div className="card-header py-2 bg-warning bg-opacity-10">
                <strong>⚠️ Clinical Notes & Limitations</strong>
              </div>
              <div className="card-body">
                <p className="small mb-2">{d.note}</p>
                <ul className="small mb-0 ps-3">
                  <li>Scalp EEG has low spatial resolution (~6 cm²) — cannot distinguish adjacent sulci</li>
                  <li>Volume conduction smears focal sources across adjacent electrodes</li>
                  <li>Single-recording analysis — multi-seizure averaging improves reliability</li>
                  <li>Must be correlated with clinical semiology, MRI, and PET/SPECT</li>
                  <li>Surgical candidates require SEEG/ECoG for definitive SOZ mapping</li>
                </ul>
              </div>
            </div>

            <div className="card shadow-sm mt-3">
              <div className="card-header py-2">
                <strong>🔗 Related EEG Viz Tools</strong>
              </div>
              <div className="card-body">
                <div className="d-flex flex-wrap gap-2">
                  {[
                    { href: '/eeg-viz', label: '🧠 EEG Viz Platform' },
                    { href: '/eeg-viz-raw', label: '📈 Raw EEG Viewer' },
                    { href: '/eeg-viz-artifacts', label: '🔬 Artifact Review' },
                    { href: '/eeg-viz-montage', label: '📡 Montage' },
                    { href: '/eeg-viz-bad-channels', label: '❌ Bad Channels' },
                  ].map(link => (
                    <Link key={link.href} href={link.href} className="btn btn-sm btn-outline-primary">
                      {link.label}
                    </Link>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
