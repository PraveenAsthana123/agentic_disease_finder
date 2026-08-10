'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

/* ─── colour helpers ─────────────────────────────────────────────────────── */
const qColor = q =>
  q === 'Good' ? '#22c55e' :
  q === 'Fair' ? '#f59e0b' :
  q === 'Poor' ? '#ef4444' : '#94a3b8';

const confColor = c =>
  c >= 0.8 ? '#22c55e' :
  c >= 0.6 ? '#3b82f6' :
  c >= 0.4 ? '#f59e0b' : '#ef4444';

const BAND_COLORS = {
  Delta: '#6366f1', Theta: '#8b5cf6', Alpha: '#22c55e',
  Beta: '#3b82f6', Gamma: '#f59e0b',
};

/* ─── shared components ─────────────────────────────────────────────────── */
function StatCard({ label, value, color = '#3b82f6', sub }) {
  return (
    <div className="col-6 col-md mb-2">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="h5 mb-0 fw-bold" style={{ color }}>{value ?? '—'}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function HBar({ label, count, total, color }) {
  const pct = total ? Math.round((count / total) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-bold">{count} <span className="text-muted">({pct}%)</span></span>
      </div>
      <div style={{ background: '#e5e7eb', borderRadius: 4, height: 10 }}>
        <div style={{ width: `${pct}%`, background: color || '#3b82f6', borderRadius: 4, height: 10 }} />
      </div>
    </div>
  );
}

function BandBar({ band, power }) {
  const pct = Math.min(100, Math.round(power * 400)); // scale 0–0.25 to 0–100
  const color = BAND_COLORS[band] || '#6b7280';
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span className="fw-semibold">{band}</span>
        <span style={{ color }}>{(power * 100).toFixed(1)}%</span>
      </div>
      <div style={{ background: '#e5e7eb', borderRadius: 4, height: 12 }}>
        <div style={{ width: `${pct}%`, background: color, borderRadius: 4, height: 12 }} />
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   OVERVIEW TAB
═══════════════════════════════════════════════════════════════════════════ */
function OverviewTab({ ov }) {
  if (!ov) return <div className="spinner-border text-primary" />;

  const qDist   = ov.signal_quality_distribution   || [];
  const bandDist = ov.band_power_distribution       || [];
  const rhythmDist = ov.background_rhythm_distribution || [];
  const predDist = ov.prediction_distribution       || [];
  const totalQ   = qDist.reduce((s, d) => s + d.value, 0);
  const totalP   = predDist.reduce((s, d) => s + d.value, 0);
  const totalR   = rhythmDist.reduce((s, d) => s + d.value, 0);

  const PRED_COLORS = ['#6366f1','#22c55e','#f59e0b','#ef4444','#8b5cf6','#3b82f6','#10b981','#dc2626','#0ea5e9'];
  const RHYTHM_COLORS = { 'Delta (<4 Hz)': '#6366f1', 'Theta (4-8 Hz)': '#8b5cf6',
    'Alpha (8-13 Hz)': '#22c55e', 'Beta (13-30 Hz)': '#3b82f6',
    'Gamma (>30 Hz)': '#f59e0b', 'Unknown': '#94a3b8' };

  return (
    <div>
      {/* KPI row */}
      <div className="row row-cols-2 row-cols-md-4 g-2 mb-4">
        {(ov.kpis || []).map((k, i) => {
          const color =
            k.label === 'Good Signal Quality' ? '#22c55e' :
            k.label === 'Flat Channel Rate'   ? '#ef4444' :
            k.label === 'Mean AI Confidence'  ? '#3b82f6' :
            k.label === 'Seizure Events'      ? '#f59e0b' :
            '#6366f1';
          return <StatCard key={i} label={k.label} value={k.value} color={color} />;
        })}
      </div>

      <div className="row g-3 mb-3">
        {/* Band Power Distribution */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f4ca; Mean Band Power Distribution</div>
            <div className="card-body">
              {bandDist.map(d => (
                <BandBar key={d.band} band={d.band} power={d.power} />
              ))}
              <div className="text-muted mt-2" style={{ fontSize: '0.7rem' }}>
                Relative power (fraction of total). Alpha dominance = normal wakefulness.
                Delta excess suggests encephalopathy.
              </div>
            </div>
          </div>
        </div>

        {/* Signal Quality */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f7e2; Signal Quality Distribution</div>
            <div className="card-body">
              {qDist.map(d => (
                <HBar key={d.name} label={d.name} count={d.value} total={totalQ}
                  color={qColor(d.name)} />
              ))}
              <div className="text-muted mt-2" style={{ fontSize: '0.7rem' }}>
                Good = minimal artifact · Fair = some contamination · Poor = significant artifact
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3">
        {/* Background Rhythm */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f9e0; Background Rhythm (Dominant Freq)</div>
            <div className="card-body">
              {rhythmDist.map(d => (
                <HBar key={d.name} label={d.name} count={d.value} total={totalR}
                  color={RHYTHM_COLORS[d.name] || '#6b7280'} />
              ))}
              <div className="text-muted mt-2" style={{ fontSize: '0.7rem' }}>
                Based on dominant spectral peak. Normal adult: Alpha 8–13 Hz.
              </div>
            </div>
          </div>
        </div>

        {/* Prediction Distribution */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f916; AI Prediction Distribution</div>
            <div className="card-body">
              {predDist.map((d, i) => (
                <HBar key={d.name} label={d.name} count={d.value} total={totalP}
                  color={PRED_COLORS[i % PRED_COLORS.length]} />
              ))}
              <div className="text-muted mt-2" style={{ fontSize: '0.7rem' }}>
                AI model label. Neurophysiologist validates each prediction in AI Validation tab.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   EEG INVENTORY TAB
═══════════════════════════════════════════════════════════════════════════ */
function InventoryTab({ bd }) {
  const [filter, setFilter] = useState('');
  const [sortBy, setSortBy] = useState('id');
  const [sortDir, setSortDir] = useState('asc');

  if (!bd) return <div className="spinner-border text-primary" />;

  const records = [...(bd.recording_inventory || [])];
  const filtered = filter
    ? records.filter(r =>
        r.patient_id?.toLowerCase().includes(filter.toLowerCase()) ||
        r.disease?.toLowerCase().includes(filter.toLowerCase()) ||
        r.signal_quality?.toLowerCase().includes(filter.toLowerCase()) ||
        r.background_rhythm?.toLowerCase().includes(filter.toLowerCase())
      )
    : records;

  filtered.sort((a, b) => {
    const va = a[sortBy] ?? '';
    const vb = b[sortBy] ?? '';
    return sortDir === 'asc'
      ? (va > vb ? 1 : va < vb ? -1 : 0)
      : (va < vb ? 1 : va > vb ? -1 : 0);
  });

  const toggle = col => {
    if (sortBy === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortBy(col); setSortDir('asc'); }
  };

  const Th = ({ col, label }) => (
    <th onClick={() => toggle(col)} style={{ cursor: 'pointer', whiteSpace: 'nowrap' }}>
      {label} {sortBy === col ? (sortDir === 'asc' ? '↑' : '↓') : ''}
    </th>
  );

  return (
    <div>
      <div className="mb-3">
        <input className="form-control form-control-sm w-auto d-inline-block"
          placeholder="Filter by patient, disease, quality, rhythm…"
          value={filter} onChange={e => setFilter(e.target.value)} />
        <span className="text-muted small ms-2">{filtered.length} recordings</span>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold">&#x1f4c2; EEG Recording Inventory</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0 small">
              <thead className="table-light">
                <tr>
                  <Th col="id" label="ID" />
                  <Th col="patient_id" label="Patient" />
                  <Th col="file" label="File" />
                  <Th col="disease" label="Disease" />
                  <Th col="n_channels" label="Ch" />
                  <Th col="sampling_rate" label="SR (Hz)" />
                  <Th col="duration_hrs" label="Dur (h)" />
                  <Th col="signal_quality" label="Quality" />
                  <Th col="flat_channels" label="Flat Ch" />
                  <Th col="background_rhythm" label="Background" />
                  <Th col="predicted_label" label="AI Label" />
                  <Th col="confidence" label="Conf" />
                </tr>
              </thead>
              <tbody>
                {filtered.map((r, i) => (
                  <tr key={i}>
                    <td className="text-muted">{r.id}</td>
                    <td><span className="badge bg-secondary">{r.patient_id}</span></td>
                    <td className="text-muted" style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.file}</td>
                    <td>{r.disease}</td>
                    <td>{r.n_channels}</td>
                    <td>{r.sampling_rate}</td>
                    <td>{r.duration_hrs ?? '—'}</td>
                    <td>
                      <span className="badge" style={{ background: qColor(r.signal_quality), color: '#fff', fontSize: '0.68rem' }}>
                        {r.signal_quality}
                      </span>
                    </td>
                    <td style={{ color: r.flat_channels > 0 ? '#ef4444' : '#374151' }}>{r.flat_channels}</td>
                    <td className="text-muted small">{r.background_rhythm}</td>
                    <td>{r.predicted_label}</td>
                    <td>
                      <span style={{ color: confColor(r.confidence), fontWeight: 600 }}>
                        {r.confidence != null ? r.confidence.toFixed(2) : '—'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div className="text-muted small mt-2">
        Ch = channels · SR = sampling rate · Dur = duration hours · Flat Ch = flat/dead channels
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SPECTRAL FEATURES TAB
═══════════════════════════════════════════════════════════════════════════ */
function SpectralTab({ bd }) {
  const [filter, setFilter] = useState('');

  if (!bd) return <div className="spinner-border text-primary" />;

  const rows = (bd.spectral_features || []).filter(r =>
    !filter || r.patient_id?.toLowerCase().includes(filter.toLowerCase())
  );

  const fmt = v => (v != null ? Number(v).toFixed(3) : '—');

  return (
    <div>
      <div className="mb-3">
        <input className="form-control form-control-sm w-auto d-inline-block"
          placeholder="Filter by patient…"
          value={filter} onChange={e => setFilter(e.target.value)} />
        <span className="text-muted small ms-2">{rows.length} records</span>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold">&#x1f4c8; Spectral &amp; Non-linear Features</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0 small">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Rec ID</th>
                  <th>Sp.Entropy</th>
                  <th>Hjorth Mob</th>
                  <th>Hjorth Cmplx</th>
                  <th>ApEn</th>
                  <th>SampEn</th>
                  <th>Hurst</th>
                  <th>DFA α</th>
                  <th>LZ Cmplx</th>
                  <th>Kurtosis</th>
                  <th>Skewness</th>
                  <th>Autocorr</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>
                    <td><span className="badge bg-secondary">{r.patient_id}</span></td>
                    <td className="text-muted">{r.recording_id}</td>
                    <td>{fmt(r.spectral_entropy)}</td>
                    <td>{fmt(r.hjorth_mobility)}</td>
                    <td>{fmt(r.hjorth_complexity)}</td>
                    <td>{fmt(r.approx_entropy)}</td>
                    <td>{fmt(r.sample_entropy)}</td>
                    <td style={{ color: r.hurst_exponent > 0.7 ? '#6366f1' : '#374151' }}>{fmt(r.hurst_exponent)}</td>
                    <td>{fmt(r.dfa_alpha)}</td>
                    <td>{fmt(r.lz_complexity)}</td>
                    <td>{fmt(r.kurtosis)}</td>
                    <td>{fmt(r.skewness)}</td>
                    <td>{fmt(r.autocorr)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div className="text-muted small mt-2">
        Sp.Entropy = Spectral Entropy · Mob = Hjorth Mobility · Cmplx = Hjorth Complexity ·
        ApEn = Approximate Entropy · SampEn = Sample Entropy · Hurst &gt;0.5 = persistent ·
        DFA α &gt;1 = long-range correlation · LZ = Lempel-Ziv Complexity
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   AI VALIDATION TAB
═══════════════════════════════════════════════════════════════════════════ */
function ValidationTab({ bd }) {
  const [filter, setFilter] = useState('');
  const [qFilter, setQFilter] = useState('All');

  if (!bd) return <div className="spinner-border text-primary" />;

  const rows = (bd.ai_validation || []).filter(r => {
    const matchText = !filter || r.patient_id?.toLowerCase().includes(filter.toLowerCase()) ||
      r.predicted_label?.toLowerCase().includes(filter.toLowerCase());
    const matchQ = qFilter === 'All' || r.signal_quality === qFilter;
    return matchText && matchQ;
  });

  return (
    <div>
      <div className="d-flex gap-2 mb-3 flex-wrap">
        <input className="form-control form-control-sm w-auto"
          placeholder="Filter by patient or label…"
          value={filter} onChange={e => setFilter(e.target.value)} />
        <select className="form-select form-select-sm w-auto"
          value={qFilter} onChange={e => setQFilter(e.target.value)}>
          <option value="All">All Quality</option>
          <option value="Good">Good</option>
          <option value="Fair">Fair</option>
          <option value="Poor">Poor</option>
        </select>
        <span className="text-muted small align-self-center">{rows.length} records</span>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold">&#x1f916; AI Label Validation Queue</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0 small">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Rec ID</th>
                  <th>AI Label</th>
                  <th>Confidence</th>
                  <th>Signal Quality</th>
                  <th>Review Status</th>
                  <th>Top Probabilities</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const probs = r.class_probabilities || {};
                  const topProbs = Object.entries(probs)
                    .sort(([, a], [, b]) => b - a)
                    .slice(0, 3);
                  return (
                    <tr key={i}>
                      <td><span className="badge bg-secondary">{r.patient_id}</span></td>
                      <td className="text-muted">{r.recording_id}</td>
                      <td>
                        <span className="badge bg-primary" style={{ fontSize: '0.68rem' }}>
                          {r.predicted_label}
                        </span>
                      </td>
                      <td>
                        <span style={{ color: confColor(r.confidence), fontWeight: 600 }}>
                          {r.confidence != null ? r.confidence.toFixed(2) : '—'}
                        </span>
                      </td>
                      <td>
                        <span className="badge" style={{ background: qColor(r.signal_quality), color: '#fff', fontSize: '0.68rem' }}>
                          {r.signal_quality}
                        </span>
                      </td>
                      <td>
                        <span className="badge bg-warning text-dark" style={{ fontSize: '0.65rem' }}>
                          {r.review_status}
                        </span>
                      </td>
                      <td className="text-muted" style={{ fontSize: '0.7rem' }}>
                        {topProbs.map(([label, prob]) =>
                          `${label}: ${(prob * 100).toFixed(0)}%`
                        ).join(' · ')}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div className="text-muted small mt-2">
        Neurophysiologist reviews AI predictions against EEG morphology and clinical context.
        Poor signal quality recordings require manual override before filing.
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   DEFINITIONS TAB
═══════════════════════════════════════════════════════════════════════════ */
function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading definitions…</div>;

  return (
    <div>
      <h6 className="fw-bold border-bottom pb-2 mb-3">&#x1f4da; Clinical Neurophysiology Glossary</h6>
      <div className="row g-2">
        {(Array.isArray(defs) ? defs : []).map((d, i) => (
          <div key={i} className="col-12 col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-body py-2">
                <div className="fw-bold small mb-1" style={{ color: '#3b82f6' }}>{d.term}</div>
                <div className="text-muted" style={{ fontSize: '0.78rem' }}>{d.definition}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 text-muted small">
        References: ACNS Guidelines · ILAE Classification · Niedermeyer's Electroencephalography ·
        Lopes da Silva EEG Reference Works.
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   ROOT PAGE
═══════════════════════════════════════════════════════════════════════════ */
export default function NeurophysiologistPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/neurophysiologist/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/neurophysiologist/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/neurophysiologist/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const TABS = [
    { id: 'overview',    label: 'Overview' },
    { id: 'inventory',   label: 'EEG Inventory' },
    { id: 'spectral',    label: 'Spectral Features' },
    { id: 'validation',  label: 'AI Validation' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f4e1; Clinical Neurophysiologist Dashboard</h3>
      <p className="text-muted small">{ov.subtitle}</p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewTab ov={ov} />}
      {tab === 'inventory'   && <InventoryTab bd={bd} />}
      {tab === 'spectral'    && <SpectralTab bd={bd} />}
      {tab === 'validation'  && <ValidationTab bd={bd} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  );
}
