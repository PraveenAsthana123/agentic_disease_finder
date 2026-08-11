'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = s =>
  s === 'Normal'   ? 'success' :
  s === 'Mild'     ? 'info' :
  s === 'Moderate' ? 'warning' :
  s === 'Severe'   ? 'danger' : 'secondary';

const patColor = p =>
  p === 'normal'           ? 'success' :
  p === 'postsynaptic_nmj' ? 'warning' :
  p === 'presynaptic_nmj'  ? 'danger' :
  p === 'mixed_nmj'        ? 'primary' : 'secondary';

const patLabel = p =>
  p === 'normal'           ? 'Normal' :
  p === 'postsynaptic_nmj' ? 'Postsynaptic NMJ' :
  p === 'presynaptic_nmj'  ? 'Presynaptic NMJ' :
  p === 'mixed_nmj'        ? 'Mixed NMJ' : p;

const TABS = [
  { id: 'overview',   label: 'Overview' },
  { id: 'sites',      label: 'Site Analysis' },
  { id: 'patients',   label: 'Per Patient' },
  { id: 'definitions',label: 'Definitions' },
];

export default function RNSPage() {
  const [ov,   setOv]   = useState(null);
  const [bk,   setBk]   = useState(null);
  const [def,  setDef]  = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [sort, setSort] = useState('severity');
  const [filt, setFilt] = useState('');
  const [err,  setErr]  = useState(null);

  useEffect(() => {
    fetch(`${API}/api/rns/overview`).then(r => r.json()).then(setOv).catch(e => setErr(e.message));
    fetch(`${API}/api/rns/breakdown`).then(r => r.json()).then(setBk).catch(() => {});
    fetch(`${API}/api/rns/definitions`).then(r => r.json()).then(setDef).catch(() => {});
  }, []);

  if (err) return <div className="p-4 alert alert-danger">Error: {err}</div>;
  if (!ov)  return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpis = ov.kpis || {};
  const sevDist = ov.severity_distribution || [];
  const patDist = ov.diagnostic_pattern_distribution || [];
  const siteRates = ov.site_abnormality_rates || [];
  const patients = ov.patient_summary || [];
  const rnsSummary = bk?.rns_summary || [];
  const decrementHist = bk?.decrement_histogram || [];
  const facilitationHist = bk?.facilitation_histogram || [];

  /* filter + sort patients */
  const filteredPats = patients
    .filter(p =>
      !filt ||
      (p.name || '').toLowerCase().includes(filt.toLowerCase()) ||
      (p.patient_id || '').toLowerCase().includes(filt.toLowerCase()) ||
      (p.overall_severity || '').toLowerCase().includes(filt.toLowerCase()) ||
      patLabel(p.diagnostic_pattern).toLowerCase().includes(filt.toLowerCase())
    )
    .sort((a, b) => {
      const sevOrder = { Severe: 0, Moderate: 1, Mild: 2, Normal: 3 };
      if (sort === 'severity') return (sevOrder[a.overall_severity] ?? 4) - (sevOrder[b.overall_severity] ?? 4);
      if (sort === 'abnormal_sites') return b.abnormal_sites - a.abnormal_sites;
      if (sort === 'name') return (a.name || a.patient_id).localeCompare(b.name || b.patient_id);
      if (sort === 'age') return b.age - a.age;
      return 0;
    });

  /* bar width helper */
  const pct = (v, max) => Math.round((v / (max || 1)) * 100);

  return (
    <div className="container-fluid py-3">
      <h3>&#x26a1; Repetitive Nerve Stimulation (RNS) Dashboard</h3>
      <p className="text-muted small">
        Neuromuscular junction (NMJ) transmission analysis — decrement &amp; facilitation testing,
        CMAP amplitude, diagnostic pattern classification (Myasthenia Gravis / LEMS / Mixed NMJ),
        proximal vs distal site comparison. Real clinical.db data.
      </p>

      {/* KPI Row */}
      <div className="row mb-3">
        {[
          { label: 'Total Studies',       value: kpis.total_studies,           color: 'primary' },
          { label: 'Abnormal',            value: `${kpis.abnormal_count} (${kpis.abnormal_rate_pct}%)`, color: 'danger' },
          { label: 'Mean Decrement',      value: `${kpis.mean_decrement_pct}%`, color: 'warning' },
          { label: 'Mean Facilitation',   value: `${kpis.mean_facilitation_pct}%`, color: 'info' },
          { label: 'Mean CMAP Amplitude', value: `${kpis.mean_baseline_cmap_mv} mV`, color: 'success' },
        ].map(k => (
          <div key={k.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h4 mb-0 text-${k.color}`}>{k.value ?? '—'}</div>
                <div className="text-muted" style={{ fontSize: '0.72rem' }}>{k.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Severity distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-body">
                <h6 className="card-title text-muted">Severity Distribution</h6>
                {sevDist.map(s => (
                  <div key={s.severity} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span><span className={`badge bg-${sevColor(s.severity)} me-1`}>{s.severity}</span></span>
                      <span className="text-muted">{s.count}</span>
                    </div>
                    <div className="progress" style={{ height: 10 }}>
                      <div
                        className={`progress-bar bg-${sevColor(s.severity)}`}
                        style={{ width: `${pct(s.count, kpis.total_studies)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Diagnostic patterns */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-body">
                <h6 className="card-title text-muted">Diagnostic Patterns</h6>
                {patDist.map(p => (
                  <div key={p.pattern} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span><span className={`badge bg-${patColor(p.pattern)} me-1`}>{p.label}</span></span>
                      <span className="text-muted">{p.count}</span>
                    </div>
                    <div className="progress" style={{ height: 10 }}>
                      <div
                        className={`progress-bar bg-${patColor(p.pattern)}`}
                        style={{ width: `${pct(p.count, kpis.total_studies)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Site abnormality rates */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-body">
                <h6 className="card-title text-muted">Site Abnormality Rates</h6>
                {siteRates.map(s => (
                  <div key={s.site} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className="text-truncate" style={{ maxWidth: 150 }}>{s.site}</span>
                      <span className={`badge bg-${s.type === 'proximal' ? 'primary' : 'secondary'}`}>
                        {s.rate_pct}%
                      </span>
                    </div>
                    <div className="progress" style={{ height: 8 }}>
                      <div className="progress-bar bg-warning" style={{ width: `${s.rate_pct}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Clinical note */}
          <div className="col-12">
            <div className="alert alert-info border-0 shadow-sm small">
              <strong>RNS Protocol (AANEM):</strong> Supramaximal 3 Hz train (6-10 stimuli),
              decrement &gt;10% between stimulus 1-4 = abnormal. Post-exercise facilitation
              &gt;100% = presynaptic (LEMS). Postsynaptic (MG): decrement ±facilitation &lt;100%.
              Exhaustion tested 2-4 min post-exercise.
            </div>
          </div>
        </div>
      )}

      {/* ── Site Analysis ── */}
      {tab === 'sites' && (
        <div>
          {/* Decrement histogram */}
          {decrementHist.length > 0 && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6 className="card-title text-muted">Decrement % Distribution (3 Hz)</h6>
                <div className="d-flex align-items-end gap-1" style={{ height: 80 }}>
                  {decrementHist.map((b, i) => (
                    <div key={i} className="text-center flex-fill">
                      <div
                        className="bg-warning rounded-top"
                        style={{
                          height: `${pct(b.count, Math.max(...decrementHist.map(x => x.count)))}%`,
                          minHeight: b.count > 0 ? 4 : 0
                        }}
                        title={`${b.bin}: ${b.count}`}
                      />
                      <div className="text-muted" style={{ fontSize: '0.6rem' }}>{b.bin}</div>
                    </div>
                  ))}
                </div>
                <div className="text-muted small mt-1">Threshold: 10% (vertical dashed line)</div>
              </div>
            </div>
          )}

          {/* Facilitation histogram */}
          {facilitationHist.length > 0 && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6 className="card-title text-muted">Post-Exercise Facilitation % Distribution</h6>
                <div className="d-flex align-items-end gap-1" style={{ height: 80 }}>
                  {facilitationHist.map((b, i) => (
                    <div key={i} className="text-center flex-fill">
                      <div
                        className="bg-info rounded-top"
                        style={{
                          height: `${pct(b.count, Math.max(...facilitationHist.map(x => x.count)))}%`,
                          minHeight: b.count > 0 ? 4 : 0
                        }}
                        title={`${b.bin}: ${b.count}`}
                      />
                      <div className="text-muted" style={{ fontSize: '0.6rem' }}>{b.bin}</div>
                    </div>
                  ))}
                </div>
                <div className="text-muted small mt-1">Facilitation &gt;100% = presynaptic (LEMS)</div>
              </div>
            </div>
          )}

          {/* Per-site table */}
          {rnsSummary.length > 0 && (
            <div className="card shadow-sm border-0">
              <div className="card-body p-0">
                <h6 className="card-title text-muted p-3 mb-0">Per-Site Metrics</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Site</th>
                        <th>Type</th>
                        <th>Innervation</th>
                        <th>Mean Dec%</th>
                        <th>Mean Fac%</th>
                        <th>Mean CMAP (mV)</th>
                        <th>Abnormal%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rnsSummary.map((s, i) => (
                        <tr key={i}>
                          <td className="small">{s.site}</td>
                          <td>
                            <span className={`badge bg-${s.type === 'proximal' ? 'primary' : 'secondary'} small`}>
                              {s.type}
                            </span>
                          </td>
                          <td className="small text-muted">{s.innervation}</td>
                          <td>
                            <span className={s.mean_decrement_pct > 10 ? 'text-danger fw-bold' : ''}>
                              {s.mean_decrement_pct?.toFixed(1)}%
                            </span>
                          </td>
                          <td>
                            <span className={s.mean_facilitation_pct > 100 ? 'text-info fw-bold' : ''}>
                              {s.mean_facilitation_pct?.toFixed(1)}%
                            </span>
                          </td>
                          <td>{s.mean_cmap_mv?.toFixed(1)}</td>
                          <td>
                            <span className={`badge bg-${s.abnormal_pct > 15 ? 'danger' : s.abnormal_pct > 0 ? 'warning' : 'success'}`}>
                              {s.abnormal_pct?.toFixed(1)}%
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Per Patient ── */}
      {tab === 'patients' && (
        <div>
          <div className="d-flex gap-2 mb-3 flex-wrap">
            <input
              className="form-control form-control-sm"
              style={{ maxWidth: 220 }}
              placeholder="Search patient / severity / pattern…"
              value={filt}
              onChange={e => setFilt(e.target.value)}
            />
            <select
              className="form-select form-select-sm"
              style={{ maxWidth: 160 }}
              value={sort}
              onChange={e => setSort(e.target.value)}
            >
              <option value="severity">Sort: Severity</option>
              <option value="abnormal_sites">Sort: Abnormal Sites</option>
              <option value="name">Sort: Name</option>
              <option value="age">Sort: Age</option>
            </select>
            <span className="text-muted small align-self-center">{filteredPats.length} patients</span>
          </div>

          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr>
                  <th>Patient ID</th>
                  <th>Name</th>
                  <th>Age</th>
                  <th>Overall Severity</th>
                  <th>Diagnostic Pattern</th>
                  <th>Abnormal Sites</th>
                </tr>
              </thead>
              <tbody>
                {filteredPats.map((p, i) => (
                  <tr key={i}>
                    <td className="small text-muted">{p.patient_id}</td>
                    <td className="small">{p.name || <em className="text-muted">—</em>}</td>
                    <td className="small">{p.age}</td>
                    <td>
                      <span className={`badge bg-${sevColor(p.overall_severity)}`}>
                        {p.overall_severity}
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${patColor(p.diagnostic_pattern)}`}>
                        {patLabel(p.diagnostic_pattern)}
                      </span>
                    </td>
                    <td>
                      <span className={p.abnormal_sites > 0 ? 'text-danger fw-bold' : 'text-success'}>
                        {p.abnormal_sites}/{p.total_sites}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && def && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-body">
                <h6 className="card-title">{def.title}</h6>
                <p className="small text-muted">{def.protocol?.description}</p>
                <h6 className="mt-3">Indications</h6>
                <ul className="small">
                  {(def.protocol?.indications || []).map((ind, i) => (
                    <li key={i}>{ind}</li>
                  ))}
                </ul>
                <div className="text-muted small mt-2">
                  Standard: {def.protocol?.standard}
                </div>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6 className="card-title">Reference Ranges</h6>
                <table className="table table-sm">
                  <tbody>
                    {Object.entries(def.reference_ranges || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td className="small text-muted">{k.replace(/_/g, ' ')}</td>
                        <td className="small">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm border-0">
              <div className="card-body">
                <h6 className="card-title">Severity Levels</h6>
                <table className="table table-sm">
                  <tbody>
                    {Object.entries(def.severity_levels || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td>
                          <span className={`badge bg-${sevColor(k.charAt(0).toUpperCase() + k.slice(1))}`}>
                            {k.charAt(0).toUpperCase() + k.slice(1)}
                          </span>
                        </td>
                        <td className="small text-muted">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Diagnostic patterns */}
          <div className="col-12">
            <div className="card shadow-sm border-0">
              <div className="card-body">
                <h6 className="card-title">Diagnostic Patterns</h6>
                <div className="row">
                  {Object.entries(def.diagnostic_patterns || {}).map(([k, v]) => (
                    <div key={k} className="col-md-6 mb-2">
                      <div className="border rounded p-2 small">
                        <strong className={`text-${patColor(k)}`}>
                          {k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                        </strong>
                        <div className="text-muted mt-1">{v}</div>
                      </div>
                    </div>
                  ))}
                </div>
                {def.reference && (
                  <div className="text-muted small mt-2">
                    Reference: {def.reference}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
