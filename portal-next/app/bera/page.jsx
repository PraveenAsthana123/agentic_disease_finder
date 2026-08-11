'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = s =>
  s === 'Normal'   ? 'success' :
  s === 'Mild'     ? 'info' :
  s === 'Moderate' ? 'warning' :
  s === 'Severe'   ? 'danger' : 'secondary';

const patColor = p =>
  p === 'normal'                      ? 'success' :
  p === 'peripheral_hearing_loss'     ? 'info' :
  p === 'auditory_neuropathy'         ? 'primary' :
  p === 'brainstem_lesion'            ? 'danger' :
  p === 'acoustic_neuroma'            ? 'warning' :
  p === 'central_auditory_dysfunction'? 'dark' : 'secondary';

const patLabel = p =>
  p === 'normal'                      ? 'Normal' :
  p === 'peripheral_hearing_loss'     ? 'Peripheral Hearing Loss' :
  p === 'auditory_neuropathy'         ? 'Auditory Neuropathy' :
  p === 'brainstem_lesion'            ? 'Brainstem Lesion' :
  p === 'acoustic_neuroma'            ? 'Acoustic Neuroma' :
  p === 'central_auditory_dysfunction'? 'Central Auditory Dysfunction' : p;

const abnBadge = (val, ref, dir) => {
  const abn = dir === 'upper' ? val > ref : val < ref;
  return <span className={abn ? 'text-danger fw-bold' : 'text-success'}>{val}{abn ? ' !' : ''}</span>;
};

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'wave',        label: 'Wave Analysis' },
  { id: 'patients',   label: 'Per Patient' },
  { id: 'definitions', label: 'Definitions' },
];

export default function BERAPage() {
  const [ov,         setOv]         = useState(null);
  const [bd,         setBd]         = useState(null);
  const [defs,       setDefs]       = useState(null);
  const [tab,        setTab]        = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);
  const [err,        setErr]        = useState(null);

  useEffect(() => {
    fetch(`${API}/api/bera/overview`).then(r => r.json()).then(setOv).catch(e => setErr(e.message));
    fetch(`${API}/api/bera/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/bera/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (err) return <div className="p-4 alert alert-danger">Error: {err}</div>;
  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpis = ov.kpis || {};
  const sevDist = ov.severity_distribution || [];
  const patDist = ov.pattern_distribution || [];
  const earRates = ov.ear_abnormality_rates || [];
  const ptSum = ov.patient_summary || [];

  return (
    <div>
      <h3>&#x1f442; Brainstem Evoked Response Audiometry (BERA / ABR)</h3>
      <p className="text-muted small">
        Real clinical.db data — Wave I-V peak latencies, inter-peak latencies (IPL), auditory brainstem pathway integrity;
        click-evoked ABR; cochlea → CN VIII → brainstem auditory nuclei.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Studies',     value: kpis.total_studies,                color: 'primary' },
          { label: 'Abnormal',          value: kpis.abnormal_count,               color: 'danger' },
          { label: 'Abnormal Rate',     value: `${kpis.abnormal_rate_pct}%`,      color: kpis.abnormal_rate_pct > 30 ? 'danger' : 'warning' },
          { label: 'Mean Wave V (ms)',  value: kpis.mean_wave_v_latency_ms,       color: kpis.mean_wave_v_latency_ms > 6.0 ? 'danger' : 'success' },
          { label: 'Mean Wave V Amp',  value: `${kpis.mean_wave_v_amplitude_uv} µV`, color: kpis.mean_wave_v_amplitude_uv < 0.25 ? 'danger' : 'success' },
          { label: 'Mean I-V IPL',     value: `${kpis.mean_ipl_i_v_ms} ms`,      color: kpis.mean_ipl_i_v_ms > 4.5 ? 'danger' : 'success' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ─────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Severity Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Severity Distribution</div>
              <div className="card-body">
                {sevDist.map(s => (
                  <div key={s.severity} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${sevColor(s.severity)}`} style={{minWidth: 70}}>{s.severity}</span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{height: 18}}>
                        <div className={`progress-bar bg-${sevColor(s.severity)}`}
                             style={{width: `${kpis.total_studies ? (s.count / kpis.total_studies * 100) : 0}%`}}>
                          {s.count > 0 ? s.count : ''}
                        </div>
                      </div>
                    </div>
                    <span className="small text-muted">{s.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Diagnostic Pattern Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Diagnostic Patterns</div>
              <div className="card-body">
                {patDist.map(p => (
                  <div key={p.pattern} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${patColor(p.pattern)} me-1`}
                          style={{minWidth: 80, fontSize: '0.65rem'}}>{patLabel(p.pattern)}</span>
                    <div className="flex-grow-1 mx-1">
                      <div className="progress" style={{height: 18}}>
                        <div className={`progress-bar bg-${patColor(p.pattern)}`}
                             style={{width: `${kpis.total_studies ? (p.count / kpis.total_studies * 100) : 0}%`}}>
                          {p.count > 0 ? p.count : ''}
                        </div>
                      </div>
                    </div>
                    <span className="small text-muted">{p.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Ear Abnormality Rates */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Abnormality by Ear</div>
              <div className="card-body">
                {earRates.map(e => (
                  <div key={e.ear} className="mb-3">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className="fw-semibold">{e.ear} Ear ({e.ear === 'Right' ? 'AD' : 'AS'})</span>
                      <span>{e.abnormal}/{e.total} ({e.rate_pct}%)</span>
                    </div>
                    <div className="progress" style={{height: 16}}>
                      <div className={`progress-bar bg-${e.rate_pct > 20 ? 'danger' : e.rate_pct > 10 ? 'warning' : 'success'}`}
                           style={{width: `${e.rate_pct}%`}} />
                    </div>
                  </div>
                ))}
                <div className="mt-3 small text-muted">
                  <strong>Reference thresholds:</strong><br/>
                  Wave V latency ≤6.00 ms<br/>
                  Wave V amplitude ≥0.25 µV<br/>
                  I-V IPL ≤4.50 ms<br/>
                  Inter-aural V diff ≤0.30 ms
                </div>
              </div>
            </div>
          </div>

          {/* Patient Summary Table */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Patient Summary (Abnormal first)</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr>
                      <th>Patient</th><th>Age</th><th>Disease</th>
                      <th>Severity</th><th>Pattern</th>
                      <th>Inter-Aural Diff</th><th>Abnormal Ears</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ptSum.map((p, i) => (
                      <tr key={i}>
                        <td className="fw-semibold small">{p.name}</td>
                        <td>{p.age}</td>
                        <td className="small">{p.disease}</td>
                        <td><span className={`badge bg-${sevColor(p.overall_severity)}`}>{p.overall_severity}</span></td>
                        <td><span className={`badge bg-${patColor(p.diagnostic_pattern)}`}
                                 style={{fontSize:'0.65rem'}}>{patLabel(p.diagnostic_pattern)}</span></td>
                        <td className={p.inter_aural_diff_ms > 0.3 ? 'text-danger fw-bold' : ''}>
                          {p.inter_aural_diff_ms} ms{p.inter_aural_diff_ms > 0.3 ? ' !' : ''}
                        </td>
                        <td>{p.abnormal_ears}/{p.total_ears}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Wave Analysis Tab ─────────────────────────────────── */}
      {tab === 'wave' && bd && (
        <div className="row">
          {/* Left Ear Summary */}
          {[bd.left_summary, bd.right_summary].map(ear => ear && (
            <div key={ear.ear} className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">{ear.ear}</div>
                <div className="card-body">
                  <table className="table table-sm mb-2">
                    <thead>
                      <tr><th>Parameter</th><th>Mean</th><th>Reference</th><th>Status</th></tr>
                    </thead>
                    <tbody>
                      {[
                        { label: 'Wave I Latency',   val: ear.mean_wave_i_ms,   ref: ear.refs.wave_i_latency_upper,   dir: 'upper', unit: 'ms' },
                        { label: 'Wave III Latency',  val: ear.mean_wave_iii_ms, ref: ear.refs.wave_iii_latency_upper, dir: 'upper', unit: 'ms' },
                        { label: 'Wave V Latency',    val: ear.mean_wave_v_ms,   ref: ear.refs.wave_v_latency_upper,   dir: 'upper', unit: 'ms' },
                        { label: 'Wave V Amplitude',  val: ear.mean_wave_v_amp_uv, ref: ear.refs.wave_v_amplitude_lower, dir: 'lower', unit: 'µV' },
                        { label: 'I-III IPL',         val: ear.mean_ipl_i_iii_ms, ref: ear.refs.ipl_i_iii_upper,      dir: 'upper', unit: 'ms' },
                        { label: 'III-V IPL',         val: ear.mean_ipl_iii_v_ms, ref: ear.refs.ipl_iii_v_upper,      dir: 'upper', unit: 'ms' },
                        { label: 'I-V IPL',           val: ear.mean_ipl_i_v_ms,   ref: ear.refs.ipl_i_v_upper,        dir: 'upper', unit: 'ms' },
                      ].map(row => {
                        const abn = row.dir === 'upper' ? row.val > row.ref : row.val < row.ref;
                        return (
                          <tr key={row.label}>
                            <td className="small">{row.label}</td>
                            <td className={abn ? 'text-danger fw-bold' : 'text-success'}>
                              {row.val} {row.unit}{abn ? ' !' : ''}
                            </td>
                            <td className="small text-muted">{row.dir === 'upper' ? '≤' : '≥'}{row.ref} {row.unit}</td>
                            <td>{abn ? <span className="badge bg-danger">Abn</span> : <span className="badge bg-success">OK</span>}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <div className="d-flex justify-content-between small text-muted">
                    <span>n={ear.count}</span>
                    <span>Abnormal: <strong className={ear.abnormal_pct > 20 ? 'text-danger' : 'text-success'}>{ear.abnormal_pct}%</strong></span>
                  </div>
                  {/* Severity mini bars */}
                  <div className="mt-2">
                    {Object.entries(ear.severity_dist || {}).map(([sev, cnt]) => (
                      <span key={sev} className={`badge bg-${sevColor(sev)} me-1`}>{sev}: {cnt}</span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}

          {/* Wave V Latency Histogram */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Wave V Latency Distribution (both ears)</div>
              <div className="card-body">
                {(bd.wave_v_latency_histogram || []).map(b => {
                  const maxC = Math.max(...(bd.wave_v_latency_histogram || []).map(x => x.count), 1);
                  return (
                    <div key={b.range} className="d-flex align-items-center mb-1">
                      <span className="small" style={{width: 60}}>{b.range} ms</span>
                      <div className="flex-grow-1">
                        <div className="progress" style={{height: 16}}>
                          <div className={`progress-bar bg-${b.abnormal ? 'danger' : 'primary'}`}
                               style={{width: `${(b.count / maxC) * 100}%`}}>
                            {b.count > 0 ? b.count : ''}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
                <div className="small text-muted mt-1">Red = abnormal range (Wave V &gt;6.00 ms)</div>
              </div>
            </div>
          </div>

          {/* Wave V Amplitude Histogram */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Wave V Amplitude Distribution (both ears)</div>
              <div className="card-body">
                {(bd.wave_v_amplitude_histogram || []).map(b => {
                  const maxC = Math.max(...(bd.wave_v_amplitude_histogram || []).map(x => x.count), 1);
                  return (
                    <div key={b.range} className="d-flex align-items-center mb-1">
                      <span className="small" style={{width: 70}}>{b.range} µV</span>
                      <div className="flex-grow-1">
                        <div className="progress" style={{height: 16}}>
                          <div className={`progress-bar bg-${b.low_range ? 'danger' : 'success'}`}
                               style={{width: `${(b.count / maxC) * 100}%`}}>
                            {b.count > 0 ? b.count : ''}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
                <div className="small text-muted mt-1">Red = low amplitude (&lt;0.25 µV)</div>
              </div>
            </div>
          </div>

          {/* I-V IPL Histogram */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">I-V Inter-Peak Latency Distribution</div>
              <div className="card-body">
                {(bd.ipl_i_v_histogram || []).map(b => {
                  const maxC = Math.max(...(bd.ipl_i_v_histogram || []).map(x => x.count), 1);
                  return (
                    <div key={b.range} className="d-flex align-items-center mb-1">
                      <span className="small" style={{width: 60}}>{b.range} ms</span>
                      <div className="flex-grow-1">
                        <div className="progress" style={{height: 16}}>
                          <div className={`progress-bar bg-${b.abnormal ? 'danger' : 'info'}`}
                               style={{width: `${(b.count / maxC) * 100}%`}}>
                            {b.count > 0 ? b.count : ''}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
                <div className="small text-muted mt-1">Red = prolonged I-V IPL (&gt;4.50 ms — brainstem conduction)</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Per Patient Tab ──────────────────────────────────── */}
      {tab === 'patients' && bd && (
        <div>
          {(bd.patient_details || []).map((pt, i) => (
            <div key={i} className="card shadow-sm mb-2">
              <div className="card-header d-flex justify-content-between align-items-center py-2"
                   style={{cursor: 'pointer'}}
                   onClick={() => setExpandedPt(expandedPt === i ? null : i)}>
                <div>
                  <span className="fw-bold">{pt.name}</span>
                  <span className="text-muted ms-2 small">Age {pt.age} | {pt.disease}</span>
                </div>
                <div>
                  <span className={`badge bg-${sevColor(pt.overall_severity)} me-1`}>{pt.overall_severity}</span>
                  <span className={`badge bg-${patColor(pt.diagnostic_pattern)}`}
                        style={{fontSize:'0.65rem'}}>{patLabel(pt.diagnostic_pattern)}</span>
                  {pt.inter_aural_abnormal &&
                    <span className="badge bg-danger ms-1">Inter-Aural !</span>}
                  <span className="ms-2 small">{expandedPt === i ? '\u25B2' : '\u25BC'}</span>
                </div>
              </div>
              {expandedPt === i && (
                <div className="card-body">
                  <div className="row">
                    {[pt.left, pt.right].map(ear => ear && (
                      <div key={ear.ear} className="col-md-6 mb-2">
                        <h6 className="fw-bold">{ear.ear} Ear ({ear.ear === 'Right' ? 'AD' : 'AS'})</h6>
                        <table className="table table-sm table-bordered">
                          <thead>
                            <tr><th>Parameter</th><th>Value</th><th>Reference</th></tr>
                          </thead>
                          <tbody>
                            {[
                              { label: 'Wave I Lat',   val: ear.wave_i_latency_ms,   ref: ear.wave_i_ref,     dir: 'upper', unit: 'ms', abn: ear.wave_i_abnormal },
                              { label: 'Wave III Lat',  val: ear.wave_iii_latency_ms, ref: ear.wave_iii_ref,   dir: 'upper', unit: 'ms', abn: ear.wave_iii_abnormal },
                              { label: 'Wave V Lat',    val: ear.wave_v_latency_ms,   ref: ear.wave_v_ref,     dir: 'upper', unit: 'ms', abn: ear.wave_v_abnormal },
                              { label: 'Wave V Amp',    val: ear.wave_v_amplitude_uv, ref: ear.wave_v_amp_ref, dir: 'lower', unit: 'µV', abn: ear.wave_v_amp_abnormal },
                              { label: 'I-III IPL',     val: ear.ipl_i_iii_ms,        ref: ear.ipl_i_iii_ref,  dir: 'upper', unit: 'ms', abn: ear.ipl_i_iii_abnormal },
                              { label: 'III-V IPL',     val: ear.ipl_iii_v_ms,        ref: ear.ipl_iii_v_ref,  dir: 'upper', unit: 'ms', abn: ear.ipl_iii_v_abnormal },
                              { label: 'I-V IPL',       val: ear.ipl_i_v_ms,          ref: ear.ipl_i_v_ref,    dir: 'upper', unit: 'ms', abn: ear.ipl_i_v_abnormal },
                            ].map(row => (
                              <tr key={row.label}>
                                <td className="small">{row.label}</td>
                                <td className={row.abn ? 'text-danger fw-bold' : 'text-success'}>
                                  {row.val} {row.unit}{row.abn ? ' !' : ''}
                                </td>
                                <td className="small text-muted">{row.dir === 'upper' ? '≤' : '≥'}{row.ref} {row.unit}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <span className={`badge bg-${sevColor(ear.severity)}`}>{ear.severity}</span>
                      </div>
                    ))}
                    {pt.inter_aural_diff_ms !== undefined && (
                      <div className="col-12 small mt-1">
                        <span className="fw-semibold">Inter-Aural Wave V Diff: </span>
                        <span className={pt.inter_aural_abnormal ? 'text-danger fw-bold' : 'text-success'}>
                          {pt.inter_aural_diff_ms} ms
                          {pt.inter_aural_abnormal ? ' ! (>0.30 ms — asymmetry marker)' : ' (normal ≤0.30 ms)'}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── Definitions Tab ──────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          {/* Protocol */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Protocol</div>
              <div className="card-body small">
                {defs.protocol && (
                  <>
                    <p>{defs.protocol.description || defs.protocol}</p>
                    {defs.protocol.stimulus && <p><strong>Stimulus:</strong> {defs.protocol.stimulus}</p>}
                    {defs.protocol.recording && <p><strong>Recording:</strong> {defs.protocol.recording}</p>}
                    {defs.protocol.indications && (
                      <>
                        <h6>Indications</h6>
                        <ul>{defs.protocol.indications.map((ind, i) => <li key={i}>{ind}</li>)}</ul>
                      </>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Parameters */}
          {defs.parameters && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Parameters ({defs.parameters.length})</div>
                <div className="card-body p-0">
                  <table className="table table-sm table-striped mb-0">
                    <thead><tr><th>Parameter</th><th>Unit</th><th>Description</th></tr></thead>
                    <tbody>
                      {defs.parameters.map((p, i) => (
                        <tr key={i}>
                          <td className="fw-semibold small">{p.name}</td>
                          <td className="small">{p.unit}</td>
                          <td className="small">{p.description}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Reference Ranges */}
          {defs.reference_ranges && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Reference Ranges (Click-Evoked ABR, Adults)</div>
                <div className="card-body small">
                  {typeof defs.reference_ranges === 'object' && !Array.isArray(defs.reference_ranges) ? (
                    Object.entries(defs.reference_ranges).map(([k, v]) => (
                      <div key={k} className="mb-2">
                        <strong>{k.replace(/_/g, ' ')}:</strong>{' '}
                        {typeof v === 'object' ? (
                          <ul className="mb-0">
                            {Object.entries(v).map(([kk, vv]) => (
                              <li key={kk}><strong>{kk.replace(/_/g, ' ')}:</strong> {vv}</li>
                            ))}
                          </ul>
                        ) : v}
                      </div>
                    ))
                  ) : <p>{JSON.stringify(defs.reference_ranges)}</p>}
                </div>
              </div>
            </div>
          )}

          {/* Diagnostic Patterns */}
          {defs.diagnostic_patterns && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Diagnostic Patterns</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <tbody>
                      {defs.diagnostic_patterns.map((p, i) => (
                        <tr key={i}>
                          <td style={{width: 140}}>
                            <span className={`badge bg-${patColor(p.pattern || p.id || '')}`}>
                              {patLabel(p.pattern || p.id || '')}
                            </span>
                          </td>
                          <td className="small">{p.description || p.criteria || ''}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Severity Levels */}
              {defs.severity_levels && (
                <div className="card shadow-sm mt-3">
                  <div className="card-header fw-bold">Severity Levels</div>
                  <div className="card-body p-0">
                    <table className="table table-sm mb-0">
                      <tbody>
                        {defs.severity_levels.map((s, i) => (
                          <tr key={i}>
                            <td style={{width: 80}}>
                              <span className={`badge bg-${sevColor(s.level)}`}>{s.level}</span>
                            </td>
                            <td className="small">{s.criteria}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Clinical Significance */}
          {defs.clinical_significance && (
            <div className="col-12 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Clinical Significance</div>
                <div className="card-body small">{defs.clinical_significance}</div>
              </div>
            </div>
          )}

          {/* Reference */}
          {defs.reference && (
            <div className="col-12 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Key References</div>
                <div className="card-body small">
                  {Array.isArray(defs.reference)
                    ? <ul>{defs.reference.map((r, i) => <li key={i}>{r}</li>)}</ul>
                    : <p>{defs.reference}</p>}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
