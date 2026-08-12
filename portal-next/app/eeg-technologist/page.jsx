'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

/* ─── colour helpers ─────────────────────────────────────────────────────── */
const gradeColor = g =>
  g === 'Good'      ? '#22c55e' :
  g === 'Fair'      ? '#f59e0b' :
  g === 'Poor'      ? '#ef4444' :
  g === 'Excellent' ? '#10b981' : '#94a3b8';

const sevColor = s =>
  s === 'mild'     ? '#f59e0b' :
  s === 'moderate' ? '#f97316' :
  s === 'severe'   ? '#ef4444' : '#94a3b8';

const ARTIFACT_COLORS = {
  eye_blink:    '#6366f1',
  muscle:       '#ef4444',
  movement:     '#f97316',
  electrode_pop:'#f59e0b',
  sweat:        '#8b5cf6',
  ECG:          '#ec4899',
};

const REC_COLORS = {
  routine:     '#3b82f6',
  video_eeg:   '#6366f1',
  ambulatory:  '#22c55e',
  LTM:         '#f59e0b',
};

/* ─── shared components ─────────────────────────────────────────────────── */
function KPICard({ label, value, unit, color = '#3b82f6' }) {
  return (
    <div className="col-6 col-md mb-2">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="h5 mb-0 fw-bold" style={{ color }}>{value ?? '—'}</div>
          {unit && <div className="text-muted" style={{ fontSize: '0.68rem' }}>{unit}</div>}
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

function ActivationBar({ label, pct, color }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-bold">{pct}%</span>
      </div>
      <div style={{ background: '#e5e7eb', borderRadius: 4, height: 10 }}>
        <div style={{ width: `${pct}%`, background: color || '#3b82f6', borderRadius: 4, height: 10 }} />
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   OVERVIEW TAB
═══════════════════════════════════════════════════════════════════════════ */
function OverviewTab({ ov }) {
  if (!ov) return <div className="spinner-border text-primary" />;

  const recTypeDist   = ov.recording_type_distribution   || {};
  const montageDist   = ov.montage_distribution          || {};
  const srDist        = ov.sampling_rate_distribution    || {};
  const qGradeDist    = ov.channel_quality_grade_distribution || {};
  const impGradeDist  = ov.impedance_grade_distribution  || {};
  const artTypeDist   = ov.artifact_type_distribution    || {};
  const artSevDist    = ov.artifact_severity_distribution || {};
  const activRates    = ov.activation_procedure_rates    || {};
  const coopDist      = ov.cooperation_distribution      || {};
  const stateDist     = ov.patient_state_distribution    || {};

  const totalRecType  = Object.values(recTypeDist).reduce((s, v) => s + v, 0);
  const totalMontage  = Object.values(montageDist).reduce((s, v) => s + v, 0);
  const totalSR       = Object.values(srDist).reduce((s, v) => s + v, 0);
  const totalQGrade   = Object.values(qGradeDist).reduce((s, v) => s + v, 0);
  const totalImpGrade = Object.values(impGradeDist).reduce((s, v) => s + v, 0);
  const totalArtType  = Object.values(artTypeDist).reduce((s, v) => s + v, 0);
  const totalArtSev   = Object.values(artSevDist).reduce((s, v) => s + v, 0);
  const totalCoop     = Object.values(coopDist).reduce((s, v) => s + v, 0);
  const totalState    = Object.values(stateDist).reduce((s, v) => s + v, 0);

  const KPI_COLORS = ['#3b82f6','#22c55e','#10b981','#6366f1','#f59e0b','#8b5cf6','#ec4899'];

  return (
    <div>
      {/* KPI Row */}
      <div className="row row-cols-2 row-cols-md-4 g-2 mb-4">
        {(ov.kpis || []).map((k, i) => (
          <KPICard key={i} label={k.label} value={k.value} unit={k.unit}
            color={KPI_COLORS[i % KPI_COLORS.length]} />
        ))}
      </div>

      <div className="row g-3 mb-3">
        {/* Recording Type */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f4fd;&#xfe0f; Recording Type</div>
            <div className="card-body">
              {Object.entries(recTypeDist).map(([k, v]) => (
                <HBar key={k} label={k} count={v} total={totalRecType}
                  color={REC_COLORS[k] || '#6b7280'} />
              ))}
            </div>
          </div>
        </div>

        {/* Montage */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f9e9; Montage Distribution</div>
            <div className="card-body">
              {Object.entries(montageDist).map(([k, v], i) => (
                <HBar key={k} label={k} count={v} total={totalMontage}
                  color={['#6366f1','#22c55e','#f59e0b'][i % 3]} />
              ))}
              <div className="text-muted mt-2" style={{ fontSize: '0.7rem' }}>
                Bipolar = chain referencing · Referential = common electrode · Average = average of all electrodes
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-3">
        {/* Channel Quality Grades */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f7e2; Channel Quality Grades</div>
            <div className="card-body">
              {['Good', 'Fair', 'Poor'].map(g => (
                <HBar key={g} label={g} count={qGradeDist[g] || 0} total={totalQGrade}
                  color={gradeColor(g)} />
              ))}
              <div className="text-muted mt-2" style={{ fontSize: '0.7rem' }}>
                Across all channels in all sessions (n={totalQGrade}). Target: &gt;80% Good.
              </div>
            </div>
          </div>
        </div>

        {/* Impedance Grades */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x26a1; Impedance Grades</div>
            <div className="card-body">
              {['Good', 'Fair', 'Poor'].map(g => (
                <HBar key={g} label={g} count={impGradeDist[g] || 0} total={totalImpGrade}
                  color={gradeColor(g)} />
              ))}
              <div className="text-muted mt-2" style={{ fontSize: '0.7rem' }}>
                Good: &lt;5 kΩ · Fair: 5–10 kΩ · Poor: &gt;10 kΩ (ACNS Guideline 1)
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-3">
        {/* Artifact Types */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x26a0;&#xfe0f; Artifact Type Distribution</div>
            <div className="card-body">
              {Object.entries(artTypeDist).map(([k, v]) => (
                <HBar key={k} label={k.replace('_', ' ')} count={v} total={totalArtType}
                  color={ARTIFACT_COLORS[k] || '#6b7280'} />
              ))}
              <div className="text-muted mt-2" style={{ fontSize: '0.7rem' }}>
                {totalArtType} total artifact annotations across {ov.kpis?.[0]?.value || '?'} recordings
              </div>
            </div>
          </div>
        </div>

        {/* Artifact Severity */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f4ca; Artifact Severity</div>
            <div className="card-body">
              {['mild', 'moderate', 'severe'].map(s => (
                <HBar key={s} label={s.charAt(0).toUpperCase() + s.slice(1)}
                  count={artSevDist[s] || 0} total={totalArtSev}
                  color={sevColor(s)} />
              ))}
              <div className="text-muted mt-2" style={{ fontSize: '0.7rem' }}>
                Mild = localized, brief · Moderate = multiple channels · Severe = obscures cerebral signal
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3">
        {/* Activation Procedures */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f9ea; Activation Procedure Coverage</div>
            <div className="card-body">
              {Object.entries(activRates).map(([k, v], i) => (
                <ActivationBar key={k} label={k} pct={v}
                  color={['#3b82f6','#6366f1','#f59e0b','#22c55e'][i % 4]} />
              ))}
              <div className="text-muted mt-2" style={{ fontSize: '0.7rem' }}>
                % of sessions where each activation was performed.
              </div>
            </div>
          </div>
        </div>

        {/* Patient State + Cooperation */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f9d8; Patient State &amp; Cooperation</div>
            <div className="card-body">
              <div className="mb-3">
                <div className="small fw-semibold mb-2 text-muted">Patient State</div>
                {Object.entries(stateDist).map(([k, v], i) => (
                  <HBar key={k} label={k.charAt(0).toUpperCase() + k.slice(1)}
                    count={v} total={totalState}
                    color={['#3b82f6','#8b5cf6','#6366f1'][i % 3]} />
                ))}
              </div>
              <div className="small fw-semibold mb-2 text-muted">Cooperation Level</div>
              {Object.entries(coopDist).map(([k, v], i) => (
                <HBar key={k} label={k.charAt(0).toUpperCase() + k.slice(1)}
                  count={v} total={totalCoop}
                  color={gradeColor(k.charAt(0).toUpperCase() + k.slice(1))} />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Recent Recordings */}
      {(ov.recent_recordings || []).length > 0 && (
        <div className="card shadow-sm mt-3">
          <div className="card-header fw-bold">&#x1f4cb; Recent Recordings</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0 small">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th>Type</th>
                    <th>Dur (min)</th>
                    <th>SR (Hz)</th>
                    <th>Montage</th>
                    <th>Study Date</th>
                  </tr>
                </thead>
                <tbody>
                  {ov.recent_recordings.map((r, i) => (
                    <tr key={i}>
                      <td><span className="badge bg-secondary">{r.patient_id}</span></td>
                      <td>
                        <span className="badge" style={{ background: REC_COLORS[r.recording_type] || '#6b7280', color: '#fff', fontSize: '0.65rem' }}>
                          {r.recording_type}
                        </span>
                      </td>
                      <td>{r.duration_min}</td>
                      <td>{r.sampling_rate}</td>
                      <td>{r.montage}</td>
                      <td className="text-muted">{r.study_date || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   CHANNEL QUALITY TAB
═══════════════════════════════════════════════════════════════════════════ */
function ChannelQualityTab({ bd }) {
  const [filter, setFilter] = useState('');
  if (!bd) return <div className="spinner-border text-primary" />;

  const patients = (bd.per_patient || []).filter(p =>
    !filter || p.patient_id?.toLowerCase().includes(filter.toLowerCase())
  );

  const snrHist = bd.snr_histogram || {};

  return (
    <div>
      {/* SNR Histogram */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">&#x1f4c8; SNR Distribution (all channels)</div>
        <div className="card-body">
          <div className="row g-2">
            {Object.entries(snrHist).map(([band, cnt], i) => {
              const total = Object.values(snrHist).reduce((s, v) => s + v, 0);
              const pct = total ? Math.round((cnt / total) * 100) : 0;
              return (
                <div key={band} className="col-6 col-md-3">
                  <div className="border rounded p-2 text-center">
                    <div className="h5 mb-0 fw-bold" style={{ color: ['#ef4444','#f59e0b','#3b82f6','#22c55e'][i] }}>
                      {cnt}
                    </div>
                    <div className="text-muted small">{band}</div>
                    <div className="text-muted" style={{ fontSize: '0.7rem' }}>{pct}%</div>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="text-muted mt-2" style={{ fontSize: '0.7rem' }}>
            SNR &ge;20 dB = diagnostic quality · &lt;10 dB = unacceptable for AI training
          </div>
        </div>
      </div>

      {/* Per-patient table */}
      <div className="mb-3">
        <input className="form-control form-control-sm w-auto d-inline-block"
          placeholder="Filter by patient…"
          value={filter} onChange={e => setFilter(e.target.value)} />
        <span className="text-muted small ms-2">{patients.length} patients</span>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold">&#x1f9ea; Per-Patient Channel Quality</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0 small">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Recording Type</th>
                  <th>Duration (min)</th>
                  <th>Usability %</th>
                  <th>Avg SNR (dB)</th>
                  <th>Impedance Pass %</th>
                  <th>Artifact Count</th>
                  <th>Artifact Types</th>
                  <th>HV</th>
                  <th>Photic</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i}>
                    <td><span className="badge bg-secondary">{p.patient_id}</span></td>
                    <td>
                      <span className="badge" style={{ background: REC_COLORS[p.recording_type] || '#6b7280', color: '#fff', fontSize: '0.65rem' }}>
                        {p.recording_type}
                      </span>
                    </td>
                    <td>{p.duration_min ?? '—'}</td>
                    <td>
                      <span style={{
                        color: p.channel_usability_pct >= 80 ? '#22c55e' :
                               p.channel_usability_pct >= 60 ? '#f59e0b' : '#ef4444',
                        fontWeight: 600
                      }}>
                        {p.channel_usability_pct != null ? p.channel_usability_pct + '%' : '—'}
                      </span>
                    </td>
                    <td>
                      <span style={{
                        color: p.avg_snr_db >= 20 ? '#22c55e' :
                               p.avg_snr_db >= 10 ? '#f59e0b' : '#ef4444',
                        fontWeight: 600
                      }}>
                        {p.avg_snr_db != null ? p.avg_snr_db.toFixed(1) : '—'}
                      </span>
                    </td>
                    <td>
                      <span style={{
                        color: p.impedance_pass_pct >= 60 ? '#22c55e' :
                               p.impedance_pass_pct >= 40 ? '#f59e0b' : '#ef4444',
                        fontWeight: 600
                      }}>
                        {p.impedance_pass_pct != null ? p.impedance_pass_pct + '%' : '—'}
                      </span>
                    </td>
                    <td style={{ color: p.artifact_count > 8 ? '#ef4444' : '#374151' }}>
                      {p.artifact_count}
                    </td>
                    <td className="text-muted" style={{ fontSize: '0.7rem' }}>{p.artifact_types}</td>
                    <td>{p.hyperventilation
                      ? <span className="badge bg-success" style={{ fontSize: '0.65rem' }}>Yes</span>
                      : <span className="badge bg-secondary" style={{ fontSize: '0.65rem' }}>No</span>
                    }</td>
                    <td>{p.photic_stimulation
                      ? <span className="badge bg-success" style={{ fontSize: '0.65rem' }}>Yes</span>
                      : <span className="badge bg-secondary" style={{ fontSize: '0.65rem' }}>No</span>
                    }</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div className="text-muted small mt-2">
        Usability target: &gt;80% Good channels. SNR &ge;20 dB = Good for AI feature extraction.
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   ARTIFACTS TAB
═══════════════════════════════════════════════════════════════════════════ */
function ArtifactsTab({ bd }) {
  if (!bd) return <div className="spinner-border text-primary" />;

  const topChannels = bd.top_artifact_channels || [];
  const matrix = bd.artifact_type_severity_matrix || {};

  const artTypes = Object.keys(matrix);
  const severities = ['mild', 'moderate', 'severe'];

  return (
    <div>
      <div className="row g-3 mb-3">
        {/* Top Artifact Channels */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f6a8; Top Artifact-Prone Channels</div>
            <div className="card-body">
              {topChannels.map((c, i) => {
                const maxCount = topChannels[0]?.count || 1;
                const pct = Math.round((c.count / maxCount) * 100);
                return (
                  <div key={i} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className="fw-semibold" style={{ fontFamily: 'monospace' }}>{c.channel}</span>
                      <span>{c.count} annotations</span>
                    </div>
                    <div style={{ background: '#e5e7eb', borderRadius: 4, height: 10 }}>
                      <div style={{ width: `${pct}%`, background: '#ef4444', borderRadius: 4, height: 10 }} />
                    </div>
                  </div>
                );
              })}
              <div className="text-muted mt-2" style={{ fontSize: '0.7rem' }}>
                Channels with frequent artifacts require impedance check, electrode repositioning,
                or exclusion from AI training features.
              </div>
            </div>
          </div>
        </div>

        {/* Artifact × Severity Matrix */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f4cb; Artifact Type × Severity Matrix</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0 small">
                  <thead className="table-light">
                    <tr>
                      <th>Type</th>
                      {severities.map(s => (
                        <th key={s} style={{ color: sevColor(s) }}>
                          {s.charAt(0).toUpperCase() + s.slice(1)}
                        </th>
                      ))}
                      <th>Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {artTypes.map(at => {
                      const row = matrix[at] || {};
                      const total = severities.reduce((s, sv) => s + (row[sv] || 0), 0);
                      return (
                        <tr key={at}>
                          <td>
                            <span style={{ color: ARTIFACT_COLORS[at] || '#6b7280', fontWeight: 600 }}>
                              {at.replace('_', ' ')}
                            </span>
                          </td>
                          {severities.map(s => (
                            <td key={s} style={{ color: (row[s] || 0) > 10 ? sevColor(s) : '#374151' }}>
                              {row[s] || 0}
                            </td>
                          ))}
                          <td className="fw-bold">{total}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="p-2 text-muted" style={{ fontSize: '0.7rem' }}>
                Severe artifacts in critical channels (e.g., Fp1/Fp2 for frontal, T3/T4 for temporal)
                require documentation and possible channel exclusion from AI feature matrices.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* AI Impact Note */}
      <div className="alert alert-warning border-0">
        <strong>AI Training Data Quality Note</strong><br />
        <span className="small">
          Artifact annotations are used to exclude contaminated epochs from AI training.
          Muscle and movement artifacts (&gt;30 Hz) degrade spectral features.
          Eye-blink artifacts at Fp1/Fp2 are removed by ICA or epoch rejection.
          ECG artifacts require BSS-based subtraction. EEG Technologist sign-off is
          required before recordings enter the AI training pipeline.
        </span>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   DEFINITIONS TAB
═══════════════════════════════════════════════════════════════════════════ */
function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;

  const concepts  = defs.concepts  || [];
  const protocols = defs.activation_protocols || [];
  const refs      = defs.references || [];
  const thresholds = defs.quality_thresholds || {};

  return (
    <div>
      <h6 className="fw-bold border-bottom pb-2 mb-3">&#x1f4da; EEG Technologist Reference — Key Concepts</h6>
      <div className="row g-2 mb-4">
        {concepts.map((c, i) => (
          <div key={i} className="col-12 col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-body py-2">
                <div className="d-flex justify-content-between align-items-start mb-1">
                  <span className="fw-bold small" style={{ color: '#3b82f6' }}>{c.term}</span>
                  {c.standard && <span className="badge bg-light text-muted" style={{ fontSize: '0.6rem' }}>{c.standard}</span>}
                </div>
                <div className="text-muted" style={{ fontSize: '0.78rem' }}>{c.definition}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Quality Thresholds */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">&#x1f4cf; Quality Thresholds</div>
        <div className="card-body">
          <div className="table-responsive">
            <table className="table table-sm table-bordered mb-0 small">
              <thead className="table-light">
                <tr><th>Parameter</th><th>Threshold</th></tr>
              </thead>
              <tbody>
                {Object.entries(thresholds).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k.replace(/_/g, ' ')}</td>
                    <td className="fw-semibold">{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Activation Protocols */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">&#x1f9ea; Standard Activation Protocols</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-bordered mb-0 small">
              <thead className="table-light">
                <tr><th>Procedure</th><th>Purpose</th><th>Duration</th></tr>
              </thead>
              <tbody>
                {protocols.map((p, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{p.procedure}</td>
                    <td>{p.purpose}</td>
                    <td className="text-muted">{p.duration}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* References */}
      <div className="card shadow-sm">
        <div className="card-header fw-bold">&#x1f4d6; References</div>
        <div className="card-body">
          <ol className="mb-0 ps-3">
            {refs.map((r, i) => (
              <li key={i} className="text-muted mb-1" style={{ fontSize: '0.78rem' }}>{r}</li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   ROOT PAGE
═══════════════════════════════════════════════════════════════════════════ */
export default function EEGTechnologistPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [err,  setErr]  = useState(null);

  useEffect(() => {
    fetch(`${API}/api/eeg-technologist/overview`)
      .then(r => r.json()).then(setOv).catch(e => setErr(e.message));
    fetch(`${API}/api/eeg-technologist/breakdown`)
      .then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/eeg-technologist/definitions`)
      .then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (err) return <div className="p-4 text-danger">Error: {err}</div>;
  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const TABS = [
    { id: 'overview',         label: 'Overview' },
    { id: 'channel-quality',  label: 'Channel Quality' },
    { id: 'artifacts',        label: 'Artifacts' },
    { id: 'definitions',      label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f4e1; EEG Technologist Dashboard</h3>
      <p className="text-muted small">
        Recording quality · impedance &amp; SNR · artifact burden · activation procedures
        &mdash; {ov.kpis?.[0]?.value || '?'} recordings · {ov.kpis?.[1]?.value || '?'} patients
      </p>

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

      {tab === 'overview'        && <OverviewTab ov={ov} />}
      {tab === 'channel-quality' && <ChannelQualityTab bd={bd} />}
      {tab === 'artifacts'       && <ArtifactsTab bd={bd} />}
      {tab === 'definitions'     && <DefinitionsTab defs={defs} />}
    </div>
  );
}
