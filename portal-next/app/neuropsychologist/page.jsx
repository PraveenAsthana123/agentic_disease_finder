'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

/* ─── colour helpers ─────────────────────────────────────────────────────── */
const iqColor = score =>
  score >= 110 ? '#22c55e' :
  score >= 90  ? '#3b82f6' :
  score >= 80  ? '#f59e0b' :
  score >= 70  ? '#ef4444' : '#991b1b';

const mocaColor = score =>
  score >= 26 ? '#22c55e' :
  score >= 18 ? '#f59e0b' : '#ef4444';

const impairColor = level =>
  level === 'none'     ? '#22c55e' :
  level === 'mild'     ? '#f59e0b' :
  level === 'moderate' ? '#ef4444' : '#991b1b';

const phqColor = score =>
  score < 5  ? '#22c55e' :
  score < 10 ? '#f59e0b' :
  score < 15 ? '#ef4444' : '#991b1b';

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

function IndexBar({ label, mean, n }) {
  const pct = Math.min(100, Math.round((mean / 15) * 100));
  const color = mean >= 10 ? '#22c55e' : mean >= 8 ? '#3b82f6' : mean >= 6 ? '#f59e0b' : '#ef4444';
  return (
    <div className="mb-3">
      <div className="d-flex justify-content-between small mb-1">
        <span className="fw-semibold">{label}</span>
        <span className="text-muted">Mean scaled score: <strong>{mean?.toFixed(1)}</strong> (n={n})</span>
      </div>
      <div style={{ background: '#e5e7eb', borderRadius: 4, height: 14 }}>
        <div style={{ width: `${pct}%`, background: color, borderRadius: 4, height: 14 }} />
      </div>
      <div className="text-muted" style={{ fontSize: '0.68rem' }}>
        Avg=10, SD=3 · &lt;7 = weakness, &lt;5 = impaired
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   OVERVIEW TAB
═══════════════════════════════════════════════════════════════════════════ */
function OverviewTab({ ov }) {
  const iqDist  = ov.iq_distribution  || [];
  const mocaDist = ov.moca_distribution || [];
  const totalIQ  = iqDist.reduce((s, d) => s + d.count, 0);
  const totalMoCA = mocaDist.reduce((s, d) => s + d.count, 0);

  const IQ_COLORS = {
    'Very Low': '#991b1b', 'Low': '#ef4444', 'Low Average': '#f59e0b',
    'Average': '#3b82f6', 'High Average': '#22c55e', 'Very Superior': '#10b981',
    'Extremely Low': '#991b1b', 'Borderline': '#dc2626', 'Extremely High': '#10b981',
  };
  const MOCA_COLORS = ['#22c55e', '#f59e0b', '#ef4444'];

  return (
    <div>
      {/* KPIs */}
      <div className="row row-cols-2 row-cols-md-4 g-2 mb-4">
        <StatCard label="Patients Assessed" value={ov.total_patients_assessed} color="#3b82f6" />
        <StatCard label="Total Assessments" value={ov.total_assessments} color="#6366f1" />
        <StatCard label="Avg FSIQ" value={ov.avg_fsiq?.toFixed(1)}
          sub="WAIS-IV Full Scale" color={iqColor(ov.avg_fsiq)} />
        <StatCard label="Below Average IQ" value={ov.iq_below_average}
          sub={`of ${ov.wais_count} WAIS`} color="#f59e0b" />
        <StatCard label="Avg MoCA" value={ov.avg_moca?.toFixed(1)}
          sub="/30 points" color={mocaColor(ov.avg_moca)} />
        <StatCard label="MoCA Impaired" value={ov.moca_impaired}
          sub={`of ${ov.moca_count} screened`} color="#ef4444" />
        <StatCard label="Avg MMSE" value={ov.avg_mmse?.toFixed(1)}
          sub="/30 points" color={mocaColor(ov.avg_mmse)} />
        <StatCard label="Avg Digit Span" value={ov.avg_digit_span?.toFixed(1)}
          sub="Working memory" color="#8b5cf6" />
      </div>

      <div className="row g-3 mb-3">
        {/* IQ distribution */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f9e0; IQ Distribution (WAIS-IV FSIQ)</div>
            <div className="card-body">
              {iqDist.map(d => (
                <HBar key={d.level} label={d.level} count={d.count}
                  total={totalIQ} color={IQ_COLORS[d.level] || '#6b7280'} />
              ))}
              <div className="text-muted" style={{ fontSize: '0.7rem' }}>
                Reference: Mean=100, SD=15. Below Average &lt;85.
              </div>
            </div>
          </div>
        </div>

        {/* MoCA distribution */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">&#x1f4cb; MoCA Screening Distribution</div>
            <div className="card-body">
              {mocaDist.map((d, i) => (
                <HBar key={d.category} label={d.category} count={d.count}
                  total={totalMoCA} color={MOCA_COLORS[i] || '#6b7280'} />
              ))}
              <div className="text-muted" style={{ fontSize: '0.7rem' }}>
                ≥26 = Normal · 18–25 = Mild impairment · &lt;18 = Moderate impairment
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* WAIS Index Profile */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">&#x1f4ca; WAIS-IV Index Score Profile (Mean Scaled Scores)</div>
        <div className="card-body">
          {(ov.index_profile || []).map(idx => (
            <IndexBar key={idx.index}
              label={`${idx.index} — ${idx.label}`}
              mean={idx.mean_score} n={idx.n} />
          ))}
        </div>
      </div>

      {/* Digit Span */}
      {(ov.digit_span_profile || []).length > 0 && (
        <div className="card shadow-sm mb-3">
          <div className="card-header fw-bold">&#x1f522; Digit Span Profile (Working Memory)</div>
          <div className="card-body">
            <div className="row g-3">
              {(ov.digit_span_profile || []).map(d => (
                <div key={d.type} className="col-4">
                  <div className="card border text-center p-2">
                    <div className="fw-bold" style={{ fontSize: '1.4rem', color: iqColor(d.mean_span * 10) }}>
                      {d.mean_span?.toFixed(1)}
                    </div>
                    <div className="text-muted small">{d.type}</div>
                    <div style={{ fontSize: '0.68rem', color: '#9ca3af' }}>n={d.n}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="text-muted mt-2" style={{ fontSize: '0.7rem' }}>
              Raw scores. Forward = auditory attention · Backward = working memory · Sequencing = flexibility
            </div>
          </div>
        </div>
      )}

      {/* Mood comorbidity */}
      {ov.mood_comorbidity && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold">&#x1f4ac; Mood Comorbidity (PHQ-9 / GAD-7)</div>
          <div className="card-body">
            <div className="row g-2">
              <div className="col-6 col-md-3 text-center">
                <div className="h4 fw-bold text-primary">{ov.mood_comorbidity.phq9_assessed}</div>
                <div className="text-muted small">PHQ-9 Assessed</div>
              </div>
              <div className="col-6 col-md-3 text-center">
                <div className="h4 fw-bold text-danger">{ov.mood_comorbidity.phq9_elevated}</div>
                <div className="text-muted small">PHQ-9 Elevated (≥10)</div>
              </div>
              <div className="col-6 col-md-3 text-center">
                <div className="h4 fw-bold text-primary">{ov.mood_comorbidity.gad7_assessed}</div>
                <div className="text-muted small">GAD-7 Assessed</div>
              </div>
              <div className="col-6 col-md-3 text-center">
                <div className="h4 fw-bold text-warning">{ov.mood_comorbidity.gad7_elevated}</div>
                <div className="text-muted small">GAD-7 Elevated (≥8)</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   WAIS PROFILES TAB
═══════════════════════════════════════════════════════════════════════════ */
function WaisProfilesTab({ bd }) {
  const [sortBy, setSortBy] = useState('fsiq');
  const [sortDir, setSortDir] = useState('asc');
  const [filter, setFilter] = useState('');

  if (!bd) return <div className="spinner-border text-primary" />;

  const profiles = [...(bd.wais_profiles || [])];
  const filtered = filter
    ? profiles.filter(p => p.patient_id?.toLowerCase().includes(filter.toLowerCase()) ||
        p.level?.toLowerCase().includes(filter.toLowerCase()))
    : profiles;

  filtered.sort((a, b) => {
    const va = a[sortBy] ?? 0;
    const vb = b[sortBy] ?? 0;
    return sortDir === 'asc' ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
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
          placeholder="Filter by patient or level…"
          value={filter} onChange={e => setFilter(e.target.value)} />
        <span className="text-muted small ms-2">{filtered.length} records</span>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold">&#x1f9e0; WAIS-IV Patient Profiles</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0 small">
              <thead className="table-light">
                <tr>
                  <Th col="patient_id" label="Patient" />
                  <Th col="fsiq" label="FSIQ" />
                  <Th col="level" label="Classification" />
                  <Th col="VCI" label="VCI" />
                  <Th col="PRI" label="PRI" />
                  <Th col="WMI" label="WMI" />
                  <Th col="PSI" label="PSI" />
                  <th>Date</th>
                  <th>Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((p, i) => (
                  <tr key={i}>
                    <td><span className="badge bg-secondary">{p.patient_id}</span></td>
                    <td>
                      <span className="fw-bold" style={{ color: iqColor(p.fsiq) }}>
                        {p.fsiq?.toFixed(0)}
                      </span>
                    </td>
                    <td>
                      <span className="badge" style={{
                        background: iqColor(p.fsiq), color: '#fff', fontSize: '0.68rem'
                      }}>{p.level}</span>
                    </td>
                    <td style={{ color: p.VCI < 7 ? '#ef4444' : '#374151' }}>{p.VCI?.toFixed(1)}</td>
                    <td style={{ color: p.PRI < 7 ? '#ef4444' : '#374151' }}>{p.PRI?.toFixed(1)}</td>
                    <td style={{ color: p.WMI < 7 ? '#ef4444' : '#374151' }}>{p.WMI?.toFixed(1)}</td>
                    <td style={{ color: p.PSI < 7 ? '#ef4444' : '#374151' }}>{p.PSI?.toFixed(1)}</td>
                    <td className="text-muted">{p.date}</td>
                    <td className="text-muted" style={{ fontSize: '0.7rem', maxWidth: 260 }}>{p.interpretation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div className="text-muted small mt-2">
        VCI=Verbal Comprehension · PRI=Perceptual Reasoning · WMI=Working Memory · PSI=Processing Speed
        · Scaled scores (mean=10, SD=3) · Red = below 7 (weakness)
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SCREENING TESTS TAB (MoCA / MMSE / Digit Span per patient)
═══════════════════════════════════════════════════════════════════════════ */
function ScreeningTab({ bd }) {
  const [filter, setFilter] = useState('');
  if (!bd) return <div className="spinner-border text-primary" />;

  const records = (bd.screening_records || []).filter(r =>
    !filter || r.patient_id?.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div>
      <div className="mb-3">
        <input className="form-control form-control-sm w-auto d-inline-block"
          placeholder="Filter by patient…"
          value={filter} onChange={e => setFilter(e.target.value)} />
        <span className="text-muted small ms-2">{records.length} records</span>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold">&#x1f4cb; Screening Battery — MoCA / MMSE / Digit Span</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0 small">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Battery</th>
                  <th>MoCA</th>
                  <th>MMSE</th>
                  <th>DS Fwd</th>
                  <th>DS Bwd</th>
                  <th>Impairment</th>
                  <th>Lateralization</th>
                  <th>Referral Reason</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {records.map((r, i) => (
                  <tr key={i}>
                    <td><span className="badge bg-secondary">{r.patient_id}</span></td>
                    <td className="text-muted">{r.battery_type}</td>
                    <td>
                      <span style={{ color: mocaColor(r.moca), fontWeight: 600 }}>{r.moca}</span>
                      <span className="text-muted">/30</span>
                    </td>
                    <td>
                      <span style={{ color: mocaColor(r.mmse), fontWeight: 600 }}>{r.mmse}</span>
                      <span className="text-muted">/30</span>
                    </td>
                    <td>{r.digit_span_forward ?? '—'}</td>
                    <td>{r.digit_span_backward ?? '—'}</td>
                    <td>
                      <span className="badge" style={{
                        background: impairColor(r.impairment_flag), color: '#fff', fontSize: '0.68rem'
                      }}>{r.impairment_flag || '—'}</span>
                    </td>
                    <td className="text-muted small">{r.lateralization_hypothesis || '—'}</td>
                    <td className="text-muted small">{r.referral_reason || '—'}</td>
                    <td className="text-muted">{r.date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   MOOD & COMORBIDITY TAB
═══════════════════════════════════════════════════════════════════════════ */
function MoodTab({ bd }) {
  if (!bd) return <div className="spinner-border text-primary" />;

  const records = bd.mood_records || [];
  const summary = bd.mood_summary || {};

  return (
    <div>
      {/* Summary cards */}
      <div className="row g-2 mb-4">
        {[
          { label: 'PHQ-9 Assessed',   value: summary.phq9_assessed,   color: '#3b82f6' },
          { label: 'PHQ-9 Elevated',   value: summary.phq9_elevated,   color: '#ef4444' },
          { label: 'GAD-7 Assessed',   value: summary.gad7_assessed,   color: '#3b82f6' },
          { label: 'GAD-7 Elevated',   value: summary.gad7_elevated,   color: '#f59e0b' },
          { label: 'Avg PHQ-9 Score',  value: summary.avg_phq9?.toFixed(1), color: phqColor(summary.avg_phq9) },
          { label: 'Avg GAD-7 Score',  value: summary.avg_gad7?.toFixed(1), color: '#8b5cf6' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2">
            <div className="card text-center py-2">
              <div className="h5 fw-bold mb-0" style={{ color: c.color }}>{c.value ?? '—'}</div>
              <div className="text-muted small">{c.label}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold">&#x1f4ac; PHQ-9 &amp; GAD-7 Per Patient</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0 small">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>PHQ-9</th>
                  <th>PHQ-9 Severity</th>
                  <th>GAD-7</th>
                  <th>GAD-7 Severity</th>
                  <th>Memory Index</th>
                  <th>Attention</th>
                  <th>Executive</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {records.map((r, i) => (
                  <tr key={i}>
                    <td><span className="badge bg-secondary">{r.patient_id}</span></td>
                    <td>
                      <span style={{ color: phqColor(r.phq9), fontWeight: 600 }}>{r.phq9 ?? '—'}</span>
                    </td>
                    <td>
                      <span className="badge" style={{
                        background: phqColor(r.phq9), color: '#fff', fontSize: '0.68rem'
                      }}>{r.phq9_severity || '—'}</span>
                    </td>
                    <td>
                      <span style={{ color: r.gad7 >= 8 ? '#ef4444' : '#22c55e', fontWeight: 600 }}>
                        {r.gad7 ?? '—'}
                      </span>
                    </td>
                    <td>
                      <span className="badge" style={{
                        background: r.gad7 >= 8 ? '#ef4444' : '#22c55e', color: '#fff', fontSize: '0.68rem'
                      }}>{r.gad7_severity || '—'}</span>
                    </td>
                    <td style={{ color: r.memory_index < 80 ? '#f59e0b' : '#374151' }}>{r.memory_index ?? '—'}</td>
                    <td style={{ color: r.attention_index < 80 ? '#f59e0b' : '#374151' }}>{r.attention_index ?? '—'}</td>
                    <td style={{ color: r.executive_index < 80 ? '#f59e0b' : '#374151' }}>{r.executive_index ?? '—'}</td>
                    <td className="text-muted">{r.date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   DEFINITIONS TAB
═══════════════════════════════════════════════════════════════════════════ */
function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading definitions…</div>;

  const concepts = defs.concepts || [];
  const scales   = defs.scales   || [];
  const cutoffs  = defs.cutoffs  || {};

  return (
    <div>
      {/* Cutoffs summary */}
      {Object.keys(cutoffs).length > 0 && (
        <div className="card shadow-sm mb-4">
          <div className="card-header fw-bold">&#x1f4d0; Score Cutoffs &amp; Reference Ranges</div>
          <div className="card-body">
            {Object.entries(cutoffs).map(([test, ranges]) => (
              <div key={test} className="mb-3">
                <h6 className="fw-semibold">{test}</h6>
                {Array.isArray(ranges)
                  ? ranges.map((r, i) => (
                    <div key={i} className="d-flex gap-2 align-items-center mb-1 small">
                      <span className="badge" style={{ background: r.color || '#6b7280', color: '#fff', minWidth: 70 }}>
                        {r.label}
                      </span>
                      <span className="text-muted">{r.range}</span>
                    </div>
                  ))
                  : <div className="text-muted small">{JSON.stringify(ranges)}</div>
                }
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Concept cards */}
      {concepts.length > 0 && (
        <div className="mb-4">
          <h6 className="fw-bold border-bottom pb-1">&#x1f4da; Key Concepts</h6>
          <div className="row g-2">
            {concepts.map((c, i) => (
              <div key={i} className="col-12 col-md-6">
                <div className="card shadow-sm h-100">
                  <div className="card-body py-2">
                    <div className="fw-bold small mb-1">{c.name}</div>
                    <div className="text-muted" style={{ fontSize: '0.78rem' }}>{c.description}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Scale reference */}
      {scales.length > 0 && (
        <div>
          <h6 className="fw-bold border-bottom pb-1">&#x1f4cb; Assessment Scales</h6>
          <div className="row g-2">
            {scales.map((s, i) => (
              <div key={i} className="col-12 col-md-6">
                <div className="card shadow-sm h-100">
                  <div className="card-body py-2">
                    <div className="fw-bold small mb-1">{s.name}</div>
                    <div className="text-muted" style={{ fontSize: '0.78rem' }}>{s.description}</div>
                    {s.range && (
                      <div className="mt-1" style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                        Range: {s.range}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Fallback: raw concepts only */}
      {concepts.length === 0 && scales.length === 0 && Object.keys(cutoffs).length === 0 && (
        <div className="row g-2">
          {Object.entries(defs).map(([k, v]) => (
            <div key={k} className="col-12 col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body py-2">
                  <div className="fw-bold small mb-1">{k}</div>
                  <div className="text-muted" style={{ fontSize: '0.78rem' }}>
                    {typeof v === 'string' ? v : JSON.stringify(v)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   ROOT PAGE
═══════════════════════════════════════════════════════════════════════════ */
export default function NeuropsychologistPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/neuropsychologist/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/neuropsychologist/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/neuropsychologist/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const TABS = [
    { id: 'overview',    label: 'Overview' },
    { id: 'wais',        label: 'WAIS Profiles' },
    { id: 'screening',   label: 'Screening Tests' },
    { id: 'mood',        label: 'Mood & Comorbidity' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f9e0; Neuropsychologist Dashboard</h3>
      <p className="text-muted small">
        Neuropsychological battery results for {ov.total_patients_assessed} epilepsy patients —
        WAIS-IV ({ov.wais_count} assessments), MoCA ({ov.moca_count}), MMSE ({ov.mmse_count}),
        Digit Span ({ov.digit_span_count}), PHQ-9 &amp; GAD-7. Real data from neuropsych table.
      </p>

      {/* Tabs */}
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
      {tab === 'wais'        && <WaisProfilesTab bd={bd} />}
      {tab === 'screening'   && <ScreeningTab bd={bd} />}
      {tab === 'mood'        && <MoodTab bd={bd} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  );
}
