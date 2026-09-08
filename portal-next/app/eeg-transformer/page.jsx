'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',      label: 'Overview' },
  { id: 'patches',       label: 'Patch Inventory' },
  { id: 'architectures', label: 'Architectures' },
  { id: 'definitions',   label: 'Definitions' },
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

function PctBar({ pct, color }) {
  const c = color || (pct >= 90 ? 'success' : pct >= 80 ? 'info' : pct >= 70 ? 'warning' : 'danger');
  return (
    <div className="progress" style={{ height: 14, borderRadius: 6 }}>
      <div
        className={`progress-bar bg-${c}`}
        style={{ width: `${Math.min(pct, 100)}%`, borderRadius: 6, transition: 'width 0.5s' }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Overview Panel
// ---------------------------------------------------------------------------
function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;

  const kpis        = ov.kpis || [];
  const attnHeads   = ov.attention_head_chart || [];
  const classDist   = ov.classification_chart || [];
  const bandPower   = ov.band_power_chart || [];
  const qualDist    = ov.quality_distribution || [];
  const daily       = ov.daily_activity || [];
  const benchmarks  = ov.literature_benchmarks || [];

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-4">
        {kpis.slice(0, 8).map((k, i) => (
          <KPI key={i} label={k.label} value={k.value}
            color={['primary','success','info','warning','secondary','dark','danger','primary'][i]}
            sub={k.sub} />
        ))}
      </div>

      <div className="row mb-4">
        {/* Attention Head Band Assignments */}
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>Attention Head → EEG Band Assignments</strong>
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>Head</th>
                    <th>Dominant Band</th>
                    <th>Range</th>
                    <th>Mean Weight</th>
                  </tr>
                </thead>
                <tbody>
                  {attnHeads.map((h, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{h.head}</td>
                      <td>
                        <span className="badge" style={{ backgroundColor: h.color }}>
                          {h.dominant_band}
                        </span>
                      </td>
                      <td className="small text-muted">{h.band_range}</td>
                      <td>{h.mean_weight}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Classification Results */}
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>Classification Results by Label</strong>
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-dark">
                  <tr><th>Label</th><th>Count</th></tr>
                </thead>
                <tbody>
                  {classDist.map((c, i) => (
                    <tr key={i}>
                      <td>{c.label}</td>
                      <td className="fw-bold">{c.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="row mb-4">
        {/* Band Power */}
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>EEG Band Power Distribution</strong>
            </div>
            <div className="card-body">
              {bandPower.map((b, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-semibold">{b.band}</span>
                    <span className="small text-muted">{b.mean_power_db} dB · {b.n_dominant} dominant</span>
                  </div>
                  <div className="progress" style={{ height: 10, borderRadius: 6 }}>
                    <div
                      className="progress-bar"
                      style={{
                        width: `${Math.min(100, (b.n_dominant / (ov.total_analyses || 1)) * 100 * 3)}%`,
                        backgroundColor: b.color,
                        borderRadius: 6,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Signal Quality */}
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>Signal Quality Distribution</strong>
            </div>
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
                        width: `${Math.round(q.count / (ov.total_analyses || 1) * 100)}%`,
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
      </div>

      {/* Literature Benchmarks */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-primary text-white">
          <strong>CHB-MIT Benchmark Comparison (Literature)</strong>
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-dark">
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Sensitivity</th>
                <th>Parameters</th>
                <th>Reference</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.map((b, i) => (
                <tr key={i} className={b.model.includes('ViT') ? 'table-success' : b.model.includes('Random') ? 'table-danger' : ''}>
                  <td className="fw-semibold">{b.model}</td>
                  <td>
                    {b.acc}%
                    <div><PctBar pct={b.acc} /></div>
                  </td>
                  <td>{b.sens}%</td>
                  <td className="small">{b.params}</td>
                  <td className="small text-muted">{b.reference}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Daily Activity */}
      {daily.length > 0 && (
        <div className="card shadow-sm mb-4">
          <div className="card-header py-2 bg-dark text-white">
            <strong>Daily Pipeline Activity (last 14 days)</strong>
          </div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-dark">
                <tr><th>Date</th><th>Events</th></tr>
              </thead>
              <tbody>
                {daily.map((d, i) => (
                  <tr key={i}>
                    <td>{d.date}</td>
                    <td>{d.events}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Patch Config */}
      {ov.patch_config && (
        <div className="card shadow-sm">
          <div className="card-header py-2 bg-info text-dark">
            <strong>Tokenisation / Patch Configuration</strong>
          </div>
          <div className="card-body">
            <div className="row">
              {Object.entries(ov.patch_config).map(([k, v], i) => (
                <div key={i} className="col-6 col-md-3 mb-2">
                  <div className="small text-muted">{k.replace(/_/g,' ')}</div>
                  <div className="fw-semibold">{String(v)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Patch Inventory Panel
// ---------------------------------------------------------------------------
function PatchesPanel({ bk }) {
  if (!bk) return <div className="text-muted p-3">Loading…</div>;

  const patches   = bk.patch_inventory || [];
  const profiles  = bk.patient_profiles || [];
  const summary   = bk.pipeline_summary || [];

  return (
    <div>
      <div className="row mb-4">
        <div className="col-md-8 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>EEG Patch Inventory (per upload)</strong>
            </div>
            <div className="card-body p-0" style={{ overflowX: 'auto' }}>
              <table className="table table-sm mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>File</th>
                    <th>Patient</th>
                    <th>Duration (s)</th>
                    <th>Channels</th>
                    <th>SR (Hz)</th>
                    <th>Windows</th>
                    <th>Patches</th>
                  </tr>
                </thead>
                <tbody>
                  {patches.length === 0
                    ? <tr><td colSpan={7} className="text-center text-muted py-3">No uploads in database</td></tr>
                    : patches.map((p, i) => (
                      <tr key={i}>
                        <td className="small">{p.filename}</td>
                        <td>{p.patient_id}</td>
                        <td>{p.duration_s}</td>
                        <td>{p.n_channels}</td>
                        <td>{p.sampling_rate}</td>
                        <td className="fw-bold">{p.n_windows}</td>
                        <td className="fw-bold text-primary">{p.n_patches.toLocaleString()}</td>
                      </tr>
                    ))
                  }
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>Pipeline Event Types</strong>
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-dark">
                  <tr><th>Event Type</th><th>Count</th></tr>
                </thead>
                <tbody>
                  {summary.map((s, i) => (
                    <tr key={i}>
                      <td className="small">{s.event_type}</td>
                      <td className="fw-bold">{s.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Patient Profiles */}
      <div className="card shadow-sm">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Patient-Level Transformer Analysis Profiles</strong>
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-dark">
              <tr>
                <th>Patient</th>
                <th>Age</th>
                <th>Sex</th>
                <th>Diagnosis</th>
                <th>Analyses</th>
                <th>Mean Conf</th>
                <th>Top Class</th>
              </tr>
            </thead>
            <tbody>
              {profiles.length === 0
                ? <tr><td colSpan={7} className="text-center text-muted py-3">No patient data</td></tr>
                : profiles.slice(0, 20).map((p, i) => (
                  <tr key={i}>
                    <td className="fw-bold">{p.patient_id}</td>
                    <td>{p.age ?? '—'}</td>
                    <td>{p.sex ?? '—'}</td>
                    <td className="small">{p.diagnosis ?? '—'}</td>
                    <td>{p.n_analyses}</td>
                    <td>{p.mean_conf ? `${(p.mean_conf * 100).toFixed(1)}%` : '—'}</td>
                    <td className="small">{p.top_class}</td>
                  </tr>
                ))
              }
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Architectures Panel
// ---------------------------------------------------------------------------
function ArchitecturesPanel({ bk }) {
  if (!bk) return <div className="text-muted p-3">Loading…</div>;

  const archs = bk.arch_comparison || [];
  const archDetail = bk.architectures || {};

  return (
    <div>
      {/* Comparison table */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>EEG Transformer Architecture Comparison (CHB-MIT)</strong>
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-dark">
              <tr>
                <th>Model</th>
                <th>Parameters</th>
                <th>Accuracy</th>
                <th>Sensitivity</th>
                <th>Pre-training</th>
                <th>Reference</th>
              </tr>
            </thead>
            <tbody>
              {archs.map((a, i) => (
                <tr key={i} className={i === 0 ? 'table-success' : ''}>
                  <td className="fw-semibold small">{a.name}</td>
                  <td>{a.params_label}</td>
                  <td>
                    {a.accuracy}%
                    <div><PctBar pct={a.accuracy} /></div>
                  </td>
                  <td>{a.sensitivity}%</td>
                  <td className="small text-muted">{a.pretrain}</td>
                  <td className="small text-muted">{a.reference}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Architecture cards */}
      {Object.entries(archDetail).map(([id, arch]) => (
        <div key={id} className="card shadow-sm mb-4">
          <div className="card-header py-2 bg-secondary text-white d-flex justify-content-between">
            <strong>{arch.name}</strong>
            <span className="badge bg-light text-dark">
              {arch.params >= 1_000_000
                ? `${(arch.params / 1e6).toFixed(2)} M params`
                : `${(arch.params / 1e3).toFixed(0)} K params`}
            </span>
          </div>
          <div className="card-body">
            <p className="small mb-3 text-muted">{arch.description}</p>
            <div className="table-responsive">
              <table className="table table-sm table-bordered mb-0">
                <thead className="table-dark">
                  <tr><th>Stage</th><th>Layer Type</th><th>Detail</th></tr>
                </thead>
                <tbody>
                  {(arch.layers || []).map((l, i) => (
                    <tr key={i}>
                      <td><span className="badge bg-secondary">{l.stage}</span></td>
                      <td className="small fw-semibold">{l.type}</td>
                      <td className="small text-muted">{l.detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-2 d-flex gap-3">
              <span className="small"><strong>CHB-MIT Acc:</strong> {arch.accuracy}%</span>
              <span className="small"><strong>Sensitivity:</strong> {arch.sensitivity}%</span>
              <span className="small text-muted">{arch.pretrain}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Definitions Panel
// ---------------------------------------------------------------------------
function DefinitionsPanel({ df }) {
  if (!df) return <div className="text-muted p-3">Loading…</div>;

  const defs   = df.definitions || [];
  const regs   = df.regulatory_context || [];
  const refs   = df.references || [];
  const cfg    = df.patch_config || {};

  return (
    <div>
      {/* Glossary */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Glossary — EEG Transformer Concepts</strong>
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-dark">
              <tr><th style={{ width: '22%' }}>Term</th><th>Definition</th></tr>
            </thead>
            <tbody>
              {defs.map((d, i) => (
                <tr key={i}>
                  <td className="fw-semibold small align-top">{d.term}</td>
                  <td className="small">{d.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Regulatory */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-warning text-dark">
          <strong>Regulatory &amp; Governance Context</strong>
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-dark">
              <tr><th style={{ width: '28%' }}>Standard</th><th>Relevance</th></tr>
            </thead>
            <tbody>
              {regs.map((r, i) => (
                <tr key={i}>
                  <td className="fw-semibold small align-top">{r.standard}</td>
                  <td className="small">{r.relevance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Patch config */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-info text-dark">
          <strong>Tokenisation Configuration</strong>
        </div>
        <div className="card-body">
          <div className="row">
            {Object.entries(cfg).map(([k, v], i) => (
              <div key={i} className="col-6 col-md-3 mb-2">
                <div className="small text-muted">{k.replace(/_/g,' ')}</div>
                <div className="fw-semibold">{String(v)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* References */}
      <div className="card shadow-sm">
        <div className="card-header py-2 bg-secondary text-white">
          <strong>References</strong>
        </div>
        <div className="card-body">
          <ol className="mb-0 small">
            {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
          </ol>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page root
// ---------------------------------------------------------------------------
export default function EegTransformerPage() {
  const [tab, setTab] = useState('overview');
  const [ov,  setOv]  = useState(null);
  const [bk,  setBk]  = useState(null);
  const [df,  setDf]  = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/eeg-transformer/overview`).then(r => r.json()),
      fetch(`${API}/api/eeg-transformer/breakdown`).then(r => r.json()),
      fetch(`${API}/api/eeg-transformer/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <div>
          <h4 className="mb-0 fw-bold">EEG Transformer — Attention-Based EEG Classifier</h4>
          <div className="text-muted small">
            EEGformer · Conformer · ViT-EEG · Multi-Head Self-Attention · Patch Tokens · Seizure Detection
          </div>
        </div>
        <div className="ms-auto d-flex gap-2 flex-wrap">
          <span className="badge bg-primary">EEGformer</span>
          <span className="badge bg-info text-dark">Conformer</span>
          <span className="badge bg-secondary">ViT-EEG</span>
          <span className="badge bg-dark">Lite</span>
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

      {tab === 'overview'      && <OverviewPanel ov={ov} />}
      {tab === 'patches'       && <PatchesPanel bk={bk} />}
      {tab === 'architectures' && <ArchitecturesPanel bk={bk} />}
      {tab === 'definitions'   && <DefinitionsPanel df={df} />}
    </div>
  );
}
