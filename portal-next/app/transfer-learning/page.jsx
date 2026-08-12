'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'patients',    label: 'Patient Results' },
  { id: 'strategies',  label: 'Strategies' },
  { id: 'definitions', label: 'Definitions' },
];

const STRATEGY_COLOR = {
  subject_independent: 'secondary',
  fine_tune:           'primary',
  domain_adversarial:  'danger',
  multi_task:          'success',
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

function AccBar({ pct, color }) {
  const c = color || (pct >= 35 ? 'success' : pct >= 15 ? 'info' : pct >= 0 ? 'warning' : 'danger');
  return (
    <div className="progress" style={{ height: 16, borderRadius: 8 }}>
      <div
        className={`progress-bar bg-${c}`}
        style={{ width: `${Math.min(Math.max(pct, 0), 100)}%`, borderRadius: 8, transition: 'width 0.6s ease' }}
      />
    </div>
  );
}

function StrategyBadge({ strategy }) {
  const color = STRATEGY_COLOR[strategy] || 'light';
  const labels = {
    subject_independent: 'Leave-One-Out',
    fine_tune:           'Fine-Tune',
    domain_adversarial:  'DANN',
    multi_task:          'Multi-Task',
  };
  return <span className={`badge bg-${color}`}>{labels[strategy] || strategy}</span>;
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted p-4">Loading…</div>;
  const k = ov.kpis || {};
  const sd = ov.strategy_distribution || [];
  const ih = ov.improvement_histogram || [];
  const ps = ov.per_patient_summary || [];

  const pct = v => `${(v * 100).toFixed(1)}%`;

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Total Patients"        value={k.total_patients}                        color="primary"   sub="Adaptation evaluated" />
        <KPI label="Mean Baseline Acc"     value={pct(k.mean_baseline_accuracy)}           color="secondary" sub="Before adaptation" />
        <KPI label="Mean Adapted Acc"      value={pct(k.mean_adapted_accuracy)}            color="success"   sub="After adaptation" />
        <KPI label="Mean Improvement"      value={`+${k.mean_improvement_pct?.toFixed(1)}%`} color="info"  sub="Relative gain" />
      </div>
      <div className="row mb-4">
        <KPI label="Adaptation Success"    value={`${k.adaptation_success_rate}%`}         color="success"   sub="Patients improved" />
        <KPI label="Avg Domain Shift (MMD)" value={k.avg_domain_shift?.toFixed(3)}         color="warning"   sub="Feature distribution gap" />
        <KPI label="Avg Convergence"       value={`${k.avg_convergence_epochs} ep`}        color="dark"      sub="Fine-tune epochs" />
        <KPI label="Strategies Evaluated"  value={sd.length}                               color="primary"   sub="Transfer methods" />
      </div>

      {/* Strategy distribution */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Strategy Distribution</strong>
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-dark">
              <tr>
                <th>Strategy</th>
                <th>Patients</th>
                <th>Share</th>
                <th>Bar</th>
              </tr>
            </thead>
            <tbody>
              {sd.map(s => (
                <tr key={s.strategy}>
                  <td><StrategyBadge strategy={s.strategy} /> {s.label}</td>
                  <td className="fw-bold">{s.count}</td>
                  <td>{s.pct}%</td>
                  <td style={{ width: '35%' }}>
                    <AccBar pct={s.pct} color={STRATEGY_COLOR[s.strategy]} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Improvement histogram */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Improvement Distribution (% relative gain per patient)</strong>
        </div>
        <div className="card-body">
          <div className="d-flex align-items-end gap-1" style={{ height: 120 }}>
            {ih.map((bin, i) => {
              const maxCount = Math.max(...ih.map(b => b.count), 1);
              const h = (bin.count / maxCount) * 100;
              const color = bin.bin_start < 0 ? '#dc3545' : bin.bin_start < 15 ? '#ffc107' : '#198754';
              return (
                <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <div style={{ fontSize: '0.65rem', color: '#6c757d', marginBottom: 2 }}>{bin.count > 0 ? bin.count : ''}</div>
                  <div
                    title={`${bin.bin_start}% to ${bin.bin_end}%: ${bin.count} patients`}
                    style={{ width: '100%', height: `${h}%`, background: color, borderRadius: '3px 3px 0 0', minHeight: bin.count > 0 ? 4 : 0 }}
                  />
                  <div style={{ fontSize: '0.55rem', color: '#6c757d', marginTop: 2, textAlign: 'center' }}>
                    {bin.bin_start}%
                  </div>
                </div>
              );
            })}
          </div>
          <div className="text-muted small text-center mt-1">
            Red = negative transfer &nbsp;|&nbsp; Yellow = marginal gain &nbsp;|&nbsp; Green = strong improvement
          </div>
        </div>
      </div>

      {/* Top/bottom performers */}
      <div className="row">
        <div className="col-md-6 mb-4">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-success text-white">
              <strong>Top 5 Responders</strong>
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead><tr><th>Patient</th><th>Strategy</th><th>Gain</th></tr></thead>
                <tbody>
                  {[...ps].sort((a, b) => b.improvement_pct - a.improvement_pct).slice(0, 5).map(p => (
                    <tr key={p.patient_id}>
                      <td>{p.patient_id}</td>
                      <td><StrategyBadge strategy={p.strategy} /></td>
                      <td className="fw-bold text-success">+{p.improvement_pct.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-4">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-danger text-white">
              <strong>Negative Transfer Cases</strong>
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead><tr><th>Patient</th><th>Strategy</th><th>Change</th></tr></thead>
                <tbody>
                  {[...ps].filter(p => !p.adaptation_helped).map(p => (
                    <tr key={p.patient_id}>
                      <td>{p.patient_id}</td>
                      <td><StrategyBadge strategy={p.strategy} /></td>
                      <td className="fw-bold text-danger">{p.improvement_pct.toFixed(1)}%</td>
                    </tr>
                  ))}
                  {[...ps].filter(p => !p.adaptation_helped).length === 0 && (
                    <tr><td colSpan={3} className="text-muted text-center py-3">No negative transfer cases</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function PatientsPanel({ ov }) {
  const [filter, setFilter] = useState('all');
  const [sortKey, setSortKey] = useState('improvement_pct');
  const [sortDir, setSortDir] = useState('desc');

  if (!ov) return <div className="text-muted p-4">Loading…</div>;
  const ps = ov.per_patient_summary || [];

  const strategies = ['all', 'subject_independent', 'fine_tune', 'domain_adversarial', 'multi_task'];
  const strategyLabels = {
    all: 'All Strategies',
    subject_independent: 'Leave-One-Out',
    fine_tune: 'Fine-Tune',
    domain_adversarial: 'DANN',
    multi_task: 'Multi-Task',
  };

  const filtered = ps.filter(p => filter === 'all' || p.strategy === filter);
  const sorted = [...filtered].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
    return sortDir === 'asc' ? av - bv : bv - av;
  });

  const toggleSort = key => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };
  const Th = ({ k, children }) => (
    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort(k)}>
      {children} {sortKey === k ? (sortDir === 'asc' ? '▲' : '▼') : ''}
    </th>
  );

  const pct = v => `${(v * 100).toFixed(1)}%`;

  return (
    <div>
      <div className="mb-3 d-flex gap-2 flex-wrap">
        {strategies.map(s => (
          <button
            key={s}
            className={`btn btn-sm ${filter === s ? `btn-${STRATEGY_COLOR[s] || 'dark'}` : 'btn-outline-secondary'}`}
            onClick={() => setFilter(s)}
          >
            {strategyLabels[s]}
          </button>
        ))}
        <span className="ms-auto text-muted small align-self-center">{sorted.length} patients</span>
      </div>
      <div className="card shadow-sm">
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-dark">
                <tr>
                  <Th k="patient_id">Patient ID</Th>
                  <Th k="disease">Disease</Th>
                  <Th k="strategy">Strategy</Th>
                  <Th k="baseline_accuracy">Baseline Acc</Th>
                  <Th k="adapted_accuracy">Adapted Acc</Th>
                  <Th k="improvement_pct">Improvement</Th>
                  <Th k="domain_shift_score">Domain Shift</Th>
                  <Th k="epochs_to_converge">Epochs</Th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map(p => (
                  <tr key={p.patient_id} className={p.adaptation_helped ? '' : 'table-danger'}>
                    <td className="fw-bold">{p.patient_id}</td>
                    <td className="text-capitalize">{p.disease}</td>
                    <td><StrategyBadge strategy={p.strategy} /></td>
                    <td>{pct(p.baseline_accuracy)}</td>
                    <td className="fw-bold">{pct(p.adapted_accuracy)}</td>
                    <td className={`fw-bold ${p.improvement_pct >= 0 ? 'text-success' : 'text-danger'}`}>
                      {p.improvement_pct >= 0 ? '+' : ''}{p.improvement_pct.toFixed(1)}%
                    </td>
                    <td>{p.domain_shift_score?.toFixed(3) ?? '—'}</td>
                    <td>{p.epochs_to_converge ?? '—'}</td>
                    <td>
                      {p.adaptation_helped
                        ? <span className="badge bg-success">Improved</span>
                        : <span className="badge bg-danger">Negative Transfer</span>
                      }
                    </td>
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

function StrategiesPanel({ defs }) {
  if (!defs) return <div className="text-muted p-4">Loading…</div>;
  const strategies = defs.strategies || [];
  const metrics = defs.metrics || [];

  return (
    <div>
      <div className="alert alert-info mb-4">
        <strong>Clinical Context:</strong> {defs.clinical_context}
      </div>

      {/* Strategy cards */}
      <div className="row mb-4">
        {strategies.map(s => (
          <div key={s.key} className="col-md-6 mb-3">
            <div className={`card border-${STRATEGY_COLOR[s.key] || 'secondary'} shadow-sm h-100`}>
              <div className={`card-header bg-${STRATEGY_COLOR[s.key] || 'secondary'} text-white py-2`}>
                <strong>{s.label}</strong>
              </div>
              <div className="card-body">
                <p className="mb-0 small">{s.description}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Metrics reference */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Metric Definitions</strong>
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-dark">
              <tr><th>Metric</th><th>Expected Range</th><th>Description</th></tr>
            </thead>
            <tbody>
              {metrics.map(m => (
                <tr key={m.key}>
                  <td className="fw-bold">{m.label}</td>
                  <td><code>{m.range}</code></td>
                  <td className="small">{m.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ defs }) {
  if (!defs) return <div className="text-muted p-4">Loading…</div>;
  const concepts = defs.concepts || [];
  const refs = defs.references || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Key Concepts</strong>
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-dark">
              <tr><th style={{ width: '25%' }}>Term</th><th>Definition</th></tr>
            </thead>
            <tbody>
              {concepts.map((c, i) => (
                <tr key={i}>
                  <td className="fw-bold">{c.term}</td>
                  <td className="small">{c.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header py-2 bg-dark text-white">
          <strong>References</strong>
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-dark">
              <tr><th>Citation</th><th>Relevance</th></tr>
            </thead>
            <tbody>
              {refs.map((r, i) => (
                <tr key={i}>
                  <td className="small fst-italic">{r.citation}</td>
                  <td className="small">{r.relevance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function TransferLearningPage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/transfer-learning/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setError(e.message));
    fetch(`${API}/api/transfer-learning/definitions`)
      .then(r => r.json())
      .then(setDefs)
      .catch(e => setError(e.message));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: '1.6rem' }}>&#x1f504;</span>
        <div>
          <h4 className="mb-0 fw-bold">Transfer Learning / Cross-Patient Adaptation</h4>
          <div className="text-muted small">
            Domain adaptation strategies for EEG-based neuropsychiatric classification across patients
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger">API error: {error}</div>}

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

      <div>
        {tab === 'overview'    && <OverviewPanel    ov={overview} />}
        {tab === 'patients'    && <PatientsPanel    ov={overview} />}
        {tab === 'strategies'  && <StrategiesPanel  defs={defs} />}
        {tab === 'definitions' && <DefinitionsPanel defs={defs} />}
      </div>
    </div>
  );
}
