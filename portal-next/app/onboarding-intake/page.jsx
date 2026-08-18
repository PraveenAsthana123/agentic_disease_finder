'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Intake Steps' },
  { id: 'definitions', label: 'Definitions' },
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

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (!data.available) return <div className="alert alert-warning">{data.note || 'No data available'}</div>;

  const s = data.summary || {};
  const groups = data.group_distribution || [];
  const steps = data.steps_table || [];
  const intakeVsDeferred = data.intake_vs_deferred || [];

  return (
    <div>
      {data.note && (
        <div className="alert alert-info small mb-3">
          <strong>Note:</strong> {data.note}
        </div>
      )}
      {data.goal && (
        <div className="alert alert-success small mb-3">
          <strong>Goal:</strong> {data.goal}
        </div>
      )}

      <div className="row mb-3">
        <KPI label="True Intake Fields" value={s.true_intake_fields} color="primary" sub="captured once at registration" />
        <KPI label="Deferred Fields" value={s.deferred_fields} color="secondary" sub="captured over time via use" />
        <KPI label="15x Reduction" value={s.reduction} color="success" sub={s.time_saved} />
        <KPI label="Intake Groups" value={s.total_groups} color="info" sub="field categories" />
      </div>
      <div className="row mb-4">
        <KPI label="Extraction Sources" value={s.extraction_sources} color="warning" sub="EEG/MRI/EMR/notes" />
        <KPI label="Deferred Sections" value={s.deferred_sections_count} color="danger" sub="filled via portal use" />
      </div>

      <div className="row mb-4">
        {/* Intake vs Deferred */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Field Split</div>
            <div className="card-body">
              {intakeVsDeferred.map(({ name, value }) => {
                const total = intakeVsDeferred.reduce((a, b) => a + b.value, 0);
                const pct = total ? Math.round((value / total) * 100) : 0;
                const isIntake = name.toLowerCase().includes('intake');
                return (
                  <div key={name} className="mb-3">
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small fw-semibold">{name}</span>
                      <span className="badge bg-primary">{value} ({pct}%)</span>
                    </div>
                    <div className="progress" style={{ height: 14 }}>
                      <div
                        className={`progress-bar ${isIntake ? 'bg-primary' : 'bg-secondary'}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Group Distribution */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Intake Field Groups</div>
            <div className="card-body">
              {groups.map(({ name, value }) => (
                <div key={name} className="d-flex justify-content-between align-items-center mb-2">
                  <span className="small">{name}</span>
                  <div className="d-flex align-items-center gap-2">
                    <div className="progress" style={{ width: 80, height: 10 }}>
                      <div
                        className="progress-bar bg-info"
                        style={{ width: `${(value / 20) * 100}%` }}
                      />
                    </div>
                    <span className="badge bg-primary">{value}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 3-Step Summary */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Onboarding Steps</div>
            <div className="card-body">
              {steps.map(({ step, title, approach }) => (
                <div key={step} className="mb-3">
                  <div className="d-flex align-items-center gap-2 mb-1">
                    <span className="badge bg-dark rounded-pill">{step}</span>
                    <span className="small fw-semibold">{title}</span>
                  </div>
                  <span className="badge bg-success bg-opacity-75 small">{approach}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (!data.available) return <div className="alert alert-warning">No data available</div>;

  const step1 = data.step1 || {};
  const step2 = data.step2 || {};
  const step3 = data.step3 || {};

  return (
    <div>
      {/* Step 1 */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold d-flex align-items-center gap-2">
          <span className="badge bg-primary rounded-pill">Step 1</span>
          {step1.title}
          <span className="badge bg-success ms-auto">{step1.approach}</span>
        </div>
        <div className="card-body">
          <p className="text-muted small mb-3">
            <strong>{step1.total_intake_fields} fields</strong> across {step1.groups?.length || 0} groups — captured once at registration.
          </p>
          <div className="row">
            {(step1.groups || []).map(({ group, n, fields }) => (
              <div key={group} className="col-md-4 mb-3">
                <div className="card border-primary h-100">
                  <div className="card-header bg-primary text-white py-1 small fw-semibold d-flex justify-content-between">
                    <span>{group}</span>
                    <span className="badge bg-light text-primary">{n} fields</span>
                  </div>
                  <div className="card-body py-2">
                    <ul className="list-unstyled mb-0">
                      {(fields || []).map(f => (
                        <li key={f} className="small text-muted">&#x2022; {f}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Step 2 */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold d-flex align-items-center gap-2">
          <span className="badge bg-success rounded-pill">Step 2</span>
          {step2.title}
          <span className="badge bg-info ms-auto">{step2.approach}</span>
        </div>
        <div className="card-body">
          {step2.note && (
            <div className="alert alert-success small mb-3">{step2.note}</div>
          )}
          <div className="row">
            {(step2.extracts || []).map(({ doc, fills }) => (
              <div key={doc} className="col-md-3 mb-3">
                <div className="card border-success h-100">
                  <div className="card-header bg-success text-white py-1 small fw-semibold">{doc}</div>
                  <div className="card-body py-2">
                    <ul className="list-unstyled mb-0">
                      {(fills || []).map(f => (
                        <li key={f} className="small text-muted">&#x2713; {f}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Step 3 */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold d-flex align-items-center gap-2">
          <span className="badge bg-secondary rounded-pill">Step 3</span>
          {step3.title}
          <span className="badge bg-secondary ms-auto">{step3.approach}</span>
        </div>
        <div className="card-body">
          {step3.note && (
            <div className="alert alert-secondary small mb-3">{step3.note}</div>
          )}
          <p className="text-muted small">
            ~{step3.deferred_field_estimate} deferred fields filled through portal use — not captured at intake.
          </p>
          <div className="row">
            {(step3.deferred_sections || []).map(({ section, capture }) => (
              <div key={section} className="col-md-4 mb-2">
                <div className="d-flex align-items-start gap-2">
                  <span className="badge bg-secondary mt-1">&#x23f0;</span>
                  <div>
                    <div className="small fw-semibold">{section}</div>
                    <div className="text-muted" style={{ fontSize: '0.75rem' }}>{capture}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (!data.available) return <div className="alert alert-warning">No data available</div>;

  const steps = data.step_descriptions || [];
  const legend = data.field_classification_legend || [];
  const glossary = data.glossary || [];

  return (
    <div>
      {/* Step Descriptions */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Step Descriptions</div>
        <div className="card-body">
          {steps.map(({ step, title, description }) => (
            <div key={step} className="mb-3">
              <div className="d-flex align-items-center gap-2 mb-1">
                <span className="badge bg-dark rounded-pill">{step}</span>
                <span className="fw-semibold">{title}</span>
              </div>
              <p className="text-muted small ms-4 mb-0">{description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Field Classification Legend */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Field Classification Legend</div>
        <div className="card-body">
          <div className="row">
            {legend.map(({ type, color, description }) => (
              <div key={type} className="col-md-4 mb-3">
                <div className="d-flex align-items-start gap-2">
                  <div style={{ width: 14, height: 14, borderRadius: 3, background: color, flexShrink: 0, marginTop: 3 }} />
                  <div>
                    <div className="small fw-semibold">{type}</div>
                    <div className="text-muted" style={{ fontSize: '0.75rem' }}>{description}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Glossary */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Glossary</div>
        <div className="card-body">
          <div className="table-responsive">
            <table className="table table-hover table-sm">
              <thead className="table-dark">
                <tr>
                  <th>Term</th>
                  <th>Definition</th>
                </tr>
              </thead>
              <tbody>
                {glossary.map(({ term, definition }) => (
                  <tr key={term}>
                    <td className="fw-semibold small">{term}</td>
                    <td className="small text-muted">{definition}</td>
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

export default function OnboardingIntakeDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/onboarding-intake/overview`).then(r => r.json()),
      fetch(`${API}/api/onboarding-intake/breakdown`).then(r => r.json()),
      fetch(`${API}/api/onboarding-intake/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading onboarding intake...</div>;

  const s = ov.summary || {};

  return (
    <div className="p-3">
      <h3>Patient Onboarding — Intake Classification</h3>
      <p className="text-muted">
        <strong>{s.true_intake_fields}</strong> true intake fields &middot;{' '}
        <strong>{s.deferred_fields}</strong> deferred (longitudinal) &middot;{' '}
        <strong className="text-success">{s.reduction} time reduction</strong> &mdash; {s.time_saved}
      </p>

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

      {tab === 'overview' && <OverviewPanel data={ov} />}
      {tab === 'breakdown' && <BreakdownPanel data={bd} />}
      {tab === 'definitions' && <DefinitionsPanel data={defs} />}
    </div>
  );
}
