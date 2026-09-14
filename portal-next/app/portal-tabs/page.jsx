'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STATUS_COLOR = { built: 'success', partial: 'warning', planned: 'secondary' };

const TAB_ICON = {
  forms:       '📋',
  campaign:    '📣',
  notification:'🔔',
  alert:       '🚨',
  inbox:       '✉️',
  medication:  '💊',
  therapy:     '🧘',
  assessments: '🧩',
  reports:     '📄',
  chat:        '💬',
  history:     '🕐',
};

export default function PortalTabsPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/portal-tabs/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/portal-tabs/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/portal-tabs/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpi = ov.kpis || {};
  const tabs_list = [
    { id: 'overview',   label: 'Overview' },
    { id: 'directory',  label: 'Tab Directory' },
    { id: 'definitions',label: 'Definitions' },
  ];

  const summaryTable = ov.summary_table || [];
  const epChart      = (ov.charts || {}).endpoints_per_tab || [];
  const maxEp        = Math.max(...epChart.map(e => e.value), 1);

  const bdTabs = (bd || {}).tabs || [];

  return (
    <div>
      <h3>🗂️ Patient Self-Service Portal — Tabs</h3>
      <p className="text-muted small">
        {ov.title} — {kpi.total_tabs} tabs covering forms, campaigns, notifications, alerts, medication,
        therapy, assessments, reports, AI chat, and history. {kpi.total_endpoints} real endpoints wired.
      </p>

      {/* KPIs */}
      <div className="row mb-3">
        {[
          { label: 'Total Tabs',       value: kpi.total_tabs,       color: 'primary' },
          { label: 'Built',            value: kpi.built,            color: 'success' },
          { label: 'Partial',          value: kpi.partial,          color: 'warning' },
          { label: 'Planned',          value: kpi.planned,          color: 'secondary' },
          { label: 'Total Endpoints',  value: kpi.total_endpoints,  color: 'info' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value ?? '—'}</div>
                <div className="text-muted" style={{ fontSize: '0.72rem' }}>{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs nav */}
      <ul className="nav nav-tabs mb-3">
        {tabs_list.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview tab ── */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Endpoints per tab chart */}
          <div className="col-md-5">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Endpoints per Tab</div>
              <div className="card-body p-3">
                {epChart.map(e => (
                  <div key={e.id} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{e.name}</span>
                      <span className="fw-bold">{e.value}</span>
                    </div>
                    <div className="progress" style={{ height: '10px' }}>
                      <div
                        className="progress-bar bg-primary"
                        style={{ width: `${(e.value / maxEp) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Status summary + tab list */}
          <div className="col-md-7">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Tab Summary</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr>
                      <th>Tab</th>
                      <th>Purpose</th>
                      <th>Status</th>
                      <th>Endpoints</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summaryTable.map(r => (
                      <tr key={r.id}>
                        <td>
                          <span className="me-1">{TAB_ICON[r.id] || '📌'}</span>
                          <span className="fw-semibold small">{r.label}</span>
                        </td>
                        <td className="small text-muted" style={{ maxWidth: 220 }}>{r.purpose}</td>
                        <td>
                          <span className={`badge bg-${STATUS_COLOR[r.status] || 'secondary'}`}>
                            {r.status}
                          </span>
                        </td>
                        <td className="small text-muted" style={{ maxWidth: 200, wordBreak: 'break-all' }}>
                          {r.maps_to}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab Directory ── */}
      {tab === 'directory' && (
        <div className="row g-3">
          {bdTabs.map(t => (
            <div key={t.id} className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header d-flex justify-content-between align-items-center">
                  <span className="fw-semibold">
                    <span className="me-2">{TAB_ICON[t.id] || '📌'}</span>
                    {t.label}
                  </span>
                  <span className={`badge bg-${STATUS_COLOR[t.status] || 'secondary'}`}>
                    {t.status}
                  </span>
                </div>
                <div className="card-body py-2">
                  <p className="text-muted small mb-2">{t.purpose}</p>
                  <div className="mb-1">
                    <span className="text-muted small fw-semibold">Endpoints ({t.endpoint_count}):</span>
                    <div className="mt-1">
                      {(t.endpoints || []).map((ep, i) => (
                        <code key={i} className="d-block small text-primary mb-1">{ep}</code>
                      ))}
                    </div>
                  </div>
                  <div className="text-muted small">
                    <span className="fw-semibold">Maps to: </span>{t.maps_to}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Definitions tab ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-md-5">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Status Legend</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Status</th><th>Meaning</th></tr></thead>
                  <tbody>
                    {(defs.status_legend || []).map(s => (
                      <tr key={s.status}>
                        <td>
                          <span className={`badge bg-${STATUS_COLOR[s.status] || 'secondary'}`}>
                            {s.status}
                          </span>
                        </td>
                        <td className="small">{s.meaning}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-7">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Glossary</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Term</th><th>Definition</th></tr></thead>
                  <tbody>
                    {(defs.glossary || []).map(g => (
                      <tr key={g.term}>
                        <td className="small fw-semibold text-nowrap">{g.term}</td>
                        <td className="small text-muted">{g.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
