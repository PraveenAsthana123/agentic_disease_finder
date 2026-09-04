'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const COLOR   = '#b71c1c';  // emergency red
const COLOR2  = '#1565c0';  // phone blue
const COLOR3  = '#2e7d32';  // resolved green
const COLOR4  = '#f57f17';  // warning amber

const TABS = [
  { id: 'overview',   label: 'Overview' },
  { id: 'patients',   label: 'Per Patient' },
  { id: 'sla',        label: 'SLA Analysis' },
  { id: 'chain',      label: 'Escalation Chain' },
  { id: 'contacts',   label: 'Contacts' },
  { id: 'definitions',label: 'Definitions' },
];

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-2">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Bar({ label, value, max, color = COLOR }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-1">
      <div className="d-flex justify-content-between small mb-1">
        <span className="text-capitalize">{String(label).replace(/-/g, ' ')}</span>
        <span className="fw-semibold">{value} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 6 }}>
        <div className="progress-bar" style={{ width: pct + '%', backgroundColor: color }} />
      </div>
    </div>
  );
}

function Sect({ title, children }) {
  return (
    <div className="card shadow-sm mb-3">
      <div className="card-header fw-semibold py-2">{title}</div>
      <div className="card-body p-2">{children}</div>
    </div>
  );
}

export default function PhoneAlertsSosDashboard() {
  const [tab, setTab] = useState('overview');
  const [ov,  setOv]  = useState(null);
  const [bd,  setBd]  = useState(null);
  const [defs,setDefs]= useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/phone-alerts-sos/overview`).then(r => r.json()),
      fetch(`${API}/api/phone-alerts-sos/breakdown`).then(r => r.json()),
      fetch(`${API}/api/phone-alerts-sos/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err)  return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov)  return <div className="text-muted p-4 text-center"><div className="spinner-border" style={{ color: COLOR }} /><p className="mt-2">Loading phone alerts…</p></div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded text-white"
           style={{ background: `linear-gradient(135deg, ${COLOR} 0%, #c62828 100%)` }}>
        <h4 className="mb-1 fw-bold">📱 Phone Alerts & SOS Escalation</h4>
        <div style={{ fontSize: '0.83rem', opacity: 0.92 }}>
          {ov.total_events} events &nbsp;·&nbsp; {ov.total_patients} patients &nbsp;·&nbsp;
          {ov.phone_initiated_events} phone-initiated ({ov.phone_pct}%) &nbsp;·&nbsp;
          {ov.auto_initiated_events} automated ({ov.auto_pct}%)
        </div>
        <div style={{ fontSize: '0.78rem', opacity: 0.85 }}>
          SLA ≤120s: {ov.sla?.pct_under_120s}% &nbsp;·&nbsp;
          Severe rate: {ov.severe_rate_pct}% &nbsp;·&nbsp;
          Contact coverage: {ov.contacts?.coverage_pct}%
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={'nav-link' + (tab === t.id ? ' active' : '')}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewTab    ov={ov} />}
      {tab === 'patients'    && <PatientsTab    bd={bd} />}
      {tab === 'sla'         && <SlaTab         ov={ov} bd={bd} />}
      {tab === 'chain'       && <ChainTab       bd={bd} />}
      {tab === 'contacts'    && <ContactsTab    ov={ov} bd={bd} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  );
}

function OverviewTab({ ov }) {
  const sla  = ov.sla || {};
  const evts = Object.entries(ov.event_type_distribution || {});
  const trgs = Object.entries(ov.trigger_method_distribution || {});
  const outs = Object.entries(ov.outcome_distribution || {});
  const totalEvt = evts.reduce((s,[,v])=>s+v,0);
  const totalTrg = trgs.reduce((s,[,v])=>s+v,0);
  const totalOut = outs.reduce((s,[,v])=>s+v,0);

  const kpis = [
    ['Total Events',    ov.total_events,            COLOR],
    ['Patients',        ov.total_patients,           COLOR2],
    ['Phone-Initiated', ov.phone_initiated_events,   COLOR2],
    ['Auto / Caregiver',ov.auto_initiated_events,    '#6a1b9a'],
    ['Severe Events',   ov.severe_events,            COLOR],
    ['Severe Rate',     ov.severe_rate_pct + '%',    COLOR],
    ['Avg Response',    Math.round(sla.avg_rt) + 's', COLOR4],
    ['SLA ≤120s',       sla.pct_under_120s + '%',   COLOR3],
    ['Location Shared', ov.location_shared_pct + '%',COLOR2],
    ['Responder Notified',ov.responder_notified_pct+'%',COLOR3],
  ];

  const outcomeColors = {
    'ems-dispatched':     COLOR,
    'er-visit':           '#c62828',
    'caregiver-responded':COLOR3,
    'resolved-home':      COLOR3,
    'false-alarm':        COLOR4,
  };

  const monthly = ov.monthly_trend || [];

  return (
    <div>
      <div className="row mb-3">
        {kpis.map(([l, v, c]) => <KPI key={l} label={l} value={v} color={c} />)}
      </div>

      <div className="row">
        <div className="col-md-4 mb-3">
          <Sect title="🚨 Event Types">
            {evts.map(([k,v]) => <Bar key={k} label={k} value={v} max={totalEvt} color={COLOR} />)}
          </Sect>
        </div>
        <div className="col-md-4 mb-3">
          <Sect title="📱 Trigger Methods">
            {trgs.map(([k,v]) => (
              <Bar key={k} label={k} value={v} max={totalTrg}
                   color={['app-button','voice-command'].includes(k) ? COLOR2 : '#6a1b9a'} />
            ))}
            <div className="mt-2 p-2 rounded small" style={{ background: '#e3f2fd' }}>
              <span className="fw-semibold" style={{ color: COLOR2 }}>Phone-initiated</span>
              {' '}(app-button, voice-command): {ov.phone_initiated_events} ({ov.phone_pct}%)
            </div>
          </Sect>
        </div>
        <div className="col-md-4 mb-3">
          <Sect title="🎯 Outcomes">
            {outs.map(([k,v]) => (
              <Bar key={k} label={k} value={v} max={totalOut}
                   color={outcomeColors[k] || '#555'} />
            ))}
          </Sect>
        </div>
      </div>

      {monthly.length > 0 && (
        <Sect title="📅 Monthly Alert Trend">
          <div className="d-flex align-items-end gap-2 overflow-auto" style={{ minHeight: 80 }}>
            {(() => {
              const maxCnt = Math.max(...monthly.map(m => m.cnt), 1);
              return monthly.map(m => (
                <div key={m.month} className="text-center" style={{ minWidth: 50 }}>
                  <div style={{
                    height: Math.round((m.cnt / maxCnt) * 70) + 10,
                    backgroundColor: COLOR,
                    borderRadius: 3,
                    marginBottom: 2,
                  }} />
                  <div style={{ fontSize: '0.65rem', color: '#555' }}>{m.month?.slice(5)}</div>
                  <div style={{ fontSize: '0.7rem', fontWeight: 600 }}>{m.cnt}</div>
                </div>
              ));
            })()}
          </div>
        </Sect>
      )}
    </div>
  );
}

function PatientsTab({ bd }) {
  const rows = bd?.patient_profile || [];
  return (
    <Sect title={`Per-Patient Alert Profile (${rows.length} patients)`}>
      <div className="table-responsive">
        <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.78rem' }}>
          <thead className="table-light">
            <tr>
              <th>Patient</th>
              <th>Total</th>
              <th>Phone</th>
              <th>Auto</th>
              <th>Avg RT (s)</th>
              <th>Severe</th>
              <th>False Alarm</th>
              <th>Location Shared</th>
              <th>Contacts</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.patient_id}>
                <td className="fw-semibold">{r.patient_id}</td>
                <td>{r.total_events}</td>
                <td style={{ color: COLOR2 }}>{r.phone_alerts}</td>
                <td style={{ color: '#6a1b9a' }}>{r.auto_alerts}</td>
                <td>
                  <span style={{ color: r.avg_rt > 300 ? COLOR : r.avg_rt > 120 ? COLOR4 : COLOR3 }}>
                    {r.avg_rt}
                  </span>
                </td>
                <td style={{ color: r.severe_events > 0 ? COLOR : '#555' }}>{r.severe_events}</td>
                <td style={{ color: r.false_alarms > 0 ? COLOR4 : '#555' }}>{r.false_alarms}</td>
                <td>{r.location_shared}</td>
                <td style={{ color: r.contacts === 0 ? COLOR : COLOR3 }}>{r.contacts}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Sect>
  );
}

function SlaTab({ ov, bd }) {
  const sla    = ov.sla || {};
  const thresh = ov.sla_thresholds || {};
  const byTrig = bd?.sla_by_trigger || [];
  const breaches = bd?.sla_breaches || [];

  const tierBars = [
    { label: `Critical ≤${thresh.critical_s}s`, count: sla.critical_met,  color: COLOR3 },
    { label: `Standard ≤${thresh.standard_s}s`, count: sla.standard_met,  color: COLOR2 },
    { label: `Extended ≤${thresh.extended_s}s`, count: sla.extended_met,  color: COLOR4 },
    { label: `Breach >${thresh.extended_s}s`,   count: sla.breach,         color: COLOR  },
  ];
  const total = ov.total_events || 1;

  return (
    <div>
      <div className="row mb-3">
        {[
          ['Avg Response', Math.round(sla.avg_rt) + 's', COLOR4],
          ['Min Response', sla.min_rt + 's', COLOR3],
          ['Max Response', sla.max_rt + 's', COLOR],
          ['≤60s Critical',sla.pct_under_60s + '%', COLOR3],
          ['≤120s Standard',sla.pct_under_120s + '%', COLOR2],
          ['Breaches', sla.breach, COLOR],
        ].map(([l,v,c]) => <KPI key={l} label={l} value={v} color={c} />)}
      </div>

      <div className="row">
        <div className="col-md-5 mb-3">
          <Sect title="SLA Tier Distribution">
            {tierBars.map(t => (
              <Bar key={t.label} label={t.label} value={t.count} max={total} color={t.color} />
            ))}
          </Sect>
        </div>
        <div className="col-md-7 mb-3">
          <Sect title="SLA Compliance by Trigger Method">
            <div className="table-responsive">
              <table className="table table-sm mb-0" style={{ fontSize: '0.78rem' }}>
                <thead className="table-light">
                  <tr><th>Trigger</th><th>Total</th><th>≤60s</th><th>≤120s</th><th>Breach</th><th>Avg RT</th></tr>
                </thead>
                <tbody>
                  {byTrig.map(r => (
                    <tr key={r.trigger_method}>
                      <td className="text-capitalize">{r.trigger_method?.replace(/-/g,' ')}</td>
                      <td>{r.total}</td>
                      <td style={{ color: COLOR3 }}>{r.critical_met}</td>
                      <td style={{ color: COLOR2 }}>{r.standard_met}</td>
                      <td style={{ color: r.breach > 0 ? COLOR : '#555' }}>{r.breach}</td>
                      <td style={{ color: r.avg_rt > 300 ? COLOR : r.avg_rt > 120 ? COLOR4 : COLOR3 }}>
                        {r.avg_rt}s
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Sect>
        </div>
      </div>

      {breaches.length > 0 && (
        <Sect title={`⚠️ SLA Breaches — >300s Response (${breaches.length} events)`}>
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.76rem' }}>
              <thead className="table-light">
                <tr><th>ID</th><th>Patient</th><th>Date</th><th>Type</th><th>Trigger</th><th>RT (s)</th><th>Outcome</th><th>Notes</th></tr>
              </thead>
              <tbody>
                {breaches.map(r => (
                  <tr key={r.id}>
                    <td>{r.id}</td>
                    <td>{r.patient_id}</td>
                    <td>{r.event_date?.slice(0, 10)}</td>
                    <td className="text-capitalize">{r.event_type?.replace(/-/g,' ')}</td>
                    <td className="text-capitalize">{r.trigger_method?.replace(/-/g,' ')}</td>
                    <td style={{ color: COLOR, fontWeight: 600 }}>{r.response_time_seconds}</td>
                    <td className="text-capitalize">{r.outcome?.replace(/-/g,' ')}</td>
                    <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Sect>
      )}
    </div>
  );
}

function ChainTab({ bd }) {
  const chain   = bd?.escalation_chain || [];
  const byType  = bd?.phone_vs_auto_by_type || [];
  const recent  = bd?.recent_events || [];

  // Group chain by trigger method
  const byTrigger = {};
  chain.forEach(r => {
    if (!byTrigger[r.trigger_method]) byTrigger[r.trigger_method] = [];
    byTrigger[r.trigger_method].push(r);
  });

  const outcomeColors = { 'ems-dispatched': COLOR, 'er-visit': '#c62828', 'caregiver-responded': COLOR3, 'resolved-home': COLOR3, 'false-alarm': COLOR4 };

  return (
    <div>
      <div className="row mb-3">
        <div className="col-md-6 mb-3">
          <Sect title="Escalation Chain — Trigger → Outcome">
            {Object.entries(byTrigger).map(([trig, rows]) => (
              <div key={trig} className="mb-3">
                <div className="fw-semibold text-capitalize mb-1" style={{ color: ['app-button','voice-command'].includes(trig) ? COLOR2 : '#6a1b9a' }}>
                  {trig.replace(/-/g,' ')} ({rows.reduce((s,r)=>s+r.cnt,0)} events)
                </div>
                {rows.map(r => (
                  <div key={r.outcome} className="d-flex justify-content-between align-items-center small px-2 mb-1">
                    <span className="text-capitalize" style={{ color: outcomeColors[r.outcome] || '#555' }}>
                      → {r.outcome?.replace(/-/g,' ')}
                    </span>
                    <span className="badge" style={{ backgroundColor: outcomeColors[r.outcome] || '#777', color:'#fff' }}>
                      {r.cnt}
                    </span>
                  </div>
                ))}
              </div>
            ))}
          </Sect>
        </div>
        <div className="col-md-6 mb-3">
          <Sect title="Phone vs. Auto by Event Type">
            <div className="table-responsive">
              <table className="table table-sm mb-0" style={{ fontSize: '0.78rem' }}>
                <thead className="table-light">
                  <tr><th>Event Type</th><th style={{color:COLOR2}}>Phone</th><th style={{color:'#6a1b9a'}}>Auto</th><th>Total</th></tr>
                </thead>
                <tbody>
                  {byType.map(r => (
                    <tr key={r.event_type}>
                      <td className="text-capitalize">{r.event_type?.replace(/-/g,' ')}</td>
                      <td style={{ color: COLOR2 }}>{r.phone}</td>
                      <td style={{ color: '#6a1b9a' }}>{r.auto}</td>
                      <td>{r.total}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Sect>
        </div>
      </div>

      <Sect title={`Recent Events (${recent.length})`}>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.75rem' }}>
            <thead className="table-light">
              <tr><th>Patient</th><th>Date</th><th>Type</th><th>Trigger</th><th>RT (s)</th><th>Notified</th><th>Location</th><th>Outcome</th></tr>
            </thead>
            <tbody>
              {recent.map(r => (
                <tr key={r.id}>
                  <td>{r.patient_id}</td>
                  <td>{r.event_date?.slice(0,10)}</td>
                  <td className="text-capitalize">{r.event_type?.replace(/-/g,' ')}</td>
                  <td className="text-capitalize"
                      style={{ color: ['app-button','voice-command'].includes(r.trigger_method) ? COLOR2 : '#6a1b9a' }}>
                    {r.trigger_method?.replace(/-/g,' ')}
                  </td>
                  <td style={{ color: r.response_time_seconds > 300 ? COLOR : r.response_time_seconds > 120 ? COLOR4 : COLOR3 }}>
                    {r.response_time_seconds}
                  </td>
                  <td>{r.responder_notified ? '✅' : '❌'}</td>
                  <td>{r.location_shared ? '📍' : '—'}</td>
                  <td className="text-capitalize"
                      style={{ color: ['ems-dispatched','er-visit'].includes(r.outcome) ? COLOR : COLOR3 }}>
                    {r.outcome?.replace(/-/g,' ')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Sect>
    </div>
  );
}

function ContactsTab({ ov, bd }) {
  const contacts = ov.contacts || {};
  const stale    = bd?.stale_contacts || [];

  return (
    <div>
      <div className="row mb-3">
        {[
          ['Total Contacts', contacts.total, COLOR2],
          ['Patients Covered', contacts.patients_covered, COLOR3],
          ['No Contact', contacts.patients_no_contact, contacts.patients_no_contact > 0 ? COLOR : COLOR3],
          ['Coverage', contacts.coverage_pct + '%', COLOR3],
          ['Stale (>180d)', stale.length, stale.length > 0 ? COLOR4 : COLOR3],
        ].map(([l,v,c]) => <KPI key={l} label={l} value={v} color={c} />)}
      </div>

      {stale.length > 0 ? (
        <Sect title={`⚠️ Stale Contacts — Unverified >180 Days (${stale.length})`}>
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.78rem' }}>
              <thead className="table-light">
                <tr><th>Patient</th><th>Contact Name</th><th>Relationship</th><th>Phone</th><th>Last Verified</th><th>Days Stale</th></tr>
              </thead>
              <tbody>
                {stale.map((r, i) => (
                  <tr key={i}>
                    <td>{r.patient_id}</td>
                    <td>{r.contact_name}</td>
                    <td className="text-capitalize">{r.relationship}</td>
                    <td>{r.phone_number}</td>
                    <td>{r.last_verified?.slice(0,10)}</td>
                    <td style={{ color: r.days_stale > 365 ? COLOR : COLOR4, fontWeight: 600 }}>{r.days_stale}d</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Sect>
      ) : (
        <div className="alert alert-success">✅ All emergency contacts verified within the last 180 days.</div>
      )}
    </div>
  );
}

function DefinitionsTab({ defs }) {
  if (!defs) return null;
  const gl   = defs.glossary || [];
  const cats = defs.trigger_categories || [];
  const tiers= defs.sla_tiers || [];
  const outs = defs.outcome_descriptions || [];
  const stds = defs.readiness_standards || [];

  const severityColor = { critical: COLOR, high: '#c62828', moderate: COLOR4, low: COLOR3 };

  return (
    <div>
      <div className="row">
        <div className="col-md-6 mb-3">
          <Sect title="📖 Glossary">
            {gl.map(g => (
              <div key={g.term} className="mb-2">
                <span className="fw-semibold">{g.term}:</span>{' '}
                <span className="text-muted small">{g.definition}</span>
              </div>
            ))}
          </Sect>
        </div>
        <div className="col-md-6 mb-3">
          <Sect title="📱 Trigger Categories">
            {cats.map(c => (
              <div key={c.category} className="mb-2">
                <div className="fw-semibold" style={{ color: c.category.includes('Phone') ? COLOR2 : '#6a1b9a' }}>{c.category}</div>
                <div className="text-muted small mb-1">{c.description}</div>
                <div>{c.triggers.map(t => <span key={t} className="badge bg-secondary me-1">{t}</span>)}</div>
              </div>
            ))}
          </Sect>

          <Sect title="⏱️ SLA Tiers">
            {tiers.map(t => (
              <div key={t.tier} className="d-flex align-items-start mb-2">
                <span className="badge me-2" style={{ backgroundColor: t.tier==='Breach' ? COLOR : t.tier==='Critical' ? COLOR3 : t.tier==='Standard' ? COLOR2 : COLOR4, color:'#fff', minWidth: 70 }}>
                  {t.label}
                </span>
                <span className="small text-muted">{t.description}</span>
              </div>
            ))}
          </Sect>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6 mb-3">
          <Sect title="🎯 Outcome Descriptions">
            {outs.map(o => (
              <div key={o.outcome} className="d-flex align-items-start mb-2">
                <span className="badge me-2" style={{ backgroundColor: severityColor[o.severity] || '#555', color:'#fff', minWidth: 55, fontSize:'0.65rem' }}>
                  {o.severity}
                </span>
                <div>
                  <span className="fw-semibold small">{o.outcome}:</span>{' '}
                  <span className="text-muted small">{o.description}</span>
                </div>
              </div>
            ))}
          </Sect>
        </div>
        <div className="col-md-6 mb-3">
          <Sect title="✅ Readiness Standards">
            <div className="table-responsive">
              <table className="table table-sm mb-0" style={{ fontSize: '0.76rem' }}>
                <thead className="table-light"><tr><th>Metric</th><th>Target</th></tr></thead>
                <tbody>
                  {stds.map(s => (
                    <tr key={s.metric}>
                      <td className="fw-semibold">{s.metric}</td>
                      <td className="text-muted">{s.target}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Sect>
        </div>
      </div>
    </div>
  );
}
