'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const CAT_COLORS = {
  team_message: 'primary', clinical_decision: 'warning',
  expert_review: 'info', form_assignment: 'success', system: 'secondary',
};
export default function InboxPage() {
  const [data, setData] = useState(null);
  const [filter, setFilter] = useState('all');
  const [expanded, setExpanded] = useState(null);
  useEffect(() => { fetch(`${API}/api/inbox`).then(r => r.json()).then(setData).catch(() => {}); }, []);
  if (!data) return <div className="p-4"><div className="spinner-border text-primary" /></div>;
  const msgs = filter === 'all' ? data.messages : data.messages.filter(m => m.category === filter);
  const cats = data.summary?.by_category || [];
  return (
    <div>
      <h3>&#x2709;&#xfe0f; Message Inbox</h3>
      <p className="text-muted">Secure messages between you and your care team</p>
      {/* Summary cards */}
      <div className="row mb-3">
        <div className="col-6 col-md-3 mb-2">
          <div className="card text-center shadow-sm border-0 cursor-pointer" onClick={() => setFilter('all')} style={{cursor:'pointer'}}>
            <div className="card-body">
              <div className="h3 mb-0 text-primary">{data.total}</div>
              <div className="text-muted small">Total Messages</div>
            </div>
          </div>
        </div>
        {cats.map(c => (
          <div key={c.category} className="col-6 col-md-3 mb-2">
            <div className={`card text-center shadow-sm border-0${filter === c.category ? ' border-primary border-2' : ''}`}
                 onClick={() => setFilter(filter === c.category ? 'all' : c.category)} style={{cursor:'pointer'}}>
              <div className="card-body">
                <div className={`h3 mb-0 text-${CAT_COLORS[c.category] || 'secondary'}`}>{c.count}</div>
                <div className="text-muted small">{c.icon} {c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
      {/* Channel breakdown */}
      {data.summary?.by_channel?.length > 0 && (
        <div className="mb-3">
          <small className="text-muted">Channels: </small>
          {data.summary.by_channel.map(ch => (
            <span key={ch.channel} className="badge bg-secondary me-1">#{ch.channel} ({ch.count})</span>
          ))}
        </div>
      )}
      {/* Messages list */}
      <div className="list-group">
        {msgs.length === 0 && <div className="list-group-item text-muted">No messages in this category.</div>}
        {msgs.map((m, i) => (
          <div key={m.id} className={`list-group-item list-group-item-action${expanded === m.id ? ' active' : ''}`}
               onClick={() => setExpanded(expanded === m.id ? null : m.id)} style={{cursor:'pointer'}}>
            <div className="d-flex justify-content-between align-items-start">
              <div>
                <span className={`badge bg-${CAT_COLORS[m.category] || 'secondary'} me-2`}>
                  {m.category === 'team_message' ? '\ud83d\udcac' : m.category === 'clinical_decision' ? '\u2696\ufe0f' : m.category === 'expert_review' ? '\ud83d\udd0d' : '\ud83d\udccb'}
                  {' '}{(m.category || '').replace('_', ' ')}
                </span>
                <strong>{m.from_role}</strong>
                {m.is_bot && <span className="badge bg-light text-dark ms-1">Bot</span>}
                {m.channel && <span className="text-muted small ms-2">#{m.channel}</span>}
              </div>
              <small className="text-muted">{m.timestamp ? new Date(m.timestamp).toLocaleString() : '—'}</small>
            </div>
            <div className="mt-1"><strong>{m.subject}</strong></div>
            {expanded === m.id && (
              <div className="mt-2">
                <p className="mb-1">{m.body}</p>
                {m.patient_id && <small className="text-muted">Patient: {m.patient_id}</small>}
                {m.confidence != null && <span className="ms-2 badge bg-info">Confidence: {m.confidence}</span>}
                {m.status && <span className="ms-2 badge bg-warning text-dark">Status: {m.status}</span>}
                {m.read_by?.length > 0 && (
                  <div className="mt-1"><small className="text-muted">Read by: {m.read_by.join(', ')}</small></div>
                )}
                {m.attachment && <div className="mt-1"><small>Attachment: {m.attachment}</small></div>}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="alert alert-info small mt-3">
        Source: <code>clinical.db</code> — team_messages + clinical_decisions + expert_reviews + form_assignments.
        All messages are derived from real patient data. No fabricated content.
      </div>
    </div>
  );
}
