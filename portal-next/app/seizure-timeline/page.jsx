'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SUBJECT_COLORS = {
  chb01:'#3b82f6', chb02:'#8b5cf6', chb03:'#22c55e', chb04:'#f59e0b',
  chb05:'#ef4444', chb06:'#06b6d4', chb07:'#a3e635', chb08:'#f97316',
  chb09:'#ec4899', chb10:'#6366f1', chb11:'#14b8a6', chb12:'#84cc16',
};

function subjectColor(id) { return SUBJECT_COLORS[id] || '#6b7280'; }

function StatCard({label, value, color='#3b82f6', sub}) {
  return (
    <div className="col-6 col-md mb-2">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="h5 mb-0" style={{color}}>{value}</div>
          {sub && <div className="text-muted" style={{fontSize:'0.7rem'}}>{sub}</div>}
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Badge({label, color}) {
  return (
    <span style={{
      display:'inline-block', padding:'2px 8px', borderRadius:'4px',
      fontSize:'0.72rem', fontWeight:600, color:'#fff',
      backgroundColor: color || '#6b7280', marginRight:'4px',
    }}>{label}</span>
  );
}

function DurationBar({sec, maxSec}) {
  const pct = Math.min(100, (sec / (maxSec || 1)) * 100);
  return (
    <div style={{background:'#e5e7eb', borderRadius:'4px', height:'8px', minWidth:'80px'}}>
      <div style={{width:`${pct}%`, background:'#ef4444', borderRadius:'4px', height:'8px'}}></div>
    </div>
  );
}

function SubjectCard({ps, maxDur}) {
  return (
    <div className="card mb-2 shadow-sm">
      <div className="card-body py-2">
        <div className="d-flex align-items-center gap-2 mb-1">
          <Badge label={ps.subject} color={subjectColor(ps.subject)}/>
          <span className="fw-bold">{ps.total_seizures} seizures</span>
          <span className="text-muted small ms-auto">{ps.mean_duration_sec}s avg</span>
        </div>
        <DurationBar sec={ps.total_duration_sec} maxSec={maxDur}/>
        <div className="text-muted small mt-1">
          Total: {ps.total_duration_sec}s &nbsp;|&nbsp; {ps.files_with_seizures.length} file(s)
        </div>
      </div>
    </div>
  );
}

function EventRow({ev, idx}) {
  return (
    <tr>
      <td className="text-center">{idx + 1}</td>
      <td><Badge label={ev.subject} color={subjectColor(ev.subject)}/></td>
      <td style={{fontFamily:'monospace', fontSize:'0.82rem'}}>{ev.file}</td>
      <td className="text-center">{ev.seizure_index}</td>
      <td className="text-center">{ev.onset_clock}</td>
      <td className="text-center">{ev.start_sec}–{ev.end_sec}s</td>
      <td className="text-center">
        <span className="badge rounded-pill" style={{background:'#ef4444', color:'#fff'}}>
          {ev.duration_sec}s
        </span>
      </td>
    </tr>
  );
}

export default function SeizureTimelineDashboard() {
  const [overview, setOverview] = useState(null);
  const [events, setEvents] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [filterSubject, setFilterSubject] = useState('all');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/seizure-timeline/overview`).then(r => r.json()),
      fetch(`${API}/api/seizure-timeline/events`).then(r => r.json()),
      fetch(`${API}/api/seizure-timeline/definitions`).then(r => r.json()),
    ])
      .then(([ov, ev, df]) => { setOverview(ov); setEvents(ev); setDefs(df); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!overview) return <div className="text-muted p-3">Loading Seizure Timeline...</div>;

  if (!overview.available) return (
    <div className="alert alert-warning m-3">
      <strong>Data unavailable:</strong> {overview.error}
      <div className="text-muted small mt-1">
        CHB-MIT EEG data expected at <code>data/real_eeg/epilepsy_physionet/chb*/</code>
      </div>
    </div>
  );

  const ps = overview.per_subject_summary || [];
  const maxDur = Math.max(...ps.map(s => s.total_duration_sec), 1);

  const timeline = events?.timeline || [];
  const filtered = filterSubject === 'all'
    ? timeline
    : timeline.filter(e => e.subject === filterSubject);

  const subjects = overview.subjects || [];

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'events', label:'Event Log'},
    {id:'definitions', label:'Dataset Info'},
  ];

  return (
    <div className="p-3">
      <h3>Seizure Timeline Dashboard</h3>
      <p className="text-muted">
        CHB-MIT PhysioNet — {overview.subject_count} subjects &nbsp;|&nbsp;
        {overview.total_seizures} seizures &nbsp;|&nbsp;
        {overview.mean_seizure_duration_sec}s mean duration &nbsp;|&nbsp;
        {Math.round(overview.total_duration_sec / 60)}min total ictal burden
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div>
          <div className="row mb-3">
            <StatCard label="Subjects" value={overview.subject_count} color="#3b82f6"/>
            <StatCard label="Total Seizures" value={overview.total_seizures} color="#ef4444"/>
            <StatCard
              label="Mean Duration"
              value={`${overview.mean_seizure_duration_sec}s`}
              color="#f59e0b"
            />
            <StatCard
              label="Total Ictal Burden"
              value={`${Math.round(overview.total_duration_sec / 60)} min`}
              color="#8b5cf6"
            />
          </div>

          <h5 className="mb-2">Per-Subject Summary</h5>
          {ps.map(s => (
            <SubjectCard key={s.subject} ps={s} maxDur={maxDur}/>
          ))}

          <div className="card mt-3 shadow-sm">
            <div className="card-header fw-bold">Subject Comparison</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>Subject</th>
                    <th className="text-center">Seizures</th>
                    <th className="text-center">Mean Duration</th>
                    <th className="text-center">Total Ictal (s)</th>
                    <th className="text-center">Files</th>
                  </tr>
                </thead>
                <tbody>
                  {ps.map(s => (
                    <tr key={s.subject}>
                      <td><Badge label={s.subject} color={subjectColor(s.subject)}/></td>
                      <td className="text-center">{s.total_seizures}</td>
                      <td className="text-center">{s.mean_duration_sec}s</td>
                      <td className="text-center">{s.total_duration_sec}s</td>
                      <td className="text-center">{s.files_with_seizures.length}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Events Tab */}
      {tab === 'events' && (
        <div>
          <div className="d-flex align-items-center gap-2 mb-3">
            <label className="fw-bold small">Subject:</label>
            <select
              className="form-select form-select-sm"
              style={{maxWidth:'160px'}}
              value={filterSubject}
              onChange={e => setFilterSubject(e.target.value)}
            >
              <option value="all">All Subjects</option>
              {subjects.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <span className="text-muted small ms-auto">
              {filtered.length} of {timeline.length} events
            </span>
          </div>

          <div className="table-responsive">
            <table className="table table-sm table-striped table-hover">
              <thead className="table-dark">
                <tr>
                  <th className="text-center">#</th>
                  <th>Subject</th>
                  <th>File</th>
                  <th className="text-center">Szr#</th>
                  <th className="text-center">Onset Clock</th>
                  <th className="text-center">Offset (s)</th>
                  <th className="text-center">Duration</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((ev, i) => <EventRow key={i} ev={ev} idx={i}/>)}
              </tbody>
            </table>
          </div>

          {filtered.length === 0 && (
            <div className="text-muted text-center p-3">No events for selected filter.</div>
          )}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="card mb-3 shadow-sm">
            <div className="card-header fw-bold">Dataset</div>
            <div className="card-body">
              <p>{defs.dataset}</p>
              <p className="text-muted small">{defs.format}</p>
              <p className="text-muted small mb-0">
                <strong>Reference:</strong> {defs.reference}
              </p>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-bold">Field Glossary</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Field</th>
                    <th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(defs.fields || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td><code>{k}</code></td>
                      <td className="text-muted small">{v}</td>
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
