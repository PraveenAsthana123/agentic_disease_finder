'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function GuidedAssessmentDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/guided-assessment/overview`).then(r=>r.json()),
      fetch(`${API}/api/guided-assessment/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/guided-assessment/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading guided assessment data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'sessions', label:'Session Log'},
    {id:'instruments', label:'Instruments'},
    {id:'patients', label:'Per Patient'},
    {id:'definitions', label:'Definitions'},
  ];

  const statusColor = s => ({completed:'success',in_progress:'warning',abandoned:'danger'}[s]||'secondary');
  const statusBadge = s => <span className={`badge bg-${statusColor(s)}`}>{s.replace('_',' ')}</span>;
  const channelIcon = c => c==='voice_ai' ? '🎙️' : '💬';

  return (<div className="p-3">
    <h3>Guided Assessment Sessions Dashboard</h3>
    <p className="text-muted">
      Conversational &amp; voice-AI-guided clinical assessments &mdash; {ov.kpis.find(k=>k.label==='Total Sessions')?.value} sessions,{' '}
      {ov.kpis.find(k=>k.label==='Patients')?.value} patients,{' '}
      {ov.kpis.find(k=>k.label==='Instruments')?.value} instruments,{' '}
      {ov.kpis.find(k=>k.label==='Completion Rate')?.value} completion rate
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {ov.kpis.map(k=>
          <div key={k.label} className="col-6 col-md-3 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className="h5 mb-0" style={{color:k.color||'#333'}}>{k.value}</div>
              <div className="text-muted small">{k.label}</div>
            </div></div>
          </div>
        )}
      </div>

      <div className="row mb-3">
        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Session Status</h6>
            {ov.status_distribution.map(s=>{
              const total = ov.status_distribution.reduce((a,b)=>a+b.value,0);
              const pct = ((s.value/total)*100).toFixed(1);
              return <div key={s.name} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span className="text-capitalize">{s.name.replace('_',' ')}</span>
                  <span>{s.value} ({pct}%)</span>
                </div>
                <div className="progress" style={{height:'16px'}}>
                  <div className={`progress-bar bg-${statusColor(s.name)}`} style={{width:`${pct}%`}}/>
                </div>
              </div>;
            })}
          </div></div>
        </div>

        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Instrument Distribution</h6>
            {ov.instrument_distribution.map(i=>{
              const total = ov.instrument_distribution.reduce((a,b)=>a+b.value,0);
              const pct = ((i.value/total)*100).toFixed(0);
              return <div key={i.name} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{i.name}</span><span>{i.value} ({pct}%)</span>
                </div>
                <div className="progress" style={{height:'14px'}}>
                  <div className="progress-bar bg-primary" style={{width:`${pct}%`}}/>
                </div>
              </div>;
            })}
          </div></div>
        </div>

        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Channel</h6>
            {ov.channel_distribution.map(c=>{
              const total = ov.channel_distribution.reduce((a,b)=>a+b.value,0);
              const pct = ((c.value/total)*100).toFixed(0);
              return <div key={c.name} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{channelIcon(c.name)} {c.name.replace('_',' ')}</span>
                  <span>{c.value} ({pct}%)</span>
                </div>
                <div className="progress" style={{height:'16px'}}>
                  <div className={`progress-bar ${c.name==='voice_ai'?'bg-success':'bg-info'}`} style={{width:`${pct}%`}}/>
                </div>
              </div>;
            })}

            <h6 className="mt-3">Score Distribution</h6>
            {ov.score_distribution.map(s=>{
              const total = ov.score_distribution.reduce((a,b)=>a+b.count,0);
              const pct = ((s.count/total)*100).toFixed(0);
              return <div key={s.label} className="mb-1">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{s.label}</span><span>{s.count}</span>
                </div>
                <div className="progress" style={{height:'12px'}}>
                  <div className="progress-bar bg-secondary" style={{width:`${pct}%`}}/>
                </div>
              </div>;
            })}
          </div></div>
        </div>
      </div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Daily Session Trend</h6>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead><tr><th>Date</th><th>Started</th><th>Completed</th><th>Abandoned</th><th>Completion %</th></tr></thead>
            <tbody>
              {ov.daily_trend.map(d=>{
                const pct = d.started>0 ? ((d.completed/d.started)*100).toFixed(0) : '-';
                return <tr key={d.date}>
                  <td>{d.date}</td>
                  <td>{d.started}</td>
                  <td><span className="text-success fw-bold">{d.completed}</span></td>
                  <td><span className={d.abandoned>0?'text-danger':''}>{d.abandoned}</span></td>
                  <td>
                    {d.started>0 && <div className="progress" style={{height:'16px',minWidth:'80px'}}>
                      <div className={`progress-bar ${parseInt(pct)>=80?'bg-success':parseInt(pct)>=50?'bg-warning':'bg-danger'}`}
                        style={{width:`${pct}%`}}>{pct}%</div>
                    </div>}
                  </td>
                </tr>;
              })}
            </tbody>
          </table>
        </div>
      </div></div>

      <div className="card shadow-sm"><div className="card-body">
        <h6>Duration Histogram</h6>
        <div className="d-flex gap-3 align-items-end" style={{height:'80px'}}>
          {ov.duration_histogram.map(h=>{
            const max = Math.max(...ov.duration_histogram.map(x=>x.count));
            const pct = max>0 ? (h.count/max)*100 : 0;
            return <div key={h.label} className="text-center flex-fill">
              <div className="bg-primary rounded-top" style={{height:`${pct}%`,minHeight:h.count>0?'8px':'0'}}/>
              <div className="small text-muted mt-1">{h.label}</div>
              <div className="small fw-bold">{h.count}</div>
            </div>;
          })}
        </div>
      </div></div>
    </div>}

    {tab==='sessions' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Session Log ({bd.session_log.length} sessions)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead><tr>
              <th>Session</th><th>Patient</th><th>Instrument</th><th>Status</th>
              <th>Progress</th><th>Score</th><th>Interpretation</th><th>Channel</th><th>Duration</th>
            </tr></thead>
            <tbody>
              {bd.session_log.map(s=><tr key={s.session_id}>
                <td><code className="small">{s.session_id}</code></td>
                <td>{s.patient_name}</td>
                <td><span className="badge bg-primary">{s.instrument}</span></td>
                <td>{statusBadge(s.status)}</td>
                <td>
                  <div className="progress" style={{height:'16px',minWidth:'60px'}}>
                    <div className={`progress-bar bg-${statusColor(s.status)}`}
                      style={{width:s.progress}}>{s.progress}</div>
                  </div>
                </td>
                <td>{s.score !== '-' ? `${s.score}/${s.max_score}` : '-'}</td>
                <td><span className="small">{s.interpretation}</span></td>
                <td>{channelIcon(s.channel)} <span className="small">{s.channel.replace('_',' ')}</span></td>
                <td>{s.duration}</td>
              </tr>)}
            </tbody>
          </table>
        </div>
      </div></div>

      {bd.active_sessions?.length>0 && <div className="card shadow-sm mt-3"><div className="card-body">
        <h6>Active Sessions ({bd.active_sessions.length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-warning">
            <thead><tr><th>Session</th><th>Patient</th><th>Instrument</th><th>Item</th><th>Started</th></tr></thead>
            <tbody>
              {bd.active_sessions.map(s=><tr key={s.session_id}>
                <td><code className="small">{s.session_id}</code></td>
                <td>{s.patient_id}</td>
                <td><span className="badge bg-warning text-dark">{s.instrument}</span></td>
                <td>{s.current_item}/{s.total_items}</td>
                <td>{s.started_at}</td>
              </tr>)}
            </tbody>
          </table>
        </div>
      </div></div>}
    </div>}

    {tab==='instruments' && bd && <div>
      <div className="row">
        {bd.instrument_summary.map(inst=><div key={inst.instrument} className="col-md-6 mb-3">
          <div className="card shadow-sm h-100"><div className="card-body">
            <h6>{inst.instrument}</h6>
            <div className="d-flex gap-2 flex-wrap mb-2">
              <span className="badge bg-primary">{inst.total} sessions</span>
              <span className="badge bg-success">{inst.completed} completed</span>
              {inst.abandoned>0 && <span className="badge bg-danger">{inst.abandoned} abandoned</span>}
              <span className="badge bg-info">{inst.patients} patients</span>
            </div>
            {inst.completion_rate && <div className="mb-2">
              <div className="d-flex justify-content-between small mb-1">
                <span>Completion Rate</span><span>{inst.completion_rate}</span>
              </div>
              <div className="progress" style={{height:'16px'}}>
                <div className={`progress-bar ${parseFloat(inst.completion_rate)>=80?'bg-success':'bg-warning'}`}
                  style={{width:inst.completion_rate}}/>
              </div>
            </div>}
            <div className="row text-center mt-2">
              {inst.avg_score && <div className="col-4">
                <div className="small text-muted">Avg Score</div>
                <div className="fw-bold">{inst.avg_score}</div>
              </div>}
              {inst.avg_duration_s && <div className="col-4">
                <div className="small text-muted">Avg Duration</div>
                <div className="fw-bold">{Math.floor(inst.avg_duration_s/60)}m {Math.round(inst.avg_duration_s%60)}s</div>
              </div>}
              {inst.avg_items_completed && <div className="col-4">
                <div className="small text-muted">Avg Items</div>
                <div className="fw-bold">{inst.avg_items_completed}</div>
              </div>}
            </div>
          </div></div>
        </div>)}
      </div>
    </div>}

    {tab==='patients' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Per-Patient Summary ({bd.patient_summary.length} patients)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead><tr>
              <th>Patient</th><th>Sessions</th><th>Completed</th><th>In Progress</th><th>Abandoned</th>
              <th>Instruments</th><th>Completion %</th>
            </tr></thead>
            <tbody>
              {bd.patient_summary.sort((a,b)=>b.total-a.total).map(p=>{
                const pct = p.total>0 ? ((p.completed/p.total)*100).toFixed(0) : 0;
                return <tr key={p.patient_id}>
                  <td>{p.patient_name || p.patient_id}</td>
                  <td>{p.total}</td>
                  <td><span className="text-success">{p.completed}</span></td>
                  <td><span className="text-warning">{p.in_progress}</span></td>
                  <td><span className={p.abandoned>0?'text-danger':''}>{p.abandoned}</span></td>
                  <td><span className="small">{p.instruments}</span></td>
                  <td>
                    <div className="progress" style={{height:'16px',minWidth:'70px'}}>
                      <div className={`progress-bar ${parseInt(pct)>=80?'bg-success':parseInt(pct)>=50?'bg-warning':'bg-danger'}`}
                        style={{width:`${pct}%`}}>{pct}%</div>
                    </div>
                  </td>
                </tr>;
              })}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='definitions' && defs && <div>
      <div className="row">
        <div className="col-md-6">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Concepts</h6>
            {defs.concepts.map(c=><div key={c.name} className="mb-2">
              <div className="fw-bold small">{c.name}</div>
              <div className="text-muted small">{c.description}</div>
            </div>)}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Instruments</h6>
            <div className="table-responsive">
              <table className="table table-sm">
                <thead><tr><th>Name</th><th>Items</th><th>Max Score</th><th>Description</th></tr></thead>
                <tbody>
                  {defs.instruments.map(i=><tr key={i.name}>
                    <td><strong>{i.name}</strong></td>
                    <td>{i.items}</td>
                    <td>{i.max_score}</td>
                    <td className="small">{i.description}</td>
                  </tr>)}
                </tbody>
              </table>
            </div>
          </div></div>

          <div className="card shadow-sm"><div className="card-body">
            <h6>Quality Metrics</h6>
            {defs.quality_metrics?.map(q=><div key={q.metric} className="mb-2">
              <div className="d-flex justify-content-between">
                <span className="fw-bold small">{q.metric}</span>
                <span className="badge bg-primary">{q.value}</span>
              </div>
              <div className="text-muted small">{q.description}</div>
            </div>)}
          </div></div>
        </div>
      </div>
    </div>}
  </div>);
}
