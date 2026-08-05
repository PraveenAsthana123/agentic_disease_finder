'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function ReferrerNotifyDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [sort, setSort] = useState({col:'total', dir:'desc'});

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/referrer-notify/overview`).then(r=>r.json()),
      fetch(`${API}/api/referrer-notify/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/referrer-notify/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading referrer notification data...</div>;

  const TABS = [
    {id:'overview',   label:'Overview'},
    {id:'queue',      label:'Notification Queue'},
    {id:'sources',    label:'By Source'},
    {id:'patients',   label:'Per Patient'},
    {id:'definitions',label:'Definitions'},
  ];

  const kpis = ov.kpis || {};
  const statusColor = s => ({notified:'success', queued:'warning', in_progress:'info',
    not_ready:'secondary', cancelled:'danger'}[s]||'info');
  const urgColor = u => ({emergent:'danger', urgent:'warning', routine:'primary', elective:'secondary'}[u]||'info');

  const sortedSources = bd ? [...bd.per_source].sort((a,b)=>{
    const av = a[sort.col]??0, bv = b[sort.col]??0;
    return sort.dir==='asc' ? av-bv : bv-av;
  }) : [];
  const toggleSort = col => setSort(s=>({col, dir: s.col===col && s.dir==='desc'?'asc':'desc'}));
  const sortIcon = col => sort.col===col?(sort.dir==='desc'?'▼':'▲'):'⇅';

  const total = kpis.total_referrals || 0;

  return (<div className="p-3">
    <h3>&#x1f4e8; Referrer Notification Dashboard</h3>
    <p className="text-muted">
      Post-report referrer notification tracking &mdash; {total} referrals &middot; {kpis.total_patients} patients
      &middot; {kpis.notified} notified &middot; {kpis.queued} queued &middot; notify rate {kpis.notify_rate_pct}%
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      {/* KPI cards */}
      <div className="row mb-3">
        {[
          ['Total Referrals', total, 'primary'],
          ['Patients', kpis.total_patients, 'info'],
          ['Notified', kpis.notified, 'success'],
          ['Queued', kpis.queued, 'warning'],
          ['In Progress', kpis.in_progress, 'info'],
          ['Not Ready', kpis.not_ready, 'secondary'],
          ['Cancelled', kpis.cancelled, 'danger'],
          ['Notify Rate', kpis.notify_rate_pct+'%', 'success'],
        ].map(([label,val,c])=>
          <div key={label} className="col-6 col-md-3 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className={`h5 mb-0 text-${c}`}>{val}</div>
              <div className="text-muted small">{label}</div>
            </div></div>
          </div>
        )}
      </div>

      <div className="row mb-3">
        {/* Notification status breakdown */}
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Notification Status</h6>
            {[
              {label:'Notified', count:kpis.notified, color:'success'},
              {label:'Queued', count:kpis.queued, color:'warning'},
              {label:'In Progress', count:kpis.in_progress, color:'info'},
              {label:'Not Ready', count:kpis.not_ready, color:'secondary'},
              {label:'Cancelled', count:kpis.cancelled, color:'danger'},
            ].map(item=>{
              const pct = total ? ((item.count/total)*100).toFixed(1) : 0;
              return <div key={item.label} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'95px'}}>{item.label}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${item.color}`} style={{width:`${pct}%`}}>
                      {item.count} ({pct}%)
                    </div>
                  </div>
                </div>
              </div>;
            })}
            {kpis.queued > 0 &&
              <div className="alert alert-warning py-1 px-2 small mt-2 mb-0">
                &#x26a0;&#xfe0f; {kpis.queued} referrers awaiting notification
              </div>
            }
          </div></div>
        </div>

        {/* Source distribution */}
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Referral Source Distribution</h6>
            {(ov.source_summary||[]).map((s,i)=>{
              const colors = ['primary','success','warning','info','secondary','danger','dark'];
              const c = colors[i%colors.length];
              return <div key={s.source} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'140px'}}>{s.label}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className={`progress-bar bg-${c}`} style={{width:`${s.pct}%`}}>
                      {s.count} ({s.pct}%)
                    </div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
      </div>

      {/* Urgency summary */}
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Urgency × Notification Status</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead><tr>
              <th>Urgency</th><th>Total</th>
              <th className="text-success">Notified</th>
              <th className="text-warning">Queued</th>
              <th className="text-info">In Progress</th>
              <th className="text-secondary">Not Ready</th>
              <th className="text-danger">Cancelled</th>
            </tr></thead>
            <tbody>
              {(ov.urgency_summary||[]).map(u=>
                <tr key={u.urgency}>
                  <td><span className={`badge bg-${u.color}`}>{u.urgency}</span></td>
                  <td className="fw-bold">{u.total}</td>
                  <td>{u.notified}</td>
                  <td>{u.queued}</td>
                  <td>{u.in_progress}</td>
                  <td>{u.not_ready}</td>
                  <td>{u.cancelled}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {(ov.urgency_summary||[]).some(u=>u.urgency==='emergent' && (u.queued+u.in_progress)>0) &&
          <div className="alert alert-danger py-1 px-2 small mb-0">
            &#x1f6a8; Emergent referrals with pending notification — SLA &lt; 2 h
          </div>
        }
      </div></div>

      {/* Monthly trend */}
      {ov.monthly_trend && ov.monthly_trend.length > 0 &&
        <div className="card shadow-sm mb-3"><div className="card-body">
          <h6>Monthly Referrals &amp; Notifications</h6>
          <table className="table table-sm table-bordered">
            <thead><tr><th>Month</th><th>Referrals</th><th>Notified</th><th>Pending</th><th>Rate %</th></tr></thead>
            <tbody>
              {ov.monthly_trend.map(m=>
                <tr key={m.month}>
                  <td>{m.month}</td>
                  <td>{m.referrals}</td>
                  <td className="text-success">{m.notified}</td>
                  <td className="text-warning">{m.referrals - m.notified}</td>
                  <td>{m.referrals ? ((m.notified/m.referrals)*100).toFixed(0)+'%' : '—'}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div></div>
      }
    </div>}

    {tab==='queue' && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Notification Queue — {(bd?.queue||[]).length} referrals pending</h6>
        <p className="text-muted small mb-2">Sorted by urgency (emergent first) then triage score.</p>
        <div className="table-responsive">
          <table className="table table-sm table-bordered table-striped">
            <thead><tr>
              <th>#</th><th>Patient</th><th>Source</th><th>Reason</th>
              <th>Urgency</th><th>Score</th><th>Assigned To</th><th>Referral Date</th><th>Status</th>
            </tr></thead>
            <tbody>
              {(bd?.queue||[]).map(q=>
                <tr key={q.id}>
                  <td className="text-muted small">{q.id}</td>
                  <td className="fw-bold small">{q.patient_id}</td>
                  <td className="small">{q.source_label}</td>
                  <td className="small">{(q.reason||'').replace(/_/g,' ')}</td>
                  <td><span className={`badge bg-${q.urgency_color}`}>{q.urgency}</span></td>
                  <td className="fw-bold">{q.triage_score?.toFixed(0)}</td>
                  <td className="small">{q.assigned_to}</td>
                  <td className="small">{q.referral_date}</td>
                  <td><span className={`badge bg-${statusColor(q.notify_status)}`}>{(q.notify_status||'').replace(/_/g,' ')}</span></td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Recently Notified (Last 10)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead><tr>
              <th>Patient</th><th>Source</th><th>Reason</th>
              <th>Urgency</th><th>Assigned To</th><th>Notified Date</th><th>Score</th>
            </tr></thead>
            <tbody>
              {(bd?.recently_notified||[]).map(r=>
                <tr key={r.id}>
                  <td className="fw-bold small">{r.patient_id}</td>
                  <td className="small">{r.source_label}</td>
                  <td className="small">{(r.reason||'').replace(/_/g,' ')}</td>
                  <td><span className={`badge bg-${urgColor(r.urgency)}`}>{r.urgency}</span></td>
                  <td className="small">{r.assigned_to}</td>
                  <td className="small">{r.notified_date?.slice(0,10)}</td>
                  <td>{r.triage_score?.toFixed(0)}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='sources' && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Per-Source Notification Performance</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered table-hover">
            <thead><tr>
              <th>Source</th>
              <th onClick={()=>toggleSort('total')} style={{cursor:'pointer'}}>Total {sortIcon('total')}</th>
              <th onClick={()=>toggleSort('notified')} style={{cursor:'pointer'}}>Notified {sortIcon('notified')}</th>
              <th onClick={()=>toggleSort('queued')} style={{cursor:'pointer'}}>Queued {sortIcon('queued')}</th>
              <th onClick={()=>toggleSort('notify_rate')} style={{cursor:'pointer'}}>Rate% {sortIcon('notify_rate')}</th>
              <th onClick={()=>toggleSort('avg_triage_score')} style={{cursor:'pointer'}}>Avg Score {sortIcon('avg_triage_score')}</th>
            </tr></thead>
            <tbody>
              {sortedSources.map(s=>
                <tr key={s.source}>
                  <td className="fw-bold small">{s.label}</td>
                  <td>{s.total}</td>
                  <td className="text-success">{s.notified}</td>
                  <td className="text-warning">{s.queued}</td>
                  <td>
                    <div className="d-flex align-items-center">
                      <div className="progress flex-grow-1 me-2" style={{height:'14px'}}>
                        <div className={`progress-bar bg-${s.notify_rate>=50?'success':'warning'}`}
                             style={{width:`${s.notify_rate}%`}}></div>
                      </div>
                      <span className="small">{s.notify_rate}%</span>
                    </div>
                  </td>
                  <td>{s.avg_triage_score}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='patients' && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Per-Patient Notification Summary — {(bd?.per_patient||[]).length} patients</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered table-striped">
            <thead><tr>
              <th>Patient ID</th><th>Referrals</th>
              <th className="text-success">Notified</th>
              <th className="text-warning">Queued</th>
              <th>Pending</th><th>Max Score</th>
            </tr></thead>
            <tbody>
              {(bd?.per_patient||[]).map(p=>
                <tr key={p.patient_id}>
                  <td className="fw-bold small">{p.patient_id}</td>
                  <td>{p.total}</td>
                  <td className="text-success">{p.notified}</td>
                  <td className="text-warning">{p.queued}</td>
                  <td>{Math.max(0, p.pending||0)}</td>
                  <td>{p.max_score?.toFixed(0) || '—'}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='definitions' && defs && <div>
      {/* Workflow */}
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Workflow</h6>
        <p className="text-muted small">{defs.workflow?.description}</p>
        <ol className="small mb-0">
          {(defs.workflow?.steps||[]).map((s,i)=><li key={i}>{s}</li>)}
        </ol>
      </div></div>

      {/* Notification statuses */}
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Notification Statuses</h6>
        <table className="table table-sm table-bordered">
          <thead><tr><th>Status</th><th>Meaning</th></tr></thead>
          <tbody>
            {(defs.notification_statuses||[]).map(s=>
              <tr key={s.status}>
                <td><span className={`badge bg-${s.color}`}>{s.status.replace(/_/g,' ')}</span></td>
                <td className="small">{s.meaning}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>

      {/* Urgency SLAs */}
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Urgency Tiers &amp; SLAs</h6>
        <table className="table table-sm table-bordered">
          <thead><tr><th>Tier</th><th>SLA</th><th>Description</th></tr></thead>
          <tbody>
            {(defs.urgency_tiers||[]).map(u=>
              <tr key={u.tier}>
                <td><span className={`badge bg-${u.color}`}>{u.tier}</span></td>
                <td className="fw-bold small">{u.sla}</td>
                <td className="small">{u.description}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>

      {/* Referral sources */}
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Referral Sources</h6>
        <table className="table table-sm table-bordered">
          <thead><tr><th>ID</th><th>Label</th><th>Note</th></tr></thead>
          <tbody>
            {(defs.referral_sources||[]).map(s=>
              <tr key={s.id}>
                <td className="small text-muted">{s.id}</td>
                <td className="fw-bold small">{s.label}</td>
                <td className="small">{s.note}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>

      {/* Triage score & channel note */}
      <div className="row">
        <div className="col-md-6">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Triage Score</h6>
            <p className="small mb-0">
              <strong>Range:</strong> {defs.triage_score?.range}<br/>
              {defs.triage_score?.description}
            </p>
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm mb-3 border-warning"><div className="card-body">
            <h6>&#x26a0;&#xfe0f; Notification Channel</h6>
            <p className="small mb-0 text-muted">{defs.channel_note}</p>
          </div></div>
        </div>
      </div>
    </div>}
  </div>);
}
