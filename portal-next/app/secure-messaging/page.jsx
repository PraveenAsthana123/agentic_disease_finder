'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function SecureMessagingDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [sort, setSort] = useState({col:'messages', dir:'desc'});

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/secure-messaging/overview`).then(r=>r.json()),
      fetch(`${API}/api/secure-messaging/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/secure-messaging/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading secure messaging data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'categories', label:'Categories'},
    {id:'priority', label:'Priority'},
    {id:'patients', label:'Per Patient'},
    {id:'definitions', label:'Definitions'},
  ];

  const priorityColor = p => ({urgent:'danger', high:'warning', normal:'primary', low:'secondary'}[p]||'info');
  const catColor = i => ['primary','success','warning','info','secondary','danger','dark','primary'][i%8];

  const sortedPatients = bd ? [...bd.per_patient].sort((a,b)=>{
    const av = a[sort.col]??0, bv = b[sort.col]??0;
    return sort.dir==='asc' ? av-bv : bv-av;
  }) : [];

  const toggleSort = col => setSort(s => ({col, dir: s.col===col && s.dir==='desc' ? 'asc' : 'desc'}));
  const sortIcon = col => sort.col===col ? (sort.dir==='desc'?'▼':'▲') : '⇅';

  const kpis = ov.kpis || ov;
  const total = kpis.total_messages || 0;
  const unreadPct = kpis.unread_pct ?? 0;

  return (<div className="p-3">
    <h3>&#x1f4ac; Secure Messaging Dashboard</h3>
    <p className="text-muted">
      Patient–provider secure messaging &mdash; {total} messages &middot; {kpis.total_patients} patients
      &middot; {kpis.inbound_count} inbound / {kpis.outbound_count} outbound
      &middot; avg response {kpis.avg_response_hours}h
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          ['Total Messages', total, 'primary'],
          ['Patients', kpis.total_patients, 'info'],
          ['Inbound', kpis.inbound_count, 'success'],
          ['Outbound', kpis.outbound_count, 'secondary'],
          ['Unread', kpis.unread_count, 'danger'],
          ['Unread %', unreadPct+'%', 'warning'],
          ['Avg Response', kpis.avg_response_hours+'h', 'info'],
          ['Median Response', kpis.median_response_hours+'h', 'primary'],
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
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Direction Split</h6>
            {(ov.direction_split||[]).map(d=>{
              const pct = ((d.count/total)*100).toFixed(1);
              return <div key={d.direction} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'80px'}}>{d.direction}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'22px'}}>
                    <div className={`progress-bar bg-${d.direction==='Inbound'?'success':'primary'}`}
                         style={{width:`${pct}%`}}>
                      {d.count} ({pct}%)
                    </div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Read vs Unread</h6>
            {[
              {label:'Read', count: total - (kpis.unread_count||0), color:'success'},
              {label:'Unread', count: kpis.unread_count||0, color:'danger'},
            ].map(item=>{
              const pct = ((item.count/total)*100).toFixed(1);
              return <div key={item.label} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'70px'}}>{item.label}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'22px'}}>
                    <div className={`progress-bar bg-${item.color}`} style={{width:`${pct}%`}}>
                      {item.count} ({pct}%)
                    </div>
                  </div>
                </div>
              </div>;
            })}
            <div className="alert alert-sm alert-warning mt-2 py-1 px-2 small mb-0">
              {kpis.unread_count} unread messages require follow-up
            </div>
          </div></div>
        </div>
      </div>

      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Category Distribution (Top 8)</h6>
            {(ov.category_distribution||[]).slice(0,8).map((c,i)=>
              <div key={c.category} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'150px'}}>{c.category.replace(/-/g,' ')}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className={`progress-bar bg-${catColor(i)}`} style={{width:`${c.pct}%`}}>
                      {c.count} ({c.pct}%)
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div></div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Priority Distribution</h6>
            {(ov.priority_breakdown||[]).map(p=>
              <div key={p.priority} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'70px', textTransform:'capitalize'}}>{p.priority}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${priorityColor(p.priority)}`} style={{width:`${p.pct}%`}}>
                      {p.count} ({p.pct}%)
                    </div>
                  </div>
                </div>
              </div>
            )}
            {(ov.priority_breakdown||[]).some(p=>p.priority==='urgent') &&
              <div className="alert alert-danger py-1 px-2 small mt-2 mb-0">
                &#x26a0;&#xfe0f; {(ov.priority_breakdown||[]).find(p=>p.priority==='urgent')?.count||0} urgent messages need immediate attention
              </div>
            }
          </div></div>
        </div>
      </div>

      {ov.monthly_volume && ov.monthly_volume.length > 0 &&
        <div className="card shadow-sm mb-3"><div className="card-body">
          <h6>Monthly Message Volume</h6>
          <table className="table table-sm table-bordered">
            <thead><tr><th>Month</th><th>Messages</th><th>Trend</th></tr></thead>
            <tbody>
              {ov.monthly_volume.map((m,i)=>{
                const prev = i>0 ? ov.monthly_volume[i-1].cnt : m.cnt;
                const diff = m.cnt - prev;
                return <tr key={m.month}>
                  <td>{m.month}</td>
                  <td className="fw-bold">{m.cnt}</td>
                  <td>{i===0?'—':diff>0?<span className="text-success">+{diff}</span>:diff<0?<span className="text-danger">{diff}</span>:<span className="text-muted">0</span>}</td>
                </tr>;
              })}
            </tbody>
          </table>
        </div></div>
      }
    </div>}

    {tab==='categories' && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>All Categories — Detailed Breakdown</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered table-striped">
            <thead><tr><th>Category</th><th>Count</th><th>Share %</th><th>Bar</th></tr></thead>
            <tbody>
              {(ov.category_distribution||[]).map((c,i)=>
                <tr key={c.category}>
                  <td className="fw-bold" style={{textTransform:'capitalize'}}>{c.category.replace(/-/g,' ')}</td>
                  <td>{c.count}</td>
                  <td>{c.pct}%</td>
                  <td style={{minWidth:'200px'}}>
                    <div className="progress" style={{height:'16px'}}>
                      <div className={`progress-bar bg-${catColor(i)}`} style={{width:`${c.pct}%`}}/>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='priority' && <div>
      <div className="row mb-3">
        {(ov.priority_breakdown||[]).map(p=>(
          <div key={p.priority} className="col-6 col-md-3 mb-2">
            <div className={`card shadow-sm border-${priorityColor(p.priority)} h-100`}>
              <div className="card-body text-center py-2">
                <div className={`h4 mb-0 text-${priorityColor(p.priority)}`}>{p.count}</div>
                <div className="text-muted small" style={{textTransform:'capitalize'}}>{p.priority}</div>
                <div className="text-muted small">{p.pct}% of total</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card shadow-sm"><div className="card-body">
        <h6>Priority Level Definitions</h6>
        {defs && Object.entries(defs.priority_levels||{}).map(([level, desc])=>
          <div key={level} className="mb-2">
            <span className={`badge bg-${priorityColor(level)} me-2`} style={{textTransform:'capitalize'}}>{level}</span>
            <span className="small">{desc}</span>
          </div>
        )}
      </div></div>
    </div>}

    {tab==='patients' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Per-Patient Message Summary ({sortedPatients.length} patients)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-bordered">
            <thead><tr>
              <th>Patient</th>
              {[
                ['messages','Messages'],
                ['inbound','Inbound'],
                ['outbound','Outbound'],
                ['unread','Unread'],
                ['avg_response_hours','Avg Response (h)'],
              ].map(([col,label])=>
                <th key={col} style={{cursor:'pointer'}} onClick={()=>toggleSort(col)}>
                  {label} {sortIcon(col)}
                </th>
              )}
            </tr></thead>
            <tbody>
              {sortedPatients.map(p=>
                <tr key={p.patient_id}>
                  <td className="fw-bold font-monospace">{p.patient_id}</td>
                  <td>{p.messages}</td>
                  <td><span className="badge bg-success">{p.inbound}</span></td>
                  <td><span className="badge bg-primary">{p.outbound}</span></td>
                  <td>{p.unread>0
                    ? <span className="badge bg-danger">{p.unread}</span>
                    : <span className="badge bg-secondary">0</span>}
                  </td>
                  <td className={p.avg_response_hours > 24 ? 'text-danger fw-bold' : ''}>
                    {p.avg_response_hours}h
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="text-muted small mt-2">
          Click column headers to sort. Response time &gt;24h highlighted in red.
        </div>
      </div></div>
    </div>}

    {tab==='definitions' && defs && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Response Time</h6>
        <p className="small mb-0">{defs.response_time_note}</p>
      </div></div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Unread Messages</h6>
        <p className="small mb-0">{defs.unread_note}</p>
      </div></div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Priority Levels</h6>
        <table className="table table-sm table-bordered">
          <thead><tr><th>Level</th><th>Description</th></tr></thead>
          <tbody>
            {Object.entries(defs.priority_levels||{}).map(([level,desc])=>
              <tr key={level}>
                <td><span className={`badge bg-${priorityColor(level)}`} style={{textTransform:'capitalize'}}>{level}</span></td>
                <td className="small">{desc}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>

      <div className="card shadow-sm"><div className="card-body">
        <h6>Message Categories</h6>
        <table className="table table-sm table-bordered">
          <thead><tr><th>Category</th><th>Description</th></tr></thead>
          <tbody>
            {Object.entries(defs.categories||{}).map(([cat,desc])=>
              <tr key={cat}>
                <td className="fw-bold small" style={{textTransform:'capitalize'}}>{cat.replace(/-/g,' ')}</td>
                <td className="small">{desc}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>
    </div>}
  </div>);
}
