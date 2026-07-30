'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const pct = (n, total) => total ? ((n / total) * 100).toFixed(1) : '0.0';
const statusColor = s => ({completed:'success',scheduled:'primary',rescheduled:'warning',cancelled:'danger','no-show':'dark'}[s]||'secondary');

export default function PatientAppointmentsDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/patient-appointments/overview`).then(r=>r.json()),
      fetch(`${API}/api/patient-appointments/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/patient-appointments/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading patient appointments data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'upcoming', label:'Upcoming'},
    {id:'completed', label:'Recent Completed'},
    {id:'noshows', label:'No-Shows'},
    {id:'patients', label:'Per Patient'},
    {id:'providers', label:'Providers'},
    {id:'definitions', label:'Definitions'},
  ];

  const total = ov.total_appointments;

  return (<div className="p-3">
    <h3>Patient Appointments Dashboard</h3>
    <p className="text-muted">
      Scheduling analytics &mdash; {total} appointments, {ov.total_patients} patients,
      {' '}{ov.completion_rate}% completion, {ov.no_show_rate}% no-show, {ov.reminder_sent_rate}% reminders sent
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          ['Total Appts', total, 'primary'],
          ['Patients', ov.total_patients, 'info'],
          ['Completion', ov.completion_rate+'%', 'success'],
          ['No-Show', ov.no_show_rate+'%', 'danger'],
          ['Cancelled', ov.cancellation_rate+'%', 'warning'],
          ['Reminders', ov.reminder_sent_rate+'%', 'primary'],
        ].map(([label,val,c])=>
          <div key={label} className="col-6 col-md-2 mb-2">
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
            <h6>Status Distribution</h6>
            {Object.entries(ov.status_distribution||{}).map(([s,count])=>{
              const p = pct(count, total);
              return <div key={s} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'100px',textTransform:'capitalize'}}>{s}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${statusColor(s)}`}
                      style={{width:`${p}%`}}>{count} ({p}%)</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Appointment Types</h6>
            {Object.entries(ov.type_distribution||{}).map(([t,count])=>{
              const p = pct(count, total);
              return <div key={t} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'160px'}}>{t}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className="progress-bar bg-info" style={{width:`${p}%`}}>{count}</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
      </div>

      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>By Provider</h6>
            <table className="table table-sm table-striped mb-0">
              <thead><tr><th>Provider</th><th>Count</th><th>Share</th></tr></thead>
              <tbody>
                {Object.entries(ov.provider_distribution||{}).map(([prov,count])=>
                  <tr key={prov}>
                    <td className="fw-bold">{prov}</td>
                    <td>{count}</td>
                    <td><span className="badge bg-primary">{pct(count, total)}%</span></td>
                  </tr>
                )}
              </tbody>
            </table>
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>By Location</h6>
            {Object.entries(ov.location_distribution||{}).map(([loc,count])=>{
              const p = pct(count, total);
              return <div key={loc} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'150px'}}>{loc}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className="progress-bar bg-success" style={{width:`${p}%`}}>{count}</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
      </div>

      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Duration Distribution</h6>
            {Object.entries(ov.duration_distribution||{}).map(([dur,count])=>{
              const p = pct(count, total);
              return <div key={dur} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'60px'}}>{dur}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className="progress-bar bg-warning" style={{width:`${p}%`}}>{count}</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-6">
          {(ov.monthly_trend||[]).length>0 && <div className="card shadow-sm"><div className="card-body">
            <h6>Monthly Trend</h6>
            <div className="d-flex align-items-end" style={{height:'120px',gap:'4px'}}>
              {ov.monthly_trend.map(m=>{
                const maxC = Math.max(...ov.monthly_trend.map(x=>x.count));
                const h = maxC ? (m.count/maxC)*100 : 0;
                return <div key={m.month} className="d-flex flex-column align-items-center flex-grow-1">
                  <small className="text-muted mb-1">{m.count}</small>
                  <div className="bg-primary rounded" style={{width:'100%',maxWidth:'40px',height:`${h}%`,minHeight:'4px'}}/>
                  <small className="text-muted mt-1">{m.month.slice(5)}</small>
                </div>;
              })}
            </div>
          </div></div>}
        </div>
      </div>
    </div>}

    {tab==='upcoming' && bd && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Upcoming Appointments ({(bd.upcoming||[]).length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr>
              <th>Patient</th><th>Type</th><th>Provider</th><th>Date</th><th>Time</th><th>Duration</th><th>Location</th><th>Reminder</th>
            </tr></thead>
            <tbody>
              {(bd.upcoming||[]).map((a,i)=>
                <tr key={i}>
                  <td className="fw-bold">{a.patient_id}</td>
                  <td>{a.appointment_type}</td>
                  <td>{a.provider_name}</td>
                  <td>{a.appointment_date}</td>
                  <td>{a.appointment_time}</td>
                  <td>{a.duration_minutes}m</td>
                  <td>{a.location}</td>
                  <td><span className={`badge bg-${a.reminder_sent?'success':'secondary'}`}>{a.reminder_sent?'Yes':'No'}</span></td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='completed' && bd && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Recently Completed ({(bd.recent_completed||[]).length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr>
              <th>Patient</th><th>Type</th><th>Provider</th><th>Date</th><th>Time</th><th>Duration</th><th>Location</th><th>Notes</th>
            </tr></thead>
            <tbody>
              {(bd.recent_completed||[]).map((a,i)=>
                <tr key={i}>
                  <td className="fw-bold">{a.patient_id}</td>
                  <td>{a.appointment_type}</td>
                  <td>{a.provider_name}</td>
                  <td>{a.appointment_date}</td>
                  <td>{a.appointment_time}</td>
                  <td>{a.duration_minutes}m</td>
                  <td>{a.location}</td>
                  <td className="small text-muted">{a.notes}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='noshows' && bd && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>No-Shows ({(bd.no_shows||[]).length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr>
              <th>Patient</th><th>Type</th><th>Provider</th><th>Date</th><th>Time</th><th>Duration</th><th>Location</th><th>Reminder Sent</th>
            </tr></thead>
            <tbody>
              {(bd.no_shows||[]).map((a,i)=>
                <tr key={i}>
                  <td className="fw-bold">{a.patient_id}</td>
                  <td>{a.appointment_type}</td>
                  <td>{a.provider_name}</td>
                  <td>{a.appointment_date}</td>
                  <td>{a.appointment_time}</td>
                  <td>{a.duration_minutes}m</td>
                  <td>{a.location}</td>
                  <td><span className={`badge bg-${a.reminder_sent?'success':'danger'}`}>{a.reminder_sent?'Yes':'No'}</span></td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='patients' && bd && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Per-Patient Summary ({(bd.per_patient||[]).length} patients)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr>
              <th>Patient</th><th>Total</th><th>Completed</th><th>Scheduled</th><th>Cancelled</th><th>No-Show</th><th>Rescheduled</th><th>Completion %</th><th>Top Type</th><th>Top Provider</th>
            </tr></thead>
            <tbody>
              {(bd.per_patient||[]).map(p=>
                <tr key={p.patient_id}>
                  <td className="fw-bold">{p.patient_id}</td>
                  <td>{p.total}</td>
                  <td>{p.completed}</td>
                  <td>{p.scheduled}</td>
                  <td>{p.cancelled}</td>
                  <td className={p.no_show>0?'text-danger':''}>{p.no_show}</td>
                  <td>{p.rescheduled}</td>
                  <td><span className={`badge bg-${p.completion_rate>=60?'success':p.completion_rate>=40?'warning':'danger'}`}>{p.completion_rate}%</span></td>
                  <td className="small">{p.top_type}</td>
                  <td className="small">{p.top_provider}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='providers' && bd && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Provider Statistics</h6>
        <table className="table table-sm table-striped mb-0">
          <thead><tr>
            <th>Provider</th><th>Total Appts</th><th>Completed</th><th>No-Show Rate</th><th>Avg Duration</th>
          </tr></thead>
          <tbody>
            {(bd.provider_stats||[]).map(p=>
              <tr key={p.provider}>
                <td className="fw-bold">{p.provider}</td>
                <td>{p.total}</td>
                <td>{p.completed}</td>
                <td><span className={`badge bg-${p.no_show_rate>10?'danger':p.no_show_rate>5?'warning':'success'}`}>{p.no_show_rate}%</span></td>
                <td>{p.avg_duration} min</td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>
    </div>}

    {tab==='definitions' && defs && <div>
      {defs.glossary && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Glossary</h6>
        <table className="table table-sm mb-0">
          <tbody>
            {(Array.isArray(defs.glossary)?defs.glossary:[]).map(g=>
              <tr key={g.term}><td className="fw-bold" style={{width:'200px'}}>{g.term}</td><td>{g.definition}</td></tr>
            )}
          </tbody>
        </table>
      </div></div>}
      {defs.appointment_types && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Appointment Types</h6>
        <table className="table table-sm mb-0">
          <tbody>
            {Object.entries(defs.appointment_types).map(([t,desc])=>
              <tr key={t}><td className="fw-bold" style={{width:'200px'}}>{t}</td><td>{desc}</td></tr>
            )}
          </tbody>
        </table>
      </div></div>}
      {defs.status_definitions && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Status Definitions</h6>
        <table className="table table-sm mb-0">
          <tbody>
            {Object.entries(defs.status_definitions).map(([s,desc])=>
              <tr key={s}><td className="fw-bold" style={{width:'200px',textTransform:'capitalize'}}>{s}</td><td>{desc}</td></tr>
            )}
          </tbody>
        </table>
      </div></div>}
      {defs.location_notes && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Location Notes</h6>
        <table className="table table-sm mb-0">
          <tbody>
            {Object.entries(defs.location_notes).map(([loc,desc])=>
              <tr key={loc}><td className="fw-bold" style={{width:'200px'}}>{loc}</td><td>{desc}</td></tr>
            )}
          </tbody>
        </table>
      </div></div>}
    </div>}
  </div>);
}
