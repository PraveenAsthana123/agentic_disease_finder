'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TYPE_COLORS = {
  physio:'#3b82f6', cognitive:'#8b5cf6', sleep:'#6366f1', safety:'#ef4444',
  mindfulness:'#22c55e', yoga:'#a3e635', breathing:'#06b6d4', progressive_relaxation:'#f59e0b',
  aerobic:'#3b82f6', strength:'#8b5cf6', flexibility:'#22c55e', balance:'#f59e0b',
  coordination:'#6366f1', aquatic:'#06b6d4',
  active:'#22c55e', completed:'#6b7280', paused:'#f59e0b',
};

function Badge({label, color}){
  const bg = color || TYPE_COLORS[label] || '#6b7280';
  return <span style={{display:'inline-block',padding:'2px 8px',borderRadius:'4px',fontSize:'0.75rem',fontWeight:600,color:'#fff',backgroundColor:bg,marginRight:'4px'}}>{label}</span>;
}

function MiniBar({value, max=100, color='#3b82f6'}){
  const pct = Math.min(100, Math.max(0, (value/max)*100));
  return <div className="progress" style={{height:'8px',minWidth:'80px'}}>
    <div className="progress-bar" style={{width:`${pct}%`,backgroundColor:color}}></div>
  </div>;
}

function StatCard({label, value, color='#3b82f6'}){
  return <div className="col-6 col-md mb-2">
    <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
      <div className="h5 mb-0" style={{color}}>{value}</div>
      <div className="text-muted small">{label}</div>
    </div></div>
  </div>;
}

export default function TherapyDashboard(){
  const [data, setData] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(()=>{
    fetch(`${API}/api/therapy`)
      .then(r=>r.json())
      .then(d=>setData(d))
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!data) return <div className="text-muted p-3">Loading Therapy data...</div>;

  const s = data.summary||{};
  const rb = s.rehab_type_breakdown||{};
  const es = s.exercise_safety||{};
  const TABS = [
    {id:'overview',label:'Overview'},
    {id:'rehab',label:'Rehab Programs'},
    {id:'exercise',label:'Exercise Plans'},
    {id:'meditation',label:'Meditation'},
    {id:'physio',label:'Physio Protocols'},
  ];

  return (<div className="p-3">
    <h3>Therapy & Rehabilitation Dashboard</h3>
    <p className="text-muted">
      {s.total_patients_with_therapy} patients | {s.total_rehab_programs} rehab programs | {s.total_meditation_programs} meditation programs | {s.total_physio_protocols} physio protocols
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* Overview */}
    {tab==='overview' && <div>
      <div className="row mb-3">
        <StatCard label="Patients in Therapy" value={s.total_patients_with_therapy} color="#3b82f6"/>
        <StatCard label="Rehab Programs" value={s.total_rehab_programs} color="#8b5cf6"/>
        <StatCard label="Meditation Programs" value={s.total_meditation_programs} color="#22c55e"/>
        <StatCard label="Physio Protocols" value={s.total_physio_protocols} color="#f59e0b"/>
        <StatCard label="Safe Prescriptions" value={es.total_safe_prescriptions} color="#06b6d4"/>
        <StatCard label="Contraindicated" value={es.total_contraindicated} color="#ef4444"/>
      </div>

      <h5>Rehab Type Breakdown</h5>
      <div className="row mb-3">
        {Object.entries(rb).map(([type, count])=>
          <div key={type} className="col-6 col-md-3 mb-2">
            <div className="card shadow-sm"><div className="card-body py-2 text-center">
              <Badge label={type}/>
              <div className="h5 mt-1 mb-0">{count}</div>
            </div></div>
          </div>
        )}
      </div>
    </div>}

    {/* Rehab Programs */}
    {tab==='rehab' && <div>
      <p className="text-muted">{data.rehab_programs?.total_patients} patients, {data.rehab_programs?.total_programs} programs</p>
      <div className="table-responsive"><table className="table table-sm table-hover">
        <thead><tr>
          <th>Patient</th><th>Program</th><th>Type</th><th>Frequency</th><th>Duration</th><th>Assigned By</th><th>Status</th><th>Progress</th>
        </tr></thead>
        <tbody>
          {(data.rehab_programs?.patients||[]).flatMap(pt=>
            pt.programs.map((p,i)=>
              <tr key={`${pt.patient_id}-${i}`}>
                {i===0 ? <td rowSpan={pt.programs.length} className="fw-bold align-middle">{pt.patient_id}</td> : null}
                <td>
                  <span className="fw-semibold">{p.program_name}</span>
                  <br/><small className="text-muted">{p.rationale}</small>
                </td>
                <td><Badge label={p.type}/></td>
                <td>{p.frequency}</td>
                <td>{p.duration_weeks}w</td>
                <td className="small">{p.assigned_by}</td>
                <td><Badge label={p.status}/></td>
                <td style={{minWidth:'100px'}}>
                  <MiniBar value={p.progress_pct} color={TYPE_COLORS[p.type]||'#3b82f6'}/>
                  <small>{p.progress_pct}%</small>
                </td>
              </tr>
            )
          )}
        </tbody>
      </table></div>
    </div>}

    {/* Exercise Plans */}
    {tab==='exercise' && <div>
      <p className="text-muted">{data.exercise_plans?.total_patients} patients with exercise plans</p>
      {(data.exercise_plans?.patients||[]).map(pt=>(
        <div key={pt.patient_id} className="card mb-2 shadow-sm">
          <div className="card-header py-2 d-flex justify-content-between align-items-center" style={{cursor:'pointer'}}
            onClick={()=>setExpandedPt(expandedPt===pt.patient_id?null:pt.patient_id)}>
            <span className="fw-bold">{pt.patient_id}</span>
            <span className="text-muted small">
              {pt.exercises?.length||0} exercises | Seizures: {pt.seizure_events||0}
              {pt.uncontrolled_seizures && <Badge label="Uncontrolled" color="#ef4444"/>}
            </span>
          </div>
          {expandedPt===pt.patient_id && <div className="card-body py-2">
            <div className="table-responsive"><table className="table table-sm mb-0">
              <thead><tr><th>Exercise</th><th>Category</th><th>Duration</th><th>Frequency</th><th>Safe?</th><th>Precautions</th></tr></thead>
              <tbody>
                {(pt.exercises||[]).map((ex,i)=>
                  <tr key={i} className={ex.contraindicated?'table-danger':''}>
                    <td className="fw-semibold">{ex.exercise_name}<br/><small className="text-muted">{ex.description}</small></td>
                    <td><Badge label={ex.category}/></td>
                    <td>{ex.duration_min}min</td>
                    <td>{ex.frequency}</td>
                    <td>{ex.contraindicated ? <Badge label="Contraindicated" color="#ef4444"/> : <Badge label="Safe" color="#22c55e"/>}</td>
                    <td className="small">{(ex.precautions||[]).join('; ')}{ex.reason && <><br/><strong>Reason:</strong> {ex.reason}</>}</td>
                  </tr>
                )}
              </tbody>
            </table></div>
          </div>}
        </div>
      ))}
    </div>}

    {/* Meditation Programs */}
    {tab==='meditation' && <div>
      <p className="text-muted">{data.meditation_programs?.total_patients} patients, {data.meditation_programs?.total_programs} programs</p>
      {(data.meditation_programs?.patients||[]).slice(0,20).map(pt=>(
        <div key={pt.patient_id} className="card mb-2 shadow-sm">
          <div className="card-header py-2 d-flex justify-content-between align-items-center" style={{cursor:'pointer'}}
            onClick={()=>setExpandedPt(expandedPt===`med-${pt.patient_id}`?null:`med-${pt.patient_id}`)}>
            <span className="fw-bold">{pt.patient_id}</span>
            <span className="text-muted small">{pt.total_programs} programs</span>
          </div>
          {expandedPt===`med-${pt.patient_id}` && <div className="card-body py-2">
            {(pt.programs||[]).map((p,i)=>
              <div key={i} className="border rounded p-2 mb-2">
                <div className="d-flex justify-content-between align-items-start">
                  <div>
                    <span className="fw-bold">{p.program_name}</span>
                    <br/><Badge label={p.type}/> <small className="text-muted">{p.frequency} | {p.session_duration_min}min sessions</small>
                  </div>
                </div>
                <div className="mt-1 small"><strong>Target:</strong> {p.target_condition}</div>
                <div className="small"><strong>Evidence:</strong> {p.evidence_level}</div>
                {p.components && <div className="mt-1 small"><strong>Components:</strong> {p.components.join(', ')}</div>}
              </div>
            )}
          </div>}
        </div>
      ))}
    </div>}

    {/* Physio Protocols */}
    {tab==='physio' && <div>
      <p className="text-muted">{data.physio_protocols?.total_patients} patients, {data.physio_protocols?.total_protocols} protocols</p>
      <div className="table-responsive"><table className="table table-sm table-hover">
        <thead><tr>
          <th>Patient</th><th>Protocol</th><th>Target Area</th><th>Sessions/wk</th><th>Duration</th><th>Therapist</th><th>Exercises</th>
        </tr></thead>
        <tbody>
          {(data.physio_protocols?.patients||[]).flatMap(pt=>
            (pt.protocols||[]).map((p,i)=>
              <tr key={`${pt.patient_id}-${i}`}>
                {i===0 ? <td rowSpan={pt.protocols.length} className="fw-bold align-middle">{pt.patient_id}</td> : null}
                <td className="fw-semibold">{p.protocol_name}</td>
                <td><Badge label={p.target_area}/></td>
                <td>{p.sessions_per_week}x</td>
                <td>{p.duration_weeks}w</td>
                <td className="small">{p.therapist_type}</td>
                <td className="small">{(p.exercises||[]).join(', ')}</td>
              </tr>
            )
          )}
        </tbody>
      </table></div>
    </div>}
  </div>);
}
