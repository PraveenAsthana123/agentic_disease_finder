'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const pct = (n, total) => total ? ((n / total) * 100).toFixed(1) : '0.0';
const readinessColor = l => ({'fully_ready':'success','mostly_ready':'info','partially_ready':'warning','not_ready':'danger'}[l]||'secondary');
const burnoutColor = l => ({'Low':'success','Moderate':'warning','High':'danger','Critical':'danger'}[l]||'secondary');

export default function CaregiverReadinessDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/caregiver-readiness/overview`).then(r=>r.json()),
      fetch(`${API}/api/caregiver-readiness/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/caregiver-readiness/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading caregiver readiness data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'profiles', label:'Caregiver Profiles'},
    {id:'burnout', label:'Burnout Analysis'},
    {id:'training', label:'Training Coverage'},
    {id:'definitions', label:'Definitions'},
  ];

  const totalCaregivers = ov.kpis?.find(k=>k.label==='Total Caregivers')?.value || 0;

  return (<div className="p-3">
    <h3>Caregiver Readiness Dashboard</h3>
    <p className="text-muted">
      Epilepsy caregiver preparedness tracking &mdash; {totalCaregivers} caregivers,
      {' '}{ov.kpis?.find(k=>k.label==='Training Completion Rate')?.value} training completion,
      {' '}{ov.kpis?.find(k=>k.label==='Rescue Med Trained')?.value} rescue-med certified,
      {' '}{ov.kpis?.find(k=>k.label==='Safety Plans Active')?.value} safety plans active
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {(ov.kpis||[]).map(k=>{
          const c = k.color==='green'?'success':k.color==='red'?'danger':k.color==='yellow'?'warning':k.color==='blue'?'primary':'info';
          return <div key={k.label} className="col-6 col-md-2 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className={`h5 mb-0 text-${c}`}>{k.value}</div>
              <div className="text-muted small">{k.label}</div>
            </div></div>
          </div>;
        })}
      </div>

      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Readiness Distribution</h6>
            {(ov.readiness_distribution||[]).map(r=>{
              const p = pct(r.count, totalCaregivers);
              return <div key={r.level} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'130px',textTransform:'capitalize'}}>{r.level.replace(/_/g,' ')}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'22px'}}>
                    <div className={`progress-bar bg-${readinessColor(r.level)}`}
                      style={{width:`${p}%`}}>{r.count} ({p}%)</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Burnout Distribution</h6>
            {(ov.burnout_distribution||[]).map(b=>{
              const p = pct(b.count, totalCaregivers);
              const c = b.color==='green'?'success':b.color==='orange'?'warning':b.color==='red'?'danger':'info';
              return <div key={b.range} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'130px'}}>{b.range}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'22px'}}>
                    <div className={`progress-bar bg-${c}`}
                      style={{width:`${p}%`}}>{b.count} ({p}%)</div>
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
            <h6>Caregiver Role Distribution</h6>
            {(ov.role_distribution||[]).map(r=>{
              const p = pct(r.count, totalCaregivers);
              return <div key={r.role} className="d-flex align-items-center mb-2">
                <span className="me-2 small fw-bold" style={{minWidth:'100px',textTransform:'capitalize'}}>{r.role}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className="progress-bar bg-primary" style={{width:`${p}%`}}>{r.count} ({p}%)</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Training Topic Coverage</h6>
            {(ov.training_topic_coverage||[]).map(t=>{
              const p = pct(t.count, totalCaregivers);
              const c = parseFloat(p)>=60?'success':parseFloat(p)>=40?'warning':'danger';
              return <div key={t.topic} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'200px'}}>{t.topic}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className={`progress-bar bg-${c}`} style={{width:`${p}%`}}>{t.count} ({p}%)</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
      </div>
    </div>}

    {tab==='profiles' && bd && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>All Caregiver Profiles ({(bd.caregiver_profiles||[]).length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr>
              <th>Patient</th><th>Caregiver</th><th>Role</th><th>Availability</th><th>Exp (yr)</th>
              <th>Training</th><th>First Aid</th><th>Rescue Med</th><th>Safety Plan</th>
              <th>Confidence</th><th>Burnout</th><th>Readiness</th>
            </tr></thead>
            <tbody>
              {(bd.caregiver_profiles||[]).map((c,i)=>
                <tr key={i}>
                  <td className="fw-bold">{c.patient_id}</td>
                  <td>{c.caregiver_name}</td>
                  <td style={{textTransform:'capitalize'}}>{c.role}</td>
                  <td className="small">{c.availability}</td>
                  <td>{c.experience_years}</td>
                  <td><span className={`badge bg-${c.training_completed?'success':'secondary'}`}>{c.training_completed?'Yes':'No'}</span></td>
                  <td><span className={`badge bg-${c.first_aid?'success':'secondary'}`}>{c.first_aid?'Yes':'No'}</span></td>
                  <td><span className={`badge bg-${c.rescue_med?'success':'secondary'}`}>{c.rescue_med?'Yes':'No'}</span></td>
                  <td><span className={`badge bg-${c.safety_plan?'success':'secondary'}`}>{c.safety_plan?'Yes':'No'}</span></td>
                  <td>{c.confidence}/10</td>
                  <td><span className={`badge bg-${burnoutColor(c.burnout_level)}`}>{c.burnout_score} ({c.burnout_level})</span></td>
                  <td><span className={`badge bg-${readinessColor(c.readiness_level)}`} style={{textTransform:'capitalize'}}>{c.readiness_level?.replace(/_/g,' ')}</span></td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='burnout' && bd && <div>
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Burnout by Role</h6>
            {(() => {
              const roles = {};
              (bd.caregiver_profiles||[]).forEach(c => {
                if(!roles[c.role]) roles[c.role] = {total:0, sum:0, critical:0};
                roles[c.role].total++;
                roles[c.role].sum += c.burnout_score;
                if(c.burnout_level==='Critical') roles[c.role].critical++;
              });
              return Object.entries(roles).sort((a,b)=>(b[1].sum/b[1].total)-(a[1].sum/a[1].total)).map(([role,v])=>{
                const avg = (v.sum/v.total).toFixed(1);
                const c = avg>=70?'danger':avg>=40?'warning':'success';
                return <div key={role} className="d-flex align-items-center mb-2">
                  <span className="me-2 small fw-bold" style={{minWidth:'100px',textTransform:'capitalize'}}>{role}</span>
                  <div className="flex-grow-1 me-2">
                    <div className="progress" style={{height:'22px'}}>
                      <div className={`progress-bar bg-${c}`} style={{width:`${avg}%`}}>Avg {avg} ({v.total} caregivers{v.critical?`, ${v.critical} critical`:''})</div>
                    </div>
                  </div>
                </div>;
              });
            })()}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>High-Burnout Caregivers (score &ge; 70)</h6>
            <table className="table table-sm table-striped mb-0">
              <thead><tr><th>Caregiver</th><th>Role</th><th>Score</th><th>Patient</th><th>Stress</th><th>Sleep</th></tr></thead>
              <tbody>
                {(bd.caregiver_profiles||[]).filter(c=>c.burnout_score>=70).sort((a,b)=>b.burnout_score-a.burnout_score).map((c,i)=>
                  <tr key={i}>
                    <td className="fw-bold">{c.caregiver_name}</td>
                    <td style={{textTransform:'capitalize'}}>{c.role}</td>
                    <td><span className={`badge bg-${burnoutColor(c.burnout_level)}`}>{c.burnout_score}</span></td>
                    <td>{c.patient_id}</td>
                    <td>{c.stress}/10</td>
                    <td>{c.sleep_quality}/10</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div></div>
        </div>
      </div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Burnout Score Distribution</h6>
        <div className="d-flex align-items-end" style={{height:'160px',gap:'4px'}}>
          {(ov.burnout_distribution||[]).map(b=>{
            const maxC = Math.max(...ov.burnout_distribution.map(x=>x.count));
            const h = maxC ? (b.count/maxC)*100 : 0;
            const c = b.color==='green'?'success':b.color==='orange'?'warning':b.color==='red'?'danger':'info';
            return <div key={b.range} className="d-flex flex-column align-items-center flex-grow-1">
              <small className="text-muted mb-1">{b.count}</small>
              <div className={`bg-${c} rounded`} style={{width:'100%',maxWidth:'80px',height:`${h}%`,minHeight:'4px'}}/>
              <small className="text-muted mt-1 text-center" style={{fontSize:'0.7rem'}}>{b.range}</small>
            </div>;
          })}
        </div>
      </div></div>
    </div>}

    {tab==='training' && bd && <div>
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Certification Summary</h6>
            {(() => {
              const profiles = bd.caregiver_profiles||[];
              const total = profiles.length;
              const metrics = [
                {label:'Training Completed', count:profiles.filter(c=>c.training_completed).length},
                {label:'First Aid Certified', count:profiles.filter(c=>c.first_aid).length},
                {label:'Rescue Med Trained', count:profiles.filter(c=>c.rescue_med).length},
                {label:'Safety Plan Active', count:profiles.filter(c=>c.safety_plan).length},
                {label:'Action Plan Active', count:profiles.filter(c=>c.action_plan).length},
              ];
              return metrics.map(m=>{
                const p = pct(m.count, total);
                const c = parseFloat(p)>=70?'success':parseFloat(p)>=50?'warning':'danger';
                return <div key={m.label} className="d-flex align-items-center mb-2">
                  <span className="me-2 small" style={{minWidth:'160px'}}>{m.label}</span>
                  <div className="flex-grow-1 me-2">
                    <div className="progress" style={{height:'22px'}}>
                      <div className={`progress-bar bg-${c}`} style={{width:`${p}%`}}>{m.count}/{total} ({p}%)</div>
                    </div>
                  </div>
                </div>;
              });
            })()}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Untrained Caregivers</h6>
            <table className="table table-sm table-striped mb-0">
              <thead><tr><th>Caregiver</th><th>Patient</th><th>Missing</th></tr></thead>
              <tbody>
                {(bd.caregiver_profiles||[]).filter(c=>!c.training_completed||!c.first_aid||!c.rescue_med).map((c,i)=>{
                  const missing = [];
                  if(!c.training_completed) missing.push('Training');
                  if(!c.first_aid) missing.push('First Aid');
                  if(!c.rescue_med) missing.push('Rescue Med');
                  return <tr key={i}>
                    <td className="fw-bold">{c.caregiver_name}</td>
                    <td>{c.patient_id}</td>
                    <td>{missing.map(m=><span key={m} className="badge bg-danger me-1">{m}</span>)}</td>
                  </tr>;
                })}
              </tbody>
            </table>
          </div></div>
        </div>
      </div>
    </div>}

    {tab==='definitions' && defs && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Key Concepts</h6>
        <table className="table table-sm mb-0">
          <thead><tr><th>Concept</th><th>Description</th></tr></thead>
          <tbody>
            {(defs.concepts||[]).map(c=>
              <tr key={c.name}>
                <td className="fw-bold" style={{minWidth:'180px'}}>{c.name}</td>
                <td>{c.description}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>

      {defs.quality_metrics && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Quality Metrics</h6>
        <table className="table table-sm mb-0">
          <thead><tr><th>Metric</th><th>Description</th></tr></thead>
          <tbody>
            {(defs.quality_metrics||[]).map(m=>
              <tr key={m.name}>
                <td className="fw-bold" style={{minWidth:'180px'}}>{m.name}</td>
                <td>{m.description}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>}

      {defs.data_sources && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Data Sources</h6>
        <ul className="mb-0">{defs.data_sources.map((s,i)=><li key={i}>{s}</li>)}</ul>
      </div></div>}
    </div>}
  </div>);
}
