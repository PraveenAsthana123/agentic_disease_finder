'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8000';

function KPI({label, value, color}){
  return (
    <div className="col-6 col-md-3 mb-2">
      <div className="card shadow-sm h-100"><div className="card-body text-center">
        <div className={`h4 mb-1 text-${color||'primary'}`}>{value}</div>
        <div className="text-muted small">{label}</div>
      </div></div>
    </div>
  );
}

function Bar({items}){
  if(!items||!items.length) return null;
  const mx = Math.max(...items.map(i=>i.value));
  return (
    <div>{items.map((it,i)=>(
      <div key={i} className="d-flex align-items-center mb-1">
        <div className="text-end small me-2" style={{width:140,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{it.label}</div>
        <div className="flex-grow-1">
          <div className="progress" style={{height:18}}>
            <div className={`progress-bar bg-${it.color||'primary'}`}
                 style={{width:`${mx?((it.value/mx)*100):0}%`}}>
              <span className="small">{it.value}</span>
            </div>
          </div>
        </div>
      </div>
    ))}</div>
  );
}

const ROUTE_COLORS = {auto_approve:'success', review:'warning', escalate:'danger', unknown:'secondary'};

export default function DecisionAIDashboard(){
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/decision-ai/overview`).then(r=>r.json()),
      fetch(`${API}/api/decision-ai/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/decision-ai/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOverview(o);setBreakdown(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load Decision AI data: {err}</div>;
  if(!overview) return <div className="text-muted p-3">Loading Decision AI dashboard...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'routing', label:'Decision Routing'},
    {id:'hitl', label:'HITL Reviews'},
    {id:'audit', label:'Audit Trail'},
    {id:'definitions', label:'Definitions'},
  ];

  const kpis = overview.kpis || {};

  return (
    <div className="container-fluid py-3">
      <h3>Decision AI Dashboard</h3>
      <p className="text-muted small mb-3">Confidence-based routing, HITL overrides, and audit trail from real clinical.db</p>

      <div className="row mb-3">
        <KPI label="Total Analyses" value={kpis.total_analyses} color="primary"/>
        <KPI label="Avg Confidence" value={kpis.avg_confidence?.toFixed(3)} color="info"/>
        <KPI label="Auto-Approve" value={kpis.auto_approve_count} color="success"/>
        <KPI label="Review Required" value={kpis.review_count} color="warning"/>
        <KPI label="Escalated" value={kpis.escalate_count} color="danger"/>
        <KPI label="HITL Overrides" value={kpis.hitl_overrides} color="danger"/>
        <KPI label="HITL Confirms" value={kpis.hitl_confirms} color="success"/>
        <KPI label="Audit Events" value={kpis.audit_events} color="secondary"/>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t=>(
          <li className="nav-item" key={t.id}>
            <button className={`nav-link${tab===t.id?' active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {tab==='overview' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm"><div className="card-body">
              <h6>Route Distribution</h6>
              <Bar items={(overview.route_distribution||[]).map(r=>({
                label: r.route.replace('_',' '), value: r.count,
                color: ROUTE_COLORS[r.route]||'primary'
              }))} />
            </div></div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm"><div className="card-body">
              <h6>Confidence Histogram</h6>
              <Bar items={(overview.confidence_histogram||[]).map(h=>({
                label: h.bucket, value: h.count, color: 'info'
              }))} />
            </div></div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm"><div className="card-body">
              <h6>Disease Summary</h6>
              <table className="table table-sm table-striped mb-0">
                <thead><tr><th>Disease</th><th>Count</th><th>Avg Conf</th><th>Routes</th></tr></thead>
                <tbody>
                  {(overview.disease_summary||[]).map((d,i)=>(
                    <tr key={i}>
                      <td>{d.disease}</td><td>{d.count}</td>
                      <td>{d.avg_confidence?.toFixed(3)}</td>
                      <td>{Object.entries(d.routes||{}).map(([k,v])=>`${k}:${v}`).join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div></div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm"><div className="card-body">
              <h6>Routing Thresholds</h6>
              <table className="table table-sm mb-0">
                <tbody>
                  <tr><td>Auto-Approve</td><td><span className="badge bg-success">&ge; {overview.thresholds?.auto_approve}</span></td></tr>
                  <tr><td>Review</td><td><span className="badge bg-warning text-dark">&ge; {overview.thresholds?.review}</span></td></tr>
                  <tr><td>Escalate</td><td><span className="badge bg-danger">&lt; {overview.thresholds?.review}</span></td></tr>
                </tbody>
              </table>
            </div></div>
          </div>
        </div>
      )}

      {tab==='routing' && breakdown && (
        <div>
          <h5>Per-Analysis Decision Routing</h5>
          <div className="table-responsive">
            <table className="table table-sm table-striped">
              <thead><tr><th>ID</th><th>Patient</th><th>Disease</th><th>Prediction</th><th>Confidence</th><th>Route</th><th>Signal Quality</th></tr></thead>
              <tbody>
                {(breakdown.per_analysis||[]).map(a=>(
                  <tr key={a.id}>
                    <td>{a.id}</td><td>{a.patient_id}</td><td>{a.disease}</td>
                    <td>{a.predicted_label}</td>
                    <td>{a.confidence?.toFixed(3)}</td>
                    <td><span className={`badge bg-${ROUTE_COLORS[a.route]||'secondary'}`}>{a.route?.replace('_',' ')}</span></td>
                    <td>{a.signal_quality}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h5 className="mt-4">Confidence Calibration</h5>
          <table className="table table-sm table-striped">
            <thead><tr><th>Bucket</th><th>Total</th><th>Reviewed</th><th>Overridden</th><th>Agreement Rate</th></tr></thead>
            <tbody>
              {(breakdown.calibration||[]).map((c,i)=>(
                <tr key={i}>
                  <td>{c.bucket}</td><td>{c.total}</td><td>{c.reviewed}</td><td>{c.overridden}</td>
                  <td>{c.agreement_rate !== null ? (c.agreement_rate * 100).toFixed(1) + '%' : 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {tab==='hitl' && breakdown && (
        <div>
          <h5>HITL Reviews</h5>
          {(breakdown.hitl_reviews||[]).length === 0 ? (
            <p className="text-muted">No HITL reviews recorded yet.</p>
          ) : (
            <table className="table table-sm table-striped">
              <thead><tr><th>ID</th><th>Patient</th><th>Analysis</th><th>Decision</th><th>AI Prediction</th><th>Human Decision</th><th>Reason</th><th>Date</th></tr></thead>
              <tbody>
                {(breakdown.hitl_reviews||[]).map(h=>(
                  <tr key={h.id}>
                    <td>{h.id}</td><td>{h.patient_id}</td><td>{h.analysis_id}</td>
                    <td><span className={`badge bg-${h.decision==='override'?'danger':'success'}`}>{h.decision}</span></td>
                    <td>{h.ai_prediction}</td><td>{h.human_decision}</td>
                    <td>{h.reason_code}</td><td className="small">{h.created_at?.slice(0,10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <h5 className="mt-4">Clinical Decisions</h5>
          {(overview.clinical_decisions||[]).length === 0 ? (
            <p className="text-muted">No clinical decisions recorded yet.</p>
          ) : (
            <table className="table table-sm table-striped">
              <thead><tr><th>Patient</th><th>AI Pred</th><th>Confidence</th><th>Agreement</th><th>Final</th><th>Reviewer</th><th>Note</th></tr></thead>
              <tbody>
                {(overview.clinical_decisions||[]).map(d=>(
                  <tr key={d.id}>
                    <td>{d.patient_id}</td><td>{d.ai_prediction}</td>
                    <td>{d.ai_confidence?.toFixed(2)}</td>
                    <td><span className={`badge bg-${d.neurologist_agreement==='Yes'?'success':'danger'}`}>{d.neurologist_agreement}</span></td>
                    <td>{d.final_decision}</td><td>{d.reviewer}</td>
                    <td className="small">{d.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <h5 className="mt-4">Per-Patient Summary</h5>
          <div className="table-responsive">
            <table className="table table-sm table-striped">
              <thead><tr><th>Patient</th><th>Analyses</th><th>Avg Conf</th><th>Routes</th><th>HITL</th><th>Overrides</th><th>Audit Events</th></tr></thead>
              <tbody>
                {(breakdown.patient_summaries||[]).filter(p=>p.analysis_count>0).map(p=>(
                  <tr key={p.patient_id}>
                    <td>{p.patient_id}</td><td>{p.analysis_count}</td>
                    <td>{p.avg_confidence?.toFixed(3)}</td>
                    <td>{Object.entries(p.routes||{}).map(([k,v])=>`${k}:${v}`).join(', ')}</td>
                    <td>{p.hitl_count}</td><td>{p.overrides}</td><td>{p.audit_events}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab==='audit' && (
        <div>
          <h5>Audit Trail Summary</h5>
          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm"><div className="card-body">
                <h6>Top Components</h6>
                <Bar items={(overview.audit_summary?.top_components||[]).map(c=>({
                  label: c.name, value: c.count, color: 'info'
                }))} />
              </div></div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm"><div className="card-body">
                <h6>Top Actions</h6>
                <Bar items={(overview.audit_summary?.top_actions||[]).map(a=>({
                  label: a.name, value: a.count, color: 'primary'
                }))} />
              </div></div>
            </div>
          </div>
          <div className="row">
            <div className="col-md-6">
              <div className="card shadow-sm"><div className="card-body">
                <h6>Top Actors</h6>
                <Bar items={(overview.audit_summary?.top_actors||[]).map(a=>({
                  label: a.name, value: a.count, color: 'success'
                }))} />
              </div></div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm"><div className="card-body">
                <h6>Event Timeline</h6>
                {(breakdown?.audit_timeline||[]).length > 0 ? (
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Month</th><th>Events</th></tr></thead>
                    <tbody>
                      {(breakdown.audit_timeline||[]).map((t,i)=>{
                        const total = Object.entries(t).filter(([k])=>k!=='month').reduce((s,[,v])=>s+v,0);
                        return <tr key={i}><td>{t.month}</td><td>{total}</td></tr>;
                      })}
                    </tbody>
                  </table>
                ) : <p className="text-muted small">No timeline data</p>}
              </div></div>
            </div>
          </div>
        </div>
      )}

      {tab==='definitions' && defs && (
        <div>
          <h5>{defs.title}</h5>
          {(defs.sections||[]).map((s,si)=>(
            <div key={si} className="mb-4">
              <h6 className="text-primary">{s.heading}</h6>
              <table className="table table-sm table-bordered">
                <tbody>
                  {(s.items||[]).map((it,ii)=>(
                    <tr key={ii}><td className="fw-bold" style={{width:'25%'}}>{it.term}</td><td>{it.definition}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
