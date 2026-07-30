'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function ClinicalDecisionsDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/clinical-decisions/overview`).then(r=>r.json()),
      fetch(`${API}/api/clinical-decisions/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/clinical-decisions/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading clinical decisions data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'reviewers', label:'Reviewer Workload'},
    {id:'disagreements', label:'Disagreements'},
    {id:'predictions', label:'AI × Decision'},
    {id:'recent', label:'Recent Decisions'},
    {id:'definitions', label:'Definitions'},
  ];

  const decColor = d => ({Confirm:'success',Override:'danger',Escalate:'warning',Defer:'secondary'}[d]||'info');
  const agrColor = a => ({Agree:'success',Disagree:'danger',Partial:'warning'}[a]||'info');

  return (<div className="p-3">
    <h3>Clinical Decisions Dashboard</h3>
    <p className="text-muted">
      AI-neurologist HITL decision audit &mdash; {ov.total_decisions} decisions, {ov.unique_patients} patients,
      {' '}{ov.unique_reviewers} reviewers, {ov.agreement_rate_pct}% agreement rate
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          ['Total Decisions', ov.total_decisions, 'primary'],
          ['Patients', ov.unique_patients, 'info'],
          ['Agreement %', ov.agreement_rate_pct+'%', 'success'],
          ['Override %', ov.override_rate_pct+'%', 'danger'],
          ['Avg Confidence', (ov.avg_confidence*100).toFixed(1)+'%', 'warning'],
          ['Reviewers', ov.unique_reviewers, 'secondary'],
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
            <h6>Agreement Distribution</h6>
            {Object.entries(ov.agreement_distribution).map(([level, count])=>{
              const pct = ((count/ov.total_decisions)*100).toFixed(1);
              return <div key={level} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'80px'}}>{level}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${agrColor(level)}`} style={{width:`${pct}%`}}>
                      {count} ({pct}%)
                    </div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Final Decision Distribution</h6>
            {Object.entries(ov.decision_distribution).map(([dec, count])=>{
              const pct = ((count/ov.total_decisions)*100).toFixed(1);
              return <div key={dec} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'80px'}}>{dec}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${decColor(dec)}`} style={{width:`${pct}%`}}>
                      {count} ({pct}%)
                    </div>
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
            <h6>AI Confidence Distribution</h6>
            {ov.confidence_distribution.map(b=>
              <div key={b.bucket} className="d-flex align-items-center mb-2">
                <span className="me-2 small font-monospace" style={{minWidth:'60px'}}>{b.bucket}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className="progress-bar bg-info" style={{width:`${(b.count/ov.total_decisions*100).toFixed(1)}%`}}>
                      {b.count}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div></div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Monthly Decision Volume</h6>
            <table className="table table-sm table-bordered">
              <thead><tr><th>Month</th><th>Decisions</th><th>Trend</th></tr></thead>
              <tbody>
                {ov.monthly_decisions.map((m,i)=>{
                  const prev = i>0 ? ov.monthly_decisions[i-1].decisions : m.decisions;
                  const diff = m.decisions - prev;
                  return <tr key={m.date}>
                    <td>{m.date}</td>
                    <td className="fw-bold">{m.decisions}</td>
                    <td>{i===0?'--':diff>=0?<span className="text-success">+{diff}</span>:<span className="text-danger">{diff}</span>}</td>
                  </tr>;
                })}
              </tbody>
            </table>
          </div></div>
        </div>
      </div>
    </div>}

    {tab==='reviewers' && bd && <div>
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Reviewer Workload</h6>
            <div className="table-responsive">
              <table className="table table-sm table-striped table-bordered">
                <thead><tr>
                  <th>Reviewer</th><th>Total</th><th>Agrees</th><th>Disagrees</th><th>Overrides</th><th>Avg Conf.</th>
                </tr></thead>
                <tbody>
                  {bd.reviewer_workload.map(r=>
                    <tr key={r.reviewer}>
                      <td className="fw-bold">{r.reviewer}</td>
                      <td>{r.total}</td>
                      <td><span className="badge bg-success">{r.agrees}</span></td>
                      <td><span className="badge bg-danger">{r.disagrees}</span></td>
                      <td><span className="badge bg-warning text-dark">{r.overrides}</span></td>
                      <td>{(r.avg_confidence*100).toFixed(1)}%</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div></div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Reviewer Performance</h6>
            <div className="table-responsive">
              <table className="table table-sm table-striped table-bordered">
                <thead><tr>
                  <th>Reviewer</th><th>Total</th><th>Agree %</th><th>Override %</th><th>Escalate %</th>
                </tr></thead>
                <tbody>
                  {bd.reviewer_performance.map(r=>
                    <tr key={r.reviewer}>
                      <td className="fw-bold">{r.reviewer}</td>
                      <td>{r.total}</td>
                      <td><span className={`badge bg-${r.agree_rate>=50?'success':'warning'}`}>{r.agree_rate.toFixed(1)}%</span></td>
                      <td><span className={`badge bg-${r.override_rate>=30?'danger':'secondary'}`}>{r.override_rate.toFixed(1)}%</span></td>
                      <td>{r.escalate_rate.toFixed(1)}%</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div></div>
        </div>
      </div>

      {bd.artifact_vs_disagreement && bd.artifact_vs_disagreement.length>0 && <div className="card shadow-sm"><div className="card-body">
        <h6>Artifact Risk vs Disagreement</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead><tr><th>Artifact Risk</th><th>Count</th><th>Disagree Rate</th></tr></thead>
            <tbody>
              {bd.artifact_vs_disagreement.map(a=>
                <tr key={a.artifact_risk||'none'}>
                  <td>{a.artifact_risk||'None'}</td>
                  <td>{a.count}</td>
                  <td><span className={`badge bg-${(a.disagree_rate||0)>=40?'danger':'success'}`}>{(a.disagree_rate||0).toFixed(1)}%</span></td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>}
    </div>}

    {tab==='disagreements' && bd && <div>
      <div className="alert alert-warning">
        <strong>Disagreement Cases:</strong> Instances where the neurologist disagreed with the AI prediction &mdash; critical for model improvement and clinical safety.
      </div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Disagreement Cases ({bd.disagreement_cases.length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-bordered">
            <thead><tr>
              <th>Patient</th><th>AI Prediction</th><th>Confidence</th><th>Top Channels</th>
              <th>Decision</th><th>Reviewer</th><th>Note</th><th>Date</th>
            </tr></thead>
            <tbody>
              {bd.disagreement_cases.map((c,i)=>
                <tr key={i}>
                  <td className="font-monospace">{c.patient_id}</td>
                  <td>{c.ai_prediction}</td>
                  <td><span className={`badge bg-${c.ai_confidence>=0.8?'success':c.ai_confidence>=0.6?'warning':'danger'}`}>
                    {(c.ai_confidence*100).toFixed(0)}%
                  </span></td>
                  <td className="small font-monospace">{c.top_channels}</td>
                  <td><span className={`badge bg-${decColor(c.final_decision)}`}>{c.final_decision}</span></td>
                  <td>{c.reviewer}</td>
                  <td className="small">{c.note}</td>
                  <td className="small">{c.created_at?.split(' ')[0]}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='predictions' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>AI Prediction &times; Final Decision Cross-Table</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered table-striped">
            <thead><tr>
              <th>AI Prediction</th><th className="text-success">Confirm</th><th className="text-secondary">Defer</th>
              <th className="text-warning">Escalate</th><th className="text-danger">Override</th><th>Total</th>
            </tr></thead>
            <tbody>
              {Object.entries(bd.prediction_decision_cross).map(([pred, decisions])=>{
                const total = Object.values(decisions).reduce((a,b)=>a+b,0);
                return <tr key={pred}>
                  <td className="fw-bold">{pred}</td>
                  <td><span className="badge bg-success">{decisions.Confirm||0}</span></td>
                  <td><span className="badge bg-secondary">{decisions.Defer||0}</span></td>
                  <td><span className="badge bg-warning text-dark">{decisions.Escalate||0}</span></td>
                  <td><span className="badge bg-danger">{decisions.Override||0}</span></td>
                  <td className="fw-bold">{total}</td>
                </tr>;
              })}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='recent' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Recent Decisions (latest {bd.recent_decisions.length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-bordered">
            <thead><tr>
              <th>Date</th><th>Patient</th><th>AI Prediction</th><th>Conf.</th>
              <th>Channels</th><th>Artifact</th><th>Agreement</th><th>Decision</th><th>Reviewer</th>
            </tr></thead>
            <tbody>
              {bd.recent_decisions.map((r,i)=>
                <tr key={i}>
                  <td className="small">{r.created_at?.split(' ')[0]}</td>
                  <td className="font-monospace">{r.patient_id}</td>
                  <td>{r.ai_prediction}</td>
                  <td><span className={`badge bg-${r.ai_confidence>=0.8?'success':r.ai_confidence>=0.6?'warning':'danger'}`}>
                    {(r.ai_confidence*100).toFixed(0)}%
                  </span></td>
                  <td className="small font-monospace">{r.top_channels}</td>
                  <td>{r.artifact_risk||'—'}</td>
                  <td><span className={`badge bg-${agrColor(r.neurologist_agreement)}`}>{r.neurologist_agreement}</span></td>
                  <td><span className={`badge bg-${decColor(r.final_decision)}`}>{r.final_decision}</span></td>
                  <td>{r.reviewer}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='definitions' && defs && <div>
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Decision Types</h6>
            <table className="table table-sm table-bordered">
              <thead><tr><th>Type</th><th>Description</th></tr></thead>
              <tbody>
                {defs.decision_types.map(t=>
                  <tr key={t.type}><td><span className={`badge bg-${decColor(t.type)}`}>{t.type}</span></td><td>{t.description}</td></tr>
                )}
              </tbody>
            </table>
          </div></div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Agreement Levels</h6>
            <table className="table table-sm table-bordered">
              <thead><tr><th>Level</th><th>Description</th></tr></thead>
              <tbody>
                {defs.agreement_levels.map(a=>
                  <tr key={a.level}><td><span className={`badge bg-${agrColor(a.level)}`}>{a.level}</span></td><td>{a.description}</td></tr>
                )}
              </tbody>
            </table>
          </div></div>
        </div>
      </div>

      {defs.ai_prediction_categories && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>AI Prediction Categories</h6>
        <table className="table table-sm table-bordered">
          <thead><tr><th>Category</th><th>Description</th></tr></thead>
          <tbody>
            {defs.ai_prediction_categories.map(c=>
              <tr key={c.category}><td className="fw-bold">{c.category}</td><td>{c.description}</td></tr>
            )}
          </tbody>
        </table>
      </div></div>}

      {defs.glossary && <div className="card shadow-sm"><div className="card-body">
        <h6>Glossary</h6>
        {defs.glossary.map(g=>
          <div key={g.term} className="mb-2">
            <strong>{g.term}:</strong> {g.definition}
          </div>
        )}
      </div></div>}
    </div>}
  </div>);
}
