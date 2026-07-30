'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function WorkbenchDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('pipeline');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/workbench/overview`).then(r=>r.json()),
      fetch(`${API}/api/workbench/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/workbench/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading workbench data...</div>;

  const TABS = [
    {id:'pipeline', label:'Pipeline Overview'},
    {id:'patients', label:'Patient Tracker'},
    {id:'cases', label:'Recent Cases'},
    {id:'reviewers', label:'Reviewer Workload'},
    {id:'xai', label:'Explainability'},
    {id:'audit', label:'Audit Trail'},
    {id:'definitions', label:'Definitions'},
  ];

  return (<div className="p-3">
    <h3>Clinical Workbench</h3>
    <p className="text-muted">
      Unified clinical decision pipeline &mdash; Patient &rarr; EEG &rarr; AI &rarr; Explainability &rarr; Human &rarr; Audit
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* ── Pipeline Overview ── */}
    {tab==='pipeline' && <div>
      {/* Stage flow */}
      <div className="d-flex flex-wrap align-items-center justify-content-center mb-4 gap-2">
        {ov.pipeline_stages.map((s,i)=><>
          <div key={s.stage} className="card shadow-sm text-center" style={{minWidth:'130px'}}>
            <div className="card-body py-2">
              <div style={{fontSize:'1.8rem'}}>{s.icon}</div>
              <div className="fw-bold">{s.stage}</div>
              <div className="h5 mb-0 text-primary">{s.count}</div>
            </div>
          </div>
          {i < ov.pipeline_stages.length - 1 && <span className="text-muted h4 mb-0">&rarr;</span>}
        </>)}
      </div>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          ['Patients', ov.total_patients, 'primary'],
          ['Analysed', ov.patients_analysed, 'info'],
          ['Analyses', ov.total_analyses, 'success'],
          ['Avg AI Confidence', (ov.avg_ai_confidence*100).toFixed(1)+'%', 'warning'],
          ['Decisions', ov.total_decisions, 'danger'],
          ['Agreement %', ov.agreement_rate_pct+'%', 'success'],
          ['Expert Reviews', ov.total_expert_reviews, 'secondary'],
          ['HITL Reviews', ov.total_hitl_reviews, 'info'],
          ['XAI Entries', ov.explainability_entries, 'warning'],
          ['Audit Events', ov.total_audit_events, 'dark'],
          ['Full Pipeline', ov.full_pipeline_patients, 'success'],
          ['Audited Patients', ov.audited_patients, 'primary'],
        ].map(([label,val,c])=>
          <div key={label} className="col-6 col-md-2 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className={`h5 mb-0 text-${c}`}>{val}</div>
              <div className="text-muted small">{label}</div>
            </div></div>
          </div>
        )}
      </div>

      {/* Disease + prediction distributions */}
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Analyses by Disease</h6>
            {Object.entries(ov.analyses_by_disease).map(([d, cnt])=>{
              const pct = ((cnt/ov.total_analyses)*100).toFixed(1);
              return <div key={d} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'100px'}}>{d}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className="progress-bar bg-info" style={{width:`${pct}%`}}>{cnt} ({pct}%)</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Decision Distribution</h6>
            {Object.entries(ov.decision_distribution).map(([d, cnt])=>{
              const pct = ((cnt/ov.total_decisions)*100).toFixed(1);
              const col = {Confirm:'success',Override:'danger',Escalate:'warning',Defer:'secondary'}[d]||'info';
              return <div key={d} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'80px'}}>{d}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${col}`} style={{width:`${pct}%`}}>{cnt} ({pct}%)</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
      </div>
    </div>}

    {/* ── Patient Pipeline Tracker ── */}
    {tab==='patients' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Patient Pipeline Status</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead><tr>
              <th>Patient</th><th>Name</th><th>Age</th><th>Sex</th>
              <th>Analyses</th><th>Decisions</th><th>Expert Reviews</th><th>Status</th>
            </tr></thead>
            <tbody>
              {bd.patient_pipeline.map(p=>{
                const complete = p.analysis_count > 0 && p.decision_count > 0;
                return <tr key={p.patient_id}>
                  <td className="fw-bold">{p.patient_id}</td>
                  <td>{p.name || '—'}</td>
                  <td>{p.age ?? '—'}</td>
                  <td>{p.sex || '—'}</td>
                  <td><span className={`badge bg-${p.analysis_count?'success':'secondary'}`}>{p.analysis_count}</span></td>
                  <td><span className={`badge bg-${p.decision_count?'info':'secondary'}`}>{p.decision_count}</span></td>
                  <td><span className={`badge bg-${p.expert_review_count?'warning':'secondary'}`}>{p.expert_review_count}</span></td>
                  <td>{complete
                    ? <span className="badge bg-success">Complete</span>
                    : <span className="badge bg-warning text-dark">In Progress</span>
                  }</td>
                </tr>;
              })}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Recent Cases ── */}
    {tab==='cases' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Recent Cases (Analysis + Decision)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead><tr>
              <th>ID</th><th>Patient</th><th>Disease</th><th>AI Label</th><th>Confidence</th>
              <th>Quality</th><th>Agreement</th><th>Decision</th><th>Reviewer</th><th>Note</th>
            </tr></thead>
            <tbody>
              {bd.recent_cases.map(c=>{
                const decCol = {Confirm:'success',Override:'danger',Escalate:'warning',Defer:'secondary'}[c.final_decision]||'info';
                return <tr key={c.analysis_id}>
                  <td>{c.analysis_id}</td>
                  <td className="fw-bold">{c.patient_id}</td>
                  <td>{c.disease}</td>
                  <td>{c.predicted_label}</td>
                  <td>{c.confidence ? (c.confidence * 100).toFixed(0) + '%' : '—'}</td>
                  <td>{c.signal_quality || '—'}</td>
                  <td>{c.neurologist_agreement || <span className="text-muted">Pending</span>}</td>
                  <td>{c.final_decision
                    ? <span className={`badge bg-${decCol}`}>{c.final_decision}</span>
                    : <span className="text-muted">—</span>
                  }</td>
                  <td>{c.reviewer || '—'}</td>
                  <td className="small">{c.decision_note || '—'}</td>
                </tr>;
              })}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Reviewer Workload ── */}
    {tab==='reviewers' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Reviewer Workload</h6>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr>
              <th>Reviewer</th><th>Total Cases</th><th>Confirms</th><th>Overrides</th><th>Escalations</th>
            </tr></thead>
            <tbody>
              {bd.reviewer_workload.map(r=>
                <tr key={r.reviewer}>
                  <td className="fw-bold">{r.reviewer}</td>
                  <td>{r.cases}</td>
                  <td><span className="badge bg-success">{r.confirms}</span></td>
                  <td><span className="badge bg-danger">{r.overrides}</span></td>
                  <td><span className="badge bg-warning text-dark">{r.escalations}</span></td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Explainability ── */}
    {tab==='xai' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Explainability Ground Truth</h6>
        {bd.explainability.length === 0
          ? <p className="text-muted">No explainability entries yet.</p>
          : bd.explainability.map((x,i)=>
            <div key={i} className="border rounded p-3 mb-3">
              <div className="fw-bold mb-1">{x.id}</div>
              <div className="mb-1"><strong>Key Features:</strong> {x.features.join(', ') || '—'}</div>
              <div className="mb-1"><strong>Channels:</strong> {x.channels.join(', ') || '—'}</div>
              <div className="mb-1"><strong>Rationale:</strong> {x.rationale || '—'}</div>
              <div className="text-muted small">Created: {x.created_at}</div>
            </div>
          )
        }
      </div></div>
    </div>}

    {/* ── Audit Trail ── */}
    {tab==='audit' && bd && <div>
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Audit Events by Category</h6>
            {bd.audit_by_category.map(a=>{
              const total = bd.audit_by_category.reduce((s,x)=>s+x.event_count,0);
              const pct = ((a.event_count/total)*100).toFixed(1);
              return <div key={a.category} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'120px'}}>{a.category}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className="progress-bar bg-dark" style={{width:`${pct}%`}}>{a.event_count} ({pct}%)</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Audit Coverage</h6>
            <p>Total audit events: <strong>{ov.total_audit_events}</strong></p>
            <p>Patients with audit trail: <strong>{ov.audited_patients}</strong> / {ov.total_patients}</p>
            <p>Full pipeline patients: <strong>{ov.full_pipeline_patients}</strong></p>
          </div></div>
        </div>
      </div>
    </div>}

    {/* ── Definitions ── */}
    {tab==='definitions' && defs && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Pipeline Stages</h6>
        {Object.entries(defs.pipeline_stages).map(([stage, desc])=>
          <div key={stage} className="mb-2">
            <strong>{stage}:</strong> {desc}
          </div>
        )}
      </div></div>
      <div className="row">
        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Decision Types</h6>
            {Object.entries(defs.decision_types).map(([k,v])=>
              <div key={k} className="mb-2"><strong>{k}:</strong> {v}</div>
            )}
          </div></div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Agreement Levels</h6>
            {Object.entries(defs.agreement_levels).map(([k,v])=>
              <div key={k} className="mb-2"><strong>{k}:</strong> {v}</div>
            )}
          </div></div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Glossary</h6>
            {Object.entries(defs.glossary).map(([k,v])=>
              <div key={k} className="mb-2"><strong>{k}:</strong> {v}</div>
            )}
          </div></div>
        </div>
      </div>
    </div>}
  </div>);
}
