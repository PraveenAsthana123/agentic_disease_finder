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

function Bar({items, colorFn}){
  if(!items||!items.length) return null;
  const mx = Math.max(...items.map(i=>Math.abs(i.value)));
  return (
    <div>{items.map((it,i)=>(
      <div key={i} className="d-flex align-items-center mb-1">
        <div className="text-end small me-2" style={{width:160,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{it.label}</div>
        <div className="flex-grow-1">
          <div className="progress" style={{height:18}}>
            <div className={`progress-bar bg-${colorFn?colorFn(it):it.color||'primary'}`}
                 style={{width:`${mx?((Math.abs(it.value)/mx)*100):0}%`}}>
              <span className="small">{typeof it.value==='number'?it.value.toFixed(4):it.value}</span>
            </div>
          </div>
        </div>
      </div>
    ))}</div>
  );
}

export default function InterpretableAIDashboard(){
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/interpretable-ai/overview`).then(r=>r.json()),
      fetch(`${API}/api/interpretable-ai/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/interpretable-ai/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOverview(o);setBreakdown(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load Interpretable AI data: {err}</div>;
  if(!overview) return <div className="text-muted p-3">Loading Interpretable AI dashboard...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'models', label:'Interpretable Models'},
    {id:'rules', label:'Decision Rules'},
    {id:'paths', label:'Patient Paths'},
    {id:'definitions', label:'Definitions'},
  ];

  const kpis = overview.kpis || [];
  const dt = overview.decision_tree || {};
  const lr = overview.logistic_regression || {};
  const rules = overview.top_decision_rules || [];
  const comparison = overview.accuracy_comparison || [];

  return (<div className="container-fluid py-3">
    <h3>Interpretable AI Dashboard</h3>
    <p className="text-muted">Intrinsically interpretable models (decision trees, logistic regression) trained on real EEG features — every prediction path is human-readable.</p>

    <div className="row mb-3">
      {kpis.map((k,i)=>(
        <KPI key={i} label={k.label} value={k.value} color={k.color||'primary'}/>
      ))}
    </div>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=>(
        <li key={t.id} className="nav-item">
          <button className={`nav-link${tab===t.id?' active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
        </li>
      ))}
    </ul>

    {/* Overview Tab */}
    {tab==='overview' && (<div>
      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100"><div className="card-body">
            <h6>Decision Tree — Feature Importance</h6>
            <p className="text-muted small">Depth: {dt.depth}, Nodes: {dt.n_nodes}, Leaves: {dt.n_leaves}, CV Accuracy: {dt.cv_accuracy_pct}%</p>
            <Bar items={(dt.feature_importance||[]).slice(0,15).map(f=>({
              label: f.feature, value: f.importance, color:'success'
            }))} colorFn={it=>it.color}/>
          </div></div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100"><div className="card-body">
            <h6>Logistic Regression — Coefficients</h6>
            <p className="text-muted small">CV Accuracy: {lr.cv_accuracy_pct}%, Non-zero coefficients: {(lr.positive_coefficients||[]).length + (lr.negative_coefficients||[]).length}</p>
            <h6 className="small text-success mt-2">Positive (push toward high-confidence)</h6>
            <Bar items={(lr.positive_coefficients||[]).slice(0,8).map(f=>({
              label: f.feature, value: f.coefficient, color:'success'
            }))} colorFn={it=>it.color}/>
            <h6 className="small text-danger mt-2">Negative (push toward low-confidence)</h6>
            <Bar items={(lr.negative_coefficients||[]).slice(0,8).map(f=>({
              label: f.feature, value: Math.abs(f.coefficient), color:'danger'
            }))} colorFn={it=>it.color}/>
          </div></div>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Accuracy Comparison — Interpretable vs Black-Box</h6>
            <div className="table-responsive">
              <table className="table table-sm table-striped">
                <thead><tr><th>Model</th><th>Type</th><th>Accuracy</th><th>Interpretable</th></tr></thead>
                <tbody>
                  {comparison.map((m,i)=>(
                    <tr key={i}>
                      <td className="fw-bold">{m.model}</td>
                      <td>{m.type}</td>
                      <td><span className={`badge bg-${(m.accuracy_pct||0)>=90?'success':(m.accuracy_pct||0)>=70?'warning':'danger'}`}>{m.accuracy_pct}%</span></td>
                      <td>{m.interpretable ? <span className="badge bg-success">Yes</span> : <span className="badge bg-secondary">No</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div></div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Top Decision Rules</h6>
            {rules.slice(0,8).map((r,i)=>(
              <div key={i} className="border rounded p-2 mb-2 small">
                <div className="fw-bold text-primary">Rule {r.rule_id}</div>
                <code className="d-block my-1">{r.path}</code>
                <div>Class: <span className="badge bg-info">{r.class_label}</span> | Samples: {r.samples} | Purity: {typeof r.purity==='number'?(r.purity*100).toFixed(1)+'%':'-'}</div>
              </div>
            ))}
          </div></div>
        </div>
      </div>
    </div>)}

    {/* Interpretable Models Tab */}
    {tab==='models' && breakdown && (<div>
      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Decision Tree Structure</h6>
            {dt.tree_text ? (
              <pre className="bg-light p-2 rounded small" style={{maxHeight:400,overflow:'auto'}}>{dt.tree_text}</pre>
            ) : <p className="text-muted">Tree text not available</p>}
          </div></div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>DT vs LR Feature Importance Comparison</h6>
            <div className="table-responsive">
              <table className="table table-sm table-striped">
                <thead><tr><th>Feature</th><th>DT Importance</th><th>LR |Coef|</th><th>Agreement</th></tr></thead>
                <tbody>
                  {(breakdown.importance_comparison||[]).slice(0,20).map((f,i)=>(
                    <tr key={i}>
                      <td className="fw-bold">{f.feature}</td>
                      <td>{typeof f.dt_importance==='number'?f.dt_importance.toFixed(4):'-'}</td>
                      <td>{typeof f.lr_abs_coef==='number'?f.lr_abs_coef.toFixed(4):'-'}</td>
                      <td><span className={`badge bg-${f.agreement==='strong'?'success':f.agreement==='moderate'?'warning':'secondary'}`}>{f.agreement}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div></div>
        </div>
      </div>

      {(breakdown.per_disease_models||[]).length > 0 && (
        <div className="card shadow-sm mb-3"><div className="card-body">
          <h6>Per-Disease Model Summary</h6>
          <div className="table-responsive">
            <table className="table table-sm">
              <thead><tr><th>Disease</th><th>Samples</th><th>DT Accuracy</th><th>LR Accuracy</th><th>Top DT Feature</th></tr></thead>
              <tbody>
                {breakdown.per_disease_models.map((d,i)=>(
                  <tr key={i}>
                    <td className="fw-bold text-capitalize">{d.disease}</td>
                    <td>{d.n_samples}</td>
                    <td>{d.dt_accuracy_pct}%</td>
                    <td>{d.lr_accuracy_pct}%</td>
                    <td>{d.top_dt_feature}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div></div>
      )}

      <div className="card shadow-sm"><div className="card-body">
        <h6>Full Coefficient Table (Logistic Regression)</h6>
        <div className="table-responsive" style={{maxHeight:400,overflow:'auto'}}>
          <table className="table table-sm table-striped">
            <thead><tr><th>Feature</th><th>Coefficient</th><th>Direction</th></tr></thead>
            <tbody>
              {(breakdown.full_coefficients||[]).map((f,i)=>(
                <tr key={i}>
                  <td>{f.feature}</td>
                  <td>{typeof f.coefficient==='number'?f.coefficient.toFixed(6):'-'}</td>
                  <td><span className={`badge bg-${(f.coefficient||0)>0?'success':'danger'}`}>{(f.coefficient||0)>0?'Positive':'Negative'}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>)}

    {/* Decision Rules Tab */}
    {tab==='rules' && breakdown && (<div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>All Decision Rules (extracted from Decision Tree)</h6>
        <p className="text-muted small">Each rule is a path from root to leaf. The path shows the feature thresholds that lead to a class prediction.</p>
        {(breakdown.all_decision_rules||[]).map((r,i)=>(
          <div key={i} className="border rounded p-2 mb-2">
            <div className="d-flex justify-content-between">
              <span className="fw-bold text-primary">Rule {r.rule_id}</span>
              <span><span className="badge bg-info me-1">{r.class_label}</span>
                <span className="badge bg-secondary">n={r.samples}</span>
                <span className={`badge ms-1 bg-${(r.purity||0)>=0.9?'success':(r.purity||0)>=0.7?'warning':'danger'}`}>
                  purity {typeof r.purity==='number'?(r.purity*100).toFixed(1)+'%':'-'}
                </span>
              </span>
            </div>
            <code className="d-block mt-1 small">{r.path}</code>
          </div>
        ))}
      </div></div>
    </div>)}

    {/* Patient Paths Tab */}
    {tab==='paths' && breakdown && (<div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Per-Patient Decision Paths</h6>
        <p className="text-muted small">For each EEG analysis, the decision tree path and logistic regression top contributors that led to the prediction.</p>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead><tr>
              <th>Patient</th><th>Analysis</th><th>DT Prediction</th><th>LR Prediction</th>
              <th>Actual Confidence</th><th>Actions</th>
            </tr></thead>
            <tbody>
              {(breakdown.per_patient_paths||[]).map((p,i)=>(
                <tr key={i}>
                  <td className="fw-bold">{p.patient_id}</td>
                  <td>{p.analysis_id}</td>
                  <td><span className={`badge bg-${p.dt_prediction==='high_confidence'?'success':'warning'}`}>{p.dt_prediction}</span></td>
                  <td><span className={`badge bg-${p.lr_prediction==='high_confidence'?'success':'warning'}`}>{p.lr_prediction}</span></td>
                  <td>{typeof p.actual_confidence==='number'?p.actual_confidence.toFixed(3):'-'}</td>
                  <td><button className="btn btn-sm btn-outline-primary" onClick={()=>setSelectedPatient(selectedPatient===i?null:i)}>
                    {selectedPatient===i?'Hide':'Show'} Path
                  </button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {selectedPatient!==null && breakdown.per_patient_paths[selectedPatient] && (
          <div className="mt-3 border rounded p-3 bg-light">
            <h6>Decision Path for {breakdown.per_patient_paths[selectedPatient].patient_id} (Analysis {breakdown.per_patient_paths[selectedPatient].analysis_id})</h6>
            <div className="mb-2">
              <strong>Decision Tree Path:</strong>
              <code className="d-block mt-1">{breakdown.per_patient_paths[selectedPatient].dt_path}</code>
            </div>
            <div>
              <strong>Logistic Regression — Top Contributors:</strong>
              <div className="mt-1">
                {(breakdown.per_patient_paths[selectedPatient].lr_top_contributors||[]).map((c,j)=>(
                  <span key={j} className={`badge me-1 mb-1 bg-${(c.contribution||0)>0?'success':'danger'}`}>
                    {c.feature}: {typeof c.contribution==='number'?c.contribution.toFixed(4):'-'}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div></div>
    </div>)}

    {/* Definitions Tab */}
    {tab==='definitions' && defs && (<div>
      {(defs.sections||[]).map((sec,i)=>(
        <div key={i} className="card shadow-sm mb-3"><div className="card-body">
          <h6>{sec.title}</h6>
          {sec.items && <ul>{sec.items.map((item,j)=>(
            <li key={j}><strong>{item.term}:</strong> {item.description}</li>
          ))}</ul>}
          {sec.text && <p className="text-muted small">{sec.text}</p>}
        </div></div>
      ))}
    </div>)}
  </div>);
}
