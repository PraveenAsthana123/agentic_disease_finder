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
  const mx = Math.max(...items.map(i=>i.value));
  return (
    <div>{items.map((it,i)=>(
      <div key={i} className="d-flex align-items-center mb-1">
        <div className="text-end small me-2" style={{width:140,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{it.label}</div>
        <div className="flex-grow-1">
          <div className="progress" style={{height:18}}>
            <div className={`progress-bar bg-${colorFn?colorFn(it):it.color||'primary'}`}
                 style={{width:`${mx?((it.value/mx)*100):0}%`}}>
              <span className="small">{typeof it.value==='number'?it.value.toFixed(3):it.value}</span>
            </div>
          </div>
        </div>
      </div>
    ))}</div>
  );
}

function PieChart({data, size}){
  if(!data||!data.length) return null;
  const sz = size || 180;
  const total = data.reduce((s,d)=>s+d.value,0);
  if(!total) return null;
  let cum = 0;
  const colors = ['#3b82f6','#ef4444','#f59e0b','#10b981','#8b5cf6','#06b6d4','#ec4899'];
  return (
    <div className="d-flex align-items-center">
      <svg width={sz} height={sz} viewBox={`0 0 ${sz} ${sz}`}>
        {data.map((d,i)=>{
          const start = cum/total*360;
          const sweep = d.value/total*360;
          cum += d.value;
          const r = sz/2-5;
          const cx = sz/2, cy = sz/2;
          const s1 = (start-90)*Math.PI/180;
          const s2 = (start+sweep-90)*Math.PI/180;
          const large = sweep>180?1:0;
          const path = `M${cx},${cy} L${cx+r*Math.cos(s1)},${cy+r*Math.sin(s1)} A${r},${r} 0 ${large},1 ${cx+r*Math.cos(s2)},${cy+r*Math.sin(s2)} Z`;
          return <path key={i} d={path} fill={d.color||colors[i%colors.length]} stroke="#fff" strokeWidth="1"/>;
        })}
      </svg>
      <div className="ms-3">
        {data.map((d,i)=>(
          <div key={i} className="small"><span style={{display:'inline-block',width:12,height:12,borderRadius:2,background:d.color||colors[i%colors.length],marginRight:4}}/>{d.label}: {d.value}</div>
        ))}
      </div>
    </div>
  );
}

export default function ExplainableAIDashboard(){
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/explainable-ai/overview`).then(r=>r.json()),
      fetch(`${API}/api/explainable-ai/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/explainable-ai/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOverview(o);setBreakdown(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load XAI data: {err}</div>;
  if(!overview) return <div className="text-muted p-3">Loading Explainable AI dashboard...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'features', label:'Feature Importance'},
    {id:'patients', label:'Patient Profiles'},
    {id:'counterfactuals', label:'Counterfactuals'},
    {id:'definitions', label:'Definitions'},
  ];

  const kpis = overview.kpis || {};
  const catColors = {'Time-Domain':'primary','Spectral':'info','Complexity':'warning','Hjorth':'success','Connectivity':'secondary'};

  // Convert disease_wise_importance dict to array
  const diseaseImportance = overview.disease_wise_importance
    ? Object.entries(overview.disease_wise_importance).map(([disease, features])=>({
        disease, n_analyses: (overview.global_feature_importance||[]).length, top_features: (features||[]).slice(0,5)
      }))
    : [];

  return (<div className="container-fluid py-3">
    <h3>Explainable AI (XAI) Dashboard</h3>
    <p className="text-muted">Feature importance, SHAP-like analysis, and counterfactual explanations from real EEG analyses.</p>

    <div className="row mb-3">
      <KPI label="Total Analyses" value={kpis.total_analyses||0} color="primary"/>
      <KPI label="Patients Analysed" value={kpis.total_patients||0} color="info"/>
      <KPI label="EEG Features" value={kpis.total_features||0} color="success"/>
      <KPI label="Avg Confidence" value={typeof kpis.avg_confidence==='number'?kpis.avg_confidence.toFixed(2):'-'} color="primary"/>
      <KPI label="Important Features" value={kpis.features_above_threshold||0} color="warning"/>
      <KPI label="Top Feature" value={kpis.top_feature||'-'} color="danger"/>
      <KPI label="Diseases Modelled" value={kpis.model_diseases||0} color="info"/>
      <KPI label="Avg Signal Quality" value={kpis.avg_signal_quality||'-'} color="success"/>
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
        <div className="col-md-7 mb-3">
          <div className="card shadow-sm h-100"><div className="card-body">
            <h6>Top 20 Feature Importance (correlation with confidence)</h6>
            <Bar items={(overview.global_feature_importance||[]).slice(0,20).map(f=>({
              label: f.feature, value: f.importance,
              color: catColors[f.category]||'secondary'
            }))} colorFn={it=>it.color}/>
          </div></div>
        </div>
        <div className="col-md-5 mb-3">
          <div className="card shadow-sm h-100"><div className="card-body">
            <h6>Importance by Feature Category</h6>
            <PieChart data={(overview.category_importance||[]).map(c=>({
              label: c.category, value: Math.round(c.total_importance*1000)/1000,
              color: {'Time-Domain':'#3b82f6','Spectral':'#06b6d4','Complexity':'#f59e0b','Hjorth':'#10b981','Connectivity':'#8b5cf6'}[c.category]||'#888'
            }))} size={200}/>
            <div className="mt-3">
              <h6>Band Power Contribution</h6>
              <Bar items={(overview.band_power_contribution||[]).map(b=>({label: b.band, value: b.mean_contribution, color:'info'}))}/>
            </div>
          </div></div>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Prediction Confidence Distribution</h6>
            <div className="d-flex align-items-end" style={{height:120}}>
              {(overview.confidence_distribution||[]).map((bin,i)=>(
                <div key={i} className="flex-fill text-center mx-1">
                  <div className="bg-primary" style={{height:`${bin.count/Math.max(...(overview.confidence_distribution||[]).map(b=>b.count),1)*100}%`,minHeight:bin.count?4:0,borderRadius:'3px 3px 0 0'}}/>
                  <div className="small text-muted mt-1">{bin.bin}</div>
                  <div className="small fw-bold">{bin.count}</div>
                </div>
              ))}
            </div>
          </div></div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Top Features by Disease</h6>
            {diseaseImportance.map(d=>(
              <div key={d.disease} className="mb-2">
                <strong className="text-capitalize">{d.disease}</strong>
                <div className="small">
                  {(d.top_features||[]).map((f,i)=>(
                    <span key={i} className={`badge bg-${catColors[f.category]||'secondary'} me-1 mb-1`}>
                      {f.feature} ({f.importance.toFixed(3)})
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div></div>
        </div>
      </div>
    </div>)}

    {/* Feature Importance Tab */}
    {tab==='features' && breakdown && (<div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Per-Feature Statistics and Importance Rankings</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead><tr>
              <th>Rank</th><th>Feature</th><th>Category</th><th>Importance</th>
              <th>Mean</th><th>Std</th><th>Min</th><th>Max</th>
            </tr></thead>
            <tbody>
              {(breakdown.per_feature_stats||[]).slice(0,30).map((f,i)=>(
                <tr key={i}>
                  <td>{f.importance_rank}</td>
                  <td className="fw-bold">{f.feature}</td>
                  <td><span className={`badge bg-${catColors[f.category]||'secondary'}`}>{f.category}</span></td>
                  <td>{typeof f.importance==='number'?f.importance.toFixed(4):'-'}</td>
                  <td>{typeof f.mean==='number'?f.mean.toFixed(4):'-'}</td>
                  <td>{typeof f.std==='number'?f.std.toFixed(4):'-'}</td>
                  <td>{typeof f.min==='number'?f.min.toFixed(4):'-'}</td>
                  <td>{typeof f.max==='number'?f.max.toFixed(4):'-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Top Feature Correlations</h6>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Feature A</th><th>Feature B</th><th>Correlation</th><th>Strength</th></tr></thead>
            <tbody>
              {(breakdown.feature_correlation_matrix||[]).slice(0,15).map((c,i)=>(
                <tr key={i}>
                  <td>{c.feature_1}</td>
                  <td>{c.feature_2}</td>
                  <td>{typeof c.correlation==='number'?c.correlation.toFixed(4):'-'}</td>
                  <td><span className={`badge bg-${Math.abs(c.correlation||0)>0.8?'danger':Math.abs(c.correlation||0)>0.5?'warning':'info'}`}>
                    {Math.abs(c.correlation||0)>0.8?'Strong':Math.abs(c.correlation||0)>0.5?'Moderate':'Weak'}
                  </span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>}

      <div className="card shadow-sm"><div className="card-body">
        <h6>Feature Direction Analysis (SHAP-like)</h6>
        <p className="text-muted small">Shows whether higher feature values push prediction toward higher or lower confidence.</p>
        <div className="row">
          {(breakdown.shap_direction_analysis||[]).slice(0,12).map((d,i)=>(
            <div key={i} className="col-md-4 col-lg-3 mb-2">
              <div className="border rounded p-2 small">
                <strong>{d.feature}</strong>
                <div className="text-muted">{d.category}</div>
                <div className="fw-bold text-info">
                  {d.direction}
                </div>
                <div className="text-muted">importance: {typeof d.importance==='number'?d.importance.toFixed(3):'-'}</div>
              </div>
            </div>
          ))}
        </div>
      </div></div>
    </div>)}

    {/* Patient Profiles Tab */}
    {tab==='patients' && breakdown && (<div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Per-Patient Explainability Profiles</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead><tr>
              <th>Patient</th><th>Name</th><th>Disease</th><th>Prediction</th>
              <th>Confidence</th><th>Top Contributing Features</th>
            </tr></thead>
            <tbody>
              {(breakdown.per_patient_profiles||[]).map((p,i)=>(
                <tr key={i}>
                  <td className="fw-bold">{p.patient_id}</td>
                  <td>{p.name}</td>
                  <td className="text-capitalize">{p.disease}</td>
                  <td>{p.predicted_label}</td>
                  <td>
                    <span className={`badge bg-${(p.confidence||0)>=0.7?'success':(p.confidence||0)>=0.5?'warning':'danger'}`}>
                      {typeof p.confidence==='number'?p.confidence.toFixed(2):'-'}
                    </span>
                  </td>
                  <td>
                    {(p.top_3_features||[]).map((f,j)=>(
                      <span key={j} className="badge bg-secondary me-1">{f.feature} ({typeof f.zscore==='number'?f.zscore.toFixed(1):'-'}{'\u03c3'})</span>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>)}

    {/* Counterfactuals Tab */}
    {tab==='counterfactuals' && breakdown && (<div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Counterfactual Explanations</h6>
        <p className="text-muted small">
          For each analysis: the feature with the highest class separation, and the delta needed to shift toward the alternative class.
        </p>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead><tr>
              <th>Patient</th><th>Current Prediction</th>
              <th>Key Feature</th><th>Category</th><th>Current Value</th>
              <th>Delta Needed</th><th>Class Separation</th><th>Flip To</th>
            </tr></thead>
            <tbody>
              {(breakdown.counterfactual_analysis||[]).map((c,i)=>(
                <tr key={i}>
                  <td className="fw-bold">{c.patient_id}</td>
                  <td>{c.predicted_label}</td>
                  <td><strong>{c.flip_feature}</strong></td>
                  <td><span className={`badge bg-${catColors[c.flip_feature_category]||'secondary'}`}>{c.flip_feature_category}</span></td>
                  <td>{typeof c.current_value==='number'?c.current_value.toFixed(4):'-'}</td>
                  <td><span className={`badge bg-${Math.abs(c.delta_needed||0)>100?'danger':'warning'}`}>
                    {typeof c.delta_needed==='number'?(c.delta_needed>0?'+':'')+c.delta_needed.toFixed(2):'-'}
                  </span></td>
                  <td>{typeof c.class_separation==='number'?c.class_separation.toFixed(2):'-'}</td>
                  <td>{c.counterfactual_class||'-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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
