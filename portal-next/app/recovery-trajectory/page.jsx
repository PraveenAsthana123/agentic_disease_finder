'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function Badge({val}){
  const m = {improving:'success', stable:'warning', declining:'danger'};
  return <span className={`badge bg-${m[val]||'secondary'} me-1`}>{val}</span>;
}

function SlopeArrow({slope}){
  if(slope > 0.1) return <span className="text-success fw-bold">▲ +{slope.toFixed(2)}</span>;
  if(slope < -0.1) return <span className="text-danger fw-bold">▼ {slope.toFixed(2)}</span>;
  return <span className="text-warning fw-bold">● {slope.toFixed(2)}</span>;
}

function MiniBar({value, max=10, color='primary'}){
  const pct = Math.min(100, Math.max(0, (value/max)*100));
  return <div className="progress" style={{height:'8px', minWidth:'80px'}}>
    <div className={`progress-bar bg-${color}`} style={{width:`${pct}%`}}></div>
  </div>;
}

export default function RecoveryTrajectoryDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/recovery-trajectory/overview`).then(r=>r.json()),
      fetch(`${API}/api/recovery-trajectory/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/recovery-trajectory/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading recovery trajectory data…</div>;

  const TABS = [
    {id:'overview', label:'📊 Overview'},
    {id:'patients', label:'👤 Patient Table'},
    {id:'declining', label:'🚨 Declining'},
    {id:'predictions', label:'🔮 Prediction Factors'},
    {id:'definitions', label:'📖 Definitions'},
  ];

  const kpi = ov.kpis;

  return (<div className="p-3">
    <h3>📈 Recovery Trajectory Dashboard</h3>
    <p className="text-muted">Functional recovery forecasting — {kpi.total_patients_tracked} patients tracked, slope-based trajectory classification</p>

    {/* Tab Nav */}
    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* ── Overview Tab ── */}
    {tab==='overview' && <div>
      {/* KPI Hero Tiles */}
      <div className="row mb-3">
        {[
          ['Patients Tracked', kpi.total_patients_tracked, 'primary'],
          ['Avg Recovery Score', kpi.avg_recovery_score?.toFixed(1), 'info'],
          ['Improving', kpi.improving_count, 'success'],
          ['Stable', kpi.stable_count, 'warning'],
          ['Declining', kpi.declining_count, 'danger'],
          ['Need Intensive Rehab', kpi.patients_needing_intensive_rehab, 'dark'],
        ].map(([k,v,c])=>
          <div key={k} className="col-6 col-md-2 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className={`h5 mb-0 text-${c}`}>{v}</div>
              <div className="text-muted small">{k}</div>
            </div></div>
          </div>
        )}
      </div>

      {/* Trajectory Distribution */}
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Trajectory Distribution</h6>
            {ov.trajectory_distribution.map(t=>
              <div key={t.name} className="d-flex align-items-center mb-2">
                <Badge val={t.name}/>
                <div className="flex-grow-1 mx-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${t.name==='improving'?'success':t.name==='stable'?'warning':'danger'}`}
                      style={{width:`${(t.count/kpi.total_patients_tracked)*100}%`}}>
                      {t.count} ({((t.count/kpi.total_patients_tracked)*100).toFixed(0)}%)
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div></div>
        </div>

        {/* Monthly Trend */}
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Monthly Avg Function Rating</h6>
            <table className="table table-sm table-bordered">
              <thead><tr><th>Month</th><th>Avg Function</th><th>Assessments</th><th>Trend</th></tr></thead>
              <tbody>
                {ov.monthly_avg_function.map((m,i)=>{
                  const prev = i>0 ? ov.monthly_avg_function[i-1].avg_daily_function : m.avg_daily_function;
                  const diff = m.avg_daily_function - prev;
                  return <tr key={m.month}>
                    <td className="small">{m.month}</td>
                    <td className="fw-bold">{m.avg_daily_function.toFixed(2)}</td>
                    <td>{m.num_assessments}</td>
                    <td>{i===0?'—':diff>=0?<span className="text-success">+{diff.toFixed(2)}</span>:<span className="text-danger">{diff.toFixed(2)}</span>}</td>
                  </tr>;
                })}
              </tbody>
            </table>
          </div></div>
        </div>
      </div>

      {/* Risk Factors Bar */}
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Risk Factor Prevalence</h6>
        <div className="row">
          {ov.risk_factor_bar.map(rf=>
            <div key={rf.risk_factor} className="col-md-3 col-6 mb-2">
              <div className="text-center">
                <div className="h4 text-danger mb-0">{rf.count}</div>
                <div className="small text-muted">{rf.risk_factor}</div>
              </div>
            </div>
          )}
        </div>
      </div></div>
    </div>}

    {/* ── Patient Table Tab ── */}
    {tab==='patients' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>All Patients — Trajectory Detail</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-hover">
            <thead className="table-dark"><tr>
              <th>Patient</th><th>Age</th><th>Gender</th><th>Trajectory</th><th>Slope</th>
              <th>Function</th><th>MoCA</th><th>PHQ-9</th><th>Fatigue</th><th>Sleep</th>
              <th>Risk Factors</th><th>Rehab</th>
            </tr></thead>
            <tbody>
              {bd.patient_table.map(p=>
                <tr key={p.patient_id} className={p.intensive_rehab_recommended?'table-danger':''}>
                  <td className="small fw-bold">{p.patient_id}</td>
                  <td>{p.age}</td>
                  <td>{p.gender}</td>
                  <td><Badge val={p.trajectory_class}/></td>
                  <td><SlopeArrow slope={p.slope}/></td>
                  <td><span className="me-1">{p.latest_function_rating}</span><MiniBar value={p.latest_function_rating}/></td>
                  <td className={p.latest_moca<25?'text-danger fw-bold':''}>{p.latest_moca}</td>
                  <td className={p.latest_phq9>10?'text-danger fw-bold':''}>{p.latest_phq9}</td>
                  <td className={p.latest_fatigue>6?'text-warning fw-bold':''}>{p.latest_fatigue}</td>
                  <td className={p.latest_sleep<6?'text-danger fw-bold':''}>{p.latest_sleep?.toFixed(1)}</td>
                  <td className="small">{p.risk_factors.length>0?p.risk_factors.join(', '):'—'}</td>
                  <td>{p.intensive_rehab_recommended?<span className="badge bg-danger">Yes</span>:<span className="badge bg-secondary">No</span>}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Declining Patients Tab ── */}
    {tab==='declining' && bd && <div>
      <div className="alert alert-danger">
        <strong>{bd.declining_patients.length} patients</strong> on a declining trajectory — prioritise for intervention.
      </div>
      <div className="row">
        {bd.declining_patients.map(p=>
          <div key={p.patient_id} className="col-md-4 col-sm-6 mb-3">
            <div className="card border-danger shadow-sm h-100"><div className="card-body">
              <div className="d-flex justify-content-between">
                <h6 className="card-title mb-1">{p.patient_id}</h6>
                {p.intensive_rehab_recommended && <span className="badge bg-danger">Rehab</span>}
              </div>
              <div className="small text-muted mb-2">{p.name} · {p.age}y · {p.disease}</div>
              <div className="mb-1"><SlopeArrow slope={p.slope}/> <span className="small text-muted ms-1">Function: {p.latest_function_rating}/10</span></div>
              <div className="small">
                <span className={p.latest_phq9>10?'text-danger':'text-muted'}>PHQ-9: {p.latest_phq9}</span>{' · '}
                <span className={p.latest_moca<25?'text-danger':'text-muted'}>MoCA: {p.latest_moca}</span>{' · '}
                <span className={p.latest_fatigue>6?'text-warning':'text-muted'}>Fatigue: {p.latest_fatigue}</span>
              </div>
              {p.risk_factors.length>0 && <div className="mt-1">{p.risk_factors.map(r=><span key={r} className="badge bg-warning text-dark me-1 small">{r}</span>)}</div>}
            </div></div>
          </div>
        )}
      </div>

      {/* Domain Averages by Trajectory */}
      {bd.domain_by_trajectory && <div className="card shadow-sm mt-3"><div className="card-body">
        <h6>Domain Averages by Trajectory Class</h6>
        <table className="table table-sm table-bordered">
          <thead><tr><th>Trajectory</th><th>Avg Daily Function</th><th>Avg Social Function</th><th>Avg Cognitive (MoCA)</th></tr></thead>
          <tbody>
            {bd.domain_by_trajectory.map(d=>
              <tr key={d.trajectory_class}>
                <td><Badge val={d.trajectory_class}/></td>
                <td>{d.avg_daily_function?.toFixed(2)}</td>
                <td>{d.avg_social_function?.toFixed(2)}</td>
                <td>{d.avg_cognitive_moca?.toFixed(1)}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>}
    </div>}

    {/* ── Prediction Factors Tab ── */}
    {tab==='predictions' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Prediction Factors for Functional Decline</h6>
        <p className="text-muted small">Variables correlated with declining recovery trajectory (negative slope).</p>
        <table className="table table-sm table-bordered">
          <thead className="table-dark"><tr><th>Variable</th><th>Correlation w/ Decline</th><th>Direction</th><th>Strength</th></tr></thead>
          <tbody>
            {bd.prediction_factors.map(f=>{
              const abs = Math.abs(f.correlation_with_decline);
              const strength = abs>=0.5?'Strong':abs>=0.3?'Moderate':'Weak';
              const color = abs>=0.5?'danger':abs>=0.3?'warning':'secondary';
              return <tr key={f.variable}>
                <td className="fw-bold">{f.variable}</td>
                <td><span className={`text-${f.correlation_with_decline<0?'success':'danger'} fw-bold`}>
                  {f.correlation_with_decline>0?'+':''}{f.correlation_with_decline.toFixed(3)}
                </span></td>
                <td>{f.direction}</td>
                <td><span className={`badge bg-${color}`}>{strength}</span></td>
              </tr>;
            })}
          </tbody>
        </table>
      </div></div>
    </div>}

    {/* ── Definitions Tab ── */}
    {tab==='definitions' && defs && <div>
      {/* Metric Definitions */}
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Metric Definitions</h6>
        <dl>
          {Object.entries(defs.metric_definitions||{}).map(([k,v])=>
            <div key={k} className="mb-2">
              <dt className="text-primary">{k.replace(/_/g,' ')}</dt>
              <dd className="small">{v}</dd>
            </div>
          )}
        </dl>
      </div></div>

      {/* Rehab Intensity Criteria */}
      {defs.rehab_intensity_criteria && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Rehab Intensity Criteria</h6>
        {Array.isArray(defs.rehab_intensity_criteria) ?
          <ul>{defs.rehab_intensity_criteria.map((c,i)=><li key={i} className="small">{typeof c==='string'?c:JSON.stringify(c)}</li>)}</ul>
          : <pre className="small">{JSON.stringify(defs.rehab_intensity_criteria, null, 2)}</pre>
        }
      </div></div>}

      {/* Functional Rating Scales */}
      {defs.functional_rating_scales && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Functional Rating Scales</h6>
        {Array.isArray(defs.functional_rating_scales) ?
          <ul>{defs.functional_rating_scales.map((s,i)=><li key={i} className="small">{typeof s==='string'?s:JSON.stringify(s)}</li>)}</ul>
          : <pre className="small">{JSON.stringify(defs.functional_rating_scales, null, 2)}</pre>
        }
      </div></div>}

      {/* Glossary */}
      {defs.glossary && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Glossary</h6>
        <dl>
          {Object.entries(defs.glossary).map(([k,v])=>
            <div key={k} className="mb-1">
              <dt className="small text-info d-inline">{k}: </dt>
              <dd className="small d-inline">{v}</dd>
            </div>
          )}
        </dl>
      </div></div>}
    </div>}
  </div>);
}
