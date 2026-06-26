'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8000';

function Badge({severity}){
  const m={high:'danger',medium:'warning',low:'info',none:'success',large:'danger',small:'warning',no_drift:'success'};
  return <span className={`badge bg-${m[severity]||'secondary'}`}>{severity}</span>;
}

export default function DriftDashboard(){
  const [overview, setOverview] = useState(null);
  const [features, setFeatures] = useState(null);
  const [severity, setSeverity] = useState(null);
  const [alerts, setAlerts] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/drift/dashboard`).then(r=>r.json()),
      fetch(`${API}/api/drift/features?limit=47`).then(r=>r.json()),
      fetch(`${API}/api/drift/severity`).then(r=>r.json()),
      fetch(`${API}/api/drift/alerts`).then(r=>r.json()),
      fetch(`${API}/api/drift/definitions`).then(r=>r.json()),
    ]).then(([o,f,s,a,d])=>{setOverview(o);setFeatures(f);setSeverity(s);setAlerts(a);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger">Failed to load drift data: {err}</div>;
  if(!overview) return <div className="text-muted p-3">Loading drift dashboard…</div>;
  if(!overview.available) return <div className="alert alert-warning">{overview.note || 'No drift data. Run scripts/drift_job.py'}</div>;

  const TABS = [
    {id:'overview', label:'📊 Overview'},
    {id:'features', label:'📋 Feature Table'},
    {id:'alerts', label:'🚨 Alerts'},
    {id:'definitions', label:'📖 Definitions'},
  ];

  return (<div>
    <h3>📉 Drift Monitoring Dashboard</h3>
    <p className="text-muted">PSI + KS-test: training-reference vs live-extractor features. Detects train/serve skew.</p>

    {/* KPI Hero Tiles */}
    <div className="row mb-3">
      {[
        ['Verdict', overview.verdict, overview.verdict?.includes('SEVERE')?'danger':overview.verdict?.includes('MODERATE')?'warning':'success'],
        ['Features Drifted', `${(overview.frac_drifted*100).toFixed(0)}%`, overview.frac_drifted>0.5?'danger':'success'],
        ['High Severity', overview.severity_breakdown?.high || 0, (overview.severity_breakdown?.high||0)>5?'danger':'success'],
        ['Reference / Live', `${overview.n_reference} / ${overview.n_live}`, 'primary'],
      ].map(([k,v,c])=>
        <div key={k} className="col-6 col-md-3 mb-2">
          <div className="card shadow-sm h-100"><div className="card-body text-center">
            <div className={`h4 mb-1 text-${c}`}>{v}</div>
            <div className="text-muted small">{k}</div>
          </div></div>
        </div>
      )}
    </div>

    {/* Tab Nav */}
    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* Overview Tab */}
    {tab==='overview' && <div>
      <div className="row">
        <div className="col-md-6">
          <div className="card mb-3"><div className="card-body">
            <h5>Severity Distribution</h5>
            {severity?.labels?.map((l,i)=>
              <div key={l} className="d-flex justify-content-between align-items-center mb-2">
                <span><Badge severity={l}/> {l}</span>
                <div className="d-flex align-items-center">
                  <div style={{width:200,height:20,background:'#e9ecef',borderRadius:4,overflow:'hidden',marginRight:8}}>
                    <div style={{width:`${(severity.values[i]/(overview.n_features||1))*100}%`,height:'100%',background:severity.colors?.[i]||'#6c757d',borderRadius:4}}/>
                  </div>
                  <strong>{severity.values[i]}</strong>
                </div>
              </div>
            )}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card mb-3"><div className="card-body">
            <h5>Summary</h5>
            <table className="table table-sm mb-0">
              <tbody>
                <tr><td className="text-muted">Disease</td><td><strong>{overview.disease}</strong></td></tr>
                <tr><td className="text-muted">Total features</td><td>{overview.n_features}</td></tr>
                <tr><td className="text-muted">High-drift features</td><td className="text-danger fw-bold">{overview.n_high_drift}</td></tr>
                <tr><td className="text-muted">Last run</td><td>{overview.last_run}</td></tr>
                <tr><td className="text-muted">Method</td><td className="small">{overview.method}</td></tr>
              </tbody>
            </table>
          </div></div>
          <div className="alert alert-warning small">{overview.recommendation}</div>
        </div>
      </div>
    </div>}

    {/* Feature Table Tab */}
    {tab==='features' && features?.available && <div>
      <p className="text-muted small">Top {features.features?.length || 0} features by PSI (Population Stability Index). KS = Kolmogorov-Smirnov.</p>
      <div className="table-responsive">
        <table className="table table-striped table-sm">
          <thead><tr><th>#</th><th>Feature</th><th>PSI</th><th>PSI Band</th><th>KS Stat</th><th>KS p-value</th><th>Severity</th></tr></thead>
          <tbody>{features.features?.map((f,i)=>
            <tr key={f.feature}>
              <td>{i+1}</td>
              <td><code>{f.feature}</code></td>
              <td className="fw-bold">{f.psi?.toFixed(3)}</td>
              <td><Badge severity={f.psi_band}/></td>
              <td>{f.ks_stat?.toFixed(4)}</td>
              <td>{f.ks_p < 0.001 ? '<0.001' : f.ks_p?.toFixed(4)}</td>
              <td><Badge severity={f.severity}/></td>
            </tr>
          )}</tbody>
        </table>
      </div>
    </div>}

    {/* Alerts Tab */}
    {tab==='alerts' && <div>
      {alerts?.available && alerts.alerts?.length > 0 ? (
        <div>
          <p className="text-muted small">{alerts.count} features exceeding PSI threshold {alerts.psi_threshold}.</p>
          {alerts.alerts?.map((a,i)=>
            <div key={a.feature} className="card mb-2 border-start border-3 border-danger">
              <div className="card-body py-2 d-flex justify-content-between align-items-center">
                <div>
                  <code className="me-2">{a.feature}</code>
                  <Badge severity={a.severity}/>
                  <span className="text-muted ms-2 small">PSI={a.psi?.toFixed(2)}</span>
                </div>
                <span className="small text-muted">{a.action}</span>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="alert alert-success">No drift alerts above threshold.</div>
      )}
    </div>}

    {/* Definitions Tab */}
    {tab==='definitions' && defs && <div>
      <h5>PSI Thresholds</h5>
      <table className="table table-sm">
        <thead><tr><th>Range</th><th>Band</th><th>Interpretation</th></tr></thead>
        <tbody>{defs.psi_thresholds?.map(t=>
          <tr key={t.band}><td>{t.range}</td><td><Badge severity={t.band}/></td><td>{t.interpretation}</td></tr>
        )}</tbody>
      </table>
      <h5 className="mt-3">Severity Levels</h5>
      <table className="table table-sm">
        <thead><tr><th>Level</th><th>Meaning</th></tr></thead>
        <tbody>{defs.severity_levels?.map(s=>
          <tr key={s.level}><td><Badge severity={s.level}/></td><td>{s.meaning}</td></tr>
        )}</tbody>
      </table>
      <h5 className="mt-3">KS-test Interpretation</h5>
      <p className="small">{defs.ks_test?.description}</p>
      <table className="table table-sm">
        <tbody>
          <tr><td className="text-muted">Null hypothesis</td><td>{defs.ks_test?.null_hypothesis}</td></tr>
          <tr><td className="text-muted">Reject when</td><td>{defs.ks_test?.reject_when}</td></tr>
          <tr><td className="text-muted">Range</td><td>{defs.ks_test?.range}</td></tr>
        </tbody>
      </table>
    </div>}

    <div className="alert alert-info small mt-3">
      Live data from <code>/api/drift/dashboard</code> + <code>/api/drift/features</code> + <code>/api/drift/severity</code> + <code>/api/drift/alerts</code>.
      Cron-driven (DRIFT cron). PSI+KS drift detection on {overview.n_features} features. Last run: {overview.last_run}.
    </div>
  </div>);
}
