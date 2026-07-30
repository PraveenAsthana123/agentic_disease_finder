'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const riskColor = r => r >= 0.7 ? 'danger' : r >= 0.3 ? 'warning' : 'success';
const statusColor = s => ({completed:'success',partial:'warning',failed:'danger',insufficient:'secondary'}[s?.toLowerCase()]||'info');
const pct = (n, total) => total ? ((n / total) * 100).toFixed(1) : '0.0';

export default function MultimodalFusionDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/multimodal-fusion/overview`).then(r=>r.json()),
      fetch(`${API}/api/multimodal-fusion/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/multimodal-fusion/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading multimodal fusion data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'methods', label:'Fusion Methods'},
    {id:'modalities', label:'Modality Coverage'},
    {id:'patients', label:'Per Patient'},
    {id:'sessions', label:'Recent Sessions'},
    {id:'definitions', label:'Definitions'},
  ];

  return (<div className="p-3">
    <h3>Multimodal Fusion Dashboard</h3>
    <p className="text-muted">
      Cross-modal AI fusion analysis &mdash; {ov.total_sessions} sessions, {ov.total_patients} patients,
      {' '}{ov.avg_modalities_per_session} avg modalities/session,
      {' '}{ov.completion_rate}% completion rate
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          ['Total Sessions', ov.total_sessions, 'primary'],
          ['Patients', ov.total_patients, 'info'],
          ['Completion', ov.completion_rate+'%', 'success'],
          ['Avg Risk', ov.avg_risk_score?.toFixed(2), 'warning'],
          ['Avg Confidence', ov.avg_confidence?.toFixed(2), 'primary'],
          ['Avg Concordance', ov.avg_concordance?.toFixed(2), 'info'],
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
            <h6>Risk Distribution</h6>
            {(ov.risk_distribution||[]).map(r=>{
              const p = pct(r.count, ov.total_sessions);
              return <div key={r.category} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'120px'}}>{r.category}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${r.category.includes('High')?'danger':r.category.includes('Low')?'success':'warning'}`}
                      style={{width:`${p}%`}}>{r.count} ({p}%)</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Session Status</h6>
            {(ov.status_distribution||[]).map(s=>{
              const p = pct(s.count, ov.total_sessions);
              return <div key={s.status} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'100px'}}>{s.status}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${statusColor(s.status)}`}
                      style={{width:`${p}%`}}>{s.count} ({p}%)</div>
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
            <h6>Modality Availability</h6>
            <table className="table table-sm table-striped mb-0">
              <thead><tr><th>Modality</th><th>Available</th><th>Missing</th><th>Rate</th></tr></thead>
              <tbody>
                {(ov.modality_availability||[]).map(m=>
                  <tr key={m.modality}>
                    <td className="fw-bold">{m.modality}</td>
                    <td>{m.available}</td>
                    <td className={m.missing>10?'text-danger':''}>{m.missing}</td>
                    <td><span className={`badge bg-${m.rate>=90?'success':m.rate>=70?'warning':'danger'}`}>{m.rate}%</span></td>
                  </tr>
                )}
              </tbody>
            </table>
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Fusion Methods Used</h6>
            {(ov.fusion_methods||[]).map(f=>{
              const p = pct(f.count, ov.total_sessions);
              return <div key={f.method} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'160px',textTransform:'capitalize'}}>{f.method.replace(/_/g,' ')}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className="progress-bar bg-primary" style={{width:`${p}%`}}>{f.count}</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
      </div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Predicted Subtypes</h6>
        <div className="row">
          {(ov.predicted_subtypes||[]).map(s=>
            <div key={s.subtype} className="col-4 col-md-2 mb-2 text-center">
              <div className="h5 text-primary mb-0">{s.count}</div>
              <div className="text-muted small" style={{textTransform:'capitalize'}}>{s.subtype.replace(/_/g,' ')}</div>
            </div>
          )}
        </div>
      </div></div>

      {(ov.monthly_trend||[]).length>0 && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Monthly Session Trend</h6>
        <div className="d-flex align-items-end" style={{height:'120px',gap:'4px'}}>
          {ov.monthly_trend.map(m=>{
            const maxS = Math.max(...ov.monthly_trend.map(x=>x.sessions));
            const h = maxS ? (m.sessions/maxS)*100 : 0;
            return <div key={m.month} className="d-flex flex-column align-items-center flex-grow-1">
              <small className="text-muted mb-1">{m.sessions}</small>
              <div className="bg-primary rounded" style={{width:'100%',maxWidth:'40px',height:`${h}%`,minHeight:'4px'}}/>
              <small className="text-muted mt-1">{m.month.slice(5)}</small>
            </div>;
          })}
        </div>
      </div></div>}
    </div>}

    {tab==='methods' && bd && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Fusion Method Comparison</h6>
        <table className="table table-sm table-striped mb-0">
          <thead><tr>
            <th>Method</th><th>Sessions</th><th>Avg Risk</th><th>Avg Confidence</th><th>Avg Concordance</th><th>Avg Time (s)</th>
          </tr></thead>
          <tbody>
            {(bd.method_comparison||[]).map(m=>
              <tr key={m.method}>
                <td className="fw-bold" style={{textTransform:'capitalize'}}>{m.method.replace(/_/g,' ')}</td>
                <td>{m.sessions}</td>
                <td><span className={`badge bg-${riskColor(m.avg_risk)}`}>{m.avg_risk.toFixed(2)}</span></td>
                <td>{m.avg_confidence.toFixed(2)}</td>
                <td>{m.avg_concordance.toFixed(2)}</td>
                <td>{m.avg_processing_sec.toFixed(1)}s</td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>
    </div>}

    {tab==='modalities' && bd && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Modality Co-occurrence Matrix</h6>
        {(bd.modality_cooccurrence||[]).length>0 && (() => {
          const cols = Object.keys(bd.modality_cooccurrence[0]).filter(k=>k!=='modality');
          return <table className="table table-sm table-bordered mb-0 text-center">
            <thead><tr><th></th>{cols.map(c=><th key={c}>{c}</th>)}</tr></thead>
            <tbody>
              {bd.modality_cooccurrence.map(row=>
                <tr key={row.modality}>
                  <td className="fw-bold">{row.modality}</td>
                  {cols.map(c=><td key={c} className={row[c]===row[row.modality]?'table-primary':''}>{row[c]}</td>)}
                </tr>
              )}
            </tbody>
          </table>;
        })()}
      </div></div>
    </div>}

    {tab==='patients' && bd && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Per-Patient Fusion Summary ({(bd.per_patient||[]).length} patients)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr>
              <th>Patient</th><th>Sessions</th><th>Completed</th><th>Completion %</th><th>Avg Risk</th><th>Avg Confidence</th><th>Avg Modalities</th>
            </tr></thead>
            <tbody>
              {(bd.per_patient||[]).map(p=>
                <tr key={p.patient_id}>
                  <td className="fw-bold">{p.patient_id}</td>
                  <td>{p.total_sessions}</td>
                  <td>{p.completed}</td>
                  <td><span className={`badge bg-${p.completion_rate>=80?'success':p.completion_rate>=50?'warning':'danger'}`}>{p.completion_rate}%</span></td>
                  <td><span className={`badge bg-${riskColor(p.avg_risk)}`}>{p.avg_risk.toFixed(2)}</span></td>
                  <td>{p.avg_confidence.toFixed(2)}</td>
                  <td>{p.avg_modalities.toFixed(1)}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='sessions' && bd && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Recent Fusion Sessions</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr>
              <th>Session</th><th>Patient</th><th>Date</th><th>Method</th><th>Modalities</th><th>Risk</th><th>Subtype</th><th>Confidence</th><th>Status</th>
            </tr></thead>
            <tbody>
              {(bd.recent_sessions||[]).slice(0,30).map(s=>
                <tr key={s.session_id}>
                  <td className="small">{s.session_id}</td>
                  <td>{s.patient_id}</td>
                  <td className="small">{s.date}</td>
                  <td style={{textTransform:'capitalize'}}>{(s.fusion_method||'').replace(/_/g,' ')}</td>
                  <td>{s.modalities}</td>
                  <td><span className={`badge bg-${riskColor(s.risk_score)}`}>{s.risk_score.toFixed(2)}</span></td>
                  <td style={{textTransform:'capitalize'}}>{(s.subtype||'').replace(/_/g,' ')}</td>
                  <td>{s.confidence.toFixed(2)}</td>
                  <td><span className={`badge bg-${statusColor(s.status)}`}>{s.status}</span></td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='definitions' && defs && <div>
      {defs.glossary && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Glossary</h6>
        <table className="table table-sm mb-0">
          <tbody>
            {Object.entries(defs.glossary).map(([term,defn])=>
              <tr key={term}><td className="fw-bold" style={{textTransform:'capitalize',width:'200px'}}>{term.replace(/_/g,' ')}</td><td>{defn}</td></tr>
            )}
          </tbody>
        </table>
      </div></div>}
      {defs.risk_tiers && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Risk Tiers</h6>
        <table className="table table-sm mb-0">
          <thead><tr><th>Tier</th><th>Range</th><th>Description</th></tr></thead>
          <tbody>
            {defs.risk_tiers.map(t=>
              <tr key={t.tier||t.label}><td className="fw-bold">{t.tier||t.label}</td><td>{t.range}</td><td>{t.description}</td></tr>
            )}
          </tbody>
        </table>
      </div></div>}
      {defs.modalities && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Modalities</h6>
        <table className="table table-sm mb-0">
          <thead><tr><th>Modality</th><th>Description</th></tr></thead>
          <tbody>
            {defs.modalities.map(m=>
              <tr key={m.name||m.modality}><td className="fw-bold">{m.name||m.modality}</td><td>{m.description}</td></tr>
            )}
          </tbody>
        </table>
      </div></div>}
    </div>}
  </div>);
}
