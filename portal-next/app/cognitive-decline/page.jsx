'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function ClassBadge({cls, color}){
  return <span className="badge me-1" style={{backgroundColor: color||'#6b7280'}}>{cls}</span>;
}

function SlopeArrow({slope}){
  if(slope > 0.5) return <span className="text-success fw-bold">▲ +{slope.toFixed(1)}</span>;
  if(slope < -0.5) return <span className="text-danger fw-bold">▼ {slope.toFixed(1)}</span>;
  return <span className="text-warning fw-bold">● {slope.toFixed(1)}</span>;
}

function MiniBar({value, max=100, color='#3b82f6'}){
  const pct = Math.min(100, Math.max(0, (value/max)*100));
  return <div className="progress" style={{height:'8px', minWidth:'80px'}}>
    <div className="progress-bar" style={{width:`${pct}%`, backgroundColor: color}}></div>
  </div>;
}

export default function CognitiveDeclineDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [selectedPt, setSelectedPt] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/cognitive-decline/overview`).then(r=>r.json()),
      fetch(`${API}/api/cognitive-decline/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/cognitive-decline/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading cognitive decline data…</div>;

  const TABS = [
    {id:'overview', label:'📊 Overview'},
    {id:'patients', label:'👤 Patient Table'},
    {id:'flagged', label:'🚨 Flagged'},
    {id:'detail', label:'🔍 Patient Detail'},
    {id:'definitions', label:'📖 Definitions'},
  ];

  const kpi = ov.kpis;

  return (<div className="p-3">
    <h3>🧠 Cognitive Decline Tracking Dashboard</h3>
    <p className="text-muted">Longitudinal cognitive monitoring — {kpi.total_patients} patients, {kpi.total_assessments} assessments, {kpi.decline_rate_pct}% showing decline</p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* ── Overview Tab ── */}
    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          ['Total Patients', kpi.total_patients, '#3b82f6'],
          ['Assessments', kpi.total_assessments, '#6366f1'],
          ['Decline Rate', `${kpi.decline_rate_pct}%`, '#ef4444'],
          ['With Decline', kpi.patients_with_decline, '#f97316'],
          ['Avg Follow-up', `${kpi.avg_follow_up_months} mo`, '#22c55e'],
        ].map(([k,v,c])=>
          <div key={k} className="col-6 col-md mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className="h5 mb-0" style={{color: c}}>{v}</div>
              <div className="text-muted small">{k}</div>
            </div></div>
          </div>
        )}
      </div>

      {/* Classification Distribution */}
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Classification Distribution</h6>
            <table className="table table-sm">
              <thead><tr><th>Classification</th><th>Count</th><th>Bar</th></tr></thead>
              <tbody>{(ov.classification_distribution||[]).map(c=>
                <tr key={c.classification}>
                  <td><ClassBadge cls={c.label} color={c.color}/></td>
                  <td>{c.count}</td>
                  <td><MiniBar value={c.count} max={kpi.total_patients} color={c.color}/></td>
                </tr>
              )}</tbody>
            </table>
          </div></div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Risk Stratification</h6>
            <table className="table table-sm">
              <thead><tr><th>Risk Level</th><th>Count</th><th>Bar</th></tr></thead>
              <tbody>{(ov.risk_stratification||[]).map(r=>
                <tr key={r.level}>
                  <td><span className="badge me-1" style={{backgroundColor: r.color}}>{r.level}</span></td>
                  <td>{r.count}</td>
                  <td><MiniBar value={r.count} max={kpi.total_patients} color={r.color}/></td>
                </tr>
              )}</tbody>
            </table>
          </div></div>
        </div>
      </div>

      {/* Domain Decline Rates */}
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Domain Decline Rates</h6>
            <table className="table table-sm">
              <thead><tr><th>Domain</th><th>Decline %</th><th>Bar</th></tr></thead>
              <tbody>{(ov.domain_decline_rates||[]).map(d=>
                <tr key={d.domain}>
                  <td className="fw-bold">{d.domain}</td>
                  <td>{d.rate_pct}%</td>
                  <td><MiniBar value={d.rate_pct} max={100} color={d.rate_pct>40?'#ef4444':'#f59e0b'}/></td>
                </tr>
              )}</tbody>
            </table>
          </div></div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>MoCA Trend (Average)</h6>
            <table className="table table-sm">
              <thead><tr><th>Timepoint</th><th>Avg MoCA</th><th>Bar</th></tr></thead>
              <tbody>{(ov.moca_trend||[]).map(t=>
                <tr key={t.timepoint}>
                  <td>T{t.timepoint}</td>
                  <td>{t.avg_moca?.toFixed(1)}</td>
                  <td><MiniBar value={t.avg_moca} max={30} color={t.avg_moca>=26?'#22c55e':t.avg_moca>=22?'#f59e0b':'#ef4444'}/></td>
                </tr>
              )}</tbody>
            </table>
          </div></div>
        </div>
      </div>
    </div>}

    {/* ── Patient Table Tab ── */}
    {tab==='patients' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>All Patients ({bd.patients?.length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead><tr>
              <th>Patient</th><th>Age</th><th>Timepoints</th>
              <th>Baseline MoCA</th><th>Latest MoCA</th><th>Change</th>
              <th>Slope</th><th>Classification</th><th>Risk</th><th>Detail</th>
            </tr></thead>
            <tbody>{(bd.patients||[]).map(p=>
              <tr key={p.patient_id}>
                <td className="fw-bold">{p.patient_id}</td>
                <td>{p.age}</td>
                <td>{p.n_timepoints}</td>
                <td>{p.baseline_moca?.toFixed(1)}</td>
                <td>{p.latest_moca?.toFixed(1)}</td>
                <td><SlopeArrow slope={p.moca_change}/></td>
                <td><SlopeArrow slope={p.moca_slope}/></td>
                <td><ClassBadge cls={p.classification_label} color={p.classification_color}/></td>
                <td><span className="badge" style={{backgroundColor: p.risk_color}}>{p.risk_level}</span></td>
                <td><button className="btn btn-sm btn-outline-primary" onClick={()=>{setSelectedPt(p);setTab('detail');}}>View</button></td>
              </tr>
            )}</tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Flagged Tab ── */}
    {tab==='flagged' && bd && <div>
      <div className="card shadow-sm border-danger"><div className="card-body">
        <h6 className="text-danger">Flagged Patients — Declining ({bd.flagged_patients?.length})</h6>
        <p className="text-muted small">Patients with significant or moderate cognitive decline requiring clinical attention</p>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead><tr>
              <th>Patient</th><th>Age</th><th>Follow-up</th>
              <th>Baseline → Latest MoCA</th><th>Slope</th>
              <th>Memory</th><th>Attention</th><th>Executive</th><th>Processing</th>
              <th>Detail</th>
            </tr></thead>
            <tbody>{(bd.flagged_patients||[]).map(p=>
              <tr key={p.patient_id}>
                <td className="fw-bold">{p.patient_id}</td>
                <td>{p.age}</td>
                <td>{p.follow_up_months} mo</td>
                <td>{p.baseline_moca?.toFixed(1)} → {p.latest_moca?.toFixed(1)}</td>
                <td><SlopeArrow slope={p.moca_slope}/></td>
                <td><SlopeArrow slope={p.domain_slopes?.memory}/></td>
                <td><SlopeArrow slope={p.domain_slopes?.attention}/></td>
                <td><SlopeArrow slope={p.domain_slopes?.executive}/></td>
                <td><SlopeArrow slope={p.domain_slopes?.processing_speed}/></td>
                <td><button className="btn btn-sm btn-outline-danger" onClick={()=>{setSelectedPt(p);setTab('detail');}}>View</button></td>
              </tr>
            )}</tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Patient Detail Tab ── */}
    {tab==='detail' && <div>
      {!selectedPt ? <div className="alert alert-info">Select a patient from the Patient Table or Flagged tab to view details.</div> :
      <div>
        <div className="card shadow-sm mb-3"><div className="card-body">
          <div className="d-flex align-items-center mb-2">
            <h5 className="mb-0 me-3">{selectedPt.patient_id}</h5>
            <ClassBadge cls={selectedPt.classification_label} color={selectedPt.classification_color}/>
            <span className="badge ms-1" style={{backgroundColor: selectedPt.risk_color}}>Risk: {selectedPt.risk_level}</span>
          </div>
          <div className="row">
            <div className="col-md-3"><strong>Age:</strong> {selectedPt.age}</div>
            <div className="col-md-3"><strong>Timepoints:</strong> {selectedPt.n_timepoints}</div>
            <div className="col-md-3"><strong>Follow-up:</strong> {selectedPt.follow_up_months} months</div>
            <div className="col-md-3"><strong>MoCA Slope:</strong> <SlopeArrow slope={selectedPt.moca_slope}/></div>
          </div>
        </div></div>

        {/* Domain Slopes */}
        <div className="card shadow-sm mb-3"><div className="card-body">
          <h6>Domain-Level Slopes (per month)</h6>
          <div className="row">
            {Object.entries(selectedPt.domain_slopes||{}).map(([domain, slope])=>
              <div key={domain} className="col-md-3 mb-2">
                <div className="card"><div className="card-body text-center py-2">
                  <div className="text-muted small text-capitalize">{domain.replace('_',' ')}</div>
                  <SlopeArrow slope={slope}/>
                </div></div>
              </div>
            )}
          </div>
        </div></div>

        {/* Longitudinal Assessments */}
        <div className="card shadow-sm"><div className="card-body">
          <h6>Longitudinal Assessments</h6>
          <div className="table-responsive">
            <table className="table table-sm">
              <thead><tr>
                <th>Date</th><th>Months</th><th>MoCA</th><th>MMSE</th>
                <th>Memory</th><th>Attention</th><th>Executive</th><th>Processing</th><th>Notes</th>
              </tr></thead>
              <tbody>{(selectedPt.timepoints||[]).map((tp,i)=>
                <tr key={i}>
                  <td>{tp.date}</td>
                  <td>{tp.months?.toFixed(1)}</td>
                  <td className="fw-bold">{tp.moca?.toFixed(1)}</td>
                  <td>{tp.mmse?.toFixed(1)}</td>
                  <td>{tp.memory?.toFixed(0)}</td>
                  <td>{tp.attention?.toFixed(0)}</td>
                  <td>{tp.executive?.toFixed(0)}</td>
                  <td>{tp.processing_speed?.toFixed(0)}</td>
                  <td className="text-muted small">{tp.notes}</td>
                </tr>
              )}</tbody>
            </table>
          </div>
        </div></div>
      </div>}
    </div>}

    {/* ── Definitions Tab ── */}
    {tab==='definitions' && defs && <div>
      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Decline Thresholds</h6>
            <table className="table table-sm">
              <thead><tr><th>Threshold</th><th>Value</th></tr></thead>
              <tbody>{Object.entries(defs.decline_thresholds||{}).map(([k,v])=>
                <tr key={k}><td className="text-capitalize">{k.replace(/_/g,' ')}</td><td>{typeof v==='object'?JSON.stringify(v):String(v)}</td></tr>
              )}</tbody>
            </table>
          </div></div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Assessment Instruments</h6>
            <table className="table table-sm">
              <thead><tr><th>Instrument</th><th>Details</th></tr></thead>
              <tbody>{(Array.isArray(defs.assessment_instruments)?defs.assessment_instruments:Object.entries(defs.assessment_instruments||{})).map((item,i)=>{
                const [k,v] = Array.isArray(item)?[item.name||i, item]:[item[0],item[1]];
                return <tr key={i}><td className="fw-bold">{typeof k==='string'?k.replace(/_/g,' '):k}</td><td>{typeof v==='object'?JSON.stringify(v):String(v)}</td></tr>;
              })}</tbody>
            </table>
          </div></div>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Risk Factors</h6>
            <ul className="list-group list-group-flush">
              {(Array.isArray(defs.risk_factors)?defs.risk_factors:Object.entries(defs.risk_factors||{})).map((item,i)=>
                <li key={i} className="list-group-item">{typeof item==='string'?item:Array.isArray(item)?`${item[0]}: ${JSON.stringify(item[1])}`:JSON.stringify(item)}</li>
              )}
            </ul>
          </div></div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Clinical Actions</h6>
            <ul className="list-group list-group-flush">
              {(Array.isArray(defs.clinical_actions)?defs.clinical_actions:Object.entries(defs.clinical_actions||{})).map((item,i)=>
                <li key={i} className="list-group-item">{typeof item==='string'?item:Array.isArray(item)?`${item[0]}: ${JSON.stringify(item[1])}`:JSON.stringify(item)}</li>
              )}
            </ul>
          </div></div>
        </div>
      </div>

      {defs.glossary && Object.keys(defs.glossary).length > 0 && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Glossary</h6>
        <table className="table table-sm">
          <thead><tr><th>Term</th><th>Definition</th></tr></thead>
          <tbody>{Object.entries(defs.glossary).map(([k,v])=>
            <tr key={k}><td className="fw-bold">{k}</td><td>{v}</td></tr>
          )}</tbody>
        </table>
      </div></div>}
    </div>}
  </div>);
}
