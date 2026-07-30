'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SEV_COLORS = {
  Good:'#22c55e', Moderate:'#f59e0b', Poor:'#ef4444', 'Very Poor':'#991b1b',
  Minimal:'#22c55e', Mild:'#a3e635', 'Moderately Severe':'#dc2626', Severe:'#991b1b',
  'No depression':'#22c55e', 'Possible MDD':'#f59e0b', 'Probable MDD':'#ef4444',
  Low:'#22c55e', High:'#ef4444',
};

function Badge({label}){
  return <span className="badge" style={{backgroundColor: SEV_COLORS[label]||'#6b7280'}}>{label}</span>;
}

function MiniBar({value, max=100, color='#3b82f6'}){
  const pct = Math.min(100, Math.max(0, (value/max)*100));
  return <div className="progress" style={{height:'8px', minWidth:'80px'}}>
    <div className="progress-bar" style={{width:`${pct}%`, backgroundColor: color}}></div>
  </div>;
}

export default function QualityOfLifeDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [selectedPt, setSelectedPt] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/quality-of-life/overview`).then(r=>r.json()),
      fetch(`${API}/api/quality-of-life/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/quality-of-life/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading Quality of Life data…</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'patients', label:'Patient Profiles'},
    {id:'trends', label:'Trends'},
    {id:'recent', label:'Recent Assessments'},
    {id:'definitions', label:'Definitions'},
  ];

  const da = ov.domain_averages||{};

  return (<div className="p-3">
    <h3>Quality of Life Dashboard</h3>
    <p className="text-muted">{ov.total_assessments} PRO assessments across {ov.patients_assessed} patients — {ov.domains_tracked} domains tracked — {ov.patients_at_risk} at-risk</p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* ── Overview Tab ── */}
    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          ['Total PRO Assessments', ov.total_assessments, '#3b82f6'],
          ['Patients Assessed', ov.patients_assessed, '#6366f1'],
          ['Coverage', `${ov.coverage_pct}%`, '#22c55e'],
          ['At-Risk Patients', ov.patients_at_risk, '#ef4444'],
          ['Domains Tracked', ov.domains_tracked, '#f59e0b'],
        ].map(([k,v,c])=>
          <div key={k} className="col-6 col-md mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className="h5 mb-0" style={{color: c}}>{v}</div>
              <div className="text-muted small">{k}</div>
            </div></div>
          </div>
        )}
      </div>

      {/* Domain Averages */}
      <div className="row mb-3">
        {[
          ['QOLIE-31', da.qolie31, 100, 'Higher = better QoL'],
          ['PHQ-9', da.phq9, 27, 'Lower = less depression'],
          ['GAD-7', da.gad7, 21, 'Lower = less anxiety'],
          ['NDDI-E', da.nddi_e, 24, 'Lower = less depression'],
          ['Mood', da.mood, 10, 'Higher = better'],
          ['Fatigue', da.fatigue, 10, 'Lower = less fatigue'],
          ['Social Function', da.social_function, 10, 'Higher = better'],
          ['Daily Function', da.daily_function, 10, 'Higher = better'],
          ['Seizure Worry', da.seizure_worry, 10, 'Lower = less worry'],
        ].map(([label, val, max, hint])=>
          <div key={label} className="col-6 col-md-4 col-lg-3 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body py-2">
              <div className="d-flex justify-content-between align-items-center mb-1">
                <span className="fw-bold small">{label}</span>
                <span className="h5 mb-0" style={{color:'#3b82f6'}}>{val ?? '—'}</span>
              </div>
              <MiniBar value={val??0} max={max} color={
                label==='QOLIE-31' ? (val>=70?'#22c55e':val>=50?'#f59e0b':'#ef4444') :
                ['PHQ-9','GAD-7','NDDI-E','Fatigue','Seizure Worry'].includes(label) ? (val<=5?'#22c55e':val<=10?'#f59e0b':'#ef4444') :
                (val>=7?'#22c55e':val>=4?'#f59e0b':'#ef4444')
              }/>
              <div className="text-muted" style={{fontSize:'0.7rem'}}>{hint}</div>
            </div></div>
          </div>
        )}
      </div>

      {/* Severity Distributions */}
      <div className="row">
        {Object.entries(ov.severity_distributions||{}).map(([domain, dist])=>
          <div key={domain} className="col-md-4 mb-3">
            <div className="card shadow-sm h-100"><div className="card-body">
              <h6>{domain.toUpperCase().replace('_',' ')}</h6>
              {Object.entries(dist).map(([cat, cnt])=>
                <div key={cat} className="d-flex justify-content-between align-items-center mb-1">
                  <Badge label={cat}/>
                  <span className="small">{cnt}</span>
                </div>
              )}
            </div></div>
          </div>
        )}
      </div>
    </div>}

    {/* ── Patient Profiles Tab ── */}
    {tab==='patients' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Patient QoL Profiles ({bd.profiles?.length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead><tr>
              <th>Patient</th><th>Assessments</th><th>QOLIE-31</th><th>PHQ-9</th>
              <th>GAD-7</th><th>NDDI-E</th><th>Mood</th><th>Seizure Worry</th><th>Detail</th>
            </tr></thead>
            <tbody>{(bd.profiles||[]).map(p=>
              <tr key={p.patient_id}>
                <td className="fw-bold">{p.patient_id}</td>
                <td>{p.assessments}</td>
                <td>{p.qolie31 ?? '—'} <Badge label={p.qolie31_category}/></td>
                <td>{p.phq9 ?? '—'} <Badge label={p.phq9_category}/></td>
                <td>{p.gad7 ?? '—'} <Badge label={p.gad7_category}/></td>
                <td>{p.nddi_e ?? '—'} <Badge label={p.nddi_e_category}/></td>
                <td>{p.mood ?? '—'}/10</td>
                <td>{p.seizure_worry ?? '—'} <Badge label={p.seizure_worry_level}/></td>
                <td><button className="btn btn-sm btn-outline-primary" onClick={()=>{setSelectedPt(p);setTab('detail');}}>View</button></td>
              </tr>
            )}</tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Patient Detail (hidden tab) ── */}
    {tab==='detail' && <div>
      {!selectedPt ? <div className="alert alert-info">Select a patient from the Patient Profiles tab.</div> :
      <div>
        <button className="btn btn-sm btn-outline-secondary mb-3" onClick={()=>setTab('patients')}>← Back to Profiles</button>
        <div className="card shadow-sm mb-3"><div className="card-body">
          <h5 className="mb-1">{selectedPt.patient_id}</h5>
          <span className="text-muted small">{selectedPt.assessments} assessments — latest: {selectedPt.latest_date}</span>
        </div></div>

        <div className="row">
          {[
            ['QOLIE-31', selectedPt.qolie31, 100, selectedPt.qolie31_category],
            ['PHQ-9', selectedPt.phq9, 27, selectedPt.phq9_category],
            ['GAD-7', selectedPt.gad7, 21, selectedPt.gad7_category],
            ['NDDI-E', selectedPt.nddi_e, 24, selectedPt.nddi_e_category],
            ['Mood', selectedPt.mood, 10, null],
            ['Fatigue', selectedPt.fatigue, 10, null],
            ['Social Function', selectedPt.social_function, 10, null],
            ['Daily Function', selectedPt.daily_function, 10, null],
            ['Seizure Worry', selectedPt.seizure_worry, 10, selectedPt.seizure_worry_level],
          ].map(([label, val, max, cat])=>
            <div key={label} className="col-md-4 mb-3">
              <div className="card shadow-sm h-100"><div className="card-body">
                <h6>{label}</h6>
                <div className="h3 mb-1" style={{color: SEV_COLORS[cat]||'#3b82f6'}}>{val ?? '—'}<span className="text-muted small">/{max}</span></div>
                <MiniBar value={val??0} max={max}/>
                {cat && <div className="mt-1"><Badge label={cat}/></div>}
              </div></div>
            </div>
          )}
        </div>

        {/* Longitudinal trend */}
        {bd?.trends?.[selectedPt.patient_id] && <div className="card shadow-sm mt-3"><div className="card-body">
          <h6>Longitudinal Trend</h6>
          <div className="table-responsive">
            <table className="table table-sm">
              <thead><tr><th>Date</th><th>QOLIE-31</th><th>PHQ-9</th><th>GAD-7</th><th>Mood</th><th>Fatigue</th></tr></thead>
              <tbody>{bd.trends[selectedPt.patient_id].map((t,i)=>
                <tr key={i}>
                  <td>{t.date}</td>
                  <td>{t.qolie31 ?? '—'}</td>
                  <td>{t.phq9 ?? '—'}</td>
                  <td>{t.gad7 ?? '—'}</td>
                  <td>{t.mood ?? '—'}</td>
                  <td>{t.fatigue ?? '—'}</td>
                </tr>
              )}</tbody>
            </table>
          </div>
        </div></div>}
      </div>}
    </div>}

    {/* ── Trends Tab ── */}
    {tab==='trends' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Longitudinal Trends (all patients)</h6>
        <p className="text-muted small">Select a patient from Profiles → View for individual trends</p>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Patient</th><th>Dates</th><th>QOLIE-31 Range</th><th>PHQ-9 Range</th><th>Trend Direction</th></tr></thead>
            <tbody>{Object.entries(bd.trends||{}).map(([pid, pts])=>{
              const q = pts.map(p=>p.qolie31).filter(v=>v!=null);
              const p9 = pts.map(p=>p.phq9).filter(v=>v!=null);
              const qTrend = q.length>=2 ? (q[q.length-1]>q[0]?'Improving':'Declining') : 'N/A';
              return <tr key={pid}>
                <td className="fw-bold">{pid}</td>
                <td className="small">{pts[0]?.date} → {pts[pts.length-1]?.date}</td>
                <td>{q.length?`${Math.min(...q)}–${Math.max(...q)}`:'—'}</td>
                <td>{p9.length?`${Math.min(...p9)}–${Math.max(...p9)}`:'—'}</td>
                <td><span style={{color: qTrend==='Improving'?'#22c55e':qTrend==='Declining'?'#ef4444':'#6b7280'}}>{qTrend}</span></td>
              </tr>;
            })}</tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Recent Assessments Tab ── */}
    {tab==='recent' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Recent PRO Assessments ({bd.recent_assessments?.length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead><tr><th>Patient</th><th>Date</th><th>QOLIE-31</th><th>PHQ-9</th><th>GAD-7</th><th>Mood</th><th>Notes</th></tr></thead>
            <tbody>{(bd.recent_assessments||[]).map((a,i)=>
              <tr key={i}>
                <td className="fw-bold">{a.patient_id}</td>
                <td className="small">{a.date}</td>
                <td>{a.qolie31 ?? '—'}</td>
                <td>{a.phq9 ?? '—'}</td>
                <td>{a.gad7 ?? '—'}</td>
                <td>{a.mood ?? '—'}</td>
                <td className="text-muted small">{a.notes||'—'}</td>
              </tr>
            )}</tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Definitions Tab ── */}
    {tab==='definitions' && defs && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>QoL Instruments</h6>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Instrument</th><th>Full Name</th><th>Range</th><th>Interpretation</th><th>Categories</th></tr></thead>
            <tbody>{(defs.instruments||[]).map(inst=>
              <tr key={inst.name}>
                <td className="fw-bold">{inst.name}</td>
                <td>{inst.full_name}</td>
                <td>{inst.range}</td>
                <td className="small">{inst.interpretation}</td>
                <td className="small">{Object.entries(inst.categories||{}).map(([k,v])=>
                  <div key={k}><Badge label={k}/> {v}</div>
                )}</td>
              </tr>
            )}</tbody>
          </table>
        </div>
      </div></div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Functional Domains</h6>
        <table className="table table-sm">
          <thead><tr><th>Domain</th><th>Range</th><th>Description</th></tr></thead>
          <tbody>{(defs.functional_domains||[]).map(d=>
            <tr key={d.name}><td className="fw-bold">{d.name}</td><td>{d.range}</td><td>{d.description}</td></tr>
          )}</tbody>
        </table>
      </div></div>

      <div className="card shadow-sm"><div className="card-body">
        <h6>Glossary</h6>
        <table className="table table-sm">
          <thead><tr><th>Term</th><th>Definition</th></tr></thead>
          <tbody>{Object.entries(defs.glossary||{}).map(([k,v])=>
            <tr key={k}><td className="fw-bold">{k}</td><td>{v}</td></tr>
          )}</tbody>
        </table>
      </div></div>
    </div>}
  </div>);
}
