'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const LEVEL_COLORS = {normal:'#22c55e', moderate:'#f59e0b', severe:'#ef4444'};

function LevelBadge({level}){
  return <span className="badge" style={{backgroundColor: LEVEL_COLORS[level]||'#6b7280'}}>{level}</span>;
}

function MiniBar({value, max=100, color='#3b82f6'}){
  const pct = Math.min(100, Math.max(0, (value/max)*100));
  return <div className="progress" style={{height:'8px', minWidth:'80px'}}>
    <div className="progress-bar" style={{width:`${pct}%`, backgroundColor: color}}></div>
  </div>;
}

export default function ADLDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [selectedPt, setSelectedPt] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/adl/overview`).then(r=>r.json()),
      fetch(`${API}/api/adl/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/adl/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading ADL assessment data…</div>;

  const TABS = [
    {id:'overview', label:'📊 Overview'},
    {id:'patients', label:'👤 Patient Profiles'},
    {id:'distributions', label:'📈 Score Distributions'},
    {id:'recent', label:'🕐 Recent Assessments'},
    {id:'definitions', label:'📖 Definitions'},
  ];

  return (<div className="p-3">
    <h3>🏃 Activities of Daily Living Dashboard</h3>
    <p className="text-muted">{ov.total_assessments} assessments across {ov.patients_assessed} patients — {ov.instruments_tracked} instruments (Barthel Index, QOLIE-31, Epworth Sleepiness Scale)</p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* ── Overview Tab ── */}
    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          ['Total Assessments', ov.total_assessments, '#3b82f6'],
          ['Patients Assessed', ov.patients_assessed, '#6366f1'],
          ['Total Patients', ov.total_patients, '#8b5cf6'],
          ['Coverage', `${ov.coverage_pct}%`, '#22c55e'],
          ['Instruments', ov.instruments_tracked, '#f59e0b'],
        ].map(([k,v,c])=>
          <div key={k} className="col-6 col-md mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className="h5 mb-0" style={{color: c}}>{v}</div>
              <div className="text-muted small">{k}</div>
            </div></div>
          </div>
        )}
      </div>

      {/* Instrument Stats */}
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Instrument Statistics</h6>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Instrument</th><th>Label</th><th>Count</th><th>Patients</th><th>Avg Score</th><th>Range</th><th>Level Distribution</th></tr></thead>
            <tbody>{(ov.instrument_stats||[]).map(inst=>
              <tr key={inst.instrument}>
                <td className="fw-bold">{inst.instrument}</td>
                <td>{inst.label}</td>
                <td>{inst.count}</td>
                <td>{inst.unique_patients}</td>
                <td>{inst.avg_score?.toFixed(1)}</td>
                <td>{inst.min_score}–{inst.max_score}</td>
                <td>{Object.entries(inst.level_distribution||{}).map(([lev,cnt])=>
                  <span key={lev} className="me-2"><LevelBadge level={lev}/> {cnt}</span>
                )}</td>
              </tr>
            )}</tbody>
          </table>
        </div>
      </div></div>

      {/* Level summary cards */}
      <div className="row mb-3">
        {(ov.instrument_stats||[]).map(inst=>
          <div key={inst.instrument} className="col-md-4 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body">
              <h6>{inst.label}</h6>
              <div className="d-flex align-items-center mb-2">
                <span className="h4 mb-0 me-2" style={{color:'#3b82f6'}}>{inst.avg_score?.toFixed(1)}</span>
                <span className="text-muted small">avg / {inst.max_score} max</span>
              </div>
              <MiniBar value={inst.avg_score} max={inst.max_score} color={inst.avg_score/inst.max_score>0.7?'#22c55e':inst.avg_score/inst.max_score>0.4?'#f59e0b':'#ef4444'}/>
              <div className="mt-2">
                {Object.entries(inst.level_distribution||{}).map(([lev,cnt])=>
                  <div key={lev} className="d-flex justify-content-between small">
                    <span><LevelBadge level={lev}/></span>
                    <span>{cnt} patients</span>
                  </div>
                )}
              </div>
            </div></div>
          </div>
        )}
      </div>
    </div>}

    {/* ── Patient Profiles Tab ── */}
    {tab==='patients' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Patient ADL Profiles ({bd.patient_profiles?.length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead><tr>
              <th>Patient</th><th>Overall</th><th>Instruments</th>
              <th>Barthel</th><th>QOLIE-31</th><th>Epworth</th><th>Detail</th>
            </tr></thead>
            <tbody>{(bd.patient_profiles||[]).map(p=>
              <tr key={p.patient_id}>
                <td className="fw-bold">{p.patient_id}</td>
                <td><LevelBadge level={p.overall_level}/></td>
                <td>{p.instruments_completed}</td>
                <td>{p.scores?.BARTHEL ? <span>{p.scores.BARTHEL.score}/{p.scores.BARTHEL.max_score} <span className="text-muted small">({p.scores.BARTHEL.interpretation})</span></span> : '—'}</td>
                <td>{p.scores?.QOLIE31 ? <span>{p.scores.QOLIE31.score?.toFixed(1)} <span className="text-muted small">({p.scores.QOLIE31.interpretation})</span></span> : '—'}</td>
                <td>{p.scores?.EPWORTH ? <span>{p.scores.EPWORTH.score}/{p.scores.EPWORTH.max_score} <span className="text-muted small">({p.scores.EPWORTH.interpretation})</span></span> : '—'}</td>
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
          <div className="d-flex align-items-center mb-2">
            <h5 className="mb-0 me-3">{selectedPt.patient_id}</h5>
            <LevelBadge level={selectedPt.overall_level}/>
            <span className="ms-2 text-muted small">{selectedPt.instruments_completed} instruments completed</span>
          </div>
        </div></div>

        <div className="row">
          {Object.entries(selectedPt.scores||{}).map(([inst, data])=>
            <div key={inst} className="col-md-4 mb-3">
              <div className="card shadow-sm h-100"><div className="card-body">
                <h6>{inst}</h6>
                <div className="h3 mb-1" style={{color: LEVEL_COLORS[data.level]||'#6b7280'}}>{data.score}{data.max_score ? `/${data.max_score}` : ''}</div>
                <MiniBar value={data.score} max={data.max_score||100} color={LEVEL_COLORS[data.level]||'#3b82f6'}/>
                <div className="mt-2">
                  <div><strong>Interpretation:</strong> {data.interpretation}</div>
                  <div><strong>Category:</strong> {data.category}</div>
                  <div><LevelBadge level={data.level}/></div>
                  <div className="text-muted small mt-1">Assessed: {new Date(data.assessed_at).toLocaleDateString()}</div>
                </div>
              </div></div>
            </div>
          )}
        </div>
      </div>}
    </div>}

    {/* ── Score Distributions Tab ── */}
    {tab==='distributions' && bd && <div>
      <div className="row">
        {Object.entries(bd.score_distributions||{}).map(([inst, buckets])=>
          <div key={inst} className="col-md-4 mb-3">
            <div className="card shadow-sm h-100"><div className="card-body">
              <h6>{inst} — Score Distribution</h6>
              <table className="table table-sm">
                <thead><tr><th>Range</th><th>Count</th><th>Bar</th></tr></thead>
                <tbody>{(buckets||[]).map((b,i)=>{
                  const maxCount = Math.max(...buckets.map(x=>x.count));
                  return <tr key={i}>
                    <td className="small">{b.bucket}</td>
                    <td>{b.count}</td>
                    <td><MiniBar value={b.count} max={maxCount||1} color={b.count===maxCount?'#3b82f6':'#93c5fd'}/></td>
                  </tr>;
                })}</tbody>
              </table>
            </div></div>
          </div>
        )}
      </div>
    </div>}

    {/* ── Recent Assessments Tab ── */}
    {tab==='recent' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Recent Assessments ({bd.recent_assessments?.length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead><tr><th>Patient</th><th>Instrument</th><th>Score</th><th>Max</th><th>Interpretation</th><th>Level</th><th>Date</th></tr></thead>
            <tbody>{(bd.recent_assessments||[]).map((a,i)=>
              <tr key={i}>
                <td className="fw-bold">{a.patient_id}</td>
                <td>{a.instrument}</td>
                <td>{a.score}</td>
                <td>{a.max_score}</td>
                <td>{a.interpretation}</td>
                <td><LevelBadge level={a.level}/></td>
                <td className="text-muted small">{new Date(a.assessed_at).toLocaleDateString()}</td>
              </tr>
            )}</tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Definitions Tab ── */}
    {tab==='definitions' && defs && <div>
      {(defs.sections||[]).map((sec,i)=>
        <div key={i} className="card shadow-sm mb-3"><div className="card-body">
          <h6>{sec.title}</h6>
          <table className="table table-sm">
            <thead><tr><th>Term</th><th>Definition</th></tr></thead>
            <tbody>{(sec.items||[]).map((item,j)=>
              <tr key={j}><td className="fw-bold">{item.term}</td><td>{item.definition}</td></tr>
            )}</tbody>
          </table>
        </div></div>
      )}
    </div>}
  </div>);
}
