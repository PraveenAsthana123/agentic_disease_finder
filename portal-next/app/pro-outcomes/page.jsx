'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SEV_COLORS = {
  Good:'#22c55e', Normal:'#22c55e', Minimal:'#22c55e',
  Fair:'#f59e0b', 'Mild Impairment':'#f59e0b', Mild:'#f59e0b',
  Poor:'#ef4444', 'Moderate Impairment':'#f97316', Moderate:'#f97316',
  Severe:'#ef4444', 'Mod-Severe':'#dc2626',
};

function SevBadge({label}){
  const c = SEV_COLORS[label] || '#6b7280';
  return <span className="badge" style={{backgroundColor:c}}>{label}</span>;
}

function MiniBar({value, max=100, color='#3b82f6', label=''}){
  const pct = Math.min(100,Math.max(0,(value/max)*100));
  return (
    <div>
      <div className="d-flex justify-content-between small text-muted mb-1">
        <span>{label}</span><span>{value}</span>
      </div>
      <div className="progress" style={{height:'8px'}}>
        <div className="progress-bar" style={{width:`${pct}%`,backgroundColor:color}}></div>
      </div>
    </div>
  );
}

function StatCard({label, value, sub, color='#3b82f6'}){
  return (
    <div className="col-6 col-md mb-2">
      <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
        <div className="h5 mb-0" style={{color}}>{value}</div>
        <div className="text-muted small">{label}</div>
        {sub && <div className="text-muted" style={{fontSize:'0.7rem'}}>{sub}</div>}
      </div></div>
    </div>
  );
}

function TrendIcon({trend}){
  if(trend==='improving') return <span className="text-success">↑ improving</span>;
  if(trend==='worsening') return <span className="text-danger">↓ worsening</span>;
  return <span className="text-secondary">→ stable</span>;
}

export default function PROOutcomesDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [selPt, setSelPt] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/pro-outcomes/overview`).then(r=>r.json()),
      fetch(`${API}/api/pro-outcomes/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/pro-outcomes/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading PRO Outcomes data…</div>;

  const TABS = [
    {id:'overview',    label:'📊 Overview'},
    {id:'instruments', label:'🔬 Instruments'},
    {id:'patients',    label:'👤 Per Patient'},
    {id:'trends',      label:'📈 QoL Trend'},
    {id:'definitions', label:'📖 Definitions'},
  ];

  const ptList = bd?.patients || [];

  /* ── domain color map ── */
  const domainColors = ['#3b82f6','#6366f1','#8b5cf6','#22c55e','#f59e0b'];

  /* ── instrument meta ── */
  const instruments = [
    {key:'avg_psqi',   label:'PSQI', full:'Sleep Quality Index', max:21, note:'≤5 good · 6-10 fair · >10 poor', low_good:true},
    {key:'avg_ess',    label:'ESS',  full:'Epworth Sleepiness',  max:24, note:'<10 normal · 10-15 mod · >15 severe', low_good:true},
    {key:'avg_phq9',   label:'PHQ-9',full:'Depression Screen',   max:27, note:'<5 minimal · 5-9 mild · 10-14 mod', low_good:true},
    {key:'avg_gad7',   label:'GAD-7',full:'Anxiety Screen',      max:21, note:'<5 minimal · 5-9 mild · 10-14 mod', low_good:true},
    {key:'avg_qolie31',label:'QOLIE-31',full:'Quality of Life',  max:100, note:'higher = better QoL', low_good:false},
    {key:'avg_moca',   label:'MoCA', full:'Cognitive Assessment',max:30, note:'≥26 normal · 18-25 mild · <18 mod', low_good:false},
    {key:'avg_nddi_e', label:'NDDI-E',full:'Neurological Depression', max:24, note:'<15 unlikely · 15-24 possible', low_good:true},
    {key:'avg_wpai',   label:'WPAI', full:'Work Productivity',   max:100, note:'% impairment — lower is better', low_good:true},
  ];

  return (
    <div className="p-3">
      <h3>📋 PRO Outcomes Dashboard</h3>
      <p className="text-muted">
        {ov.total_assessments} assessments · {ov.total_patients} patients · 8 validated instruments
        (PSQI / ESS / PHQ-9 / GAD-7 / QOLIE-31 / MoCA / NDDI-E / WPAI)
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t=>(
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ── */}
      {tab==='overview' && <div>
        <div className="row mb-3">
          <StatCard label="Total Assessments" value={ov.total_assessments} color="#3b82f6"/>
          <StatCard label="Patients" value={ov.total_patients} color="#6366f1"/>
          <StatCard label="Avg QOLIE-31" value={ov.avg_qolie31} sub="QoL (0-100)" color="#22c55e"/>
          <StatCard label="Avg PHQ-9" value={ov.avg_phq9} sub="Depression (lower=better)" color="#f59e0b"/>
          <StatCard label="Avg MoCA" value={ov.avg_moca} sub="Cognition (/30)" color="#8b5cf6"/>
        </div>

        {/* Domain Averages */}
        <div className="card shadow-sm mb-3"><div className="card-body">
          <h6>Domain Health Snapshot</h6>
          <div className="row">
            {(ov.domain_averages||[]).map((d,i)=>(
              <div key={d.domain} className="col-6 col-md mb-2">
                <div className="small text-muted mb-1">{d.domain}</div>
                <MiniBar value={d.score} max={100} color={domainColors[i%domainColors.length]}/>
              </div>
            ))}
          </div>
        </div></div>

        {/* Distributions row */}
        <div className="row mb-3">
          {/* Sleep Quality */}
          <div className="col-md-4 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body">
              <h6>Sleep Quality (PSQI)</h6>
              <table className="table table-sm mb-0">
                <tbody>
                  {(ov.sleep_quality_distribution||[]).map(r=>(
                    <tr key={r.category}>
                      <td><SevBadge label={r.category}/></td>
                      <td>{r.count}</td>
                      <td><div className="progress" style={{height:'6px'}}>
                        <div className="progress-bar" style={{width:`${Math.round(r.count/ov.total_assessments*100)}%`,backgroundColor:SEV_COLORS[r.category]||'#6b7280'}}></div>
                      </div></td>
                      <td className="text-muted small">{Math.round(r.count/ov.total_assessments*100)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div></div>
          </div>

          {/* Depression */}
          <div className="col-md-4 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body">
              <h6>Depression Severity (PHQ-9)</h6>
              <table className="table table-sm mb-0">
                <tbody>
                  {(ov.depression_severity||[]).map(r=>(
                    <tr key={r.severity}>
                      <td><SevBadge label={r.severity}/></td>
                      <td>{r.count}</td>
                      <td><div className="progress" style={{height:'6px'}}>
                        <div className="progress-bar" style={{width:`${Math.round(r.count/ov.total_assessments*100)}%`,backgroundColor:SEV_COLORS[r.severity]||'#6b7280'}}></div>
                      </div></td>
                      <td className="text-muted small">{Math.round(r.count/ov.total_assessments*100)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div></div>
          </div>

          {/* Anxiety */}
          <div className="col-md-4 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body">
              <h6>Anxiety Severity (GAD-7)</h6>
              <table className="table table-sm mb-0">
                <tbody>
                  {(ov.anxiety_severity||[]).map(r=>(
                    <tr key={r.severity}>
                      <td><SevBadge label={r.severity}/></td>
                      <td>{r.count}</td>
                      <td><div className="progress" style={{height:'6px'}}>
                        <div className="progress-bar" style={{width:`${Math.round(r.count/ov.total_assessments*100)}%`,backgroundColor:SEV_COLORS[r.severity]||'#6b7280'}}></div>
                      </div></td>
                      <td className="text-muted small">{Math.round(r.count/ov.total_assessments*100)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div></div>
          </div>
        </div>

        {/* Cognitive Status */}
        <div className="card shadow-sm mb-3"><div className="card-body">
          <h6>Cognitive Status (MoCA)</h6>
          <div className="d-flex flex-wrap gap-3">
            {(ov.cognitive_status||[]).map(r=>(
              <div key={r.status} className="text-center">
                <div className="h4 mb-0" style={{color:SEV_COLORS[r.status]||'#6b7280'}}>{r.count}</div>
                <div className="small text-muted">{r.status}</div>
              </div>
            ))}
          </div>
        </div></div>

        {/* Correlation note */}
        <div className="alert alert-light small">
          <strong>Mood vs Seizure Worry correlation:</strong> r = {ov.mood_vs_seizure_worry} &nbsp;
          (weak negative — higher seizure worry associated with lower mood ratings)
        </div>
      </div>}

      {/* ── Instruments Tab ── */}
      {tab==='instruments' && <div>
        <div className="card shadow-sm mb-3"><div className="card-body">
          <h6>Cohort-Level Instrument Averages</h6>
          <div className="table-responsive">
            <table className="table table-sm align-middle">
              <thead className="table-light">
                <tr>
                  <th>Instrument</th><th>Full Name</th><th>Avg Score</th><th>Scale</th><th>Bar</th><th>Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {instruments.map(ins=>{
                  const val = ov[ins.key];
                  const pct = Math.min(100, Math.max(0, (val/ins.max)*100));
                  const color = ins.low_good
                    ? (pct<33?'#22c55e':pct<66?'#f59e0b':'#ef4444')
                    : (pct>66?'#22c55e':pct>33?'#f59e0b':'#ef4444');
                  return (
                    <tr key={ins.key}>
                      <td><strong>{ins.label}</strong></td>
                      <td className="text-muted small">{ins.full}</td>
                      <td style={{color, fontWeight:'600'}}>{val}</td>
                      <td className="text-muted small">/{ins.max}</td>
                      <td style={{minWidth:'120px'}}>
                        <div className="progress" style={{height:'8px'}}>
                          <div className="progress-bar" style={{width:`${pct}%`,backgroundColor:color}}></div>
                        </div>
                      </td>
                      <td className="text-muted small">{ins.note}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div></div>

        <div className="row">
          {instruments.map(ins=>{
            const val = ov[ins.key];
            const pct = Math.min(100, Math.max(0, (val/ins.max)*100));
            const color = ins.low_good
              ? (pct<33?'#22c55e':pct<66?'#f59e0b':'#ef4444')
              : (pct>66?'#22c55e':pct>33?'#f59e0b':'#ef4444');
            return (
              <div key={ins.key} className="col-6 col-md-3 mb-3">
                <div className="card shadow-sm h-100"><div className="card-body text-center">
                  <div className="h3 mb-0" style={{color}}>{val}</div>
                  <div className="fw-semibold small">{ins.label}</div>
                  <div className="text-muted" style={{fontSize:'0.7rem'}}>{ins.full}</div>
                  <div className="progress mt-2" style={{height:'6px'}}>
                    <div className="progress-bar" style={{width:`${pct}%`,backgroundColor:color}}></div>
                  </div>
                  <div className="text-muted mt-1" style={{fontSize:'0.68rem'}}>{ins.note}</div>
                </div></div>
              </div>
            );
          })}
        </div>
      </div>}

      {/* ── Per Patient Tab ── */}
      {tab==='patients' && <div>
        <div className="row mb-3">
          <div className="col-md-5 mb-2">
            <div className="card shadow-sm" style={{maxHeight:'75vh', overflowY:'auto'}}>
              <div className="card-header small fw-semibold">Patient List ({ptList.length})</div>
              <ul className="list-group list-group-flush">
                {ptList.map(p=>(
                  <li key={p.patient_id}
                    className={`list-group-item list-group-item-action py-2 ${selPt?.patient_id===p.patient_id?'active':''}`}
                    onClick={()=>setSelPt(p)} style={{cursor:'pointer'}}>
                    <div className="d-flex justify-content-between align-items-center">
                      <strong>{p.patient_id}</strong>
                      <span className="badge bg-secondary">{p.total_assessments} Ass.</span>
                    </div>
                    <div className="small d-flex gap-2 flex-wrap mt-1">
                      <span>QoL: <strong>{p.latest_qolie31}</strong></span>
                      <span>PHQ-9: <strong>{p.latest_phq9}</strong></span>
                      <TrendIcon trend={p.phq9_trend}/>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="col-md-7 mb-2">
            {!selPt ? (
              <div className="alert alert-light">Select a patient to view their PRO scores.</div>
            ) : (
              <div className="card shadow-sm h-100"><div className="card-body">
                <h6>{selPt.patient_id} — Latest PRO Scores</h6>
                <div className="row mb-3">
                  {[
                    ['Assessments', selPt.total_assessments, '#3b82f6'],
                    ['QOLIE-31', selPt.latest_qolie31, '#22c55e'],
                    ['PHQ-9', selPt.latest_phq9, '#f59e0b'],
                    ['MoCA', selPt.latest_moca, '#8b5cf6'],
                  ].map(([k,v,c])=>(
                    <div key={k} className="col-6 mb-2">
                      <div className="card bg-light border-0"><div className="card-body text-center py-2">
                        <div className="h5 mb-0" style={{color:c}}>{v}</div>
                        <div className="text-muted small">{k}</div>
                      </div></div>
                    </div>
                  ))}
                </div>
                <table className="table table-sm">
                  <tbody>
                    {[
                      ['PSQI (Sleep)', selPt.latest_psqi, '/21', true],
                      ['ESS (Sleepiness)', selPt.latest_ess, '/24', true],
                      ['PHQ-9 (Depression)', selPt.latest_phq9, '/27', true],
                      ['GAD-7 (Anxiety)', selPt.latest_gad7, '/21', true],
                      ['QOLIE-31 (QoL)', selPt.latest_qolie31, '/100', false],
                      ['MoCA (Cognition)', selPt.latest_moca, '/30', false],
                      ['NDDI-E (Depression)', selPt.latest_nddi_e, '/24', true],
                      ['WPAI (Work impair.)', selPt.latest_wpai, '%', true],
                    ].map(([lbl,val,unit,lowGood])=>{
                      const maxMap = {'/21':21,'/24':24,'/27':27,'/21':21,'/100':100,'/30':30,'%':100};
                      const max = maxMap[unit]||100;
                      const pct = Math.min(100,Math.max(0,(val/max)*100));
                      const color = lowGood
                        ? (pct<33?'#22c55e':pct<66?'#f59e0b':'#ef4444')
                        : (pct>66?'#22c55e':pct>33?'#f59e0b':'#ef4444');
                      return (
                        <tr key={lbl}>
                          <td className="small">{lbl}</td>
                          <td style={{color,fontWeight:'600'}}>{val}{unit}</td>
                          <td style={{minWidth:'80px'}}>
                            <div className="progress" style={{height:'6px'}}>
                              <div className="progress-bar" style={{width:`${pct}%`,backgroundColor:color}}></div>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <div className="d-flex gap-3 small mt-2">
                  <span>PHQ-9 trend: <TrendIcon trend={selPt.phq9_trend}/></span>
                  <span>QoL trend: <TrendIcon trend={selPt.qolie31_trend}/></span>
                </div>
                {selPt.recent_assessments?.length > 0 && (
                  <div className="mt-3">
                    <div className="small fw-semibold mb-1">Recent Assessments</div>
                    <div className="table-responsive" style={{maxHeight:'200px', overflowY:'auto'}}>
                      <table className="table table-sm table-striped">
                        <thead className="table-light"><tr>
                          <th>Date</th><th>PSQI</th><th>PHQ-9</th><th>QOLIE-31</th><th>MoCA</th><th>Notes</th>
                        </tr></thead>
                        <tbody>
                          {selPt.recent_assessments.map((a,i)=>(
                            <tr key={i}>
                              <td className="small">{a.assessment_date}</td>
                              <td>{a.psqi_score}</td>
                              <td>{a.phq9_score}</td>
                              <td>{a.qolie31_score}</td>
                              <td>{a.moca_score}</td>
                              <td className="small text-muted" style={{maxWidth:'150px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{a.notes}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div></div>
            )}
          </div>
        </div>
      </div>}

      {/* ── QoL Trend Tab ── */}
      {tab==='trends' && <div>
        <div className="card shadow-sm mb-3"><div className="card-body">
          <h6>Monthly QOLIE-31 Quality of Life Trend</h6>
          <div className="d-flex align-items-end gap-2 flex-wrap" style={{height:'160px'}}>
            {(ov.qol_trend||[]).map((m,i)=>{
              const h = Math.round((m.avg_score/100)*140);
              return (
                <div key={m.month} className="text-center d-flex flex-column align-items-center" style={{flex:1, minWidth:'60px'}}>
                  <div className="small text-muted mb-1">{m.avg_score}</div>
                  <div style={{width:'40px',height:`${h}px`,backgroundColor:'#3b82f6',borderRadius:'4px 4px 0 0'}}></div>
                  <div className="small text-muted mt-1">{m.month.slice(5)}/{m.month.slice(2,4)}</div>
                </div>
              );
            })}
          </div>
          <p className="text-muted small mt-2">
            QOLIE-31 range 0-100 (higher = better QoL). Avg cohort score: <strong>{ov.avg_qolie31}</strong>.
            Scores below 50 indicate significantly impaired health-related quality of life.
          </p>
        </div></div>

        <div className="card shadow-sm"><div className="card-body">
          <h6>Cohort PRO Summary Table</h6>
          <div className="table-responsive">
            <table className="table table-sm">
              <thead className="table-light">
                <tr><th>Instrument</th><th>Cohort Avg</th><th>Domain</th><th>Interpretation Range</th></tr>
              </thead>
              <tbody>
                {instruments.map(ins=>(
                  <tr key={ins.key}>
                    <td><strong>{ins.label}</strong> <span className="text-muted small">({ins.full})</span></td>
                    <td><strong>{ov[ins.key]}</strong>/{ins.max}</td>
                    <td className="text-muted small">{ins.note.split('·')[0].trim()}</td>
                    <td className="text-muted small">{ins.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div></div>
      </div>}

      {/* ── Definitions Tab ── */}
      {tab==='definitions' && <div>
        {(defs?.concepts||[]).map(c=>(
          <div key={c.name} className="card shadow-sm mb-3"><div className="card-body">
            <h6 className="text-primary">{c.name}</h6>
            <p className="mb-0 small">{c.description}</p>
          </div></div>
        ))}
      </div>}
    </div>
  );
}
