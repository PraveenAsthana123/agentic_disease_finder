'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const pct = (n, total) => total ? ((n / total) * 100).toFixed(1) : '0.0';
const engelColor = c => ({'I':'success','II':'info','III':'warning','IV':'danger'}[c]||'secondary');

export default function SurgicalOutcomesDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/surgical-outcomes/overview`).then(r=>r.json()),
      fetch(`${API}/api/surgical-outcomes/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/surgical-outcomes/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading surgical outcomes data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'engel', label:'Engel Classification'},
    {id:'surgeries', label:'All Surgeries'},
    {id:'by-type', label:'By Surgery Type'},
    {id:'followup', label:'Follow-up Trend'},
    {id:'definitions', label:'Definitions'},
  ];

  const total = ov.total_surgeries;

  return (<div className="p-3">
    <h3>Surgical Outcomes Dashboard</h3>
    <p className="text-muted">
      Epilepsy surgery outcome tracking &mdash; {total} surgeries, {ov.total_patients} patients,
      {' '}{ov.seizure_free_rate}% seizure-free (Engel IA), {ov.complication_rate}% complication rate,
      {' '}{ov.aed_reduction_rate}% AED reduction, mean follow-up {ov.mean_followup} months
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          ['Total Surgeries', total, 'primary'],
          ['Patients', ov.total_patients, 'info'],
          ['Seizure-Free', ov.seizure_free_count+' ('+ov.seizure_free_rate+'%)', 'success'],
          ['Complications', ov.complication_count+' ('+ov.complication_rate+'%)', 'danger'],
          ['AED Reduction', ov.aed_reduction_count+' ('+ov.aed_reduction_rate+'%)', 'warning'],
          ['Mean Follow-up', ov.mean_followup+' mo', 'primary'],
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
            <h6>Engel Outcome (Major Classes)</h6>
            {(ov.engel_major||[]).map(e=>{
              const p = pct(e.count, total);
              return <div key={e.class} className="d-flex align-items-center mb-2">
                <span className="me-2 small fw-bold" style={{minWidth:'120px'}}>Class {e.class}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'22px'}}>
                    <div className={`progress-bar bg-${engelColor(e.class)}`}
                      style={{width:`${p}%`}}>{e.count} ({p}%)</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Surgery Type Distribution</h6>
            {(ov.surgery_type_distribution||[]).map(s=>{
              const p = pct(s.count, total);
              return <div key={s.type} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'150px',textTransform:'capitalize'}}>{s.type}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className="progress-bar bg-info" style={{width:`${p}%`}}>{s.count} ({p}%)</div>
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
            <h6>ILAE Outcome Distribution</h6>
            {(ov.ilae_distribution||[]).map(i=>{
              const p = pct(i.count, total);
              const c = i.outcome<=2?'success':i.outcome<=4?'warning':'danger';
              return <div key={i.outcome} className="d-flex align-items-center mb-2">
                <span className="me-2 small fw-bold" style={{minWidth:'100px'}}>ILAE {i.outcome}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${c}`} style={{width:`${p}%`}}>{i.count} ({p}%)</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Follow-up Periods</h6>
            <div className="d-flex align-items-end" style={{height:'140px',gap:'6px'}}>
              {(ov.followup_trend||[]).map(f=>{
                const maxC = Math.max(...ov.followup_trend.map(x=>x.total));
                const h = maxC ? (f.total/maxC)*100 : 0;
                const sfH = maxC ? (f.seizure_free/maxC)*100 : 0;
                return <div key={f.period} className="d-flex flex-column align-items-center flex-grow-1">
                  <small className="text-muted mb-1">{f.total}</small>
                  <div style={{position:'relative',width:'100%',maxWidth:'50px',height:`${h}%`,minHeight:'4px'}}>
                    <div className="bg-primary rounded" style={{width:'100%',height:'100%'}}/>
                    <div className="bg-success rounded" style={{position:'absolute',bottom:0,width:'100%',height:`${f.total?((f.seizure_free/f.total)*100):0}%`}}/>
                  </div>
                  <small className="text-muted mt-1 text-center" style={{fontSize:'0.65rem'}}>{f.period}</small>
                </div>;
              })}
            </div>
            <div className="d-flex justify-content-center mt-2 gap-3">
              <small><span className="badge bg-primary">&nbsp;</span> Total</small>
              <small><span className="badge bg-success">&nbsp;</span> Seizure-free</small>
            </div>
          </div></div>
        </div>
      </div>
    </div>}

    {tab==='engel' && <div>
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Engel Subclass Detail</h6>
            <table className="table table-sm table-striped mb-0">
              <thead><tr><th>Class</th><th>Count</th><th>Share</th></tr></thead>
              <tbody>
                {(ov.engel_detail||[]).map(e=>
                  <tr key={e.class}>
                    <td className="fw-bold">{e.class}</td>
                    <td>{e.count}</td>
                    <td><span className={`badge bg-${engelColor(e.class.replace(/[A-D]$/,''))}`}>{pct(e.count,total)}%</span></td>
                  </tr>
                )}
              </tbody>
            </table>
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Engel Class Summary</h6>
            {(ov.engel_major||[]).map(e=>{
              const label = e.class==='I'?'Seizure-free':e.class==='II'?'Rare seizures':e.class==='III'?'Worthwhile improvement':'No improvement';
              return <div key={e.class} className="mb-3">
                <div className="d-flex justify-content-between mb-1">
                  <span className="fw-bold">Class {e.class} — {label}</span>
                  <span className={`badge bg-${engelColor(e.class)}`}>{e.count} ({pct(e.count,total)}%)</span>
                </div>
                <div className="progress" style={{height:'12px'}}>
                  <div className={`progress-bar bg-${engelColor(e.class)}`} style={{width:`${pct(e.count,total)}%`}}/>
                </div>
              </div>;
            })}
          </div></div>
        </div>
      </div>
    </div>}

    {tab==='surgeries' && bd && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>All Surgical Records ({(bd.surgeries||[]).length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr>
              <th>Patient</th><th>Surgery Type</th><th>Date</th><th>Hemisphere</th><th>Pathology</th>
              <th>Engel</th><th>ILAE</th><th>Follow-up</th><th>Seizure-Free</th><th>AED Reduced</th>
              <th>Complications</th><th>Pre Freq</th><th>Post Freq</th>
            </tr></thead>
            <tbody>
              {(bd.surgeries||[]).map((s,i)=>
                <tr key={i}>
                  <td className="fw-bold">{s.patient_id}</td>
                  <td style={{textTransform:'capitalize'}}>{s.surgery_type}</td>
                  <td>{s.surgery_date}</td>
                  <td>{s.hemisphere}</td>
                  <td className="small">{s.pathology}</td>
                  <td><span className={`badge bg-${engelColor(s.engel_class?.replace(/[A-D]$/,''))}`}>{s.engel_class}</span></td>
                  <td>{s.ilae_outcome}</td>
                  <td>{s.follow_up_months} mo</td>
                  <td><span className={`badge bg-${s.seizure_free?'success':'secondary'}`}>{s.seizure_free?'Yes':'No'}</span></td>
                  <td><span className={`badge bg-${s.aed_reduction?'info':'secondary'}`}>{s.aed_reduction?'Yes':'No'}</span></td>
                  <td>{s.complications?<span className={`badge bg-${s.complication_severity==='major'?'danger':'warning'}`}>{s.complications}</span>:<span className="text-muted">None</span>}</td>
                  <td>{s.pre_surgery_frequency}/mo</td>
                  <td>{s.post_surgery_frequency}/mo</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='by-type' && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Outcome by Surgery Type</h6>
        <table className="table table-sm table-striped mb-0">
          <thead><tr>
            <th>Surgery Type</th><th>Total</th><th>Seizure-Free</th><th>SF Rate</th>
            <th>Avg Follow-up</th><th>Complications</th><th>Comp Rate</th>
          </tr></thead>
          <tbody>
            {(ov.outcome_by_type||[]).map(o=>
              <tr key={o.surgery_type}>
                <td className="fw-bold" style={{textTransform:'capitalize'}}>{o.surgery_type}</td>
                <td>{o.total}</td>
                <td>{o.seizure_free}</td>
                <td><span className={`badge bg-${(o.seizure_free/o.total*100)>=50?'success':(o.seizure_free/o.total*100)>=25?'warning':'danger'}`}>{pct(o.seizure_free,o.total)}%</span></td>
                <td>{o.avg_followup} mo</td>
                <td>{o.complications}</td>
                <td><span className={`badge bg-${o.complications>0?'warning':'success'}`}>{pct(o.complications,o.total)}%</span></td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>
    </div>}

    {tab==='followup' && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Seizure Freedom by Follow-up Period</h6>
        <table className="table table-sm table-striped mb-0">
          <thead><tr><th>Period</th><th>Total</th><th>Seizure-Free</th><th>SF Rate</th></tr></thead>
          <tbody>
            {(ov.followup_trend||[]).map(f=>
              <tr key={f.period}>
                <td className="fw-bold">{f.period}</td>
                <td>{f.total}</td>
                <td>{f.seizure_free}</td>
                <td><span className={`badge bg-${f.total&&(f.seizure_free/f.total*100)>=50?'success':'warning'}`}>{pct(f.seizure_free,f.total)}%</span></td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Follow-up Distribution</h6>
        <div className="d-flex align-items-end" style={{height:'160px',gap:'8px'}}>
          {(ov.followup_trend||[]).map(f=>{
            const maxC = Math.max(...ov.followup_trend.map(x=>x.total));
            const h = maxC ? (f.total/maxC)*100 : 0;
            return <div key={f.period} className="d-flex flex-column align-items-center flex-grow-1">
              <small className="text-success mb-1">{f.seizure_free}/{f.total}</small>
              <div style={{position:'relative',width:'100%',maxWidth:'60px',height:`${h}%`,minHeight:'4px'}}>
                <div className="bg-primary rounded" style={{width:'100%',height:'100%'}}/>
                <div className="bg-success rounded" style={{position:'absolute',bottom:0,width:'100%',height:`${f.total?((f.seizure_free/f.total)*100):0}%`}}/>
              </div>
              <small className="text-muted mt-1 text-center">{f.period}</small>
            </div>;
          })}
        </div>
        <div className="d-flex justify-content-center mt-2 gap-3">
          <small><span className="badge bg-primary">&nbsp;</span> Total surgeries</small>
          <small><span className="badge bg-success">&nbsp;</span> Seizure-free</small>
        </div>
      </div></div>
    </div>}

    {tab==='definitions' && defs && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Engel Classification</h6>
        <table className="table table-sm mb-0">
          <thead><tr><th>Class</th><th>Label</th><th>Subclasses</th></tr></thead>
          <tbody>
            {(defs.engel_classification||[]).map(e=>
              <tr key={e.class}>
                <td className="fw-bold align-top">{e.class}</td>
                <td className="align-top">{e.label}</td>
                <td><ul className="mb-0 ps-3">{(e.subclasses||[]).map(s=>
                  <li key={s.sub}><strong>{s.sub}</strong>: {s.definition}</li>
                )}</ul></td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>ILAE Outcome Scale</h6>
        <table className="table table-sm mb-0">
          <thead><tr><th>Outcome</th><th>Definition</th></tr></thead>
          <tbody>
            {(defs.ilae_scale||[]).map(i=>
              <tr key={i.outcome}>
                <td className="fw-bold">ILAE {i.outcome}</td>
                <td>{i.definition}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Surgery Types</h6>
        <table className="table table-sm mb-0">
          <thead><tr><th>Type</th><th>Description</th></tr></thead>
          <tbody>
            {(defs.surgery_types||[]).map(s=>
              <tr key={s.type}>
                <td className="fw-bold" style={{minWidth:'180px'}}>{s.type}</td>
                <td>{s.description}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>

      {defs.data_sources && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Data Sources</h6>
        <ul className="mb-0">{defs.data_sources.map((s,i)=><li key={i}>{s}</li>)}</ul>
      </div></div>}
    </div>}
  </div>);
}
