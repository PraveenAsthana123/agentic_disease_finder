'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

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

function Bar({items, colorKey}){
  if(!items||!items.length) return null;
  const mx = Math.max(...items.map(i=>i.value));
  return (
    <div>{items.map((it,i)=>(
      <div key={i} className="d-flex align-items-center mb-1">
        <div className="text-end small me-2" style={{width:140,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{it.label}</div>
        <div className="flex-grow-1">
          <div className="progress" style={{height:18}}>
            <div className={`progress-bar bg-${it.color||colorKey||'primary'}`}
                 style={{width:`${mx?((it.value/mx)*100):0}%`}}>
              <span className="small">{it.value}</span>
            </div>
          </div>
        </div>
      </div>
    ))}</div>
  );
}

const STATUS_COLORS = {complete:'success', partial:'warning', pending:'secondary'};

export default function ReportingAIDashboard(){
  const [summary, setSummary] = useState(null);
  const [reports, setReports] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/patient-reports/summary`).then(r=>r.json()),
      fetch(`${API}/api/patient-reports`).then(r=>r.json()),
      fetch(`${API}/api/patient-reports/definitions`).then(r=>r.json()),
    ]).then(([s,r,d])=>{setSummary(s);setReports(r);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load Reporting AI data: {err}</div>;
  if(!summary) return <div className="text-muted p-3">Loading Reporting AI dashboard...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'instruments', label:'Assessment Instruments'},
    {id:'patients', label:'Patient Reports'},
    {id:'predictions', label:'AI Predictions'},
    {id:'definitions', label:'Definitions'},
  ];

  const reps = reports?.reports || [];
  const complete = reps.filter(r=>r.status==='complete').length;
  const partial = reps.filter(r=>r.status==='partial').length;
  const pending = reps.filter(r=>r.status==='pending').length;
  const avgCompleteness = reps.length ? Math.round(reps.reduce((s,r)=>s+r.completeness,0)/reps.length) : 0;

  return (
    <div className="container-fluid py-3">
      <h3>Reporting AI Dashboard</h3>
      <p className="text-muted small mb-3">Patient report assembly from real clinical.db — EEG analyses, assessments, seizure diaries, expert reviews</p>

      <div className="row mb-3">
        <KPI label="Total Patients" value={summary.total_patients} color="primary"/>
        <KPI label="Total Analyses" value={summary.total_analyses} color="info"/>
        <KPI label="Total Assessments" value={summary.total_assessments} color="success"/>
        <KPI label="Seizure Events" value={summary.total_seizure_events} color="warning"/>
        <KPI label="Expert Reviews" value={summary.total_expert_reviews} color="info"/>
        <KPI label="Clinical Decisions" value={summary.total_clinical_decisions} color="primary"/>
        <KPI label="HITL Reviews" value={summary.total_hitl_reviews} color="danger"/>
        <KPI label="Avg Completeness" value={`${avgCompleteness}%`} color="success"/>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t=>(
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab===t.id?' active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {tab==='overview' && (
        <div className="row">
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm"><div className="card-body">
              <h6>Report Status Distribution</h6>
              <Bar items={[
                {label:'Complete',value:complete,color:'success'},
                {label:'Partial',value:partial,color:'warning'},
                {label:'Pending',value:pending,color:'secondary'},
              ]}/>
            </div></div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm"><div className="card-body">
              <h6>Disease Breakdown</h6>
              <Bar items={(summary.disease_breakdown||[]).map(d=>({label:d.disease,value:d.count}))} colorKey="info"/>
            </div></div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm"><div className="card-body">
              <h6>Analysis Coverage</h6>
              <Bar items={[
                {label:'With Analysis',value:summary.patients_with_analysis,color:'success'},
                {label:'Without Analysis',value:summary.patients_without_analysis,color:'secondary'},
              ]}/>
            </div></div>
          </div>
        </div>
      )}

      {tab==='instruments' && (
        <div className="row">
          <div className="col-md-8 mb-3">
            <div className="card shadow-sm"><div className="card-body">
              <h6>Assessment Instrument Distribution ({summary.total_assessments} total)</h6>
              <Bar items={(summary.instrument_breakdown||[]).map(d=>({label:d.instrument,value:d.count}))} colorKey="primary"/>
            </div></div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm"><div className="card-body">
              <h6>Instrument Count</h6>
              <p className="h2 text-primary">{(summary.instrument_breakdown||[]).length}</p>
              <p className="text-muted small">Unique validated instruments tracked across {summary.total_patients} patients</p>
            </div></div>
          </div>
        </div>
      )}

      {tab==='patients' && (
        <div className="card shadow-sm"><div className="card-body">
          <h6>Patient Report Cards ({reps.length} patients)</h6>
          <div className="table-responsive" style={{maxHeight:500,overflowY:'auto'}}>
            <table className="table table-sm table-striped">
              <thead className="table-dark sticky-top">
                <tr>
                  <th>Patient ID</th>
                  <th>Name</th>
                  <th>Disease</th>
                  <th>Department</th>
                  <th>Status</th>
                  <th>Completeness</th>
                  <th>Analyses</th>
                  <th>Assessments</th>
                  <th>Seizures</th>
                  <th>Reviews</th>
                  <th>Meds</th>
                </tr>
              </thead>
              <tbody>
                {reps.map(r=>(
                  <tr key={r.patient_id}>
                    <td className="fw-bold small">{r.patient_id}</td>
                    <td className="small">{r.name}</td>
                    <td className="small">{r.disease}</td>
                    <td className="small">{r.department}</td>
                    <td><span className={`badge bg-${STATUS_COLORS[r.status]||'secondary'}`}>{r.status}</span></td>
                    <td>
                      <div className="progress" style={{height:16,minWidth:60}}>
                        <div className={`progress-bar bg-${r.completeness>=60?'success':r.completeness>0?'warning':'secondary'}`}
                             style={{width:`${r.completeness}%`}}>
                          <span className="small">{r.completeness}%</span>
                        </div>
                      </div>
                    </td>
                    <td className="small">{r.latest_analysis? `${r.latest_analysis.predicted_label} (${(r.latest_analysis.confidence*100).toFixed(0)}%)` : '—'}</td>
                    <td className="small">{r.assessment_count}</td>
                    <td className="small">{r.seizure_diary_entries}</td>
                    <td className="small">{r.expert_reviews?.length||0}</td>
                    <td className="small">{r.medication_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div></div>
      )}

      {tab==='predictions' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm"><div className="card-body">
              <h6>AI Prediction Breakdown</h6>
              <Bar items={(summary.prediction_breakdown||[]).map(d=>({label:d.label,value:d.count}))} colorKey="info"/>
            </div></div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm"><div className="card-body">
              <h6>Patients with AI Analysis</h6>
              <table className="table table-sm table-striped">
                <thead className="table-dark">
                  <tr><th>Patient</th><th>Prediction</th><th>Confidence</th><th>Signal Quality</th><th>Date</th></tr>
                </thead>
                <tbody>
                  {reps.filter(r=>r.latest_analysis).map(r=>(
                    <tr key={r.patient_id}>
                      <td className="small fw-bold">{r.patient_id}</td>
                      <td className="small">{r.latest_analysis.predicted_label}</td>
                      <td className="small">{r.latest_analysis.confidence!=null?(r.latest_analysis.confidence*100).toFixed(1)+'%':'—'}</td>
                      <td className="small">{r.latest_analysis.signal_quality||'—'}</td>
                      <td className="small">{r.latest_analysis.date||'—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div></div>
          </div>
        </div>
      )}

      {tab==='definitions' && defs && (
        <div className="card shadow-sm"><div className="card-body">
          <h6>Report Metric Definitions</h6>
          <div className="row">
            {Object.entries(defs.definitions||{}).map(([term, desc])=>(
              <div key={term} className="col-md-6 mb-3">
                <div className="card border"><div className="card-body py-2">
                  <strong className="text-primary">{term}</strong>
                  <p className="mb-0 small text-muted">{desc}</p>
                </div></div>
              </div>
            ))}
          </div>
        </div></div>
      )}
    </div>
  );
}
