'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function MedicationRefillsDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/medication-refills/overview`).then(r=>r.json()),
      fetch(`${API}/api/medication-refills/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/medication-refills/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading medication refills data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'drugs', label:'Drug Details'},
    {id:'patients', label:'Per Patient'},
    {id:'gaps', label:'Gap Analysis'},
    {id:'recent', label:'Recent Refills'},
    {id:'definitions', label:'Definitions'},
  ];

  const k = ov.kpis;

  return (<div className="p-3">
    <h3>Medication Refills Dashboard</h3>
    <p className="text-muted">AED refill tracking &mdash; {k.total_refills} refills, {k.total_patients} patients, {k.total_drugs} drugs, {k.total_pharmacies} pharmacies</p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          ['Total Refills', k.total_refills, 'primary'],
          ['Patients', k.total_patients, 'info'],
          ['Drugs', k.total_drugs, 'success'],
          ['Avg Quantity', k.avg_quantity, 'warning'],
          ['Avg Days Supply', k.avg_days_supply, 'secondary'],
          ['Auto-Refill %', k.auto_refill_pct+'%', 'dark'],
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
            <h6>Drug Distribution</h6>
            {ov.drug_distribution.map(d=>
              <div key={d.drug_name} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'110px'}}>{d.drug_name}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className="progress-bar bg-primary" style={{width:`${d.pct}%`}}>
                      {d.count} ({d.pct}%)
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div></div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Pharmacy Distribution</h6>
            {ov.pharmacy_distribution.map(p=>
              <div key={p.pharmacy} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'130px'}}>{p.pharmacy}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className="progress-bar bg-success" style={{width:`${p.pct}%`}}>
                      {p.count} ({p.pct}%)
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div></div>
        </div>
      </div>

      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Monthly Refill Trend</h6>
            <table className="table table-sm table-bordered">
              <thead><tr><th>Month</th><th>Refills</th><th>Trend</th></tr></thead>
              <tbody>
                {ov.monthly_trend.map((m,i)=>{
                  const prev = i>0 ? ov.monthly_trend[i-1].refills : m.refills;
                  const diff = m.refills - prev;
                  return <tr key={m.month}>
                    <td>{m.month}</td>
                    <td className="fw-bold">{m.refills}</td>
                    <td>{i===0?'--':diff>=0?<span className="text-success">+{diff}</span>:<span className="text-danger">{diff}</span>}</td>
                  </tr>;
                })}
              </tbody>
            </table>
          </div></div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Auto vs Manual Refills</h6>
            {ov.auto_refill_breakdown.map(a=>
              <div key={a.type} className="d-flex align-items-center mb-2">
                <span className={`badge bg-${a.type==='Auto'?'success':'warning'} me-2`}>{a.type}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'22px'}}>
                    <div className={`progress-bar bg-${a.type==='Auto'?'success':'warning'}`}
                      style={{width:`${a.pct}%`}}>
                      {a.count} ({a.pct}%)
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div></div>
        </div>
      </div>
    </div>}

    {tab==='drugs' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Drug-Level Refill Details</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-bordered">
            <thead><tr>
              <th>Drug</th><th>Total Refills</th><th>Patients</th><th>Avg Qty</th><th>Avg Days</th><th>Auto-Refill %</th>
            </tr></thead>
            <tbody>
              {bd.drug_details.map(d=>
                <tr key={d.drug_name}>
                  <td className="fw-bold">{d.drug_name}</td>
                  <td>{d.total_refills}</td>
                  <td>{d.unique_patients}</td>
                  <td>{d.avg_quantity.toFixed(1)}</td>
                  <td>{d.avg_days_supply.toFixed(1)}</td>
                  <td>
                    <span className={`badge bg-${d.auto_refill_pct>=60?'success':d.auto_refill_pct>=40?'warning':'danger'}`}>
                      {d.auto_refill_pct.toFixed(1)}%
                    </span>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='patients' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Per-Patient Refill Summary ({bd.per_patient.length} patients)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-bordered">
            <thead><tr>
              <th>Patient</th><th>Refills</th><th>Drugs</th><th>Avg Qty</th><th>Avg Days Supply</th>
            </tr></thead>
            <tbody>
              {bd.per_patient.map(p=>
                <tr key={p.patient_id}>
                  <td className="fw-bold font-monospace">{p.patient_id}</td>
                  <td>{p.refills}</td>
                  <td>{p.drugs}</td>
                  <td>{p.avg_quantity.toFixed(1)}</td>
                  <td>{p.avg_days_supply.toFixed(1)}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='gaps' && bd && <div>
      <div className="alert alert-warning">
        <strong>Gap Analysis:</strong> Patients whose refill interval exceeds the previous days supply, indicating potential non-adherence and increased seizure risk.
      </div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Refill Gaps ({bd.gap_analysis.length} detected)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-bordered">
            <thead><tr>
              <th>Patient</th><th>Drug</th><th>Last Refill</th><th>Previous</th><th>Gap (days)</th><th>Supply (days)</th><th>Status</th>
            </tr></thead>
            <tbody>
              {bd.gap_analysis.map((g,i)=>
                <tr key={i}>
                  <td className="font-monospace">{g.patient_id}</td>
                  <td>{g.drug_name}</td>
                  <td>{g.last_refill}</td>
                  <td>{g.previous_refill}</td>
                  <td className="fw-bold text-danger">{g.gap_days}</td>
                  <td>{g.days_supply}</td>
                  <td><span className="badge bg-danger">{g.status}</span></td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='recent' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Recent Refills (latest 20)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-bordered">
            <thead><tr>
              <th>Date</th><th>Patient</th><th>Drug</th><th>Qty</th><th>Days</th><th>Pharmacy</th><th>Auto</th>
            </tr></thead>
            <tbody>
              {bd.recent_refills.map((r,i)=>
                <tr key={i}>
                  <td>{r.refill_date}</td>
                  <td className="font-monospace">{r.patient_id}</td>
                  <td>{r.drug_name}</td>
                  <td>{r.quantity}</td>
                  <td>{r.days_supply}</td>
                  <td>{r.pharmacy}</td>
                  <td>{r.auto_refill ? <span className="badge bg-success">Auto</span> : <span className="badge bg-secondary">Manual</span>}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='definitions' && defs && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Definitions &amp; Drug Classes</h6>
        <div className="mb-3">
          <p><strong>Auto-Refill:</strong> {defs.auto_refill_note}</p>
          <p><strong>Days Supply:</strong> {defs.days_supply_note}</p>
          <p><strong>Gap Analysis:</strong> {defs.gap_analysis_note}</p>
          <p><strong>Pharmacy:</strong> {defs.pharmacy_note}</p>
        </div>
        <h6>AED Drug Classes</h6>
        <table className="table table-sm table-bordered">
          <thead><tr><th>Drug</th><th>Mechanism / Class</th></tr></thead>
          <tbody>
            {Object.entries(defs.drug_classes).map(([drug, desc])=>
              <tr key={drug}><td className="fw-bold">{drug}</td><td>{desc}</td></tr>
            )}
          </tbody>
        </table>
      </div></div>
    </div>}
  </div>);
}
