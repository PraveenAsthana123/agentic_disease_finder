'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function ConsentManagementDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/consent-management/overview`).then(r=>r.json()),
      fetch(`${API}/api/consent-management/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/consent-management/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading consent management data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'matrix', label:'Type × Status'},
    {id:'patients', label:'Per Patient'},
    {id:'expiring', label:'Expiring Soon'},
    {id:'withdrawn', label:'Withdrawn'},
    {id:'definitions', label:'Definitions'},
  ];

  const statusColor = s => ({granted:'success',pending:'warning',withdrawn:'danger',declined:'secondary',expired:'dark'}[s]||'info');

  return (<div className="p-3">
    <h3>Consent Management Dashboard</h3>
    <p className="text-muted">
      Informed consent tracking &mdash; {ov.total_records} records, {ov.total_patients} patients,
      {' '}{Object.keys(ov.consent_type_distribution).length} consent types,
      {' '}{ov.compliance_rate_pct}% compliance rate
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          ['Total Records', ov.total_records, 'primary'],
          ['Patients', ov.total_patients, 'info'],
          ['Granted', ov.status_distribution.granted||0, 'success'],
          ['Pending', ov.status_distribution.pending||0, 'warning'],
          ['Compliance %', ov.compliance_rate_pct+'%', 'primary'],
          ['Expiring Soon', ov.expiring_soon, 'danger'],
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
            <h6>Consent Status Distribution</h6>
            {Object.entries(ov.status_distribution).map(([status, count])=>{
              const pct = ((count/ov.total_records)*100).toFixed(1);
              return <div key={status} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'90px',textTransform:'capitalize'}}>{status}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${statusColor(status)}`} style={{width:`${pct}%`}}>
                      {count} ({pct}%)
                    </div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Consent Type Distribution</h6>
            {Object.entries(ov.consent_type_distribution).map(([type, count])=>{
              const pct = ((count/ov.total_records)*100).toFixed(1);
              return <div key={type} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'120px'}}>{type.replace(/_/g,' ')}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className="progress-bar bg-primary" style={{width:`${pct}%`}}>
                      {count} ({pct}%)
                    </div>
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
            <h6>Witness Coverage</h6>
            {Object.entries(ov.witness_distribution).map(([witness, count])=>{
              const pct = ((count/ov.total_records)*100).toFixed(1);
              return <div key={witness} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'150px'}}>{witness}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className={`progress-bar bg-${witness==='Unwitnessed'?'danger':'info'}`} style={{width:`${pct}%`}}>
                      {count} ({pct}%)
                    </div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Monthly Volume</h6>
            <table className="table table-sm table-bordered">
              <thead><tr><th>Month</th><th>Records</th><th>Trend</th></tr></thead>
              <tbody>
                {ov.monthly_volume.map((m,i)=>{
                  const prev = i>0 ? ov.monthly_volume[i-1].cnt : m.cnt;
                  const diff = m.cnt - prev;
                  return <tr key={m.month}>
                    <td>{m.month}</td>
                    <td className="fw-bold">{m.cnt}</td>
                    <td>{i===0?'--':diff>=0?<span className="text-success">+{diff}</span>:<span className="text-danger">{diff}</span>}</td>
                  </tr>;
                })}
              </tbody>
            </table>
          </div></div>
        </div>
      </div>
    </div>}

    {tab==='matrix' && ov && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Consent Type &times; Status Matrix</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered table-striped">
            <thead><tr>
              <th>Consent Type</th><th className="text-success">Granted</th><th className="text-warning">Pending</th>
              <th className="text-danger">Withdrawn</th><th>Declined</th><th>Expired</th><th>Total</th>
            </tr></thead>
            <tbody>
              {ov.type_status_matrix.map(r=>{
                const total = (r.granted||0)+(r.pending||0)+(r.withdrawn||0)+(r.declined||0)+(r.expired||0);
                return <tr key={r.consent_type}>
                  <td className="fw-bold" style={{textTransform:'capitalize'}}>{r.consent_type.replace(/_/g,' ')}</td>
                  <td><span className="badge bg-success">{r.granted||0}</span></td>
                  <td><span className="badge bg-warning text-dark">{r.pending||0}</span></td>
                  <td><span className="badge bg-danger">{r.withdrawn||0}</span></td>
                  <td><span className="badge bg-secondary">{r.declined||0}</span></td>
                  <td><span className="badge bg-dark">{r.expired||0}</span></td>
                  <td className="fw-bold">{total}</td>
                </tr>;
              })}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='patients' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Per-Patient Consent Summary ({bd.per_patient.length} patients)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-bordered">
            <thead><tr>
              <th>Patient</th><th>Total</th><th className="text-success">Granted</th>
              <th className="text-warning">Pending</th><th className="text-danger">Withdrawn</th>
              <th>Declined</th><th>Expired</th><th>Latest Date</th>
            </tr></thead>
            <tbody>
              {bd.per_patient.map(p=>
                <tr key={p.patient_id}>
                  <td className="fw-bold font-monospace">{p.patient_id}</td>
                  <td>{p.total}</td>
                  <td><span className="badge bg-success">{p.granted}</span></td>
                  <td><span className="badge bg-warning text-dark">{p.pending}</span></td>
                  <td>{p.withdrawn>0?<span className="badge bg-danger">{p.withdrawn}</span>:0}</td>
                  <td>{p.declined>0?<span className="badge bg-secondary">{p.declined}</span>:0}</td>
                  <td>{p.expired>0?<span className="badge bg-dark">{p.expired}</span>:0}</td>
                  <td>{p.latest_date}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='expiring' && bd && <div>
      <div className="alert alert-warning">
        <strong>Expiring Soon:</strong> Consents within 30 days of expiry that require re-consent action.
      </div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Expiring Consents ({bd.expiring_soon_list.length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-bordered">
            <thead><tr>
              <th>Patient</th><th>Type</th><th>Status</th><th>Granted</th><th>Expiry</th><th>Days Left</th><th>Witness</th>
            </tr></thead>
            <tbody>
              {bd.expiring_soon_list.map(c=>
                <tr key={c.id}>
                  <td className="font-monospace">{c.patient_id}</td>
                  <td style={{textTransform:'capitalize'}}>{c.consent_type.replace(/_/g,' ')}</td>
                  <td><span className={`badge bg-${statusColor(c.status)}`}>{c.status}</span></td>
                  <td>{c.granted_date}</td>
                  <td>{c.expiry_date}</td>
                  <td className="fw-bold text-danger">{c.days_left}</td>
                  <td>{c.witness||'—'}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='withdrawn' && bd && <div>
      <div className="alert alert-danger">
        <strong>Withdrawn Consents:</strong> Patients who revoked previously granted consent. Activities must cease immediately per withdrawal protocol.
      </div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Withdrawn Consents ({bd.withdrawn_list.length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-bordered">
            <thead><tr>
              <th>Patient</th><th>Type</th><th>Granted</th><th>Expiry</th><th>Witness</th><th>Notes</th>
            </tr></thead>
            <tbody>
              {bd.withdrawn_list.map(c=>
                <tr key={c.id}>
                  <td className="font-monospace">{c.patient_id}</td>
                  <td style={{textTransform:'capitalize'}}>{c.consent_type.replace(/_/g,' ')}</td>
                  <td>{c.granted_date}</td>
                  <td>{c.expiry_date}</td>
                  <td>{c.witness||'—'}</td>
                  <td className="small">{c.notes}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='definitions' && defs && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Consent Types</h6>
        <table className="table table-sm table-bordered">
          <thead><tr><th>Type</th><th>Description</th></tr></thead>
          <tbody>
            {defs.consent_types.map(t=>
              <tr key={t.type}><td className="fw-bold" style={{textTransform:'capitalize'}}>{t.type.replace(/_/g,' ')}</td><td>{t.description}</td></tr>
            )}
          </tbody>
        </table>
      </div></div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Consent Statuses</h6>
        <table className="table table-sm table-bordered">
          <thead><tr><th>Status</th><th>Description</th></tr></thead>
          <tbody>
            {defs.statuses.map(s=>
              <tr key={s.status}>
                <td><span className={`badge bg-${statusColor(s.status)}`}>{s.status}</span></td>
                <td>{s.description}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div></div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Glossary</h6>
        {defs.glossary.map(g=>
          <div key={g.term} className="mb-2">
            <strong>{g.term}:</strong> {g.definition}
          </div>
        )}
      </div></div>

      <div className="card shadow-sm"><div className="card-body">
        <h6>Compliance Notes</h6>
        {defs.compliance_notes.map(n=>
          <div key={n.title} className="mb-2">
            <strong>{n.title}:</strong> {n.detail}
          </div>
        )}
      </div></div>
    </div>}
  </div>);
}
