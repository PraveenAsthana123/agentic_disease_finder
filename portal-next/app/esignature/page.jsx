'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function StatusBadge({status}){
  const m = {signed:'success', pending:'warning', rejected:'danger'};
  return <span className={`badge bg-${m[status]||'secondary'}`}>{status}</span>;
}

function KpiCard({label, value, sub, color='primary'}){
  return <div className="col-md-2 col-sm-4 col-6 mb-3">
    <div className={`card border-${color} h-100`}>
      <div className="card-body p-2 text-center">
        <div className={`fs-4 fw-bold text-${color}`}>{value}</div>
        <div className="small text-muted">{label}</div>
        {sub && <div className="small text-muted">{sub}</div>}
      </div>
    </div>
  </div>;
}

function MiniBar({value, max, color='primary'}){
  const pct = max>0?Math.min(100,Math.max(0,(value/max)*100)):0;
  return <div className="progress" style={{height:'8px'}}>
    <div className={`progress-bar bg-${color}`} style={{width:`${pct}%`}}/>
  </div>;
}

export default function ESignatureDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [search, setSearch] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/esignature/overview`).then(r=>r.json()),
      fetch(`${API}/api/esignature/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/esignature/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading e-signature data…</div>;

  const TABS = [
    {id:'overview',   label:'📋 Overview'},
    {id:'reports',    label:'📄 All Reports'},
    {id:'signers',    label:'👨‍⚕️ By Signer'},
    {id:'compliance', label:'⚖️ Compliance'},
    {id:'definitions',label:'📖 Definitions'},
  ];

  const kpi = ov.kpis;
  const thr = ov.thresholds;

  const allReports = bd?.all_reports || [];
  const filtered = allReports.filter(r=>{
    const matchStatus = filterStatus==='all' || r.status===filterStatus;
    const matchSearch = !search ||
      r.patient_id?.toLowerCase().includes(search.toLowerCase()) ||
      r.disease?.toLowerCase().includes(search.toLowerCase()) ||
      r.signer_name?.toLowerCase().includes(search.toLowerCase());
    return matchStatus && matchSearch;
  });

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-2">
        <h4 className="mb-0 me-3">🖊️ Clinical Report E-Signature</h4>
        <span className={`badge bg-${kpi.cfr11_compliant?'success':'danger'} me-2`}>
          {kpi.cfr11_compliant?'✅ 21 CFR Part 11 Compliant':'⚠️ Below Compliance Threshold'}
        </span>
        <span className="badge bg-secondary">133 reports · 5 signers</span>
      </div>
      <div className="text-muted small mb-3">
        AI-generated EEG report signature workflow — sign / reject / pending tracking with audit trail.
        Regulation: <strong>21 CFR Part 11</strong> + IEC 62304 + ISO 13485.
      </div>

      <div className="row g-2 mb-3">
        <KpiCard label="Total Reports" value={kpi.total_reports} color="secondary"/>
        <KpiCard label="Signed" value={kpi.signed} sub={`${kpi.sign_rate_pct}%`} color="success"/>
        <KpiCard label="Pending" value={kpi.pending} sub={kpi.pending>=thr.pending_alert_count?'⚠️ Alert':''} color="warning"/>
        <KpiCard label="Rejected" value={kpi.rejected} sub={`${kpi.reject_rate_pct}%`} color={kpi.reject_rate_pct>=thr.reject_rate_alert_pct?'danger':'secondary'}/>
        <KpiCard label="Avg Turnaround" value={`${kpi.avg_turnaround_hours}h`} sub={kpi.avg_turnaround_hours>thr.max_turnaround_hours?'⚠️ Over 72h':'≤72h OK'} color={kpi.avg_turnaround_hours>thr.max_turnaround_hours?'danger':'info'}/>
        <KpiCard label="Sign Rate" value={`${kpi.sign_rate_pct}%`} sub={kpi.cfr11_compliant?'≥60% ✅':'<60% ⚠️'} color={kpi.cfr11_compliant?'success':'danger'}/>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t=>(
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab==='overview' && (
        <div className="row g-3">
          <div className="col-md-5">
            <div className="card">
              <div className="card-header py-2"><strong>📊 By Disease</strong></div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Disease</th><th>Total</th><th>Signed</th><th>Pending</th><th>Rejected</th><th>Sign %</th></tr></thead>
                  <tbody>
                    {ov.disease_breakdown.map(d=>(
                      <tr key={d.disease}>
                        <td className="text-capitalize">{d.disease.replace('_',' ')}</td>
                        <td>{d.total}</td>
                        <td><span className="badge bg-success">{d.signed}</span></td>
                        <td><span className="badge bg-warning text-dark">{d.pending}</span></td>
                        <td><span className="badge bg-danger">{d.rejected}</span></td>
                        <td>{d.total>0?Math.round(d.signed/d.total*100):0}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-7">
            <div className="card">
              <div className="card-header py-2"><strong>🕐 Recent Activity (last 10)</strong></div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Patient</th><th>Disease</th><th>Status</th><th>Signer</th><th>Signed At</th></tr></thead>
                  <tbody>
                    {ov.recent_activity.map(r=>(
                      <tr key={r.analysis_id}>
                        <td><code>{r.patient_id}</code></td>
                        <td className="text-capitalize">{r.disease.replace('_',' ')}</td>
                        <td><StatusBadge status={r.status}/></td>
                        <td><small>{r.signer_name}</small></td>
                        <td><small>{r.signed_at?.replace('T',' ')}</small></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-12">
            <div className="card">
              <div className="card-header py-2"><strong>📈 AI Confidence vs Signature Status</strong></div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Confidence Bucket</th><th>Total</th><th>Signed</th><th>Pending</th><th>Rejected</th><th>Sign Rate</th></tr></thead>
                  <tbody>
                    {(bd?.confidence_vs_status||[]).map(c=>(
                      <tr key={c.bucket}>
                        <td><strong>{c.bucket}</strong></td>
                        <td>{c.total||0}</td>
                        <td><span className="badge bg-success">{c.signed||0}</span></td>
                        <td><span className="badge bg-warning text-dark">{c.pending||0}</span></td>
                        <td><span className="badge bg-danger">{c.rejected||0}</span></td>
                        <td>
                          <MiniBar value={c.signed||0} max={c.total||1} color="success"/>
                          <small>{c.total>0?Math.round((c.signed||0)/c.total*100):0}%</small>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {tab==='reports' && (
        <div>
          <div className="row g-2 mb-3">
            <div className="col-md-6">
              <input className="form-control form-control-sm"
                placeholder="Search patient / disease / signer…"
                value={search} onChange={e=>setSearch(e.target.value)}/>
            </div>
            <div className="col-md-3">
              <select className="form-select form-select-sm"
                value={filterStatus} onChange={e=>setFilterStatus(e.target.value)}>
                <option value="all">All Statuses</option>
                <option value="signed">Signed</option>
                <option value="pending">Pending</option>
                <option value="rejected">Rejected</option>
              </select>
            </div>
            <div className="col-md-3 text-muted small d-flex align-items-center">
              Showing {filtered.length} of {allReports.length} reports
            </div>
          </div>
          <div className="card">
            <div className="card-body p-0" style={{overflowX:'auto'}}>
              <table className="table table-sm table-hover mb-0">
                <thead>
                  <tr>
                    <th>#</th><th>Patient</th><th>Disease</th><th>Prediction</th>
                    <th>Conf</th><th>Signal</th><th>Status</th><th>Signer</th>
                    <th>Signed At</th><th>Hash</th><th>Reject Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.slice(0,100).map(r=>(
                    <tr key={r.analysis_id}>
                      <td><small>{r.analysis_id}</small></td>
                      <td><code>{r.patient_id}</code></td>
                      <td className="text-capitalize"><small>{r.disease.replace('_',' ')}</small></td>
                      <td><small>{r.predicted_label}</small></td>
                      <td><span className={`badge bg-${r.confidence>=0.85?'success':r.confidence>=0.70?'info':'warning'} text-dark`}>{(r.confidence*100).toFixed(0)}%</span></td>
                      <td><small>{r.signal_quality}</small></td>
                      <td><StatusBadge status={r.status}/></td>
                      <td><small>{r.signer_name}</small></td>
                      <td><small>{r.signed_at?.replace('T',' ')||'—'}</small></td>
                      <td><code style={{fontSize:'10px'}}>{r.signature_hash||'—'}</code></td>
                      <td><small className="text-danger">{r.reject_reason||'—'}</small></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {tab==='signers' && (
        <div className="row g-3">
          {ov.signer_metrics.map(s=>(
            <div key={s.signer_id} className="col-md-4">
              <div className="card h-100">
                <div className="card-header py-2">
                  <strong>{s.signer_name}</strong>
                  <span className="badge bg-secondary ms-2">{s.signer_role}</span>
                </div>
                <div className="card-body">
                  <div className="row text-center mb-2">
                    <div className="col-4">
                      <div className="fw-bold text-success">{s.signed}</div>
                      <div className="small text-muted">Signed</div>
                    </div>
                    <div className="col-4">
                      <div className="fw-bold text-warning">{s.pending}</div>
                      <div className="small text-muted">Pending</div>
                    </div>
                    <div className="col-4">
                      <div className="fw-bold text-danger">{s.rejected}</div>
                      <div className="small text-muted">Rejected</div>
                    </div>
                  </div>
                  <div className="mb-1">
                    <small className="text-muted">Sign Rate</small>
                    <MiniBar value={s.signed} max={s.total} color="success"/>
                    <small>{s.total>0?Math.round(s.signed/s.total*100):0}%</small>
                  </div>
                  <div className="text-muted small">Total assigned: {s.total}</div>
                </div>
              </div>
            </div>
          ))}
          <div className="col-12">
            <div className="card">
              <div className="card-header py-2"><strong>⛔ Rejection Reason Distribution</strong></div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Reason</th><th>Count</th><th>Distribution</th></tr></thead>
                  <tbody>
                    {(bd?.reject_reasons||[]).map((r,i)=>(
                      <tr key={i}>
                        <td>{r.reason}</td>
                        <td>{r.count}</td>
                        <td style={{width:'200px'}}>
                          <MiniBar value={r.count} max={Math.max(...(bd?.reject_reasons||[{count:1}]).map(x=>x.count))} color="danger"/>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {tab==='compliance' && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card">
              <div className="card-header py-2"><strong>⚖️ 21 CFR Part 11 Compliance Status</strong></div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Metric</th><th>Target</th><th>Actual</th><th>Status</th></tr></thead>
                  <tbody>
                    <tr>
                      <td>Sign Rate</td>
                      <td>≥{thr.sign_rate_min_pct}%</td>
                      <td>{kpi.sign_rate_pct}%</td>
                      <td><span className={`badge bg-${kpi.sign_rate_pct>=thr.sign_rate_min_pct?'success':'danger'}`}>{kpi.sign_rate_pct>=thr.sign_rate_min_pct?'PASS':'FAIL'}</span></td>
                    </tr>
                    <tr>
                      <td>Reject Rate</td>
                      <td>&lt;{thr.reject_rate_alert_pct}%</td>
                      <td>{kpi.reject_rate_pct}%</td>
                      <td><span className={`badge bg-${kpi.reject_rate_pct<thr.reject_rate_alert_pct?'success':'danger'}`}>{kpi.reject_rate_pct<thr.reject_rate_alert_pct?'PASS':'FAIL'}</span></td>
                    </tr>
                    <tr>
                      <td>Avg Turnaround</td>
                      <td>≤{thr.max_turnaround_hours}h</td>
                      <td>{kpi.avg_turnaround_hours}h</td>
                      <td><span className={`badge bg-${kpi.avg_turnaround_hours<=thr.max_turnaround_hours?'success':'danger'}`}>{kpi.avg_turnaround_hours<=thr.max_turnaround_hours?'PASS':'FAIL'}</span></td>
                    </tr>
                    <tr>
                      <td>Pending Queue</td>
                      <td>&lt;{thr.pending_alert_count}</td>
                      <td>{kpi.pending}</td>
                      <td><span className={`badge bg-${kpi.pending<thr.pending_alert_count?'success':'warning'}`}>{kpi.pending<thr.pending_alert_count?'PASS':'ALERT'}</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card">
              <div className="card-header py-2"><strong>📜 Applicable Standards</strong></div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Standard</th><th>Body</th><th>Relevance</th></tr></thead>
                  <tbody>
                    {(defs?.standards||[]).map((s,i)=>(
                      <tr key={i}>
                        <td><strong>{s.standard}</strong></td>
                        <td><span className="badge bg-secondary">{s.body}</span></td>
                        <td><small>{s.relevance}</small></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {tab==='definitions' && (
        <div className="row g-3">
          <div className="col-md-8">
            <div className="card">
              <div className="card-header py-2"><strong>📖 Concepts</strong></div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th style={{width:'200px'}}>Term</th><th>Definition</th></tr></thead>
                  <tbody>
                    {(defs?.concepts||[]).map((c,i)=>(
                      <tr key={i}>
                        <td><strong>{c.term}</strong></td>
                        <td><small>{c.definition}</small></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="card mb-3">
              <div className="card-header py-2"><strong>🎯 Thresholds</strong></div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Metric</th><th>Target</th><th>Alert</th></tr></thead>
                  <tbody>
                    {(defs?.thresholds||[]).map((t,i)=>(
                      <tr key={i}>
                        <td><strong>{t.metric}</strong></td>
                        <td className="text-success">{t.target}</td>
                        <td className="text-danger">{t.alert}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card">
              <div className="card-header py-2"><strong>📚 References</strong></div>
              <div className="card-body">
                <ol className="small mb-0">
                  {(defs?.references||[]).map((r,i)=><li key={i} className="mb-1">{r}</li>)}
                </ol>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
