'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SEV_COLOR = {critical:'danger', warning:'warning', info:'info'};
const STATUS_COLOR = {online:'success', offline:'secondary'};

function KpiCard({label, value, color='primary', sub}){
  return (
    <div className="col-6 col-md-2 mb-2">
      <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
        <div className={`h5 mb-0 text-${color}`}>{value}</div>
        <div className="text-muted small">{label}</div>
        {sub && <div className="text-muted" style={{fontSize:'0.7rem'}}>{sub}</div>}
      </div></div>
    </div>
  );
}

export default function DeviceTelemetryDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [filter, setFilter] = useState('all');

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/device-telemetry/overview`).then(r=>r.json()),
      fetch(`${API}/api/device-telemetry/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/device-telemetry/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading device telemetry data...</div>;

  const k = ov.kpis;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'iot', label:'IoT Devices'},
    {id:'wearables', label:'Wearables'},
    {id:'alerts', label:'Alerts'},
    {id:'gateway', label:'Gateways'},
    {id:'definitions', label:'Definitions'},
  ];

  const batteryColor = pct => pct < 20 ? 'danger' : pct < 40 ? 'warning' : 'success';
  const signalColor = dbm => dbm < -70 ? 'danger' : dbm < -60 ? 'warning' : 'success';

  // filtered IoT devices
  const iotDevices = bd?.iot_devices || [];
  const filteredIoT = filter === 'all' ? iotDevices
    : filter === 'low_batt' ? iotDevices.filter(d => d.battery_pct < 20)
    : filter === 'offline' ? iotDevices.filter(d => d.status === 'offline')
    : filter === 'weak_signal' ? iotDevices.filter(d => d.signal_strength_dbm < -65)
    : iotDevices;

  const wearDevices = bd?.wearable_devices || [];
  const filteredWear = filter === 'all' ? wearDevices
    : filter === 'low_batt' ? wearDevices.filter(d => d.battery_pct < 20)
    : filter === 'offline' ? wearDevices.filter(d => d.status === 'offline')
    : filter === 'weak_signal' ? wearDevices.filter(d => d.signal_strength_dbm < -65)
    : wearDevices;

  return (<div className="p-3">
    <h3>📡 Device Telemetry Dashboard</h3>
    <p className="text-muted small">
      Fleet-wide health monitoring — battery, signal, connectivity, alerts &amp; gateways
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* ── OVERVIEW ── */}
    {tab==='overview' && <div>
      <div className="row mb-3">
        <KpiCard label="Total Devices" value={k.total_devices} color="primary"/>
        <KpiCard label="Online" value={k.online_count} color="success" sub={`${(k.online_pct*100).toFixed(0)}%`}/>
        <KpiCard label="Offline" value={k.offline_count} color="secondary" sub={`${(k.offline_pct*100).toFixed(0)}%`}/>
        <KpiCard label="Avg Battery" value={`${k.avg_battery.toFixed(0)}%`} color={k.avg_battery<40?'warning':'success'}/>
        <KpiCard label="Low Battery" value={k.low_battery_count} color="warning"/>
        <KpiCard label="Weak Signal" value={k.weak_signal_count} color="warning"/>
        <KpiCard label="Avg Latency" value={`${k.avg_latency_ms.toFixed(0)}ms`} color="info"/>
        <KpiCard label="Total Alerts" value={k.total_alerts} color="danger"/>
        <KpiCard label="Unresolved" value={k.unresolved_alerts} color="danger"/>
        <KpiCard label="IoT Devices" value={k.total_iot} color="primary"/>
        <KpiCard label="Wearables" value={k.total_wearable} color="info"/>
        <KpiCard label="Resolved Alerts" value={k.resolved_alerts} color="success"/>
      </div>

      <div className="row mb-3">
        {/* Severity breakdown */}
        <div className="col-md-4">
          <div className="card shadow-sm h-100"><div className="card-body">
            <h6>Alert Severity</h6>
            {ov.severity_breakdown.map(({severity,count,pct})=>(
              <div key={severity} className="d-flex align-items-center mb-2">
                <span className={`badge bg-${SEV_COLOR[severity]||'secondary'} me-2`} style={{minWidth:'70px'}}>{severity}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className={`progress-bar bg-${SEV_COLOR[severity]||'secondary'}`} style={{width:`${(pct*100).toFixed(0)}%`}}>
                      {count}
                    </div>
                  </div>
                </div>
                <span className="small text-muted">{(pct*100).toFixed(0)}%</span>
              </div>
            ))}
          </div></div>
        </div>

        {/* Battery distribution */}
        <div className="col-md-4">
          <div className="card shadow-sm h-100"><div className="card-body">
            <h6>Battery Distribution</h6>
            {ov.battery_distribution.map(({bucket,count})=>{
              const maxCount = Math.max(...ov.battery_distribution.map(b=>b.count));
              const pct = (count/maxCount*100).toFixed(0);
              const isLow = bucket==='0-20'||bucket==='20-40';
              return <div key={bucket} className="d-flex align-items-center mb-2">
                <span className="me-2 small text-muted" style={{minWidth:'45px'}}>{bucket}%</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className={`progress-bar bg-${isLow?'danger':'success'}`} style={{width:`${pct}%`}}>
                      {count}
                    </div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>

        {/* Device type breakdown */}
        <div className="col-md-4">
          <div className="card shadow-sm h-100"><div className="card-body">
            <h6>By Device Type</h6>
            <div style={{maxHeight:'200px', overflowY:'auto'}}>
              {ov.device_type_breakdown.map(({device_type,count,pct})=>(
                <div key={device_type} className="d-flex justify-content-between border-bottom py-1">
                  <span className="small">{device_type.replace(/_/g,' ')}</span>
                  <span>
                    <span className="badge bg-primary me-1">{count}</span>
                    <span className="text-muted small">{(pct*100).toFixed(0)}%</span>
                  </span>
                </div>
              ))}
            </div>
          </div></div>
        </div>
      </div>

      {/* Recent alerts */}
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Recent Alerts</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead><tr>
              <th>Alert Type</th><th>Severity</th><th>Device</th>
              <th>Resolved</th><th>Acknowledged</th><th>Timestamp</th>
            </tr></thead>
            <tbody>
              {ov.recent_alerts.map((a,i)=>(
                <tr key={i}>
                  <td className="small">{a.alert_type.replace(/_/g,' ')}</td>
                  <td><span className={`badge bg-${SEV_COLOR[a.severity]||'secondary'}`}>{a.severity}</span></td>
                  <td className="small fw-semibold">{a.device_id}</td>
                  <td>{a.resolved ? <span className="text-success">&#10003;</span> : <span className="text-danger">&#10007;</span>}</td>
                  <td>{a.acknowledged ? <span className="text-success">&#10003;</span> : <span className="text-muted">—</span>}</td>
                  <td className="small text-muted">{a.timestamp?.slice(0,16)?.replace('T',' ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── IOT DEVICES ── */}
    {tab==='iot' && bd && <div>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5>IoT Device Fleet ({k.total_iot} devices)</h5>
        <div className="btn-group btn-group-sm">
          {[['all','All'],['low_batt','Low Battery'],['offline','Offline'],['weak_signal','Weak Signal']].map(([v,l])=>(
            <button key={v} className={`btn btn-${filter===v?'primary':'outline-secondary'}`} onClick={()=>setFilter(v)}>{l}</button>
          ))}
        </div>
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-striped table-hover">
          <thead className="table-dark"><tr>
            <th>Device ID</th><th>Type</th><th>Patient</th>
            <th>Battery</th><th>Signal</th><th>Status</th>
            <th>Latency</th><th>Firmware</th><th>Location</th><th>Last Seen</th>
          </tr></thead>
          <tbody>
            {filteredIoT.map(d=>(
              <tr key={d.device_id}>
                <td className="fw-semibold small">{d.device_id}</td>
                <td className="small">{d.device_type?.replace(/_/g,' ')}</td>
                <td className="small">{d.patient_id}</td>
                <td>
                  <div className="d-flex align-items-center gap-1">
                    <div className="progress flex-grow-1" style={{height:'10px',minWidth:'50px'}}>
                      <div className={`progress-bar bg-${batteryColor(d.battery_pct)}`} style={{width:`${d.battery_pct}%`}}></div>
                    </div>
                    <span className="small">{d.battery_pct?.toFixed(0)}%</span>
                  </div>
                </td>
                <td className={`small text-${signalColor(d.signal_strength_dbm)}`}>{d.signal_strength_dbm?.toFixed(1)} dBm</td>
                <td><span className={`badge bg-${STATUS_COLOR[d.status]||'secondary'}`}>{d.status}</span></td>
                <td className="small">{d.latency_ms?.toFixed(0)}ms</td>
                <td className="small text-muted">{d.firmware_version}</td>
                <td className="small">{d.location}</td>
                <td className="small text-muted">{d.last_seen?.slice(0,10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {filteredIoT.length===0 && <div className="alert alert-info">No devices match filter.</div>}
    </div>}

    {/* ── WEARABLES ── */}
    {tab==='wearables' && bd && <div>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5>Wearable Devices ({k.total_wearable} devices)</h5>
        <div className="btn-group btn-group-sm">
          {[['all','All'],['low_batt','Low Battery'],['offline','Offline'],['weak_signal','Weak Signal']].map(([v,l])=>(
            <button key={v} className={`btn btn-${filter===v?'primary':'outline-secondary'}`} onClick={()=>setFilter(v)}>{l}</button>
          ))}
        </div>
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-striped table-hover">
          <thead className="table-dark"><tr>
            <th>Device ID</th><th>Type</th><th>Patient</th>
            <th>Battery</th><th>Signal</th><th>Status</th>
            <th>Latency</th><th>Brand</th><th>Features</th><th>Last Seen</th>
          </tr></thead>
          <tbody>
            {filteredWear.map(d=>(
              <tr key={d.device_id}>
                <td className="fw-semibold small">{d.device_id}</td>
                <td className="small">{d.device_type?.replace(/_/g,' ')}</td>
                <td className="small">{d.patient_id}</td>
                <td>
                  <div className="d-flex align-items-center gap-1">
                    <div className="progress flex-grow-1" style={{height:'10px',minWidth:'50px'}}>
                      <div className={`progress-bar bg-${batteryColor(d.battery_pct)}`} style={{width:`${d.battery_pct}%`}}></div>
                    </div>
                    <span className="small">{d.battery_pct?.toFixed(0)}%</span>
                  </div>
                </td>
                <td className={`small text-${signalColor(d.signal_strength_dbm)}`}>{d.signal_strength_dbm?.toFixed(1)} dBm</td>
                <td><span className={`badge bg-${STATUS_COLOR[d.status]||'secondary'}`}>{d.status}</span></td>
                <td className="small">{d.latency_ms?.toFixed(0)}ms</td>
                <td className="small">{d.brand || '—'}</td>
                <td className="small">{d.features ? d.features.slice(0,2).join(', ') + (d.features.length>2?'…':'') : '—'}</td>
                <td className="small text-muted">{d.last_seen?.slice(0,10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {filteredWear.length===0 && <div className="alert alert-info">No devices match filter.</div>}
    </div>}

    {/* ── ALERTS ── */}
    {tab==='alerts' && bd && <div>
      <h5>Alerts by Type</h5>
      {bd.per_alert_type && <div className="row mb-3">
        {bd.per_alert_type.map(({alert_type,count,unresolved,resolved})=>(
          <div key={alert_type} className="col-md-4 mb-2">
            <div className="card shadow-sm"><div className="card-body py-2">
              <div className="fw-semibold small">{alert_type.replace(/_/g,' ')}</div>
              <div className="d-flex gap-2 mt-1">
                <span className="badge bg-secondary">Total: {count}</span>
                <span className="badge bg-danger">Unresolved: {unresolved}</span>
                <span className="badge bg-success">Resolved: {resolved}</span>
              </div>
            </div></div>
          </div>
        ))}
      </div>}
    </div>}

    {/* ── GATEWAYS ── */}
    {tab==='gateway' && bd && <div>
      <h5>Gateway Health</h5>
      <div className="table-responsive">
        <table className="table table-sm table-striped">
          <thead className="table-dark"><tr>
            <th>Gateway ID</th><th>Location</th><th>Status</th>
            <th>Connected Devices</th><th>Uptime</th><th>Last Seen</th>
          </tr></thead>
          <tbody>
            {(bd.gateway_health||[]).map(g=>(
              <tr key={g.gateway_id}>
                <td className="fw-semibold small">{g.gateway_id}</td>
                <td className="small">{g.location}</td>
                <td><span className={`badge bg-${STATUS_COLOR[g.status]||'secondary'}`}>{g.status}</span></td>
                <td>{g.connected_devices}</td>
                <td className="small">{g.uptime_pct?.toFixed(1)}%</td>
                <td className="small text-muted">{g.last_seen?.slice(0,16)?.replace('T',' ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>}

    {/* ── DEFINITIONS ── */}
    {tab==='definitions' && defs && <div>
      <h5>Telemetry Reference</h5>
      <div className="row">
        {defs.signal_strength_thresholds && <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100"><div className="card-body">
            <h6>Signal Strength Thresholds</h6>
            {defs.signal_strength_thresholds.map(t=>(
              <div key={t.label} className="mb-1">
                <span className={`badge bg-${t.color||'secondary'} me-2`}>{t.label}</span>
                <span className="small">{t.range}</span>
                {t.action && <div className="text-muted small ms-3">{t.action}</div>}
              </div>
            ))}
          </div></div>
        </div>}
        {defs.battery_thresholds && <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100"><div className="card-body">
            <h6>Battery Thresholds</h6>
            {defs.battery_thresholds.map(t=>(
              <div key={t.label} className="mb-1">
                <span className={`badge bg-${t.color||'secondary'} me-2`}>{t.label}</span>
                <span className="small">{t.range}</span>
                {t.action && <div className="text-muted small ms-3">{t.action}</div>}
              </div>
            ))}
          </div></div>
        </div>}
        {defs.alert_severity_definitions && <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100"><div className="card-body">
            <h6>Alert Severity Definitions</h6>
            {defs.alert_severity_definitions.map(s=>(
              <div key={s.level} className="mb-2">
                <span className={`badge bg-${SEV_COLOR[s.level]||'secondary'} me-2`}>{s.level}</span>
                <span className="small">{s.description}</span>
                {s.response_time && <div className="text-muted small ms-3">Response: {s.response_time}</div>}
              </div>
            ))}
          </div></div>
        </div>}
        {defs.device_types_glossary && <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100"><div className="card-body">
            <h6>Device Types Glossary</h6>
            <div style={{maxHeight:'200px',overflowY:'auto'}}>
              {defs.device_types_glossary.map(t=>(
                <div key={t.type} className="mb-1 border-bottom pb-1">
                  <strong className="small">{t.type?.replace(/_/g,' ')}</strong>
                  <div className="text-muted small">{t.description}</div>
                </div>
              ))}
            </div>
          </div></div>
        </div>}
        {defs.clinical_importance && <div className="col-md-12 mb-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Clinical Importance</h6>
            <p className="small mb-0">{defs.clinical_importance}</p>
          </div></div>
        </div>}
      </div>
    </div>}
  </div>);
}
