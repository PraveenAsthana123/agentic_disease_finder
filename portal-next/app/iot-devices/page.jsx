'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function IoTDevicesDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/iot-devices/overview`).then(r=>r.json()),
      fetch(`${API}/api/iot-devices/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/iot-devices/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading IoT device fleet data...</div>;

  const s = ov.summary;
  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'fleet', label:'Device Fleet'},
    {id:'modes', label:'Connectivity Matrix'},
    {id:'data', label:'Data Streams'},
    {id:'offline', label:'Offline Strategy'},
    {id:'definitions', label:'Definitions'},
  ];

  const statusBadge = st => ({built:'success',partial:'warning',planned:'secondary'}[st]||'info');

  return (<div className="p-3">
    <h3>IoT Devices &amp; Wearable Fleet Dashboard</h3>
    <p className="text-muted">
      Device fleet management &mdash; {s.total_devices} devices, {s.device_types} types,
      {' '}{s.unique_data_streams} data streams, {s.alert_capable} alert-capable
    </p>
    {s.honest_note && <div className="alert alert-info small py-1 px-2">{s.honest_note}</div>}

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          ['Total Devices', s.total_devices, 'primary'],
          ['Device Types', s.device_types, 'info'],
          ['Built', s.built, 'success'],
          ['Partial', s.partial, 'warning'],
          ['Planned', s.planned, 'secondary'],
          ['Data Streams', s.unique_data_streams, 'primary'],
          ['Alert-Capable', s.alert_capable, 'danger'],
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
        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Status Distribution</h6>
            {ov.status_distribution.map(({name,value})=>{
              const pct = ((value/s.total_devices)*100).toFixed(0);
              return <div key={name} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'70px'}}>{name}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${statusBadge(name.toLowerCase())}`} style={{width:`${pct}%`}}>
                      {value} ({pct}%)
                    </div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>By Device Type</h6>
            {ov.type_distribution.map(({name,value})=>
              <div key={name} className="d-flex justify-content-between border-bottom py-1">
                <span className="small">{name}</span>
                <span className="badge bg-primary">{value}</span>
              </div>
            )}
          </div></div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Connectivity Modes</h6>
            {ov.mode_distribution.map(({name,value})=>
              <div key={name} className="d-flex justify-content-between border-bottom py-1">
                <span className="small">{name}</span>
                <span className="badge bg-info">{value}</span>
              </div>
            )}
            <div className="mt-2 small text-muted">
              <strong>Alert pipeline:</strong> {ov.alert_pipeline}
            </div>
          </div></div>
        </div>
      </div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Connectivity Model</h6>
        <div className="row">
          {Object.entries(ov.connectivity_model).map(([mode,desc])=>
            <div key={mode} className="col-md-4 mb-2">
              <div className="border rounded p-2 h-100">
                <strong className="text-capitalize">{mode}</strong>
                <div className="small text-muted mt-1">{desc}</div>
              </div>
            </div>
          )}
        </div>
      </div></div>
    </div>}

    {tab==='fleet' && bd && <div>
      <h5>Device Fleet by Type</h5>
      {Object.entries(bd.by_type).map(([type, devices])=>
        <div key={type} className="card shadow-sm mb-3"><div className="card-body">
          <h6 className="text-capitalize">{type} ({devices.length})</h6>
          <div className="table-responsive">
            <table className="table table-sm table-striped">
              <thead><tr>
                <th>Device</th><th>Channels</th><th>Modes</th><th>Data Streams</th>
                <th>Online</th><th>Offline</th><th>Alert</th><th>Status</th>
              </tr></thead>
              <tbody>
                {devices.map(d=><tr key={d.id}>
                  <td className="fw-semibold">{d.name}</td>
                  <td>{d.channels || '—'}</td>
                  <td>{d.modes.join(', ')}</td>
                  <td><span className="small">{d.data.join(', ')}</span></td>
                  <td className="small">{d.online || '—'}</td>
                  <td className="small">{d.offline || '—'}</td>
                  <td className="small">{d.alert || '—'}</td>
                  <td><span className={`badge bg-${statusBadge(d.status)}`}>{d.status}</span></td>
                </tr>)}
              </tbody>
            </table>
          </div>
        </div></div>
      )}
    </div>}

    {tab==='modes' && bd && <div>
      <h5>Connectivity Mode Matrix</h5>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead><tr>
              <th>Device</th><th>Status</th>
              {bd.mode_matrix.modes.map(m=><th key={m} className="text-capitalize text-center">{m}</th>)}
            </tr></thead>
            <tbody>
              {bd.mode_matrix.rows.map(r=><tr key={r.id}>
                <td className="fw-semibold">{r.device}</td>
                <td><span className={`badge bg-${statusBadge(r.status)}`}>{r.status}</span></td>
                {bd.mode_matrix.modes.map(m=><td key={m} className="text-center">
                  {r[m] ? <span className="text-success fw-bold">&#10003;</span> : <span className="text-muted">—</span>}
                </td>)}
              </tr>)}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='data' && bd && <div>
      <h5>Data Stream Matrix</h5>
      <p className="text-muted small">{bd.data_matrix.streams.length} unique data streams across {bd.data_matrix.rows.length} devices</p>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <div className="table-responsive">
          <table className="table table-sm table-bordered" style={{fontSize:'0.75rem'}}>
            <thead><tr>
              <th style={{position:'sticky',left:0,background:'#fff',zIndex:1}}>Device</th>
              {bd.data_matrix.streams.map(st=><th key={st} className="text-center" style={{writingMode:'vertical-rl',transform:'rotate(180deg)',maxWidth:'30px',whiteSpace:'nowrap'}}>{st}</th>)}
            </tr></thead>
            <tbody>
              {bd.data_matrix.rows.map(r=><tr key={r.id}>
                <td style={{position:'sticky',left:0,background:'#fff',zIndex:1}} className="fw-semibold small">{r.device}</td>
                {bd.data_matrix.streams.map(st=><td key={st} className="text-center">
                  {r[st] ? <span className="text-success">&#9679;</span> : ''}
                </td>)}
              </tr>)}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='offline' && bd && <div>
      <h5>Offline-First Strategy</h5>
      <div className="row">
        {Object.entries(bd.offline_strategy).map(([key,val])=>
          <div key={key} className="col-md-6 mb-3">
            <div className="card shadow-sm h-100"><div className="card-body">
              <h6 className="text-capitalize">{key.replace(/_/g,' ')}</h6>
              <p className="mb-0 small">{val}</p>
            </div></div>
          </div>
        )}
      </div>
    </div>}

    {tab==='definitions' && defs && <div>
      <h5>Definitions &amp; Legend</h5>
      {defs.connectivity_model && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Connectivity Modes</h6>
        {Object.entries(defs.connectivity_model).map(([k,v])=>
          <div key={k} className="mb-2">
            <strong className="text-capitalize">{v.label || k}:</strong>{' '}
            <span className="small">{v.description || v}</span>
          </div>
        )}
      </div></div>}
      {defs.status_legend && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Status Legend</h6>
        {defs.status_legend.map(s=>
          <div key={s.status} className="mb-1">
            <span className={`badge bg-${statusBadge(s.status)} me-2`}>{s.status}</span>
            <span className="small">{s.description}</span>
          </div>
        )}
      </div></div>}
      {defs.device_types && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Device Types</h6>
        {defs.device_types.map(dt=>
          <div key={dt.type} className="mb-1">
            <strong>{dt.type}:</strong> <span className="small">{dt.description}</span>
          </div>
        )}
      </div></div>}
    </div>}
  </div>);
}
