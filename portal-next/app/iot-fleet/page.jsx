'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KPICard({label, value, color='primary'}) {
  return (
    <div className="col-6 col-md-2 mb-2">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className={`h5 mb-0 text-${color}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function DistBar({items, total, colorFn}) {
  return items.map(({name, count, pct}) => {
    const pctVal = pct != null ? (pct * 100).toFixed(0) : ((count / total) * 100).toFixed(0);
    return (
      <div key={name} className="d-flex align-items-center mb-2">
        <span className="me-2 small text-capitalize" style={{minWidth:'110px'}}>{name.replace(/_/g,' ')}</span>
        <div className="flex-grow-1 me-2">
          <div className="progress" style={{height:'20px'}}>
            <div className={`progress-bar bg-${colorFn ? colorFn(name) : 'primary'}`} style={{width:`${pctVal}%`}}>
              {count} ({pctVal}%)
            </div>
          </div>
        </div>
      </div>
    );
  });
}

export default function IoTFleetDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [filter, setFilter] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/iot-fleet/overview`).then(r => r.json()),
      fetch(`${API}/api/iot-fleet/breakdown`).then(r => r.json()),
      fetch(`${API}/api/iot-fleet/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading IoT fleet data...</div>;

  const k = ov.kpis;
  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'devices', label:'Device Table'},
    {id:'alerts', label:'Alerts'},
    {id:'definitions', label:'Definitions'},
  ];

  const statusColor = s => ({online:'success', offline:'danger', maintenance:'warning'}[s] || 'secondary');
  const severityColor = s => ({critical:'danger', warning:'warning', info:'info'}[s] || 'secondary');

  const filteredDevices = (bd?.device_table || []).filter(d => {
    const q = filter.toLowerCase();
    return !q || d.device_id?.toLowerCase().includes(q) || d.type?.toLowerCase().includes(q) ||
      d.patient_id?.toLowerCase().includes(q) || d.status?.toLowerCase().includes(q) ||
      d.location?.toLowerCase().includes(q);
  });

  return (
    <div className="p-3">
      <h3>&#x1f4e1; IoT Fleet Dashboard</h3>
      <p className="text-muted">
        Real-time device fleet — {k.total_devices} devices · {k.unique_patients_covered} patients covered ·
        {k.total_gateways} gateways · {k.total_alerts} alerts ({k.unresolved_alerts} unresolved)
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && (
        <div>
          {/* KPI Row 1 */}
          <div className="row mb-3">
            <KPICard label="Total Devices" value={k.total_devices} color="primary" />
            <KPICard label="Online" value={k.online_devices} color="success" />
            <KPICard label="Availability" value={`${(k.device_availability_rate * 100).toFixed(0)}%`} color="info" />
            <KPICard label="Avg Battery" value={`${k.avg_battery_pct?.toFixed(1)}%`} color={k.avg_battery_pct < 30 ? 'danger' : 'success'} />
            <KPICard label="Low Battery" value={k.low_battery_devices} color="warning" />
            <KPICard label="Avg Latency" value={`${k.avg_latency_ms?.toFixed(0)} ms`} color="primary" />
          </div>
          {/* KPI Row 2 */}
          <div className="row mb-3">
            <KPICard label="Gateways" value={k.total_gateways} color="primary" />
            <KPICard label="GW Online" value={k.online_gateways} color="success" />
            <KPICard label="GW Uptime" value={`${k.avg_gateway_uptime_pct?.toFixed(1)}%`} color="info" />
            <KPICard label="Total Alerts" value={k.total_alerts} color="warning" />
            <KPICard label="Unresolved" value={k.unresolved_alerts} color="danger" />
            <KPICard label="Critical" value={k.critical_unresolved} color="danger" />
          </div>

          <div className="row mb-3">
            {/* Device Status */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Device Status</h6>
                  <DistBar items={ov.device_status_distribution} total={k.total_devices} colorFn={statusColor} />
                </div>
              </div>
            </div>

            {/* Device Types */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Device Types</h6>
                  {ov.device_type_distribution.map(({name, count}) => (
                    <div key={name} className="d-flex justify-content-between border-bottom py-1">
                      <span className="small text-capitalize">{name.replace(/_/g, ' ')}</span>
                      <span className="badge bg-primary">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Alert Severity */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Alert Severity</h6>
                  <DistBar items={ov.alert_severity_distribution} total={k.total_alerts} colorFn={severityColor} />
                </div>
              </div>
            </div>
          </div>

          <div className="row mb-3">
            {/* Location Distribution */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-body">
                  <h6>Devices by Location</h6>
                  {ov.location_distribution.map(({name, count}) => (
                    <div key={name} className="d-flex justify-content-between border-bottom py-1">
                      <span className="small">{name}</span>
                      <span className="badge bg-secondary">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Gateway Status */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-body">
                  <h6>Gateway Status</h6>
                  {ov.gateway_status_distribution.map(({name, count}) => (
                    <div key={name} className="d-flex justify-content-between border-bottom py-1">
                      <span className="small text-capitalize">{name}</span>
                      <span className={`badge bg-${statusColor(name)}`}>{count}</span>
                    </div>
                  ))}
                  <div className="mt-2 small text-muted">
                    Avg uptime: {k.avg_gateway_uptime_pct?.toFixed(1)}% ·
                    Connected devices: {k.total_connected_devices}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {tab === 'devices' && bd && (
        <div>
          <div className="d-flex align-items-center mb-3 gap-2">
            <input
              type="text"
              className="form-control"
              style={{maxWidth:'340px'}}
              placeholder="Filter by device ID, type, patient, location, status..."
              value={filter}
              onChange={e => setFilter(e.target.value)}
            />
            <span className="text-muted small">{filteredDevices.length} of {bd.device_table.length}</span>
          </div>
          <div className="card shadow-sm">
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-striped mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Device ID</th>
                      <th>Type</th>
                      <th>Status</th>
                      <th>Patient</th>
                      <th>Location</th>
                      <th>Battery %</th>
                      <th>Signal dBm</th>
                      <th>Latency ms</th>
                      <th>Firmware</th>
                      <th>Last Seen</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredDevices.map(d => (
                      <tr key={d.device_id}>
                        <td className="fw-semibold small">{d.device_id}</td>
                        <td className="small text-capitalize">{d.type?.replace(/_/g,' ')}</td>
                        <td>
                          <span className={`badge bg-${statusColor(d.status)}`}>{d.status}</span>
                        </td>
                        <td className="small">{d.patient_id || '—'}</td>
                        <td className="small">{d.location}</td>
                        <td className="small">
                          <span className={d.battery_pct < 20 ? 'text-danger fw-bold' : ''}>
                            {d.battery_pct?.toFixed(1)}%
                          </span>
                        </td>
                        <td className="small">{d.signal_dbm?.toFixed(1)}</td>
                        <td className="small">{d.latency_ms?.toFixed(1)}</td>
                        <td className="small font-monospace">{d.firmware}</td>
                        <td className="small">{d.last_seen ? d.last_seen.replace('T',' ').slice(0,16) : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {tab === 'alerts' && bd && (
        <div>
          <div className="row mb-3">
            <div className="col-md-4">
              <div className={`card border-danger shadow-sm`}>
                <div className="card-body text-center">
                  <div className="h3 text-danger">{k.critical_unresolved}</div>
                  <div className="text-muted small">Critical Unresolved</div>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card border-warning shadow-sm">
                <div className="card-body text-center">
                  <div className="h3 text-warning">{k.unresolved_alerts}</div>
                  <div className="text-muted small">Total Unresolved</div>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card shadow-sm">
                <div className="card-body text-center">
                  <div className="h3 text-info">{k.unacknowledged_alerts}</div>
                  <div className="text-muted small">Unacknowledged</div>
                </div>
              </div>
            </div>
          </div>
          <div className="card shadow-sm">
            <div className="card-body">
              <h6>Alert Severity Distribution</h6>
              {ov.alert_severity_distribution.map(({name, count, pct}) => (
                <div key={name} className="d-flex align-items-center mb-2">
                  <span className={`badge bg-${severityColor(name)} me-2`} style={{minWidth:'70px'}}>{name}</span>
                  <div className="flex-grow-1 me-2">
                    <div className="progress" style={{height:'22px'}}>
                      <div className={`progress-bar bg-${severityColor(name)}`} style={{width:`${(pct*100).toFixed(0)}%`}}>
                        {count} ({(pct*100).toFixed(0)}%)
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div>
          {defs.device_types && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6>Device Types</h6>
                {Object.entries(defs.device_types).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <strong className="text-capitalize">{k.replace(/_/g,' ')}:</strong>{' '}
                    <span className="small text-muted">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {defs.connectivity_modes && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6>Connectivity Modes</h6>
                {Object.entries(defs.connectivity_modes).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <strong className="text-capitalize">{k}:</strong>{' '}
                    <span className="small text-muted">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {defs.alert_types && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6>Alert Types</h6>
                {Object.entries(defs.alert_types).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <strong className="text-capitalize">{k.replace(/_/g,' ')}:</strong>{' '}
                    <span className="small text-muted">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {defs.severity_levels && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6>Severity Levels</h6>
                {Object.entries(defs.severity_levels).map(([k, v]) => (
                  <div key={k} className="mb-1">
                    <span className={`badge bg-${severityColor(k)} me-2`}>{k}</span>
                    <span className="small">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {defs.kpi_definitions && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6>KPI Definitions</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-striped">
                    <thead><tr><th>KPI</th><th>Definition</th><th>Good Threshold</th></tr></thead>
                    <tbody>
                      {defs.kpi_definitions.map(d => (
                        <tr key={d.kpi}>
                          <td className="fw-semibold small">{d.kpi}</td>
                          <td className="small">{d.definition}</td>
                          <td className="small text-success">{d.good_threshold || '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
