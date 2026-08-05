'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KPICard({label, value, color='primary', sub}) {
  return (
    <div className="col-6 col-md-2 mb-2">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className={`h5 mb-0 text-${color}`}>{value ?? '—'}</div>
          {sub && <div className="text-muted" style={{fontSize:'0.7rem'}}>{sub}</div>}
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function DistBar({items, total, labelKey='tier', colorFn}) {
  return (items || []).map((item) => {
    const name = item[labelKey] ?? item.version ?? item.tier ?? item.name ?? '?';
    const count = item.count ?? 0;
    const pctVal = item.pct != null ? (item.pct * 100).toFixed(0) : total ? ((count / total) * 100).toFixed(0) : 0;
    const bg = colorFn ? colorFn(name) : 'primary';
    return (
      <div key={name} className="d-flex align-items-center mb-2">
        <span className="me-2 small text-capitalize" style={{minWidth:'120px'}}>{name}</span>
        <div className="flex-grow-1 me-2">
          <div className="progress" style={{height:'20px'}}>
            <div className={`progress-bar bg-${bg}`} style={{width:`${pctVal}%`}}>
              {count} ({pctVal}%)
            </div>
          </div>
        </div>
      </div>
    );
  });
}

function StatusBadge({status}) {
  const color = status === 'online' ? 'success' : status === 'offline' ? 'danger' : 'warning';
  return <span className={`badge bg-${color}`}>{status}</span>;
}

function UptimeBadge({pct}) {
  const color = pct >= 95 ? 'success' : pct >= 90 ? 'info' : pct >= 80 ? 'warning' : 'danger';
  return <span className={`badge bg-${color}`}>{pct}%</span>;
}

export default function IoTGatewayDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/iot-gateway/overview`).then(r => r.json()),
      fetch(`${API}/api/iot-gateway/breakdown`).then(r => r.json()),
      fetch(`${API}/api/iot-gateway/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="container mt-4"><div className="alert alert-danger">{err}</div></div>;
  if (!ov) return <div className="container mt-4"><div className="text-center py-5"><div className="spinner-border text-primary"/><p className="mt-2">Loading IoT Gateway data…</p></div></div>;

  const k = ov.kpis || {};
  const tabs = ['overview', 'gateways', 'firmware', 'alerts', 'definitions'];

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-1">IoT Gateway Dashboard</h4>
      <p className="text-muted small mb-3">
        {k.total_gateways} gateways · {k.online_gateways} online · avg uptime {k.avg_uptime_pct}% · {k.total_connected_devices} devices connected
      </p>

      {/* KPI Row */}
      <div className="row g-2 mb-3">
        <KPICard label="Total Gateways" value={k.total_gateways} color="primary"/>
        <KPICard label="Online" value={k.online_gateways} color="success"/>
        <KPICard label="Availability" value={`${(k.gateway_availability_rate*100).toFixed(0)}%`} color="info"/>
        <KPICard label="Avg Uptime" value={`${k.avg_uptime_pct}%`} color="primary"/>
        <KPICard label="Devices Connected" value={k.total_connected_devices} color="secondary"/>
        <KPICard label="Unresolved Alerts" value={k.unresolved_gateway_alerts} color={k.unresolved_gateway_alerts > 0 ? 'danger' : 'success'}/>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab===t?'active':''}`} onClick={()=>setTab(t)}>
              {t.charAt(0).toUpperCase()+t.slice(1)}
            </button>
          </li>
        ))}
      </ul>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div className="row">
          <div className="col-md-5 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Uptime Tier Distribution</div>
              <div className="card-body">
                <DistBar items={ov.uptime_tier_distribution} total={k.total_gateways} labelKey="tier"
                  colorFn={t => t==='excellent'?'success':t==='good'?'info':t==='fair'?'warning':'danger'}/>
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Firmware Distribution</div>
              <div className="card-body">
                <DistBar items={ov.firmware_distribution} total={k.total_gateways} labelKey="version"
                  colorFn={v => v==='4.0.0'?'success':v==='3.2.0'?'warning':'danger'}/>
                {bd && bd.outdated_count > 0 && (
                  <div className="alert alert-warning mt-2 mb-0 py-1 small">
                    {bd.outdated_count} gateway(s) below latest firmware ({bd.latest_firmware})
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="col-md-3 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Gateway Stats</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <tbody>
                    <tr><td>Min Uptime</td><td><strong>{k.min_uptime_pct}%</strong></td></tr>
                    <tr><td>Max Uptime</td><td><strong>{k.max_uptime_pct}%</strong></td></tr>
                    <tr><td>Avg Load</td><td><strong>{k.avg_devices_per_gateway} dev/gw</strong></td></tr>
                    <tr><td>Firmware Versions</td><td><strong>{k.unique_firmware_versions}</strong></td></tr>
                    <tr><td>Critical Alerts</td><td><strong className={k.critical_alerts>0?'text-danger':''}>{k.critical_alerts}</strong></td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Location Summary</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th>Gateway</th><th>Location</th><th>Status</th><th>Uptime</th><th>Devices</th></tr>
                  </thead>
                  <tbody>
                    {(ov.location_summary||[]).map(loc => (
                      <tr key={loc.gateway_id}>
                        <td><code>{loc.gateway_id}</code></td>
                        <td>{loc.location}</td>
                        <td><StatusBadge status={loc.status}/></td>
                        <td><UptimeBadge pct={loc.uptime_pct}/></td>
                        <td>{loc.connected_devices}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Gateways Tab */}
      {tab === 'gateways' && bd && (
        <div>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">Per-Gateway Detail</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Gateway ID</th><th>Location</th><th>Status</th>
                      <th>Uptime</th><th>Devices</th><th>Firmware</th>
                      <th>Last Heartbeat</th><th>Alerts</th><th>Unresolved</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.gateway_table||[]).map(g => (
                      <tr key={g.gateway_id}>
                        <td><code>{g.gateway_id}</code></td>
                        <td>{g.location}</td>
                        <td><StatusBadge status={g.status}/></td>
                        <td><UptimeBadge pct={g.uptime_pct}/></td>
                        <td>{g.connected_devices}</td>
                        <td><code>{g.firmware_version}</code></td>
                        <td><span className="small text-muted">{g.last_heartbeat ? g.last_heartbeat.replace('T',' ').slice(0,16) : '—'}</span></td>
                        <td>{g.alerts}</td>
                        <td>{g.unresolved_alerts > 0 ? <span className="text-danger fw-bold">{g.unresolved_alerts}</span> : <span className="text-success">0</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Device Load by Location</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Gateway</th><th>Location</th><th>Reported Connected</th><th>Devices in Location</th></tr>
                </thead>
                <tbody>
                  {(bd.device_load||[]).map(d => (
                    <tr key={d.gateway_id}>
                      <td><code>{d.gateway_id}</code></td>
                      <td>{d.location}</td>
                      <td>{d.reported_connected}</td>
                      <td>{d.devices_in_location}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Firmware Tab */}
      {tab === 'firmware' && bd && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Firmware Version Distribution</div>
              <div className="card-body">
                <DistBar items={ov.firmware_distribution} total={k.total_gateways} labelKey="version"
                  colorFn={v => v==='4.0.0'?'success':v==='3.2.0'?'warning':'danger'}/>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className={`card-header fw-bold ${bd.outdated_count>0?'bg-warning':'bg-success text-white'}`}>
                Firmware Gap — {bd.outdated_count} gateways need upgrade to {bd.latest_firmware}
              </div>
              <div className="card-body p-0">
                {bd.outdated_firmware && bd.outdated_firmware.length > 0 ? (
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Gateway ID</th><th>Location</th><th>Current Firmware</th><th>Required</th></tr>
                    </thead>
                    <tbody>
                      {bd.outdated_firmware.map(g => (
                        <tr key={g.gateway_id}>
                          <td><code>{g.gateway_id}</code></td>
                          <td>{g.location}</td>
                          <td><code className="text-warning">{g.firmware_version}</code></td>
                          <td><code className="text-success">{bd.latest_firmware}</code></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="p-3 text-success">All gateways running latest firmware.</div>
                )}
              </div>
            </div>
          </div>
          {defs && (
            <div className="col-12 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Firmware Policy</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Rule</th><th>Version</th><th>Status</th></tr>
                    </thead>
                    <tbody>
                      {(defs.firmware_policy||[]).map(p => (
                        <tr key={p.version}>
                          <td>{p.rule}</td>
                          <td><code>{p.version}</code></td>
                          <td>{p.status}</td>
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

      {/* Alerts Tab */}
      {tab === 'alerts' && bd && (
        <div>
          <div className="row g-2 mb-3">
            <div className="col-md-3">
              <div className="card text-center shadow-sm">
                <div className="card-body py-2">
                  <div className="h4 mb-0 text-danger">{k.unresolved_gateway_alerts}</div>
                  <div className="small text-muted">Unresolved Alerts</div>
                </div>
              </div>
            </div>
            <div className="col-md-3">
              <div className="card text-center shadow-sm">
                <div className="card-body py-2">
                  <div className="h4 mb-0 text-warning">{k.critical_alerts}</div>
                  <div className="small text-muted">Critical Alerts</div>
                </div>
              </div>
            </div>
            <div className="col-md-3">
              <div className="card text-center shadow-sm">
                <div className="card-body py-2">
                  <div className="h4 mb-0">{k.total_gateway_alerts}</div>
                  <div className="small text-muted">Total Gateway Alerts</div>
                </div>
              </div>
            </div>
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Recent Alert Log</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th>Gateway</th><th>Type</th><th>Severity</th><th>Resolved</th><th>Ack'd</th><th>Timestamp</th></tr>
                  </thead>
                  <tbody>
                    {(bd.alert_log||[]).map((a, i) => (
                      <tr key={i}>
                        <td><code>{a.gateway_id}</code></td>
                        <td>{a.alert_type?.replace(/_/g,' ')}</td>
                        <td>
                          <span className={`badge bg-${a.severity==='critical'?'danger':a.severity==='high'?'warning':a.severity==='medium'?'info':'secondary'}`}>
                            {a.severity}
                          </span>
                        </td>
                        <td>{a.resolved ? <span className="text-success">Yes</span> : <span className="text-danger">No</span>}</td>
                        <td>{a.acknowledged ? <span className="text-success">Yes</span> : <span className="text-muted">No</span>}</td>
                        <td><span className="small text-muted">{a.timestamp ? a.timestamp.replace('T',' ').slice(0,16) : '—'}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Glossary</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Term</th><th>Definition</th></tr></thead>
                  <tbody>
                    {(defs.glossary||[]).map(g => (
                      <tr key={g.term}><td className="fw-bold text-nowrap">{g.term}</td><td>{g.definition}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold">Uptime Tiers</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Tier</th><th>Range</th><th>Action</th></tr></thead>
                  <tbody>
                    {(defs.uptime_tiers||[]).map(t => (
                      <tr key={t.tier}><td className="fw-bold">{t.tier}</td><td>{t.range}</td><td>{t.action}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold">Connectivity Modes</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Mode</th><th>Use Case</th></tr></thead>
                  <tbody>
                    {(defs.connectivity_modes||[]).map(m => (
                      <tr key={m.mode}><td className="fw-bold">{m.mode}</td><td>{m.use}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Data Sources</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Table</th><th>Rows</th><th>Key Fields</th></tr></thead>
                  <tbody>
                    {(defs.data_sources||[]).map(s => (
                      <tr key={s.table}><td><code>{s.table}</code></td><td>{s.rows}</td><td><span className="small text-muted">{s.fields}</span></td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
