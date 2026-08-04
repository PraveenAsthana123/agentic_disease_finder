'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function EEGAIStackDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/eeg-ai-stack/overview`).then(r=>r.json()),
      fetch(`${API}/api/eeg-ai-stack/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/eeg-ai-stack/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading EEG AI Stack data...</div>;

  const k = ov.kpis || {};
  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'layers', label:'Layer Details'},
    {id:'tools', label:'All Tools'},
    {id:'endpoints', label:'Live Endpoints'},
    {id:'definitions', label:'Definitions'},
  ];

  const statusColor = s => ({installed:'primary', built:'success', external:'secondary', cataloged:'warning'}[s]||'info');

  return (<div className="p-3">
    <h3>EEG AI Stack Dashboard</h3>
    <p className="text-muted">
      {k.layers} layers, {k.total_tools} tools — complete EEG AI processing pipeline
      from raw signal to responsible deployment.
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=>(
        <li key={t.id} className="nav-item">
          <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
        </li>
      ))}
    </ul>

    {tab==='overview' && <OverviewPanel ov={ov} statusColor={statusColor}/>}
    {tab==='layers' && <LayersPanel ov={ov} bd={bd} statusColor={statusColor}/>}
    {tab==='tools' && <ToolsPanel ov={ov} bd={bd} statusColor={statusColor}/>}
    {tab==='endpoints' && <EndpointsPanel bd={bd} statusColor={statusColor}/>}
    {tab==='definitions' && <DefinitionsPanel defs={defs}/>}
  </div>);
}

function KPI({label, value, color, sub}){
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color||'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{fontSize:'0.7rem'}}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function OverviewPanel({ov, statusColor}){
  const k = ov.kpis || {};
  const sd = ov.status_distribution || [];
  const layers = ov.layers || [];
  const pipeline = ov.recommended_pipeline || '';

  return (<div>
    {/* KPI cards */}
    <div className="row mb-3">
      <KPI label="Total Tools" value={k.total_tools} color="primary" sub="across all layers"/>
      <KPI label="Layers" value={k.layers} color="info" sub="processing pipeline stages"/>
      <KPI label="Installed" value={k.installed} color="primary" sub="verified importable"/>
      <KPI label="Built" value={k.built} color="success" sub="live endpoints + dashboards"/>
    </div>
    <div className="row mb-4">
      <KPI label="External" value={k.external} color="secondary" sub="MATLAB / desktop / server"/>
      <KPI label="With Endpoints" value={k.with_endpoints} color="success" sub="live API routes"/>
      <KPI label="EDC Tools" value={k.edc_tools} color="warning" sub="assessment platforms"/>
    </div>

    {/* Status distribution */}
    <div className="card mb-3">
      <div className="card-header fw-semibold">Status Distribution</div>
      <div className="card-body">
        <table className="table table-sm mb-0">
          <tbody>
            {sd.map((s,i)=>{
              const pct = k.total_tools>0 ? ((s.count/k.total_tools)*100).toFixed(1) : 0;
              return (<tr key={i}>
                <td style={{width:'25%'}}>
                  <span className={`badge bg-${statusColor(s.status)}`}>{s.status}</span>
                </td>
                <td style={{width:'50%'}}>
                  <div className="progress" style={{height:12}}>
                    <div className={`progress-bar bg-${statusColor(s.status)}`} style={{width:`${pct}%`}}/>
                  </div>
                </td>
                <td className="text-end small">{s.count} ({pct}%)</td>
              </tr>);
            })}
          </tbody>
        </table>
      </div>
    </div>

    {/* Layer overview */}
    <div className="card mb-3">
      <div className="card-header fw-semibold">Layer Coverage ({layers.length} layers)</div>
      <div className="card-body">
        <table className="table table-sm table-striped mb-0">
          <thead><tr>
            <th>Layer</th><th className="text-center">Tools</th><th className="text-center">Active %</th>
            <th className="text-center">Installed</th><th className="text-center">Built</th><th className="text-center">External</th>
          </tr></thead>
          <tbody>
            {layers.map((l,i)=>(
              <tr key={i}>
                <td className="small fw-semibold">{l.layer}</td>
                <td className="text-center">{l.total}</td>
                <td className="text-center">
                  <span className={`badge bg-${l.active_pct>=80?'success':l.active_pct>=50?'warning':'danger'}`}>
                    {l.active_pct}%
                  </span>
                </td>
                <td className="text-center">{l.installed||'—'}</td>
                <td className="text-center">{l.built||'—'}</td>
                <td className="text-center">{l.external||'—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>

    {/* Recommended pipeline */}
    {pipeline && <div className="card mb-3">
      <div className="card-header fw-semibold">Recommended Pipeline</div>
      <div className="card-body">
        <div className="d-flex flex-wrap gap-2 align-items-center">
          {pipeline.split(' → ').map((step,i,arr)=>(
            <span key={i}>
              <span className="badge bg-primary">{step}</span>
              {i < arr.length-1 && <span className="mx-1 text-muted">→</span>}
            </span>
          ))}
        </div>
      </div>
    </div>}

    {ov.honest_note && <div className="alert alert-info small">{ov.honest_note}</div>}
  </div>);
}

function LayersPanel({ov, bd, statusColor}){
  const perLayer = bd?.tools ? groupByLayer(bd.tools) : [];

  return (<div>
    {(ov.layers||[]).map((layer,i)=>{
      const layerTools = (bd?.tools||[]).filter(t=>t.layer===layer.layer || t.layer_name===layer.layer);
      return (<div key={i} className="card mb-3">
        <div className="card-header fw-semibold d-flex justify-content-between">
          <span>{layer.layer}</span>
          <span>
            <span className="badge bg-primary me-1">{layer.total} tools</span>
            <span className={`badge bg-${layer.active_pct>=80?'success':'warning'}`}>{layer.active_pct}% active</span>
          </span>
        </div>
        <div className="card-body">
          <table className="table table-sm mb-0">
            <thead><tr><th>Tool</th><th>Status</th><th>Use</th><th>Endpoints</th></tr></thead>
            <tbody>
              {layerTools.map((t,j)=>{
                const eps = t.endpoints||[];
                return (<tr key={j}>
                  <td className="fw-semibold small">{t.name}</td>
                  <td><span className={`badge bg-${statusColor(t.status)}`}>{t.status}</span></td>
                  <td className="small text-muted">{t.use||'—'}</td>
                  <td className="small">{eps.length>0 ? eps.map((e,k)=>(
                    <div key={k}><code className="text-success">{e}</code></div>
                  )) : <span className="text-muted">—</span>}</td>
                </tr>);
              })}
            </tbody>
          </table>
        </div>
      </div>);
    })}

    {/* EDC Tools */}
    {bd?.edc_assessment_tools?.length > 0 && <div className="card mb-3">
      <div className="card-header fw-semibold">EDC / Assessment Tools</div>
      <div className="card-body">
        <table className="table table-sm mb-0">
          <thead><tr><th>Tool</th><th>Status</th><th>Use</th><th>Endpoints</th></tr></thead>
          <tbody>
            {bd.edc_assessment_tools.map((t,i)=>(
              <tr key={i}>
                <td className="fw-semibold small">{t.name}</td>
                <td><span className={`badge bg-${statusColor(t.status)}`}>{t.status}</span></td>
                <td className="small text-muted">{t.use||'—'}</td>
                <td className="small">{(t.endpoints||[]).length>0 ? t.endpoints.map((e,k)=>(
                  <div key={k}><code className="text-success">{e}</code></div>
                )) : <span className="text-muted">—</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>}
  </div>);
}

function ToolsPanel({ov, bd, statusColor}){
  const tools = bd?.tools || ov?.tools || [];
  const [filter, setFilter] = useState('all');

  const statuses = [...new Set(tools.map(t=>t.status))];
  const filtered = filter==='all' ? tools : tools.filter(t=>t.status===filter);

  return (<div>
    <div className="mb-3">
      <span className="me-2 small fw-semibold">Filter:</span>
      <button className={`btn btn-sm me-1 ${filter==='all'?'btn-primary':'btn-outline-primary'}`} onClick={()=>setFilter('all')}>All ({tools.length})</button>
      {statuses.map(s=>(
        <button key={s} className={`btn btn-sm me-1 ${filter===s?`btn-${statusColor(s)}`:`btn-outline-${statusColor(s)}`}`}
          onClick={()=>setFilter(s)}>
          {s} ({tools.filter(t=>t.status===s).length})
        </button>
      ))}
    </div>
    <div className="card">
      <div className="card-body p-0">
        <table className="table table-sm table-striped mb-0">
          <thead><tr><th>#</th><th>Tool</th><th>Layer</th><th>Status</th><th>Use</th></tr></thead>
          <tbody>
            {filtered.map((t,i)=>(
              <tr key={i}>
                <td className="text-muted small">{i+1}</td>
                <td className="fw-semibold small">{t.name}</td>
                <td className="small text-muted">{t.layer}</td>
                <td><span className={`badge bg-${statusColor(t.status)}`}>{t.status}</span></td>
                <td className="small">{t.use||'—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  </div>);
}

function EndpointsPanel({bd, statusColor}){
  const tools = (bd?.tools||[]).filter(t=>(t.endpoints||[]).length>0);
  const edc = (bd?.edc_assessment_tools||[]).filter(t=>(t.endpoints||[]).length>0);
  const allEndpoints = [...tools, ...edc];
  const totalEps = allEndpoints.reduce((a,t)=>a+(t.endpoints||[]).length, 0);

  return (<div>
    <div className="alert alert-success small mb-3">
      <strong>{allEndpoints.length}</strong> tools with <strong>{totalEps}</strong> live API endpoints
    </div>
    {allEndpoints.map((t,i)=>(
      <div key={i} className="card mb-2">
        <div className="card-body py-2 d-flex justify-content-between align-items-start">
          <div>
            <span className="fw-semibold">{t.name}</span>
            <span className="text-muted small ms-2">{t.layer || 'EDC'}</span>
            <div className="mt-1">
              {(t.endpoints||[]).map((e,j)=>(
                <code key={j} className="d-block text-success small">{e}</code>
              ))}
            </div>
          </div>
          <span className={`badge bg-${statusColor(t.status)}`}>{t.status}</span>
        </div>
      </div>
    ))}
  </div>);
}

function DefinitionsPanel({defs}){
  if(!defs) return <div className="text-muted">Loading...</div>;

  const layers = defs.layers || [];
  const legend = defs.status_legend || [];
  const glossary = defs.glossary || [];
  const notes = defs.clinical_notes || [];
  const refs = defs.references || [];

  return (<div>
    {/* Status legend */}
    <div className="card mb-3">
      <div className="card-header fw-semibold">Status Legend</div>
      <div className="card-body">
        <table className="table table-sm mb-0">
          <thead><tr><th>Status</th><th>Meaning</th></tr></thead>
          <tbody>
            {legend.map((l,i)=>(
              <tr key={i}>
                <td><span className="badge bg-primary">{l.status}</span></td>
                <td className="small">{l.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>

    {/* Layer descriptions */}
    {layers.length > 0 && <div className="card mb-3">
      <div className="card-header fw-semibold">Layer Descriptions</div>
      <div className="card-body">
        <table className="table table-sm mb-0">
          <thead><tr><th>Layer</th><th>Description</th></tr></thead>
          <tbody>
            {layers.map((l,i)=>(
              <tr key={i}>
                <td className="fw-semibold small">{l.layer}</td>
                <td className="small">{l.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>}

    {/* Glossary */}
    {glossary.length > 0 && <div className="card mb-3">
      <div className="card-header fw-semibold">Glossary</div>
      <div className="card-body">
        <table className="table table-sm mb-0">
          <thead><tr><th>Term</th><th>Definition</th></tr></thead>
          <tbody>
            {glossary.map((g,i)=>(
              <tr key={i}>
                <td className="fw-semibold small text-nowrap">{g.term}</td>
                <td className="small">{g.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>}

    {/* Clinical notes */}
    {notes.length > 0 && <div className="card mb-3">
      <div className="card-header fw-semibold">Clinical Notes</div>
      <div className="card-body">
        <ul className="mb-0">
          {notes.map((n,i)=><li key={i} className="small">{n}</li>)}
        </ul>
      </div>
    </div>}

    {/* References */}
    {refs.length > 0 && <div className="card mb-3">
      <div className="card-header fw-semibold">References</div>
      <div className="card-body">
        <ol className="mb-0">
          {refs.map((r,i)=><li key={i} className="small">{r}</li>)}
        </ol>
      </div>
    </div>}
  </div>);
}

function groupByLayer(tools){
  const map = {};
  for(const t of tools){
    const l = t.layer || 'Other';
    if(!map[l]) map[l] = [];
    map[l].push(t);
  }
  return Object.entries(map).map(([layer,tools])=>({layer, tools}));
}
