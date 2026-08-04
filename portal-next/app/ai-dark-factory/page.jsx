'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function AIDarkFactoryDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/ai-dark-factory/overview`).then(r=>r.json()),
      fetch(`${API}/api/ai-dark-factory/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/ai-dark-factory/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading AI Dark Factory data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'pipeline', label:'Pipeline Stages'},
    {id:'tools', label:'Tool Catalog'},
    {id:'planes', label:'Planes & Patterns'},
    {id:'definitions', label:'Definitions'},
  ];

  const statusColor = s => ({built:'success',cataloged:'info',planned:'warning',adapter:'primary'}[s]||'secondary');
  const statusBadge = s => <span className={`badge bg-${statusColor(s)} me-1`}>{s}</span>;

  const kpi = ov.kpis || {};

  return (<div className="p-3">
    <h3>AI Dark Factory Dashboard</h3>
    <p className="text-muted">
      {ov.title} &mdash; {kpi.total_stages} pipeline stages, {kpi.total_tools} tools,
      {' '}{kpi.tool_categories} categories, {kpi.planes} planes, {kpi.patterns} patterns
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          ['Pipeline Stages', kpi.total_stages, 'primary'],
          ['Built', kpi.built, 'success'],
          ['Cataloged', kpi.cataloged, 'info'],
          ['Planned', kpi.planned, 'warning'],
          ['Tools', kpi.total_tools, 'primary'],
          ['Categories', kpi.tool_categories, 'secondary'],
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
            <h6>Flow Status Distribution</h6>
            {(ov.flow_status_distribution||[]).map(({name,value})=>{
              const pct = ((value/kpi.total_stages)*100).toFixed(1);
              return <div key={name} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'90px',textTransform:'capitalize'}}>{name}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${statusColor(name)}`} style={{width:`${pct}%`}}>
                      {value} ({pct}%)</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Tool Status Distribution</h6>
            {(ov.tool_status_distribution||[]).map(({name,value})=>{
              const pct = ((value/kpi.total_tools)*100).toFixed(1);
              return <div key={name} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'90px',textTransform:'capitalize'}}>{name}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${statusColor(name)}`} style={{width:`${pct}%`}}>
                      {value} ({pct}%)</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
      </div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Tools per Category</h6>
        {(ov.tools_per_category||[]).map(({name,value})=>{
          const pct = ((value/kpi.total_tools)*100).toFixed(1);
          return <div key={name} className="d-flex align-items-center mb-2">
            <span className="me-2 small" style={{minWidth:'140px'}}>{name}</span>
            <div className="flex-grow-1 me-2">
              <div className="progress" style={{height:'20px'}}>
                <div className="progress-bar bg-primary" style={{width:`${pct}%`}}>
                  {value}</div>
              </div>
            </div>
          </div>;
        })}
      </div></div>

      {ov.description && <div className="alert alert-info small">{ov.description}</div>}
    </div>}

    {tab==='pipeline' && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>11-Stage Autonomous Pipeline</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead><tr>
              <th>#</th><th>Stage</th><th>Tool</th><th>Produces</th><th>Status</th><th>Note</th>
            </tr></thead>
            <tbody>
              {(ov.flow_table||[]).map(s=>
                <tr key={s.n}>
                  <td>{s.n}</td>
                  <td className="fw-bold">{s.stage}</td>
                  <td>{s.tool}</td>
                  <td>{s.produces||'—'}</td>
                  <td>{statusBadge(s.status)}</td>
                  <td className="small text-muted">{s.note||''}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='tools' && bd && <div>
      {(bd.per_category||[]).map(cat=>
        <div key={cat.category} className="card shadow-sm mb-3"><div className="card-body">
          <h6>{cat.category} <span className="badge bg-secondary">{cat.tools.length}</span></h6>
          <div className="table-responsive">
            <table className="table table-sm">
              <thead><tr><th>Tool</th><th>Purpose</th><th>Status</th><th>Note</th></tr></thead>
              <tbody>
                {cat.tools.map(t=>
                  <tr key={t.tool}>
                    <td className="fw-bold">{t.tool}</td>
                    <td>{t.for}</td>
                    <td>{statusBadge(t.status)}</td>
                    <td className="small text-muted">{t.note||''}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div></div>
      )}
    </div>}

    {tab==='planes' && bd && <div>
      <div className="row mb-3">
        {(bd.planes||[]).map(p=>
          <div key={p.plane} className="col-md-4 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body">
              <h6>{p.plane} {statusBadge(p.status)}</h6>
              <div className="mb-2">
                {p.components.map(c=><span key={c} className="badge bg-light text-dark border me-1 mb-1">{c}</span>)}
              </div>
              {p.note && <div className="small text-muted">{p.note}</div>}
            </div></div>
          </div>
        )}
      </div>

      <h5 className="mt-3">Architecture Patterns</h5>
      <div className="row">
        {(bd.patterns||[]).map(p=>
          <div key={p.id} className="col-md-6 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body">
              <h6>{p.id.replace(/_/g,' ')} {statusBadge(p.status)}</h6>
              <p className="small mb-1"><strong>Description:</strong> {p.desc}</p>
              <p className="small mb-1"><strong>Best for:</strong> {p.best_for}</p>
              <p className="small mb-1 text-danger"><strong>Failure mode:</strong> {p.failure_mode}</p>
              {p.note && <p className="small text-muted mb-0">{p.note}</p>}
            </div></div>
          </div>
        )}
      </div>
    </div>}

    {tab==='definitions' && defs && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Status Legend</h6>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Status</th><th>Meaning</th></tr></thead>
            <tbody>
              {(defs.status_legend||[]).map(s=>
                <tr key={s.status}>
                  <td>{statusBadge(s.status)}</td>
                  <td>{s.description}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Glossary</h6>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {(defs.glossary||[]).map(g=>
                <tr key={g.term}>
                  <td className="fw-bold">{g.term}</td>
                  <td>{g.definition}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>

      {defs.adoption_gates && <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>6-Gate Adoption Process</h6>
        <ol>
          {defs.adoption_gates.map((g,i)=><li key={i} className="mb-1">{g}</li>)}
        </ol>
      </div></div>}

      {defs.clinical_notes && <div className="alert alert-warning small">
        <strong>Clinical Notes:</strong>
        <ul className="mb-0 mt-1">
          {defs.clinical_notes.map((n,i)=><li key={i}>{n}</li>)}
        </ul>
      </div>}

      {defs.references && <div className="card shadow-sm"><div className="card-body">
        <h6>References</h6>
        <ul className="mb-0">
          {defs.references.map((r,i)=><li key={i}><strong>{r.ref}:</strong> {r.detail}</li>)}
        </ul>
      </div></div>}
    </div>}
  </div>);
}
