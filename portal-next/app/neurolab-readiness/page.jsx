'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function NeuroLabReadinessDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/neurolab-readiness/overview`).then(r=>r.json()),
      fetch(`${API}/api/neurolab-readiness/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/neurolab-readiness/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading NeuroLab Readiness data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'stakeholders', label:'Stakeholders'},
    {id:'processes', label:'Processes'},
    {id:'business', label:'Business Case'},
    {id:'roadmap', label:'Roadmap'},
    {id:'gaps', label:'Gap Analysis'},
    {id:'definitions', label:'Definitions'},
  ];

  const statusBadge = (st) => {
    const colors = {built:'success', partial:'warning', missing:'danger'};
    return <span className={`badge bg-${colors[st]||'secondary'} me-1`}>{st}</span>;
  };

  const pctBar = (pct, label) => (
    <div className="mb-2">
      <div className="d-flex justify-content-between small">
        <span>{label}</span><span>{pct}%</span>
      </div>
      <div className="progress" style={{height:'8px'}}>
        <div className="progress-bar bg-primary" style={{width:`${pct}%`}}/>
      </div>
    </div>
  );

  return (<div className="p-3">
    <h3>NeuroLab Readiness Dashboard</h3>
    <p className="text-muted">Deployment readiness &amp; business case for NeuroLab AI in a clinical EEG laboratory</p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* ── Overview ── */}
    {tab==='overview' && <div>
      {/* KPI row */}
      <div className="row g-3 mb-4">
        {[
          {label:'Overall Readiness', value:`${ov.kpis.readiness_pct}%`, color:'primary'},
          {label:'Stakeholders', value:ov.kpis.total_stakeholders, color:'info'},
          {label:'Processes Built', value:`${ov.kpis.processes_built}/${ov.kpis.total_processes}`, color:'success'},
          {label:'Functionality Built', value:`${ov.kpis.functionality_built}/${ov.kpis.total_functionality}`, color:'success'},
          {label:'Missing Items', value:ov.kpis.total_missing_items, color:'danger'},
        ].map(k=><div key={k.label} className="col-md-2 col-sm-4">
          <div className={`card border-${k.color} h-100`}>
            <div className="card-body text-center py-2">
              <div className="small text-muted">{k.label}</div>
              <div className={`h4 mb-0 text-${k.color}`}>{k.value}</div>
            </div>
          </div>
        </div>)}
      </div>

      {/* Readiness Radar (bar-chart approximation) */}
      <div className="card mb-4">
        <div className="card-header fw-bold">Readiness by Dimension</div>
        <div className="card-body">
          {ov.readiness_radar.dimensions.map((dim, i) =>
            pctBar(ov.readiness_radar.values[i], dim)
          )}
        </div>
      </div>

      {/* Stakeholder readiness */}
      <div className="card mb-4">
        <div className="card-header fw-bold">Stakeholder Readiness</div>
        <div className="card-body">
          {ov.stakeholder_readiness.map(s =>
            <div key={s.role} className="mb-2">
              {pctBar(s.readiness_pct, `${s.icon} ${s.role} (${s.built_count} built / ${s.missing_count} missing)`)}
            </div>
          )}
        </div>
      </div>

      {/* Phase progress */}
      <div className="card mb-4">
        <div className="card-header fw-bold">Implementation Phases</div>
        <div className="card-body">
          <div className="d-flex flex-wrap gap-2">
            {ov.phase_progress.map(p =>
              <div key={p.phase} className="card" style={{minWidth:'180px'}}>
                <div className="card-body py-2 text-center">
                  {statusBadge(p.status)}
                  <div className="fw-bold small mt-1">{p.phase}</div>
                  <div className="text-muted" style={{fontSize:'0.75rem'}}>{p.scope}</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>}

    {/* ── Stakeholders ── */}
    {tab==='stakeholders' && bd && <div>
      <div className="row g-3">
        {bd.stakeholder_detail.map(s => <div key={s.role} className="col-md-6">
          <div className="card h-100">
            <div className="card-header d-flex justify-content-between">
              <span>{s.icon} {s.role}</span>
              <span className="badge bg-primary">{s.readiness_pct}% ready</span>
            </div>
            <div className="card-body">
              <h6 className="text-success">Built ({s.built.length})</h6>
              <ul className="small mb-2">
                {s.built.map((b,i)=><li key={i}>{b}</li>)}
              </ul>
              {s.missing.length > 0 && <>
                <h6 className="text-danger">Missing ({s.missing.length})</h6>
                <ul className="small mb-0">
                  {s.missing.map((m,i)=><li key={i}>{m}</li>)}
                </ul>
              </>}
            </div>
          </div>
        </div>)}
      </div>
    </div>}

    {/* ── Processes ── */}
    {tab==='processes' && bd && <div>
      <table className="table table-striped">
        <thead><tr><th>Process</th><th>Status</th><th>Mapped To</th></tr></thead>
        <tbody>
          {bd.process_detail.map((p,i) => <tr key={i}>
            <td>{p.name}</td>
            <td>{statusBadge(p.status)}</td>
            <td className="text-muted small">{p.maps_to || '-'}</td>
          </tr>)}
        </tbody>
      </table>

      <h5 className="mt-4">Functionality Status</h5>
      <table className="table table-striped">
        <thead><tr><th>Capability</th><th>Status</th></tr></thead>
        <tbody>
          {bd.functionality_detail.map((f,i) => <tr key={i}>
            <td>{f.capability}</td>
            <td>{statusBadge(f.status)}</td>
          </tr>)}
        </tbody>
      </table>
    </div>}

    {/* ── Business Case ── */}
    {tab==='business' && bd && <div>
      <div className="row g-3">
        <div className="col-md-4">
          <div className="card border-success h-100">
            <div className="card-header bg-success text-white fw-bold">Cost Decrease</div>
            <div className="card-body">
              {bd.business_case.cost_decrease.map((c,i)=>
                <div key={i} className="mb-2">
                  <div className="fw-bold small">{c.lever}</div>
                  <div className="text-muted small">{c.impact}</div>
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card border-primary h-100">
            <div className="card-header bg-primary text-white fw-bold">Revenue Increase</div>
            <div className="card-body">
              {bd.business_case.revenue_increase.map((r,i)=>
                <div key={i} className="mb-2">
                  <div className="fw-bold small">{r.lever}</div>
                  <div className="text-muted small">{r.impact}</div>
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card border-info h-100">
            <div className="card-header bg-info text-white fw-bold">Productivity Increase</div>
            <div className="card-body">
              {bd.business_case.productivity_increase.map((p,i)=>
                <div key={i} className="mb-2">
                  <div className="fw-bold small">{p.lever}</div>
                  <div className="text-muted small">{p.impact}</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>}

    {/* ── Roadmap ── */}
    {tab==='roadmap' && bd && <div>
      <div className="position-relative">
        {bd.implementation_roadmap.map((ph,i) =>
          <div key={ph.phase} className="d-flex align-items-start mb-3">
            <div className="me-3 text-center" style={{minWidth:'40px'}}>
              <div className={`rounded-circle d-inline-flex align-items-center justify-content-center ${ph.status==='built'?'bg-success':'bg-secondary'}`}
                style={{width:'32px',height:'32px',color:'#fff',fontWeight:'bold'}}>
                {i}
              </div>
              {i < bd.implementation_roadmap.length-1 && <div style={{width:'2px',height:'30px',margin:'0 auto',background:ph.status==='built'?'#198754':'#dee2e6'}}/>}
            </div>
            <div className="flex-grow-1">
              <div className="fw-bold">{ph.phase} {statusBadge(ph.status)}</div>
              <div className="text-muted small">{ph.scope}</div>
            </div>
          </div>
        )}
      </div>
    </div>}

    {/* ── Gap Analysis ── */}
    {tab==='gaps' && bd && <div>
      {bd.gap_analysis.map(g => <div key={g.role} className="card mb-3">
        <div className="card-header fw-bold">{g.icon} {g.role} ({g.total_gaps} gaps)</div>
        <div className="card-body">
          <div className="row g-2">
            {Object.entries(g.categories).map(([cat, items]) =>
              <div key={cat} className="col-md-4">
                <div className="card bg-light h-100">
                  <div className="card-body py-2">
                    <div className="fw-bold small text-capitalize">{cat.replace(/_/g,' ')}</div>
                    <ul className="small mb-0">
                      {items.map((item,i) => <li key={i}>{item}</li>)}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>)}
    </div>}

    {/* ── Definitions ── */}
    {tab==='definitions' && defs && <div>
      <table className="table">
        <thead><tr><th>Term</th><th>Definition</th></tr></thead>
        <tbody>
          {defs.terms.map((t,i) => <tr key={i}>
            <td className="fw-bold text-nowrap">{t.term}</td>
            <td>{t.definition}</td>
          </tr>)}
        </tbody>
      </table>
    </div>}
  </div>);
}
