'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const tierBadge = t => t === 1
  ? <span className="badge bg-danger">Tier 1 — Mandatory</span>
  : <span className="badge bg-info">Tier 2 — Recommended</span>;

const dataCellColor = v => ({yes:'table-success', optional:'table-warning', no:'table-light', metadata:'table-info', aggregated:'table-secondary'}[v] || '');

export default function ConsultantMatrixDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [selected, setSelected] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/consultant-matrix/overview`).then(r=>r.json()),
      fetch(`${API}/api/consultant-matrix/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/consultant-matrix/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading consultant matrix...</div>;

  const s = ov.summary || {};
  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'roles', label:'Role Details'},
    {id:'data', label:'Data Matrix'},
    {id:'ai', label:'AI Solutions'},
    {id:'definitions', label:'Definitions'},
  ];

  return (<div className="p-3">
    <h3>Consultant Matrix Dashboard</h3>
    <p className="text-muted">
      {s.total_consultants} consultants ({s.mandatory} mandatory, {s.optional} recommended) &mdash;
      {' '}{s.total_tasks} tasks, {s.total_challenges} challenges, {s.total_ai_solutions} AI solutions,
      {' '}{s.ai_coverage_pct}% AI coverage
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* ── OVERVIEW ── */}
    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          ['Consultants', s.total_consultants, 'primary'],
          ['Mandatory', s.mandatory, 'danger'],
          ['Recommended', s.optional, 'info'],
          ['Tasks', s.total_tasks, 'success'],
          ['Challenges', s.total_challenges, 'warning'],
          ['AI Solutions', s.total_ai_solutions, 'primary'],
          ['AI Coverage', s.ai_coverage_pct+'%', 'success'],
          ['Tools', s.total_tools, 'secondary'],
          ['Assessments', s.total_assessments, 'info'],
          ['Documents', s.total_documents, 'dark'],
          ['Compliance Docs', s.total_compliance_docs, 'danger'],
        ].map(([label,val,c])=>
          <div key={label} className="col-6 col-md-2 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className={`h5 mb-0 text-${c}`}>{val}</div>
              <div className="text-muted small">{label}</div>
            </div></div>
          </div>
        )}
      </div>

      {/* Tier Distribution */}
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Tier Distribution</h6>
            {(ov.tier_distribution||[]).map(t=>{
              const p = s.total_consultants ? ((t.value/s.total_consultants)*100).toFixed(0) : 0;
              return <div key={t.name} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'60px'}}>{t.name}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'22px'}}>
                    <div className={`progress-bar bg-${t.name==='Tier 1'?'danger':'info'}`} style={{width:p+'%'}}>{t.value} ({p}%)</div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Data Coverage by Field</h6>
            <table className="table table-sm table-bordered mb-0">
              <thead><tr><th>Field</th><th className="text-center">Yes</th><th className="text-center">Optional</th><th className="text-center">No</th></tr></thead>
              <tbody>
                {Object.entries(ov.data_coverage||{}).map(([field,counts])=>
                  <tr key={field}>
                    <td className="fw-bold small">{field.replace(/_/g,' ')}</td>
                    <td className="text-center table-success">{counts.yes}</td>
                    <td className="text-center table-warning">{counts.optional}</td>
                    <td className="text-center table-light">{counts.no}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div></div>
        </div>
      </div>

      {/* Role Summary Table */}
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Consultant Summary</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-hover mb-0">
            <thead><tr>
              <th>Consultant</th><th>Role</th><th>Tier</th>
              <th className="text-center">Tasks</th><th className="text-center">Challenges</th>
              <th className="text-center">AI Solutions</th><th className="text-center">Tools</th>
              <th className="text-center">Assessments</th>
            </tr></thead>
            <tbody>
              {(ov.role_summary||[]).map(r=>
                <tr key={r.id} style={{cursor:'pointer'}} onClick={()=>{setSelected(r.id);setTab('roles');}}>
                  <td className="fw-bold">{r.name}</td>
                  <td className="text-muted small">{r.role}</td>
                  <td>{tierBadge(r.tier)}</td>
                  <td className="text-center">{r.tasks}</td>
                  <td className="text-center">{r.challenges}</td>
                  <td className="text-center">{r.ai_solutions}</td>
                  <td className="text-center">{r.tools}</td>
                  <td className="text-center">{r.assessments}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="text-muted small mt-1">Click a row to see full details</div>
      </div></div>
    </div>}

    {/* ── ROLE DETAILS ── */}
    {tab==='roles' && bd && <div>
      <div className="row">
        <div className="col-md-3 mb-3">
          <div className="list-group">
            {(bd.roles||[]).map(r=>
              <button key={r.id}
                className={`list-group-item list-group-item-action ${selected===r.id?'active':''}`}
                onClick={()=>setSelected(r.id)}>
                <div className="fw-bold small">{r.name}</div>
                <div className="text-muted small">{r.role}</div>
              </button>
            )}
          </div>
        </div>
        <div className="col-md-9">
          {(()=>{
            const role = (bd.roles||[]).find(r=>r.id===selected) || bd.roles?.[0];
            if(!role) return <div className="text-muted">No role selected</div>;
            return <div>
              <div className="card shadow-sm mb-3"><div className="card-body">
                <div className="d-flex justify-content-between align-items-start mb-2">
                  <div>
                    <h5 className="mb-1">{role.name}</h5>
                    <div className="text-muted">{role.role} &mdash; {role.objective}</div>
                  </div>
                  {tierBadge(role.tier)}
                </div>

                <div className="row mt-3">
                  <div className="col-md-6 mb-3">
                    <h6>Tasks</h6>
                    <ul className="list-group list-group-flush">
                      {(role.tasks||[]).map((t,i)=><li key={i} className="list-group-item py-1 small">{t}</li>)}
                    </ul>
                  </div>
                  <div className="col-md-6 mb-3">
                    <h6>Challenges</h6>
                    <ul className="list-group list-group-flush">
                      {(role.challenges||[]).map((c,i)=><li key={i} className="list-group-item py-1 small list-group-item-warning">{c}</li>)}
                    </ul>
                  </div>
                </div>

                <div className="row">
                  <div className="col-md-6 mb-3">
                    <h6>Tools</h6>
                    {(role.tools||[]).map((t,i)=><span key={i} className="badge bg-secondary me-1 mb-1">{t}</span>)}
                  </div>
                  <div className="col-md-6 mb-3">
                    <h6>Assessments</h6>
                    {(role.assessment||[]).map((a,i)=><span key={i} className="badge bg-primary me-1 mb-1">{a}</span>)}
                  </div>
                </div>

                <div className="row">
                  <div className="col-md-6 mb-3">
                    <h6>Documents</h6>
                    <ul className="list-group list-group-flush">
                      {(role.documents||[]).map((d,i)=><li key={i} className="list-group-item py-1 small">{d}</li>)}
                    </ul>
                  </div>
                  <div className="col-md-6 mb-3">
                    <h6>Compliance Documents</h6>
                    <ul className="list-group list-group-flush">
                      {(role.compliance_docs||[]).map((d,i)=><li key={i} className="list-group-item py-1 small list-group-item-danger">{d}</li>)}
                    </ul>
                  </div>
                </div>

                <div className="row">
                  <div className="col-md-6 mb-3">
                    <h6>Internal Tasks</h6>
                    <ul className="list-group list-group-flush">
                      {(role.internal_tasks||[]).map((t,i)=><li key={i} className="list-group-item py-1 small">{t}</li>)}
                    </ul>
                  </div>
                  <div className="col-md-6 mb-3">
                    <h6>Patient Questionnaire</h6>
                    <ul className="list-group list-group-flush">
                      {(role.patient_questionnaire||[]).map((q,i)=><li key={i} className="list-group-item py-1 small">{q}</li>)}
                    </ul>
                  </div>
                </div>

                {role.patient_documents && role.patient_documents.length > 0 && <div className="mb-3">
                  <h6>Patient Documents</h6>
                  {role.patient_documents.map((d,i)=><span key={i} className="badge bg-success me-1 mb-1">{d}</span>)}
                </div>}

                <h6>AI Solutions</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead><tr><th>Challenge</th><th>AI Solution</th></tr></thead>
                    <tbody>
                      {(role.ai_solutions||[]).map((a,i)=>
                        <tr key={i}><td className="small">{a.challenge}</td><td className="small text-success">{a.ai}</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>

                <h6 className="mt-3">Data Requirements</h6>
                <div className="d-flex flex-wrap gap-2">
                  {Object.entries(role.data||{}).map(([field,val])=>
                    <span key={field} className={`badge ${val==='yes'?'bg-success':val==='optional'?'bg-warning text-dark':'bg-light text-dark border'}`}>
                      {field.replace(/_/g,' ')}: {val}
                    </span>
                  )}
                </div>
              </div></div>
            </div>;
          })()}
        </div>
      </div>
    </div>}

    {/* ── DATA MATRIX ── */}
    {tab==='data' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Data Requirement Matrix (per Consultant)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0">
            <thead><tr>
              <th>Consultant</th>
              {(bd.data_fields||[]).map(f=><th key={f} className="text-center small">{f.replace(/_/g,' ')}</th>)}
            </tr></thead>
            <tbody>
              {(bd.data_matrix||[]).map(row=>
                <tr key={row.id}>
                  <td className="fw-bold small">{row.consultant}</td>
                  {(bd.data_fields||[]).map(f=>
                    <td key={f} className={`text-center small ${dataCellColor(row[f])}`}>{row[f]}</td>
                  )}
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="mt-2 d-flex gap-3 small">
          <span><span className="badge bg-success">yes</span> Required</span>
          <span><span className="badge bg-warning text-dark">optional</span> Helpful</span>
          <span><span className="badge bg-light text-dark border">no</span> Not needed</span>
        </div>
      </div></div>

      <div className="row mt-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Core Team</h6>
            <ul className="list-group list-group-flush">
              {(bd.core_team||[]).map(id=>{
                const r = (bd.roles||[]).find(x=>x.id===id);
                return <li key={id} className="list-group-item py-1 d-flex justify-content-between">
                  <span className="fw-bold small">{r?.name||id}</span>
                  <span className="badge bg-danger">Mandatory</span>
                </li>;
              })}
            </ul>
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Recommended Add-ons</h6>
            <ul className="list-group list-group-flush">
              {(bd.recommended_addons||[]).map(id=>{
                const r = (bd.roles||[]).find(x=>x.id===id);
                return <li key={id} className="list-group-item py-1 d-flex justify-content-between">
                  <span className="fw-bold small">{r?.name||id}</span>
                  <span className="badge bg-info">Recommended</span>
                </li>;
              })}
            </ul>
          </div></div>
        </div>
      </div>
    </div>}

    {/* ── AI SOLUTIONS ── */}
    {tab==='ai' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>AI Solutions for Consultant Challenges ({(bd.ai_solutions||[]).length} total, {s.ai_coverage_pct}% coverage)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr><th>Consultant</th><th>Challenge</th><th>AI Solution</th></tr></thead>
            <tbody>
              {(bd.ai_solutions||[]).map((a,i)=>
                <tr key={i}>
                  <td className="fw-bold small">{a.consultant}</td>
                  <td className="small text-warning">{a.challenge}</td>
                  <td className="small text-success">{a.ai}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── DEFINITIONS ── */}
    {tab==='definitions' && defs && <div>
      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Tier Definitions</h6>
            <table className="table table-sm mb-0">
              <thead><tr><th>Tier</th><th>Label</th><th>Description</th></tr></thead>
              <tbody>
                {(defs.tiers||[]).map(t=>
                  <tr key={t.tier}><td>{t.tier}</td><td className="fw-bold">{t.label}</td><td className="small">{t.description}</td></tr>
                )}
              </tbody>
            </table>
          </div></div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Data Requirement Legend</h6>
            <table className="table table-sm mb-0">
              <thead><tr><th>Value</th><th>Meaning</th></tr></thead>
              <tbody>
                {(defs.data_requirement_legend||[]).map(l=>
                  <tr key={l.value}><td><span className={`badge ${l.value==='yes'?'bg-success':l.value==='optional'?'bg-warning text-dark':'bg-light text-dark border'}`}>{l.value}</span></td><td className="small">{l.meaning}</td></tr>
                )}
              </tbody>
            </table>
          </div></div>
        </div>
      </div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Glossary</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr><th>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {(defs.glossary||[]).map(g=>
                <tr key={g.term}><td className="fw-bold">{g.term}</td><td className="small">{g.definition}</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}
  </div>);
}
