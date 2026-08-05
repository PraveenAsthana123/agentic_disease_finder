'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function StrengthBadge({val}){
  const m = {strong:'success', moderate:'warning', weak:'secondary', high:'danger', medium:'warning', low:'info'};
  return <span className={`badge bg-${m[val]||'secondary'} me-1`}>{val}</span>;
}

function RiskBadge({val}){
  const m = {high:'danger', medium:'warning', low:'success'};
  return <span className={`badge bg-${m[val]||'secondary'}`}>{val}</span>;
}

function MiniBar({value, max, color='primary'}){
  const pct = max > 0 ? Math.min(100, Math.max(0, (value/max)*100)) : 0;
  return <div className="progress" style={{height:'8px', minWidth:'80px'}}>
    <div className={`progress-bar bg-${color}`} style={{width:`${pct}%`}}></div>
  </div>;
}

export default function CausalAIDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [search, setSearch] = useState('');

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/causal-ai/overview`).then(r=>r.json()),
      fetch(`${API}/api/causal-ai/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/causal-ai/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading causal AI data…</div>;

  const TABS = [
    {id:'overview', label:'📊 Overview'},
    {id:'medications', label:'💊 Medication Links'},
    {id:'triggers', label:'⚡ Trigger Chains'},
    {id:'graph', label:'🕸️ Causal Graph'},
    {id:'patients', label:'👤 Per Patient'},
    {id:'definitions', label:'📖 Definitions'},
  ];

  const kpi = ov.kpis;
  const maxTriggerCount = Math.max(...(ov.trigger_chains||[]).map(t=>t.count), 1);
  const maxSeizures = Math.max(...(ov.age_severity||[]).map(a=>a.seizures), 1);

  // Filter patients
  const patients = (bd?.patient_profiles || []).filter(p =>
    !search || p.patient_id?.toLowerCase().includes(search.toLowerCase()) ||
    p.name?.toLowerCase().includes(search.toLowerCase())
  );

  return (<div className="p-3">
    <h3>🔗 Causal AI Dashboard</h3>
    <p className="text-muted">
      Causal inference over epilepsy data — {kpi.total_patients} patients, {kpi.total_causal_pathways} causal pathways,{' '}
      {kpi.medications_analysed} medications analysed
    </p>

    {/* Tab Nav */}
    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* ── Overview Tab ── */}
    {tab==='overview' && <div>
      {/* KPI Hero */}
      <div className="row mb-3">
        {[
          ['Total Patients', kpi.total_patients, 'primary'],
          ['Patients w/ Seizures', kpi.patients_with_seizures, 'warning'],
          ['Seizure Events', kpi.total_seizure_events, 'danger'],
          ['Causal Pathways', kpi.total_causal_pathways, 'info'],
          ['Strong Pathways', kpi.strong_pathways, 'success'],
          ['Medications Analysed', kpi.medications_analysed, 'dark'],
          ['Triggers Identified', kpi.triggers_identified, 'secondary'],
          ['Graph Nodes', kpi.causal_graph_nodes, 'primary'],
        ].map(([k,v,c])=>
          <div key={k} className="col-6 col-md-3 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className={`h5 mb-0 text-${c}`}>{v}</div>
              <div className="text-muted small">{k}</div>
            </div></div>
          </div>
        )}
      </div>

      <div className="row mb-3">
        {/* Age × Severity */}
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Age Group × Seizure Severity</h6>
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Age Group</th><th>Seizures</th><th>Mild</th><th>Severe</th><th>Avg Duration</th></tr>
              </thead>
              <tbody>
                {ov.age_severity.map(a=><tr key={a.group}>
                  <td><strong>{a.group}</strong></td>
                  <td><span className="badge bg-primary">{a.seizures}</span></td>
                  <td>{a.severity_dist?.Mild||0}</td>
                  <td>{a.severity_dist?.Severe||0}</td>
                  <td>{a.avg_duration > 0 ? `${Math.round(a.avg_duration)}s` : '—'}</td>
                </tr>)}
              </tbody>
            </table>
          </div></div>
        </div>

        {/* Gender Analysis */}
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Gender × Seizure Rate</h6>
            {ov.gender_analysis.filter(g=>g.gender!=='Unknown').map(g=>
              <div key={g.gender} className="mb-3">
                <div className="d-flex justify-content-between mb-1">
                  <span>{g.gender}</span>
                  <span className="text-muted small">{g.patients} patients · {g.seizures} seizures · rate {g.seizure_rate.toFixed(2)}</span>
                </div>
                <MiniBar value={g.seizures} max={Math.max(...ov.gender_analysis.map(x=>x.seizures), 1)}
                  color={g.gender==='Female'?'danger':'primary'}/>
              </div>
            )}
            <div className="text-muted small mt-2">{ov.gender_analysis.find(g=>g.gender==='Unknown')?.patients||0} patients gender-unknown excluded</div>
          </div></div>
        </div>
      </div>

      {/* Top Trigger Summary */}
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Trigger Risk Summary</h6>
        <div className="d-flex flex-wrap gap-2">
          {ov.trigger_chains.map(t=>
            <div key={t.trigger} className="border rounded p-2 text-center" style={{minWidth:'140px'}}>
              <div className="fw-bold small">{t.trigger}</div>
              <div className="text-muted small">{t.count} events · {Math.round(t.avg_duration_sec)}s avg</div>
              <RiskBadge val={t.risk_level}/>
            </div>
          )}
        </div>
      </div></div>
    </div>}

    {/* ── Medication Links Tab ── */}
    {tab==='medications' && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Medication → Seizure Causal Links</h6>
        <p className="text-muted small">Strength classification: Strong (3+ patients), Moderate (2 patients), Weak (1 patient)</p>
        <table className="table table-hover">
          <thead className="table-dark">
            <tr><th>Drug</th><th>Patients</th><th>Avg Dose (mg)</th><th>Total Seizures</th><th>Seizures/Patient</th><th>Causal Strength</th></tr>
          </thead>
          <tbody>
            {ov.medication_seizure_links.map(m=><tr key={m.drug}>
              <td><strong>💊 {m.drug}</strong></td>
              <td>{m.patients}</td>
              <td>{m.avg_dose_mg.toFixed(0)}</td>
              <td>{m.total_seizures}</td>
              <td>{m.avg_seizures_per_patient.toFixed(1)}</td>
              <td><StrengthBadge val={m.causal_strength}/></td>
            </tr>)}
          </tbody>
        </table>
        <div className="alert alert-info small mt-2">
          ℹ️ All medications show 0 concurrent seizures — consistent with effective AED therapy in this cohort.
          Causal strength reflects independent patient observations supporting the association.
        </div>
      </div></div>
    </div>}

    {/* ── Trigger Chains Tab ── */}
    {tab==='triggers' && <div>
      <div className="row">
        {ov.trigger_chains.map(t=>
          <div key={t.trigger} className="col-md-4 mb-3">
            <div className="card shadow-sm h-100"><div className="card-body">
              <div className="d-flex justify-content-between align-items-start mb-2">
                <h6 className="mb-0">⚡ {t.trigger}</h6>
                <RiskBadge val={t.risk_level}/>
              </div>
              <div className="mb-2">
                <div className="d-flex justify-content-between small text-muted mb-1">
                  <span>Events</span><span>{t.count}</span>
                </div>
                <MiniBar value={t.count} max={maxTriggerCount} color={t.risk_level==='high'?'danger':t.risk_level==='medium'?'warning':'info'}/>
              </div>
              <div className="mb-2">
                <small className="text-muted">Avg Duration</small>
                <div className="fw-bold">{Math.round(t.avg_duration_sec)}s</div>
              </div>
              <div>
                <small className="text-muted">Severity Distribution</small>
                <div className="d-flex gap-2 mt-1">
                  {Object.entries(t.severity_distribution).map(([sev, cnt])=>
                    <span key={sev} className={`badge bg-${sev==='Severe'?'danger':'warning'}`}>{sev}: {cnt}</span>
                  )}
                </div>
              </div>
            </div></div>
          </div>
        )}
      </div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Trigger-to-Outcome Pathway</h6>
        <p className="text-muted small">Temporal precedence: trigger identified before seizure event in diary records.</p>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          {ov.trigger_chains.map((t,i)=><>
            <div key={t.trigger} className="border rounded px-3 py-2 text-center bg-light">
              <div className="small fw-bold">{t.trigger}</div>
              <RiskBadge val={t.risk_level}/>
            </div>
            {i < ov.trigger_chains.length-1 && <span key={`arr${i}`} className="text-muted">→</span>}
          </>)}
          <span className="text-muted">→</span>
          <div className="border rounded px-3 py-2 text-center bg-danger text-white">
            <div className="small fw-bold">Seizure Event</div>
            <div className="small">{kpi.total_seizure_events} events</div>
          </div>
        </div>
      </div></div>
    </div>}

    {/* ── Causal Graph Tab ── */}
    {tab==='graph' && <div>
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>🕸️ Graph Nodes ({ov.causal_graph.nodes.length})</h6>
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Node ID</th><th>Label</th><th>Type</th><th>Size (weight)</th></tr>
              </thead>
              <tbody>
                {ov.causal_graph.nodes.map(n=><tr key={n.id}>
                  <td><code>{n.id}</code></td>
                  <td>{n.label}</td>
                  <td>
                    <span className={`badge bg-${n.type==='Medication'?'primary':n.type==='Trigger'?'warning':'danger'}`}>
                      {n.type}
                    </span>
                  </td>
                  <td>{n.size}</td>
                </tr>)}
              </tbody>
            </table>
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>🔗 Graph Edges ({ov.causal_graph.edges.length})</h6>
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Source</th><th>Relation</th><th>Target</th><th>Strength</th><th>Weight</th></tr>
              </thead>
              <tbody>
                {ov.causal_graph.edges.map((e,i)=><tr key={i}>
                  <td><code>{e.source}</code></td>
                  <td><span className="text-muted small">{e.relation}</span></td>
                  <td><code>{e.target}</code></td>
                  <td><StrengthBadge val={e.strength}/></td>
                  <td>{e.weight}</td>
                </tr>)}
              </tbody>
            </table>
          </div></div>
        </div>
      </div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Graph Summary</h6>
        <div className="row text-center">
          {[
            ['Nodes', ov.causal_graph.nodes.length, 'primary'],
            ['Edges', ov.causal_graph.edges.length, 'info'],
            ['Medication nodes', ov.causal_graph.nodes.filter(n=>n.type==='Medication').length, 'primary'],
            ['Trigger nodes', ov.causal_graph.nodes.filter(n=>n.type==='Trigger').length, 'warning'],
            ['Outcome nodes', ov.causal_graph.nodes.filter(n=>n.type==='Outcome').length, 'danger'],
          ].map(([k,v,c])=>
            <div key={k} className="col">
              <div className={`h4 text-${c}`}>{v}</div>
              <div className="text-muted small">{k}</div>
            </div>
          )}
        </div>
      </div></div>
    </div>}

    {/* ── Per Patient Tab ── */}
    {tab==='patients' && <div>
      <div className="mb-3">
        <input className="form-control" placeholder="Search by patient ID or name…"
          value={search} onChange={e=>setSearch(e.target.value)} style={{maxWidth:'320px'}}/>
      </div>
      <div className="card shadow-sm"><div className="card-body p-0">
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-dark">
              <tr>
                <th>Patient</th><th>Age</th><th>Gender</th><th>Seizures</th>
                <th>Medications</th><th>Triggers</th><th>Avg Duration</th>
                <th>Assessments</th><th>Causal Factors</th><th>Risk</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p=><tr key={p.patient_id}>
                <td><strong>{p.patient_id}</strong><br/><span className="text-muted small">{p.name}</span></td>
                <td>{p.age}</td>
                <td>{p.gender}</td>
                <td>{p.seizure_count > 0 ? <span className="badge bg-danger">{p.seizure_count}</span> : <span className="text-muted">0</span>}</td>
                <td>{p.medications?.length > 0 ? p.medications.join(', ') : <span className="text-muted">—</span>}</td>
                <td>{p.triggers?.length > 0 ? p.triggers.join(', ') : <span className="text-muted">—</span>}</td>
                <td>{p.avg_duration_sec > 0 ? `${Math.round(p.avg_duration_sec)}s` : '—'}</td>
                <td>{p.assessment_count}</td>
                <td>{p.causal_factors}</td>
                <td><RiskBadge val={p.risk_level}/></td>
              </tr>)}
            </tbody>
          </table>
        </div>
      </div></div>
      <div className="text-muted small mt-2">Showing {patients.length} of {bd?.patient_profiles?.length||0} patients</div>
    </div>}

    {/* ── Definitions Tab ── */}
    {tab==='definitions' && defs && <div>
      {defs.sections?.map(sec=><div key={sec.title} className="card shadow-sm mb-3">
        <div className="card-header fw-bold">{sec.title}</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th style={{width:'220px'}}>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {sec.items?.map(item=><tr key={item.term}>
                <td><strong>{item.term}</strong></td>
                <td className="text-muted small">{item.definition}</td>
              </tr>)}
            </tbody>
          </table>
        </div>
      </div>)}
    </div>}
  </div>);
}
