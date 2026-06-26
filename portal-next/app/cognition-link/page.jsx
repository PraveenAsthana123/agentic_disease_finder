'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8000';

function Badge({val, kind}){
  const m = {large:'danger', medium:'warning', small:'info', high:'danger', moderate:'warning'};
  return <span className={`badge bg-${m[val]||m[kind]||'secondary'} me-1`}>{val}</span>;
}

function RVal({r}){
  const c = r > 0 ? (r >= 0.5 ? 'primary' : 'info') : (r <= -0.5 ? 'danger' : 'warning');
  return <span className={`fw-bold text-${c}`}>{r > 0 ? '+' : ''}{r.toFixed(2)}</span>;
}

function HeatCell({val}){
  if(val === null || val === undefined) return <td className="text-center text-muted" style={{background:'#f1f1f1'}}>—</td>;
  const abs = Math.abs(val);
  const r = val < 0 ? Math.round(200 + abs*55) : Math.round(200 - abs*55);
  const g = Math.round(200 - abs*80);
  const b = val > 0 ? Math.round(200 + abs*55) : Math.round(200 - abs*55);
  return <td className="text-center small fw-bold" style={{background:`rgb(${r},${g},${b})`, color: abs>=0.4?'#fff':'#333'}} title={`r = ${val.toFixed(2)}`}>
    {val.toFixed(2)}
  </td>;
}

export default function CognitionLinkDashboard(){
  const [ov, setOv] = useState(null);
  const [matrix, setMatrix] = useState(null);
  const [heat, setHeat] = useState(null);
  const [domains, setDomains] = useState(null);
  const [alerts, setAlerts] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/cognition-link/overview`).then(r=>r.json()),
      fetch(`${API}/api/cognition-link/matrix`).then(r=>r.json()),
      fetch(`${API}/api/cognition-link/heatmap`).then(r=>r.json()),
      fetch(`${API}/api/cognition-link/domains`).then(r=>r.json()),
      fetch(`${API}/api/cognition-link/alerts`).then(r=>r.json()),
      fetch(`${API}/api/cognition-link/definitions`).then(r=>r.json()),
    ]).then(([o,m,h,d,a,df])=>{setOv(o);setMatrix(m);setHeat(h);setDomains(d);setAlerts(a);setDefs(df);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading cognition link data…</div>;

  const TABS = [
    {id:'overview', label:'📊 Overview'},
    {id:'heatmap', label:'🔥 Heatmap'},
    {id:'correlations', label:'📋 Full Matrix'},
    {id:'domains', label:'🧠 By Domain'},
    {id:'alerts', label:'🚨 Alerts'},
    {id:'definitions', label:'📖 Definitions'},
  ];

  return (<div>
    <h3>🔗 Cognition Link Dashboard</h3>
    <p className="text-muted">{ov.subtitle} — {ov.eeg_features_count} EEG features × {ov.cognitive_tests_count} tests</p>

    {/* KPI Hero Tiles */}
    <div className="row mb-3">
      {[
        ['Total Pairs Tested', ov.total_pairs_tested, 'primary'],
        ['Significant (p<.05)', ov.significant_pairs, 'success'],
        ['EEG Features', ov.eeg_features_count, 'info'],
        ['Cognitive Tests', ov.cognitive_tests_count, 'info'],
      ].map(([k,v,c])=>
        <div key={k} className="col-6 col-md-3 mb-2">
          <div className="card shadow-sm h-100"><div className="card-body text-center">
            <div className={`h4 mb-1 text-${c}`}>{v}</div>
            <div className="text-muted small">{k}</div>
          </div></div>
        </div>
      )}
    </div>

    {/* Tab bar */}
    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* ─── Overview ─── */}
    {tab==='overview' && <>
      <h5>Top 5 Strongest Correlations</h5>
      <table className="table table-sm table-striped">
        <thead><tr><th>EEG Feature</th><th>Cognitive Test</th><th>r</th><th>p</th><th>Clinical Note</th></tr></thead>
        <tbody>{ov.top_correlations?.map((c,i)=>
          <tr key={i}>
            <td className="fw-semibold">{c.eeg_feature}</td>
            <td>{c.cognitive_test}</td>
            <td><RVal r={c.r}/></td>
            <td>{c.p.toFixed(4)}</td>
            <td className="small text-muted">{c.note}</td>
          </tr>
        )}</tbody>
      </table>

      <h5 className="mt-4">Domain Summary</h5>
      <div className="row">
        {ov.domain_summary?.map(d=>
          <div key={d.domain} className="col-sm-6 col-lg-4 mb-3">
            <div className="card h-100 shadow-sm"><div className="card-body">
              <h6 className="card-title">{d.domain}</h6>
              <div className="d-flex justify-content-between small">
                <span>Sig. pairs: <strong>{d.n_significant}</strong></span>
                <span>Mean |r|: <strong>{d.mean_abs_r}</strong></span>
                <span>Max |r|: <strong>{d.max_abs_r}</strong></span>
              </div>
            </div></div>
          </div>
        )}
      </div>
    </>}

    {/* ─── Heatmap ─── */}
    {tab==='heatmap' && heat && <>
      <h5>EEG × Cognitive Test Correlation Heatmap</h5>
      <p className="small text-muted">{heat.note}</p>
      <div style={{overflowX:'auto'}}>
        <table className="table table-bordered table-sm" style={{fontSize:'0.78rem'}}>
          <thead><tr><th></th>{heat.cognitive_tests?.map(t=><th key={t} className="text-center" style={{writingMode:'vertical-lr',transform:'rotate(180deg)',maxWidth:30}}>{t}</th>)}</tr></thead>
          <tbody>
            {heat.eeg_features?.map((f,ri)=>
              <tr key={f}><td className="fw-semibold text-nowrap">{f}</td>
                {heat.matrix[ri]?.map((v,ci)=><HeatCell key={ci} val={v}/>)}
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </>}

    {/* ─── Full Matrix ─── */}
    {tab==='correlations' && matrix && <>
      <h5>Full Correlation Table ({matrix.n} pairs)</h5>
      <div style={{overflowX:'auto'}}>
        <table className="table table-sm table-striped" style={{fontSize:'0.82rem'}}>
          <thead><tr><th>EEG Feature</th><th>Band</th><th>Region</th><th>Test</th><th>Domain</th><th>r</th><th>p</th><th>Effect</th><th>Sig</th><th>Clinical Note</th></tr></thead>
          <tbody>{matrix.correlations?.map((c,i)=>
            <tr key={i} className={!c.significant?'opacity-50':''}>
              <td className="fw-semibold">{c.eeg_feature}</td>
              <td><span className="badge bg-secondary">{c.eeg_band}</span></td>
              <td className="small">{c.eeg_region}</td>
              <td>{c.test_name}</td>
              <td className="small text-muted">{c.test_domain}</td>
              <td><RVal r={c.r}/></td>
              <td>{c.p.toFixed(4)}</td>
              <td><Badge val={c.effect_size}/></td>
              <td>{c.significant ? '✓' : '—'}</td>
              <td className="small text-muted" style={{maxWidth:250}}>{c.clinical_note}</td>
            </tr>
          )}</tbody>
        </table>
      </div>
    </>}

    {/* ─── Domains ─── */}
    {tab==='domains' && domains && <>
      <h5>Cognitive Domain Profiles</h5>
      {domains.profiles?.map(p=>
        <div key={p.domain} className="card mb-3 shadow-sm">
          <div className="card-header d-flex justify-content-between">
            <strong>🧠 {p.domain}</strong>
            <span className="badge bg-primary">{p.n_significant} sig. links</span>
          </div>
          <div className="card-body">
            <p className="small mb-2">Strongest EEG predictor: <strong>{p.strongest_eeg_predictor}</strong> (r = <RVal r={p.strongest_r}/>)</p>
            <table className="table table-sm mb-0" style={{fontSize:'0.82rem'}}>
              <thead><tr><th>EEG Feature</th><th>Band</th><th>Test</th><th>r</th><th>p</th></tr></thead>
              <tbody>{p.correlations?.map((c,i)=>
                <tr key={i}><td>{c.eeg_feature}</td><td><span className="badge bg-secondary">{c.eeg_band}</span></td><td>{c.test}</td><td><RVal r={c.r}/></td><td>{c.p.toFixed(4)}</td></tr>
              )}</tbody>
            </table>
          </div>
        </div>
      )}
    </>}

    {/* ─── Alerts ─── */}
    {tab==='alerts' && alerts && <>
      <h5>Clinical Alerts ({alerts.n})</h5>
      {alerts.alerts?.length === 0 && <div className="alert alert-success">No strong alerts at this time.</div>}
      {alerts.alerts?.map((a,i)=>
        <div key={i} className={`alert alert-${a.severity==='high'?'danger':'warning'} shadow-sm`}>
          <div className="d-flex justify-content-between align-items-start">
            <div>
              <strong>{a.eeg_feature}</strong> ↔ <strong>{a.cognitive_test}</strong>
              <span className="ms-2"><RVal r={a.r}/> (p={a.p.toFixed(4)})</span>
            </div>
            <Badge val={a.severity}/>
          </div>
          <p className="small mt-1 mb-1">{a.clinical_note}</p>
          <p className="small text-muted mb-0">💡 {a.recommendation}</p>
        </div>
      )}
    </>}

    {/* ─── Definitions ─── */}
    {tab==='definitions' && defs && <>
      <h5>Effect Size Thresholds</h5>
      <table className="table table-sm table-striped">
        <thead><tr><th>Size</th><th>|r| Range</th><th>Interpretation</th></tr></thead>
        <tbody>{Object.entries(defs.effect_size_thresholds||{}).map(([k,v])=>
          <tr key={k}><td><Badge val={k}/></td><td>{v.min}–{v.max}</td><td className="small">{v.interpretation}</td></tr>
        )}</tbody>
      </table>

      <h5 className="mt-4">EEG Frequency Bands</h5>
      <table className="table table-sm table-striped">
        <thead><tr><th>Band</th><th>Frequency</th><th>Clinical Significance</th></tr></thead>
        <tbody>{Object.entries(defs.eeg_bands||{}).map(([k,v])=>
          <tr key={k}><td><span className="badge bg-secondary">{k}</span></td><td>{v.range_hz} Hz</td><td className="small">{v.clinical}</td></tr>
        )}</tbody>
      </table>

      <h5 className="mt-4">Cognitive Domain → EEG Marker Map</h5>
      <table className="table table-sm table-striped">
        <thead><tr><th>Domain</th><th>Tests</th><th>Key EEG Markers</th></tr></thead>
        <tbody>{defs.cognitive_domains?.map(d=>
          <tr key={d.domain}><td className="fw-semibold">{d.domain}</td><td>{d.tests?.join(', ')}</td><td className="small">{d.eeg_markers?.join(', ')}</td></tr>
        )}</tbody>
      </table>

      <h5 className="mt-4">References</h5>
      <ol className="small">{defs.references?.map((r,i)=><li key={i}>{r}</li>)}</ol>
    </>}
  </div>);
}
