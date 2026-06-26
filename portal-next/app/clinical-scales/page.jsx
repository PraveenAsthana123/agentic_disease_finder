'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8000';

const CAT_COLORS = {
  'Functional': 'primary', 'Consciousness': 'danger', 'Disability': 'warning',
  'Stroke Severity': 'danger', 'Psychiatric': 'info', 'Depression': 'secondary',
  'Seizure Outcome': 'success', 'AED Side Effects': 'warning', 'Mortality Risk': 'danger',
  'Adherence': 'primary', 'Sleep Quality': 'info',
};
function catColor(c) { return CAT_COLORS[c] || (/Cognitive/.test(c) ? 'info' : 'secondary'); }

function DirectionBadge({higher}) {
  if (higher === 'better') return <span className="badge bg-success">Higher = Better</span>;
  if (higher === 'worse')  return <span className="badge bg-danger">Higher = Worse</span>;
  return <span className="badge bg-secondary">Mixed</span>;
}

export default function ClinicalScalesDashboard() {
  const [catalog, setCatalog] = useState(null);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailDefs, setDetailDefs] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [filterCat, setFilterCat] = useState('All');
  const [tab, setTab] = useState('catalog');
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/neuro-scales/catalog`).then(r => r.json()).then(setCatalog).catch(e => setErr(String(e)));
  }, []);

  function loadScale(scaleId) {
    setSelected(scaleId);
    setDetailLoading(true);
    setDetail(null);
    setDetailDefs(null);
    setTab('detail');
    Promise.all([
      fetch(`${API}/api/neuro-scales/${scaleId}`).then(r => r.json()),
      fetch(`${API}/api/neuro-scales/${scaleId}/definitions`).then(r => r.json()),
    ]).then(([d, defs]) => { setDetail(d); setDetailDefs(defs); setDetailLoading(false); })
      .catch(e => { setErr(`Failed to load ${scaleId}: ${e}`); setDetailLoading(false); });
  }

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!catalog) return <div className="text-muted p-3">Loading clinical scales catalog...</div>;

  const TABS = [
    {id: 'catalog', label: 'Catalog'},
    {id: 'by-category', label: 'By Category'},
    ...(selected ? [{id: 'detail', label: `${selected.toUpperCase()} Detail`}] : []),
  ];

  const filtered = filterCat === 'All' ? catalog.scales : catalog.scales.filter(s => s.category === filterCat);
  const grouped = {};
  catalog.scales.forEach(s => { (grouped[s.category] = grouped[s.category] || []).push(s); });

  return (<div>
    <h3>Clinical & Neuropsychological Scales</h3>
    <p className="text-muted">{catalog.subtitle}</p>

    {/* KPI Hero Tiles */}
    <div className="row mb-3">
      {[
        ['Total Scales', catalog.total_scales, 'primary'],
        ['Patients in DB', catalog.total_patients, 'success'],
        ['Categories', catalog.categories.length, 'info'],
        ['Cognitive Tests', catalog.scales.filter(s => /Cognitive/.test(s.category)).length, 'warning'],
      ].map(([k, v, c]) =>
        <div key={k} className="col-6 col-md-3 mb-2">
          <div className="card shadow-sm h-100"><div className="card-body text-center">
            <div className={`h4 mb-1 text-${c}`}>{v}</div>
            <div className="text-muted small">{k}</div>
          </div></div>
        </div>
      )}
    </div>

    {/* Tabs */}
    <ul className="nav nav-tabs mb-3">
      {TABS.map(t => <li key={t.id} className="nav-item">
        <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* ── Catalog Tab ── */}
    {tab === 'catalog' && <>
      <div className="mb-3">
        <select className="form-select form-select-sm d-inline-block w-auto" value={filterCat} onChange={e => setFilterCat(e.target.value)}>
          <option value="All">All categories ({catalog.total_scales})</option>
          {catalog.categories.map(c => <option key={c} value={c}>{c} ({catalog.scales.filter(s => s.category === c).length})</option>)}
        </select>
      </div>
      <div className="row">
        {filtered.map(s =>
          <div key={s.id} className="col-12 col-md-6 col-lg-4 mb-3">
            <div className="card h-100 shadow-sm" style={{cursor: 'pointer', borderLeft: `4px solid var(--bs-${catColor(s.category)})`}} onClick={() => loadScale(s.id)}>
              <div className="card-body">
                <h6 className="card-title mb-1">{s.name}</h6>
                <div className="mb-2">
                  <span className={`badge bg-${catColor(s.category)} me-1`}>{s.category}</span>
                  <DirectionBadge higher={s.higher} />
                  <span className="badge bg-light text-dark ms-1">{s.range}</span>
                </div>
                <p className="card-text small text-muted mb-0">{s.description}</p>
              </div>
              <div className="card-footer bg-transparent border-top-0 pt-0">
                <button className="btn btn-sm btn-outline-primary">View Dashboard</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>}

    {/* ── By Category Tab ── */}
    {tab === 'by-category' && <>
      {Object.entries(grouped).map(([cat, scales]) =>
        <div key={cat} className="mb-4">
          <h5><span className={`badge bg-${catColor(cat)} me-2`}>{cat}</span> {scales.length} scale{scales.length > 1 ? 's' : ''}</h5>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead><tr><th>Scale</th><th>Range</th><th>Direction</th><th>Description</th><th></th></tr></thead>
              <tbody>
                {scales.map(s => <tr key={s.id}>
                  <td className="fw-bold">{s.name}</td>
                  <td><code>{s.range}</code></td>
                  <td><DirectionBadge higher={s.higher} /></td>
                  <td className="small">{s.description}</td>
                  <td><button className="btn btn-sm btn-outline-primary" onClick={() => loadScale(s.id)}>View</button></td>
                </tr>)}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>}

    {/* ── Detail Tab ── */}
    {tab === 'detail' && selected && <>
      <button className="btn btn-sm btn-outline-secondary mb-3" onClick={() => { setTab('catalog'); setSelected(null); }}>Back to Catalog</button>
      {detailLoading && <div className="text-muted">Loading {selected} data...</div>}
      {detail && <ScaleDetail data={detail} defs={detailDefs} scaleId={selected} />}
    </>}
  </div>);
}

function ScaleDetail({data, defs, scaleId}) {
  const [subTab, setSubTab] = useState('overview');

  // Adaptively render based on data structure
  const hasPatients = data.patients && data.patients.length > 0;
  const hasSummary = data.summary || data.total_patients || data.population_summary;
  const hasItems = data.items || data.components || data.domains || data.subscales;
  const hasTrend = data.trend || data.trends;

  const SUB_TABS = [
    {id: 'overview', label: 'Overview'},
    ...(hasPatients ? [{id: 'patients', label: `Patients (${data.patients?.length || 0})`}] : []),
    ...(hasItems ? [{id: 'items', label: 'Scale Items'}] : []),
    ...(defs ? [{id: 'definitions', label: 'Definitions'}] : []),
  ];

  return (<div>
    <h5 className="mb-1">{data.title || data.scale_name || scaleId.toUpperCase()}</h5>
    <p className="text-muted small">{data.subtitle || data.description || ''}</p>

    {/* Summary KPIs */}
    {hasSummary && <div className="row mb-3">
      {Object.entries(data.summary || data.population_summary || {}).filter(([,v]) => typeof v === 'number' || typeof v === 'string').slice(0, 6).map(([k, v]) =>
        <div key={k} className="col-6 col-md-2 mb-2">
          <div className="card bg-light"><div className="card-body p-2 text-center">
            <div className="fw-bold">{typeof v === 'number' ? (v % 1 === 0 ? v : v.toFixed(2)) : v}</div>
            <div className="text-muted small" style={{fontSize: '0.7rem'}}>{k.replace(/_/g, ' ')}</div>
          </div></div>
        </div>
      )}
    </div>}

    <ul className="nav nav-pills nav-fill mb-3">
      {SUB_TABS.map(t => <li key={t.id} className="nav-item">
        <button className={`nav-link ${subTab === t.id ? 'active' : ''}`} onClick={() => setSubTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {subTab === 'overview' && <OverviewSection data={data} />}
    {subTab === 'patients' && hasPatients && <PatientsTable patients={data.patients} />}
    {subTab === 'items' && hasItems && <ItemsSection data={data} />}
    {subTab === 'definitions' && defs && <DefsSection defs={defs} />}
  </div>);
}

function OverviewSection({data}) {
  // Render key fields adaptively
  const entries = Object.entries(data).filter(([k]) => !['patients', 'items', 'components', 'domains', 'subscales', 'title', 'subtitle', 'scale_name', 'description', 'summary', 'population_summary', 'trend', 'trends'].includes(k));
  return (<div className="row">
    {entries.filter(([,v]) => typeof v === 'object' && v !== null && !Array.isArray(v)).map(([k, v]) =>
      <div key={k} className="col-12 col-md-6 mb-3">
        <div className="card"><div className="card-body">
          <h6 className="card-title">{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</h6>
          <table className="table table-sm mb-0">
            <tbody>{Object.entries(v).filter(([,val]) => val !== null && typeof val !== 'object').map(([sk, sv]) =>
              <tr key={sk}><td className="text-muted small">{sk.replace(/_/g, ' ')}</td><td className="fw-bold">{typeof sv === 'number' ? (sv % 1 === 0 ? sv : sv.toFixed(2)) : String(sv)}</td></tr>
            )}</tbody>
          </table>
        </div></div>
      </div>
    )}
    {entries.filter(([,v]) => Array.isArray(v) && v.length > 0 && v.length <= 20).map(([k, v]) =>
      <div key={k} className="col-12 mb-3">
        <h6>{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())} ({v.length})</h6>
        {typeof v[0] === 'object' ? (
          <div className="table-responsive"><table className="table table-sm table-striped">
            <thead><tr>{Object.keys(v[0]).map(h => <th key={h} className="small">{h.replace(/_/g, ' ')}</th>)}</tr></thead>
            <tbody>{v.map((row, i) => <tr key={i}>{Object.values(row).map((cell, j) =>
              <td key={j} className="small">{cell === null ? '—' : typeof cell === 'number' ? (cell % 1 === 0 ? cell : cell.toFixed(2)) : String(cell)}</td>
            )}</tr>)}</tbody>
          </table></div>
        ) : <ul className="list-group list-group-flush">{v.map((item, i) => <li key={i} className="list-group-item py-1 small">{String(item)}</li>)}</ul>}
      </div>
    )}
  </div>);
}

function PatientsTable({patients}) {
  if (!patients || patients.length === 0) return <p className="text-muted">No patient data available.</p>;
  const cols = Object.keys(patients[0]);
  return (<div className="table-responsive">
    <table className="table table-sm table-hover table-striped">
      <thead className="table-dark"><tr>{cols.map(c => <th key={c} className="small">{c.replace(/_/g, ' ')}</th>)}</tr></thead>
      <tbody>{patients.map((p, i) => <tr key={i}>{cols.map(c =>
        <td key={c} className="small">{p[c] === null ? '—' : typeof p[c] === 'number' ? (p[c] % 1 === 0 ? p[c] : p[c].toFixed(2)) : String(p[c])}</td>
      )}</tr>)}</tbody>
    </table>
  </div>);
}

function ItemsSection({data}) {
  const items = data.items || data.components || data.domains || data.subscales || [];
  if (!items || items.length === 0) return <p className="text-muted">No item-level data.</p>;
  if (typeof items[0] === 'object') {
    const cols = Object.keys(items[0]);
    return (<div className="table-responsive">
      <table className="table table-sm">
        <thead><tr>{cols.map(c => <th key={c} className="small">{c.replace(/_/g, ' ')}</th>)}</tr></thead>
        <tbody>{items.map((item, i) => <tr key={i}>{cols.map(c =>
          <td key={c} className="small">{item[c] === null ? '—' : String(item[c])}</td>
        )}</tr>)}</tbody>
      </table>
    </div>);
  }
  return <ul>{items.map((item, i) => <li key={i}>{String(item)}</li>)}</ul>;
}

function DefsSection({defs}) {
  if (!defs) return null;
  return (<div>
    {typeof defs === 'object' && !Array.isArray(defs) && Object.entries(defs).map(([k, v]) =>
      <div key={k} className="mb-3">
        <h6>{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</h6>
        {typeof v === 'string' ? <p className="small">{v}</p> :
         Array.isArray(v) ? (
           typeof v[0] === 'object' ?
             <div className="table-responsive"><table className="table table-sm">
               <thead><tr>{Object.keys(v[0]).map(h => <th key={h} className="small">{h.replace(/_/g, ' ')}</th>)}</tr></thead>
               <tbody>{v.map((row, i) => <tr key={i}>{Object.values(row).map((cell, j) =>
                 <td key={j} className="small">{cell === null ? '—' : String(cell)}</td>
               )}</tr>)}</tbody>
             </table></div>
           : <ul className="list-group list-group-flush">{v.map((item, i) => <li key={i} className="list-group-item py-1 small">{String(item)}</li>)}</ul>
         ) : typeof v === 'object' ?
           <table className="table table-sm"><tbody>{Object.entries(v).map(([sk, sv]) =>
             <tr key={sk}><td className="text-muted small">{sk.replace(/_/g, ' ')}</td><td className="small">{String(sv)}</td></tr>
           )}</tbody></table>
         : <p className="small">{String(v)}</p>}
      </div>
    )}
  </div>);
}
