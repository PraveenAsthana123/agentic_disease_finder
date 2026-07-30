'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const RISK_COLORS = { high: 'danger', moderate: 'warning', low: 'success' };
const MET_COLORS = {
  'Poor Metabolizer': 'danger', 'Intermediate Metabolizer': 'warning',
  'Normal Metabolizer': 'success', 'Rapid Metabolizer': 'info',
  'Ultra-rapid Metabolizer': 'primary', 'Carrier': 'dark',
};
const EV_COLORS = { '1A': 'danger', '1B': 'warning', '2A': 'info', '2B': 'secondary', '3': 'light' };

export default function PharmacogenomicsPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/pharmacogenomics/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/pharmacogenomics/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/pharmacogenomics/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const k = overview.kpis || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'high-risk', label: 'High Risk' },
    { id: 'patients', label: 'Per Patient' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f9ec; Pharmacogenomics Dashboard</h3>
      <p className="text-muted">Gene-drug interactions for AED prescribing — CYP2C9, CYP2C19, HLA-B, SCN1A and more from clinical.db ({k.total_tests} tests, {k.total_patients} patients)</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Tests', value: k.total_tests, color: 'primary' },
          { label: 'Patients', value: k.total_patients, color: 'info' },
          { label: 'Unique Genes', value: k.unique_genes, color: 'success' },
          { label: 'High Risk', value: k.high_risk_count, color: 'danger' },
          { label: 'Actionable', value: k.actionable_results, color: 'warning' },
          { label: 'Poor Metabolizers', value: k.poor_metabolizer_count, color: 'dark' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value ?? '\u2014'}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Gene distribution */}
      <div className="card mb-3 shadow-sm border-0">
        <div className="card-body">
          <h6 className="card-title">Gene Distribution</h6>
          <div className="progress" style={{height: '28px'}}>
            {(overview.gene_distribution || []).map((g, i) => {
              const colors = ['primary','success','info','warning','danger','secondary','dark'];
              return (
                <div key={g.gene} className={`progress-bar bg-${colors[i % colors.length]}`}
                  style={{width: `${g.pct}%`}} title={`${g.gene}: ${g.count} (${g.pct}%)`}>
                  {g.gene} {g.count}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Source + Evidence row */}
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-body">
              <h6 className="card-title">Source Distribution</h6>
              {(overview.source_distribution || []).map(s => (
                <div key={s.source} className="d-flex justify-content-between mb-1">
                  <span>{s.source}</span>
                  <span className="badge bg-primary">{s.count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-body">
              <h6 className="card-title">Evidence Levels</h6>
              {(overview.evidence_level_distribution || []).map(e => (
                <div key={e.evidence_level} className="d-flex justify-content-between mb-1">
                  <span>Level {e.evidence_level}</span>
                  <span className={`badge bg-${EV_COLORS[e.evidence_level] || 'secondary'}`}>{e.count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Metabolizer distribution */}
      <div className="card mb-3 shadow-sm border-0">
        <div className="card-body">
          <h6 className="card-title">Metabolizer Status Distribution</h6>
          <div className="d-flex flex-wrap gap-2">
            {(overview.metabolizer_distribution || []).map(m => (
              <span key={m.metabolizer_status} className={`badge bg-${MET_COLORS[m.metabolizer_status] || 'secondary'} fs-6`}>
                {m.metabolizer_status}: {m.count}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview tab — high-risk genes */}
      {tab === 'overview' && (
        <div>
          <h5>High-Risk Genes</h5>
          <div className="row mb-3">
            {(overview.high_risk_by_gene || []).map(g => (
              <div key={g.gene} className="col-md-4 mb-2">
                <div className="card shadow-sm border-0">
                  <div className="card-body text-center">
                    <div className="h4 text-danger mb-0">{g.count}</div>
                    <div className="text-muted">{g.gene}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* High Risk tab */}
      {tab === 'high-risk' && breakdown && (
        <div>
          <h5>High-Risk Results ({(breakdown.high_risk_results || []).length})</h5>
          <div className="table-responsive">
            <table className="table table-hover table-sm">
              <thead className="table-dark">
                <tr><th>Patient</th><th>Gene</th><th>Variant</th><th>Significance</th><th>Affected Drugs</th><th>Recommendation</th><th>Evidence</th></tr>
              </thead>
              <tbody>
                {(breakdown.high_risk_results || []).map((r, i) => (
                  <tr key={i}>
                    <td><code>{r.patient_id}</code></td>
                    <td><strong>{r.gene}</strong></td>
                    <td className="text-muted small">{r.variant}</td>
                    <td><span className="badge bg-danger">{r.clinical_significance}</span></td>
                    <td className="small">{r.affected_drugs}</td>
                    <td className="small">{r.recommendation}</td>
                    <td><span className={`badge bg-${EV_COLORS[r.evidence_level] || 'secondary'}`}>{r.evidence_level}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Poor metabolizers */}
          <h5 className="mt-4">Poor Metabolizers ({(breakdown.poor_metabolizers || []).length})</h5>
          <div className="table-responsive">
            <table className="table table-hover table-sm">
              <thead className="table-dark">
                <tr><th>Patient</th><th>Gene</th><th>Variant</th><th>Status</th><th>Affected Drugs</th><th>Recommendation</th></tr>
              </thead>
              <tbody>
                {(breakdown.poor_metabolizers || []).map((r, i) => (
                  <tr key={i}>
                    <td><code>{r.patient_id}</code></td>
                    <td><strong>{r.gene}</strong></td>
                    <td className="text-muted small">{r.variant}</td>
                    <td><span className="badge bg-warning text-dark">{r.metabolizer_status}</span></td>
                    <td className="small">{r.affected_drugs}</td>
                    <td className="small">{r.recommendation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Per Patient tab */}
      {tab === 'patients' && breakdown && (
        <div>
          <h5>Per-Patient Pharmacogenomic Profile</h5>
          <div className="table-responsive">
            <table className="table table-hover table-sm">
              <thead className="table-dark">
                <tr><th>Patient</th><th>Tests</th><th>Genes</th><th>High Risk</th><th>Actionable</th><th>Details</th></tr>
              </thead>
              <tbody>
                {(breakdown.per_patient || []).map(p => (
                  <>
                    <tr key={p.patient_id}>
                      <td><code>{p.patient_id}</code></td>
                      <td>{p.total_tests}</td>
                      <td>{p.genes_tested}</td>
                      <td>{p.high_risk_count > 0 ? <span className="badge bg-danger">{p.high_risk_count}</span> : <span className="text-muted">0</span>}</td>
                      <td>{p.actionable_count > 0 ? <span className="badge bg-warning text-dark">{p.actionable_count}</span> : <span className="text-muted">0</span>}</td>
                      <td>
                        <button className="btn btn-sm btn-outline-primary" onClick={() => setExpandedPt(expandedPt === p.patient_id ? null : p.patient_id)}>
                          {expandedPt === p.patient_id ? 'Hide' : 'Show'}
                        </button>
                      </td>
                    </tr>
                    {expandedPt === p.patient_id && (p.results || []).map((r, i) => (
                      <tr key={`${p.patient_id}-${i}`} className="table-light">
                        <td></td>
                        <td colSpan={2}><strong>{r.gene}</strong> {r.variant}</td>
                        <td><span className={`badge bg-${MET_COLORS[r.metabolizer_status] || 'secondary'}`}>{r.metabolizer_status}</span></td>
                        <td colSpan={2} className="small">{r.recommendation}</td>
                      </tr>
                    ))}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div>
          <h5>Gene Descriptions</h5>
          {Object.entries(defs.gene_descriptions || {}).map(([gene, desc]) => (
            <div key={gene} className="card mb-2 shadow-sm border-0">
              <div className="card-body py-2">
                <strong>{gene}</strong>
                <p className="mb-0 small text-muted">{desc}</p>
              </div>
            </div>
          ))}

          <h5 className="mt-4">Metabolizer Categories</h5>
          {Object.entries(defs.metabolizer_categories || {}).map(([cat, desc]) => (
            <div key={cat} className="d-flex justify-content-between border-bottom py-1">
              <strong className="small">{cat}</strong>
              <span className="small text-muted ms-3">{desc}</span>
            </div>
          ))}

          <h5 className="mt-4">Evidence Levels</h5>
          {Object.entries(defs.evidence_levels || {}).map(([lvl, desc]) => (
            <div key={lvl} className="d-flex align-items-center gap-2 mb-1">
              <span className={`badge bg-${EV_COLORS[lvl] || 'secondary'}`}>{lvl}</span>
              <span className="small">{desc}</span>
            </div>
          ))}

          <h5 className="mt-4">Clinical Notes</h5>
          <ul className="list-group list-group-flush">
            {(defs.clinical_notes || []).map((note, i) => (
              <li key={i} className="list-group-item small">{note}</li>
            ))}
          </ul>

          {defs.glossary && (
            <>
              <h5 className="mt-4">Glossary</h5>
              {Object.entries(defs.glossary).map(([term, def]) => (
                <div key={term} className="mb-1">
                  <strong className="small">{term}:</strong> <span className="small text-muted">{def}</span>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
