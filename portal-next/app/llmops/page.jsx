'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = s => s === 'high' ? 'danger' : s === 'medium' ? 'warning' : 'success';
const okBadge  = ok => ok ? 'success' : 'danger';
const pctColor = p => p > 80 ? 'danger' : p > 50 ? 'warning' : 'success';
const fmt      = n => n >= 1e6 ? `${(n/1e6).toFixed(1)}M` : n >= 1e3 ? `${(n/1e3).toFixed(1)}K` : n;

export default function LLMOpsPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/llmops/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/llmops/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/llmops/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',     label: 'Overview' },
    { id: 'prompts',      label: 'Prompts & Versions' },
    { id: 'monitoring',   label: 'Cost & Latency' },
    { id: 'rag',          label: 'RAG & Hallucination' },
    { id: 'definitions',  label: 'Definitions' },
  ];

  return (
    <div>
      <h3>LLMOps Dashboard</h3>
      <p className="text-muted">Prompt versioning, token/cost tracking, latency monitoring, hallucination detection, RAG evaluation</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Active Prompts',  value: ov.kpis.active_prompts,                     color: 'primary' },
          { label: 'Avg Accuracy',    value: `${(ov.kpis.avg_accuracy*100).toFixed(1)}%`, color: ov.kpis.avg_accuracy >= 0.90 ? 'success' : 'warning' },
          { label: 'Halluc Rate',     value: `${(ov.kpis.avg_hallucination_rate*100).toFixed(2)}%`, color: ov.kpis.avg_hallucination_rate > 0.05 ? 'danger' : 'success' },
          { label: '30d Cost',        value: `$${ov.kpis.total_cost_30d_usd}`,           color: pctColor(ov.kpis.budget_pct) },
          { label: 'Budget Used',     value: `${ov.kpis.budget_pct}%`,                   color: pctColor(ov.kpis.budget_pct) },
          { label: 'RAG Faithfulness',value: `${(ov.kpis.avg_rag_faithfulness*100).toFixed(1)}%`, color: ov.kpis.avg_rag_faithfulness >= 0.85 ? 'success' : 'warning' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ─────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Cost by Provider */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Cost by Provider (30d)</div>
              <div className="card-body">
                {Object.entries(ov.cost_by_provider || {}).map(([prov, cost]) => {
                  const maxC = Math.max(...Object.values(ov.cost_by_provider || {}), 1);
                  return (
                    <div key={prov} className="mb-2">
                      <div className="d-flex justify-content-between small"><span>{prov}</span><span className="fw-bold">${cost}</span></div>
                      <div className="progress" style={{height:'10px'}}><div className="progress-bar bg-info" style={{width:`${cost/maxC*100}%`}} /></div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Cost by Model */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Cost by Model (30d)</div>
              <div className="card-body">
                {(ov.cost_by_model || []).map(m => {
                  const maxC = Math.max(...(ov.cost_by_model||[]).map(x=>x.cost_usd), 1);
                  return (
                    <div key={m.model} className="mb-2">
                      <div className="d-flex justify-content-between small">
                        <span>{m.model}</span>
                        <span>${m.cost_usd} ({fmt(m.requests)} req)</span>
                      </div>
                      <div className="progress" style={{height:'10px'}}><div className={`progress-bar ${m.cost_usd === 0 ? 'bg-secondary' : 'bg-primary'}`} style={{width:`${Math.max(m.cost_usd/maxC*100, 2)}%`}} /></div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Hallucination by Category */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Hallucinations by Category</div>
              <div className="card-body">
                {Object.entries(ov.hallucination_by_category || {}).map(([cat, count]) => {
                  const maxH = Math.max(...Object.values(ov.hallucination_by_category || {}), 1);
                  return (
                    <div key={cat} className="mb-2">
                      <div className="d-flex justify-content-between small">
                        <span>{cat.replace(/_/g, ' ')}</span><span className="fw-bold">{count}</span>
                      </div>
                      <div className="progress" style={{height:'8px'}}><div className="progress-bar bg-warning" style={{width:`${count/maxH*100}%`}} /></div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Prompt Summary Table */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Prompt Template Summary</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark"><tr>
                      <th>Prompt</th><th>Category</th><th>Model</th><th>Versions</th>
                      <th>Accuracy</th><th>Halluc Rate</th><th>Latency (ms)</th>
                    </tr></thead>
                    <tbody>
                      {(ov.prompt_summary || []).map(p => (
                        <tr key={p.prompt}>
                          <td>{p.prompt}</td>
                          <td><span className="badge bg-secondary">{p.category}</span></td>
                          <td>{p.model}</td>
                          <td>{p.versions}</td>
                          <td><span className={`badge bg-${p.accuracy >= 0.90 ? 'success' : 'warning'}`}>{(p.accuracy*100).toFixed(1)}%</span></td>
                          <td><span className={`badge bg-${p.halluc_rate > 0.05 ? 'danger' : 'success'}`}>{(p.halluc_rate*100).toFixed(2)}%</span></td>
                          <td>{p.latency_ms}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Prompts & Versions Tab ───────────────────────────── */}
      {tab === 'prompts' && bd && (
        <div>
          {(bd.prompt_versions || []).map(pv => (
            <div key={pv.prompt_id} className="card shadow-sm mb-3">
              <div className="card-header d-flex justify-content-between align-items-center">
                <span className="fw-bold">{pv.label}</span>
                <span className="badge bg-info">v{pv.current_version} active &middot; {pv.total_versions} versions</span>
              </div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead className="table-light"><tr>
                      <th>Version</th><th>Status</th><th>Tokens (avg)</th>
                      <th>Accuracy</th><th>Halluc Rate</th><th>Latency (ms)</th><th>Age (days)</th>
                    </tr></thead>
                    <tbody>
                      {pv.versions.map(v => (
                        <tr key={v.version} className={v.status === 'active' ? 'table-success' : ''}>
                          <td>v{v.version}</td>
                          <td><span className={`badge bg-${v.status === 'active' ? 'success' : 'secondary'}`}>{v.status}</span></td>
                          <td>{fmt(v.avg_tokens)}</td>
                          <td>{(v.accuracy*100).toFixed(1)}%</td>
                          <td>{(v.hallucination_rate*100).toFixed(2)}%</td>
                          <td>{v.avg_latency_ms}</td>
                          <td>{v.created_days_ago}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Cost & Latency Tab ───────────────────────────────── */}
      {tab === 'monitoring' && bd && (
        <div className="row">
          {/* Model Cost Totals */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Model Cost Summary (30d) &middot; Budget ${bd.budget_usd}</div>
              <div className="card-body">
                <div className="mb-2">
                  <div className="d-flex justify-content-between small fw-bold">
                    <span>Total: ${bd.total_cost_30d_usd}</span>
                    <span className={`text-${pctColor(bd.budget_pct)}`}>{bd.budget_pct}% of budget</span>
                  </div>
                  <div className="progress mb-3" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${pctColor(bd.budget_pct)}`} style={{width:`${Math.min(bd.budget_pct,100)}%`}}>{bd.budget_pct}%</div>
                  </div>
                </div>
                {Object.entries(bd.model_totals || {}).map(([mid, mt]) => (
                  <div key={mid} className="d-flex justify-content-between border-bottom py-1 small">
                    <span>{mid}</span>
                    <span>{fmt(mt.requests)} req &middot; {fmt(mt.input_tokens+mt.output_tokens)} tok &middot; <strong>${mt.cost_usd}</strong></span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Latency Metrics */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Latency Percentiles</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead className="table-dark"><tr>
                      <th>Prompt</th><th>P50</th><th>P95</th><th>P99</th><th>Timeout%</th>
                    </tr></thead>
                    <tbody>
                      {(bd.latency_metrics || []).map(l => (
                        <tr key={l.prompt_id}>
                          <td className="small">{l.label}</td>
                          <td><span className={`badge bg-${okBadge(l.p50_ok)}`}>{l.p50_ms}ms</span></td>
                          <td><span className={`badge bg-${okBadge(l.p95_ok)}`}>{l.p95_ms}ms</span></td>
                          <td><span className={`badge bg-${okBadge(l.p99_ok)}`}>{l.p99_ms}ms</span></td>
                          <td>{(l.timeout_rate*100).toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="p-2 small text-muted">
                  Targets: P50 &le; {bd.latency_thresholds?.p50_target_ms}ms &middot;
                  P95 &le; {bd.latency_thresholds?.p95_target_ms}ms &middot;
                  P99 &le; {bd.latency_thresholds?.p99_target_ms}ms
                </div>
              </div>
            </div>
          </div>

          {/* Daily Token Usage (last 14 days) */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Daily Token Usage (14d)</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-light"><tr>
                      <th>Day</th>
                      {(bd.daily_token_usage?.[0] ? Object.keys(bd.daily_token_usage[0].models || {}) : []).map(m => (
                        <th key={m} colSpan="2" className="text-center small">{m}</th>
                      ))}
                    </tr>
                    <tr>
                      <th></th>
                      {(bd.daily_token_usage?.[0] ? Object.keys(bd.daily_token_usage[0].models || {}) : []).map(m => (
                        <><th key={`${m}-tok`} className="small">Tokens</th><th key={`${m}-cost`} className="small">Cost</th></>
                      ))}
                    </tr>
                    </thead>
                    <tbody>
                      {(bd.daily_token_usage || []).slice(0, 7).map(d => (
                        <tr key={d.day_offset}>
                          <td>D-{d.day_offset}</td>
                          {Object.entries(d.models || {}).map(([mid, mv]) => (
                            <><td key={`${mid}-t`} className="small">{fmt(mv.input_tokens + mv.output_tokens)}</td><td key={`${mid}-c`} className="small">${mv.cost_usd}</td></>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── RAG & Hallucination Tab ──────────────────────────── */}
      {tab === 'rag' && bd && (
        <div className="row">
          {/* RAG Evaluation */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">RAG Pipeline Evaluation</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead className="table-dark"><tr>
                      <th>Pipeline</th><th>Vector DB</th><th>Chunk</th><th>Docs</th>
                      <th>Recall@5</th><th>Prec@5</th><th>MRR</th><th>NDCG</th>
                      <th>Faithfulness</th><th>Relevance</th><th>Latency</th><th>Index</th>
                    </tr></thead>
                    <tbody>
                      {(bd.rag_evaluation || []).map(r => (
                        <tr key={r.pipeline_id}>
                          <td className="fw-bold">{r.label}</td>
                          <td>{r.vector_db}</td>
                          <td>{r.chunk_size}/{r.overlap}</td>
                          <td>{fmt(r.total_documents)}</td>
                          <td><span className={`badge bg-${r.recall_at_5 >= 0.80 ? 'success' : 'warning'}`}>{(r.recall_at_5*100).toFixed(1)}%</span></td>
                          <td><span className={`badge bg-${r.precision_at_5 >= 0.75 ? 'success' : 'warning'}`}>{(r.precision_at_5*100).toFixed(1)}%</span></td>
                          <td>{(r.mrr*100).toFixed(1)}%</td>
                          <td>{(r.ndcg*100).toFixed(1)}%</td>
                          <td><span className={`badge bg-${r.faithfulness >= 0.85 ? 'success' : 'warning'}`}>{(r.faithfulness*100).toFixed(1)}%</span></td>
                          <td>{(r.relevance*100).toFixed(1)}%</td>
                          <td>{r.retrieval_latency_ms}ms</td>
                          <td><span className={`badge bg-${r.index_stale ? 'danger' : 'success'}`}>{r.index_stale ? 'stale' : 'fresh'} ({r.last_indexed_days_ago}d)</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Hallucination Detail */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Hallucination Report by Prompt</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead className="table-dark"><tr>
                      <th>Prompt</th><th>Evaluations</th><th>Detected</th><th>Rate</th><th>Severity</th>
                      <th>Factual</th><th>Unsupported</th><th>Contradicts</th><th>Fabricated</th><th>Numerical</th><th>Temporal</th>
                    </tr></thead>
                    <tbody>
                      {(bd.hallucination_report || []).map(h => (
                        <tr key={h.prompt_id}>
                          <td className="small">{h.label}</td>
                          <td>{h.total_evaluations}</td>
                          <td>{h.hallucinations_detected}</td>
                          <td><span className={`badge bg-${sevColor(h.severity)}`}>{(h.hallucination_rate*100).toFixed(2)}%</span></td>
                          <td><span className={`badge bg-${sevColor(h.severity)}`}>{h.severity}</span></td>
                          {['factual_error','unsupported_claim','contradicts_source','fabricated_reference','numerical_error','temporal_confusion'].map(cat => (
                            <td key={cat}>{h.by_category?.[cat] || 0}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ──────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Models</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Model</th><th>Provider</th><th>Input $/1K</th><th>Output $/1K</th></tr></thead>
                  <tbody>{(defs.models||[]).map(m => (
                    <tr key={m.id}><td>{m.label}</td><td>{m.provider}</td><td>${m.cost_per_1k_input}</td><td>${m.cost_per_1k_output}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Prompt Templates</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Prompt</th><th>Category</th><th>Model</th></tr></thead>
                  <tbody>{(defs.prompt_templates||[]).map(pt => (
                    <tr key={pt.id}><td>{pt.label}</td><td><span className="badge bg-secondary">{pt.category}</span></td><td>{pt.model_id}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">RAG Pipelines</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Pipeline</th><th>Vector DB</th><th>Chunk Size</th><th>Overlap</th></tr></thead>
                  <tbody>{(defs.rag_pipelines||[]).map(r => (
                    <tr key={r.id}><td>{r.label}</td><td>{r.vector_db}</td><td>{r.chunk_size}</td><td>{r.overlap}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Hallucination Categories</div>
              <div className="card-body">
                {Object.entries(defs.hallucination_categories || {}).map(([cat, desc]) => (
                  <div key={cat} className="mb-2"><strong>{cat.replace(/_/g,' ')}:</strong> <span className="text-muted">{desc}</span></div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">RAG Metrics</div>
              <div className="card-body">
                {Object.entries(defs.rag_metrics || {}).map(([met, desc]) => (
                  <div key={met} className="mb-2"><strong>{met}:</strong> <span className="text-muted">{desc}</span></div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Latency Thresholds</div>
              <div className="card-body">
                {Object.entries(defs.latency_thresholds || {}).map(([k, v]) => (
                  <div key={k} className="d-flex justify-content-between border-bottom py-1">
                    <span>{k.replace(/_/g,' ')}</span><span className="fw-bold">{v}ms</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Clinical Relevance</div>
              <div className="card-body"><p className="mb-0">{defs.clinical_relevance}</p></div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
