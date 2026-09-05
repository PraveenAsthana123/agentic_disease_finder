'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// Aortic-CTD Atlas color palette — vascular / connective tissue
const COLOR  = '#880e4f';  // deep burgundy — aortic / vascular danger
const LIGHT  = '#fce4ec';  // light pink tint
const COLOR2 = '#1a237e';  // deep blue — LDS / TGF-β pathway
const COLOR3 = '#1b5e20';  // dark green — treatable / celiprolol
const COLOR4 = '#e65100';  // orange — vEDS / WARNING
const COLOR5 = '#4a148c';  // dark purple — SMAD3 / dual phenotype
const COLOR6 = '#37474f';  // blue-grey — MYH11 / PDA
const COLOR7 = '#bf360c';  // dark orange — SKI / craniosynostosis

const GENE_COLORS = {
  FBN1:   '#1565c0',  // blue — Marfan / extracellular matrix
  TGFBR2: '#c62828',  // red — LDS2 / aggressive aortic
  TGFBR1: '#ad1457',  // dark pink — LDS1 / craniosynostosis
  COL3A1: '#e65100',  // orange — vEDS / NO surgery WARNING
  ACTA2:  '#6a1b9a',  // purple — FTAAD / Moya-Moya
  SMAD3:  '#00695c',  // teal — AOS / dual OA+aneurysm
  MYH11:  '#37474f',  // blue-grey — FTAAD+PDA
  SKI:    '#bf360c',  // dark orange — Shprintzen-Goldberg
};

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function BarRow({ label, pct, color = COLOR, note }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between mb-1">
        <span className="small fw-semibold">{label}</span>
        <span className="small text-muted">{typeof pct === 'number' ? `${pct}%` : pct}{note ? ` — ${note}` : ''}</span>
      </div>
      {typeof pct === 'number' && (
        <div className="progress" style={{ height: 8 }}>
          <div className="progress-bar" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color }} />
        </div>
      )}
    </div>
  );
}

function AlertBox({ type = 'info', title, children }) {
  const icons = { danger: '🚨', warning: '⚠️', info: 'ℹ️', success: '✅' };
  return (
    <div className={`alert alert-${type} py-2 px-3 mb-3`}>
      <strong>{icons[type]} {title}</strong>
      <div className="small mt-1">{children}</div>
    </div>
  );
}

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border" style={{ color: COLOR }} />
      <div className="mt-2 text-muted small">Loading Aortic-CTD Atlas…</div>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger">{msg}</div>;
}

// ── Tab: Overview ──────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const sp = ov.severity_prevalence || {};
  const cf = ov.clinical_features_prevalence || {};
  const sev = ov.severity || {};
  const cat = ov.disease_category_breakdown || {};
  const kpis = ov.kpis || [];

  return (
    <div>
      <h5 className="fw-bold mb-1" style={{ color: COLOR }}>{ov.full_name}</h5>
      <p className="text-muted small mb-3">{ov.subtitle}</p>
      <p className="mb-3">{ov.description}</p>

      {/* Drug / clinical alerts */}
      {(ov.drug_alerts || []).map((a, i) => (
        <AlertBox key={i}
          type={a.type || 'info'}
          title={a.title}>
          {a.body}
        </AlertBox>
      ))}

      {/* Critical Rules */}
      {(ov.critical_rules || []).length > 0 && (
        <div className="alert alert-secondary py-2 px-3 mb-3">
          <strong>📋 Critical Clinical Rules</strong>
          <ul className="mb-0 mt-1 small">
            {ov.critical_rules.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        </div>
      )}

      {/* KPIs */}
      <div className="row g-2 mb-4">
        {kpis.map((k, i) => (
          <KPI key={i} label={k.label} value={k.value}
            color={i % 4 === 0 ? COLOR : i % 4 === 1 ? COLOR2 : i % 4 === 2 ? COLOR3 : COLOR4} />
        ))}
      </div>

      <div className="row g-3">
        {/* Severity */}
        <div className="col-md-4">
          <div className="card shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Severity Distribution</h6>
              {Object.entries(sp).map(([s, v]) => (
                <BarRow key={s} label={s} pct={v}
                  color={s === 'Severe' ? '#c62828' : s === 'Moderate' ? '#e65100' : '#2e7d32'} />
              ))}
            </div>
          </div>
        </div>

        {/* Clinical Features */}
        <div className="col-md-4">
          <div className="card shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3" style={{ color: COLOR2 }}>Clinical Features Prevalence</h6>
              {Object.entries(cf).map(([f, v]) => (
                <BarRow key={f} label={f} pct={v} color={COLOR2} />
              ))}
            </div>
          </div>
        </div>

        {/* Disease Categories */}
        <div className="col-md-4">
          <div className="card shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3" style={{ color: COLOR3 }}>Disease Category Breakdown</h6>
              {Object.entries(cat).map(([c, v]) => (
                <BarRow key={c} label={c} pct={v} color={COLOR3} />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Gene Surgery Rate */}
      {ov.gene_surgery_pct && (
        <div className="card shadow-sm mt-3">
          <div className="card-body">
            <h6 className="fw-bold mb-3" style={{ color: COLOR4 }}>Aortic Surgery Rate by Gene (%)</h6>
            <div className="row">
              {Object.entries(ov.gene_surgery_pct).map(([g, v]) => (
                <div key={g} className="col-6 col-md-3 mb-2">
                  <BarRow label={g} pct={v} color={GENE_COLORS[g] || COLOR} />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Tab: Gene Table ────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h5 className="fw-bold mb-3" style={{ color: COLOR }}>Per-Gene Clinical Breakdown</h5>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th>
              <th>Disease</th>
              <th>Locus</th>
              <th>Surgery %</th>
              <th>Dissection %</th>
              <th>Lens Ectopia %</th>
              <th>Skeletal %</th>
              <th>De Novo %</th>
              <th>Mean Onset</th>
            </tr>
          </thead>
          <tbody>
            {data.map((g) => (
              <tr key={g.gene}>
                <td>
                  <span className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>
                    {g.gene}
                  </span>
                </td>
                <td className="small text-muted" style={{ maxWidth: 200 }}>
                  <span title={g.disease_type}>{g.disease_type?.slice(0, 50)}…</span>
                </td>
                <td><code className="small">{g.locus}</code></td>
                <td>
                  <span className={`badge ${g.aortic_surgery_pct >= 60 ? 'bg-danger' : g.aortic_surgery_pct >= 40 ? 'bg-warning text-dark' : 'bg-success'}`}>
                    {g.aortic_surgery_pct}%
                  </span>
                </td>
                <td>
                  <span className={`badge ${g.dissection_pct >= 50 ? 'bg-danger' : g.dissection_pct >= 30 ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                    {g.dissection_pct}%
                  </span>
                </td>
                <td>{g.lens_ectopia_pct}%</td>
                <td>{g.skeletal_features_pct}%</td>
                <td>
                  <span className={`badge ${g.de_novo_pct >= 70 ? 'bg-primary' : g.de_novo_pct >= 40 ? 'bg-info text-dark' : 'bg-secondary'}`}>
                    {g.de_novo_pct}%
                  </span>
                </td>
                <td>{g.mean_onset_age_y}y</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-gene cards */}
      <div className="row g-3 mt-2">
        {data.map((g) => (
          <div key={g.gene} className="col-md-6 col-lg-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold py-2" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR, color: '#fff' }}>
                {g.gene} — {g.protein?.split(' (')[0]}
              </div>
              <div className="card-body small">
                <p className="mb-2"><strong>Key distinguishing:</strong> {g.key_distinguishing}</p>
                {(g.drug_alerts || []).map((a, i) => (
                  <div key={i} className={`alert alert-${a.type || 'info'} py-1 px-2 mb-2`}>
                    <strong className="small">{a.title}</strong>
                    <div className="small mt-1">{a.body}</div>
                  </div>
                ))}
                <strong>Clinical Rules:</strong>
                <ul className="mb-0 mt-1">
                  {(g.clinical_rules || []).map((r, i) => <li key={i}>{r}</li>)}
                </ul>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Tab: Clinical Atlas ────────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h5 className="fw-bold mb-3" style={{ color: COLOR }}>Clinical Atlas — Treatment & Pathomechanism</h5>
      {data.map((g) => (
        <div key={g.gene} className="card shadow-sm mb-4">
          <div className="card-header fw-bold py-2" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR, color: '#fff' }}>
            {g.gene} ({g.locus}) — OMIM Gene #{g.omim_gene} / Disease #{g.omim_disease}
          </div>
          <div className="card-body">
            <div className="row g-3">
              <div className="col-md-7">
                <h6 className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>Disease Type</h6>
                <p className="small mb-3">{g.disease_type}</p>
                <h6 className="fw-bold">Treatment Options</h6>
                <ul className="small">
                  {(g.treatment_options || []).map((t, i) => <li key={i} className="mb-1">{t}</li>)}
                </ul>
              </div>
              <div className="col-md-5">
                <h6 className="fw-bold">Per-Gene Stats</h6>
                <BarRow label="Surgery Rate" pct={g.aortic_surgery_pct} color={GENE_COLORS[g.gene] || COLOR} />
                <BarRow label="Dissection Risk" pct={g.dissection_pct} color="#c62828" />
                <BarRow label="Lens Ectopia" pct={g.lens_ectopia_pct} color="#1565c0" />
                <BarRow label="Skeletal Features" pct={g.skeletal_features_pct} color="#e65100" />
                <BarRow label="De Novo Variants" pct={g.de_novo_pct} color="#6a1b9a" />
                <div className="mt-2 small">
                  <span className="badge bg-secondary me-1">Prevalence: ~{g.prevalence_per_100k}/100k</span>
                  <span className="badge bg-dark">Onset: ~{g.mean_onset_age_y}y</span>
                </div>
                <h6 className="fw-bold mt-3">Alerts</h6>
                {(g.drug_alerts || []).map((a, i) => (
                  <AlertBox key={i} type={a.type || 'info'} title={a.title}>
                    {a.body}
                  </AlertBox>
                ))}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const [q, setQ] = useState('');
  const filtered = q
    ? data.filter(d => d.term.toLowerCase().includes(q.toLowerCase()) || d.definition.toLowerCase().includes(q.toLowerCase()))
    : data;

  return (
    <div>
      <h5 className="fw-bold mb-3" style={{ color: COLOR }}>Clinical Terminology — Aortic & CTD Atlas</h5>
      <input
        className="form-control mb-3"
        placeholder="Search terms…"
        value={q}
        onChange={e => setQ(e.target.value)}
      />
      <div className="row g-3">
        {filtered.map((d, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-body">
                <h6 className="fw-bold mb-2" style={{ color: COLOR }}>{d.term}</h6>
                <p className="small mb-0">{d.definition}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
      {filtered.length === 0 && <p className="text-muted">No matching terms.</p>}
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────
export default function AorticCTDAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    const load = async (path, setter) => {
      try {
        const r = await fetch(`${API}${path}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        setter(await r.json());
      } catch (e) {
        setErr(e.message);
      }
    };
    load('/api/aortic-ctd-atlas/overview', setOverview);
    load('/api/aortic-ctd-atlas/breakdown', setBreakdown);
    load('/api/aortic-ctd-atlas/definitions', setDefinitions);
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2 flex-wrap">
        <span style={{ fontSize: 28 }}>🫀</span>
        <div>
          <h4 className="fw-bold mb-0" style={{ color: COLOR }}>Aortic-CTD Atlas</h4>
          <div className="text-muted small">
            Complete 8-Gene Hereditary Aortic &amp; Connective Tissue Disorders Atlas ·
            FBN1·TGFBR2·TGFBR1·COL3A1·ACTA2·SMAD3·MYH11·SKI · 320 patients (8×40, seeds 1110–1117)
          </div>
        </div>
      </div>

      {err && <ErrorMsg msg={err} />}

      {/* Tab navigation */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active fw-bold' : ''}`}
              style={tab === t ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(t)}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'      && <OverviewTab data={overview} />}
      {tab === 'Gene Table'    && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'   && <DefinitionsTab data={definitions} />}
    </div>
  );
}
