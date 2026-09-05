'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ATP7B:    '#5c4033',   // dark brown — copper
  HFE:      '#b71c1c',   // deep red — iron
  SERPINA1: '#e65100',   // deep orange — lung/liver
  ABCB11:   '#1a237e',   // deep navy — bile
  ATP8B1:   '#1b5e20',   // deep green — FIC1
  ABCB4:    '#4a148c',   // deep purple — MDR3
  JAG1:     '#006064',   // teal — Notch/biliary
  SLC25A13: '#880e4f',   // dark magenta — citrin
};

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-warning" role="status" />
      <p className="mt-3 text-muted">Loading hepatic atlas…</p>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return (
    <div className="alert alert-danger m-4">
      <strong>Error:</strong> {msg}
    </div>
  );
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm border-0">
        <div className="card-body text-center p-3">
          <div className="fw-bold fs-3" style={{ color: color || '#5c4033' }}>{value}</div>
          <div className="small text-muted">{label}</div>
        </div>
      </div>
    </div>
  );
}

function BarRow({ label, pct, color }) {
  const p = Math.min(100, Math.max(0, Math.round(pct || 0)));
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-bold">{p}%</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${p}%`, backgroundColor: color || '#5c4033' }} />
      </div>
    </div>
  );
}

/* ── Overview tab ─────────────────────────────────────────────────────── */
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const agg = ov.aggregate_clinical || {};
  const alerts = ov.drug_alerts || [];
  const rules = ov.critical_rules || [];
  const pathways = ov.pathway_targets || {};
  const kpis = ov.kpis || [];

  return (
    <div>
      {/* Hero */}
      <div className="rounded-3 p-4 mb-4 text-white" style={{ background: 'linear-gradient(135deg,#5c4033,#1a237e)' }}>
        <h3 className="mb-1">{ov.atlas_name || 'Hepatic Atlas'}</h3>
        <p className="mb-1 opacity-75">{ov.atlas_subtitle}</p>
        <div className="d-flex flex-wrap gap-3 mt-3">
          <span className="badge bg-white text-dark fs-6">{ov.n_genes} Genes</span>
          <span className="badge bg-white text-dark fs-6">{ov.n_patients} Patients</span>
          <span className="badge bg-white text-dark fs-6">Seeds {ov.seeds}</span>
        </div>
      </div>

      {/* Drug / Clinical Alerts */}
      <h5 className="mb-3">Critical Drug &amp; Clinical Alerts</h5>
      {alerts.map((a, i) => (
        <div key={i} className={`alert alert-${a.type === 'danger' ? 'danger' : 'warning'} mb-3`}>
          <strong>{a.title}</strong>
          <p className="mb-0 mt-1 small">{a.body}</p>
        </div>
      ))}

      {/* KPIs */}
      <h5 className="mt-4 mb-3">Aggregate Cohort KPIs (n={ov.n_patients})</h5>
      <div className="row">
        {kpis.map((k, i) => (
          <KPI key={i} label={k.label} value={k.value} color={Object.values(GENE_COLORS)[i % 8]} />
        ))}
      </div>

      {/* Aggregate bars */}
      <div className="row mt-2">
        <div className="col-md-6">
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: '#fff8e1' }}>Clinical Rates (Aggregate)</div>
            <div className="card-body">
              <BarRow label="Liver Disease"          pct={agg.liver_disease_pct}        color="#5c4033" />
              <BarRow label="Neuro Involvement"      pct={agg.neuro_involvement_pct}    color="#1a237e" />
              <BarRow label="KF Rings (Wilson's)"    pct={agg.kf_rings_pct}             color="#4a148c" />
              <BarRow label="Haemolytic Anaemia"     pct={agg.haemolytic_anaemia_pct}   color="#b71c1c" />
              <BarRow label="Psychiatric Symptoms"   pct={agg.psychiatric_involvement_pct} color="#880e4f" />
              <BarRow label="Renal Involvement"      pct={agg.renal_involvement_pct}    color="#006064" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: '#fff8e1' }}>Pathway Targets</div>
            <div className="card-body">
              {Object.entries(pathways).map(([key, pathway]) => (
                <div key={key} className="d-flex align-items-start mb-2">
                  <span className="badge me-2 flex-shrink-0"
                    style={{ background: GENE_COLORS[key.split('_')[0]] || '#6c757d', fontSize: '0.72rem', minWidth: 70 }}>
                    {key}
                  </span>
                  <small className="text-muted">{pathway}</small>
                </div>
              ))}
            </div>
          </div>
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: '#fff8e1' }}>Outcomes &amp; Process</div>
            <div className="card-body">
              <BarRow label="Transplant Rate"        pct={agg.transplant_rate_pct}      color="#5c4033" />
              <BarRow label="Drug Error"             pct={agg.drug_error_pct}           color="#b71c1c" />
              <BarRow label="Delayed Diagnosis"      pct={agg.diagnosis_delayed_pct}    color="#e65100" />
              <BarRow label="Surveillance Adherent"  pct={agg.surveillance_adherent_pct} color="#1b5e20" />
            </div>
          </div>
        </div>
      </div>

      {/* Critical Rules */}
      {rules.length > 0 && (
        <div className="card border-danger border-2 mb-3">
          <div className="card-header bg-danger text-white fw-bold">Critical Rules — Must Know</div>
          <div className="card-body">
            <ul className="mb-0">
              {rules.map((r, i) => <li key={i} className="mb-1 small">{r}</li>)}
            </ul>
          </div>
        </div>
      )}

      {/* Description */}
      {ov.description && (
        <div className="alert alert-light border mt-2">
          <small className="text-muted">{ov.description}</small>
        </div>
      )}
    </div>
  );
}

/* ── Gene Table tab ───────────────────────────────────────────────────── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || [];

  return (
    <div>
      <h5 className="mb-3">Per-Gene Cohort Breakdown (40 patients each)</h5>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th>
              <th>Protein</th>
              <th>Locus</th>
              <th>n</th>
              <th>Liver %</th>
              <th>Neuro %</th>
              <th>Transplant %</th>
              <th>Drug Err %</th>
              <th>GGT Pattern</th>
            </tr>
          </thead>
          <tbody>
            {genes.map(g => (
              <tr key={g.gene}>
                <td>
                  <span className="badge" style={{ background: GENE_COLORS[g.gene] || '#6c757d', fontSize: '0.85rem' }}>
                    {g.gene}
                  </span>
                </td>
                <td><small>{g.protein}</small></td>
                <td><code>{g.locus}</code></td>
                <td>{g.cohort_n}</td>
                <td><span className="fw-bold">{g.liver_disease_pct}%</span></td>
                <td>{g.neuro_involvement_pct}%</td>
                <td>{g.transplant_pct}%</td>
                <td>{g.drug_error_pct}%</td>
                <td><small className="text-muted">{(g.ggt_pattern || '').slice(0, 40)}…</small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-gene cards */}
      {genes.map(g => (
        <div key={g.gene} className="card mb-3 border-0 shadow-sm">
          <div className="card-header text-white fw-bold" style={{ background: GENE_COLORS[g.gene] || '#6c757d' }}>
            {g.gene} — {g.protein}
            <span className="ms-3 badge bg-white text-dark">{g.locus}</span>
            <span className="ms-2 badge bg-white text-dark">{g.aa}</span>
            <span className="ms-2 badge bg-white text-dark">OMIM {g.omim_disease}</span>
          </div>
          <div className="card-body">
            <div className="row">
              <div className="col-md-7">
                <p className="mb-2"><strong>Phenotype:</strong> <small>{g.phenotype}</small></p>
                <p className="mb-2"><strong>Inheritance:</strong> <small>{g.inheritance}</small></p>
                <p className="mb-2"><strong>Hallmark:</strong> <small className="text-danger fw-semibold">{g.hallmark}</small></p>
                <p className="mb-2"><strong>Key DDx:</strong> <small>{g.key_ddx}</small></p>
                <p className="mb-2"><strong>Treatment Alert:</strong> <small className="text-warning fw-semibold">{g.treatment_alert}</small></p>
              </div>
              <div className="col-md-5">
                <p className="mb-1"><strong>Liver Biochem:</strong> <small>{g.liver_biochem}</small></p>
                <p className="mb-1"><strong>GGT Pattern:</strong> <small>{g.ggt_pattern}</small></p>
                <p className="mb-1"><strong>Primary Complication:</strong> <small>{g.primary_complication}</small></p>
                <div className="mt-2">
                  <BarRow label="Liver Disease"     pct={g.liver_disease_pct}        color={GENE_COLORS[g.gene]} />
                  <BarRow label="Neuro Involvement" pct={g.neuro_involvement_pct}    color={GENE_COLORS[g.gene]} />
                  <BarRow label="Transplant"        pct={g.transplant_pct}           color={GENE_COLORS[g.gene]} />
                  <BarRow label="Delayed Dx"        pct={g.diagnosis_delayed_pct}    color="#e65100" />
                </div>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Clinical Atlas tab ───────────────────────────────────────────────── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || [];

  return (
    <div>
      <h5 className="mb-3">Clinical Atlas — Full Disease Descriptions</h5>
      {genes.map(g => (
        <div key={g.gene} className="mb-4">
          <div className="d-flex align-items-center mb-2">
            <span className="badge fs-5 me-2" style={{ background: GENE_COLORS[g.gene] || '#6c757d' }}>{g.gene}</span>
            <span className="fw-bold">{g.protein}</span>
            <span className="ms-2 text-muted small">— {g.locus} · {g.aa} · {g.gene_class}</span>
          </div>
          <div className="card border-0 shadow-sm">
            <div className="card-body">
              <div className="alert alert-light border-start border-4 mb-3" style={{ borderColor: GENE_COLORS[g.gene] + ' !important' }}>
                <small>{g.disease}</small>
              </div>
              <div className="row">
                <div className="col-md-6">
                  <p className="mb-1"><strong>🔴 Hallmark:</strong> <small>{g.hallmark}</small></p>
                  <p className="mb-1"><strong>⚠️ DDx:</strong> <small>{g.key_ddx}</small></p>
                </div>
                <div className="col-md-6">
                  <p className="mb-1"><strong>💊 Treatment Alert:</strong> <small className="text-danger">{g.treatment_alert}</small></p>
                  <p className="mb-1"><strong>🧬 Inheritance:</strong> <small>{g.inheritance}</small></p>
                </div>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Definitions tab ──────────────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h5 className="mb-3">Hepatic Atlas — Clinical Term Definitions</h5>
      {data.map((d, i) => (
        <div key={i} className="card mb-3 border-0 shadow-sm">
          <div className="card-header fw-semibold" style={{ background: '#fff8e1' }}>
            {d.term}
          </div>
          <div className="card-body">
            <small className="text-muted">{d.definition}</small>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Main Page ────────────────────────────────────────────────────────── */
export default function HepaticAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hepatic-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hepatic-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hepatic-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => {
        setOverview(ov);
        setBreakdown(bk);
        setDefinitions(df);
      })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center mb-3">
        <span style={{ fontSize: 32 }}>🫀</span>
        <div className="ms-3">
          <h2 className="mb-0">Hepatic-Atlas</h2>
          <p className="text-muted mb-0 small">
            ATP7B · HFE · SERPINA1 · ABCB11 · ATP8B1 · ABCB4 · JAG1 · SLC25A13 — 320 patients (8×40, seeds 1150–1157)
          </p>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active fw-semibold' : ''}`}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'       && <OverviewTab      data={overview}     />}
      {tab === 'Gene Table'     && <GeneTableTab     data={breakdown}    />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown}    />}
      {tab === 'Definitions'    && <DefinitionsTab   data={definitions}  />}
    </div>
  );
}
