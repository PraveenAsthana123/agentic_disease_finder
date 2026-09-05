'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  CYP21A2: '#1a237e',  // deep navy — most common CAH
  CYP11B1: '#880e4f',  // deep rose — hypertension-CAH
  CYP11B2: '#006064',  // deep teal — isolated mineralocorticoid deficiency
  CYP17A1: '#4a148c',  // deep violet — HTN + sexual infantilism
  STAR:    '#e65100',  // deep orange — most severe, all steroids absent
  NR0B1:   '#1b5e20',  // deep green — X-linked AHC + HH
  MC2R:    '#37474f',  // dark slate — isolated GC deficiency
  AAAS:    '#b71c1c',  // deep red — Triple-A/Allgrove
};

const GENE_DISEASE = {
  CYP21A2: 'CAH21 (21-OHase)',
  CYP11B1: 'CAH-11β (11β-OHase)',
  CYP11B2: 'CMO I/II (Aldo Synth)',
  CYP17A1: '17α-OHase/17,20-Lyase',
  STAR:    'Lipoid CAH (CLAH)',
  NR0B1:   'AHC + HH (DAX1)',
  MC2R:    'FGD1 (ACTH-R)',
  AAAS:    'Triple-A (Allgrove)',
};

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Adrenal Disorders atlas…</p>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger m-4"><strong>Error:</strong> {msg}</div>;
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm border-0">
        <div className="card-body text-center p-3">
          <div className="fw-bold fs-3" style={{ color: color || '#1a237e' }}>{value}</div>
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
        <div className="progress-bar" style={{ width: `${p}%`, backgroundColor: color || '#1a237e' }} />
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const agg = ov.aggregate_clinical || {};
  const alerts = ov.drug_alerts || [];
  const pearls = ov.clinical_pearls || [];

  return (
    <div>
      {/* Hero */}
      <div className="rounded-3 p-4 mb-4 text-white" style={{ background: 'linear-gradient(135deg,#1a237e,#006064)' }}>
        <h2 className="fw-bold">{ov.atlas_name}</h2>
        <p className="mb-1 opacity-90">{ov.atlas_subtitle}</p>
        <div className="d-flex gap-3 flex-wrap mt-2">
          <span className="badge bg-light text-dark">{ov.n_genes} Genes</span>
          <span className="badge bg-light text-dark">{ov.n_patients} Patients</span>
          <span className="badge bg-light text-dark">Seeds {ov.seeds}</span>
          <span className="badge bg-light text-dark">8 Genes: {(ov.genes || []).join(' · ')}</span>
        </div>
      </div>

      {/* KPIs */}
      <div className="row g-3 mb-4">
        <KPI label="Salt Wasting %" value={`${agg.salt_wasting_pct || 0}%`} color="#1a237e" />
        <KPI label="Hypertension %" value={`${agg.hypertension_pct || 0}%`} color="#880e4f" />
        <KPI label="Virilisation %" value={`${agg.virilisation_pct || 0}%`} color="#4a148c" />
        <KPI label="Hyperpigment %" value={`${agg.hyperpigmentation_pct || 0}%`} color="#e65100" />
        <KPI label="Hypoglycaemia %" value={`${agg.hypoglycaemia_pct || 0}%`} color="#b71c1c" />
        <KPI label="Adrenal Crisis %" value={`${agg.adrenal_crisis_pct || 0}%`} color="#37474f" />
      </div>

      {/* Description */}
      <div className="card border-0 shadow-sm mb-4">
        <div className="card-body">
          <h5 className="fw-bold mb-2">Atlas Description</h5>
          <p className="text-muted mb-0" style={{ lineHeight: 1.7 }}>{ov.description}</p>
        </div>
      </div>

      {/* Aggregate bars */}
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Steroid &amp; Clinical Features</h6>
              <BarRow label="Salt-wasting (mineralocorticoid deficit)" pct={agg.salt_wasting_pct} color="#1a237e" />
              <BarRow label="Hypertension (mineralocorticoid excess)" pct={agg.hypertension_pct} color="#880e4f" />
              <BarRow label="Virilisation in 46,XX patients" pct={agg.virilisation_pct} color="#4a148c" />
              <BarRow label="Adrenal crisis at presentation" pct={agg.adrenal_crisis_pct} color="#b71c1c" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Syndromic &amp; Associated Features</h6>
              <BarRow label="Hyperpigmentation (ACTH elevation)" pct={agg.hyperpigmentation_pct} color="#e65100" />
              <BarRow label="Hypoglycaemia" pct={agg.hypoglycaemia_pct} color="#006064" />
              <BarRow label="Alacrima (Triple-A / AAAS)" pct={agg.alacrima_pct} color="#37474f" />
              <BarRow label="Hypogonadotropic HH (NR0B1)" pct={agg.hh_pct} color="#1b5e20" />
            </div>
          </div>
        </div>
      </div>

      {/* Drug Alerts */}
      {alerts.length > 0 && (
        <div className="mb-4">
          <h5 className="fw-bold mb-3">Critical Drug &amp; Management Alerts</h5>
          {alerts.map((a, i) => (
            <div key={i} className="alert alert-warning border-warning shadow-sm mb-3">
              <div className="fw-bold mb-1">&#9888; {a.title}</div>
              <div className="small">{a.body}</div>
            </div>
          ))}
        </div>
      )}

      {/* Clinical Pearls */}
      {pearls.length > 0 && (
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-body">
            <h5 className="fw-bold mb-3">Clinical Pearls — Adrenal Disorders Hierarchy</h5>
            <ul className="mb-0 small" style={{ lineHeight: 2 }}>
              {pearls.map((p, i) => <li key={i}>{p}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  return (
    <div>
      <h5 className="fw-bold mb-3">Per-Gene Summary Table</h5>
      <div className="table-responsive">
        <table className="table table-bordered table-hover align-middle small">
          <thead className="table-dark">
            <tr>
              <th>Gene</th>
              <th>Disease</th>
              <th>Locus</th>
              <th>aa / kDa</th>
              <th>Inheritance</th>
              <th>Mineralocorticoid</th>
              <th>Glucocorticoid</th>
              <th>SW %</th>
              <th>HTN %</th>
              <th>Viril %</th>
            </tr>
          </thead>
          <tbody>
            {genes.map(g => {
              const s = g.stats || {};
              return (
                <tr key={g.gene}>
                  <td>
                    <span className="badge" style={{ backgroundColor: GENE_COLORS[g.gene] || '#555' }}>
                      {g.gene}
                    </span>
                  </td>
                  <td>{GENE_DISEASE[g.gene] || g.gene}</td>
                  <td className="text-nowrap">{g.locus}</td>
                  <td className="text-nowrap">{g.aa} / {g.kDa}</td>
                  <td><span className="badge bg-secondary text-wrap">{g.inheritance}</span></td>
                  <td className="small" style={{ maxWidth: 160 }}>{g.mineralocorticoid_status}</td>
                  <td className="small" style={{ maxWidth: 160 }}>{g.glucocorticoid_status}</td>
                  <td className="fw-bold" style={{ color: '#1a237e' }}>{s.salt_wasting_pct}%</td>
                  <td className="fw-bold" style={{ color: '#880e4f' }}>{s.hypertension_pct}%</td>
                  <td>{s.virilisation_pct}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const [selected, setSelected] = useState(null);
  const genes = Object.values(data);

  return (
    <div>
      <h5 className="fw-bold mb-3">Clinical Atlas — Select a Gene</h5>
      <div className="row g-2 mb-4">
        {genes.map(g => (
          <div key={g.gene} className="col-6 col-md-3">
            <button
              className={`btn w-100 fw-bold ${selected?.gene === g.gene ? 'text-white' : 'btn-outline-secondary'}`}
              style={selected?.gene === g.gene ? { backgroundColor: GENE_COLORS[g.gene] } : {}}
              onClick={() => setSelected(g)}
            >
              {g.gene}
              <div className="small fw-normal">{GENE_DISEASE[g.gene]}</div>
            </button>
          </div>
        ))}
      </div>

      {selected && (
        <div className="card border-0 shadow">
          <div className="card-header text-white fw-bold" style={{ backgroundColor: GENE_COLORS[selected.gene] || '#1a237e' }}>
            {selected.gene} — {selected.protein}
          </div>
          <div className="card-body">
            <div className="row g-4">
              <div className="col-md-6">
                <h6 className="fw-bold text-muted">Gene / Protein</h6>
                <p className="small">{selected.alias}</p>

                <h6 className="fw-bold text-muted mt-3">Molecular Mechanism</h6>
                <p className="small">{selected.gene_class}</p>

                <h6 className="fw-bold text-muted mt-3">Phenotype</h6>
                <p className="small">{selected.phenotype}</p>
              </div>
              <div className="col-md-6">
                <h6 className="fw-bold text-danger">Hallmark / Red Flag</h6>
                <p className="small">{selected.hallmark}</p>

                <h6 className="fw-bold text-primary mt-3">Treatment Alert</h6>
                <p className="small">{selected.treatment_alert}</p>

                <h6 className="fw-bold text-muted mt-3">Differential Diagnosis</h6>
                <p className="small">{selected.key_ddx}</p>

                <div className="row g-2 mt-2">
                  <div className="col-12">
                    <div className="bg-light rounded p-2 small">
                      <strong>Mineralocorticoid:</strong> {selected.mineralocorticoid_status}
                    </div>
                  </div>
                  <div className="col-12">
                    <div className="bg-light rounded p-2 small">
                      <strong>Glucocorticoid:</strong> {selected.glucocorticoid_status}
                    </div>
                  </div>
                  <div className="col-12">
                    <div className="bg-light rounded p-2 small">
                      <strong>Androgen:</strong> {selected.androgen_status}
                    </div>
                  </div>
                </div>

                {/* Mini stats */}
                {selected.stats && (
                  <div className="mt-3">
                    <h6 className="fw-bold text-muted">Cohort Stats ({selected.cohort_n} patients)</h6>
                    <div className="row g-2 text-center">
                      {[
                        ['SW', `${selected.stats.salt_wasting_pct}%`, '#1a237e'],
                        ['HTN', `${selected.stats.hypertension_pct}%`, '#880e4f'],
                        ['Viril', `${selected.stats.virilisation_pct}%`, '#4a148c'],
                        ['Hyperpig', `${selected.stats.hyperpigmentation_pct}%`, '#e65100'],
                        ['HypoGly', `${selected.stats.hypoglycaemia_pct}%`, '#b71c1c'],
                        ['Crisis', `${selected.stats.adrenal_crisis_pct}%`, '#37474f'],
                      ].map(([l, v, c]) => (
                        <div key={l} className="col-4">
                          <div className="border rounded p-1">
                            <div className="fw-bold small" style={{ color: c }}>{v}</div>
                            <div className="text-muted" style={{ fontSize: 10 }}>{l}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {!selected && (
        <div className="text-center text-muted py-5">
          <div style={{ fontSize: 48 }}>&#x1f9ec;</div>
          <p>Select a gene above to view its full clinical profile</p>
        </div>
      )}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const terms = data.terms || [];
  return (
    <div>
      <h5 className="fw-bold mb-3">Clinical Definitions — Adrenal Disorders</h5>
      <div className="accordion" id="defAccordion">
        {terms.map((t, i) => (
          <div key={i} className="accordion-item border-0 shadow-sm mb-2">
            <h2 className="accordion-header">
              <button
                className="accordion-button collapsed fw-bold"
                type="button"
                data-bs-toggle="collapse"
                data-bs-target={`#def${i}`}
              >
                {t.term}
              </button>
            </h2>
            <div id={`def${i}`} className="accordion-collapse collapse" data-bs-parent="#defAccordion">
              <div className="accordion-body small text-muted" style={{ lineHeight: 1.7 }}>
                {t.definition}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function AdrenalDisordersAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/adrenal-disorders-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/adrenal-disorders-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/adrenal-disorders-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4 px-3 px-md-4">
      <h1 className="fw-bold mb-1" style={{ color: '#1a237e' }}>
        &#x1f9ec; Adrenal Disorders Atlas
      </h1>
      <p className="text-muted mb-3">
        Complete 8-Gene Hereditary Adrenal &amp; Steroidogenesis Disorders Reference —
        CYP21A2 · CYP11B1 · CYP11B2 · CYP17A1 · STAR · NR0B1 · MC2R · AAAS
        (320 patients, seeds 1254–1261)
      </p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === t ? ' active fw-bold' : ''}`}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Gene Table' && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
