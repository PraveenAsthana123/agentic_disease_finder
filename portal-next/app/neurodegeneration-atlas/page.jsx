'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PSEN1:   '#1a237e',  // deep navy — FAD3, most common FAD
  PSEN2:   '#283593',  // dark indigo — FAD4, Volga German
  APP:     '#b71c1c',  // deep red — CAA / lobar ICH
  MAPT:    '#4a148c',  // deep purple — FTDP-17, PSP/CBS
  GRN:     '#1b5e20',  // deep green — FTD-TDP, progranulin
  C9orf72: '#e65100',  // deep orange — ALS/FTD spectrum
  LRRK2:   '#006064',  // deep teal — Parkinson
  SNCA:    '#37474f',  // dark slate — alpha-synuclein / Lewy body
};

const GENE_DISEASE = {
  PSEN1:   'FAD3 (Alzheimer)',
  PSEN2:   'FAD4 (Alzheimer)',
  APP:     'FAD2 / CAA',
  MAPT:    'FTDP-17 / FTD',
  GRN:     'FTD-TDP type A',
  C9orf72: 'ALS / FTD',
  LRRK2:   'PD (Parkinson)',
  SNCA:    'PD / DLB / MSA',
};

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Neurodegeneration atlas…</p>
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
      <div className="rounded-3 p-4 mb-4 text-white" style={{ background: 'linear-gradient(135deg,#1a237e,#b71c1c)' }}>
        <h2 className="fw-bold">{ov.atlas_name}</h2>
        <p className="mb-1 opacity-90">{ov.atlas_subtitle}</p>
        <div className="d-flex gap-3 flex-wrap mt-2">
          <span className="badge bg-light text-dark">{ov.n_genes} Genes</span>
          <span className="badge bg-light text-dark">{ov.n_patients} Patients</span>
          <span className="badge bg-light text-dark">Seeds {ov.seeds}</span>
          <span className="badge bg-light text-dark">8 Genes: {(ov.genes||[]).join(' · ')}</span>
        </div>
      </div>

      {/* KPIs */}
      <div className="row g-3 mb-4">
        <KPI label="Amyloid PET+" value={`${agg.amyloid_pet_positive_pct||0}%`} color="#1a237e" />
        <KPI label="Parkinsonism" value={`${agg.parkinsonism_pct||0}%`} color="#006064" />
        <KPI label="Behavioural" value={`${agg.behavioural_pct||0}%`} color="#4a148c" />
        <KPI label="Mean Onset (yr)" value={agg.mean_onset_age||'—'} color="#b71c1c" />
        <KPI label="Mean MMSE" value={agg.mean_mmse_at_diagnosis||'—'} color="#1b5e20" />
        <KPI label="Family Hx +" value={`${agg.family_history_pct||0}%`} color="#e65100" />
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
              <h6 className="fw-bold mb-3">Imaging &amp; Biomarker Features</h6>
              <BarRow label="Amyloid PET positive (PSEN1/PSEN2/APP)" pct={agg.amyloid_pet_positive_pct} color="#1a237e" />
              <BarRow label="Lobar ICH — CAA (APP-CAA dominant)" pct={agg.lobar_ich_pct} color="#b71c1c" />
              <BarRow label="Plasma PGRN low (GRN diagnostic)" pct={agg.pgrn_low_pct} color="#1b5e20" />
              <BarRow label="Seizures (PSEN1 dominant)" pct={agg.seizures_pct} color="#e65100" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Motor &amp; Behavioural Features</h6>
              <BarRow label="Parkinsonism (LRRK2/SNCA dominant)" pct={agg.parkinsonism_pct} color="#006064" />
              <BarRow label="RBD prodrome (SNCA 80%)" pct={agg.rbd_pct} color="#37474f" />
              <BarRow label="Behavioural FTD (MAPT/GRN/C9)" pct={agg.behavioural_pct} color="#4a148c" />
              <BarRow label="ALS signs (C9orf72 dominant)" pct={agg.als_signs_pct} color="#e65100" />
            </div>
          </div>
        </div>
      </div>

      {/* Drug Alerts */}
      {alerts.length > 0 && (
        <div className="mb-4">
          <h5 className="fw-bold mb-3">Critical Drug &amp; Management Alerts</h5>
          {alerts.map((a, i) => (
            <div key={i} className="alert alert-danger border-danger shadow-sm mb-3">
              <div className="fw-bold mb-1">🚨 {a.title}</div>
              <div className="small">{a.body}</div>
            </div>
          ))}
        </div>
      )}

      {/* Clinical Pearls */}
      {pearls.length > 0 && (
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-body">
            <h5 className="fw-bold mb-3">Clinical Pearls — Hereditary Neurodegeneration</h5>
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
              <th>Amyloid PET+</th>
              <th>Parkinsonism</th>
              <th>RBD %</th>
              <th>Behavioural %</th>
              <th>ALS %</th>
              <th>Mean Onset</th>
              <th>MMSE</th>
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
                  <td><span className="badge bg-secondary text-wrap" style={{fontSize:10}}>{g.inheritance}</span></td>
                  <td className="fw-bold" style={{ color: '#1a237e' }}>{s.amyloid_pet_positive_pct}%</td>
                  <td style={{ color: '#006064' }}>{s.parkinsonism_pct}%</td>
                  <td>{s.rbd_pct}%</td>
                  <td style={{ color: '#4a148c' }}>{s.behavioural_pct}%</td>
                  <td style={{ color: '#e65100' }}>{s.als_signs_pct}%</td>
                  <td>{s.mean_onset_age} yr</td>
                  <td>{s.mean_mmse_at_diagnosis}</td>
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
                      <strong>Onset:</strong> {selected.onset_pattern}
                    </div>
                  </div>
                  <div className="col-12">
                    <div className="bg-light rounded p-2 small">
                      <strong>Biomarkers:</strong> {selected.biomarker_pattern}
                    </div>
                  </div>
                  <div className="col-12">
                    <div className="bg-light rounded p-2 small">
                      <strong>Motor:</strong> {selected.motor_pattern}
                    </div>
                  </div>
                </div>

                {/* Mini stats */}
                {selected.stats && (
                  <div className="mt-3">
                    <h6 className="fw-bold text-muted">Cohort Stats ({selected.cohort_n} patients)</h6>
                    <div className="row g-2 text-center">
                      {[
                        ['Amyloid PET+', `${selected.stats.amyloid_pet_positive_pct}%`, '#1a237e'],
                        ['Parkinson', `${selected.stats.parkinsonism_pct}%`, '#006064'],
                        ['RBD', `${selected.stats.rbd_pct}%`, '#37474f'],
                        ['Behav.', `${selected.stats.behavioural_pct}%`, '#4a148c'],
                        ['ALS', `${selected.stats.als_signs_pct}%`, '#e65100'],
                        ['MMSE', selected.stats.mean_mmse_at_diagnosis, '#1b5e20'],
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
          <div style={{ fontSize: 48 }}>🧠</div>
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
      <h5 className="fw-bold mb-3">Clinical Definitions — Hereditary Neurodegeneration</h5>
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

export default function NeurodegenerationAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/neurodegeneration-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/neurodegeneration-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/neurodegeneration-atlas/definitions`).then(r => r.json()),
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
        🧠 Hereditary Neurodegeneration Atlas
      </h1>
      <p className="text-muted mb-3">
        Complete 8-Gene Adult-Onset Neurodegenerative Disease Reference —
        PSEN1 · PSEN2 · APP · MAPT · GRN · C9orf72 · LRRK2 · SNCA
        (320 patients, seeds 1238–1245)
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
