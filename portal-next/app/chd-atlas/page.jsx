'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'NKX2-5': '#1565c0',  // deep blue — cardiac TF master
  GATA4:    '#2e7d32',  // deep green — GATA4/TBX5 interaction
  TBX5:     '#6a1b9a',  // deep violet — Holt-Oram, radial ray
  TBX20:    '#00695c',  // teal — late DCM
  GATA6:    '#e65100',  // deep orange — pancreatic agenesis + TOF
  JAG1:     '#b71c1c',  // deep red — Alagille, Notch ligand
  NOTCH1:   '#37474f',  // dark slate — BAV, calcific valve
  MYH6:     '#4e342e',  // dark brown — sick sinus, alpha-MHC
};

const GENE_DISEASE = {
  'NKX2-5': 'CHD + AV Block (Progressive)',
  GATA4:    'ASD / VSD / AVSD',
  TBX5:     'Holt-Oram (Radial Ray + ASD)',
  TBX20:    'ASD + MVP + Late DCM',
  GATA6:    'TOF + Pancreatic Agenesis',
  JAG1:     'Alagille (Peripheral PS + Cholestasis)',
  NOTCH1:   'BAV + Calcific Aortic Valve Disease',
  MYH6:     'ASD3 + Sick Sinus Syndrome',
};

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Congenital Heart Disease atlas…</p>
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
          <div className="fw-bold fs-3" style={{ color: color || '#1565c0' }}>{value}</div>
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
        <div className="progress-bar" style={{ width: `${p}%`, backgroundColor: color || '#1565c0' }} />
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
      <div className="rounded-3 p-4 mb-4 text-white" style={{ background: 'linear-gradient(135deg,#1565c0,#b71c1c)' }}>
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
        <KPI label="ASD %" value={`${agg.asd_pct || 0}%`} color="#1565c0" />
        <KPI label="TOF %" value={`${agg.tof_pct || 0}%`} color="#e65100" />
        <KPI label="AV Block %" value={`${agg.av_block_pct || 0}%`} color="#6a1b9a" />
        <KPI label="Pacemaker %" value={`${agg.pacemaker_pct || 0}%`} color="#37474f" />
        <KPI label="Peripheral PS %" value={`${agg.pps_pct || 0}%`} color="#b71c1c" />
        <KPI label="Cholestasis %" value={`${agg.cholestasis_pct || 0}%`} color="#2e7d32" />
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
              <h6 className="fw-bold mb-3">Structural CHD Distribution</h6>
              <BarRow label="ASD (NKX2-5 · GATA4 · TBX5 · MYH6)" pct={agg.asd_pct} color="#1565c0" />
              <BarRow label="TOF (GATA6 · JAG1)" pct={agg.tof_pct} color="#e65100" />
              <BarRow label="BAV / Calcific AVD (NOTCH1)" pct={agg.bav_pct} color="#37474f" />
              <BarRow label="Peripheral PS (JAG1 Alagille)" pct={agg.pps_pct} color="#b71c1c" />
              <BarRow label="VSD (GATA4 · NKX2-5)" pct={agg.vsd_pct} color="#2e7d32" />
              <BarRow label="AVSD (GATA4 · GATA6)" pct={agg.avsd_pct} color="#6a1b9a" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Conduction + Extracardiac Features</h6>
              <BarRow label="AV Block (NKX2-5 · TBX5)" pct={agg.av_block_pct} color="#6a1b9a" />
              <BarRow label="Sick Sinus (MYH6)" pct={agg.sick_sinus_pct} color="#4e342e" />
              <BarRow label="Pacemaker Implanted" pct={agg.pacemaker_pct} color="#37474f" />
              <BarRow label="Radial Ray Anomaly (TBX5)" pct={agg.radial_ray_pct} color="#6a1b9a" />
              <BarRow label="Cholestasis (JAG1)" pct={agg.cholestasis_pct} color="#2e7d32" />
              <BarRow label="Pancreatic Agenesis (GATA6)" pct={agg.pancreatic_agenesis_pct} color="#e65100" />
              <BarRow label="Butterfly Vertebrae (JAG1)" pct={agg.butterfly_vertebrae_pct} color="#b71c1c" />
              <BarRow label="Post-Repair Arrhythmia" pct={agg.post_repair_arrhythmia_pct} color="#4e342e" />
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
            <h5 className="fw-bold mb-3">Clinical Pearls — CHD Gene Hierarchy</h5>
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
              <th>Structural Defect</th>
              <th>Conduction</th>
              <th>ASD %</th>
              <th>AV Block %</th>
              <th>Pacemaker %</th>
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
                  <td><span className="badge bg-secondary text-wrap">{(g.inheritance || '').split(';')[0]}</span></td>
                  <td className="small" style={{ maxWidth: 160 }}>{g.structural_defect}</td>
                  <td className="small" style={{ maxWidth: 120 }}>
                    <span className={`badge ${(g.conduction_defect || '').includes('AV block') ? 'bg-danger' : (g.conduction_defect || '').includes('sinus') ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                      {(g.conduction_defect || 'None').split(' ')[0]}
                    </span>
                  </td>
                  <td className="fw-bold" style={{ color: '#1565c0' }}>{s.asd_pct}%</td>
                  <td className="fw-bold" style={{ color: '#6a1b9a' }}>{s.av_block_pct}%</td>
                  <td className="fw-bold" style={{ color: '#37474f' }}>{s.pacemaker_pct}%</td>
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
          <div className="card-header text-white fw-bold" style={{ backgroundColor: GENE_COLORS[selected.gene] || '#1565c0' }}>
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

                <div className="row g-2 mt-2">
                  <div className="col-12">
                    <div className="bg-light rounded p-2 small">
                      <strong>Structural Defect:</strong> {selected.structural_defect}
                    </div>
                  </div>
                  <div className="col-12">
                    <div className="bg-light rounded p-2 small">
                      <strong>Conduction Defect:</strong> {selected.conduction_defect}
                    </div>
                  </div>
                  <div className="col-12">
                    <div className="rounded p-2 small text-white"
                      style={{ backgroundColor: selected.extracardiac && selected.extracardiac !== 'NONE — isolated cardiac phenotype' ? '#c62828' : '#546e7a' }}>
                      <strong>Extracardiac:</strong> {selected.extracardiac}
                    </div>
                  </div>
                </div>
              </div>
              <div className="col-md-6">
                <h6 className="fw-bold text-danger">Hallmark / Red Flag</h6>
                <p className="small">{selected.hallmark}</p>

                <h6 className="fw-bold text-primary mt-3">Treatment Alert</h6>
                <p className="small">{selected.treatment_alert}</p>

                <h6 className="fw-bold text-muted mt-3">Differential Diagnosis</h6>
                <p className="small">{selected.key_ddx}</p>

                {/* Mini stats */}
                {selected.stats && (
                  <div className="mt-3">
                    <h6 className="fw-bold text-muted">Cohort Stats ({selected.cohort_n} patients)</h6>
                    <div className="row g-2 text-center">
                      {[
                        ['ASD', `${selected.stats.asd_pct ?? 0}%`, '#1565c0'],
                        ['VSD', `${selected.stats.vsd_pct ?? 0}%`, '#2e7d32'],
                        ['TOF', `${selected.stats.tof_pct ?? 0}%`, '#e65100'],
                        ['AV Blk', `${selected.stats.av_block_pct ?? 0}%`, '#6a1b9a'],
                        ['Sick Sn', `${selected.stats.sick_sinus_pct ?? 0}%`, '#4e342e'],
                        ['PPM', `${selected.stats.pacemaker_pct ?? 0}%`, '#37474f'],
                      ].map(([l, v, c]) => (
                        <div key={l} className="col-4">
                          <div className="border rounded p-1">
                            <div className="fw-bold small" style={{ color: c }}>{v}</div>
                            <div className="text-muted" style={{ fontSize: 10 }}>{l}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="row g-2 text-center mt-1">
                      {[
                        ['PPS', `${selected.stats.pps_pct ?? 0}%`, '#b71c1c'],
                        ['BAV', `${selected.stats.bav_pct ?? 0}%`, '#37474f'],
                        ['Rad Ray', `${selected.stats.radial_ray_pct ?? 0}%`, '#6a1b9a'],
                        ['PancAg', `${selected.stats.pancreatic_agenesis_pct ?? 0}%`, '#e65100'],
                        ['Cholest', `${selected.stats.cholestasis_pct ?? 0}%`, '#2e7d32'],
                        ['CalcAVD', `${selected.stats.calcific_avd_pct ?? 0}%`, '#37474f'],
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
          <div style={{ fontSize: 48 }}>&#x2764;&#xfe0f;</div>
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
      <h5 className="fw-bold mb-3">Clinical Definitions — Congenital Heart Disease Genetics</h5>
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

export default function CHDAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/chd-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/chd-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/chd-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4 px-3 px-md-4">
      <h1 className="fw-bold mb-1" style={{ color: '#1565c0' }}>
        &#x2764;&#xfe0f; Congenital Heart Disease (CHD) Atlas
      </h1>
      <p className="text-muted mb-3">
        Complete 8-Gene Hereditary Structural CHD Reference —
        NKX2-5 · GATA4 · TBX5 · TBX20 · GATA6 · JAG1 · NOTCH1 · MYH6
        (320 patients, seeds 1270–1277)
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
