'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  G6PD:    '#b71c1c',  // deep red — X-linked G6PD deficiency, most common enzymopathy
  PKLR:    '#e65100',  // burnt orange — pyruvate kinase deficiency, 2,3-BPG paradox
  ANK1:    '#1565c0',  // deep blue — ankyrin-1, most common hereditary spherocytosis
  SPTA1:   '#4a148c',  // deep purple — alpha-spectrin, HE/HPP, αLELY modifier
  SLC4A1:  '#006064',  // dark teal — band 3, HS4 + SAO + dRTA
  EPB42:   '#1b5e20',  // deep green — protein 4.2, AR spherocytosis, Japanese founder
  HK1:     '#37474f',  // dark slate — hexokinase, severe non-spherocytic HA
  PIEZO1:  '#880e4f',  // deep maroon — xerocytosis, splenectomy CONTRAINDICATED
};

const GENE_DISEASE = {
  G6PD:    'G6PD Deficiency (X-Linked) — >400M affected; rasburicase/primaquine CI; test 3 months post-crisis',
  PKLR:    'PK Deficiency (AR) — 2,3-BPG paradox tolerate low Hb; mitapivat FDA 2022; iron overload without transfusion',
  ANK1:    'Hereditary Spherocytosis 1 (AD) — most common HS 40–65%; EMA flow cytometry; splenectomy curative',
  SPTA1:   'Hereditary Elliptocytosis/HPP (AD/AR) — αLELY modifier converts mild HE to severe HPP; genotype always',
  SLC4A1:  'HS4 + SAO + dRTA (AD/AR) — band 3; SAO Δ400–408 malaria protection; homozygous SAO lethal',
  EPB42:   'Hereditary Spherocytosis 5 (AR) — protein 4.2; AR unlike most HS; Japanese founder p.Ala142Thr',
  HK1:     'Hexokinase Deficiency (AR) — non-spherocytic HA; EMA/osmotic fragility NORMAL; severe neonatal',
  PIEZO1:  'Xerocytosis/DHS (AD GOF) — mechanosensory channel; MCHC elevated; splenectomy ABSOLUTELY CI',
};

const ENZYME_GENES    = ['G6PD', 'PKLR', 'HK1'];
const MEMBRANE_GENES  = ['ANK1', 'SPTA1', 'SLC4A1', 'EPB42'];
const CHANNEL_GENES   = ['PIEZO1'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Red Cell Disorders Atlas…</p>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger m-4"><strong>Error:</strong> {msg}</div>;
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-sm-4 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-body text-center p-2" style={{ borderTop: `4px solid ${color}` }}>
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function AlertBadge({ text, color = '#37474f' }) {
  return (
    <span className="badge me-1 mb-1" style={{ background: color, fontSize: '0.7rem' }}>
      {text}
    </span>
  );
}

/* ── OVERVIEW TAB ── */
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats;

  const statItems = [
    { key: 'haemolytic_anaemia_any',           label: 'Haemolytic Anaemia (any)',             color: '#b71c1c' },
    { key: 'splenomegaly_any',                  label: 'Splenomegaly',                         color: '#1565c0' },
    { key: 'gallstones',                        label: 'Gallstones',                           color: '#e65100' },
    { key: 'neonatal_jaundice_any',             label: 'Neonatal Jaundice',                    color: '#880e4f' },
    { key: 'splenectomy_performed',             label: 'Splenectomy Performed',                color: '#4a148c' },
    { key: 'elevated_reticulocytes',            label: 'Elevated Reticulocytes',               color: '#006064' },
    { key: 'aplastic_crisis_parvovirus',        label: 'Aplastic Crisis (Parvovirus B19)',     color: '#b71c1c' },
    { key: 'iron_overload_elevated_ferritin',   label: 'Iron Overload (elevated ferritin)',    color: '#e65100' },
    { key: 'folate_supplementation',            label: 'On Folate Supplementation',            color: '#1b5e20' },
    { key: 'g6pd_haemoglobinuria',             label: 'Haemoglobinuria (G6PD crisis)',        color: '#b71c1c' },
    { key: 'piezo1_pseudohyperkalaemia',        label: 'Pseudohyperkalaemia (PIEZO1)',         color: '#880e4f' },
    { key: 'piezo1_elevated_mchc',              label: 'Elevated MCHC (xerocytosis)',          color: '#880e4f' },
  ];

  const alertColors = {
    G6PD: '#b71c1c', PKLR: '#e65100', ANK1: '#1565c0',
    SPTA1: '#4a148c', SLC4A1: '#006064', EPB42: '#1b5e20',
    HK1: '#37474f', PIEZO1: '#880e4f',
  };

  return (
    <div>
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={data.total_patients} color="#1565c0" />
        <KPI label="Genes Covered" value={data.genes?.length ?? 8} color="#e65100" />
        <KPI label="Cohort Seeds" value={data.seed_range} color="#4a148c" />
        <KPI label="Patients/Gene" value="40" color="#1b5e20" />
        <KPI label="Enzyme Deficiencies" value="3" color="#b71c1c" />
        <KPI label="Membrane Disorders" value="4" color="#1565c0" />
      </div>

      <h5 className="mb-3">Aggregate Clinical Statistics (320 patients)</h5>
      <div className="row g-2 mb-4">
        {statItems.map(({ key, label, color }) => (
          <KPI key={key} label={label} value={`${s[key] ?? '—'}%`} color={color} />
        ))}
      </div>

      <h5 className="mb-2">Gene Categories</h5>
      <div className="row mb-4">
        <div className="col-md-4 mb-2">
          <div className="card border-0 shadow-sm">
            <div className="card-body p-3" style={{ borderLeft: '4px solid #b71c1c' }}>
              <div className="fw-bold mb-1" style={{ color: '#b71c1c' }}>Glycolytic Enzyme Deficiencies</div>
              {ENZYME_GENES.map(g => (
                <div key={g} className="small text-muted">{g}: {GENE_DISEASE[g]?.split('—')[0].trim()}</div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-4 mb-2">
          <div className="card border-0 shadow-sm">
            <div className="card-body p-3" style={{ borderLeft: '4px solid #1565c0' }}>
              <div className="fw-bold mb-1" style={{ color: '#1565c0' }}>Red Cell Membrane Disorders</div>
              {MEMBRANE_GENES.map(g => (
                <div key={g} className="small text-muted">{g}: {GENE_DISEASE[g]?.split('—')[0].trim()}</div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-4 mb-2">
          <div className="card border-0 shadow-sm">
            <div className="card-body p-3" style={{ borderLeft: '4px solid #880e4f' }}>
              <div className="fw-bold mb-1" style={{ color: '#880e4f' }}>Ion Channel / Volume Disorders</div>
              {CHANNEL_GENES.map(g => (
                <div key={g} className="small text-muted">{g}: {GENE_DISEASE[g]?.split('—')[0].trim()}</div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <h5 className="mb-2">Critical Alerts</h5>
      <div>
        {(data.top_alerts || []).map((a, i) => {
          const gene = Object.keys(alertColors).find(g => a.startsWith(g + '-') || a.startsWith(g + ' '));
          const color = alertColors[gene] || '#37474f';
          return <AlertBadge key={i} text={a} color={color} />;
        })}
      </div>
    </div>
  );
}

/* ── GENE TABLE TAB ── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);

  return (
    <div className="table-responsive">
      <table className="table table-sm table-hover align-middle">
        <thead className="table-dark">
          <tr>
            <th>Gene</th>
            <th>Protein</th>
            <th>Locus</th>
            <th>OMIM</th>
            <th>Inheritance</th>
            <th>Organ System</th>
            <th>N</th>
          </tr>
        </thead>
        <tbody>
          {genes.map(g => (
            <tr key={g.gene}>
              <td>
                <span className="badge" style={{ background: GENE_COLORS[g.gene] || '#555' }}>
                  {g.gene}
                </span>
              </td>
              <td className="small">{g.protein?.split('(')[0].trim()}</td>
              <td className="small font-monospace">{g.locus}</td>
              <td className="small">
                <a href={`https://omim.org/entry/${g.omim_disease}`} target="_blank" rel="noreferrer"
                   className="text-decoration-none">#{g.omim_disease}</a>
              </td>
              <td className="small">{g.inheritance?.split('—')[0].trim()}</td>
              <td className="small">{g.organ_system}</td>
              <td className="small text-center">{g.n_patients}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ── */
function ClinicalAtlasTab({ data }) {
  const [selected, setSelected] = useState(null);
  if (!data) return <Loading />;
  const genes = Object.keys(data);
  const gene = selected || genes[0];
  const g = data[gene];
  if (!g) return <Loading />;

  return (
    <div>
      <div className="d-flex flex-wrap gap-2 mb-3">
        {genes.map(gn => (
          <button key={gn} className="btn btn-sm" onClick={() => setSelected(gn)}
            style={{
              background: gene === gn ? GENE_COLORS[gn] : '#f8f9fa',
              color: gene === gn ? '#fff' : '#333',
              border: `2px solid ${GENE_COLORS[gn] || '#aaa'}`,
            }}>
            {gn}
          </button>
        ))}
      </div>

      <div className="card border-0 shadow-sm mb-3">
        <div className="card-body" style={{ borderLeft: `6px solid ${GENE_COLORS[gene] || '#555'}` }}>
          <h5 style={{ color: GENE_COLORS[gene] }}>{g.gene} — {g.protein?.split('(')[0].trim()}</h5>
          <div className="small text-muted mb-2">
            {g.locus} · {g.aa} · OMIM gene #{g.omim_gene} / disease #{g.omim_disease} · {g.inheritance}
          </div>
          <p className="small mb-0">{g.gene_class}</p>
        </div>
      </div>

      <div className="row g-3 mb-3">
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: GENE_COLORS[gene], color: '#fff' }}>
              Clinical Hallmarks
            </div>
            <ul className="list-group list-group-flush">
              {(g.hallmarks || []).map((h, i) => (
                <li key={i} className="list-group-item small py-1">{h}</li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: '#b71c1c', color: '#fff' }}>
              Treatment Alerts
            </div>
            <ul className="list-group list-group-flush">
              {(g.treatment_alerts || []).map((a, i) => (
                <li key={i} className="list-group-item small py-1">{a}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div className="card border-0 shadow-sm mb-3">
        <div className="card-header fw-bold small">Primary Treatment</div>
        <div className="card-body small">{g.primary_treatment}</div>
      </div>

      <div className="row g-3">
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header fw-bold small">Etiology Distribution</div>
            <ul className="list-group list-group-flush">
              {(g.etiology_distribution || []).map((e, i) => (
                <li key={i} className="list-group-item small py-1 d-flex justify-content-between">
                  <span>{e.etiology}</span>
                  <span className="badge bg-secondary ms-2">{Math.round(e.fraction * 100)}%</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header fw-bold small">Simulated Cohort Stats (n={g.n_patients})</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm mb-0">
                  <tbody>
                    {Object.entries(g.stats || {}).slice(0, 12).map(([k, v]) => (
                      <tr key={k}>
                        <td className="small ps-2">{k.replace(/_/g, ' ')}</td>
                        <td className="small text-end pe-2">{v}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── DEFINITIONS TAB ── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;

  return (
    <div>
      {data.classification && (
        <div className="mb-4">
          <h5>Disease Classification</h5>
          {Object.entries(data.classification).map(([category, entries]) => (
            <div key={category} className="card border-0 shadow-sm mb-2">
              <div className="card-header fw-bold small text-capitalize">
                {category.replace(/_/g, ' ')}
              </div>
              <ul className="list-group list-group-flush">
                {Object.entries(entries).map(([k, v]) => (
                  <li key={k} className="list-group-item small py-1">
                    <span className="fw-semibold">{k.replace(/_/g, ' ')}: </span>{v}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}

      {data.key_diagnostic_rules && (
        <div className="mb-4">
          <h5>Key Diagnostic Rules</h5>
          {Object.entries(data.key_diagnostic_rules).map(([rule, text]) => (
            <div key={rule} className="card border-0 shadow-sm mb-2">
              <div className="card-header fw-bold small" style={{ background: '#37474f', color: '#fff' }}>
                {rule.replace(/_/g, ' ')}
              </div>
              <div className="card-body small">{text}</div>
            </div>
          ))}
        </div>
      )}

      {data.treatment_hierarchy && (
        <div className="mb-4">
          <h5>Treatment Hierarchies</h5>
          {Object.entries(data.treatment_hierarchy).map(([gene, steps]) => (
            <div key={gene} className="card border-0 shadow-sm mb-2">
              <div className="card-header fw-bold small"
                   style={{ background: GENE_COLORS[gene.split('_')[0]] || '#555', color: '#fff' }}>
                {gene.replace(/_/g, ' ')}
              </div>
              <ol className="list-group list-group-flush list-group-numbered">
                {steps.map((s, i) => (
                  <li key={i} className="list-group-item small py-1">{s}</li>
                ))}
              </ol>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── MAIN PAGE ── */
export default function RedCellDisordersAtlasPage() {
  const [tab, setTab]         = useState('Overview');
  const [overview, setOv]     = useState(null);
  const [breakdown, setBd]    = useState(null);
  const [definitions, setDf]  = useState(null);
  const [error, setError]     = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/red-cell-disorders-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/red-cell-disorders-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/red-cell-disorders-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => { setOv(ov); setBd(bd); setDf(df); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div>
          <h2 className="mb-0 fw-bold" style={{ color: '#b71c1c' }}>
            🩸 Red-Cell-Disorders-Atlas
          </h2>
          <div className="text-muted small">
            Complete 8-Gene Hereditary Haemolytic Anaemia Atlas —
            Enzyme Deficiencies (G6PD · PKLR · HK1) + Membrane Disorders (ANK1 · SPTA1 · SLC4A1 · EPB42) + Ion Channel (PIEZO1)
            · 320 patients · seeds 1422–1429
          </div>
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
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
