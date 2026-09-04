'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#4a0072';  // deep purple — lysosomal/MPS
const LIGHT  = '#f3e5f5';
const COLOR2 = '#b71c1c';  // fatal / critical CI
const COLOR3 = '#0d47a1';  // ERT available / biomarker
const COLOR4 = '#e65100';  // warning / no ERT
const COLOR5 = '#1b5e20';  // treatable / HSCT
const COLOR6 = '#37474f';  // gene class / struct
const COLOR7 = '#006064';  // Sanfilippo / gene therapy

const SUBGROUP_COLORS = {
  'Heparan/dermatan sulfate degradation (IDUA · IDS · SGSH · NAGLU · HGSNAT)': '#0d47a1',
  'Keratan sulfate / chondroitin-6-S degradation (GALNS)':                      '#e65100',
  'Dermatan sulfate degradation (ARSB)':                                         '#1b5e20',
  'Multi-GAG degradation DS+HS+CS (GUSB)':                                      '#b71c1c',
};

const GENE_COLORS = {
  IDUA:   '#0d47a1',
  IDS:    '#1565c0',
  SGSH:   '#006064',
  NAGLU:  '#00838f',
  GALNS:  '#e65100',
  ARSB:   '#2e7d32',
  GUSB:   '#b71c1c',
  HGSNAT: '#4a148c',
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

function BarRow({ label, pct, color = COLOR }) {
  return (
    <div className="mb-1">
      <div className="d-flex justify-content-between mb-0" style={{ fontSize: '0.78rem' }}>
        <span>{label}</span><span className="fw-semibold">{pct}%</span>
      </div>
      <div className="progress" style={{ height: '7px' }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

export default function MPSAtlasPage() {
  const [tab, setTab]             = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [err, setErr]             = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mps-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/mps-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mps-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return (
    <div className="d-flex align-items-center justify-content-center" style={{ minHeight: '60vh' }}>
      <div className="spinner-border" style={{ color: COLOR }} role="status" />
      <span className="ms-3 text-muted">Loading MPS-Atlas…</span>
    </div>
  );
  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;

  const genes = breakdown?.genes || [];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="rounded-3 px-4 py-3 mb-3 text-white"
           style={{ background: `linear-gradient(135deg, ${COLOR} 0%, #6a1b9a 100%)` }}>
        <h3 className="mb-1 fw-bold">&#x1f9ec; MPS-Atlas</h3>
        <div style={{ fontSize: '0.85rem', opacity: 0.9 }}>
          Complete 8-Gene Mucopolysaccharidoses Atlas &nbsp;·&nbsp;
          IDUA · IDS · SGSH · NAGLU · GALNS · ARSB · GUSB · HGSNAT &nbsp;·&nbsp;
          320-patient aggregate cohort (8 × 40, seeds 912–919)
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active fw-semibold' : ''}`}
              style={tab === t ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(t)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <>
          <div className="row g-2 mb-3">
            <KPI label="Genes" value={overview.n_genes} color={COLOR} />
            <KPI label="Patients" value={overview.n_patients} color={COLOR} />
            <KPI label="Subgroups" value={Object.keys(overview.gene_subgroups || {}).length} color={COLOR3} />
            <KPI label="Seeds" value={`${overview.seeds?.[0]}–${overview.seeds?.[overview.seeds.length-1]}`} color={COLOR6} />
            <KPI label="ERT Available" value="5/8 genes" color={COLOR5} />
            <KPI label="No Approved ERT" value="3/8 genes" color={COLOR4} />
          </div>

          {/* Subgroups */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ backgroundColor: LIGHT, color: COLOR }}>
              MPS Gene Subgroups — GAG Pathway
            </div>
            <div className="card-body">
              <div className="row">
                {Object.entries(overview.gene_subgroups || {}).map(([group, genes_list]) => (
                  <div key={group} className="col-12 col-md-6 mb-2">
                    <div className="p-2 rounded" style={{ backgroundColor: '#f8f9fa', borderLeft: `4px solid ${SUBGROUP_COLORS[group] || COLOR}` }}>
                      <div className="fw-semibold small" style={{ color: SUBGROUP_COLORS[group] || COLOR }}>{group}</div>
                      <div className="text-muted small">{genes_list.join(' · ')}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Critical Rules */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ backgroundColor: '#fce4ec', color: COLOR2 }}>
              ⚠️ Critical Clinical Rules (MPS)
            </div>
            <div className="card-body">
              <ul className="mb-0" style={{ fontSize: '0.85rem' }}>
                {(overview.critical_clinical_rules || []).map((rule, i) => (
                  <li key={i} className="mb-2">{rule}</li>
                ))}
              </ul>
            </div>
          </div>

          {/* Gene summary bars */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ backgroundColor: LIGHT, color: COLOR }}>
              Gene Summary — Mean Age at Diagnosis
            </div>
            <div className="card-body">
              {(overview.gene_summary || []).map(g => (
                <div key={g.gene} className="mb-2">
                  <div className="d-flex justify-content-between" style={{ fontSize: '0.8rem' }}>
                    <span className="fw-semibold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</span>
                    <span className="text-muted">{g.mean_age_dx_y} yr dx</span>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: '#555', marginBottom: '2px' }}>{g.phenotype}</div>
                  <div className="progress" style={{ height: '6px' }}>
                    <div className="progress-bar"
                         style={{ width: `${Math.min(100, g.mean_age_dx_y * 12)}%`, backgroundColor: GENE_COLORS[g.gene] || COLOR }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* NBS note */}
          <div className="alert alert-secondary small mb-0">
            <strong>NBS:</strong> {overview.nbs_note}
          </div>
        </>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && (
        <div className="table-responsive">
          <table className="table table-bordered table-sm align-middle" style={{ fontSize: '0.78rem' }}>
            <thead style={{ backgroundColor: LIGHT }}>
              <tr>
                <th style={{ color: COLOR }}>Gene</th>
                <th>MPS Type</th>
                <th>Locus</th>
                <th>Size</th>
                <th>GAG</th>
                <th>Pts</th>
                <th>Dx (yr)</th>
                <th>Normal IQ %</th>
                <th>Corneal %</th>
                <th>HSM %</th>
                <th>AAI %</th>
                <th>Hydrops %</th>
                <th>ERT</th>
              </tr>
            </thead>
            <tbody>
              {genes.map(g => (
                <tr key={g.gene}>
                  <td className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</td>
                  <td style={{ maxWidth: '160px' }}>{g.alias.split('·')[1]?.trim() || g.alias.split('—')[1]?.trim() || ''}</td>
                  <td>{g.locus}</td>
                  <td>{g.aa} / {g.kDa}</td>
                  <td style={{ maxWidth: '120px', fontSize: '0.7rem' }}>{g.mps_subgroup.split('(')[0].trim()}</td>
                  <td>{g.n_patients}</td>
                  <td>{g.mean_age_dx_y}</td>
                  <td>
                    <span className={`badge ${g.pct_normal_iq >= 80 ? 'bg-success' : g.pct_normal_iq >= 30 ? 'bg-warning text-dark' : 'bg-danger'}`}>
                      {g.pct_normal_iq}%
                    </span>
                  </td>
                  <td>
                    <span className="badge bg-info text-dark">{g.pct_corneal_clouding}%</span>
                  </td>
                  <td>{g.pct_hepatosplenomegaly}%</td>
                  <td>
                    {g.pct_atlantoaxial_instability > 0
                      ? <span className="badge bg-danger">{g.pct_atlantoaxial_instability}%</span>
                      : <span className="text-muted">—</span>}
                  </td>
                  <td>
                    {g.pct_hydrops_fetalis > 0
                      ? <span className="badge bg-warning text-dark">{g.pct_hydrops_fetalis}%</span>
                      : <span className="text-muted">—</span>}
                  </td>
                  <td>
                    {g.diet_treatment?.includes('ERT:') && !g.diet_treatment?.includes('No approved')
                      ? <span className="badge" style={{ backgroundColor: COLOR5 }}>✓</span>
                      : <span className="badge bg-secondary">✗</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && (
        <div>
          {genes.map(g => (
            <div key={g.gene} className="card shadow-sm mb-4">
              <div className="card-header text-white fw-bold"
                   style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR }}>
                {g.gene} — {g.alias}
                <span className="float-end badge bg-light text-dark">{g.locus} · {g.aa}</span>
              </div>
              <div className="card-body" style={{ fontSize: '0.82rem' }}>
                <div className="row mb-2">
                  <div className="col-md-6">
                    <BarRow label="Normal IQ %" pct={g.pct_normal_iq} color={g.pct_normal_iq >= 80 ? COLOR5 : COLOR2} />
                    <BarRow label="Corneal Clouding %" pct={g.pct_corneal_clouding} color={COLOR3} />
                    <BarRow label="Hepatosplenomegaly %" pct={g.pct_hepatosplenomegaly} color={COLOR} />
                  </div>
                  <div className="col-md-6">
                    <BarRow label="Atlantoaxial Instability %" pct={g.pct_atlantoaxial_instability} color={COLOR2} />
                    <BarRow label="Hydrops Fetalis %" pct={g.pct_hydrops_fetalis} color={COLOR4} />
                    <BarRow label="HSCT Eligible (IDUA <2yr) %" pct={g.pct_hsct_eligible} color={COLOR5} />
                  </div>
                </div>

                <div className="row g-2 mt-1" style={{ fontSize: '0.78rem' }}>
                  <div className="col-12">
                    <strong>Phenotype:</strong> {g.phenotype}
                  </div>
                  <div className="col-12 col-md-6">
                    <strong>Severity Spectrum:</strong> {g.severity_spectrum}
                  </div>
                  <div className="col-12 col-md-6">
                    <strong>Key Biomarker:</strong> {g.key_biomarker}
                  </div>
                  <div className="col-12">
                    <strong>Hallmarks:</strong>
                    <p className="mt-1 mb-1" style={{ whiteSpace: 'pre-line' }}>{g.hallmark}</p>
                  </div>
                  <div className="col-12">
                    <strong>Treatment:</strong> {g.diet_treatment}
                  </div>
                  <div className="col-12">
                    <strong style={{ color: COLOR2 }}>Critical CI:</strong>{' '}
                    <span style={{ color: COLOR2 }}>{g.critical_ci}</span>
                  </div>
                  <div className="col-12">
                    <strong>Gene Therapy Status:</strong> {g.gene_therapy_status}
                  </div>
                  <div className="col-12">
                    <strong>Key Variants:</strong>{' '}
                    {(g.key_variants || []).map((v, i) => (
                      <span key={i} className="badge me-1" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR, fontSize: '0.7rem' }}>{v}</span>
                    ))}
                  </div>
                  <div className="col-12">
                    <strong>Founder Variant:</strong> {g.founder_variant}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && defs && (
        <div>
          <div className="alert mb-3" style={{ backgroundColor: LIGHT, color: COLOR, border: `1px solid ${COLOR}` }}>
            <strong>{defs.mps_overview?.full_name}</strong> &nbsp;|&nbsp;
            {defs.mps_overview?.genes_in_atlas} genes &nbsp;|&nbsp;
            Incidence: {defs.mps_overview?.collective_incidence} &nbsp;|&nbsp;
            {defs.mps_overview?.nbs_note}
          </div>
          {(defs.definitions || []).map((d, i) => (
            <div key={i} className="card shadow-sm mb-2">
              <div className="card-header fw-semibold" style={{ backgroundColor: LIGHT, color: COLOR, fontSize: '0.85rem' }}>
                {d.term}
              </div>
              <div className="card-body py-2" style={{ fontSize: '0.82rem' }}>
                {d.definition}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
