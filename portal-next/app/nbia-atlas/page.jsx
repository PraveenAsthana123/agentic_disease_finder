'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#37474f';  // iron-grey — brain iron accumulation
const LIGHT  = '#eceff1';
const COLOR2 = '#b71c1c';  // critical / eye-of-tiger PKAN
const COLOR3 = '#1565c0';  // WDR45 / X-linked dominant
const COLOR4 = '#e65100';  // warning / systemic features
const COLOR5 = '#1b5e20';  // pathway / treatable
const COLOR6 = '#4a148c';  // lysosomal / PARK
const COLOR7 = '#880e4f';  // nucleolar / WSS

const SUBGROUP_COLORS = {
  'CoA biosynthesis (PANK2 · COASY)':                  '#37474f',
  'Phospholipid remodelling (PLA2G6 · FA2H)':          '#1565c0',
  'Autophagy pathway (WDR45)':                          '#4a148c',
  'Lysosomal / PARK-NBIA (ATP13A2)':                   '#1b5e20',
  'Mitochondria-associated membrane (C19orf12)':        '#e65100',
  'Nucleolar / ubiquitin (DCAF17)':                     '#880e4f',
};

const GENE_COLORS = {
  PANK2:    '#b71c1c',
  PLA2G6:   '#1565c0',
  C19orf12: '#e65100',
  FA2H:     '#2e7d32',
  WDR45:    '#4a148c',
  COASY:    '#37474f',
  ATP13A2:  '#1b5e20',
  DCAF17:   '#880e4f',
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
        <div className="progress-bar" style={{ width: `${Math.min(100, pct)}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

export default function NBIAAtlasPage() {
  const [tab, setTab]             = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [err, setErr]             = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/nbia-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/nbia-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nbia-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return (
    <div className="d-flex align-items-center justify-content-center" style={{ minHeight: '60vh' }}>
      <div className="spinner-border" style={{ color: COLOR }} role="status" />
      <span className="ms-3 text-muted">Loading NBIA-Atlas&#x2026;</span>
    </div>
  );
  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: COLOR, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">&#x1f9e0; NBIA-Atlas — 8-Gene Neurodegeneration with Brain Iron Accumulation</h4>
        <div className="small opacity-75">
          PANK2 (PKAN) · PLA2G6 (INAD/PLAN/PARK14) · C19orf12 (MPAN) · FA2H (FAHN/SPG35) ·
          WDR45 (BPAN) · COASY (CoPAN) · ATP13A2 (Kufor-Rakeb/PARK9) · DCAF17 (Woodhouse-Sakati)
          &nbsp;·&nbsp; {overview?.n_patients} patients ({overview?.n_genes} genes × 40 each)
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === t ? ' active fw-semibold' : ''}`}
              onClick={() => setTab(t)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          {/* KPIs */}
          <div className="row mb-4">
            <KPI label="Total Genes" value={overview.n_genes} color={COLOR} />
            <KPI label="Total Patients" value={overview.n_patients} color={COLOR} />
            <KPI label="Eye-of-Tiger (PKAN)" value={overview.n_eye_of_tiger} color={COLOR2} />
            <KPI label="Iron on MRI" value={overview.n_iron_on_mri} color={COLOR} />
            <KPI label="Optic Atrophy (MPAN)" value={overview.n_optic_atrophy} color={COLOR4} />
            <KPI label="Systemic (WSS)" value={overview.n_systemic_features} color={COLOR7} />
            <KPI label="WDR45 De Novo" value={overview.n_de_novo_wdr45} color={COLOR3} />
          </div>

          {/* Subgroup breakdown */}
          <div className="card mb-4">
            <div className="card-header fw-semibold" style={{ background: LIGHT }}>
              Pathway Subgroups
            </div>
            <div className="card-body">
              <div className="row">
                {overview.gene_subgroups && Object.entries(overview.gene_subgroups).map(([sg, genes]) => (
                  <div key={sg} className="col-md-6 col-lg-4 mb-3">
                    <div className="p-2 rounded h-100" style={{ border: `2px solid ${SUBGROUP_COLORS[sg] || COLOR}` }}>
                      <div className="fw-semibold small mb-1" style={{ color: SUBGROUP_COLORS[sg] || COLOR }}>{sg}</div>
                      <div className="d-flex flex-wrap gap-1">
                        {genes.map(g => (
                          <span key={g} className="badge" style={{ background: GENE_COLORS[g] || COLOR }}>{g}</span>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Critical Clinical Rules */}
          <div className="card mb-4">
            <div className="card-header fw-semibold" style={{ background: '#fce4ec' }}>
              &#x26a0;&#xfe0f; Critical Clinical Rules — NBIA High-Stakes Pearls
            </div>
            <div className="card-body p-0">
              <ul className="list-group list-group-flush">
                {overview.critical_clinical_rules?.map((r, i) => (
                  <li key={i} className="list-group-item" style={{ fontSize: '0.82rem', borderLeft: `4px solid ${COLOR2}` }}>
                    {r}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Gene summary cards */}
          <h6 className="fw-semibold mb-2">Gene Summary ({overview.n_genes} Genes)</h6>
          <div className="row">
            {overview.gene_summary?.map(g => (
              <div key={g.gene} className="col-md-6 col-lg-4 mb-3">
                <div className="card h-100 shadow-sm" style={{ borderTop: `4px solid ${GENE_COLORS[g.gene] || COLOR}` }}>
                  <div className="card-body p-3">
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <span className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</span>
                      <span className="badge" style={{ background: GENE_COLORS[g.gene] || COLOR }}>{g.n_patients} pts</span>
                    </div>
                    <div className="text-muted small mb-1" style={{ fontSize: '0.72rem' }}>{g.alias?.split('—')[1]?.trim()}</div>
                    <div className="small mb-1"><strong>Locus:</strong> {g.locus}</div>
                    <div className="small mb-1"><strong>Pathway:</strong> <span className="text-muted">{g.gene_class?.split(':')[0]}</span></div>
                    <div className="small mb-1"><strong>Mean Dx Age:</strong> {g.mean_age_dx_y}y</div>
                    <div className="small mb-1"><strong>Phenotype:</strong> <span className="text-muted" style={{ fontSize: '0.72rem' }}>{g.phenotype?.substring(0, 120)}…</span></div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* MRI note */}
          {overview.mri_note && (
            <div className="alert mt-3" style={{ background: '#e3f2fd', borderLeft: `4px solid ${COLOR3}` }}>
              <strong>MRI:</strong> {overview.mri_note}
            </div>
          )}
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <div className="mb-2 text-muted small">
            {breakdown.total} genes · {breakdown.total_patients} total patients
          </div>
          <div className="table-responsive">
            <table className="table table-bordered table-hover table-sm" style={{ fontSize: '0.78rem' }}>
              <thead style={{ background: COLOR, color: '#fff' }}>
                <tr>
                  <th>Gene</th><th>Locus</th><th>aa</th><th>Pathway / Subgroup</th>
                  <th>Inheritance</th><th>Pts</th><th>Mean Dx Age</th>
                  <th>Eye-of-Tiger</th><th>Iron MRI</th><th>Key Biomarker</th>
                </tr>
              </thead>
              <tbody>
                {breakdown.genes?.map(g => (
                  <tr key={g.gene}>
                    <td className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</td>
                    <td>{g.locus}</td>
                    <td>{g.aa?.split('(')[0]?.trim()}</td>
                    <td><span style={{ fontSize: '0.70rem' }}>{g.nbia_subgroup}</span></td>
                    <td><span className="badge bg-secondary">{g.inheritance?.split('.')[0]}</span></td>
                    <td>{g.n_patients}</td>
                    <td>{g.mean_age_dx_y}y</td>
                    <td>
                      {g.gene === 'PANK2'
                        ? <span className="badge" style={{ background: COLOR2 }}>&#x1f441;&#xfe0f; PATHOGNOMONIC</span>
                        : <span className="badge bg-secondary">Absent</span>
                      }
                    </td>
                    <td>{g.n_iron_mri} / {g.n_patients}</td>
                    <td style={{ fontSize: '0.70rem' }}>{g.key_biomarker?.substring(0, 80)}…</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {breakdown.genes?.map(g => (
            <div key={g.gene} className="card mb-4 shadow-sm" style={{ borderLeft: `6px solid ${GENE_COLORS[g.gene] || COLOR}` }}>
              <div className="card-header py-2" style={{ background: LIGHT }}>
                <span className="fw-bold fs-6" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</span>
                &nbsp;—&nbsp;<span className="text-muted small">{g.alias?.split('—')[1]?.trim()}</span>
                &nbsp;·&nbsp;<span className="badge bg-secondary">{g.locus}</span>
                &nbsp;·&nbsp;<span className="badge bg-light text-dark border">{g.aa}</span>
                &nbsp;·&nbsp;<span className="badge" style={{ background: SUBGROUP_COLORS[g.nbia_subgroup] || COLOR, fontSize: '0.65rem' }}>{g.nbia_subgroup}</span>
              </div>
              <div className="card-body" style={{ fontSize: '0.80rem' }}>
                <div className="row">
                  <div className="col-md-6">
                    <p><strong style={{ color: COLOR2 }}>Hallmark:</strong> {g.hallmark}</p>
                    <p><strong style={{ color: COLOR3 }}>Inheritance:</strong> {g.inheritance}</p>
                    <p><strong>Phenotype:</strong> {g.phenotype}</p>
                    <p><strong style={{ color: '#f57c00' }}>DDx:</strong> {g.key_ddx}</p>
                  </div>
                  <div className="col-md-6">
                    <p><strong style={{ color: COLOR5 }}>Treatment:</strong> {g.diet_treatment}</p>
                    <p><strong>Key Biomarker:</strong> {g.key_biomarker}</p>
                    <p><strong>NBS Marker:</strong> {g.nbs_marker}</p>
                    {g.founder_variant && (
                      <p><strong>Founder Variant:</strong> <span className="text-muted">{g.founder_variant}</span></p>
                    )}
                    {g.key_variants?.length > 0 && (
                      <p><strong>Key Variants:</strong> {g.key_variants.join(' · ')}</p>
                    )}
                  </div>
                </div>
                <div className="mt-2 p-2 rounded" style={{ background: '#fff3e0', borderLeft: `3px solid ${COLOR2}` }}>
                  <strong style={{ color: COLOR2 }}>&#x26a0;&#xfe0f; Critical CI:</strong> {g.critical_ci}
                </div>
                <div className="mt-2 p-2 rounded" style={{ background: '#e8f5e9', borderLeft: `3px solid ${COLOR5}` }}>
                  <strong style={{ color: COLOR5 }}>Gene Therapy Status:</strong> {g.gene_therapy_status}
                </div>
                <div className="mt-2 p-2 rounded" style={{ background: '#e3f2fd' }}>
                  <strong>Severity Spectrum:</strong> {g.severity_spectrum}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && defs && (
        <div>
          <div className="card mb-4">
            <div className="card-header fw-semibold" style={{ background: LIGHT }}>NBIA Overview</div>
            <div className="card-body" style={{ fontSize: '0.82rem' }}>
              {defs.nbia_overview && Object.entries(defs.nbia_overview).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <strong>{k.replace(/_/g, ' ')}: </strong>
                  <span className="text-muted">{v}</span>
                </div>
              ))}
            </div>
          </div>
          {defs.definitions?.map((d, i) => (
            <div key={i} className="card mb-3 shadow-sm">
              <div className="card-header py-2 fw-semibold" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
                {d.term}
              </div>
              <div className="card-body" style={{ fontSize: '0.81rem', whiteSpace: 'pre-wrap' }}>
                {d.definition}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
