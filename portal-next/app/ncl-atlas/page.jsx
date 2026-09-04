'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// NCL color palette — amber/brown for ceroid lipofuscinosis storage pigment
const COLOR  = '#4e342e';  // dark brown — ceroid lipopigment
const LIGHT  = '#efebe9';
const COLOR2 = '#b71c1c';  // critical / ERT / CLN2
const COLOR3 = '#1565c0';  // CLN3 most common / vacuolated lymphocytes
const COLOR4 = '#e65100';  // AD / CLN4B dominant
const COLOR5 = '#1b5e20';  // gene therapy / treatable
const COLOR6 = '#4a148c';  // CLN10 congenital / rare

const GENE_COLORS = {
  CLN1:  '#b71c1c',   // infantile, severe, earliest
  CLN2:  '#1b5e20',   // ERT available — highlight green
  CLN3:  '#1565c0',   // most common
  CLN4B: '#e65100',   // AD — dominant
  CLN5:  '#37474f',   // Finnish variant
  CLN6:  '#4a148c',   // ER-resident
  CLN7:  '#880e4f',   // Turkish variant
  CLN10: '#4e342e',   // congenital, earliest of all
};

const EM_COLORS = {
  'GRODs':                              '#b71c1c',
  'Curvilinear bodies (CB)':            '#1b5e20',
  'Fingerprint profiles (FP)':          '#1565c0',
  'Mixed curvilinear + fingerprint':    '#37474f',
  'Rectilinear profiles':               '#4a148c',
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

export default function NCLAtlasPage() {
  const [tab, setTab]             = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [err, setErr]             = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ncl-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/ncl-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ncl-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return (
    <div className="d-flex align-items-center justify-content-center" style={{ minHeight: '60vh' }}>
      <div className="spinner-border" style={{ color: COLOR }} role="status" />
      <span className="ms-3 text-muted">Loading NCL-Atlas&#x2026;</span>
    </div>
  );
  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: COLOR, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">&#x1f9e0; NCL-Atlas — 8-Gene Neuronal Ceroid Lipofuscinosis (Batten Disease)</h4>
        <div className="small opacity-75">
          CLN1 (PPT1) · CLN2 (TPP1) · CLN3 · CLN4B (DNAJC5) · CLN5 · CLN6 · CLN7 (MFSD8) · CLN10 (CTSD)
          &nbsp;·&nbsp; {overview?.n_patients} patients ({overview?.n_genes} genes × 40 each)
          &nbsp;·&nbsp; Most common inherited progressive neurodegenerative storage disorders of childhood
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
            <KPI label="Visual Failure" value={overview.n_visual_failure} color={COLOR} />
            <KPI label="Seizures" value={overview.n_seizures} color={COLOR2} />
            <KPI label="On ERT (CLN2)" value={overview.n_on_ert_cerliponase} color={COLOR5} />
            <KPI label="HCM (CLN10)" value={overview.n_hcm_cln10} color={COLOR6} />
            <KPI label="Vacoulated Lymph (CLN3)" value={overview.n_vacuolated_lymphocytes_cln3} color={COLOR3} />
            <KPI label="CLN3 1kb Deletion" value={overview.n_cln3_deletion} color={COLOR3} />
          </div>

          {/* ERT Highlight — CLN2 only */}
          <div className="alert mb-4" style={{ background: '#e8f5e9', borderLeft: `6px solid ${COLOR5}`, borderRadius: '6px' }}>
            <div className="fw-bold mb-1" style={{ color: COLOR5 }}>&#x2705; CLN2/TPP1 — ONLY APPROVED NCL ERT: Cerliponase Alfa (Brineura)</div>
            <div className="small text-muted">
              300 mg ICV every 2 weeks via Ommaya reservoir · FDA approved 2017 · Slows motor decline ·
              Start BEFORE symptom onset if sibling diagnosed · ALL CLN2 patients must have ERT access arranged
            </div>
          </div>

          {/* NCL Subgroups */}
          <div className="card mb-4">
            <div className="card-header fw-semibold" style={{ background: LIGHT }}>
              NCL Subgroups by Onset / Gene
            </div>
            <div className="card-body">
              <div className="row">
                {overview.gene_subgroups && Object.entries(overview.gene_subgroups).map(([sg, genes]) => (
                  <div key={sg} className="col-md-6 col-lg-4 mb-3">
                    <div className="p-2 rounded h-100" style={{ border: `2px solid ${GENE_COLORS[genes[0]] || COLOR}` }}>
                      <div className="fw-semibold small mb-1" style={{ color: GENE_COLORS[genes[0]] || COLOR }}>{sg}</div>
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

          {/* EM Ultrastructure Key */}
          {overview.em_key && (
            <div className="card mb-4">
              <div className="card-header fw-semibold" style={{ background: LIGHT }}>
                &#x1f52c; EM Ultrastructure — Gene-Specific Storage Bodies
              </div>
              <div className="card-body">
                <div className="row">
                  {Object.entries(overview.em_key).map(([em, genes]) => (
                    <div key={em} className="col-md-6 col-lg-4 mb-3">
                      <div className="p-2 rounded" style={{ border: `2px solid ${EM_COLORS[em] || COLOR}` }}>
                        <div className="fw-semibold small" style={{ color: EM_COLORS[em] || COLOR }}>{em}</div>
                        <div className="d-flex flex-wrap gap-1 mt-1">
                          {genes.map(g => (
                            <span key={g} className="badge" style={{ background: GENE_COLORS[g] || COLOR }}>{g}</span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="alert mt-2 mb-0" style={{ background: '#fff3e0', fontSize: '0.80rem' }}>
                  <strong>Pearl:</strong> GRODs are seen in CLN1 AND CLN10 — distinguish by onset (CLN1 infantile, CLN10 congenital + HCM).
                  CLN4B/DNAJC5 shows fingerprint profiles but is the ONLY dominant NCL and has NO visual failure.
                </div>
              </div>
            </div>
          )}

          {/* Critical Clinical Rules */}
          <div className="card mb-4">
            <div className="card-header fw-semibold" style={{ background: '#fce4ec' }}>
              &#x26a0;&#xfe0f; Critical Clinical Rules — NCL High-Stakes Pearls
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
                    <div className="text-muted small mb-1" style={{ fontSize: '0.72rem' }}>
                      {g.ncl_subgroup}
                    </div>
                    <div className="small mb-1"><strong>Locus:</strong> {g.locus}</div>
                    <div className="small mb-1"><strong>EM:</strong> <span className="text-muted" style={{ fontSize: '0.72rem' }}>{g.em_finding?.substring(0, 80)}…</span></div>
                    <div className="small mb-1"><strong>Mean Dx Age:</strong> {g.mean_age_dx_y}y</div>
                    <div className="small"><strong>Phenotype:</strong> <span className="text-muted" style={{ fontSize: '0.72rem' }}>{g.phenotype?.substring(0, 100)}…</span></div>
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
                  <th>Gene</th><th>Locus</th><th>aa</th><th>NCL Subgroup</th>
                  <th>Inheritance</th><th>Pts</th><th>Mean Dx Age</th>
                  <th>EM Finding</th><th>Visual Failure</th><th>ERT/Therapy</th>
                </tr>
              </thead>
              <tbody>
                {breakdown.genes?.map(g => (
                  <tr key={g.gene}>
                    <td className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</td>
                    <td>{g.locus}</td>
                    <td>{g.aa?.split('(')[0]?.trim()}</td>
                    <td><span style={{ fontSize: '0.70rem' }}>{g.ncl_subgroup}</span></td>
                    <td>
                      <span className={`badge ${g.gene === 'CLN4B' ? '' : 'bg-secondary'}`}
                        style={g.gene === 'CLN4B' ? { background: COLOR4 } : {}}>
                        {g.gene === 'CLN4B' ? 'AD ⚠' : 'AR'}
                      </span>
                    </td>
                    <td>{g.n_patients}</td>
                    <td>{g.mean_age_dx_y}y</td>
                    <td>
                      <span className="badge" style={{ background: EM_COLORS[g.em_finding?.split('—')[0]?.trim()] || '#607d8b', fontSize: '0.68rem' }}>
                        {g.em_finding?.split('—')[0]?.trim()?.substring(0, 20)}
                      </span>
                    </td>
                    <td>
                      {g.gene === 'CLN4B'
                        ? <span className="badge" style={{ background: COLOR4 }}>SPARED</span>
                        : <span className="badge bg-secondary">Present</span>
                      }
                    </td>
                    <td style={{ fontSize: '0.70rem' }}>
                      {g.gene === 'CLN2'
                        ? <span className="badge" style={{ background: COLOR5 }}>ERT: Cerliponase ✓</span>
                        : <span style={{ color: '#999' }}>{g.gene_therapy_status?.substring(0, 50)}…</span>
                      }
                    </td>
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
                &nbsp;/&nbsp;<span className="fw-semibold text-muted">{g.protein}</span>
                &nbsp;—&nbsp;<span className="text-muted small">{g.ncl_subgroup}</span>
                &nbsp;·&nbsp;<span className="badge bg-secondary">{g.locus}</span>
                &nbsp;·&nbsp;<span className="badge bg-light text-dark border">{g.aa}</span>
                {g.gene === 'CLN4B' && (
                  <span className="badge ms-1" style={{ background: COLOR4 }}>AD — DOMINANT ⚠</span>
                )}
                {g.gene === 'CLN2' && (
                  <span className="badge ms-1" style={{ background: COLOR5 }}>ERT APPROVED ✓</span>
                )}
              </div>
              <div className="card-body" style={{ fontSize: '0.80rem' }}>
                <div className="row">
                  <div className="col-md-6">
                    <p><strong style={{ color: COLOR2 }}>EM Finding:</strong> {g.em_finding}</p>
                    <p><strong style={{ color: COLOR }}>Hallmark:</strong> {g.hallmark}</p>
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
            <div className="card-header fw-semibold" style={{ background: LIGHT }}>NCL Overview</div>
            <div className="card-body" style={{ fontSize: '0.82rem' }}>
              {defs.ncl_overview && Object.entries(defs.ncl_overview).map(([k, v]) => (
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
