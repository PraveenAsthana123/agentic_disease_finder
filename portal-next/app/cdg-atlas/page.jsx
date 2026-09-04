'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#1a237e';  // deep indigo — glycosylation / glycan chemistry
const LIGHT  = '#e8eaf6';
const COLOR2 = '#b71c1c';  // fatal / no treatment
const COLOR3 = '#1b5e20';  // treatable / controlled
const COLOR4 = '#e65100';  // neuro / exercise
const COLOR5 = '#006064';  // transporter / Bombay
const COLOR6 = '#4a148c';  // Golgi / COG complex
const COLOR7 = '#37474f';  // biomarker / IEF

const CDG_TYPE_COLORS = {
  'Type I (LLO defect)':            '#1a237e',
  'Mixed (Type I + II features)':   '#880e4f',
  'Type II (Golgi processing)':     '#bf360c',
  'Type I (LLO initiation defect)': '#1a237e',
  'Type I (dolichol synthesis defect)': '#283593',
  'Type I or Mixed (Golgi ion homeostasis defect)': '#880e4f',
};

const CLASS_COLORS = {
  phosphomutase:       '#1a237e',
  isomerase:           '#006064',
  glucosyltransferase: '#4a148c',
  transferase:         '#bf360c',
  reductase:           '#e65100',
  golgi_tethering:     '#6a1b9a',
  transporter:         '#00695c',
};

const CLASS_LABELS = {
  phosphomutase:       'Phosphomutase — PMM2 (CDG-Ia, most common), PGM1 (CDG-It, galactose-treatable)',
  isomerase:           'Isomerase — MPI (CDG-Ib, mannose-TREATABLE, NO neuro)',
  glucosyltransferase: 'Glucosyltransferase — ALG6 (CDG-Ic, 2nd most common N-CDG)',
  transferase:         'Transferase — DPAGT1 (CDG-Ij, myasthenic CMS), MGAT2 (CDG-IIa, bleeding)',
  reductase:           'Reductase — SRD5A3 (CDG-Iq, dolichol synthesis, retinal dystrophy)',
  golgi_tethering:     'Golgi Tethering — COG7 (CDG-IIe, most severe Golgi-CDG, wrinkled skin)',
  transporter:         'Transporter — SLC35C1 (CDG-IIc/LAD-II, Bombay blood group), TMEM165 (CDG-IIk, Mn2+)',
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

export default function CDGAtlasPage() {
  const [tab, setTab]             = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [err, setErr]             = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/cdg-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/cdg-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/cdg-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center"><div className="spinner-border" style={{ color: COLOR }} /><p className="mt-2">Loading CDG-Atlas…</p></div>;
  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;

  const ac = overview?.aggregate_clinical || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-2 gap-3">
        <div style={{ width: 8, height: 48, background: COLOR, borderRadius: 4 }} />
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>CDG-Atlas — Congenital Disorders of Glycosylation</h4>
          <small className="text-muted">
            10 genes · {overview?.n_patients} patients (10×40, seeds {overview?.seeds?.[0]}–{overview?.seeds?.[overview.seeds.length-1]}) ·
            PMM2 · MPI · ALG6 · DPAGT1 · SRD5A3 · PGM1 · COG7 · SLC35C1 · MGAT2 · TMEM165
          </small>
        </div>
      </div>

      {/* Tab Nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === t ? ' active fw-semibold' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'Overview' && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Genes" value={overview?.n_genes} color={COLOR} />
            <KPI label="Patients" value={overview?.n_patients} color={COLOR} />
            <KPI label="Gene Classes" value="7" color={COLOR} />
            <KPI label="Tx Controlled" value={`${ac.pct_tx_controlled}%`} color={COLOR3} />
            <KPI label="Neurological" value={`${ac.pct_neurological}%`} color={COLOR4} />
            <KPI label="Deceased" value={`${ac.pct_deceased}%`} color={COLOR2} />
          </div>

          {/* CDG Type legend */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>Transferrin IEF Pattern by Gene</div>
            <div className="card-body py-2">
              {overview?.cdg_types && Object.entries(overview.cdg_types).map(([type, genes]) => (
                <div key={type} className="mb-2">
                  <span className="badge me-2" style={{ background: CDG_TYPE_COLORS[type] || COLOR }}>{type}</span>
                  <span className="small">{genes.join(' · ')}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Gene class legend */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>Gene Class Map</div>
            <div className="card-body py-2">
              {Object.entries(CLASS_LABELS).map(([cls, label]) => (
                <div key={cls} className="mb-1">
                  <span className="badge me-2" style={{ background: CLASS_COLORS[cls] || COLOR }}>{cls}</span>
                  <span className="small">{label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Aggregate clinical bars */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>Aggregate Clinical (400 patients)</div>
            <div className="card-body">
              <BarRow label="Treatment Controlled" pct={ac.pct_tx_controlled} color={COLOR3} />
              <BarRow label="Liver Disease" pct={ac.pct_liver_disease} color={COLOR5} />
              <BarRow label="Neurological Involvement" pct={ac.pct_neurological} color={COLOR4} />
              <BarRow label="Deceased" pct={ac.pct_deceased} color={COLOR2} />
            </div>
          </div>

          {/* Gene summary cards */}
          <h6 className="fw-semibold mb-2" style={{ color: COLOR }}>Gene-Level Summary</h6>
          <div className="row g-2 mb-3">
            {overview?.gene_summary?.map(g => (
              <div key={g.gene} className="col-12 col-md-6 col-lg-4">
                <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${CDG_TYPE_COLORS[g.cdg_type] || COLOR}` }}>
                  <div className="card-body py-2 px-3">
                    <div className="fw-bold" style={{ color: CDG_TYPE_COLORS[g.cdg_type] || COLOR }}>{g.gene}</div>
                    <div className="text-muted small mb-1">{g.locus} · {g.cdg_type}</div>
                    <BarRow label="Tx controlled" pct={g.pct_tx} color={COLOR3} />
                    <BarRow label="Liver disease" pct={g.pct_liver} color={COLOR5} />
                    <BarRow label="Neurological" pct={g.pct_neuro} color={COLOR4} />
                    <BarRow label="Deceased" pct={g.pct_deceased} color={COLOR2} />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Critical clinical rules */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: '#ffebee', color: COLOR2 }}>
              Critical Clinical Rules (10 Rules — One Per Gene)
            </div>
            <ul className="list-group list-group-flush">
              {overview?.critical_clinical_rules?.map((r, i) => (
                <li key={i} className="list-group-item py-2" style={{ fontSize: '0.82rem' }}>
                  <span className="badge me-2" style={{ background: COLOR, fontSize: '0.7rem' }}>{i + 1}</span>
                  {r}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* ── Gene Table ── */}
      {tab === 'Gene Table' && breakdown?.genes && (
        <div>
          <div className="table-responsive">
            <table className="table table-bordered table-sm table-hover" style={{ fontSize: '0.79rem' }}>
              <thead style={{ background: COLOR, color: '#fff' }}>
                <tr>
                  <th>Gene</th><th>Disease / CDG Type</th><th>aa / kDa</th><th>Locus</th>
                  <th>IEF Pattern</th><th>Class</th><th>N</th>
                  <th>Tx%</th><th>Liver%</th><th>Neuro%</th><th>†%</th><th>Avg Dx (y)</th>
                </tr>
              </thead>
              <tbody>
                {breakdown.genes.map(g => (
                  <tr key={g.gene}>
                    <td className="fw-bold" style={{ color: CDG_TYPE_COLORS[g.cdg_type] || COLOR }}>{g.gene}</td>
                    <td style={{ maxWidth: 220 }}>{g.phenotype?.split(';')[0]}</td>
                    <td className="text-nowrap">{g.aa} / {g.kDa}</td>
                    <td>{g.locus}</td>
                    <td><span className="badge" style={{ background: CDG_TYPE_COLORS[g.cdg_type] || COLOR, fontSize: '0.68rem' }}>{g.cdg_type?.split(' ')[1]}</span></td>
                    <td><span className="badge" style={{ background: CLASS_COLORS[g.gene_class] || COLOR7, fontSize: '0.68rem' }}>{g.gene_class}</span></td>
                    <td>{g.n_patients}</td>
                    <td style={{ color: COLOR3 }}>{g.pct_tx}%</td>
                    <td style={{ color: COLOR5 }}>{g.pct_liver}%</td>
                    <td style={{ color: COLOR4 }}>{g.pct_neuro}%</td>
                    <td style={{ color: COLOR2 }}>{g.pct_deceased}%</td>
                    <td>{g.mean_age_dx_y}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-muted small">
            {breakdown.total} genes · {breakdown.total_patients} total patients · IEF = transferrin isoelectric focusing pattern class
          </p>
        </div>
      )}

      {/* ── Clinical Atlas ── */}
      {tab === 'Clinical Atlas' && breakdown?.genes && (
        <div>
          {breakdown.genes.map(g => (
            <div key={g.gene} className="card shadow-sm mb-3" style={{ borderLeft: `5px solid ${CDG_TYPE_COLORS[g.cdg_type] || COLOR}` }}>
              <div className="card-header d-flex justify-content-between align-items-center"
                   style={{ background: LIGHT }}>
                <div>
                  <span className="fw-bold fs-6 me-2" style={{ color: CDG_TYPE_COLORS[g.cdg_type] || COLOR }}>{g.gene}</span>
                  <span className="text-muted small">{g.aa} · {g.locus} · OMIM Gene {g.omim_gene}</span>
                </div>
                <div className="d-flex gap-1">
                  <span className="badge" style={{ background: CDG_TYPE_COLORS[g.cdg_type] || COLOR, fontSize: '0.68rem' }}>{g.cdg_type}</span>
                  <span className="badge" style={{ background: CLASS_COLORS[g.gene_class] || COLOR7, fontSize: '0.68rem' }}>{g.gene_class}</span>
                </div>
              </div>
              <div className="card-body">
                <div className="row g-2 mb-2">
                  <div className="col-6 col-md-3"><small className="text-muted d-block">Tx Controlled</small><strong style={{ color: COLOR3 }}>{g.pct_tx}%</strong></div>
                  <div className="col-6 col-md-3"><small className="text-muted d-block">Liver Disease</small><strong style={{ color: COLOR5 }}>{g.pct_liver}%</strong></div>
                  <div className="col-6 col-md-3"><small className="text-muted d-block">Neurological</small><strong style={{ color: COLOR4 }}>{g.pct_neuro}%</strong></div>
                  <div className="col-6 col-md-3"><small className="text-muted d-block">Deceased</small><strong style={{ color: COLOR2 }}>{g.pct_deceased}%</strong></div>
                </div>
                <p className="small mb-1"><strong>Phenotype:</strong> {g.phenotype}</p>
                <p className="small mb-1"><strong>Inheritance:</strong> {g.inheritance}</p>
                <details className="mb-1">
                  <summary className="small fw-semibold" style={{ color: COLOR, cursor: 'pointer' }}>Hallmarks (expand)</summary>
                  <p className="small mt-1 mb-0" style={{ whiteSpace: 'pre-wrap' }}>{g.hallmark}</p>
                </details>
                <details className="mb-1">
                  <summary className="small fw-semibold" style={{ color: COLOR2, cursor: 'pointer' }}>Critical CI / Contraindications</summary>
                  <p className="small mt-1 mb-0" style={{ whiteSpace: 'pre-wrap' }}>{g.critical_ci}</p>
                </details>
                <details className="mb-1">
                  <summary className="small fw-semibold" style={{ color: COLOR4, cursor: 'pointer' }}>DDx (expand)</summary>
                  <p className="small mt-1 mb-0" style={{ whiteSpace: 'pre-wrap' }}>{g.key_ddx}</p>
                </details>
                <p className="small mb-1"><strong>Treatment:</strong> {g.diet_treatment}</p>
                <p className="small mb-1"><strong>Gene Therapy:</strong> {g.gene_therapy_status}</p>
                <p className="small mb-1"><strong>Key Biomarker:</strong> {g.key_biomarker}</p>
                <p className="small mb-1"><strong>Severity Spectrum:</strong> {g.severity_spectrum}</p>
                <p className="small mb-0"><strong>Founder Variant:</strong> {g.founder_variant}</p>
                {g.key_variants?.length > 0 && (
                  <div className="mt-1">
                    <strong className="small">Key Variants:</strong>
                    <ul className="mb-0 ps-3">
                      {g.key_variants.map((v, i) => <li key={i} className="small">{v}</li>)}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'Definitions' && defs && (
        <div>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>
              CDG Overview — {defs.cdg_overview?.genes_in_atlas} Genes · {defs.cdg_overview?.total_known_cdg_types}
            </div>
            <div className="card-body">
              <p className="small mb-1"><strong>Full Name:</strong> {defs.cdg_overview?.full_name}</p>
              <p className="small mb-1"><strong>Incidence:</strong> {defs.cdg_overview?.collective_incidence}</p>
              <p className="small mb-1"><strong>Inheritance:</strong> {defs.cdg_overview?.inheritance_note}</p>
              <p className="small mb-0"><strong>NBS Note:</strong> {defs.cdg_overview?.nbs_note}</p>
            </div>
          </div>
          {defs.definitions?.map((d, i) => (
            <div key={i} className="card shadow-sm mb-2">
              <div className="card-header py-2 fw-semibold" style={{ background: LIGHT, color: COLOR, fontSize: '0.85rem' }}>
                {d.term}
              </div>
              <div className="card-body py-2">
                <p className="small mb-0">{d.definition}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
