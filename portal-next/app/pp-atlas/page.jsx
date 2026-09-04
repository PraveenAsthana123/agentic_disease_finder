'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#4a148c';  // deep purple — purine/pyrimidine nucleotide
const LIGHT  = '#f3e5f5';
const COLOR2 = '#b71c1c';  // fatal / absolute CI
const COLOR3 = '#0d47a1';  // biomarker / immunodeficiency
const COLOR4 = '#e65100';  // neurological / neuro features
const COLOR5 = '#1b5e20';  // treatable / curative
const COLOR6 = '#006064';  // gene therapy / ERT
const COLOR7 = '#37474f';  // gene class / struct

const SUBGROUP_COLORS = {
  'Purine salvage pathway (HPRT1 · APRT)':                        '#4a148c',
  'Purine catabolism pathway (ADA · PNP · XDH)':                  '#0d47a1',
  'Purine de novo synthesis pathway (ADSL · ATIC)':               '#e65100',
  'Pyrimidine de novo synthesis pathway (UMPS)':                  '#1b5e20',
};

const GENE_COLORS = {
  HPRT1: '#4a148c',
  APRT:  '#7b1fa2',
  ADA:   '#0d47a1',
  PNP:   '#1565c0',
  XDH:   '#1976d2',
  ADSL:  '#e65100',
  ATIC:  '#f57c00',
  UMPS:  '#1b5e20',
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

export default function PPAtlasPage() {
  const [tab, setTab]             = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [err, setErr]             = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/pp-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/pp-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pp-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center"><div className="spinner-border" style={{ color: COLOR }} /><p className="mt-2">Loading PP-Atlas…</p></div>;
  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-2 gap-3">
        <div style={{ width: 8, height: 48, background: COLOR, borderRadius: 4 }} />
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>PP-Atlas — Purine &amp; Pyrimidine Metabolism Disorders</h4>
          <small className="text-muted">
            {overview?.n_genes} genes · {overview?.n_patients} patients (8×40, seeds {overview?.seeds?.[0]}–{overview?.seeds?.[overview.seeds.length-1]}) ·
            HPRT1 · ADA · ADSL · PNP · ATIC · APRT · XDH · UMPS
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
            <KPI label="Pathways" value="4" color={COLOR3} />
            <KPI label="Treatable" value="5/8" color={COLOR5} />
            <KPI label="Gene Therapy" value="ADA-SCID" color={COLOR6} />
            <KPI label="HSCT Curative" value="ADA · PNP" color={COLOR3} />
          </div>

          {/* Pathway groups */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>
              Purine &amp; Pyrimidine Metabolic Pathway Subgroups
            </div>
            <div className="card-body py-2">
              {overview?.gene_subgroups && Object.entries(overview.gene_subgroups).map(([grp, genes]) => (
                <div key={grp} className="mb-2 d-flex align-items-start gap-2">
                  <span className="badge flex-shrink-0" style={{ background: SUBGROUP_COLORS[grp] || COLOR, fontSize: '0.68rem' }}>
                    {grp.split('(')[0].trim()}
                  </span>
                  <span className="small text-muted">
                    <strong style={{ color: SUBGROUP_COLORS[grp] || COLOR }}>{genes.join(' · ')}</strong>
                  </span>
                </div>
              ))}
              <div className="alert alert-info py-1 mb-0 mt-2" style={{ fontSize: '0.78rem' }}>
                <strong>Key principle:</strong> Purines (adenine, guanine) and pyrimidines (uracil, cytosine, thymine)
                are synthesised <em>de novo</em> (expensive, multi-step) and recycled via the <em>salvage pathway</em> (efficient).
                Defects cause substrate accumulation (toxic) or product deficiency (starved cells).
              </div>
            </div>
          </div>

          {/* Critical clinical rules */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: '#ffebee', color: COLOR2 }}>
              Critical Clinical Rules — Prescribing Traps &amp; Diagnostic Pitfalls
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

          {/* Per-gene summary cards */}
          <h6 className="fw-semibold mb-2" style={{ color: COLOR }}>Per-Gene Summary (8 genes · 40 patients each)</h6>
          <div className="row g-2 mb-3">
            {overview?.gene_summary?.map(g => (
              <div key={g.gene} className="col-12 col-md-6 col-lg-3">
                <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${GENE_COLORS[g.gene] || COLOR}` }}>
                  <div className="card-body py-2 px-3">
                    <div className="fw-bold mb-0" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</div>
                    <div className="text-muted" style={{ fontSize: '0.72rem' }}>{g.locus} · {g.gene_class}</div>
                    <div className="small mt-1 mb-1" style={{ color: '#555' }}>{g.phenotype}</div>
                    <div className="small"><strong>Avg Dx:</strong> {g.mean_age_dx_y}y</div>
                    <div className="small text-muted" style={{ fontSize: '0.7rem' }}>{g.pp_subgroup}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* NBS note */}
          <div className="alert alert-warning py-2 mb-0" style={{ fontSize: '0.8rem' }}>
            <strong>NBS Status:</strong> None of these 8 disorders is in standard NBS panels.
            ADA-SCID detected by low TRECs on expanded NBS. Others require targeted metabolite
            testing (uric acid, orotic acid, purine panel) triggered by clinical suspicion.
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
                  <th>Gene</th><th>Pathway</th><th>aa / kDa</th><th>Locus</th>
                  <th>N</th><th>Inheritance</th><th>Avg Dx (y)</th><th>Phenotype (brief)</th>
                </tr>
              </thead>
              <tbody>
                {breakdown.genes.map(g => (
                  <tr key={g.gene}>
                    <td className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</td>
                    <td style={{ maxWidth: 140, fontSize: '0.73rem' }}>{g.gene_class}</td>
                    <td className="text-nowrap">{g.aa} / {g.kDa}</td>
                    <td>{g.locus}</td>
                    <td>{g.n_patients}</td>
                    <td style={{ fontSize: '0.73rem' }}>{g.inheritance?.split('.')[0]}</td>
                    <td>{g.mean_age_dx_y}</td>
                    <td style={{ maxWidth: 200, fontSize: '0.73rem' }}>{g.phenotype}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-muted small">
            {breakdown.total} genes · {breakdown.total_patients} total patients ·
            PP = Purine &amp; Pyrimidine · DBS = dried blood spot
          </p>
        </div>
      )}

      {/* ── Clinical Atlas ── */}
      {tab === 'Clinical Atlas' && breakdown?.genes && (
        <div>
          {breakdown.genes.map(g => (
            <div key={g.gene} className="card shadow-sm mb-3" style={{ borderLeft: `5px solid ${GENE_COLORS[g.gene] || COLOR}` }}>
              <div className="card-header d-flex justify-content-between align-items-center"
                   style={{ background: LIGHT }}>
                <div>
                  <span className="fw-bold fs-6 me-2" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</span>
                  <span className="text-muted small">{g.aa} · {g.locus} · OMIM Gene {g.omim_gene}</span>
                </div>
                <div className="d-flex gap-1 flex-wrap">
                  <span className="badge" style={{ background: GENE_COLORS[g.gene] || COLOR, fontSize: '0.65rem' }}>{g.gene_class}</span>
                  <span className="badge" style={{ background: COLOR7, fontSize: '0.65rem' }}>{g.pp_subgroup?.split('(')[0]?.trim()}</span>
                </div>
              </div>
              <div className="card-body">
                <p className="small mb-1"><strong>Phenotype:</strong> {g.phenotype}</p>
                <p className="small mb-1"><strong>Inheritance:</strong> {g.inheritance}</p>
                <p className="small mb-1"><strong>Key Biomarker:</strong> {g.key_biomarker}</p>
                <p className="small mb-1"><strong>NBS Marker:</strong> {g.nbs_marker}</p>
                <p className="small mb-1"><strong>Severity Spectrum:</strong> {g.severity_spectrum}</p>
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
                <p className="small mb-1"><strong>Gene Therapy / HSCT Status:</strong> {g.gene_therapy_status}</p>
                <p className="small mb-1"><strong>Founder Variant:</strong> {g.founder_variant}</p>
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
              PP-Atlas Overview — {defs.pp_overview?.genes_in_atlas} Genes
            </div>
            <div className="card-body">
              <p className="small mb-1"><strong>Full Name:</strong> {defs.pp_overview?.full_name}</p>
              <p className="small mb-1"><strong>Collective Incidence:</strong> {defs.pp_overview?.collective_incidence}</p>
              <p className="small mb-0"><strong>NBS Note:</strong> {defs.pp_overview?.nbs_note}</p>
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
