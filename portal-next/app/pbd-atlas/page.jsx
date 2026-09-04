'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#1b5e20';  // deep forest green — peroxisome/organelle biogenesis
const LIGHT  = '#e8f5e9';
const COLOR2 = '#b71c1c';  // fatal / Lorenzo Oil CI
const COLOR3 = '#0d47a1';  // VLCFA / biomarker
const COLOR4 = '#e65100';  // retinal / neurological
const COLOR5 = '#4a148c';  // SNHL / severe
const COLOR6 = '#006064';  // DHA / treatable partial
const COLOR7 = '#37474f';  // gene class / struct

const SUBGROUP_COLORS = {
  'AAA-ATPase recycling complex (PEX1·PEX6·PEX26)':        '#1b5e20',
  'RING-finger E3 ubiquitin-ligase triad (PEX2·PEX10·PEX12*)': '#4a148c',
  'Membrane biogenesis + PTS receptor axis (PEX3·PEX5·PEX16)': '#0d47a1',
};

const GENE_COLORS = {
  PEX1:  '#1b5e20',
  PEX6:  '#2e7d32',
  PEX26: '#388e3c',
  PEX10: '#6a1b9a',
  PEX2:  '#4a148c',
  PEX3:  '#0d47a1',
  PEX5:  '#1565c0',
  PEX16: '#1976d2',
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

export default function PBDAtlasPage() {
  const [tab, setTab]             = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [err, setErr]             = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/pbd-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/pbd-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pbd-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center"><div className="spinner-border" style={{ color: COLOR }} /><p className="mt-2">Loading PBD-Atlas…</p></div>;
  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;

  const ac = overview?.aggregate_clinical || {};
  const pheno = overview?.phenotypic_distribution || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-2 gap-3">
        <div style={{ width: 8, height: 48, background: COLOR, borderRadius: 4 }} />
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>PBD-Atlas — Peroxisomal Biogenesis Disorders</h4>
          <small className="text-muted">
            8 genes · {overview?.n_patients} patients (8×40, seeds {overview?.seeds?.[0]}–{overview?.seeds?.[overview.seeds.length-1]}) ·
            PEX1 · PEX6 · PEX26 · PEX10 · PEX2 · PEX3 · PEX5 · PEX16
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
            <KPI label="VLCFA Elevated" value={`${ac.pct_vlcfa_elevated}%`} color={COLOR3} />
            <KPI label="Plasmalogens Low" value={`${ac.pct_plasmalogen_low}%`} color={COLOR3} />
            <KPI label="Retinal Dystrophy" value={`${ac.pct_retinal_dystrophy}%`} color={COLOR4} />
            <KPI label="SNHL" value={`${ac.pct_snhl}%`} color={COLOR5} />
            <KPI label="Deceased" value={`${ac.pct_deceased}%`} color={COLOR2} />
            <KPI label="DHA Therapy" value={`${ac.pct_dha_therapy}%`} color={COLOR6} />
          </div>

          {/* Phenotypic distribution */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>
              ZSD Phenotypic Spectrum (all 8 genes · {overview?.n_patients} patients)
            </div>
            <div className="card-body">
              <div className="row g-3 text-center mb-2">
                <div className="col-4">
                  <div className="fw-bold fs-5" style={{ color: COLOR2 }}>{pheno.ZS?.pct}%</div>
                  <div className="small text-muted">ZS — Zellweger Syndrome</div>
                  <div className="small text-muted">(n={pheno.ZS?.n})</div>
                  <div className="small">Most severe; neonatal; die &lt;12 months</div>
                </div>
                <div className="col-4">
                  <div className="fw-bold fs-5" style={{ color: COLOR4 }}>{pheno.NALD?.pct}%</div>
                  <div className="small text-muted">NALD — Neonatal ALD</div>
                  <div className="small text-muted">(n={pheno.NALD?.n})</div>
                  <div className="small">Intermediate; progressive leukodystrophy</div>
                </div>
                <div className="col-4">
                  <div className="fw-bold fs-5" style={{ color: COLOR6 }}>{pheno.IRD?.pct}%</div>
                  <div className="small text-muted">IRD — Infantile Refsum Disease</div>
                  <div className="small text-muted">(n={pheno.IRD?.n})</div>
                  <div className="small">Mildest; adult survival; RP + SNHL</div>
                </div>
              </div>
              <div className="alert alert-warning py-1 mb-0" style={{ fontSize: '0.78rem' }}>
                <strong>ZSD is ONE continuum</strong> — severity is genotype-driven (residual peroxin function).
                G843D/G843D (PEX1) → NALD/IRD (never ZS). Null/null → severe ZS.
              </div>
            </div>
          </div>

          {/* Gene subgroup legend */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>Peroxin Functional Subgroups</div>
            <div className="card-body py-2">
              {overview?.gene_subgroups && Object.entries(overview.gene_subgroups).map(([grp, genes]) => (
                <div key={grp} className="mb-2">
                  <span className="badge me-2" style={{ background: SUBGROUP_COLORS[grp] || COLOR }}>{grp.split('(')[1]?.replace(')', '') || grp}</span>
                  <span className="small fw-semibold me-1" style={{ color: SUBGROUP_COLORS[grp] || COLOR }}>{grp.split(' (')[0]}</span>
                  <span className="small text-muted">→ {genes.join(' · ')}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Aggregate clinical bars */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>
              Aggregate Clinical Outcomes (320 patients — all 8 genes)
            </div>
            <div className="card-body">
              <BarRow label="VLCFA Elevated (plasma C26:0)" pct={ac.pct_vlcfa_elevated} color={COLOR3} />
              <BarRow label="Plasmalogens Low (erythrocyte)" pct={ac.pct_plasmalogen_low} color={COLOR3} />
              <BarRow label="Retinal Dystrophy (cone-rod)" pct={ac.pct_retinal_dystrophy} color={COLOR4} />
              <BarRow label="Sensorineural Hearing Loss (SNHL)" pct={ac.pct_snhl} color={COLOR5} />
              <BarRow label="Liver Disease" pct={ac.pct_liver_disease} color={COLOR7} />
              <BarRow label="Seizures" pct={ac.pct_seizures} color={COLOR4} />
              <BarRow label="DHA Therapy (NALD/IRD survivors)" pct={ac.pct_dha_therapy} color={COLOR6} />
              <BarRow label="Deceased" pct={ac.pct_deceased} color={COLOR2} />
            </div>
          </div>

          {/* Gene summary cards */}
          <h6 className="fw-semibold mb-2" style={{ color: COLOR }}>Per-Gene Summary (ZS / NALD / IRD split)</h6>
          <div className="row g-2 mb-3">
            {overview?.gene_summary?.map(g => (
              <div key={g.gene} className="col-12 col-md-6 col-lg-3">
                <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${GENE_COLORS[g.gene] || COLOR}` }}>
                  <div className="card-body py-2 px-3">
                    <div className="fw-bold mb-0" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</div>
                    <div className="text-muted" style={{ fontSize: '0.72rem' }}>{g.locus}</div>
                    <div className="d-flex gap-1 mb-1" style={{ fontSize: '0.7rem' }}>
                      <span className="badge" style={{ background: COLOR2 }}>ZS {g.pct_zs}%</span>
                      <span className="badge" style={{ background: COLOR4 }}>NALD {g.pct_nald}%</span>
                      <span className="badge" style={{ background: COLOR6 }}>IRD {g.pct_ird}%</span>
                    </div>
                    <BarRow label="Retinal" pct={g.pct_retinal} color={COLOR4} />
                    <BarRow label="SNHL" pct={g.pct_snhl} color={COLOR5} />
                    <BarRow label="Deceased" pct={g.pct_deceased} color={COLOR2} />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Critical clinical rules */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ background: '#ffebee', color: COLOR2 }}>
              Critical Clinical Rules (10 Rules)
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
                  <th>Gene</th><th>Subgroup</th><th>aa / kDa</th><th>Locus</th>
                  <th>N</th><th>ZS%</th><th>NALD%</th><th>IRD%</th>
                  <th>Retinal%</th><th>SNHL%</th><th>†%</th><th>Avg Dx (y)</th>
                </tr>
              </thead>
              <tbody>
                {breakdown.genes.map(g => (
                  <tr key={g.gene}>
                    <td className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</td>
                    <td style={{ maxWidth: 160, fontSize: '0.73rem' }}>{g.gene_class}</td>
                    <td className="text-nowrap">{g.aa} / {g.kDa}</td>
                    <td>{g.locus}</td>
                    <td>{g.n_patients}</td>
                    <td style={{ color: COLOR2 }}>{g.pct_zs}%</td>
                    <td style={{ color: COLOR4 }}>{g.pct_nald}%</td>
                    <td style={{ color: COLOR6 }}>{g.pct_ird}%</td>
                    <td style={{ color: COLOR4 }}>{g.pct_retinal}%</td>
                    <td style={{ color: COLOR5 }}>{g.pct_snhl}%</td>
                    <td style={{ color: COLOR2 }}>{g.pct_deceased}%</td>
                    <td>{g.mean_age_dx_y}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-muted small">
            {breakdown.total} genes · {breakdown.total_patients} total patients ·
            ZS = Zellweger Syndrome · NALD = Neonatal ALD · IRD = Infantile Refsum Disease
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
                  <span className="badge" style={{ background: COLOR2, fontSize: '0.65rem' }}>ZS {g.pct_zs}%</span>
                  <span className="badge" style={{ background: COLOR4, fontSize: '0.65rem' }}>NALD {g.pct_nald}%</span>
                  <span className="badge" style={{ background: COLOR6, fontSize: '0.65rem' }}>IRD {g.pct_ird}%</span>
                  <span className="badge" style={{ background: COLOR7, fontSize: '0.65rem' }}>{g.gene_class}</span>
                </div>
              </div>
              <div className="card-body">
                <div className="row g-2 mb-2">
                  <div className="col-6 col-md-3"><small className="text-muted d-block">Retinal Dystrophy</small><strong style={{ color: COLOR4 }}>{g.pct_retinal}%</strong></div>
                  <div className="col-6 col-md-3"><small className="text-muted d-block">SNHL</small><strong style={{ color: COLOR5 }}>{g.pct_snhl}%</strong></div>
                  <div className="col-6 col-md-3"><small className="text-muted d-block">Liver Disease</small><strong style={{ color: COLOR7 }}>{g.pct_liver}%</strong></div>
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
                <p className="small mb-1"><strong>Gene Therapy Status:</strong> {g.gene_therapy_status}</p>
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
              PBD Overview — {defs.pbd_overview?.genes_in_atlas} Genes · {defs.pbd_overview?.most_common_gene}
            </div>
            <div className="card-body">
              <p className="small mb-1"><strong>Full Name:</strong> {defs.pbd_overview?.full_name}</p>
              <p className="small mb-1"><strong>Spectrum:</strong> {defs.pbd_overview?.spectrum_name}</p>
              <p className="small mb-1"><strong>Incidence:</strong> {defs.pbd_overview?.collective_incidence}</p>
              <p className="small mb-1"><strong>Most Common Gene:</strong> {defs.pbd_overview?.most_common_gene}</p>
              <p className="small mb-0"><strong>NBS Note:</strong> {defs.pbd_overview?.nbs_note}</p>
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
