'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR   = '#e65100';  // deep orange — FAO / fat oxidation / energy
const LIGHT   = '#fff3e0';
const COLOR2  = '#c62828';  // KD absolute CI (danger)
const COLOR3  = '#1565c0';  // KD tolerated (safe)
const COLOR4  = '#2e7d32';  // MCT treatment (treatment)
const COLOR5  = '#6a1b9a';  // carnitine treatment
const COLOR6  = '#37474f';  // NBS detected
const COLOR7  = '#880e4f';  // VPA risk
const COLOR8  = '#004d40';  // LCHAD maternal
const COLOR9  = '#bf360c';  // fasting forbidden

const CLASS_COLORS = {
  acyl_coa_dehydrogenase_medium_chain:    '#e65100',
  acyl_coa_dehydrogenase_very_long_chain: '#b71c1c',
  mtp_alpha_lchad_hydratase:              '#1565c0',
  mtp_beta_thiolase:                      '#1a237e',
  short_chain_hydroxyacyl_coa_dehydrogenase: '#6a1b9a',
  acyl_coa_dehydrogenase_short_chain:     '#558b2f',
  carnitine_palmitoyltransferase_1a_liver:'#ef6c00',
  carnitine_palmitoyltransferase_2:       '#bf360c',
  carnitine_acylcarnitine_translocase:    '#880e4f',
  octn2_carnitine_transporter:            '#00695c',
};

const CLASS_LABELS = {
  acyl_coa_dehydrogenase_medium_chain:    'Acyl-CoA dehydrogenase — Medium chain (C6–C12): MCAD',
  acyl_coa_dehydrogenase_very_long_chain: 'Acyl-CoA dehydrogenase — Very long chain (≥C14): VLCAD',
  mtp_alpha_lchad_hydratase:              'MTP alpha subunit — LCHAD + 2-enoyl-CoA hydratase (long-chain)',
  mtp_beta_thiolase:                      'MTP beta subunit — Long-chain 3-ketoacyl-CoA thiolase',
  short_chain_hydroxyacyl_coa_dehydrogenase: 'Short-chain 3-hydroxyacyl-CoA dehydrogenase — SCHAD (hyperinsulinism)',
  acyl_coa_dehydrogenase_short_chain:     'Acyl-CoA dehydrogenase — Short chain (C4–C6): SCAD (debated significance)',
  carnitine_palmitoyltransferase_1a_liver:'CPT1A — Carnitine palmitoyltransferase 1A (liver, rate-limiting CPT step)',
  carnitine_palmitoyltransferase_2:       'CPT2 — Carnitine palmitoyltransferase 2 (IMM inner face, 3 forms)',
  carnitine_acylcarnitine_translocase:    'CACT — Carnitine-acylcarnitine translocase (IMM antiporter, neonatal emergency)',
  octn2_carnitine_transporter:            'OCTN2 — Organic cation/carnitine transporter 2 (primary carnitine deficiency)',
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

export default function FaoAtlasPage() {
  const [tab, setTab]           = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]         = useState(null);
  const [loading, setLoading]   = useState(true);
  const [err, setErr]           = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/fao-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/fao-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/fao-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center"><div className="spinner-border" style={{ color: COLOR }} /><p className="mt-2">Loading FAO-Atlas…</p></div>;
  if (err)     return <div className="p-4 alert alert-danger">Error: {err}</div>;

  const ac = overview?.aggregate_clinical || {};
  const genes = breakdown?.genes || [];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${COLOR} 0%, #bf360c 100%)` }}>
        <h4 className="mb-1 fw-bold">&#x1f9ec; FAO-Atlas — Complete 10-Gene Mitochondrial Fatty Acid Oxidation Disorders Atlas</h4>
        <p className="mb-1 small">
          ACADM(MCAD) · ACADVL(VLCAD) · HADHA(LCHAD/MTP-α) · HADHB(MTP-β) · HADH(SCHAD) ·
          ACADS(SCAD) · CPT1A · CPT2 · SLC25A20(CACT) · SLC22A5(OCTN2) |&nbsp;
          400-patient aggregate (10×40, seeds 813–822)
        </p>
        <p className="mb-0 small opacity-75">
          KD-ABSOLUTE-CI-Long-Chain-FAO · MCT-Diet-VLCAD-LCHAD-MTP-CACT · Fasting-FORBIDDEN ·
          VPA-HIGH-RISK-ALL · MCAD-Most-Common-NBS-C8 · LCHAD-Maternal-AFLP-HELLP-79pct ·
          SCHAD-Hyperinsulinism-GDH-Disinhibition · CPT2-Neonatal-Brain-Malformations · OCTN2-LCarnitine-Dramatic-Response
        </p>
      </div>

      {/* KPI bar */}
      <div className="row g-2 mb-3">
        <KPI label="Genes" value={overview?.n_genes} color={COLOR} />
        <KPI label="Patients" value={overview?.n_patients} color={COLOR} />
        <KPI label="Hypoglycaemia" value={`${ac.hypoglycaemia_pct}%`} color={COLOR2} />
        <KPI label="HCM" value={`${ac.hcm_pct}%`} color={COLOR2} />
        <KPI label="Hepatopathy" value={`${ac.hepatopathy_pct}%`} color={COLOR} />
        <KPI label="Myopathy" value={`${ac.myopathy_pct}%`} color={COLOR} />
        <KPI label="Rhabdomyolysis" value={`${ac.rhabdomyolysis_pct}%`} color={COLOR7} />
        <KPI label="Retinopathy" value={`${ac.retinopathy_pct}%`} color={COLOR5} />
        <KPI label="Hyperinsulinism" value={`${ac.hyperinsulinism_pct}%`} color={COLOR5} />
        <KPI label="Neonatal Crisis" value={`${ac.neonatal_crisis_pct}%`} color={COLOR2} />
        <KPI label="KD Absolute CI" value={`${ac.kd_absolute_ci_pct}%`} color={COLOR2} />
        <KPI label="MCT Treatment" value={`${ac.mct_treatment_pct}%`} color={COLOR4} />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && (
        <div className="row g-3">
          {/* Description */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold" style={{ background: LIGHT }}>Atlas Description</div>
              <div className="card-body">
                <p className="mb-0">{overview?.description}</p>
              </div>
            </div>
          </div>

          {/* Chain length classes */}
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: LIGHT }}>Chain-Length Classification</div>
              <div className="card-body">
                {Object.entries(overview?.chain_length_classes || {}).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <span className="badge me-2" style={{ background: COLOR }}>{k.replace(/_/g, ' ')}</span>
                    <small>{v}</small>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Carnitine cycle */}
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: LIGHT }}>Carnitine Cycle (CPT1A → CACT → CPT2)</div>
              <div className="card-body">
                {Object.entries(overview?.carnitine_cycle || {}).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <span className="badge me-2" style={{ background: '#6a1b9a' }}>{k}</span>
                    <small>{v}</small>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Aggregate phenotype bars */}
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: LIGHT }}>Aggregate Clinical Phenotypes (400 patients)</div>
              <div className="card-body">
                <BarRow label="Hypoglycaemia" pct={ac.hypoglycaemia_pct} color={COLOR2} />
                <BarRow label="Hepatopathy" pct={ac.hepatopathy_pct} color={COLOR} />
                <BarRow label="Myopathy" pct={ac.myopathy_pct} color={COLOR} />
                <BarRow label="HCM (cardiomyopathy)" pct={ac.hcm_pct} color={COLOR2} />
                <BarRow label="Rhabdomyolysis" pct={ac.rhabdomyolysis_pct} color={COLOR7} />
                <BarRow label="Encephalopathy" pct={ac.encephalopathy_pct} color={COLOR9} />
                <BarRow label="Lactic acidosis" pct={ac.lactic_acidosis_pct} color={COLOR9} />
                <BarRow label="Retinopathy (LCHAD/MTP)" pct={ac.retinopathy_pct} color={COLOR5} />
                <BarRow label="Hyperinsulinism (SCHAD)" pct={ac.hyperinsulinism_pct} color={COLOR5} />
                <BarRow label="Neonatal crisis" pct={ac.neonatal_crisis_pct} color={COLOR2} />
                <BarRow label="Epilepsy" pct={ac.epilepsy_pct} color={COLOR} />
                <BarRow label="NBS detected" pct={ac.nbs_detected_pct} color={COLOR4} />
              </div>
            </div>
          </div>

          {/* Drug CIs */}
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: '#ffebee' }}>Drug Contraindications</div>
              <div className="card-body">
                {Object.entries(overview?.drug_contraindications || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 p-2 rounded" style={{ background: '#fff8f6', borderLeft: `4px solid ${COLOR2}` }}>
                    <div className="fw-semibold small text-danger">{k.replace(/_/g, ' ').toUpperCase()}</div>
                    <div className="small">{v}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* MCT diet */}
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: '#e8f5e9' }}>MCT Diet — Bypass Strategy</div>
              <div className="card-body">
                <div className="mb-2"><strong>Mechanism:</strong> {overview?.mct_diet_rationale?.mechanism}</div>
                <div className="mb-2"><strong>Primary treatment for:</strong> {(overview?.mct_diet_rationale?.genes_where_primary_treatment || []).join(' · ')}</div>
                <div><strong>DHA supplement:</strong> {overview?.mct_diet_rationale?.dha_supplement}</div>
              </div>
            </div>
          </div>

          {/* NBS markers */}
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold" style={{ background: LIGHT }}>NBS Acylcarnitine Markers</div>
              <div className="card-body">
                {Object.entries(overview?.nbs_markers || {}).map(([gene, marker]) => (
                  <div key={gene} className="mb-1 d-flex">
                    <span className="badge me-2" style={{ background: CLASS_COLORS[genes.find(g=>g.gene===gene)?.gene_class] || COLOR, minWidth: 80 }}>{gene}</span>
                    <small>{marker}</small>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Maternal risk */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold" style={{ background: '#fce4ec' }}>Maternal Risk (Pregnancy)</div>
              <div className="card-body row g-2">
                {Object.entries(overview?.maternal_risk || {}).map(([k, v]) => (
                  <div key={k} className="col-md-6">
                    <div className="p-2 rounded" style={{ background: '#fff', borderLeft: `4px solid ${COLOR8}` }}>
                      <div className="fw-semibold small" style={{ color: COLOR8 }}>{k.replace(/_/g, ' ')}</div>
                      <div className="small">{v}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Key rules */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold" style={{ background: LIGHT }}>Key Clinical Rules</div>
              <div className="card-body row g-2">
                {Object.entries(overview?.key_rules || {}).map(([k, v]) => (
                  <div key={k} className="col-md-6">
                    <div className="p-2 rounded" style={{ background: '#fff3e0', borderLeft: `4px solid ${COLOR}` }}>
                      <div className="fw-semibold small" style={{ color: COLOR }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                      <div className="small">{v}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold" style={{ background: LIGHT }}>All 10 FAO Genes — Summary Table</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Gene</th><th>Alias</th><th>aa/kDa</th><th>Locus</th>
                    <th>Phenotype</th><th>KD CI</th><th>MCT</th><th>L-Carn</th><th>NBS Marker</th>
                  </tr>
                </thead>
                <tbody>
                  {genes.map(g => (
                    <tr key={g.gene}>
                      <td>
                        <span className="badge" style={{ background: CLASS_COLORS[g.gene_class] || COLOR }}>{g.gene}</span>
                      </td>
                      <td style={{ fontSize: '0.76rem', maxWidth: 160, whiteSpace: 'normal' }}>{g.alias}</td>
                      <td style={{ fontSize: '0.76rem' }}>{g.aa} / {g.kDa}</td>
                      <td style={{ fontSize: '0.76rem' }}>{g.locus}</td>
                      <td style={{ fontSize: '0.73rem', maxWidth: 200, whiteSpace: 'normal' }}>{g.phenotype}</td>
                      <td className="text-center">
                        {g.kd_absolute_ci ? (
                          <span className="badge bg-danger">ABSOLUTE CI</span>
                        ) : g.kd_tolerated ? (
                          <span className="badge bg-success">Tolerated</span>
                        ) : (
                          <span className="badge bg-secondary">—</span>
                        )}
                      </td>
                      <td className="text-center">
                        {g.mct_treatment ? <span className="badge bg-success">PRIMARY Tx</span> : <span className="text-muted">—</span>}
                      </td>
                      <td className="text-center">
                        {g.lcarnitine_treatment ? <span className="badge bg-primary">Tx</span> : <span className="text-muted">—</span>}
                      </td>
                      <td style={{ fontSize: '0.73rem', maxWidth: 180, whiteSpace: 'normal' }}>{g.nbs_marker?.split('—')[0]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && (
        <div className="row g-3">
          {genes.map(g => (
            <div key={g.gene} className="col-12 col-xl-6">
              <div className="card shadow-sm h-100" style={{ borderTop: `4px solid ${CLASS_COLORS[g.gene_class] || COLOR}` }}>
                <div className="card-header d-flex align-items-center gap-2" style={{ background: LIGHT }}>
                  <span className="badge fs-6" style={{ background: CLASS_COLORS[g.gene_class] || COLOR }}>{g.gene}</span>
                  <span className="small fw-semibold">{g.alias}</span>
                  <span className="ms-auto small text-muted">{g.aa} · {g.locus}</span>
                </div>
                <div className="card-body">
                  <div className="mb-2">
                    <span className="badge" style={{ background: CLASS_COLORS[g.gene_class] || COLOR }}>
                      {CLASS_LABELS[g.gene_class] || g.gene_class}
                    </span>
                  </div>
                  <p className="small mb-2"><strong>Phenotype:</strong> {g.phenotype}</p>
                  <div className="mb-2 p-2 rounded" style={{ background: '#fff8f6', fontSize: '0.76rem' }}>
                    <strong>Hallmark:</strong> {g.hallmark?.slice(0, 400)}{g.hallmark?.length > 400 ? '…' : ''}
                  </div>

                  {/* Phenotype rates */}
                  <div className="row g-1 mb-2">
                    {[
                      ['Hypoglycaemia', g.hypoglycaemia_pct],
                      ['HCM', g.hcm_pct],
                      ['Hepatopathy', g.hepatopathy_pct],
                      ['Myopathy', g.myopathy_pct],
                      ['Rhabdomyolysis', g.rhabdomyolysis_pct],
                      ['Retinopathy', g.retinopathy_pct],
                      ['Hyperinsulinism', g.hyperinsulinism_pct],
                      ['Neonatal Crisis', g.neonatal_crisis_pct],
                      ['Epilepsy', g.epilepsy_pct],
                      ['NBS Detected', g.nbs_detected_pct],
                    ].map(([lbl, pct]) => pct > 0 && (
                      <div key={lbl} className="col-6">
                        <div style={{ fontSize: '0.72rem' }} className="d-flex justify-content-between">
                          <span>{lbl}</span><span className="fw-bold">{pct}%</span>
                        </div>
                        <div className="progress" style={{ height: '5px' }}>
                          <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: CLASS_COLORS[g.gene_class] || COLOR }} />
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Drug flags */}
                  <div className="d-flex flex-wrap gap-1 mb-2">
                    {g.kd_absolute_ci && <span className="badge bg-danger">KD ABSOLUTE CI</span>}
                    {g.kd_tolerated && <span className="badge bg-success">KD Tolerated</span>}
                    {g.mct_treatment && <span className="badge bg-success">MCT = PRIMARY Tx</span>}
                    {g.lcarnitine_treatment && <span className="badge bg-primary">L-Carnitine Tx</span>}
                    {g.fasting_forbidden && <span className="badge bg-danger">Fasting FORBIDDEN</span>}
                    {g.nbs_detected && <span className="badge bg-info text-dark">NBS Detected</span>}
                  </div>

                  <div className="small mb-1 p-1 rounded" style={{ background: '#fff3e0', borderLeft: `3px solid ${COLOR7}` }}>
                    <strong>VPA:</strong> {g.vpa_risk}
                  </div>
                  <div className="small p-1 rounded" style={{ background: '#e8f5e9', borderLeft: '3px solid #2e7d32' }}>
                    <strong>Acute Tx:</strong> {g.acute_treatment?.slice(0, 200)}{g.acute_treatment?.length > 200 ? '…' : ''}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && (
        <div>
          {(Array.isArray(defs) ? defs : []).map((d, i) => (
            <div key={i} className="card shadow-sm mb-3">
              <div className="card-header fw-bold" style={{ background: LIGHT, color: COLOR }}>{d.term}</div>
              <div className="card-body">
                <p className="mb-0 small">{d.definition}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-4 text-muted small text-center">
        FAO-Atlas · 10 nuclear-encoded genes · 400-patient aggregate · seeds 813–822 · 3 endpoints /api/fao-atlas/overview|breakdown|definitions ·{' '}
        <Link href="/" className="text-muted">← Dashboard</Link>
      </div>
    </div>
  );
}
