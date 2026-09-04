'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#1a237e';  // deep indigo — BH4 cofactor ring structure
const LIGHT  = '#e8eaf6';
const COLOR2 = '#b71c1c';  // critical / folinic acid MANDATORY
const COLOR3 = '#0d47a1';  // NBS / biomarker
const COLOR4 = '#e65100';  // warning / NBS miss
const COLOR5 = '#1b5e20';  // treatable / good response
const COLOR6 = '#37474f';  // gene class / struct
const COLOR7 = '#4a148c';  // chaperone / DNAJC12

const SUBGROUP_COLORS = {
  'BH4 de novo synthesis (GCH1 · PTS · SPR)':        '#0d47a1',
  'BH4 regeneration auxiliary (PCBD1 · QDPR)':        '#b71c1c',
  'BH4 cofactor utilisation / chaperone (DNAJC12)':   '#4a148c',
};

const GENE_COLORS = {
  GCH1:    '#1a237e',
  PTS:     '#1565c0',
  QDPR:    '#b71c1c',
  SPR:     '#e65100',
  PCBD1:   '#2e7d32',
  DNAJC12: '#4a148c',
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

export default function BH4AtlasPage() {
  const [tab, setTab]             = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [err, setErr]             = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/bh4-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/bh4-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/bh4-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return (
    <div className="d-flex align-items-center justify-content-center" style={{ minHeight: '60vh' }}>
      <div className="spinner-border" style={{ color: COLOR }} role="status" />
      <span className="ms-3 text-muted">Loading BH4-Atlas&#x2026;</span>
    </div>
  );
  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;

  const genes = breakdown?.genes || [];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="rounded-3 px-4 py-3 mb-3 text-white"
           style={{ background: `linear-gradient(135deg, ${COLOR} 0%, #283593 100%)` }}>
        <h3 className="mb-1 fw-bold">&#x1f9ec; BH4-Atlas</h3>
        <div style={{ fontSize: '0.85rem', opacity: 0.9 }}>
          Complete 6-Gene Tetrahydrobiopterin Disorders Atlas &nbsp;&#xb7;&nbsp;
          GCH1 &nbsp;&#xb7;&nbsp; PTS &nbsp;&#xb7;&nbsp; QDPR &nbsp;&#xb7;&nbsp; SPR &nbsp;&#xb7;&nbsp; PCBD1 &nbsp;&#xb7;&nbsp; DNAJC12 &nbsp;&#xb7;&nbsp;
          240-patient aggregate cohort (6 &times; 40, seeds 920&#x2013;925)
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
            <KPI label="BH4-Loading Responsive" value={overview.n_bh4_loading_responsive} color={COLOR5} />
            <KPI label="NBS Missed (SPR+DRD)" value={overview.n_nbs_missed} color={COLOR4} />
            <KPI label="Seeds" value={`${overview.seeds?.[0]}&#x2013;${overview.seeds?.[overview.seeds.length-1]}`} color={COLOR6} />
            <KPI label="Folinic Acid Mandatory" value="QDPR only" color={COLOR2} />
          </div>

          {/* Alert: NBS Miss */}
          <div className="alert mb-3" style={{ backgroundColor: '#fff3e0', borderLeft: `4px solid ${COLOR4}`, fontSize: '0.85rem' }}>
            <strong style={{ color: COLOR4 }}>&#x26a0;&#xfe0f; NBS Diagnostic Gap:</strong>&nbsp;
            GCH1-AD-DRD and SPR have <strong>NORMAL Phe</strong> on newborn screening and are <strong>missed by standard NBS</strong>.
            Any unexplained childhood dystonia, motor disorder, or ataxia with normal NBS must have CSF neurotransmitter analysis (HVA + 5-HIAA).
          </div>

          {/* Subgroups */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ backgroundColor: LIGHT, color: COLOR }}>
              BH4 Gene Subgroups — Pathway Position
            </div>
            <div className="card-body">
              <div className="row">
                {Object.entries(overview.gene_subgroups || {}).map(([group, genes_list]) => (
                  <div key={group} className="col-12 col-md-4 mb-2">
                    <div className="p-2 rounded" style={{ backgroundColor: '#f8f9fa', borderLeft: `4px solid ${SUBGROUP_COLORS[group] || COLOR}` }}>
                      <div className="fw-semibold small" style={{ color: SUBGROUP_COLORS[group] || COLOR }}>{group}</div>
                      <div className="text-muted small">{genes_list.join(' &#xb7; ')}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Critical Rules */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold" style={{ backgroundColor: '#fce4ec', color: COLOR2 }}>
              &#x26a0;&#xfe0f; Critical Clinical Rules (BH4 Disorders)
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
              Gene Summary &#x2014; Mean Age at Diagnosis
            </div>
            <div className="card-body">
              {(overview.gene_summary || []).map(g => (
                <div key={g.gene} className="mb-2">
                  <div className="d-flex justify-content-between" style={{ fontSize: '0.8rem' }}>
                    <span className="fw-semibold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</span>
                    <span className="text-muted">{g.mean_age_dx_y} yr dx</span>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: '#555', marginBottom: '2px' }}>{g.phenotype?.substring(0, 120)}{g.phenotype?.length > 120 ? '...' : ''}</div>
                  <div className="progress" style={{ height: '6px' }}>
                    <div className="progress-bar"
                         style={{ width: `${Math.min(100, g.mean_age_dx_y * 8)}%`, backgroundColor: GENE_COLORS[g.gene] || COLOR }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* NBS note */}
          <div className="alert alert-secondary small mb-0">
            <strong>NBS Note:</strong> {overview.nbs_note}
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
                <th>Locus</th>
                <th>Size</th>
                <th>BH4 Role</th>
                <th>Pts</th>
                <th>Dx (yr)</th>
                <th>NBS Detected</th>
                <th>Pterin Hallmark</th>
                <th>Folinic Acid</th>
                <th>Sapropterin</th>
                <th>Key Treatment</th>
              </tr>
            </thead>
            <tbody>
              {genes.map(g => {
                // Determine pterin hallmark per gene
                const pterinMap = {
                  GCH1:    'Neopterin LOW + Biopterin LOW',
                  PTS:     'Neopterin VERY HIGH + Biopterin LOW + Primapterin',
                  QDPR:    'Biopterin HIGH + Neopterin NORMAL',
                  SPR:     'Urine: near-normal; CSF: Biopterin HIGH + Sepiapterin',
                  PCBD1:   'Primapterin in urine + Neopterin NORMAL',
                  DNAJC12: 'ALL NORMAL (diagnostic trap)',
                };
                const folinicMap = {
                  GCH1: false, PTS: false, QDPR: true, SPR: true, PCBD1: false, DNAJC12: false,
                };
                const sapropterinMap = {
                  GCH1: 'AR only', PTS: 'Yes', QDPR: 'Yes', SPR: 'NO', PCBD1: 'Short course', DNAJC12: 'Yes (primary)',
                };
                const nbsMap = {
                  GCH1: 'AR: Yes / AD-DRD: NO',
                  PTS: 'Yes', QDPR: 'Yes', SPR: 'NO (Phe NORMAL)', PCBD1: 'Yes', DNAJC12: 'Yes',
                };
                const treatMap = {
                  GCH1: 'L-DOPA 1-2mg/kg/d (DRD) / BH4+L-DOPA+5-HTP (AR)',
                  PTS: 'BH4 + L-DOPA + 5-HTP (classic); BH4 alone (peripheral)',
                  QDPR: 'BH4 + L-DOPA + 5-HTP + Folinic acid',
                  SPR: 'L-DOPA + 5-HTP + Folinic acid (NOT sapropterin)',
                  PCBD1: 'Usually none; short BH4 if Phe very high',
                  DNAJC12: 'Sapropterin ± L-DOPA + 5-HTP',
                };
                return (
                  <tr key={g.gene}>
                    <td className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</td>
                    <td>{g.locus}</td>
                    <td>{g.aa} / {g.kDa}</td>
                    <td style={{ maxWidth: '140px', fontSize: '0.7rem' }}>{g.gene_class?.substring(0, 80)}{g.gene_class?.length > 80 ? '...' : ''}</td>
                    <td>{g.n_patients}</td>
                    <td>{g.mean_age_dx_y}</td>
                    <td>
                      {nbsMap[g.gene]?.includes('NO')
                        ? <span className="badge" style={{ backgroundColor: COLOR4 }}>&#x26a0; {nbsMap[g.gene]}</span>
                        : <span className="badge bg-success">{nbsMap[g.gene]}</span>}
                    </td>
                    <td style={{ fontSize: '0.7rem', maxWidth: '150px' }}>{pterinMap[g.gene] || '—'}</td>
                    <td>
                      {folinicMap[g.gene]
                        ? <span className="badge" style={{ backgroundColor: COLOR2 }}>MANDATORY</span>
                        : <span className="text-muted">—</span>}
                    </td>
                    <td style={{ fontSize: '0.7rem' }}>{sapropterinMap[g.gene] || '—'}</td>
                    <td style={{ maxWidth: '160px', fontSize: '0.7rem' }}>{treatMap[g.gene] || '—'}</td>
                  </tr>
                );
              })}
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
                {g.gene} &#x2014; {g.alias?.split('—')[0]?.trim() || g.gene}
                <span className="float-end badge bg-light text-dark">{g.locus} &#xb7; {g.aa}</span>
              </div>
              <div className="card-body" style={{ fontSize: '0.82rem' }}>

                {/* QDPR folinic acid alert */}
                {g.gene === 'QDPR' && (
                  <div className="alert alert-danger py-1 px-2 mb-2" style={{ fontSize: '0.78rem' }}>
                    <strong>&#x26a0;&#xfe0f; FOLINIC ACID MANDATORY:</strong> qBH2 inhibits DHFR &#x2192; cerebral folate deficiency.
                    Must give 5-formyl-THF (leucovorin) 15-20 mg/day. Folic acid does NOT work.
                  </div>
                )}

                {/* SPR NBS miss alert */}
                {g.gene === 'SPR' && (
                  <div className="alert alert-warning py-1 px-2 mb-2" style={{ fontSize: '0.78rem' }}>
                    <strong>&#x26a0;&#xfe0f; NBS MISS:</strong> Phe is NORMAL &#x2192; SPR not detected by standard screening.
                    Diagnosis requires CSF analysis (HVA + 5-HIAA + biopterin). Sapropterin NOT indicated.
                  </div>
                )}

                {/* GCH1 DRD alert */}
                {g.gene === 'GCH1' && (
                  <div className="alert alert-warning py-1 px-2 mb-2" style={{ fontSize: '0.78rem' }}>
                    <strong>&#x26a0;&#xfe0f; AD-DRD NBS MISS:</strong> GCH1 AD (Segawa DRD) has NORMAL Phe.
                    Diurnal dystonia + dramatic L-DOPA response at 1-2 mg/kg/day = pathognomonic.
                  </div>
                )}

                {/* DNAJC12 normal pterin alert */}
                {g.gene === 'DNAJC12' && (
                  <div className="alert" style={{ backgroundColor: '#ede7f6', borderLeft: `3px solid ${COLOR7}`, fontSize: '0.78rem', padding: '4px 10px' }}>
                    <strong style={{ color: COLOR7 }}>&#x26a0;&#xfe0f; DIAGNOSTIC TRAP:</strong> Pterin profile NORMAL.
                    BH4-responsive HPA + normal pterins + low CSF HVA/5-HIAA = DNAJC12 until proven otherwise.
                  </div>
                )}

                <div className="row g-2 mt-1" style={{ fontSize: '0.78rem' }}>
                  <div className="col-12">
                    <strong>Phenotype:</strong> {g.phenotype}
                  </div>
                  <div className="col-12 col-md-6">
                    <strong>Severity Spectrum:</strong> {g.severity_spectrum}
                  </div>
                  <div className="col-12 col-md-6">
                    <strong>Key Biomarker:</strong> {g.key_biomarker?.substring(0, 200)}{g.key_biomarker?.length > 200 ? '...' : ''}
                  </div>
                  <div className="col-12">
                    <strong>Hallmarks:</strong>
                    <p className="mt-1 mb-1" style={{ whiteSpace: 'pre-line' }}>{g.hallmark}</p>
                  </div>
                  <div className="col-12">
                    <strong>Treatment:</strong> {g.diet_treatment}
                  </div>
                  <div className="col-12">
                    <strong style={{ color: COLOR2 }}>Critical CI / Non-Negotiable:</strong>{' '}
                    <span style={{ color: COLOR2 }}>{g.critical_ci}</span>
                  </div>
                  <div className="col-12">
                    <strong>NBS / Diagnostic:</strong> {g.nbs_marker}
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
                    <strong>Founder / Common Variant:</strong> {g.founder_variant}
                  </div>
                  <div className="col-12">
                    <strong>Key DDx:</strong> <span style={{ color: '#555' }}>{g.key_ddx}</span>
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
            <strong>{defs.bh4_overview?.full_name}</strong> &nbsp;|&nbsp;
            {defs.bh4_overview?.genes_in_atlas} genes &nbsp;|&nbsp;
            Incidence: {defs.bh4_overview?.collective_incidence?.substring(0, 120)}... &nbsp;|&nbsp;
            {defs.bh4_overview?.nbs_note}
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
