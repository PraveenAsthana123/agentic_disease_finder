'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Subunit Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#1a237e';   // deep indigo — nuclear CI atlas
const LIGHT  = '#e8eaf6';
const COLOR2 = '#1b5e20';   // dark green — structural subunits
const COLOR3 = '#4a148c';   // deep purple — assembly factors
const COLOR4 = '#b71c1c';   // dark red — absolute CIs
const COLOR5 = '#e65100';   // orange — hallmarks
const COLOR6 = '#006064';   // teal — modules
const COLOR7 = '#880e4f';   // dark pink — X-linked / NDUFA4 caveat

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

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        {title && <h6 className="fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>}
        {children}
      </div>
    </div>
  );
}

function Badge({ text, color = COLOR }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: '0.72rem' }}>
      {text}
    </span>
  );
}

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const sb   = data.series_breakdown || {};
  const mod  = data.module_architecture || {};
  const xlnk = data.x_linked_ci_genes || {};
  const coh  = data.cohort || {};
  const agg  = data.aggregate_clinical || {};
  const uci  = data.universal_absolute_ci || [];
  const acid = data.absolute_contraindicated || [];
  const hcaut= data.high_caution || [];
  const uman = data.universal_mandatory || [];
  const hall = data.hallmark_phenotypes || {};
  const cist = data.ci_structure || {};

  return (
    <>
      {/* Atlas banner */}
      <SectionCard title="CI-Subunit-Atlas — Complete 42-Gene Nuclear-Encoded Complex I Reference">
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">Complex I: </span>NADH:Ubiquinone Oxidoreductase — L-shaped 45-subunit ~980 kDa</div>
            <div><span className="fw-semibold">Total subunits: </span>{cist.total_subunits} ({cist.mtDNA_encoded} mtDNA + {cist.nuclear_encoded} nuclear)</div>
            <div><span className="fw-semibold">Assembly factors: </span>{cist.assembly_factors_nuclear} (all nuclear)</div>
            <div className="mt-1 text-success fw-semibold">✅ WES detects all 42 nuclear CI genes</div>
            <div className="text-muted small">{cist.note}</div>
          </div>
          <div className="col-12 col-md-6">
            <div className="alert alert-warning py-1 px-2 mb-2 small">
              <strong>NDUFA4 caveat:</strong> {data.ndufa4_caveat}
            </div>
            <div><span className="fw-semibold">Cohort: </span>{coh.total_patients?.toLocaleString()} patients ({coh.genes_included} genes × {coh.patients_per_gene})</div>
            <div><span className="fw-semibold">Seeds: </span>{coh.seed_range}</div>
          </div>
        </div>
      </SectionCard>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total CI Nuclear Genes" value="42" color={COLOR} />
        <KPI label="Structural Subunits" value="34" color={COLOR2} />
        <KPI label="Assembly Factors" value="8" color={COLOR3} />
        <KPI label="X-Linked Genes" value="2" color={COLOR7} />
        <KPI label="Aggregate Cohort" value={coh.total_patients?.toLocaleString()} color={COLOR5} />
        <KPI label="Leigh MRI %" value={`${agg.leigh_mri_pct}%`} color={COLOR4} />
        <KPI label="Mean CI Activity" value={`${agg.mean_ci_activity_pct}%`} color={COLOR6} />
        <KPI label="HCM (NDUFV2)" value={`${agg.hcm_pct}%`} color={COLOR5} />
      </div>

      {/* Series breakdown */}
      <SectionCard title="Subunit Series Breakdown (NDUFS / NDUFV / NDUFA / NDUFB / NDUFAF)" borderColor={COLOR2}>
        <div className="row g-2 small">
          {Object.entries(sb).map(([series, info]) => (
            <div key={series} className="col-12 col-sm-6 col-lg-4 col-xl-2dot4">
              <div className="card h-100 p-2" style={{ borderLeft: `3px solid ${series === 'NDUFAF' ? COLOR3 : COLOR2}` }}>
                <div className="fw-bold fs-6" style={{ color: series === 'NDUFAF' ? COLOR3 : COLOR2 }}>
                  {series} ({info.count})
                </div>
                <div className="text-muted small mb-2">{info.note}</div>
                <div className="d-flex flex-wrap gap-1">
                  {(info.genes || []).map((g, i) => (
                    <Badge key={i} text={g} color={series === 'NDUFAF' ? COLOR3 : COLOR2} />
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Module architecture */}
      <SectionCard title="CI Module Architecture — Peripheral Arm (N/Q) + Membrane Arm (PP/PD)" borderColor={COLOR6}>
        <div className="row g-3 small">
          {Object.entries(mod).map(([mname, minfo]) => (
            <div key={mname} className="col-12 col-md-6">
              <div className="card p-3 h-100" style={{ borderLeft: `3px solid ${COLOR6}` }}>
                <div className="fw-bold mb-1" style={{ color: COLOR6 }}>{mname.replace(/_/g, '-')}</div>
                <div className="text-muted mb-1">{minfo.location}</div>
                <div className="mb-2">{minfo.function}</div>
                <div className="d-flex flex-wrap gap-1 mb-2">
                  {(minfo.key_genes || []).map((g, i) => (
                    <Badge key={i} text={g} color={COLOR6} />
                  ))}
                </div>
                <div className="small fst-italic text-muted">{minfo.clinical_note}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* X-linked genes */}
      <SectionCard title="X-Linked CI Genes (Xq24 + Xp11.3)" borderColor={COLOR7}>
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ backgroundColor: '#fce4ec' }}>
              <div className="fw-bold" style={{ color: COLOR7 }}>NDUFA1 (Xq24)</div>
              <div>PP-module (MWFE) | Hemizygous males: severe Leigh | Carrier females: mosaic/mild, NO cardiac</div>
            </div>
          </div>
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ backgroundColor: '#fce4ec' }}>
              <div className="fw-bold" style={{ color: COLOR7 }}>NDUFB11 (Xp11.3)</div>
              <div>PP-module (ESSS) | Hemizygous males: lethal/severe | Female carriers: mosaic + CARDIAC FEATURES</div>
            </div>
          </div>
        </div>
        <div className="small text-muted mt-2">{xlnk.note}</div>
      </SectionCard>

      {/* Clinical aggregate */}
      <SectionCard title="Aggregate Clinical Profile (1,680 Patients, 42 Genes × 40)" borderColor={COLOR4}>
        <div className="row g-3 small">
          <div className="col-6 col-md-3">
            <div className="fw-bold text-danger">{agg.leigh_mri_pct}%</div>
            <div className="text-muted">Leigh MRI</div>
          </div>
          <div className="col-6 col-md-3">
            <div className="fw-bold text-danger">{agg.lactic_acidosis_pct}%</div>
            <div className="text-muted">Lactic Acidosis</div>
          </div>
          <div className="col-6 col-md-3">
            <div className="fw-bold" style={{ color: COLOR5 }}>{agg.hcm_pct}%</div>
            <div className="text-muted">HCM (NDUFV2-dominant)</div>
          </div>
          <div className="col-6 col-md-3">
            <div className="fw-bold" style={{ color: COLOR3 }}>{agg.peripheral_neuropathy_pct}%</div>
            <div className="text-muted">Peripheral Neuropathy (NDUFS1)</div>
          </div>
          <div className="col-6 col-md-3">
            <div className="fw-bold" style={{ color: COLOR6 }}>{agg.median_onset_months} mo</div>
            <div className="text-muted">Median Onset</div>
          </div>
          <div className="col-6 col-md-3">
            <div className="fw-bold" style={{ color: COLOR2 }}>{agg.mean_ci_activity_pct}%</div>
            <div className="text-muted">Mean CI Activity</div>
          </div>
        </div>
        <div className="small text-muted mt-2">{agg.note}</div>
      </SectionCard>

      {/* Universal absolute CIs */}
      <SectionCard title="Universal Absolute Contraindications (ALL 42 CI Genes)" borderColor={COLOR4}>
        <div className="row g-2 small">
          {uci.map((ci, i) => (
            <div key={i} className="col-12 col-md-6">
              <div className="p-2 rounded" style={{ backgroundColor: '#ffebee' }}>
                <div className="fw-bold text-danger">{ci.drug}</div>
                <div>{ci.mechanism}</div>
                <div className="text-muted">{ci.applies_to}</div>
              </div>
            </div>
          ))}
          {acid.map((ci, i) => (
            <div key={`acid-${i}`} className="col-12 col-md-6">
              <div className="p-2 rounded" style={{ backgroundColor: '#fff3e0' }}>
                <div className="fw-bold text-warning">{ci.drug}</div>
                <div>{ci.mechanism}</div>
                <div className="text-muted">{ci.applies_to}</div>
              </div>
            </div>
          ))}
        </div>
        {hcaut.length > 0 && (
          <>
            <div className="fw-semibold small mt-3 mb-1">High Caution:</div>
            <div className="d-flex flex-wrap gap-2 small">
              {hcaut.map((h, i) => (
                <div key={i} className="p-2 rounded" style={{ backgroundColor: '#fffde7' }}>
                  <span className="fw-bold">{h.drug}: </span>{h.mechanism}
                </div>
              ))}
            </div>
          </>
        )}
      </SectionCard>

      {/* Hallmark phenotypes */}
      <SectionCard title="Key Distinguishing Hallmarks by Gene" borderColor={COLOR5}>
        <div className="row g-2 small">
          {Object.entries(hall).map(([key, ph]) => (
            <div key={key} className="col-12 col-md-6 col-lg-4">
              <div className="card h-100 p-2" style={{ borderLeft: `3px solid ${COLOR5}` }}>
                <div className="fw-bold" style={{ color: COLOR5 }}>{ph.gene}</div>
                <div className="text-muted">{ph.note}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Mandatory protocol */}
      <SectionCard title="Universal Mandatory Protocol (ALL 42 CI Genes)" borderColor={COLOR2}>
        <ul className="mb-0 small">
          {uman.map((m, i) => <li key={i}>{m}</li>)}
        </ul>
      </SectionCard>
    </>
  );
}

// ── Tab: Subunit Table ────────────────────────────────────────────────────────
function SubunitTableTab({ data }) {
  const [filter, setFilter] = useState('all');
  const [series, setSeries] = useState('all');
  if (!data) return <p className="text-muted">Loading…</p>;
  const rows = data.genes || [];

  let filtered = rows;
  if (filter !== 'all') filtered = filtered.filter(r => r.gene_class === filter);
  if (series !== 'all') filtered = filtered.filter(r => r.subunit_series === series);

  const classColor = (cls, ser) => {
    if (cls === 'assembly_factor') return COLOR3;
    return COLOR2;
  };

  return (
    <>
      <div className="mb-3 d-flex gap-2 flex-wrap">
        {['all','structural_subunit','assembly_factor'].map(f => (
          <button key={f} className="btn btn-sm" onClick={() => setFilter(f)}
            style={{ backgroundColor: filter===f ? (f === 'assembly_factor' ? COLOR3 : COLOR) : '#eee', color: filter===f ? '#fff' : '#333' }}>
            {f === 'all' ? 'All 42 Genes' : f === 'structural_subunit' ? 'Structural Subunits (34)' : 'Assembly Factors (8)'}
          </button>
        ))}
        <span className="text-muted small align-self-center">|</span>
        {['all','S','V','A','B','AF'].map(s => (
          <button key={s} className="btn btn-sm btn-outline-secondary" onClick={() => setSeries(s)}
            style={{ fontWeight: series===s ? 'bold' : 'normal' }}>
            {s === 'all' ? 'All Series' : `NDUF${s === 'AF' ? 'AF' : s}`}
          </button>
        ))}
      </div>

      <div className="small text-muted mb-2">Showing {filtered.length} genes</div>

      <div className="table-responsive">
        <table className="table table-sm table-hover table-bordered small">
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Class</th><th>Series</th><th>kDa</th>
              <th>Module / Location</th><th>Chr</th><th>OMIM Gene</th>
              <th>CI Activity (mean%)</th><th>Leigh MRI%</th><th>HCM%</th>
              <th>Inheritance</th><th>Key Hallmark</th><th>Founder Variant</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => {
              const isAF = r.gene_class === 'assembly_factor';
              const isXL = r.inheritance?.includes('X-linked');
              const isNDUFA4 = r.gene === 'NDUFA4';
              return (
                <tr key={i}
                  className={isXL ? 'table-info' : isNDUFA4 ? 'table-warning' : isAF ? 'table-light' : ''}>
                  <td className="fw-bold" style={{ color: classColor(r.gene_class, r.subunit_series) }}>
                    {r.gene}
                    {isXL && <span className="ms-1 text-info fw-bold">X</span>}
                    {isNDUFA4 && <span className="ms-1 text-warning fw-bold">CIV!</span>}
                  </td>
                  <td>
                    <span className="badge" style={{ backgroundColor: isAF ? COLOR3 : COLOR2, fontSize: '0.65rem' }}>
                      {isAF ? 'AF' : 'Sub'}
                    </span>
                  </td>
                  <td className="fw-semibold">{r.subunit_series === 'AF' ? 'AF' : `NDUF${r.subunit_series}`}</td>
                  <td className="text-nowrap">{r.kDa || '—'}</td>
                  <td style={{ maxWidth: 200 }} className="small">{r.ci_module}</td>
                  <td className="text-nowrap">{r.chromosome}</td>
                  <td>{r.omim_gene}</td>
                  <td className="text-center">
                    <span style={{ color: COLOR4 }}>{r.ci_activity_mean_pct}%</span>
                  </td>
                  <td className="text-center">{r.leigh_mri_pct}%</td>
                  <td className="text-center" style={{ color: r.hcm_pct > 30 ? COLOR5 : 'inherit' }}>
                    {r.hcm_pct}%{r.hcm_pct > 30 ? ' ⬆' : ''}
                  </td>
                  <td className="small">{r.inheritance?.includes('X-linked') ? <span className="fw-bold text-info">X-linked</span> : r.inheritance}</td>
                  <td style={{ maxWidth: 200 }} className="small">{r.hallmark}</td>
                  <td style={{ maxWidth: 180 }} className="small text-primary">{r.founder_variant}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="small mt-1 d-flex gap-3 flex-wrap">
        <span><span className="badge me-1" style={{ backgroundColor: '#cfe2ff', color: '#333' }}>X</span>= X-linked gene</span>
        <span><span className="badge me-1" style={{ backgroundColor: '#fff3cd', color: '#333' }}>CIV!</span>= NDUFA4 is actually a CIV (not CI) subunit</span>
        <span><span className="badge me-1" style={{ backgroundColor: COLOR3, color: '#fff' }}>AF</span>= Assembly factor (not in mature CI)</span>
      </div>
    </>
  );
}

// ── Tab: Clinical Atlas ───────────────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const rows = data.genes || [];
  const structural = rows.filter(r => r.gene_class === 'structural_subunit');
  const assembly   = rows.filter(r => r.gene_class === 'assembly_factor');
  const xlinked    = rows.filter(r => r.inheritance?.includes('X-linked'));
  const ndufa4     = rows.find(r => r.gene === 'NDUFA4');

  return (
    <>
      {/* CI subunit vs assembly factor comparison */}
      <SectionCard title="Structural Subunits vs Assembly Factors — Clinical Implications" borderColor={COLOR2}>
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div className="fw-bold mb-2" style={{ color: COLOR2 }}>Structural Subunits (34 genes)</div>
            <div>• Loss = absent or severely reduced CI in mature holocomplex</div>
            <div>• BN-PAGE: absent CI band (scaffold-loss) OR sub-assembly intermediates (junction mutations)</div>
            <div>• Standard phenotype: isolated CI deficiency 5–22% (CII/CIII/CIV NORMAL)</div>
            <div>• WES detects all 34 structural subunits</div>
            <div className="mt-1 fw-semibold">Notable hallmarks:</div>
            <div>• <strong>NDUFS4:</strong> olfactory bulb MRI 52-65% (PATHOGNOMONIC)</div>
            <div>• <strong>NDUFV1:</strong> leukodystrophy 40-50%</div>
            <div>• <strong>NDUFV2:</strong> HCM 80% (highest cardiac rate)</div>
            <div>• <strong>NDUFS1:</strong> peripheral neuropathy 50%</div>
            <div>• <strong>NDUFB3:</strong> first nuclear CI mutation published (Andreu 1999)</div>
          </div>
          <div className="col-12 col-md-6">
            <div className="fw-bold mb-2" style={{ color: COLOR3 }}>Assembly Factors (8 genes)</div>
            <div>• Not present in mature CI holocomplex (transient scaffolds/enzymes)</div>
            <div>• Loss = CI biogenesis arrested at specific intermediate</div>
            <div>• Same CI deficiency phenotype as structural subunits</div>
            <div>• WES detects all 8 NDUFAF genes</div>
            <div className="mt-1 fw-semibold">Notable hallmarks:</div>
            <div>• <strong>NDUFAF1:</strong> obligate ACAD9 partner (CIA30); NO riboflavin response (unlike ACAD9)</div>
            <div>• <strong>NDUFAF2:</strong> LATER-ONSET (4–6 yr vs infantile); Moroccan+Ashkenazi founders</div>
            <div>• <strong>NDUFAF6:</strong> only CI AF with enzymatic 2OG-Fe dioxygenase activity</div>
            <div>• <strong>NDUFAF7:</strong> only CI AF that is an arginine methyltransferase</div>
          </div>
        </div>
      </SectionCard>

      {/* NDUFA4 CIV warning */}
      <SectionCard title="NDUFA4 — Critical: CIV (NOT CI) Subunit" borderColor={COLOR7}>
        <div className="small">
          <div className="alert alert-warning mb-2">
            <strong>NDUFA4 encodes a 14th Complex IV (COX) subunit</strong>, not a Complex I subunit.
            It was historically listed as a CI subunit before Balsa et al. (2012, Cell Metab) demonstrated
            it is a CIV structural subunit.
          </div>
          {ndufa4 && (
            <div className="row g-2">
              <div className="col-12 col-md-6">
                <div><span className="fw-semibold">Gene: </span>NDUFA4 (OMIM {ndufa4.omim_gene})</div>
                <div><span className="fw-semibold">Chromosome: </span>{ndufa4.chromosome}</div>
                <div><span className="fw-semibold">Disease: </span>{ndufa4.disease_summary}</div>
              </div>
              <div className="col-12 col-md-6">
                <div className="text-danger fw-semibold">Biochemistry: COX deficiency (CIV↓, CI NORMAL)</div>
                <div>Clinically resembles SURF1/COX10/COX15 (CIV-Leigh), NOT standard isolated CI-Leigh</div>
                <div className="text-muted">{ndufa4.key_ddx}</div>
              </div>
            </div>
          )}
        </div>
      </SectionCard>

      {/* X-linked CI genes */}
      <SectionCard title="X-Linked CI Genes — NDUFA1 and NDUFB11" borderColor={COLOR7}>
        <div className="row g-3 small">
          {xlinked.map((g, i) => (
            <div key={i} className="col-12 col-md-6">
              <div className="card p-3 h-100" style={{ borderLeft: `4px solid ${COLOR7}` }}>
                <div className="fw-bold fs-6 mb-1" style={{ color: COLOR7 }}>{g.gene} ({g.chromosome})</div>
                <div className="mb-1">{g.ci_module}</div>
                <div className="mb-2 fst-italic">{g.disease_summary}</div>
                <div className="fw-semibold text-danger">Inheritance: {g.inheritance}</div>
                <div className="text-muted mt-1">{g.hallmark}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Assembly factor sequence */}
      <SectionCard title="CI Assembly Factor Pathway (NDUFAF1–8) — Temporal Sequence" borderColor={COLOR3}>
        <div className="small">
          <div className="mb-2 text-muted">CI biogenesis follows a defined temporal order; assembly factors act at specific intermediates then are released from the mature holocomplex.</div>
          <div className="table-responsive">
            <table className="table table-sm table-bordered small mb-0">
              <thead style={{ backgroundColor: COLOR3, color: '#fff' }}>
                <tr>
                  <th>Assembly Factor</th><th>Stage</th><th>Function</th><th>Key Feature</th><th>Disease</th>
                </tr>
              </thead>
              <tbody>
                {assembly.map((af, i) => (
                  <tr key={i}>
                    <td className="fw-bold" style={{ color: COLOR3 }}>{af.gene}</td>
                    <td>{af.ci_module?.includes('early') ? 'Early' : af.ci_module?.includes('late') ? 'Late' : 'Mid'}</td>
                    <td style={{ maxWidth: 180 }}>{af.ci_module}</td>
                    <td style={{ maxWidth: 180 }} className="small">{af.hallmark}</td>
                    <td style={{ maxWidth: 150 }} className="small">{af.disease_summary}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </SectionCard>

      {/* Cross-gene drug safety */}
      <SectionCard title="Cross-Gene Drug Safety Summary" borderColor={COLOR4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead className="table-dark">
              <tr><th>Drug/Class</th><th>Structural Subunits (34)</th><th>Assembly Factors (8)</th><th>Mechanism</th></tr>
            </thead>
            <tbody>
              <tr><td className="fw-bold text-danger">Metformin</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td>Direct CI inhibitor — fatal lactic acidosis</td></tr>
              <tr><td className="fw-bold text-danger">VPA (Valproate)</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td>CoA sequestration + CI inhibition + POLG block</td></tr>
              <tr><td className="fw-bold text-danger">Propofol</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td>PRIS — uncouples OXPHOS; fatal cardiac failure</td></tr>
              <tr><td className="fw-bold text-danger">Linezolid</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td>Blocks mt-23S rRNA → halts 7 mt-ND subunit synthesis → CI further impaired</td></tr>
              <tr><td className="fw-bold text-danger">Chloramphenicol</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td>Inhibits mt-ribosome peptidyl transferase</td></tr>
              <tr><td className="fw-bold text-danger">Ketogenic Diet</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td>Forces NADH → β-oxidation; CI cannot re-oxidise; metabolic crisis</td></tr>
              <tr><td className="fw-bold text-warning">Phenobarbital</td><td className="text-center text-warning">High caution</td><td className="text-center text-warning">High caution</td><td>Secondary CI inhibitor; prefer LEV first-line</td></tr>
              <tr><td className="fw-bold text-warning">Amiodarone</td><td className="text-center text-warning">Avoid</td><td className="text-center text-warning">Avoid</td><td>OXPHOS inhibitor; especially hazardous if HCM (NDUFV2)</td></tr>
              <tr className="table-success"><td className="fw-bold">LEV (preferred)</td><td className="text-center">Preferred AED</td><td className="text-center">Preferred AED</td><td>Renal excretion; no CYP450; no mito toxicity</td></tr>
              <tr className="table-success"><td className="fw-bold">Succinate</td><td className="text-center">Level C</td><td className="text-center">Level C</td><td>CII bypass — provides electrons to ubiquinol pool bypassing failed CI</td></tr>
              <tr className="table-success"><td className="fw-bold">Riboflavin (B2)</td><td className="text-center">Level C (CI-specific)</td><td className="text-center">Level C</td><td>FMN at NDUFV1 N-module; partial benefit in missense alleles</td></tr>
            </tbody>
          </table>
        </div>
        <div className="alert alert-warning py-1 px-2 mt-2 small mb-0">
          <strong>NDUFA4 note:</strong> NDUFA4 = CIV (COX) deficiency — drug safety profile follows CIV (SURF1-like), not standard CI-Leigh. Metformin/VPA/Propofol/KD absolute CIs still apply (all OXPHOS disease).
        </div>
      </SectionCard>

      {/* See also */}
      <SectionCard title="Cross-Atlas Navigation" borderColor={COLOR}>
        <div className="d-flex flex-wrap gap-2 small">
          <Link href="/mt-genome-atlas" className="btn btn-sm btn-outline-primary">MT-Genome-Atlas (mtDNA CI: MT-ND1–6, ND4L)</Link>
          <Link href="/mt-trna-atlas" className="btn btn-sm btn-outline-secondary">MT-tRNA-Atlas (22 tRNA genes)</Link>
          <Link href="/mtrnr1" className="btn btn-sm btn-outline-secondary">MT-RNR1 (12S rRNA AISNHL)</Link>
          <Link href="/mtrnr2" className="btn btn-sm btn-outline-secondary">MT-RNR2 (16S rRNA)</Link>
          <Link href="/ndufs4" className="btn btn-sm btn-outline-success">NDUFS4 (olfactory Leigh)</Link>
          <Link href="/ndufv1" className="btn btn-sm btn-outline-success">NDUFV1 (leukodystrophy)</Link>
          <Link href="/ndufv2" className="btn btn-sm btn-outline-success">NDUFV2 (HCM 80%)</Link>
          <Link href="/ndufb3" className="btn btn-sm btn-outline-success">NDUFB3 (first nuclear CI mutation)</Link>
          <Link href="/ndufaf2" className="btn btn-sm btn-outline-success">NDUFAF2 (later-onset)</Link>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const entries = Object.entries(data);
  const half = Math.ceil(entries.length / 2);
  const col1 = entries.slice(0, half);
  const col2 = entries.slice(half);

  return (
    <>
      <SectionCard title="CI-Subunit-Atlas — Key Terms & Definitions" borderColor={COLOR}>
        <div className="row g-2">
          {[col1, col2].map((col, ci) => (
            <div key={ci} className="col-12 col-md-6">
              {col.map(([term, def], i) => (
                <div key={i} className="mb-3">
                  <div className="fw-semibold small" style={{ color: COLOR }}>{term.replace(/_/g, ' ')}</div>
                  <div className="small text-muted">{def}</div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </SectionCard>
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function CiSubunitAtlasPage() {
  const [tab,  setTab]  = useState(0);
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defn, setDefn] = useState(null);
  const [err,  setErr]  = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const [r1, r2, r3] = await Promise.all([
          fetch(`${API}/api/ci-subunit-atlas/overview`),
          fetch(`${API}/api/ci-subunit-atlas/breakdown`),
          fetch(`${API}/api/ci-subunit-atlas/definitions`),
        ]);
        setOv(await r1.json());
        setBd(await r2.json());
        setDefn(await r3.json());
      } catch (e) { setErr(e.message); }
    };
    load();
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <div>
          <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
            🧬 CI-Subunit-Atlas — Complete 42-Gene Nuclear-Encoded Complex I Atlas
          </h4>
          <div className="text-muted small">
            34 Structural Subunits (NDUFS/V/A/B) + 8 Assembly Factors (NDUFAF) · 1,680-Patient Aggregate (42×40) · All Nuclear WES-Detectable
          </div>
        </div>
        <div className="ms-auto d-flex gap-2 flex-wrap">
          <Link href="/mt-genome-atlas" className="btn btn-sm btn-outline-secondary">MT-Genome Atlas</Link>
          <Link href="/mt-trna-atlas" className="btn btn-sm btn-outline-secondary">MT-tRNA Atlas</Link>
          <Link href="/expert-dashboards-catalog" className="btn btn-sm btn-outline-primary">All Dashboards</Link>
        </div>
      </div>

      {err && <div className="alert alert-danger small">{err}</div>}

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}>{t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={ov} />}
      {tab === 1 && <SubunitTableTab data={bd} />}
      {tab === 2 && <ClinicalAtlasTab data={bd} />}
      {tab === 3 && <DefinitionsTab data={defn} />}
    </div>
  );
}
