'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#1a237e';   // deep indigo — CIII nuclear atlas
const LIGHT  = '#e8eaf6';
const COLOR2 = '#0d47a1';   // dark blue — structural subunits
const COLOR3 = '#4a148c';   // deep purple — assembly factors
const COLOR4 = '#b71c1c';   // dark red — severe / GRACILE
const COLOR5 = '#e65100';   // orange — Leigh / lactic acidosis
const COLOR6 = '#1b5e20';   // dark green — cardiomyopathy
const COLOR7 = '#006064';   // teal — BN-PAGE / biochemistry
const COLOR8 = '#880e4f';   // dark pink — drug CIs

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
  const ciii   = data.complex_iii || {};
  const scope  = data.atlas_scope || {};
  const biochem= data.biochemical_fingerprint || {};
  const pheno  = data.phenotypic_spectrum || {};
  const agg    = data.aggregate_clinical || {};
  const drug   = data.drug_contraindications || {};
  const wes    = data.wes_utility || {};

  return (
    <>
      {/* Atlas banner */}
      <SectionCard title="CIII-Subunit-Atlas — Complete 15-Gene Nuclear-Encoded Complex III Reference">
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">Complex III: </span>{ciii.full_name}</div>
            <div><span className="fw-semibold">Total subunits: </span>{ciii.subunits_total} ({ciii.subunits_nuclear} nuclear + {ciii.subunits_mtDNA} mtDNA)</div>
            <div><span className="fw-semibold">mtDNA subunit: </span>{ciii.mtDNA_subunit}</div>
            <div><span className="fw-semibold">Assembly factors: </span>{ciii.assembly_factors} nuclear-encoded</div>
            <div><span className="fw-semibold">Function: </span>{ciii.function_q_cycle}</div>
          </div>
          <div className="col-12 col-md-6">
            <div className="alert alert-primary py-1 px-2 mb-2 small">
              <strong>Supercomplex:</strong> {ciii.supercomplex}
            </div>
            <div className="alert alert-info py-1 px-2 mb-0 small">
              <strong>Mobile subunit:</strong> {ciii.risp_mobile}
            </div>
          </div>
        </div>
        <div className="row g-2 mt-2">
          <KPI label="Structural Subunits" value={scope.nuclear_structural_subunits} color={COLOR2} />
          <KPI label="Assembly Factors"    value={scope.nuclear_assembly_factors}    color={COLOR3} />
          <KPI label="Total Nuclear Genes" value={scope.total_nuclear_genes}         color={COLOR}  />
          <KPI label="Total Patients"      value={scope.total_patients}              color={COLOR}  />
          <KPI label="Patients/Gene"       value={scope.patients_per_gene}           color={COLOR}  />
          <KPI label="Seeds"               value={scope.seeds}                       color={COLOR}  />
        </div>
        <div className="alert alert-warning py-1 px-2 mt-2 small">
          <strong>Note:</strong> {scope.mtDNA_subunit_note}
        </div>
      </SectionCard>

      {/* Biochemical fingerprint */}
      <SectionCard title="⚗️ Biochemical Fingerprint — Isolated CIII Deficiency" borderColor={COLOR7}>
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div className="mb-1"><span className="fw-semibold">CIII: </span>{biochem.ciii_deficiency}</div>
            <div className="mb-1">
              <span className="badge me-1" style={{ backgroundColor: COLOR5 }}>CII ALWAYS NORMAL</span>
              {biochem.cii_always_normal}
            </div>
            <div className="mb-1"><span className="fw-semibold">CI also low: </span>{biochem.ci_also_low}</div>
          </div>
          <div className="col-12 col-md-6">
            <div className="mb-1"><span className="fw-semibold">CIV: </span>{biochem.civ_usually_normal}</div>
            <div className="alert alert-secondary py-1 px-2 small">
              <strong>Pre-CIII BN-PAGE:</strong> {biochem.pre_ciii_bnpage}
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Phenotypic spectrum */}
      <SectionCard title="🌈 Phenotypic Spectrum — CIII Disorders" borderColor={COLOR4}>
        <div className="row g-3 small">
          {pheno.GRACILE && (
            <div className="col-12 col-md-6">
              <div className="fw-bold" style={{ color: COLOR4 }}>GRACILE Syndrome — {pheno.GRACILE.gene}</div>
              <div>{pheno.GRACILE.features}</div>
              <div><span className="fw-semibold">Onset: </span>{pheno.GRACILE.onset} | <span className="fw-semibold">Survival: </span>{pheno.GRACILE.survival}</div>
              <div><span className="fw-semibold">Founder: </span>{pheno.GRACILE.founder}</div>
            </div>
          )}
          {pheno.Bjornstad && (
            <div className="col-12 col-md-6">
              <div className="fw-bold" style={{ color: COLOR3 }}>Bjornstad Syndrome — {pheno.Bjornstad.gene}</div>
              <div>{pheno.Bjornstad.features}</div>
              <div><span className="fw-semibold">Onset: </span>{pheno.Bjornstad.onset}</div>
              <div className="text-muted">{pheno.Bjornstad.note}</div>
            </div>
          )}
          {pheno.Progressive_Neurodegeneration && (
            <div className="col-12 col-md-6">
              <div className="fw-bold" style={{ color: COLOR }}>{pheno.Progressive_Neurodegeneration.gene} — Progressive Neurodegeneration</div>
              <div>{pheno.Progressive_Neurodegeneration.features}</div>
              <div><span className="fw-semibold">Onset: </span>{pheno.Progressive_Neurodegeneration.onset}</div>
              <div className="text-muted">{pheno.Progressive_Neurodegeneration.note}</div>
            </div>
          )}
          {pheno.Leigh_like && (
            <div className="col-12 col-md-6">
              <div className="fw-bold" style={{ color: COLOR5 }}>Leigh/Leigh-like — {pheno.Leigh_like.genes}</div>
              <div>{pheno.Leigh_like.features}</div>
              <div><span className="fw-semibold">Onset: </span>{pheno.Leigh_like.onset}</div>
            </div>
          )}
          {pheno.Cardiomyopathy_dominant && (
            <div className="col-12 col-md-6">
              <div className="fw-bold" style={{ color: COLOR6 }}>Cardiomyopathy-Dominant — {pheno.Cardiomyopathy_dominant.genes}</div>
              <div>{pheno.Cardiomyopathy_dominant.features}</div>
            </div>
          )}
        </div>
      </SectionCard>

      {/* Aggregate clinical */}
      <SectionCard title={`📊 Aggregate Clinical — ${scope.total_patients} Patients (15×40, seeds ${scope.seeds})`} borderColor={COLOR}>
        <div className="row g-2 small">
          <KPI label="GRACILE Syndrome"    value={`${agg.gracile_pct}%`}       color={COLOR4} />
          <KPI label="Bjornstad Syndrome"  value={`${agg.bjornstad_pct}%`}     color={COLOR3} />
          <KPI label="Leigh MRI"           value={`${agg.leigh_mri_pct}%`}     color={COLOR5} />
          <KPI label="Leukoencephalopathy" value={`${agg.leuko_pct}%`}         color={COLOR}  />
          <KPI label="Cardiomyopathy"      value={`${agg.cardiac_pct}%`}       color={COLOR6} />
          <KPI label="Hepatopathy"         value={`${agg.hepatopathy_pct}%`}   color={COLOR4} />
          <KPI label="Lactic Acidosis"     value={`${agg.lactic_ac_pct}%`}     color={COLOR5} />
          <KPI label="Pre-CIII BN-PAGE"    value={`${agg.pre_ciii_bnpage_pct}%`} color={COLOR7} />
          <KPI label="CI also Low"         value={`${agg.ci_also_low_pct}%`}   color={COLOR2} />
          <KPI label="Mean CIII Activity"  value={`${agg.mean_ciii_activity_pct}%`} color={COLOR} />
        </div>
      </SectionCard>

      {/* Drug CIs */}
      <SectionCard title="💊 Drug Contraindications — All 15 CIII Genes" borderColor={COLOR8}>
        {(drug.absolute_ci_all_15_genes || []).map((d, i) => (
          <div key={i} className="mb-1 small">
            <Badge text="ABSOLUTE CI" color={COLOR8} />
            <span className="fw-semibold">{d.drug}: </span>{d.mechanism}
          </div>
        ))}
        <hr className="my-2" />
        <div className="fw-semibold small mb-1" style={{ color: COLOR5 }}>Mandatory Workup (all CIII):</div>
        {(drug.mandatory_workup || []).map((m, i) => (
          <div key={i} className="small mb-1">• {m}</div>
        ))}
      </SectionCard>

      {/* GRACILE-specific management */}
      <SectionCard title="🚨 GRACILE-Specific Management (BCS1L p.Ser78Gly)" borderColor={COLOR4}>
        {(drug.gracile_specific || []).map((m, i) => (
          <div key={i} className="small mb-1">• {m}</div>
        ))}
      </SectionCard>

      {/* Bjornstad-specific */}
      <SectionCard title="💇 Bjornstad Syndrome Management (BCS1L — milder alleles)" borderColor={COLOR3}>
        {(drug.bjornstad_specific || []).map((m, i) => (
          <div key={i} className="small mb-1">• {m}</div>
        ))}
      </SectionCard>

      {/* WES utility */}
      <SectionCard title="🧬 WES Utility — Nuclear vs mtDNA CIII Genes" borderColor={COLOR}>
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div className="fw-semibold mb-1" style={{ color: COLOR }}>Nuclear (WES detects all 15):</div>
            <div>{wes.nuclear_genes_detectable}</div>
          </div>
          <div className="col-12 col-md-6">
            <div className="alert alert-warning py-1 px-2 mb-2 small">
              <strong>WES MISSES MT-CYB:</strong> {wes.mtDNA_missed}
            </div>
            <div className="small text-muted">{wes.panel_note}</div>
            <div className="small mt-1"><span className="fw-semibold">BN-PAGE value: </span>{wes.bnpage_value}</div>
          </div>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Gene Table ───────────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const genes = data.genes || [];
  return (
    <SectionCard title={`CIII Nuclear Gene Reference Table — ${genes.length} Genes, ${data.total_patients} Patients`}>
      <div className="table-responsive">
        <table className="table table-sm table-bordered table-hover small align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Class</th><th>Module</th><th>OMIM Gene</th><th>Chr</th>
              <th>Phenotype</th><th>CIII Activity (mean%)</th><th>Median Onset (mo)</th>
              <th>Leigh MRI%</th><th>Leuko%</th><th>Cardiac%</th><th>Hepato%</th>
              <th>Hypoglycaemia%</th><th>Pre-CIII BN-PAGE%</th><th>CI Low%</th>
            </tr>
          </thead>
          <tbody>
            {genes.map(g => (
              <tr key={g.gene}>
                <td className="fw-bold" style={{ color: g.gene_class === 'assembly_factor' ? COLOR3 : COLOR2 }}>{g.gene}</td>
                <td>
                  <span className="badge" style={{ backgroundColor: g.gene_class === 'assembly_factor' ? COLOR3 : COLOR2 }}>
                    {g.gene_class === 'assembly_factor' ? 'AF' : g.subunit_series}
                  </span>
                </td>
                <td style={{ maxWidth: 200, fontSize: '0.68rem' }}>{g.ciii_module}</td>
                <td>{g.omim_gene}</td>
                <td>{g.chromosome}</td>
                <td style={{ fontSize: '0.72rem' }}>
                  <span className="badge" style={{
                    backgroundColor:
                      g.gene === 'BCS1L' ? COLOR4 :
                      g.gene === 'TTC19' ? COLOR :
                      g.gene_class === 'assembly_factor' ? COLOR3 : COLOR5
                  }}>{g.phenotype?.split('—')[0]?.trim()}</span>
                </td>
                <td className="text-center fw-bold">{g.ciii_activity_mean_pct}%</td>
                <td className="text-center">{g.median_onset_months} mo</td>
                <td className="text-center">{g.leigh_mri_pct}%</td>
                <td className="text-center">{g.leuko_pct}%</td>
                <td className="text-center" style={{ color: g.cardiac_pct > 50 ? COLOR6 : 'inherit', fontWeight: g.cardiac_pct > 50 ? 'bold' : 'normal' }}>
                  {g.cardiac_pct}%
                </td>
                <td className="text-center">{g.hepatopathy_pct}%</td>
                <td className="text-center" style={{ color: g.hypoglycaemia_pct > 40 ? COLOR4 : 'inherit' }}>
                  {g.hypoglycaemia_pct}%
                </td>
                <td className="text-center" style={{ color: g.pre_ciii_bnpage_pct > 50 ? COLOR7 : 'inherit' }}>
                  {g.pre_ciii_bnpage_pct}%
                </td>
                <td className="text-center" style={{ color: g.ci_low_pct > 40 ? COLOR2 : 'inherit' }}>
                  {g.ci_low_pct}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SectionCard>
  );
}

// ── Tab: Clinical Atlas ───────────────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const genes = data.genes || [];
  return (
    <>
      {genes.map(g => (
        <SectionCard
          key={g.gene}
          title={`${g.gene} — ${g.subunit_series} (${g.chromosome}) — ${g.inheritance}`}
          borderColor={g.gene_class === 'assembly_factor' ? COLOR3 : COLOR2}
        >
          <div className="row g-3 small">
            <div className="col-12 col-md-6">
              <div><span className="fw-semibold">Disease: </span>{g.disease_summary}</div>
              <div><span className="fw-semibold">Inheritance: </span>{g.inheritance}</div>
              <div><span className="fw-semibold">OMIM gene: </span>{g.omim_gene} | Disease OMIM: {g.disease_omim}</div>
              <div><span className="fw-semibold">Module: </span>{g.ciii_module}</div>
              {g.ci_also_low && (
                <div className="alert alert-info py-1 px-2 mt-1 small">
                  <strong>⚠️ CI also low</strong> — supercomplex (SC I+III₂) destabilisation
                </div>
              )}
            </div>
            <div className="col-12 col-md-6">
              <div className="alert alert-secondary py-1 px-2 mb-1" style={{ fontSize: '0.75rem' }}>
                <strong>Hallmark:</strong> {g.hallmark}
              </div>
              <div><span className="fw-semibold">Founder variant: </span>{g.founder_variant}</div>
            </div>
          </div>
          <div className="row g-2 mt-2 small">
            <div className="col-6 col-md-2"><span className="fw-semibold">Patients: </span>{g.n_patients}</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">CIII mean: </span>{g.ciii_activity_mean_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Leigh MRI: </span>{g.leigh_mri_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Leuko: </span>{g.leuko_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Cardiac: </span>
              <span style={{ color: g.cardiac_pct > 50 ? COLOR6 : 'inherit', fontWeight: g.cardiac_pct > 50 ? 'bold' : 'normal' }}>
                {g.cardiac_pct}%
              </span>
            </div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Hepato: </span>{g.hepatopathy_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Hypoglycaemia: </span>
              <span style={{ color: g.hypoglycaemia_pct > 40 ? COLOR4 : 'inherit' }}>{g.hypoglycaemia_pct}%</span>
            </div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Pre-CIII BN-PAGE: </span>
              <span style={{ color: g.pre_ciii_bnpage_pct > 50 ? COLOR7 : 'inherit' }}>{g.pre_ciii_bnpage_pct}%</span>
            </div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Lactic acidosis: </span>{g.lactic_acidosis_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Median onset: </span>{g.median_onset_months} mo</div>
          </div>
        </SectionCard>
      ))}
    </>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  return (
    <SectionCard title="CIII Subunit Atlas — Glossary & Definitions">
      <div className="row g-2 small">
        {Object.entries(data).map(([k, v]) => (
          <div key={k} className="col-12 col-md-6 mb-2">
            <span className="fw-bold" style={{ color: COLOR }}>{k.replace(/_/g,' ')}: </span>
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </div>
    </SectionCard>
  );
}

// ── Page root ─────────────────────────────────────────────────────────────────
export default function CIIISubunitAtlasPage() {
  const [tab, setTab]     = useState(0);
  const [overview, setOv] = useState(null);
  const [breakdown, setBk]= useState(null);
  const [defs, setDefs]   = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ciii-subunit-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/ciii-subunit-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ciii-subunit-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => { setOv(ov); setBk(bk); setDefs(df); })
      .catch(e => setError(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3" style={{ background: LIGHT, minHeight: '100vh' }}>
      <div className="mb-3">
        <Link href="/" className="btn btn-sm btn-outline-secondary me-2">← Home</Link>
        <Link href="/dashboard-catalog" className="btn btn-sm btn-outline-secondary">Dashboard Catalog</Link>
      </div>

      <div className="card shadow-sm mb-4" style={{ borderTop: `4px solid ${COLOR}` }}>
        <div className="card-body py-2 px-3">
          <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
            🧬 CIII-Subunit-Atlas — Complete 15-Gene Nuclear-Encoded Complex III
          </h4>
          <div className="small text-muted">
            9 Structural Subunits (UQCRC1/UQCRC2/CYC1/UQCRFS1/UQCRB/UQCRQ/UQCR10/UQCR11/UQCRH) ·
            6 Assembly Factors (BCS1L/TTC19/LYRM7/UQCC1/UQCC2/UQCC3) ·
            GRACILE + Bjornstad (BCS1L) · Progressive Neuro (TTC19) · SC I+III₂ (UQCRC1/UQCRC2) ·
            Pre-CIII BN-PAGE Hallmark · MT-CYB (mtDNA) WES-missed ·
            600-Patient Aggregate (15×40, seeds 710–724)
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottom: `2px solid ${COLOR}` } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <GeneTableTab data={breakdown} />}
      {tab === 2 && <ClinicalAtlasTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={defs} />}
    </div>
  );
}
