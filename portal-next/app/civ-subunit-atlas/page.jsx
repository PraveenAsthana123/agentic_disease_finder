'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#004d40';   // deep teal — CIV nuclear atlas
const LIGHT  = '#e0f2f1';
const COLOR2 = '#006064';   // dark cyan — structural subunits
const COLOR3 = '#1a237e';   // deep indigo — assembly factors
const COLOR4 = '#b71c1c';   // dark red — severe / SCO2 HCM
const COLOR5 = '#e65100';   // orange — Leigh / lactic acidosis
const COLOR6 = '#1b5e20';   // dark green — cardiomyopathy
const COLOR7 = '#4a148c';   // deep purple — copper/heme chemistry
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
  const agg    = data.aggregate_clinical || {};
  const drug   = data.drug_contraindications || {};
  const wes    = data.wes_utility || {};
  const pheno  = data.hallmark_phenotypes || {};
  const genes  = data.gene_list || {};
  const copper = data.copper_centres || {};
  const heme   = data.heme_centres || {};

  return (
    <>
      {/* Atlas banner */}
      <SectionCard title="CIV-Subunit-Atlas — Complete 19-Gene Nuclear-Encoded Complex IV Reference">
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">Complex IV: </span>{data.complex} — {data.function}</div>
            <div><span className="fw-semibold">Total Genes: </span>{data.n_genes} nuclear ({data.n_structural_subunits} structural + {data.n_assembly_factors} assembly factors)</div>
            <div><span className="fw-semibold">Cohort: </span>{data.cohort_formula}</div>
            <div><span className="fw-semibold">mtDNA subunits (WES missed): </span>{data.mtDNA_subunits}</div>
            <div className="alert alert-info py-1 px-2 mt-2 small">
              <strong>NDUFA4 Reclassification:</strong> {data.reclassification}
            </div>
          </div>
          <div className="col-12 col-md-6">
            <div className="fw-semibold mb-1" style={{ color: COLOR2 }}>4 Nuclear Structural Subunits:</div>
            <div className="mb-2">{(genes.structural_subunits_4_nuclear || []).join(' · ')}</div>
            <div className="fw-semibold mb-1" style={{ color: COLOR3 }}>15 Nuclear Assembly Factors:</div>
            <div className="small">{(genes.assembly_factors_15_nuclear || []).join(' · ')}</div>
          </div>
        </div>
        <div className="row g-2 mt-2">
          <KPI label="Structural Subunits" value={data.n_structural_subunits}  color={COLOR2} />
          <KPI label="Assembly Factors"    value={data.n_assembly_factors}     color={COLOR3} />
          <KPI label="Total Nuclear Genes" value={data.n_genes}                color={COLOR}  />
          <KPI label="Total Patients"      value={data.n_patients}             color={COLOR}  />
        </div>
        <div className="alert alert-warning py-1 px-2 mt-2 small">
          <strong>CII ALWAYS NORMAL:</strong> {data.cii_always_normal}
        </div>
      </SectionCard>

      {/* Copper & Heme centres */}
      <SectionCard title="⚗️ CIV Metal Centres — Copper (CuA/CuB) + Haem (a/a3)" borderColor={COLOR7}>
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div className="fw-semibold mb-1" style={{ color: COLOR7 }}>Copper Centres:</div>
            <div className="mb-1"><span className="fw-semibold">CuA: </span>{copper.CuA}</div>
            <div><span className="fw-semibold">CuB: </span>{copper.CuB}</div>
          </div>
          <div className="col-12 col-md-6">
            <div className="fw-semibold mb-1" style={{ color: COLOR7 }}>Haem Centres:</div>
            <div className="mb-1"><span className="fw-semibold">Haem a: </span>{heme.haem_a}</div>
            <div><span className="fw-semibold">Haem a3: </span>{heme.haem_a3}</div>
          </div>
        </div>
      </SectionCard>

      {/* Hallmark phenotypes */}
      <SectionCard title="🌈 Hallmark Phenotypes — CIV Disorders" borderColor={COLOR4}>
        <div className="row g-3 small">
          {Object.entries(pheno).map(([key, val]) => (
            <div key={key} className="col-12 col-md-6">
              <div className="fw-bold" style={{ color:
                key === 'HCM_100pct' ? COLOR4 :
                key === 'PINDAC_Triad' ? COLOR5 :
                key === 'SCA_Neuropathy' ? COLOR3 :
                key === 'Hepatic_Dominant' ? COLOR6 : COLOR
              }}>{key.replace(/_/g,' ')} — {val.gene}</div>
              <div className="text-muted">{val.note}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Aggregate clinical */}
      <SectionCard title={`📊 Aggregate Clinical — ${data.n_patients} Patients (19×40, seeds 725–743)`} borderColor={COLOR}>
        <div className="row g-2 small">
          <KPI label="Leigh MRI"          value={`${agg.leigh_mri_pct}%`}      color={COLOR5} />
          <KPI label="Cardiomyopathy"     value={`${agg.cardiac_pct}%`}        color={COLOR6} />
          <KPI label="HCM"                value={`${agg.hcm_pct}%`}            color={COLOR4} />
          <KPI label="Hepatopathy"        value={`${agg.hepatopathy_pct}%`}    color={COLOR4} />
          <KPI label="Neuropathy"         value={`${agg.neuropathy_pct}%`}     color={COLOR3} />
          <KPI label="Lactic Acidosis"    value={`${agg.lactic_ac_pct}%`}      color={COLOR5} />
          <KPI label="PINDAC Syndrome"    value={`${agg.pindac_pct}%`}         color={COLOR5} />
          <KPI label="SCA/Neuropathy"     value={`${agg.scl_ataxia_pct}%`}     color={COLOR3} />
          <KPI label="French-Canadian"    value={`${agg.french_canadian_pct}%`} color={COLOR}  />
          <KPI label="Mean COX Activity"  value={`${agg.mean_cox_activity_pct}%`} color={COLOR} />
        </div>
      </SectionCard>

      {/* Drug CIs */}
      <SectionCard title="💊 Drug Contraindications — All 19 CIV Genes" borderColor={COLOR8}>
        {(drug.absolute_ci_all_19_genes || []).map((d, i) => (
          <div key={i} className="mb-1 small">
            <Badge text="ABSOLUTE CI" color={d.drug === 'Propofol' ? COLOR4 : COLOR8} />
            <span className="fw-semibold">{d.drug}: </span>{d.mechanism}
          </div>
        ))}
        <hr className="my-2" />
        <div className="fw-semibold small mb-1" style={{ color: COLOR7 }}>Copper Therapy (SCO1/SCO2/COA6):</div>
        {(drug.copper_therapy || []).map((m, i) => (
          <div key={i} className="small mb-1">• {m}</div>
        ))}
        <hr className="my-2" />
        <div className="fw-semibold small mb-1" style={{ color: COLOR5 }}>Mandatory Workup (all CIV):</div>
        {(drug.mandatory_workup || []).map((m, i) => (
          <div key={i} className="small mb-1">• {m}</div>
        ))}
      </SectionCard>

      {/* WES utility */}
      <SectionCard title="🧬 WES Utility — Nuclear vs mtDNA CIV Genes" borderColor={COLOR}>
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div className="fw-semibold mb-1" style={{ color: COLOR }}>Nuclear (WES detects all 19):</div>
            <div>{wes.nuclear_genes_detectable}</div>
          </div>
          <div className="col-12 col-md-6">
            <div className="alert alert-warning py-1 px-2 mb-2 small">
              <strong>WES MISSES MT-CO1/CO2/CO3:</strong> {wes.mtDNA_missed}
            </div>
            <div className="small text-muted">{wes.panel_note}</div>
            <div className="small mt-1"><span className="fw-semibold">Enzymatic note: </span>{wes.enzymatic_distinction}</div>
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
    <SectionCard title={`CIV Nuclear Gene Reference Table — ${genes.length} Genes, ${data.total_patients} Patients`}>
      <div className="table-responsive">
        <table className="table table-sm table-bordered table-hover small align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Class</th><th>Module</th><th>OMIM</th><th>Chr</th>
              <th>Phenotype</th><th>COX%</th><th>Onset(mo)</th>
              <th>Leigh%</th><th>HCM%</th><th>Cardiac%</th><th>Hepato%</th>
              <th>Neuro%</th><th>PINDAC%</th><th>SCA%</th><th>FC%</th>
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
                <td style={{ maxWidth: 180, fontSize: '0.68rem' }}>{g.civ_module}</td>
                <td>{g.omim_gene}</td>
                <td>{g.chromosome}</td>
                <td style={{ fontSize: '0.72rem' }}>
                  <span className="badge" style={{
                    backgroundColor:
                      g.gene === 'SCO2'   ? COLOR4 :
                      g.gene === 'SURF1'  ? COLOR5 :
                      g.gene === 'LRPPRC' ? COLOR  :
                      g.gene === 'COA7'   ? COLOR3 :
                      g.gene_class === 'assembly_factor' ? COLOR3 : COLOR2
                  }}>{g.phenotype?.split('—')[0]?.trim()}</span>
                </td>
                <td className="text-center fw-bold">{g.cox_activity_mean_pct}%</td>
                <td className="text-center">{g.median_onset_months} mo</td>
                <td className="text-center" style={{ color: g.leigh_mri_pct > 60 ? COLOR5 : 'inherit', fontWeight: g.leigh_mri_pct > 60 ? 'bold' : 'normal' }}>
                  {g.leigh_mri_pct}%
                </td>
                <td className="text-center" style={{ color: g.hcm_pct > 50 ? COLOR4 : 'inherit', fontWeight: g.hcm_pct > 50 ? 'bold' : 'normal' }}>
                  {g.hcm_pct}%
                </td>
                <td className="text-center" style={{ color: g.cardiac_pct > 50 ? COLOR6 : 'inherit', fontWeight: g.cardiac_pct > 50 ? 'bold' : 'normal' }}>
                  {g.cardiac_pct}%
                </td>
                <td className="text-center">{g.hepatopathy_pct}%</td>
                <td className="text-center" style={{ color: g.neuropathy_pct > 50 ? COLOR3 : 'inherit' }}>
                  {g.neuropathy_pct}%
                </td>
                <td className="text-center" style={{ color: g.pindac_pct > 50 ? COLOR5 : 'inherit' }}>
                  {g.pindac_pct}%
                </td>
                <td className="text-center" style={{ color: g.scl_ataxia_pct > 50 ? COLOR7 : 'inherit' }}>
                  {g.scl_ataxia_pct}%
                </td>
                <td className="text-center" style={{ color: g.french_canadian_pct > 50 ? COLOR : 'inherit' }}>
                  {g.french_canadian_pct}%
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
              <div><span className="fw-semibold">Phenotype: </span>{g.phenotype}</div>
              <div><span className="fw-semibold">Inheritance: </span>{g.inheritance}</div>
              <div><span className="fw-semibold">OMIM gene: </span>{g.omim_gene} | Disease: {g.disease_omim}</div>
              <div><span className="fw-semibold">Module: </span>{g.civ_module}</div>
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
            <div className="col-6 col-md-2"><span className="fw-semibold">COX mean: </span>{g.cox_activity_mean_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Leigh MRI: </span>{g.leigh_mri_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">HCM: </span>
              <span style={{ color: g.hcm_pct > 50 ? COLOR4 : 'inherit', fontWeight: g.hcm_pct > 50 ? 'bold' : 'normal' }}>
                {g.hcm_pct}%
              </span>
            </div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Cardiac: </span>
              <span style={{ color: g.cardiac_pct > 50 ? COLOR6 : 'inherit' }}>{g.cardiac_pct}%</span>
            </div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Hepato: </span>{g.hepatopathy_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Neuropathy: </span>
              <span style={{ color: g.neuropathy_pct > 50 ? COLOR3 : 'inherit' }}>{g.neuropathy_pct}%</span>
            </div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Lactic acid: </span>{g.lactic_acidosis_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">PINDAC: </span>{g.pindac_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">SCA/neuro: </span>{g.scl_ataxia_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Fr-Canadian: </span>{g.french_canadian_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Onset: </span>{g.median_onset_months} mo</div>
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
    <SectionCard title="CIV Subunit Atlas — Glossary & Definitions">
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
export default function CIVSubunitAtlasPage() {
  const [tab, setTab]     = useState(0);
  const [overview, setOv] = useState(null);
  const [breakdown, setBk]= useState(null);
  const [defs, setDefs]   = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/civ-subunit-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/civ-subunit-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/civ-subunit-atlas/definitions`).then(r => r.json()),
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
            🧬 CIV-Subunit-Atlas — Complete 19-Gene Nuclear-Encoded Complex IV (Cytochrome c Oxidase)
          </h4>
          <div className="small text-muted">
            4 Structural Subunits (COX4I1/COX6B1/COX8A/NDUFA4-reclassified) ·
            15 Assembly Factors (SURF1/SCO1/SCO2/COX10/COX15/COX20/COA3/COA5/COA6/COA7/TACO1/LRPPRC/PET100/COX14/FASTKD2) ·
            SURF1-Most-Common-CIV-Leigh · SCO2-HCM-100pct · LRPPRC-French-Canadian-LSFC ·
            COX4I1-PINDAC-Triad · COA7-SCA-Neuropathy-ONLY-CIV ·
            MT-CO1/CO2/CO3 mtDNA WES-missed · 760-Patient Aggregate (19×40, seeds 725–743)
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
