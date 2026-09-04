'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#1a237e';   // deep indigo — CV nuclear atlas
const LIGHT  = '#e8eaf6';
const COLOR2 = '#283593';   // F1 structural subunits
const COLOR3 = '#0d47a1';   // F0 structural subunits
const COLOR4 = '#b71c1c';   // dark red — HCM / severe cardiac
const COLOR5 = '#e65100';   // orange — 3-MGA / TMEM70
const COLOR6 = '#1b5e20';   // dark green — assembly factors
const COLOR7 = '#4a148c';   // deep purple — rotary mechanism
const COLOR8 = '#880e4f';   // dark pink — drug CIs / absolute CI

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
  const rot    = data.rotary_mechanism || {};
  const dimer  = data.cv_dimer || {};
  const mga    = data.three_mga_types || {};

  return (
    <>
      {/* Atlas banner */}
      <SectionCard title="CV-Subunit-Atlas — Complete 16-Gene Nuclear-Encoded Complex V (F1F0-ATP Synthase) Reference">
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">Complex V: </span>{data.complex} — {data.function}</div>
            <div><span className="fw-semibold">Total Genes: </span>{data.n_genes} nuclear ({data.n_f1_structural} F1 structural + {data.n_f0_structural} F0 structural + {data.n_assembly_factors} assembly factors)</div>
            <div><span className="fw-semibold">Cohort: </span>{data.cohort_formula}</div>
            <div><span className="fw-semibold">mtDNA subunits (WES missed): </span>{data.mtDNA_subunits}</div>
            <div className="alert alert-info py-1 px-2 mt-2 small">
              <strong>3-MGA Hallmark:</strong> {data.three_mga_hallmark}
            </div>
          </div>
          <div className="col-12 col-md-6">
            <div className="fw-semibold mb-1" style={{ color: COLOR2 }}>5 Nuclear F1 Structural Subunits:</div>
            <div className="mb-2">{(genes.f1_structural_5_nuclear || []).join(' · ')}</div>
            <div className="fw-semibold mb-1" style={{ color: COLOR3 }}>8 Nuclear F0 Structural Subunits:</div>
            <div className="mb-2 small">{(genes.f0_structural_8_nuclear || []).join(' · ')}</div>
            <div className="fw-semibold mb-1" style={{ color: COLOR6 }}>3 Nuclear Assembly Factors:</div>
            <div className="small">{(genes.assembly_factors_3_nuclear || []).join(' · ')}</div>
          </div>
        </div>
        <div className="row g-2 mt-2">
          <KPI label="F1 Structural"     value={data.n_f1_structural}   color={COLOR2} />
          <KPI label="F0 Structural"     value={data.n_f0_structural}   color={COLOR3} />
          <KPI label="Assembly Factors"  value={data.n_assembly_factors} color={COLOR6} />
          <KPI label="Total Genes"       value={data.n_genes}           color={COLOR}  />
          <KPI label="Total Patients"    value={data.n_patients}        color={COLOR}  />
        </div>
        <div className="alert alert-warning py-1 px-2 mt-2 small">
          <strong>CII ALWAYS NORMAL:</strong> {data.cii_always_normal}
        </div>
      </SectionCard>

      {/* Rotary mechanism */}
      <SectionCard title="⚙️ CV Rotary Catalysis — F1F0 Mechanism" borderColor={COLOR7}>
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div className="fw-semibold mb-1" style={{ color: COLOR7 }}>Rotor (rotating assembly):</div>
            <div className="mb-1"><span className="fw-semibold">c-ring: </span>{rot.c_ring_stoichiometry}</div>
            <div className="mb-1"><span className="fw-semibold">Proton channel: </span>{rot.proton_channel}</div>
            <div><span className="fw-semibold">Rotor components: </span>{rot.rotor}</div>
          </div>
          <div className="col-12 col-md-6">
            <div className="fw-semibold mb-1" style={{ color: COLOR2 }}>Stator (fixed assembly):</div>
            <div className="mb-1"><span className="fw-semibold">Stator: </span>{rot.stator}</div>
            <div className="mb-1"><span className="fw-semibold">Catalytic sites: </span>{rot.catalytic_sites}</div>
          </div>
        </div>
        <hr className="my-2" />
        <div className="fw-semibold mb-1 small" style={{ color: COLOR7 }}>CV₂ Dimer Module (eukaryote-specific):</div>
        <div className="small mb-1"><span className="fw-semibold">Dimerisation: </span>{dimer.dimerisation_module}</div>
        <div className="small mb-1"><span className="fw-semibold">Cristae shaping: </span>{dimer.cristae_shaping}</div>
        <div className="small"><span className="fw-semibold">Stoichiometry: </span>{dimer.cv2_stoichiometry}</div>
      </SectionCard>

      {/* 3-MGA types */}
      <SectionCard title="🧪 3-Methylglutaconic Aciduria (3-MGA) — CV Biomarker (Type V)" borderColor={COLOR5}>
        <div className="row g-3 small">
          {Object.entries(mga).map(([k, v]) => (
            <div key={k} className="col-12 col-md-6">
              <span className="fw-bold" style={{ color: k === 'Type_V' ? COLOR5 : COLOR }}>{k.replace(/_/g,' ')}: </span>
              <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Hallmark phenotypes */}
      <SectionCard title="🌈 Hallmark Phenotypes — CV Disorders" borderColor={COLOR4}>
        <div className="row g-3 small">
          {Object.entries(pheno).map(([key, val]) => (
            <div key={key} className="col-12 col-md-6">
              <div className="fw-bold" style={{ color:
                key === 'TMEM70_3MGA_HCM_Roma'  ? COLOR5 :
                key === 'ATP5F1A_HCM_Neonatal'   ? COLOR4 :
                key === 'ATPAF2_F1_Assembly_First'? COLOR6 :
                key === 'ATP5MC3_Cardiac_60pct'  ? COLOR4 :
                COLOR7
              }}>{key.replace(/_/g,' ')} — {val.gene}</div>
              <div className="text-muted">{val.note}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Aggregate clinical */}
      <SectionCard title={`📊 Aggregate Clinical — ${data.n_patients} Patients (16×40, seeds 744–759)`} borderColor={COLOR}>
        <div className="row g-2 small">
          <KPI label="3-MGA Aciduria"    value={`${agg.three_mga_pct}%`}       color={COLOR5} />
          <KPI label="HCM"               value={`${agg.hcm_pct}%`}             color={COLOR4} />
          <KPI label="Cardiac"           value={`${agg.cardiac_pct}%`}         color={COLOR4} />
          <KPI label="Leigh MRI"         value={`${agg.leigh_mri_pct}%`}       color={COLOR3} />
          <KPI label="Hepatopathy"       value={`${agg.hepatopathy_pct}%`}     color={COLOR} />
          <KPI label="Lactic Acidosis"   value={`${agg.lactic_ac_pct}%`}       color={COLOR5} />
          <KPI label="Neuropathy"        value={`${agg.neuropathy_pct}%`}      color={COLOR7} />
          <KPI label="Hyperammonaemia"   value={`${agg.hyperammonemia_pct}%`}  color={COLOR5} />
          <KPI label="Mean ATP Activity" value={`${agg.mean_atp_activity_pct}%`} color={COLOR} />
        </div>
      </SectionCard>

      {/* Drug CIs */}
      <SectionCard title="💊 Drug Contraindications — All 16 CV Genes" borderColor={COLOR8}>
        {(drug.absolute_ci_all_16_genes || []).map((d, i) => (
          <div key={i} className="mb-2 small">
            <Badge text="ABSOLUTE CI" color={
              d.drug.startsWith('Oligomycin') ? COLOR7 :
              d.drug.startsWith('Propofol')   ? COLOR4 :
              COLOR8
            } />
            <span className="fw-semibold">{d.drug}: </span>{d.mechanism}
          </div>
        ))}
        <hr className="my-2" />
        <div className="fw-semibold small mb-1" style={{ color: COLOR5 }}>Mandatory Workup (all CV):</div>
        {(drug.mandatory_workup || []).map((m, i) => (
          <div key={i} className="small mb-1">• {m}</div>
        ))}
        <hr className="my-2" />
        <div className="small text-muted">
          <span className="fw-semibold">KD note: </span>{drug.ketogenic_diet}
        </div>
      </SectionCard>

      {/* WES utility */}
      <SectionCard title="🧬 WES Utility — Nuclear vs mtDNA CV Genes" borderColor={COLOR}>
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div className="fw-semibold mb-1" style={{ color: COLOR }}>Nuclear (WES detects all 16):</div>
            <div>{wes.nuclear_genes_detectable}</div>
          </div>
          <div className="col-12 col-md-6">
            <div className="alert alert-warning py-1 px-2 mb-2 small">
              <strong>WES MISSES MT-ATP6 + MT-ATP8:</strong> {wes.mtDNA_missed}
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
    <SectionCard title={`CV Nuclear Gene Reference Table — ${genes.length} Genes, ${data.total_patients} Patients`}>
      <div className="table-responsive">
        <table className="table table-sm table-bordered table-hover small align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Alias</th><th>Class</th><th>Module</th><th>OMIM</th><th>Chr</th>
              <th>Phenotype</th><th>ATP%</th><th>Onset(mo)</th>
              <th>3-MGA%</th><th>HCM%</th><th>Cardiac%</th>
              <th>Leigh%</th><th>Hepato%</th><th>NH3%</th>
            </tr>
          </thead>
          <tbody>
            {genes.map(g => (
              <tr key={g.gene}>
                <td className="fw-bold" style={{ color:
                  g.gene_class === 'assembly_factor' ? COLOR6 :
                  g.gene_class === 'f0_structural'   ? COLOR3 : COLOR2
                }}>{g.gene}</td>
                <td style={{ fontSize: '0.68rem', color: '#666' }}>{g.alias}</td>
                <td>
                  <span className="badge" style={{ backgroundColor:
                    g.gene_class === 'assembly_factor' ? COLOR6 :
                    g.gene_class === 'f0_structural'   ? COLOR3 : COLOR2,
                    fontSize: '0.65rem'
                  }}>
                    {g.gene_class === 'assembly_factor' ? 'AF' :
                     g.gene_class === 'f0_structural'   ? 'F0' : 'F1'}
                  </span>
                </td>
                <td style={{ maxWidth: 160, fontSize: '0.65rem' }}>{g.subunit_series}</td>
                <td>{g.omim_gene}</td>
                <td>{g.chromosome}</td>
                <td style={{ fontSize: '0.68rem' }}>
                  <span className="badge" style={{
                    backgroundColor:
                      g.gene === 'TMEM70'   ? COLOR5 :
                      g.gene === 'ATP5F1A'  ? COLOR4 :
                      g.gene === 'ATPAF2'   ? COLOR6 :
                      g.gene === 'ATP5MC3'  ? COLOR4 :
                      g.gene_class === 'assembly_factor' ? COLOR6 :
                      g.gene_class === 'f0_structural'   ? COLOR3 : COLOR2,
                    fontSize: '0.65rem'
                  }}>{g.phenotype?.split('—')[0]?.trim()}</span>
                </td>
                <td className="text-center fw-bold">{g.atp_activity_mean_pct}%</td>
                <td className="text-center">{g.median_onset_months} mo</td>
                <td className="text-center" style={{ color: g.three_mga_pct > 50 ? COLOR5 : 'inherit', fontWeight: g.three_mga_pct > 50 ? 'bold' : 'normal' }}>
                  {g.three_mga_pct}%
                </td>
                <td className="text-center" style={{ color: g.hcm_pct > 50 ? COLOR4 : 'inherit', fontWeight: g.hcm_pct > 50 ? 'bold' : 'normal' }}>
                  {g.hcm_pct}%
                </td>
                <td className="text-center" style={{ color: g.cardiac_pct > 50 ? COLOR4 : 'inherit' }}>
                  {g.cardiac_pct}%
                </td>
                <td className="text-center" style={{ color: g.leigh_mri_pct > 50 ? COLOR3 : 'inherit' }}>
                  {g.leigh_mri_pct}%
                </td>
                <td className="text-center">{g.hepatopathy_pct}%</td>
                <td className="text-center" style={{ color: g.hyperammonemia_pct > 40 ? COLOR5 : 'inherit', fontWeight: g.hyperammonemia_pct > 40 ? 'bold' : 'normal' }}>
                  {g.hyperammonemia_pct}%
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
          title={`${g.gene} (${g.alias}) — ${g.subunit_series} (${g.chromosome}) — ${g.inheritance}`}
          borderColor={
            g.gene_class === 'assembly_factor' ? COLOR6 :
            g.gene_class === 'f0_structural'   ? COLOR3 : COLOR2
          }
        >
          <div className="row g-3 small">
            <div className="col-12 col-md-6">
              <div><span className="fw-semibold">Phenotype: </span>{g.phenotype}</div>
              <div><span className="fw-semibold">Inheritance: </span>{g.inheritance}</div>
              <div><span className="fw-semibold">OMIM gene: </span>{g.omim_gene} | Disease: {g.disease_omim}</div>
              <div><span className="fw-semibold">Size: </span>{g.aa} / {g.kDa}</div>
              <div style={{ fontSize: '0.72rem' }}><span className="fw-semibold">Module: </span>{g.cv_module?.slice(0, 120)}…</div>
            </div>
            <div className="col-12 col-md-6">
              <div className="alert alert-secondary py-1 px-2 mb-1" style={{ fontSize: '0.73rem' }}>
                <strong>Hallmark:</strong> {g.hallmark?.slice(0, 200)}
              </div>
              <div style={{ fontSize: '0.72rem' }}><span className="fw-semibold">Founder: </span>{g.founder_variant}</div>
              <div style={{ fontSize: '0.72rem' }}><span className="fw-semibold">DDx: </span>{g.key_ddx?.slice(0, 120)}…</div>
            </div>
          </div>
          <div className="row g-2 mt-2 small">
            <div className="col-6 col-md-2"><span className="fw-semibold">Patients: </span>{g.n_patients}</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">ATP mean: </span>{g.atp_activity_mean_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">3-MGA: </span>
              <span style={{ color: g.three_mga_pct > 50 ? COLOR5 : 'inherit', fontWeight: g.three_mga_pct > 50 ? 'bold' : 'normal' }}>
                {g.three_mga_pct}%
              </span>
            </div>
            <div className="col-6 col-md-2"><span className="fw-semibold">HCM: </span>
              <span style={{ color: g.hcm_pct > 50 ? COLOR4 : 'inherit', fontWeight: g.hcm_pct > 50 ? 'bold' : 'normal' }}>
                {g.hcm_pct}%
              </span>
            </div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Cardiac: </span>
              <span style={{ color: g.cardiac_pct > 50 ? COLOR4 : 'inherit' }}>{g.cardiac_pct}%</span>
            </div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Leigh MRI: </span>{g.leigh_mri_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Hepato: </span>{g.hepatopathy_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Lactic acid: </span>{g.lactic_ac_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Neuropathy: </span>{g.neuropathy_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">NH3: </span>
              <span style={{ color: g.hyperammonemia_pct > 40 ? COLOR5 : 'inherit', fontWeight: g.hyperammonemia_pct > 40 ? 'bold' : 'normal' }}>
                {g.hyperammonemia_pct}%
              </span>
            </div>
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
    <SectionCard title="CV Subunit Atlas — Glossary & Definitions">
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
export default function CVSubunitAtlasPage() {
  const [tab, setTab]     = useState(0);
  const [overview, setOv] = useState(null);
  const [breakdown, setBk]= useState(null);
  const [defs, setDefs]   = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/cv-subunit-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/cv-subunit-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/cv-subunit-atlas/definitions`).then(r => r.json()),
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
            🧬 CV-Subunit-Atlas — Complete 16-Gene Nuclear-Encoded Complex V (F1F0-ATP Synthase)
          </h4>
          <div className="small text-muted">
            5 F1 Structural Subunits (ATP5F1A/B/C/D/E) ·
            8 F0 Structural Subunits (ATP5PO/PB/MC1/MC2/MC3/PD/ME/MF) ·
            3 Assembly Factors (TMEM70/ATPAF1/ATPAF2) ·
            TMEM70-Most-Common-CV-3-MGA-70pct-Roma-Founder ·
            ATP5F1A-HCM-75pct-First-Nuclear-CV-Structural ·
            ATPAF2-First-F1-Chaperone-Disease-2004 ·
            3-MGA-Type-V-CV-Biomarker ·
            MT-ATP6/MT-ATP8-mtDNA-WES-Missed ·
            640-Patient Aggregate (16×40, seeds 744–759)
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
