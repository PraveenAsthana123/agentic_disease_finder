'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#1b5e20';   // deep green — nuclear CII atlas (all nuclear)
const LIGHT  = '#e8f5e9';
const COLOR2 = '#0d47a1';   // dark blue — structural subunits
const COLOR3 = '#4a148c';   // deep purple — assembly factors
const COLOR4 = '#b71c1c';   // dark red — PGL/PHEO / metastatic
const COLOR5 = '#e65100';   // orange — Leigh / CII deficiency
const COLOR6 = '#006064';   // teal — imprinting genes
const COLOR7 = '#880e4f';   // dark pink — IHC / diagnostics

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
  const cii  = data.complex_ii || {};
  const sb   = data.series_breakdown || {};
  const imp  = data.imprinting_genes || {};
  const ihc  = data.ihc_diagnostic || {};
  const met  = data.metastatic_risk || {};
  const coh  = data.cohort || {};
  const agg  = data.aggregate_clinical || {};
  const drug = data.drug_considerations || {};
  const surv = data.surveillance_protocols || {};
  const struc= data.sdh_structure || {};
  const carn = data.carney_triad || {};
  const cii0 = data.cii_always_normal_rule || {};
  const wes  = data.wes_utility || {};
  const leigh_drug = drug.leigh_phenotype || {};
  const pgl_drug   = drug.pgl_pheo_phenotype || {};

  return (
    <>
      {/* Atlas banner */}
      <SectionCard title="CII-Deficiency-Atlas — Complete 6-Gene Nuclear-Encoded Complex II Reference">
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">Complex II: </span>{cii.full_name}</div>
            <div><span className="fw-semibold">Total subunits: </span>{cii.subunits_total} (ALL {cii.subunits_nuclear} nuclear — ZERO mtDNA)</div>
            <div><span className="fw-semibold">Assembly factors: </span>{cii.assembly_factors}</div>
            <div><span className="fw-semibold">Total genes: </span>{cii.total_genes} | Size: {cii.size_kDa}</div>
            <div className="mt-1 text-success fw-semibold">✅ WES detects all 6 CII nuclear genes</div>
          </div>
          <div className="col-12 col-md-6">
            <div className="alert alert-success py-1 px-2 mb-2 small">
              <strong>Unique Feature:</strong> {cii.unique_feature}
            </div>
            <div className="small text-muted"><strong>OXPHOS function:</strong> {cii.function_oxphos}</div>
            <div className="small text-muted"><strong>TCA function:</strong> {cii.function_tca}</div>
          </div>
        </div>
        <div className="row g-2 mt-2">
          <KPI label="Structural Subunits" value={sb.structural_subunits} color={COLOR2} />
          <KPI label="Assembly Factors"    value={sb.assembly_factors}    color={COLOR3} />
          <KPI label="Total Genes"         value={sb.total_genes}         color={COLOR}  />
          <KPI label="Total Patients"      value={coh.total_patients}     color={COLOR}  />
          <KPI label="PGL/PHEO Cases"      value={coh.pgl_pheo_count}     color={COLOR4} />
          <KPI label="CII-Leigh Cases"     value={coh.leigh_cii_count}    color={COLOR5} />
        </div>
      </SectionCard>

      {/* CII Always Normal Rule */}
      <SectionCard title="⚡ CII Always Normal Rule — The Biochemical Fingerprint Anchor" borderColor={COLOR5}>
        <div className="alert alert-warning py-2 px-3 small mb-3">
          <strong>RULE: {cii0.rule}</strong>
        </div>
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">Implication: </span>{cii0.implication}</div>
          </div>
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">Internal Reference: </span>{cii0.internal_reference}</div>
          </div>
        </div>
      </SectionCard>

      {/* Imprinting */}
      <SectionCard title="🧬 Parental Imprinting — SDHD (PGL1) + SDHAF2 (PGL2)" borderColor={COLOR6}>
        <div className="alert alert-info py-2 px-3 small mb-3">
          <strong>CRITICAL:</strong> {imp.note}
        </div>
        <div className="row g-3 small">
          {imp.sdhd && (
            <div className="col-12 col-md-6">
              <div className="fw-semibold" style={{ color: COLOR6 }}>{imp.sdhd.gene} — {imp.sdhd.mechanism}</div>
              <div>{imp.sdhd.rule}</div>
            </div>
          )}
          {imp.sdhaf2 && (
            <div className="col-12 col-md-6">
              <div className="fw-semibold" style={{ color: COLOR6 }}>{imp.sdhaf2.gene} — {imp.sdhaf2.mechanism}</div>
              <div>{imp.sdhaf2.rule}</div>
            </div>
          )}
        </div>
      </SectionCard>

      {/* IHC Diagnostic Algorithm */}
      <SectionCard title="🔬 SDHB/SDHA IHC Diagnostic Algorithm" borderColor={COLOR7}>
        <div className="row g-3 small">
          <div className="col-12 col-md-4">
            <div className="fw-semibold" style={{ color: COLOR7 }}>SDHB (Universal Surrogate)</div>
            <div>{ihc.sdhb_universal}</div>
          </div>
          <div className="col-12 col-md-4">
            <div className="fw-semibold" style={{ color: COLOR7 }}>SDHA (SDHA-specific)</div>
            <div>{ihc.sdha_specific}</div>
          </div>
          <div className="col-12 col-md-4">
            <div className="fw-semibold" style={{ color: COLOR7 }}>Algorithm</div>
            <div>{ihc.algorithm}</div>
          </div>
        </div>
      </SectionCard>

      {/* Metastatic Risk */}
      <SectionCard title="⚠️ Metastatic Risk by Gene (PGL/PHEO)" borderColor={COLOR4}>
        <div className="row g-2 small">
          {Object.entries(met).map(([gene, risk]) => (
            <div key={gene} className="col-12 col-md-6">
              <span className="fw-bold" style={{ color: gene === 'SDHB' ? COLOR4 : COLOR }}>{gene}: </span>{risk}
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Aggregate Clinical */}
      <SectionCard title={`📊 Aggregate Clinical — ${coh.total_patients} Patients (6 × 40, seeds ${coh.seeds})`} borderColor={COLOR}>
        <div className="row g-2 small">
          <KPI label="PGL/PHEO Rate"          value={`${agg.pgl_pheo_rate_pct}%`}        color={COLOR4} />
          <KPI label="CII-Leigh Rate"          value={`${agg.leigh_cii_rate_pct}%`}        color={COLOR5} />
          <KPI label="HNPGL among PGL"         value={`${agg.hnpgl_among_pgl_pct}%`}      color={COLOR}  />
          <KPI label="SDHB Metastatic Rate"    value={`${agg.sdhb_metastatic_rate_pct}%`} color={COLOR4} />
          <KPI label="CII Mean (Leigh pts)"    value={`${agg.cii_deficiency_mean_leigh_pct}%`} color={COLOR5} />
        </div>
      </SectionCard>

      {/* SDH Structure */}
      <SectionCard title="🔧 SDH Tetrameric Structure" borderColor={COLOR2}>
        <div className="row g-3 small">
          {Object.entries(struc).map(([k, v]) => (
            <div key={k} className="col-12 col-md-6">
              <span className="fw-semibold" style={{ color: COLOR2 }}>{k.replace(/_/g,' ')}: </span>{v}
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Drug CIs — Leigh phenotype */}
      <SectionCard title="💊 Drug Contraindications (CII-Leigh: SDHA / SDHAF1)" borderColor={COLOR4}>
        <div className="small mb-2 text-muted">Applies to SDHA (Leigh subtype) and SDHAF1 CII-Leigh; PGL/PHEO patients see separate section below</div>
        {(leigh_drug.absolute_ci || []).map((d, i) => (
          <div key={i} className="mb-1">
            <Badge text="ABSOLUTE CI" color={COLOR4} />
            <span className="fw-semibold">{d.drug}: </span>{d.mechanism}
          </div>
        ))}
        <hr className="my-2" />
        <div className="fw-semibold small mb-1" style={{ color: COLOR5 }}>Mandatory:</div>
        {(leigh_drug.mandatory || []).map((m, i) => (
          <div key={i} className="small mb-1">• {m}</div>
        ))}
      </SectionCard>

      {/* Drug — PGL/PHEO */}
      <SectionCard title="💉 Management Notes — PGL/PHEO Phenotype" borderColor={COLOR}>
        <div className="small mb-2">{pgl_drug.note}</div>
        {(pgl_drug.pheo_specific || []).map((p, i) => (
          <div key={i} className="small mb-1">• {p}</div>
        ))}
      </SectionCard>

      {/* Surveillance */}
      <SectionCard title="📅 Surveillance Protocols by Gene" borderColor={COLOR6}>
        <div className="row g-3 small">
          {Object.entries(surv).map(([gene, proto]) => (
            <div key={gene} className="col-12 col-md-6">
              <span className="fw-bold" style={{ color: COLOR6 }}>{gene}: </span>{proto}
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Carney Triad */}
      <SectionCard title="🎭 Carney Triad & SDHC" borderColor={COLOR3}>
        <div className="row g-3 small">
          <div className="col-12 col-md-4"><span className="fw-semibold">Definition: </span>{carn.definition}</div>
          <div className="col-12 col-md-4"><span className="fw-semibold">SDH link: </span>{carn.sdh_link}</div>
          <div className="col-12 col-md-4"><span className="fw-semibold">DDx from germline: </span>{carn.ddx_from_germline_sdhc}</div>
        </div>
      </SectionCard>

      {/* WES utility */}
      <SectionCard title="🧬 WES Utility — All 6 CII Genes Nuclear" borderColor={COLOR}>
        <div className="row g-2 small">
          {['SDHA','SDHB','SDHC','SDHD','SDHAF1','SDHAF2'].map(g => (
            <div key={g} className="col-12 col-md-4">
              <Badge text={g} color={COLOR} /> {wes[g]}
            </div>
          ))}
        </div>
        <div className="small text-muted mt-2">{wes.panel_note}</div>
      </SectionCard>
    </>
  );
}

// ── Tab: Gene Table ───────────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const genes = data.genes || [];
  return (
    <SectionCard title={`CII Gene Reference Table — ${genes.length} Genes, ${data.total_patients} Patients`}>
      <div className="table-responsive">
        <table className="table table-sm table-bordered table-hover small align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Class</th><th>Module</th><th>OMIM</th><th>Chr</th>
              <th>Inheritance / Imprinting</th><th>Phenotype</th>
              <th>CII Activity (mean %)</th><th>Median Onset (mo)</th>
              <th>PGL/PHEO %</th><th>Leigh MRI %</th><th>Metastatic %</th>
              <th>SDHB IHC</th><th>Metastatic Risk</th>
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
                <td style={{ maxWidth: 200, fontSize: '0.68rem' }}>{g.cii_module}</td>
                <td>{g.omim_gene}</td>
                <td>{g.chromosome}</td>
                <td>
                  {g.inheritance}
                  {g.imprinting && <span className="badge ms-1" style={{ backgroundColor: COLOR6 }}>Paternal imprint</span>}
                </td>
                <td>
                  <span className="badge" style={{ backgroundColor: g.phenotype?.includes('Leigh') || g.phenotype?.includes('CII') ? COLOR5 : COLOR4 }}>
                    {g.phenotype}
                  </span>
                </td>
                <td className="text-center">{g.cii_activity_mean_pct}%</td>
                <td className="text-center">{g.median_onset_months} mo</td>
                <td className="text-center">{g.pgl_pheo_rate_pct}%</td>
                <td className="text-center">{g.leigh_mri_pct}%</td>
                <td className="text-center fw-bold" style={{ color: g.metastatic_pct_of_pgl > 20 ? COLOR4 : 'inherit' }}>
                  {g.metastatic_pct_of_pgl > 0 ? `${g.metastatic_pct_of_pgl}%` : '—'}
                </td>
                <td style={{ fontSize: '0.68rem', maxWidth: 180 }}>{g.sdhb_ihc}</td>
                <td style={{ color: g.metastatic_risk?.includes('HIGHEST') ? COLOR4 : 'inherit', fontSize: '0.72rem' }}>
                  {g.metastatic_risk}
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
          title={`${g.gene} — ${g.subunit_series} (${g.chromosome}) — ${g.phenotype}`}
          borderColor={g.gene_class === 'assembly_factor' ? COLOR3 : COLOR2}
        >
          <div className="row g-3 small">
            <div className="col-12 col-md-6">
              <div><span className="fw-semibold">Disease: </span>{g.disease_summary}</div>
              <div><span className="fw-semibold">Inheritance: </span>{g.inheritance}
                {g.imprinting && <span className="badge ms-1" style={{ backgroundColor: COLOR6 }}>Paternal imprint</span>}
              </div>
              <div><span className="fw-semibold">OMIM gene: </span>{g.omim_gene}</div>
              <div><span className="fw-semibold">CII module: </span>{g.cii_module}</div>
            </div>
            <div className="col-12 col-md-6">
              <div className="alert alert-secondary py-1 px-2 mb-1" style={{ fontSize: '0.75rem' }}>
                <strong>Hallmark:</strong> {g.hallmark}
              </div>
              <div><span className="fw-semibold">Founder variant: </span>{g.founder_variant}</div>
              <div><span className="fw-semibold">Key DDx: </span>{g.key_ddx}</div>
            </div>
          </div>
          <div className="row g-2 mt-2 small">
            <div className="col-6 col-md-2"><span className="fw-semibold">Patients: </span>{g.n_patients}</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">CII mean: </span>{g.cii_activity_mean_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">PGL/PHEO: </span>{g.pgl_pheo_rate_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Leigh MRI: </span>{g.leigh_mri_pct}%</div>
            <div className="col-6 col-md-2"><span className="fw-semibold">Metastatic: </span>
              {g.metastatic_pct_of_pgl > 0 ? `${g.metastatic_pct_of_pgl}%` : 'N/A'}
            </div>
            <div className="col-6 col-md-2"><span className="fw-semibold">White matter: </span>{g.white_matter_pct}%</div>
          </div>
          <div className="small mt-2"><span className="fw-semibold">SDHB IHC: </span>
            <span style={{ color: COLOR7 }}>{g.sdhb_ihc}</span>
          </div>
          <div className="small mt-1"><span className="fw-semibold">Metastatic risk: </span>
            <span style={{ color: g.metastatic_risk?.includes('HIGHEST') ? COLOR4 : COLOR }}>{g.metastatic_risk}</span>
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
    <SectionCard title="CII Deficiency Atlas — Glossary & Definitions">
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
export default function CIIDeficiencyAtlasPage() {
  const [tab, setTab]     = useState(0);
  const [overview, setOv] = useState(null);
  const [breakdown, setBk]= useState(null);
  const [defs, setDefs]   = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/cii-deficiency-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/cii-deficiency-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/cii-deficiency-atlas/definitions`).then(r => r.json()),
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
            🧬 CII-Deficiency-Atlas — Complete 6-Gene Nuclear-Encoded Complex II
          </h4>
          <div className="small text-muted">
            4 Structural Subunits (SDHA/B/C/D) + 2 Assembly Factors (SDHAF1/SDHAF2) ·
            All Nuclear-Encoded (ZERO mtDNA) · CII Always Normal in mtDNA Disorders ·
            PGL/PHEO + Leigh · Parental Imprinting (SDHD/SDHAF2) · SDHB IHC Surrogate ·
            240-Patient Aggregate (6×40, seeds 703–708)
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
