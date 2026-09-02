'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Features', 'DDx & Treatment', 'Definitions'];
const COLOR_AR  = '#b71c1c';   // deep red — AR Leigh / metabolic
const COLOR_AD  = '#4a148c';   // deep purple — AD PGL5 / tumor
const COLOR     = COLOR_AR;
const LIGHT     = '#ffebee';

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Bar({ label, value, color = COLOR }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${value}%`, backgroundColor: color }} />
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

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ar = data.ar_summary || {};
  const ad = data.ad_summary || {};

  return (
    <div>
      {/* Gene header */}
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `5px solid ${COLOR}` }}>
        <h5 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 SDHA — Succinate Dehydrogenase Subunit A (Flavoprotein Catalytic Subunit)
        </h5>
        <p className="mb-1 small">
          <strong>OMIM Gene:</strong> *{data.omim_gene} &nbsp;|&nbsp;
          <strong>Chr:</strong> {data.chromosome} &nbsp;|&nbsp;
          <strong>Size:</strong> {data.protein_size} &nbsp;|&nbsp;
          <strong>FAD site:</strong> {data.fad_site}
        </p>
        <p className="mb-1 small">
          <strong>Disease 1:</strong> Complex II Deficiency / Leigh Syndrome (AR, OMIM #{data.omim_disease_cii}) — biallelic &nbsp;|&nbsp;
          <strong>Disease 2:</strong> Paraganglioma 5 PGL5 (AD, OMIM #{data.omim_disease_pgl5}) — monoallelic &nbsp;|&nbsp;
          <strong>Disease 3:</strong> Carney-Stratakis Syndrome (AD, OMIM #{data.omim_disease_cs})
        </p>
        <p className="mb-0 small text-danger fw-semibold">
          ⚠️ DUAL-DISEASE GENE: Biallelic loss → AR Leigh syndrome; Monoallelic loss → AD PGL5. NOT maternally imprinted (unlike SDHD/SDHAF2).
        </p>
      </div>

      {/* Dual cohort KPIs */}
      <h6 className="fw-bold mb-2" style={{ color: COLOR_AR }}>AR Leigh / CII Cohort (n={ar.n})</h6>
      <div className="row mb-3">
        <KPI label="n (AR Leigh)" value={ar.n} color={COLOR_AR} />
        <KPI label="Mean onset" value={`${ar.mean_onset_months}mo`} color={COLOR_AR} />
        <KPI label="Mean CII residual" value={`${ar.mean_cii_residual_pct}%`} color={COLOR_AR} />
        <KPI label="Leigh MRI" value={`${ar.leigh_mri_pct}%`} color={COLOR_AR} />
        <KPI label="Cardiomyopathy" value={`${ar.cardiomyopathy_pct}%`} color={COLOR_AR} />
        <KPI label="MRS succinate↑" value={`${ar.mrs_succinate_pct}%`} color={COLOR_AR} />
      </div>

      <h6 className="fw-bold mb-2" style={{ color: COLOR_AD }}>AD PGL5 / Carney-Stratakis Cohort (n={ad.n})</h6>
      <div className="row mb-4">
        <KPI label="n (AD PGL5)" value={ad.n} color={COLOR_AD} />
        <KPI label="Penetrance" value={`~${ad.penetrance_pct}%`} color={COLOR_AD} />
        <KPI label="GIST (CS)" value={`${ad.gist_pct}%`} color={COLOR_AD} />
        <KPI label="Malignant" value={`${ad.malignant_pct}%`} color={COLOR_AD} />
        <KPI label="Bilateral PGL" value={`${ad.bilateral_pct}%`} color={COLOR_AD} />
        <KPI label="IHC" value={ad.ihc_pattern ? 'SDHA+SDHB null' : '—'} color={COLOR_AD} />
      </div>

      {/* Key facts */}
      <SectionCard title="Key Clinical Facts" borderColor={COLOR}>
        <ul className="mb-0 small">
          {(data.key_facts || []).map((f, i) => (
            <li key={i} className="mb-1">{f}</li>
          ))}
        </ul>
      </SectionCard>

      {/* IHC pattern highlight */}
      <SectionCard title="IHC Pattern (Critical DDx Tool)" borderColor={COLOR_AD}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: COLOR_AD, color: '#fff' }}>
              <tr><th>Mutation</th><th>SDHA IHC</th><th>SDHB IHC</th><th>Interpretation</th></tr>
            </thead>
            <tbody>
              <tr className="table-danger">
                <td><strong>SDHA lost</strong></td>
                <td>NULL</td><td>NULL</td>
                <td>→ SDHA mutation (germline sequencing required)</td>
              </tr>
              <tr>
                <td>SDHB, SDHC, or SDHD lost</td>
                <td>Proficient</td><td>NULL</td>
                <td>→ SDHB/C/D mutation (NOT SDHA)</td>
              </tr>
              <tr className="table-success">
                <td>SDH-proficient</td>
                <td>Proficient</td><td>Proficient</td>
                <td>→ KIT/PDGFRA or other mutation</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="small text-muted mt-2 mb-0">
          SDHA null + SDHB null is UNIQUE to SDHA loss — only SDHA mutations cause dual SDHA/SDHB null IHC.
        </p>
      </SectionCard>
    </div>
  );
}

// ── Tab: Variants & Features ──────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const variants = data.variant_breakdown || [];
  const leigh    = data.leigh_features  || [];
  const pgl5     = data.pgl5_features   || [];

  const arVariants = variants.filter(v => v.phenotype && v.phenotype.includes('AR'));
  const adVariants = variants.filter(v => v.phenotype && v.phenotype.includes('AD'));

  return (
    <div>
      {/* AR Leigh variants */}
      <SectionCard title={`AR Leigh / CII Variants (n=${arVariants.length}) — Biallelic Required`} borderColor={COLOR_AR}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead style={{ background: COLOR_AR, color: '#fff' }}>
              <tr><th>cDNA</th><th>Protein</th><th>Domain</th><th>Severity</th><th>Notes</th></tr>
            </thead>
            <tbody>
              {arVariants.map((v, i) => (
                <tr key={i}>
                  <td><code>{v.hgvs_c}</code></td>
                  <td><code>{v.hgvs_p}</code></td>
                  <td className="small">{v.domain?.substring(0, 50)}…</td>
                  <td>
                    <span className={`badge ${v.severity === 'severe' ? 'bg-danger' : v.severity === 'moderate' ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                      {v.severity}
                    </span>
                  </td>
                  <td className="small text-muted">{v.notes?.substring(0, 80)}…</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* AD PGL5 variants */}
      <SectionCard title={`AD PGL5 Variants (n=${adVariants.length}) — Monoallelic + Somatic Second-Hit`} borderColor={COLOR_AD}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead style={{ background: COLOR_AD, color: '#fff' }}>
              <tr><th>cDNA</th><th>Protein</th><th>Domain</th><th>Phenotype</th><th>Notes</th></tr>
            </thead>
            <tbody>
              {adVariants.map((v, i) => (
                <tr key={i}>
                  <td><code>{v.hgvs_c}</code></td>
                  <td><code>{v.hgvs_p}</code></td>
                  <td className="small">{v.domain?.substring(0, 50)}…</td>
                  <td><span className="badge bg-purple text-white" style={{ background: COLOR_AD }}>{v.phenotype?.substring(0, 30)}</span></td>
                  <td className="small text-muted">{v.notes?.substring(0, 80)}…</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Dual clinical features */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="AR Leigh / CII Features" borderColor={COLOR_AR}>
            {leigh.map((f, i) => <Bar key={i} label={f.feature} value={f.freq_pct} color={COLOR_AR} />)}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="AD PGL5 Features" borderColor={COLOR_AD}>
            {pgl5.map((f, i) => <Bar key={i} label={f.feature} value={f.freq_pct} color={COLOR_AD} />)}
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

// ── Tab: DDx & Treatment ──────────────────────────────────────────────────────
function DDxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ddx       = data.ddx_table || [];
  const txLeigh   = data.treatment_cii_leigh || {};
  const txPgl5    = data.treatment_pgl5 || {};

  return (
    <div>
      {/* DDx table */}
      <SectionCard title="Critical DDx Table" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr><th>Gene</th><th>Locus</th><th>Disease</th><th>Key DDx vs SDHA</th><th>Malignancy</th><th>Imprinting</th></tr>
            </thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i}>
                  <td><strong>{d.gene}</strong></td>
                  <td><code>{d.locus}</code></td>
                  <td>{d.disease}</td>
                  <td className="small">{d.key_ddx}</td>
                  <td>{d.malignancy}</td>
                  <td>{d.imprinting}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Treatment: AR Leigh */}
      <SectionCard title="Treatment — AR CII Deficiency / Leigh Syndrome" borderColor={COLOR_AR}>
        <h6 className="text-danger">Absolute Contraindications</h6>
        {(txLeigh.absolute_contraindications || []).map((ci, i) => (
          <div key={i} className="alert alert-danger py-2 mb-2 small">
            <strong>{ci.drug}:</strong> {ci.reason}
          </div>
        ))}
        <h6 className="text-success mt-3">Recommended</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead className="table-success">
              <tr><th>Drug</th><th>Dose</th><th>Level</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(txLeigh.recommended_treatments || []).map((t, i) => (
                <tr key={i}>
                  <td><strong>{t.drug}</strong></td>
                  <td>{t.dose}</td>
                  <td><span className="badge bg-secondary">{t.level}</span></td>
                  <td className="small">{t.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <h6 className="mt-3">Supportive Care</h6>
        <ul className="small mb-0">
          {(txLeigh.supportive || []).map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      </SectionCard>

      {/* Treatment: AD PGL5 */}
      <SectionCard title="Treatment — AD PGL5 / Carney-Stratakis" borderColor={COLOR_AD}>
        <h6 className="text-danger">Critical Sequence Requirement</h6>
        {(txPgl5.absolute_contraindications || []).map((ci, i) => (
          <div key={i} className="alert alert-danger py-2 mb-2 small">
            <strong>{ci.drug}:</strong> {ci.reason}
          </div>
        ))}
        <h6 className="text-success mt-3">Recommended</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: COLOR_AD, color: '#fff' }}>
              <tr><th>Drug</th><th>Dose</th><th>Level</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(txPgl5.recommended_treatments || []).map((t, i) => (
                <tr key={i}>
                  <td><strong>{t.drug}</strong></td>
                  <td>{t.dose}</td>
                  <td><span className="badge bg-secondary">{t.level}</span></td>
                  <td className="small">{t.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <h6 className="mt-3">Surveillance Protocol</h6>
        <ul className="small mb-0">
          {(txPgl5.surveillance || []).map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const gene  = data.gene  || {};
  const dis   = data.diseases || {};
  const imp   = data.imprinting_comparison || {};
  const ihc   = data.ihc_interpretation   || {};
  const path  = data.pathway || {};
  const refs  = data.key_references || [];
  const mon   = data.monitoring_protocol || {};

  return (
    <div>
      {/* Gene */}
      <SectionCard title="Gene / Protein" borderColor={COLOR}>
        <dl className="row small mb-0">
          <dt className="col-sm-3">Full name</dt><dd className="col-sm-9">{gene.full_name}</dd>
          <dt className="col-sm-3">OMIM Gene</dt><dd className="col-sm-9">*{gene.omim_gene}</dd>
          <dt className="col-sm-3">Chromosome</dt><dd className="col-sm-9">{gene.chromosome}</dd>
          <dt className="col-sm-3">Size</dt><dd className="col-sm-9">{gene.size_aa} aa, ~{gene.size_kda} kDa</dd>
          <dt className="col-sm-3">Cofactor</dt><dd className="col-sm-9">{gene.cofactor}</dd>
          <dt className="col-sm-3">Function</dt><dd className="col-sm-9">{gene.function}</dd>
          <dt className="col-sm-3">Assembly</dt><dd className="col-sm-9">{gene.assembly}</dd>
          <dt className="col-sm-3">Domains</dt>
          <dd className="col-sm-9">
            <ul className="mb-0">{(gene.domains || []).map((d, i) => <li key={i}>{d}</li>)}</ul>
          </dd>
        </dl>
      </SectionCard>

      {/* Diseases */}
      <SectionCard title="Disease Entities" borderColor={COLOR_AR}>
        {dis.cii_deficiency_leigh && (
          <div className="mb-3">
            <h6 className="text-danger">1. CII Deficiency / Leigh Syndrome (OMIM #{dis.cii_deficiency_leigh.omim})</h6>
            <dl className="row small mb-0">
              <dt className="col-sm-3">Inheritance</dt><dd className="col-sm-9">{dis.cii_deficiency_leigh.inheritance}</dd>
              <dt className="col-sm-3">Onset</dt><dd className="col-sm-9">{dis.cii_deficiency_leigh.onset}</dd>
              <dt className="col-sm-3">MRI</dt><dd className="col-sm-9">{dis.cii_deficiency_leigh.mri}</dd>
              <dt className="col-sm-3">Biochemistry</dt><dd className="col-sm-9">{dis.cii_deficiency_leigh.biochemistry}</dd>
              <dt className="col-sm-3">Brain MRS</dt><dd className="col-sm-9">{dis.cii_deficiency_leigh.mrs}</dd>
              <dt className="col-sm-3">Cardiomyopathy</dt><dd className="col-sm-9">{dis.cii_deficiency_leigh.cardiomyopathy}</dd>
              <dt className="col-sm-3">KD</dt><dd className="col-sm-9 text-danger fw-bold">{dis.cii_deficiency_leigh.kd}</dd>
              <dt className="col-sm-3">Prognosis</dt><dd className="col-sm-9">{dis.cii_deficiency_leigh.prognosis}</dd>
            </dl>
          </div>
        )}
        {dis.pgl5 && (
          <div className="mb-3">
            <h6 style={{ color: COLOR_AD }}>2. Paraganglioma 5 — PGL5 (OMIM #{dis.pgl5.omim})</h6>
            <dl className="row small mb-0">
              <dt className="col-sm-3">Inheritance</dt><dd className="col-sm-9">{dis.pgl5.inheritance}</dd>
              <dt className="col-sm-3">Penetrance</dt><dd className="col-sm-9">{dis.pgl5.penetrance}</dd>
              <dt className="col-sm-3">Sites</dt><dd className="col-sm-9">{dis.pgl5.sites}</dd>
              <dt className="col-sm-3">Malignancy</dt><dd className="col-sm-9">{dis.pgl5.malignancy}</dd>
              <dt className="col-sm-3">IHC pattern</dt><dd className="col-sm-9 fw-bold">{dis.pgl5.ihc_pattern}</dd>
              <dt className="col-sm-3">NOT imprinted</dt><dd className="col-sm-9 text-danger fw-bold">{dis.pgl5.not_imprinted}</dd>
            </dl>
          </div>
        )}
        {dis.carney_stratakis && (
          <div>
            <h6 style={{ color: COLOR_AD }}>3. Carney-Stratakis Syndrome (OMIM #{dis.carney_stratakis.omim})</h6>
            <dl className="row small mb-0">
              <dt className="col-sm-3">GIST type</dt><dd className="col-sm-9">{dis.carney_stratakis.gist}</dd>
              <dt className="col-sm-3">GIST DDx</dt><dd className="col-sm-9">{dis.carney_stratakis.gist_ddx}</dd>
              <dt className="col-sm-3">Treatment</dt><dd className="col-sm-9">{dis.carney_stratakis.treatment}</dd>
              <dt className="col-sm-3">IHC</dt><dd className="col-sm-9">{dis.carney_stratakis.ihc}</dd>
            </dl>
          </div>
        )}
      </SectionCard>

      {/* Imprinting comparison */}
      <SectionCard title="Imprinting Comparison (Critical for Genetic Counselling)" borderColor={COLOR_AD}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: COLOR_AD, color: '#fff' }}>
              <tr><th>Gene / Disease</th><th>Imprinting status</th></tr>
            </thead>
            <tbody>
              {Object.entries(imp).map(([k, v], i) => (
                <tr key={i} className={k === 'sdha_pgl5' ? 'table-warning' : ''}>
                  <td><strong>{k.replace(/_/g, ' ').toUpperCase()}</strong></td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Pathway */}
      <SectionCard title="CII Assembly Pathway Context" borderColor={COLOR}>
        <h6 className="small fw-bold">Assembly Sequence</h6>
        <ol className="small mb-3">
          {(path.cii_assembly_sequence || []).map((s, i) => <li key={i} className="mb-1">{s}</li>)}
        </ol>
        {path.sdha_sdhaf2_link && (
          <><h6 className="small fw-bold">SDHA ↔ SDHAF2 Link</h6>
          <p className="small mb-2">{path.sdha_sdhaf2_link}</p></>
        )}
        {path.sdha_sdhaf1_link && (
          <><h6 className="small fw-bold">SDHA ↔ SDHAF1 Link</h6>
          <p className="small mb-0">{path.sdha_sdhaf1_link}</p></>
        )}
      </SectionCard>

      {/* References */}
      <SectionCard title="Key References" borderColor={COLOR}>
        <ol className="small mb-0">
          {refs.map((r, i) => (
            <li key={i} className="mb-2">
              <em>{r.citation}</em><br />
              <span className="text-muted">{r.relevance}</span>
            </li>
          ))}
        </ol>
      </SectionCard>

      {/* Monitoring */}
      <SectionCard title="Monitoring Protocol" borderColor={COLOR}>
        {mon.AR_CII_Leigh && (
          <div className="mb-3">
            <h6 className="text-danger">AR CII Leigh Surveillance</h6>
            <dl className="row small mb-0">
              {Object.entries(mon.AR_CII_Leigh).map(([k, v], i) => (
                <><dt key={`k${i}`} className="col-sm-3">{k}</dt><dd key={`v${i}`} className="col-sm-9">{v}</dd></>
              ))}
            </dl>
          </div>
        )}
        {mon.AD_PGL5 && (
          <div>
            <h6 style={{ color: COLOR_AD }}>AD PGL5 Surveillance</h6>
            <dl className="row small mb-0">
              {Object.entries(mon.AD_PGL5).map(([k, v], i) => (
                <><dt key={`k${i}`} className="col-sm-3">{k}</dt><dd key={`v${i}`} className="col-sm-9">{v}</dd></>
              ))}
            </dl>
          </div>
        )}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function SDHAPage() {
  const [tab,      setTab]      = useState(0);
  const [overview, setOverview] = useState(null);
  const [bkdown,   setBkdown]   = useState(null);
  const [defs,     setDefs]     = useState(null);
  const [error,    setError]    = useState(null);

  useEffect(() => {
    const headers = { 'Content-Type': 'application/json' };
    Promise.all([
      fetch(`${API}/api/sdha/overview`,    { headers }).then(r => r.json()),
      fetch(`${API}/api/sdha/breakdown`,   { headers }).then(r => r.json()),
      fetch(`${API}/api/sdha/definitions`, { headers }).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBkdown(bd); setDefs(df); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return (
    <div className="container py-4">
      <div className="alert alert-danger">Error: {error}</div>
    </div>
  );

  const tabContent = [
    <OverviewTab  key="ov"  data={overview} />,
    <VariantsTab  key="vr"  data={bkdown}   />,
    <DDxTab       key="dd"  data={bkdown}   />,
    <DefinitionsTab key="df" data={defs}    />,
  ];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="row mb-3">
        <div className="col">
          <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
            🧬 SDHA — Succinate Dehydrogenase Subunit A
          </h4>
          <p className="text-muted small mb-0">
            Complex II Deficiency / Leigh Syndrome (AR biallelic) + Paraganglioma 5 / Carney-Stratakis (AD monoallelic) · 5p15.33 · Seed {overview?.seed || 705}
          </p>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tabContent[tab]}
    </div>
  );
}
