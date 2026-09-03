'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Audiometry', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#006064';   // teal/dark-cyan — hearing/cochlear theme
const LIGHT  = '#e0f7fa';
const COLOR2 = '#00838f';
const COLOR3 = '#b71c1c';   // red — absolute contraindication / emergency
const COLOR4 = '#e65100';   // deep orange — warning
const COLOR5 = '#2e7d32';   // green — safe / cochlear implant benefit

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
  const s = data.cohort_statistics || {};
  const feats = data.cohort_summary_features || [];
  const pheno = data.phenotype_distribution || {};
  const contrast = data.contrast_with_oxphos_genes || {};

  return (
    <div>
      {/* Gene header */}
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `5px solid ${COLOR}` }}>
        <h5 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 MT-RNR1 — 12S Ribosomal RNA / Aminoglycoside-Induced SNHL (AISNHL) · m.1555A>G
        </h5>
        <p className="mb-1 small">
          <strong>OMIM Gene:</strong> *{data.omim_gene} &nbsp;|&nbsp;
          <strong>Genome:</strong> {data.chromosome} &nbsp;|&nbsp;
          <strong>Product:</strong> {data.product} &nbsp;|&nbsp;
          <strong>Population freq:</strong> {data.population_frequency}
        </p>
        <p className="mb-1 small">
          <strong>Inheritance:</strong> {data.inheritance} &nbsp;|&nbsp;
          <strong>Cohort:</strong> {data.cohort_n} patients (seed {data.seed})
        </p>
        <p className="mb-0 small fw-semibold" style={{ color: COLOR3 }}>
          🔴 UNIQUE among mtDNA genes: ISOLATED SNHL — NO OXPHOS deficiency (CI/CII/CIII/CIV/CV all NORMAL).
          <span style={{ color: COLOR3 }}> m.1555A>G: ~1 in 500 people — MOST COMMON pathogenic mtDNA variant.</span>
          <span style={{ color: COLOR3 }}> AMINOGLYCOSIDES ABSOLUTELY CONTRAINDICATED — even 1 dose causes permanent deafness.</span>
          HOMOPLASMIC (blood DNA sufficient — no muscle biopsy needed).
        </p>
      </div>

      {/* Emergency alert box */}
      <div className="alert alert-danger mb-4" style={{ borderLeft: `5px solid ${COLOR3}` }}>
        <strong>🚨 ABSOLUTE CONTRAINDICATION — ALL AMINOGLYCOSIDES:</strong> Gentamicin · Amikacin · Tobramycin · Streptomycin · Neomycin · Kanamycin · Spectinomycin · Paromomycin
        <br /><span className="small">ANY DOSE → permanent severe-to-profound SNHL within 24–72 h. Cochlear implant is the only rehabilitation. NO RECOVERY.</span>
        <br /><span className="small fw-semibold">Safe alternatives for gram-negatives: piperacillin-tazobactam · cefepime · meropenem · aztreonam</span>
      </div>

      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Patients (cohort)" value={data.cohort_n} />
        <KPI label="Severe/Profound SNHL" value={`${s.severe_profound_pct}%`} color={COLOR3} />
        <KPI label="Aminoglycoside Exposed" value={`${s.aminoglycoside_exposed_pct}%`} color={COLOR3} />
        <KPI label="Cochlear Implant" value={`${s.cochlear_implant_pct}%`} color={COLOR2} />
        <KPI label="Asymptomatic Carriers" value={`${s.asymptomatic_carrier_pct}%`} color={COLOR4} />
        <KPI label="m.1555A>G (main)" value={`${s.m1555_pct}%`} />
        <KPI label="Bilateral SNHL" value={`${s.bilateral_pct}%`} />
        <KPI label="Maternal Family SNHL" value={`${s.maternal_family_snhl_pct}%`} color={COLOR4} />
        <KPI label="Tinnitus" value={`${s.tinnitus_pct}%`} color={COLOR2} />
        <KPI label="Hearing Aid Benefit" value={`${s.hearing_aid_pct}%`} color={COLOR5} />
        <KPI label="NO OXPHOS Deficiency" value={`${s.no_oxphos_deficiency_pct}%`} color={COLOR5} />
        <KPI label="TRMU Modifier" value={`${s.trmu_modifier_pct}%`} color={COLOR4} />
      </div>

      {/* Key Message */}
      <SectionCard title="🔑 Unique Position of MT-RNR1 Among All mtDNA Genes" borderColor={COLOR}>
        <p className="mb-2 small">{data.key_message}</p>
        <div className="row">
          <div className="col-md-6">
            <strong className="small" style={{ color: COLOR5 }}>ABSENT in MT-RNR1 (unlike all other mtDNA genes):</strong>
            <ul className="small mt-1 mb-0">
              {Object.entries(contrast).map(([k, v]) => (
                <li key={k}><strong>{k.replace(/_/g,' ')}:</strong> {v}</li>
              ))}
            </ul>
          </div>
          <div className="col-md-6">
            <strong className="small" style={{ color: COLOR3 }}>PRESENT (unique to MT-RNR1):</strong>
            <ul className="small mt-1 mb-0">
              <li><strong>m.1555A>G population freq:</strong> ~1 in 500–1000 people — most common pathogenic mtDNA variant</li>
              <li><strong>AMINOGLYCOSIDE hypersensitivity:</strong> 100% SNHL penetrance with any aminoglycoside dose</li>
              <li><strong>Homoplasmic transmission:</strong> 100% of children of carrier mothers inherit variant</li>
              <li><strong>Blood DNA diagnostic:</strong> No muscle biopsy required (unlike heteroplasmic mtDNA genes)</li>
              <li><strong>Cochlear implant:</strong> Excellent outcomes — cochlear nerve intact; hair cells only affected</li>
            </ul>
          </div>
        </div>
      </SectionCard>

      {/* Clinical features */}
      <SectionCard title="📊 Cohort Clinical Feature Prevalence" borderColor={COLOR}>
        <div className="row">
          <div className="col-md-6">
            {feats.slice(0, Math.ceil(feats.length / 2)).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct}
                   color={f.feature.includes('NO OXPHOS') ? COLOR5 :
                          f.feature.includes('Aminoglycoside') || f.feature.includes('Severe') ? COLOR3 :
                          f.feature.includes('Cochlear') ? COLOR5 : COLOR} />
            ))}
          </div>
          <div className="col-md-6">
            {feats.slice(Math.ceil(feats.length / 2)).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct}
                   color={f.feature.includes('NO OXPHOS') ? COLOR5 :
                          f.feature.includes('Aminoglycoside') || f.feature.includes('Severe') ? COLOR3 :
                          f.feature.includes('Cochlear') ? COLOR5 : COLOR} />
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Phenotype breakdown */}
      <SectionCard title="🎯 Phenotype Distribution" borderColor={COLOR2}>
        <div className="row">
          <div className="col-md-4 text-center">
            <div className="fw-bold fs-4" style={{ color: COLOR3 }}>{pheno.aminoglycoside_induced_snhl_pct}%</div>
            <div className="small text-muted">Aminoglycoside-Induced SNHL</div>
            <div className="small" style={{ color: COLOR3 }}>Severe–Profound; 24–72 h onset</div>
          </div>
          <div className="col-md-4 text-center">
            <div className="fw-bold fs-4" style={{ color: COLOR2 }}>{pheno.non_aminoglycoside_snhl_pct}%</div>
            <div className="small text-muted">Non-Aminoglycoside NSHL</div>
            <div className="small" style={{ color: COLOR2 }}>Progressive over years; nuclear modifier-dependent</div>
          </div>
          <div className="col-md-4 text-center">
            <div className="fw-bold fs-4" style={{ color: COLOR4 }}>{pheno.asymptomatic_carrier_pct}%</div>
            <div className="small text-muted">Asymptomatic Carriers</div>
            <div className="small" style={{ color: COLOR4 }}>Normal hearing; at risk if aminoglycosides given</div>
          </div>
        </div>
      </SectionCard>

      {/* Alerts */}
      <SectionCard title="⚠️ Clinical Alerts" borderColor={COLOR3}>
        {(data.key_clinical_alerts || []).map((a, i) => (
          <div key={i} className="mb-2 small p-2 rounded"
               style={{ background: a.startsWith('🚨') ? '#ffebee' : a.startsWith('🚫') ? '#fff3e0' : a.startsWith('⚠️') ? '#fff8e1' : '#e8f5e9', borderLeft: `3px solid ${a.startsWith('🚨') ? COLOR3 : a.startsWith('🚫') ? COLOR4 : a.startsWith('⚠️') ? '#f9a825' : COLOR5}` }}>
            {a}
          </div>
        ))}
      </SectionCard>

      {/* Sample patients */}
      <SectionCard title={`👥 Sample Patients (first 10 of ${data.cohort_n})`} borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>ID</th><th>Sex</th><th>Variant</th><th>AG Exposed</th>
                <th>SNHL Severity</th><th>Laterality</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.patient_id}>
                  <td>{p.patient_id}</td>
                  <td>{p.sex}</td>
                  <td><code className="small">{p.variant}</code></td>
                  <td>
                    <span className={`badge ${p.aminoglycoside_exposed ? 'bg-danger' : 'bg-secondary'}`}>
                      {p.aminoglycoside_exposed ? `Yes — ${p.aminoglycoside_agent}` : 'No'}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${p.snhl_severity === 'Profound' ? 'bg-danger' : p.snhl_severity === 'Severe' ? 'bg-warning text-dark' : p.snhl_severity === 'Asymptomatic carrier' ? 'bg-secondary' : 'bg-info text-dark'}`}>
                      {p.snhl_severity}
                    </span>
                  </td>
                  <td>{p.laterality}</td>
                  <td className="small">{p.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Variants & Audiometry ─────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const variants = data.all_variants || [];
  const varDist = data.variant_distribution || [];
  const sevDist = data.severity_distribution || [];
  const audDist = data.audiogram_distribution || [];
  const agDist = data.aminoglycoside_agent_distribution || [];
  const indDist = data.aminoglycoside_indication_distribution || [];
  const s = data.cohort_statistics || {};
  const mods = data.nuclear_modifier_breakdown || {};

  return (
    <div>
      {/* Variant table */}
      <SectionCard title="🧬 MT-RNR1 Pathogenic Variants" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr>
                <th>Variant</th><th>Location</th><th>Type</th><th>Severity</th>
                <th>Phenotype</th><th>Cohort %</th>
              </tr>
            </thead>
            <tbody>
              {variants.map(v => (
                <tr key={v.change}>
                  <td><code className="small fw-bold">{v.change}</code></td>
                  <td className="small">{v.location}</td>
                  <td className="small">{v.type}</td>
                  <td>
                    <span className={`badge ${v.severity.includes('Profound') || v.severity.includes('Severe') ? 'bg-danger' : v.severity.includes('Moderate') ? 'bg-warning text-dark' : 'bg-info text-dark'}`}>
                      {v.severity}
                    </span>
                  </td>
                  <td className="small">{v.phenotype}</td>
                  <td className="fw-bold">{v.allele_freq_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-2">
          {variants.map(v => (
            <div key={v.change} className="mb-2 p-2 rounded small" style={{ background: '#f9f9f9', borderLeft: `3px solid ${COLOR}` }}>
              <strong><code>{v.change}</code></strong>: {v.notes}
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Variant distribution */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="📊 Variant Distribution (Cohort)" borderColor={COLOR}>
            {varDist.map(v => (
              <div key={v.variant} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <code>{v.variant}</code>
                  <span className="text-muted">{v.count} pts ({v.allele_freq_pct}%)</span>
                </div>
                <div className="progress" style={{ height: 10 }}>
                  <div className="progress-bar" style={{ width: `${v.allele_freq_pct}%`, backgroundColor: v.variant === 'm.1555A>G' ? COLOR3 : v.variant === 'm.1494C>T' ? COLOR4 : COLOR }} />
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="🔊 SNHL Severity Distribution" borderColor={COLOR2}>
            {sevDist.map(s => (
              <div key={s.severity} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{s.severity}</span>
                  <span className="text-muted">{s.count} pts ({s.pct}%)</span>
                </div>
                <div className="progress" style={{ height: 10 }}>
                  <div className="progress-bar" style={{ width: `${s.pct}%`, backgroundColor: s.severity === 'Profound' ? COLOR3 : s.severity === 'Severe' ? '#ef5350' : s.severity === 'Asymptomatic carrier' ? '#9e9e9e' : COLOR2 }} />
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Audiogram patterns */}
      <SectionCard title="📈 Audiogram Pattern Distribution" borderColor={COLOR}>
        <div className="row">
          {audDist.map(a => (
            <div key={a.pattern} className="col-md-6 mb-2">
              <div className="d-flex justify-content-between small mb-1">
                <span className="small">{a.pattern}</span>
                <span className="text-muted">{a.pct}%</span>
              </div>
              <div className="progress" style={{ height: 10 }}>
                <div className="progress-bar" style={{ width: `${a.pct}%`, backgroundColor: a.pattern.includes('Profound') ? COLOR3 : a.pattern.includes('Normal') ? COLOR5 : COLOR }} />
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Aminoglycoside agents */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="💉 Precipitating Aminoglycoside Agents" borderColor={COLOR3}>
            {agDist.length > 0 ? agDist.map(a => (
              <div key={a.agent} className="d-flex justify-content-between small mb-1">
                <span className="badge bg-danger">{a.agent}</span>
                <span>{a.count} patients ({a.pct}%)</span>
              </div>
            )) : <p className="text-muted small">No aminoglycoside exposures recorded</p>}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="🏥 Clinical Indication (AG Exposure)" borderColor={COLOR4}>
            {indDist.length > 0 ? indDist.map((i, idx) => (
              <div key={idx} className="mb-1 small">
                <span className="text-muted">• </span>{i.indication} <span className="badge bg-warning text-dark ms-1">{i.count}</span>
              </div>
            )) : <p className="text-muted small">No exposures</p>}
          </SectionCard>
        </div>
      </div>

      {/* Nuclear modifiers */}
      <SectionCard title="🔬 Nuclear Modifier Genes (Penetrance Modulators)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Nuclear Gene</th><th>Role</th></tr></thead>
            <tbody>
              {Object.entries(mods).map(([k, v]) => (
                <tr key={k}>
                  <td><strong>{k}</strong></td>
                  <td className="small">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small text-muted mb-0">
          Nuclear modifiers explain why only ~20–30% of m.1555A>G carriers develop NSHL without aminoglycosides.
          TRMU and MTO1/GTPBP3 variants impair mt-tRNA modification → compound the mt-translation deficit.
        </p>
      </SectionCard>
    </div>
  );
}

// ── Tab: DDx & Treatment ───────────────────────────────────────────────────────
function DDxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ddx = data.key_ddx || [];
  const contrasts = data.key_contrasts || {};
  const aClasses = data.aminoglycoside_classes || {};
  const ciList = aClasses.ABSOLUTE_CI || [];
  const safeAlts = aClasses.Safe_alternatives_for_gram_negatives || [];
  const tx = data.treatment_uptake || {};

  return (
    <div>
      {/* Aminoglycoside contraindication panel */}
      <div className="alert alert-danger mb-4">
        <h6 className="fw-bold mb-2">🚨 ABSOLUTE CONTRAINDICATIONS — ALL AMINOGLYCOSIDE ANTIBIOTICS</h6>
        <div className="row">
          {ciList.map((c, i) => (
            <div key={i} className="col-md-6 mb-1 small">
              <span className="text-danger">✗</span> {c}
            </div>
          ))}
        </div>
        <hr className="my-2" />
        <strong className="small">Safe non-aminoglycoside alternatives for gram-negative coverage:</strong>
        <div className="row mt-1">
          {safeAlts.map((s, i) => (
            <div key={i} className="col-md-6 mb-1 small">
              <span className="text-success">✓</span> {s}
            </div>
          ))}
        </div>
      </div>

      {/* DDx */}
      <SectionCard title="🔍 Differential Diagnosis" borderColor={COLOR}>
        {ddx.map((d, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: '#f5f5f5', borderLeft: `3px solid ${COLOR}` }}>
            <div className="fw-bold small mb-1" style={{ color: COLOR }}>{d.condition}</div>
            <div className="small">{d.distinguishing}</div>
          </div>
        ))}
      </SectionCard>

      {/* Key contrasts */}
      <SectionCard title="⚖️ MT-RNR1 vs Other mtDNA Gene Groups" borderColor={COLOR2}>
        {Object.entries(contrasts).map(([k, v]) => (
          <div key={k} className="mb-2 p-2 rounded small" style={{ background: '#e0f7fa', borderLeft: `3px solid ${COLOR2}` }}>
            <strong>{k.replace(/_/g,' ')}:</strong> {v}
          </div>
        ))}
      </SectionCard>

      {/* Treatment uptake */}
      <SectionCard title="💊 Treatment Uptake (Cohort)" borderColor={COLOR5}>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Intervention</th><th>Uptake</th></tr></thead>
            <tbody>
              {Object.entries(tx).map(([k, v]) => (
                <tr key={k}>
                  <td>{k}</td>
                  <td><span className="badge bg-info text-dark">{v}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Screening / prevention */}
      <SectionCard title="🛡️ Prevention & Screening" borderColor={COLOR5}>
        <ul className="small mb-0">
          <li><strong>Newborn screening (NBS) for m.1555A>G:</strong> Implemented in UK, China, several European countries. Identify carriers before any aminoglycoside exposure.</li>
          <li><strong>Pre-aminoglycoside genetic testing (elective settings):</strong> Rapid PCR-RFLP or Sanger test for m.1555A>G + m.1494C>T — MANDATORY before elective aminoglycoside use.</li>
          <li><strong>Cascade maternal family testing:</strong> ALL maternal relatives (mother, maternal siblings, maternal aunts, maternal cousins) require blood mtDNA testing.</li>
          <li><strong>Medical alert bracelet:</strong> All carriers — "AVOID ALL AMINOGLYCOSIDE ANTIBIOTICS — MT-RNR1 m.1555A>G — Risk of permanent deafness".</li>
          <li><strong>Electronic health record flag:</strong> Aminoglycoside allergy/contraindication flag in EHR for all carriers.</li>
          <li><strong>Cochlear implant referral:</strong> All severe-to-profound AISNHL cases — cochlear nerve intact; excellent CI outcomes in MT-RNR1.</li>
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const terms = data.terms || [];
  const refs = data.key_references || [];
  const gc = data.genetic_counselling || {};
  const kvars = data.key_variants || [];
  const ci = data.absolute_contraindications || [];
  const rec = data.recommended_treatments || [];

  return (
    <div>
      {/* Gene card */}
      <SectionCard title="🧬 Gene & Disease Summary" borderColor={COLOR}>
        <div className="row">
          <div className="col-md-6">
            <table className="table table-sm small">
              <tbody>
                <tr><th>Gene</th><td>{data.gene} ({data.alias})</td></tr>
                <tr><th>OMIM Gene</th><td>*{data.omim_gene}</td></tr>
                <tr><th>Disease</th><td>{data.disease_name}</td></tr>
                <tr><th>Chromosome</th><td>{data.chromosome}</td></tr>
                <tr><th>Inheritance</th><td>{data.inheritance}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="col-md-6">
            {data.product && (
              <div className="p-2 rounded small" style={{ background: LIGHT }}>
                <strong>RNA product:</strong> {data.product.type}<br />
                <strong>Length:</strong> {data.product.length_nt} nt<br />
                <strong>Unit:</strong> {data.product.ribosome_unit}<br />
                <strong>Function:</strong> {data.product.function}
              </div>
            )}
          </div>
        </div>
      </SectionCard>

      {/* Key variants */}
      <SectionCard title="🔑 Key Variants — m.1555A>G and m.1494C>T" borderColor={COLOR3}>
        {kvars.map(v => (
          <div key={v.variant} className="mb-3 p-2 rounded" style={{ background: '#ffebee', borderLeft: `3px solid ${COLOR3}` }}>
            <div className="fw-bold small" style={{ color: COLOR3 }}><code>{v.variant}</code> — Population frequency: {v.frequency}</div>
            <div className="small mt-1"><strong>Mechanism:</strong> {v.mechanism}</div>
            <div className="small"><strong>With aminoglycosides:</strong> {v.penetrance_with_AG}</div>
            <div className="small"><strong>Without aminoglycosides:</strong> {v.penetrance_without_AG}</div>
          </div>
        ))}
      </SectionCard>

      {/* Contraindications */}
      <SectionCard title="🚫 Absolute Contraindications" borderColor={COLOR3}>
        <ul className="small mb-0">
          {ci.map((c, i) => <li key={i} className="text-danger">{c}</li>)}
        </ul>
      </SectionCard>

      {/* Treatment */}
      <SectionCard title="✅ Recommended Treatments & Prevention" borderColor={COLOR5}>
        <ul className="small mb-0">
          {rec.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ul>
      </SectionCard>

      {/* Genetic counselling */}
      <SectionCard title="🧬 Genetic Counselling" borderColor={COLOR2}>
        {Object.entries(gc).map(([k, v]) => (
          <div key={k} className="mb-2">
            <strong className="small">{k.replace(/_/g,' ')}:</strong>
            <p className="small mb-0 ms-2">{v}</p>
          </div>
        ))}
      </SectionCard>

      {/* Glossary */}
      <SectionCard title="📖 Glossary" borderColor={COLOR}>
        {terms.map(t => (
          <div key={t.term} className="mb-3">
            <div className="fw-bold small" style={{ color: COLOR }}>{t.term}</div>
            <div className="small text-muted">{t.definition}</div>
          </div>
        ))}
      </SectionCard>

      {/* References */}
      <SectionCard title="📚 Key References" borderColor={COLOR2}>
        <ol className="small mb-0">
          {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function MTRNR1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const ep = tab === 0 ? 'overview' : tab === 1 ? 'breakdown' : tab === 2 ? 'breakdown' : 'definitions';
    const setter = tab === 0 ? setOverview : tab === 1 ? setBreakdown : tab === 2 ? setBreakdown : setDefinitions;
    const already = tab === 0 ? overview : tab === 1 ? breakdown : tab === 2 ? breakdown : definitions;
    if (already) { setLoading(false); return; }
    fetch(`${API}/api/mtrnr1/${ep}`)
      .then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); })
      .then(d => { setter(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3" style={{ borderLeft: `5px solid ${COLOR}`, paddingLeft: 12 }}>
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
          🧬 MT-RNR1 — 12S Ribosomal RNA
        </h4>
        <div className="text-muted small">
          Aminoglycoside-Induced SNHL (AISNHL) · m.1555A>G (~1 in 500 people) · Maternally Inherited Hearing Loss ·
          UNIQUE: ISOLATED SNHL — NO OXPHOS Deficiency · HOMOPLASMIC · ABSOLUTE CI: All Aminoglycosides ·
          mtDNA H-strand rCRS 648–1601 (954 nt) · OMIM *561000
        </div>
      </div>

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

      {loading && <div className="text-center py-4"><div className="spinner-border" style={{ color: COLOR }} /></div>}
      {error && <div className="alert alert-danger">Error: {error}</div>}

      {!loading && !error && (
        <>
          {tab === 0 && <OverviewTab data={overview} />}
          {tab === 1 && <VariantsTab data={breakdown} />}
          {tab === 2 && <DDxTab data={breakdown} />}
          {tab === 3 && <DefinitionsTab data={definitions} />}
        </>
      )}
    </div>
  );
}
