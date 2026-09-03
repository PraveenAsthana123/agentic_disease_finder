'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & OXPHOS', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#4527a0';   // deep purple — mt-LSU / 16S rRNA theme
const LIGHT  = '#ede7f6';
const COLOR2 = '#6a1b9a';
const COLOR3 = '#b71c1c';   // red — absolute contraindication / emergency
const COLOR4 = '#e65100';   // deep orange — warning / caution
const COLOR5 = '#2e7d32';   // green — safe / cochlear implant / treatment benefit
const COLOR6 = '#0277bd';   // blue — humanin / neuroprotection

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

// ── Tab: Overview ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const s = data.cohort_statistics || {};
  const feats = data.cohort_summary_features || [];
  const pheno = data.phenotype_distribution || {};
  const contrast = data.contrast_with_mtrnr1 || {};
  const humanin = data.humanin_orf || {};

  return (
    <div>
      {/* Gene header */}
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `5px solid ${COLOR}` }}>
        <h5 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 MT-RNR2 — 16S Ribosomal RNA / Mitoribosome Large Subunit (mt-LSU / 39S)
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
        <p className="mb-0 small fw-semibold" style={{ color: COLOR }}>
          🔬 mt-LSU scaffold RNA — PEPTIDYL TRANSFERASE CENTRE (PTC) — synthesises ALL 13 mt-OXPHOS subunits.
          <span style={{ color: COLOR6 }}> Contains HUMANIN ORF (rCRS ~2706–2768): 21-aa neuroprotective microprotein.</span>
          <span style={{ color: COLOR4 }}> Variants cause COMBINED OXPHOS deficiency (unlike MT-RNR1 which causes isolated SNHL only).</span>
        </p>
      </div>

      {/* Humanin alert */}
      <div className="alert mb-4" style={{ background: '#e3f2fd', borderLeft: `5px solid ${COLOR6}` }}>
        <strong style={{ color: COLOR6 }}>🧠 UNIQUE: HUMANIN microprotein encoded within MT-RNR2 (rCRS ~2706–2768)</strong>
        <br /><span className="small">
          <strong>Function:</strong> {humanin.function} &nbsp;|&nbsp;
          <strong>Length:</strong> {humanin.length_aa} amino acids &nbsp;|&nbsp;
          <strong>Position:</strong> {humanin.position}
        </span>
        <br /><span className="small text-muted">{humanin.clinical_relevance}</span>
      </div>

      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Patients (cohort)" value={data.cohort_n} />
        <KPI label="OXPHOS Deficiency" value={`${s.oxphos_deficiency_pct}%`} color={COLOR3} />
        <KPI label="Cardiomyopathy" value={`${s.cardiomyopathy_pct}%`} color={COLOR3} />
        <KPI label="Hypertension (MIHH)" value={`${s.hypertension_pct}%`} color={COLOR4} />
        <KPI label="Hypercholesterolaemia" value={`${s.hypercholesterolaemia_pct}%`} color={COLOR4} />
        <KPI label="SNHL" value={`${s.snhl_pct}%`} color={COLOR2} />
        <KPI label="Optic Neuropathy" value={`${s.optic_neuropathy_pct}%`} color={COLOR2} />
        <KPI label="Myopathy (RRF)" value={`${s.myopathy_pct}%`} color={COLOR4} />
        <KPI label="Elevated Lactate" value={`${s.elevated_lactate_pct}%`} color={COLOR3} />
        <KPI label="Maternal Family Affected" value={`${s.maternal_family_affected_pct}%`} color={COLOR4} />
        <KPI label="m.2336T>C (MIHH)" value={`${s.m2336_pct}%`} />
        <KPI label="Avg Heteroplasmy (blood)" value={`${s.avg_heteroplasmy_blood}%`} color={COLOR2} />
      </div>

      {/* Key message */}
      <SectionCard title="🔑 MT-RNR2 vs MT-RNR1 — Critical Distinctions" borderColor={COLOR}>
        <p className="mb-2 small">{data.key_message}</p>
        <div className="row">
          <div className="col-md-6">
            <strong className="small" style={{ color: COLOR3 }}>MT-RNR2 (16S rRNA — mt-LSU):</strong>
            <ul className="small mt-1 mb-0">
              {Object.entries(contrast).map(([k, v]) => (
                <li key={k}><strong>{k.replace(/_/g,' ')}:</strong> {v}</li>
              ))}
            </ul>
          </div>
          <div className="col-md-6">
            <strong className="small" style={{ color: COLOR6 }}>HUMANIN (unique to MT-RNR2):</strong>
            <div className="p-2 rounded small mt-1" style={{ background: '#e3f2fd', borderLeft: `3px solid ${COLOR6}` }}>
              <strong>21-aa neuroprotective microprotein</strong><br />
              Encoded within 16S rRNA ORF at rCRS ~2706–2768<br />
              Anti-apoptotic · Alzheimer disease protection · Ischaemia-reperfusion protection<br />
              Circulating humanin declines with age — MT-RNR2 variants may reduce neuroprotection
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Clinical features */}
      <SectionCard title="📊 Cohort Clinical Feature Prevalence" borderColor={COLOR}>
        <div className="row">
          <div className="col-md-6">
            {feats.slice(0, Math.ceil(feats.length / 2)).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct}
                   color={f.feature.includes('OXPHOS') || f.feature.includes('Cardiomyopathy') ? COLOR3 :
                          f.feature.includes('Hypertension') || f.feature.includes('Myopathy') ? COLOR4 :
                          f.feature.includes('Maternal') ? COLOR4 : COLOR} />
            ))}
          </div>
          <div className="col-md-6">
            {feats.slice(Math.ceil(feats.length / 2)).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct}
                   color={f.feature.includes('OXPHOS') || f.feature.includes('Cardiomyopathy') ? COLOR3 :
                          f.feature.includes('Hypertension') || f.feature.includes('Myopathy') ? COLOR4 :
                          f.feature.includes('Maternal') ? COLOR4 : COLOR} />
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Phenotype distribution */}
      <SectionCard title="🎯 Phenotype Distribution" borderColor={COLOR2}>
        <div className="row text-center">
          <div className="col-md-3">
            <div className="fw-bold fs-4" style={{ color: COLOR4 }}>{pheno.mihh_pct}%</div>
            <div className="small text-muted">MIHH (m.2336T>C)</div>
            <div className="small" style={{ color: COLOR4 }}>Hypertension + Hypercholesterolaemia</div>
          </div>
          <div className="col-md-3">
            <div className="fw-bold fs-4" style={{ color: COLOR3 }}>{pheno.cardiomyopathy_pct}%</div>
            <div className="small text-muted">Cardiomyopathy (m.3260A>G)</div>
            <div className="small" style={{ color: COLOR3 }}>CI+CIV deficiency; annual echo</div>
          </div>
          <div className="col-md-3">
            <div className="fw-bold fs-4" style={{ color: COLOR2 }}>{pheno.optic_neuropathy_pct}%</div>
            <div className="small text-muted">LHON-like Optic Neuropathy</div>
            <div className="small" style={{ color: COLOR2 }}>m.2617G>A; males &gt; females</div>
          </div>
          <div className="col-md-3">
            <div className="fw-bold fs-4" style={{ color: COLOR }}>{pheno.snhl_pct}%</div>
            <div className="small text-muted">SNHL (m.3093G>A)</div>
            <div className="small" style={{ color: COLOR }}>Progressive; heteroplasmy-dependent</div>
          </div>
        </div>
      </SectionCard>

      {/* Clinical alerts */}
      <SectionCard title="⚠️ Clinical Alerts" borderColor={COLOR3}>
        {(data.key_clinical_alerts || []).map((a, i) => (
          <div key={i} className="mb-2 small p-2 rounded"
               style={{ background: a.startsWith('🚨') ? '#ffebee' : a.startsWith('🧠') ? '#e3f2fd' : a.startsWith('⚠️') ? '#fff8e1' : '#e8f5e9',
                        borderLeft: `3px solid ${a.startsWith('🚨') ? COLOR3 : a.startsWith('🧠') ? COLOR6 : a.startsWith('⚠️') ? '#f9a825' : COLOR5}` }}>
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
                <th>ID</th><th>Sex</th><th>Variant</th><th>Het %</th>
                <th>Phenotype</th><th>Cardiomyopathy</th><th>SNHL</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.patient_id}>
                  <td>{p.patient_id}</td>
                  <td>{p.sex}</td>
                  <td><code className="small">{p.variant}</code></td>
                  <td>
                    <span className={`badge ${p.heteroplasmy_blood_pct > 80 ? 'bg-danger' : p.heteroplasmy_blood_pct > 50 ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                      {p.heteroplasmy_blood_pct}%
                    </span>
                  </td>
                  <td className="small">{p.primary_phenotype}</td>
                  <td>
                    <span className={`badge ${p.cardiomyopathy ? 'bg-danger' : 'bg-secondary'}`}>
                      {p.cardiomyopathy ? 'Yes' : 'No'}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${p.snhl ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                      {p.snhl ? 'Yes' : 'No'}
                    </span>
                  </td>
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

// ── Tab: Variants & OXPHOS ────────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const variants = data.all_variants || [];
  const varDist = data.variant_distribution || [];
  const phenoDist = data.phenotype_distribution || [];
  const sevDist = data.severity_distribution || [];
  const hetDist = data.heteroplasmy_distribution || [];
  const oxphos = data.oxphos_profile || {};
  const nuclDdx = data.nuclear_ddx || {};

  return (
    <div>
      {/* Variant table */}
      <SectionCard title="🧬 MT-RNR2 Pathogenic Variants" borderColor={COLOR}>
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
                    <span className={`badge ${v.severity.includes('Severe') ? 'bg-danger' : v.severity.includes('Moderate') ? 'bg-warning text-dark' : 'bg-info text-dark'}`}>
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

      {/* Distribution row */}
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
                  <div className="progress-bar" style={{ width: `${v.allele_freq_pct}%`, backgroundColor: v.variant === 'm.2336T>C' ? COLOR4 : v.variant === 'm.3260A>G' ? COLOR3 : v.variant === 'm.2617G>A' ? COLOR2 : COLOR }} />
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="🧬 Heteroplasmy Distribution (Blood)" borderColor={COLOR2}>
            {hetDist.map(h => (
              <div key={h.band} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{h.band}</span>
                  <span className="text-muted">{h.count} pts ({h.pct}%)</span>
                </div>
                <div className="progress" style={{ height: 10 }}>
                  <div className="progress-bar" style={{ width: `${h.pct}%`, backgroundColor: h.band === '>90%' ? COLOR3 : h.band === '70–90%' ? COLOR4 : h.band === '40–70%' ? COLOR2 : COLOR5 }} />
                </div>
              </div>
            ))}
            <p className="small text-muted mt-2 mb-0">
              ⚠️ Blood heteroplasmy may underestimate tissue (muscle/heart/retina) heteroplasmy — muscle biopsy required for OXPHOS enzymology.
            </p>
          </SectionCard>
        </div>
      </div>

      {/* OXPHOS profile */}
      <SectionCard title="🔬 OXPHOS Respiratory Chain Profile (Patients with Deficiency)" borderColor={COLOR3}>
        <div className="row">
          {[
            { label: 'CI (NADH dehydrogenase)', value: oxphos.CI_avg_pct_normal, key: 'CI' },
            { label: 'CII (Succinate dehydrogenase)', value: oxphos.CII_avg_pct_normal, key: 'CII' },
            { label: 'CIII (Cytochrome bc1)', value: oxphos.CIII_avg_pct_normal, key: 'CIII' },
            { label: 'CIV (Cytochrome c oxidase)', value: oxphos.CIV_avg_pct_normal, key: 'CIV' },
            { label: 'CV (ATP synthase)', value: oxphos.CV_avg_pct_normal, key: 'CV' },
          ].map(c => (
            <div key={c.key} className="col-md-6 mb-3">
              <div className="d-flex justify-content-between small mb-1">
                <span className="fw-semibold">{c.label}</span>
                <span className={c.key === 'CII' ? 'text-success fw-bold' : 'text-danger fw-bold'}>
                  {c.value}% of normal {c.key === 'CII' ? '✓ NORMAL' : '↓ REDUCED'}
                </span>
              </div>
              <div className="progress" style={{ height: 14 }}>
                <div className="progress-bar" style={{ width: `${Math.min(c.value, 110)}%`, backgroundColor: c.key === 'CII' ? COLOR5 : COLOR3 }} />
              </div>
            </div>
          ))}
        </div>
        <div className="alert alert-success py-2 small mb-0">
          <strong>✅ CII (Succinate Dehydrogenase): NORMAL</strong> — CII is entirely nuclear-encoded; NOT mt-translated.
          A reduced CI+CIII+CIV+CV with NORMAL CII is the <strong>diagnostic fingerprint</strong> of impaired mt-translation (MT-RNR2 variants → mt-LSU defect → all mt-encoded OXPHOS subunits affected).
          {oxphos.CII_interpretation && <span> {oxphos.CII_interpretation}</span>}
        </div>
        <p className="small text-muted mt-2 mb-0">
          Based on {oxphos.n_patients_with_deficiency} patients with OXPHOS deficiency; average % of normal range.
        </p>
      </SectionCard>

      {/* Phenotype distribution */}
      <SectionCard title="🏥 Phenotype Distribution" borderColor={COLOR2}>
        {phenoDist.map(p => (
          <div key={p.phenotype} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span>{p.phenotype}</span>
              <span className="text-muted">{p.count} pts ({p.pct}%)</span>
            </div>
            <div className="progress" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${p.pct}%`, backgroundColor: p.phenotype.includes('MIHH') ? COLOR4 : p.phenotype.includes('Cardio') ? COLOR3 : p.phenotype.includes('Optic') ? COLOR2 : p.phenotype.includes('SNHL') ? COLOR : COLOR5 }} />
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Nuclear DDx */}
      <SectionCard title="🔬 Nuclear Gene DDx (WES-Detectable mt-LSU Assembly Defects)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Nuclear Gene</th><th>Clinical Context</th></tr></thead>
            <tbody>
              {Object.entries(nuclDdx).map(([k, v]) => (
                <tr key={k}>
                  <td><strong>{k}</strong></td>
                  <td className="small">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small text-muted mb-0">
          WES detects nuclear mt-LSU assembly gene variants; WES MISSES MT-RNR2 and all mtDNA variants.
          Always order dedicated mtDNA panel when nuclear mt-LSU genes are negative in suspected mitochondrial disease.
        </p>
      </SectionCard>
    </div>
  );
}

// ── Tab: DDx & Treatment ──────────────────────────────────────────────────────
function DDxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const contrasts = data.key_contrasts || {};
  const tx = data.treatment_uptake || {};
  const drugCI = data.absolute_drug_contraindications || {};

  return (
    <div>
      {/* Drug contraindications */}
      <div className="alert alert-danger mb-4">
        <h6 className="fw-bold mb-2">🚨 ABSOLUTE CONTRAINDICATIONS — m.3260A>G + LargeDeletion (OXPHOS deficiency)</h6>
        <div className="row">
          {(drugCI.mt_3260_and_largedel || []).map((c, i) => (
            <div key={i} className="col-md-6 mb-1 small">
              <span className="text-danger">✗</span> {c}
            </div>
          ))}
        </div>
      </div>

      <div className="alert alert-warning mb-4">
        <h6 className="fw-bold mb-2">⚠️ CAUTION — m.2336T>C MIHH (Statins + CoQ10)</h6>
        <div className="row">
          {(drugCI.mihh_m2336 || []).map((c, i) => (
            <div key={i} className="col-md-12 mb-1 small">
              <span className="text-warning">⚠</span> {c}
            </div>
          ))}
        </div>
      </div>

      <div className="alert mb-4" style={{ background: '#fff3e0', borderLeft: `5px solid ${COLOR4}` }}>
        <h6 className="fw-bold mb-2" style={{ color: COLOR4 }}>🚫 ABSOLUTE AVOID — m.2617G>A LHON-like Optic Neuropathy</h6>
        <div className="row">
          {(drugCI.lhon_m2617 || []).map((c, i) => (
            <div key={i} className="col-md-6 mb-1 small">
              <span style={{ color: COLOR4 }}>✗</span> {c}
            </div>
          ))}
        </div>
      </div>

      {/* DDx */}
      <SectionCard title="🔍 Differential Diagnosis" borderColor={COLOR}>
        {Object.entries(contrasts).map(([k, v]) => (
          <div key={k} className="mb-3 p-3 rounded" style={{ background: '#f5f5f5', borderLeft: `3px solid ${COLOR}` }}>
            <div className="fw-bold small mb-1" style={{ color: COLOR }}>{k.replace(/_/g, ' ')}</div>
            <div className="small">{v}</div>
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

      {/* Screening panel */}
      <SectionCard title="🛡️ Surveillance & Screening Protocol" borderColor={COLOR5}>
        <ul className="small mb-0">
          <li><strong>Annual echocardiography + ECG:</strong> ALL m.3260A>G and LargeDeletion patients — cardiomyopathy surveillance.</li>
          <li><strong>Annual visual evoked potentials (VEP) + OCT:</strong> m.2617G>A LHON-like — retinal nerve fibre layer thickness; central visual field.</li>
          <li><strong>Annual audiometry:</strong> m.3093G>A SNHL — serial pure-tone audiogram; monitor progression.</li>
          <li><strong>Blood pressure + fasting lipids (annually):</strong> ALL m.2336T>C MIHH patients; statin with CoQ10 monitoring.</li>
          <li><strong>Blood + urine heteroplasmy:</strong> Serial quantification in heteroplasmic variants — track disease progression.</li>
          <li><strong>Muscle biopsy:</strong> If blood heteroplasmy low but clinical phenotype present — tissue-specific heteroplasmy may be higher.</li>
          <li><strong>Cascade maternal family testing:</strong> ALL maternal relatives (mother, maternal siblings, maternal aunts) — blood mtDNA sequencing.</li>
          <li><strong>BTBGD exclusion (SLC19A3 sequencing):</strong> MANDATORY before confirming MT-RNR2 diagnosis — treatable Leigh-like mimic.</li>
          <li><strong>Idebenone trial (Raxone):</strong> m.2617G>A LHON-like optic neuropathy — 900 mg/day; evidence extrapolated from canonical LHON (MT-ND4).</li>
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const terms = data.terms || [];
  const refs = data.key_references || [];
  const gc = data.genetic_counselling || {};
  const kvars = data.key_variants || [];
  const ci = data.absolute_contraindications || [];
  const rec = data.recommended_treatments || [];
  const ddx = data.key_ddx || [];
  const humanin = data.humanin || {};

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

      {/* Humanin */}
      {humanin.definition && (
        <SectionCard title="🧠 HUMANIN — Neuroprotective Microprotein Encoded Within MT-RNR2" borderColor={COLOR6}>
          <p className="small mb-0" style={{ color: '#0d47a1' }}>{humanin.definition}</p>
        </SectionCard>
      )}

      {/* Key variants */}
      <SectionCard title="🔑 Key Pathogenic Variants" borderColor={COLOR3}>
        {kvars.map(v => (
          <div key={v.variant} className="mb-3 p-2 rounded" style={{ background: '#fce4ec', borderLeft: `3px solid ${COLOR3}` }}>
            <div className="fw-bold small" style={{ color: COLOR3 }}><code>{v.variant}</code> — Frequency: {v.frequency}</div>
            <div className="small mt-1"><strong>Mechanism:</strong> {v.mechanism}</div>
            <div className="small"><strong>Penetrance:</strong> {v.penetrance}</div>
            <div className="small"><strong>OXPHOS deficiency:</strong> {v.oxphos_deficiency}</div>
          </div>
        ))}
      </SectionCard>

      {/* DDx table */}
      <SectionCard title="🔍 Differential Diagnosis" borderColor={COLOR}>
        {ddx.map((d, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f5f5f5', borderLeft: `3px solid ${COLOR}` }}>
            <div className="fw-bold small mb-1" style={{ color: COLOR }}>{d.condition}</div>
            <div className="small">{d.distinguishing}</div>
          </div>
        ))}
      </SectionCard>

      {/* Contraindications */}
      <SectionCard title="🚫 Absolute Contraindications" borderColor={COLOR3}>
        <ul className="small mb-0">
          {ci.map((c, i) => (
            <li key={i} className={c.includes('CAUTION') ? 'text-warning' : 'text-danger'}>{c}</li>
          ))}
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
            <strong className="small">{k.replace(/_/g, ' ')}:</strong>
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

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MTRNR2Page() {
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
    fetch(`${API}/api/mtrnr2/${ep}`)
      .then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); })
      .then(d => { setter(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3" style={{ borderLeft: `5px solid ${COLOR}`, paddingLeft: 12 }}>
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
          🧬 MT-RNR2 — 16S Ribosomal RNA (mt-LSU / 39S)
        </h4>
        <div className="text-muted small">
          MIHH (m.2336T>C) · Cardiomyopathy-Myopathy (m.3260A>G) · LHON-like Optic Neuropathy (m.2617G>A) · SNHL (m.3093G>A) ·
          Combined OXPHOS Deficiency (CI+CIII+CIV+CV) · HUMANIN microprotein encoded within 16S rRNA ·
          mtDNA H-strand rCRS 1671–3229 (1559 nt) · OMIM *561010
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
