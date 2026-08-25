'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Treatments & Genetics', 'Definitions'];

// CPT2 colour scheme — deep teal/cyan (step 3; muscle; rhabdomyolysis; exercise trigger)
const ACCENT  = '#006064';   // deep teal — CPT2 / carnitine shuttle step 3 / matrix
const ACCENT2 = '#00838f';   // medium teal — C16/C18:1 elevated / acylcarnitines
const ACCENT3 = '#1b5e20';   // deep green — KEY POSITIVES / MCT therapeutic / no cardiomyopathy
const ACCENT4 = '#b71c1c';   // deep red — absolute CI / lethal neonatal / severe
const ACCENT5 = '#4a148c';   // dark purple — genetics / p.Ser113Leu
const ACCENT6 = '#e65100';   // deep orange — rhabdomyolysis / exercise trigger / CK elevated
const ACCENT7 = '#37474f';   // dark slate — NBS / epidemiology
const ACCENT8 = '#01579b';   // dark blue — myopathic form (adult; most common)

function KPI({ label, value, color }) {
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

function PctBar({ label, pct, color = ACCENT }) {
  const numPct = typeof pct === 'string' ? parseInt(pct) : pct;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${numPct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function InfoBox({ title, children, color = ACCENT }) {
  return (
    <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold" style={{ color }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: '0.72rem' }}>
      {text}
    </span>
  );
}

// ── Overview tab ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  const b = data.biomarkers || {};
  const cf = data.clinical_features || {};
  const pd = data.phenotype_distribution || {};

  return (
    <div>
      {/* KPIs */}
      <div className="row g-3 mb-4">
        <KPI label="Patients" value={data.n_patients} color={ACCENT} />
        <KPI label="Seed" value={data.seed} color={ACCENT7} />
        <KPI label="OMIM Gene" value={data.omim_gene} color={ACCENT5} />
        <KPI label="Locus" value={data.locus} color={ACCENT2} />
        <KPI label="Inheritance" value="AR" color={ACCENT7} />
        <KPI label="Prevalence" value="~1:100K" color={ACCENT8} />
      </div>

      {/* Primary banner */}
      <div className="alert mb-4" style={{ backgroundColor: '#e0f7fa', borderLeft: `5px solid ${ACCENT}` }}>
        <h6 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🔵 CPT2 — CARNITINE SHUTTLE STEP 3 (Inner IMM, Matrix Face): MOST COMMON FAO Rhabdomyolysis Disorder
        </h6>
        <p className="mb-0 small">
          CPT2 deficiency completes the carnitine shuttle trilogy: Step 1 (CPT1A) → Step 2 (CACT) →{' '}
          <strong>Step 3 (CPT2)</strong>. CPT2 converts long-chain acylcarnitines back to acyl-CoA
          inside the mitochondrial matrix. The <strong>myopathic form (&gt;90%)</strong> presents as
          adult exercise-induced rhabdomyolysis with <strong>NO cardiomyopathy</strong> — key exam trap
          vs CACT/LCHAD/VLCAD. <strong>p.Ser113Leu</strong> is temperature-sensitive (~3–5% of general
          population are heterozygous carriers). <strong>NBS may be normal</strong> between episodes
          in myopathic CPT2. Lethal neonatal form: renal cysts + brain malformations — KEY NEGATIVE vs CACT.
        </p>
      </div>

      {/* Key biomarkers */}
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <InfoBox title="🔵 Average C16 (Palmitoylcarnitine) — Elevated" color={ACCENT2}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT2 }}>{b.avg_c16_umol} μmol/L</div>
            <div className="text-muted small">NBS flag: ≥1.5 μmol/L · CPT2 myopathic: often mildly elevated · Severe: 3–9 μmol/L</div>
            <div className="small mt-1" style={{ color: ACCENT7 }}>⚠️ NBS may be NORMAL in myopathic between episodes (crisis-dependent)</div>
          </InfoBox>
        </div>
        <div className="col-md-6">
          <InfoBox title="🟠 Rhabdomyolysis Rate" color={ACCENT6}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT6 }}>{cf.rhabdomyolysis}/{data.n_patients}</div>
            <div className="text-muted small">HALLMARK (myopathic) · Exercise-induced · CK often &gt;10,000 U/L · Myoglobinuria</div>
          </InfoBox>
        </div>
        <div className="col-md-6">
          <InfoBox title="✅ NO Cardiomyopathy (Myopathic)" color={ACCENT3}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT3 }}>
              {cf.cardiomyopathy}/{data.n_patients} total
            </div>
            <div className="text-muted small">KEY EXAM FACT: NO cardiomyopathy in myopathic CPT2 (only in severe infantile/neonatal) · CACT = hallmark cardiac</div>
          </InfoBox>
        </div>
        <div className="col-md-6">
          <InfoBox title="🫙 Renal Cysts (Lethal Neonatal)" color={ACCENT4}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT4 }}>{cf.renal_cysts}/{data.n_patients}</div>
            <div className="text-muted small">HALLMARK lethal neonatal CPT2 · KEY DISCRIMINATOR vs CACT (CACT has NO renal cysts)</div>
          </InfoBox>
        </div>
      </div>

      {/* Phenotype distribution */}
      <InfoBox title="📊 Phenotype Distribution (40-Patient Cohort)" color={ACCENT7}>
        <div className="row">
          {Object.entries(pd).map(([grp, n]) => (
            <div key={grp} className="col-md-4 mb-2">
              <PctBar
                label={grp}
                pct={Math.round(n / data.n_patients * 100)}
                color={
                  grp.includes('Myopathic') ? ACCENT8 :
                  grp.includes('Lethal') ? ACCENT4 : ACCENT6
                }
              />
              <div className="text-muted small">{n} patients</div>
            </div>
          ))}
        </div>
        <div className="mt-2">
          <Badge text="Myopathic >90%: adult rhabdomyolysis" color={ACCENT8} />
          <Badge text="NO cardiomyopathy in myopathic" color={ACCENT3} />
          <Badge text="Renal cysts = lethal neonatal only" color={ACCENT4} />
        </div>
      </InfoBox>

      {/* Clinical features */}
      <InfoBox title="🏥 Clinical Features (40-Patient Cohort)" color={ACCENT7}>
        <div className="row">
          {Object.entries(cf).map(([feat, n]) => (
            <div key={feat} className="col-6 col-md-4 mb-2">
              <div className="d-flex justify-content-between small">
                <span style={{
                  color: feat === 'rhabdomyolysis' ? ACCENT6 :
                         feat === 'myoglobinuria' ? ACCENT6 :
                         feat === 'cardiomyopathy' ? ACCENT4 : 'inherit'
                }}>
                  {feat === 'rhabdomyolysis' ? '🟠 ' : feat === 'myoglobinuria' ? '🟠 ' : ''}
                  {feat.replace(/_/g, ' ')}
                </span>
                <span className="fw-bold" style={{ color: ACCENT }}>
                  {n}/{data.n_patients}
                </span>
              </div>
              <div className="progress" style={{ height: 6 }}>
                <div className="progress-bar" style={{
                  width: `${n / data.n_patients * 100}%`,
                  backgroundColor: feat === 'rhabdomyolysis' || feat === 'myoglobinuria' ? ACCENT6 :
                                   feat === 'cardiomyopathy' ? ACCENT4 : ACCENT,
                }} />
              </div>
            </div>
          ))}
        </div>
        <div className="mt-2">
          <Badge text="Rhabdomyolysis = HALLMARK (myopathic)" color={ACCENT6} />
          <Badge text="NO cardiomyopathy in myopathic (KEY EXAM)" color={ACCENT3} />
          <Badge text="Renal cysts = lethal neonatal KEY DISCRIMINATOR vs CACT" color={ACCENT4} />
          <Badge text="NBS may be false-negative in myopathic" color={ACCENT7} />
        </div>
      </InfoBox>

      {/* Key exam facts */}
      <InfoBox title="🎯 Highest-Yield Exam Facts" color={ACCENT}>
        <ol className="mb-0 small">
          {(data.key_exam_facts || []).map((f, i) => (
            <li key={i} className="mb-1"
              style={{
                color: f.includes('ABSOLUTE') ? ACCENT4 :
                       f.includes('NO cardiac') || f.includes('NOT ROUTINE') || f.includes('NORMAL') ? ACCENT3 :
                       f.includes('rhabdo') || f.includes('RHABDO') || f.includes('exercise') || f.includes('exercise') ? ACCENT6 :
                       f.includes('p.Ser113') || f.includes('STEP 3') ? ACCENT5 :
                       f.includes('MYOPATHIC') ? ACCENT8 : 'inherit'
              }}>
              {f}
            </li>
          ))}
        </ol>
      </InfoBox>

      {/* Pathway diagram */}
      <InfoBox title="🔬 Carnitine Shuttle — Where CPT2 Acts (Step 3)" color={ACCENT2}>
        <div className="font-monospace small p-2 rounded" style={{ backgroundColor: '#e0f7fa' }}>
          <div className="text-muted">Step 1 (CPT1A): Long-chain acyl-CoA + Carnitine → Acylcarnitine + CoA-SH [outer IMM, cytosolic face]</div>
          <div className="text-muted">Step 2 (CACT/SLC25A20): Acylcarnitine IN ↔ Free carnitine OUT [antiport through IMM]</div>
          <div><strong style={{ color: ACCENT4 }}>← CPT2 BLOCK (inner IMM, MATRIX FACE — Step 3)</strong></div>
          <div style={{ color: ACCENT }}>Step 3 (CPT2): Acylcarnitine + CoA-SH → Acyl-CoA + Free carnitine [matrix — FINAL shuttle step]</div>
          <div className="text-muted">Steps 4–7: Beta-oxidation (VLCAD → HADHA → HADHB → ACAT1 for long chain)</div>
          <div className="mt-1" style={{ color: ACCENT3 }}>
            ✅ MCT (C8/C10) BYPASSES CPT2 — medium-chain FA enter mitochondria via MCT1 → THERAPEUTIC (Level A severe / B myopathic)
          </div>
          <div style={{ color: ACCENT6 }}>
            🟠 p.Ser113Leu: CPT2 normal at 37°C; markedly reduced at 41°C (fever/exercise) → explains episodic triggers
          </div>
          <div style={{ color: ACCENT8 }}>
            🔵 NBS may be NORMAL in myopathic between episodes — C16 crisis-dependent; false-negative risk
          </div>
        </div>
      </InfoBox>
    </div>
  );
}

// ── Patients & Biomarkers tab ─────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  const patients = data.patients || [];
  const byPheno  = data.by_phenotype || {};
  const nbs      = data.nbs_profile_summary || {};

  return (
    <div>
      {/* By phenotype summary */}
      <InfoBox title="📊 By Phenotype — Biomarker + Clinical Summary" color={ACCENT}>
        <div className="row">
          {Object.entries(byPheno).map(([grp, s]) => (
            <div key={grp} className="col-md-4 mb-3">
              <div className="card h-100 shadow-sm" style={{
                borderTop: `3px solid ${grp.includes('Myopathic') ? ACCENT8 : grp.includes('Lethal') ? ACCENT4 : ACCENT6}`
              }}>
                <div className="card-body p-2">
                  <div className="small fw-bold mb-1" style={{
                    color: grp.includes('Myopathic') ? ACCENT8 : grp.includes('Lethal') ? ACCENT4 : ACCENT6
                  }}>{grp}</div>
                  <table className="table table-sm small mb-0">
                    <tbody>
                      <tr><td>n</td><td className="fw-bold">{s.n}</td></tr>
                      <tr><td>avg C16 (μmol/L)</td><td className="fw-bold" style={{ color: ACCENT2 }}>{s.avg_c16}</td></tr>
                      <tr><td>avg C18:1 (μmol/L)</td><td className="fw-bold" style={{ color: ACCENT2 }}>{s.avg_c18_1}</td></tr>
                      <tr><td>avg C0 (μmol/L)</td><td className="fw-bold">{s.avg_c0}</td></tr>
                      <tr><td>avg glucose (mmol/L)</td><td className="fw-bold">{s.avg_glucose}</td></tr>
                      <tr><td>avg ammonia (μmol/L)</td><td className="fw-bold">{s.avg_ammonia}</td></tr>
                      <tr><td>cardiomyopathy %</td>
                        <td className="fw-bold" style={{ color: s.cardiomyopathy_rate > 30 ? ACCENT4 : ACCENT3 }}>
                          {s.cardiomyopathy_rate}%
                        </td>
                      </tr>
                      <tr><td>rhabdomyolysis %</td>
                        <td className="fw-bold" style={{ color: ACCENT6 }}>{s.rhabdomyolysis_rate}%</td>
                      </tr>
                      <tr><td>myoglobinuria %</td>
                        <td className="fw-bold" style={{ color: ACCENT6 }}>{s.myoglobinuria_rate}%</td>
                      </tr>
                      {s.renal_cysts_rate > 0 && (
                        <tr><td>renal cysts %</td>
                          <td className="fw-bold" style={{ color: ACCENT4 }}>{s.renal_cysts_rate}%</td>
                        </tr>
                      )}
                      <tr><td>good response %</td><td className="fw-bold" style={{ color: ACCENT3 }}>{s.good_response_rate}%</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ))}
        </div>
      </InfoBox>

      {/* NBS summary */}
      <InfoBox title="🧪 NBS / Biomarker Profile Summary" color={ACCENT7}>
        <div className="row">
          {Object.entries(nbs).map(([k, v]) => (
            <div key={k} className="col-md-4 mb-2">
              <div className="d-flex justify-content-between small">
                <span>{k.replace(/_/g, ' ')}</span>
                <span className="fw-bold" style={{ color: ACCENT }}>{v}%</span>
              </div>
              <div className="progress" style={{ height: 6 }}>
                <div className="progress-bar" style={{ width: `${v}%`, backgroundColor: ACCENT }} />
              </div>
            </div>
          ))}
        </div>
        <div className="mt-2 small text-muted">
          ⚠️ NBS false-negative risk: myopathic CPT2 C16 normalises between episodes. Diagnosis often
          requires CK monitoring during/after exercise + molecular confirmation.
        </div>
      </InfoBox>

      {/* Patient table */}
      <InfoBox title="👤 Individual Patients (40-Patient Cohort, seed=283)" color={ACCENT7}>
        <div style={{ overflowX: 'auto' }}>
          <table className="table table-sm small mb-0">
            <thead>
              <tr>
                <th>ID</th><th>Phenotype</th><th>Variant</th><th>Onset (mo)</th>
                <th>C16</th><th>C18:1</th><th>C0</th>
                <th>Glucose</th><th>NH₃</th>
                <th>Cardiac</th><th>Rhabdo</th><th>RenCyst</th>
                <th>Trigger</th><th>Response</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.id}>
                  <td className="font-monospace text-muted">{p.id}</td>
                  <td style={{ color: p.phenotype === 'Myopathic' ? ACCENT8 : p.phenotype === 'Lethal Neonatal' ? ACCENT4 : ACCENT6 }}>
                    {p.phenotype}
                  </td>
                  <td className="font-monospace" style={{ fontSize: '0.65rem' }}>{p.variant}</td>
                  <td>{p.onset_age_months}</td>
                  <td style={{ color: p.c16_umol >= 1.5 ? ACCENT2 : 'inherit' }}>{p.c16_umol}</td>
                  <td>{p.c18_1_umol}</td>
                  <td>{p.c0_umol}</td>
                  <td style={{ color: p.glucose_mmol < 2.5 ? ACCENT4 : 'inherit' }}>{p.glucose_mmol}</td>
                  <td style={{ color: p.ammonia_umol > 80 ? ACCENT4 : 'inherit' }}>{p.ammonia_umol}</td>
                  <td style={{ color: p.cardiomyopathy ? ACCENT4 : ACCENT3 }}>
                    {p.cardiomyopathy ? '❤️' : '✅'}
                  </td>
                  <td style={{ color: p.rhabdomyolysis ? ACCENT6 : 'inherit' }}>
                    {p.rhabdomyolysis ? '🟠' : '—'}
                  </td>
                  <td style={{ color: p.renal_cysts ? ACCENT4 : 'inherit' }}>
                    {p.renal_cysts ? '🫙' : '—'}
                  </td>
                  <td style={{ fontSize: '0.65rem' }}>{p.trigger}</td>
                  <td style={{ color: p.response.includes('Good') ? ACCENT3 : p.response.includes('Critical') ? ACCENT4 : ACCENT6 }}>
                    {p.response}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </InfoBox>
    </div>
  );
}

// ── Treatments & Genetics tab ─────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  const vc = data.variant_counts || {};
  const ts = data.treatment_summary || {};

  return (
    <div>
      {/* Treatment summary */}
      <InfoBox title="💊 Treatment Summary (40-Patient Cohort)" color={ACCENT3}>
        <div className="row">
          {Object.entries(ts).map(([k, v]) => (
            <div key={k} className="col-md-4 mb-2">
              <div className="d-flex justify-content-between small">
                <span>{k.replace(/_/g, ' ')}</span>
                <span className="fw-bold" style={{ color: ACCENT }}>{v} pts</span>
              </div>
              <div className="progress" style={{ height: 6 }}>
                <div className="progress-bar" style={{ width: `${v / 40 * 100}%`, backgroundColor: ACCENT3 }} />
              </div>
            </div>
          ))}
        </div>
      </InfoBox>

      {/* Treatment hierarchy */}
      <InfoBox title="🏥 Treatment by Phenotype (Myopathic vs Severe)" color={ACCENT}>
        <div className="row">
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: ACCENT8 }}>MYOPATHIC FORM (adult; >90%)</div>
            <ul className="small mb-0">
              <li><strong>Avoid prolonged exercise</strong> — primary prevention; pace activities; warm-up</li>
              <li><strong>High-carb pre-exercise diet</strong> — Level B; maintains glycogen; reduces FAO demand</li>
              <li><strong>MCT oil</strong> — Level B; bypasses CPT2; medium chain via MCT1</li>
              <li><strong>IV glucose + IV fluids</strong> — emergency rhabdomyolysis; target UO ≥3 ml/kg/hr</li>
              <li><strong>L-carnitine NOT ROUTINE</strong> — Level C only (may worsen by ↑ acylcarnitines)</li>
              <li><strong>Avoid statins</strong> — HIGH RISK (rhabdomyolysis risk)</li>
              <li><strong>Avoid general anaesthesia</strong> without glucose pre-loading</li>
              <li><strong>Bezafibrate</strong> — Level C investigational (PPAR-α agonist; may ↑ residual CPT2)</li>
            </ul>
          </div>
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: ACCENT4 }}>SEVERE INFANTILE / LETHAL NEONATAL</div>
            <ul className="small mb-0">
              <li><strong>MCT oil</strong> — Level A (bypasses CPT2; medium chain via MCT1)</li>
              <li><strong>Long-chain fat restriction</strong> — Level A</li>
              <li><strong>L-carnitine</strong> — Level A if C0 depleted (severe forms)</li>
              <li><strong>Fasting ABSOLUTE CI</strong></li>
              <li><strong>KD ABSOLUTE CI</strong> (long-chain fat floods blocked pathway)</li>
              <li><strong>VPA ABSOLUTE CI</strong> (inhibits FAO; carnitine depletion; especially with cardiomyopathy)</li>
              <li><strong>IV glucose 10%</strong> — emergency anti-catabolic</li>
              <li><strong>Cardiac management</strong> — severe infantile/neonatal with cardiomyopathy</li>
            </ul>
          </div>
        </div>
        <div className="mt-3">
          <Badge text="Myopathic: L-carnitine NOT ROUTINE (Level C only)" color={ACCENT8} />
          <Badge text="Severe: L-carnitine Level A if C0 depleted" color={ACCENT4} />
          <Badge text="MCT: Level A (severe) / Level B (myopathic)" color={ACCENT3} />
          <Badge text="Statins: HIGH RISK ALL phenotypes" color={ACCENT4} />
        </div>
      </InfoBox>

      {/* CI box */}
      <InfoBox title="🚫 Contraindications by Phenotype" color={ACCENT4}>
        <div className="row">
          <div className="col-md-6">
            <div className="small fw-bold mb-1" style={{ color: ACCENT8 }}>Myopathic (adult)</div>
            <ul className="small mb-0">
              <li style={{ color: ACCENT4 }}>STATINS — HIGH RISK (rhabdomyolysis)</li>
              <li style={{ color: ACCENT4 }}>PROLONGED EXERCISE without carb loading</li>
              <li style={{ color: ACCENT4 }}>VPA — HIGH RISK</li>
              <li style={{ color: ACCENT4 }}>General anaesthesia without metabolic prep</li>
              <li>L-carnitine routine — not recommended (Level C only)</li>
              <li>Fasting — avoid; not absolute CI in well-compensated myopathic</li>
            </ul>
          </div>
          <div className="col-md-6">
            <div className="small fw-bold mb-1" style={{ color: ACCENT4 }}>Severe Infantile / Lethal Neonatal</div>
            <ul className="small mb-0">
              <li style={{ color: ACCENT4 }}>FASTING — ABSOLUTE CONTRAINDICATION</li>
              <li style={{ color: ACCENT4 }}>KD — ABSOLUTE CONTRAINDICATION</li>
              <li style={{ color: ACCENT4 }}>VPA — ABSOLUTE CONTRAINDICATION</li>
              <li style={{ color: ACCENT4 }}>STATINS — HIGH RISK (all phenotypes)</li>
              <li style={{ color: ACCENT4 }}>HIGH DIETARY LONG-CHAIN FAT</li>
            </ul>
          </div>
        </div>
      </InfoBox>

      {/* Variant counts */}
      <InfoBox title="🧬 Variant Distribution (Cohort)" color={ACCENT5}>
        <div className="row">
          {Object.entries(vc).map(([v, n]) => (
            <div key={v} className="col-md-6 mb-2">
              <div className="d-flex justify-content-between small">
                <span className="font-monospace text-muted" style={{ fontSize: '0.72rem' }}>{v}</span>
                <span className="fw-bold" style={{ color: ACCENT5 }}>{n} pts</span>
              </div>
              <div className="progress" style={{ height: 6 }}>
                <div className="progress-bar" style={{ width: `${n / 40 * 100}%`, backgroundColor: ACCENT5 }} />
              </div>
            </div>
          ))}
        </div>
        <div className="mt-2 small text-muted">
          p.Ser113Leu is the most common myopathic allele (~30–50% allele frequency; temperature-sensitive
          hypomorph; ~3–5% of general population are heterozygous carriers). Biallelic required for disease.
        </div>
      </InfoBox>
    </div>
  );
}

// ── Definitions tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;

  return (
    <div>
      <InfoBox title="🔬 Disease Overview" color={ACCENT}>
        <dl className="small mb-0">
          <dt>Disease</dt><dd>{data.disease_name}</dd>
          <dt>Gene</dt><dd>{data.gene}</dd>
          <dt>OMIM Gene</dt><dd>{data.omim_gene}</dd>
          <dt>OMIM Disease</dt><dd>{data.omim_disease}</dd>
          <dt>Inheritance</dt><dd>{data.inheritance}</dd>
          <dt>Protein</dt><dd>{data.protein}</dd>
          <dt>Enzymatic function</dt><dd>{data.enzymatic_function}</dd>
          <dt>Metabolic block</dt><dd>{data.metabolic_block}</dd>
        </dl>
      </InfoBox>

      <InfoBox title="📊 Three Clinical Phenotypes (KEY EXAM)" color={ACCENT8}>
        {Object.entries(data.three_phenotypes || {}).map(([k, v]) => (
          <div key={k} className="mb-2">
            <div className="small fw-bold" style={{
              color: k.includes('Myopathic') ? ACCENT8 : k.includes('Lethal') ? ACCENT4 : ACCENT6
            }}>{k.replace(/_/g, ' ')}</div>
            <div className="small text-muted">{v}</div>
          </div>
        ))}
      </InfoBox>

      <InfoBox title="🧪 NBS Marker Profile" color={ACCENT2}>
        <p className="small mb-2">{data.nbs_marker}</p>
        <table className="table table-sm small mb-0">
          <thead><tr><th>Biomarker</th><th>Status in CPT2</th></tr></thead>
          <tbody>
            {Object.entries(data.key_biomarkers || {}).map(([k, v]) => (
              <tr key={k}>
                <td className="fw-bold font-monospace text-muted" style={{ width: 220 }}>{k.replace(/_/g, ' ')}</td>
                <td style={{
                  color: v.includes('ELEVATED') ? ACCENT2 :
                         v.includes('NORMAL') || v.includes('NOT ROUTINE') ? ACCENT3 :
                         v.includes('HIGH RISK') || v.includes('false-negative') ? ACCENT4 :
                         'inherit'
                }}>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </InfoBox>

      <InfoBox title="🏥 Clinical Features" color={ACCENT7}>
        <table className="table table-sm small mb-0">
          <tbody>
            {Object.entries(data.clinical_features || {}).map(([k, v]) => (
              <tr key={k}>
                <td className="fw-bold" style={{
                  width: 240,
                  color: k.includes('Rhabdo') || k.includes('Myoglo') ? ACCENT6 :
                         k.includes('NO_cardiac') ? ACCENT3 :
                         k.includes('Renal') || k.includes('Brain') ? ACCENT4 : 'inherit'
                }}>{k.replace(/_/g, ' ')}</td>
                <td className="text-muted small">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </InfoBox>

      <InfoBox title="💊 Treatment" color={ACCENT3}>
        <table className="table table-sm small mb-0">
          <tbody>
            {Object.entries(data.treatment || {}).map(([k, v]) => (
              <tr key={k}>
                <td className="fw-bold" style={{
                  width: 260,
                  color: k.includes('ABSOLUTE') || k.includes('Statin') ? ACCENT4 :
                         k.includes('MCT') || k.includes('Carb') ? ACCENT3 : 'inherit'
                }}>{k.replace(/_/g, ' ')}</td>
                <td className="text-muted small">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </InfoBox>

      <InfoBox title="🚫 Contraindications" color={ACCENT4}>
        <ul className="small mb-0">
          {(data.contraindications || []).map((c, i) => (
            <li key={i} style={{ color: c.includes('ABSOLUTE') || c.includes('HIGH RISK') ? ACCENT4 : 'inherit' }}>{c}</li>
          ))}
        </ul>
      </InfoBox>

      <InfoBox title="🧬 Genetics — Key Variants" color={ACCENT5}>
        <table className="table table-sm small mb-0">
          <tbody>
            {Object.entries((data.genetics || {}).key_variants || {}).map(([v, desc]) => (
              <tr key={v}>
                <td className="fw-bold font-monospace text-muted" style={{ width: 260 }}>{v}</td>
                <td>{desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {(data.genetics || {}).population_note && (
          <p className="small mt-2 mb-0 text-muted">{data.genetics.population_note}</p>
        )}
      </InfoBox>

      <InfoBox title="⚖️ Key Distinguishing Facts" color={ACCENT2}>
        <ul className="small mb-0">
          {(data.key_distinguishing_facts || []).map((f, i) => <li key={i}>{f}</li>)}
        </ul>
      </InfoBox>

      <InfoBox title="📐 Carnitine Shuttle Context" color={ACCENT7}>
        <pre className="small mb-0" style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
          {data.carnitine_shuttle_context}
        </pre>
      </InfoBox>

      <InfoBox title="⚖️ Comparison Table" color={ACCENT2}>
        {Object.entries(data.comparison_table || {}).map(([k, v]) => (
          <div key={k} className="mb-2">
            <div className="small fw-bold" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}</div>
            <div className="small text-muted">{v}</div>
          </div>
        ))}
      </InfoBox>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function CPT2Page() {
  const [tab, setTab]          = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);

  useEffect(() => {
    fetch(`${API}/api/cpt2/overview`).then(r => r.json()).then(setOverview).catch(() => setError('Backend offline'));
    fetch(`${API}/api/cpt2/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/cpt2/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3 d-flex align-items-center gap-2 flex-wrap">
        <Link href="/" className="btn btn-sm btn-outline-secondary">← Home</Link>
        <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
          🔵 CPT2 Epilepsy Dashboard
        </h4>
        <span className="badge" style={{ backgroundColor: ACCENT }}>CPT2 Deficiency</span>
        <span className="badge" style={{ backgroundColor: ACCENT5 }}>1p32.3</span>
        <span className="badge" style={{ backgroundColor: ACCENT7 }}>AR</span>
        <span className="badge" style={{ backgroundColor: ACCENT8 }}>Myopathic &gt;90%</span>
        <span className="badge" style={{ backgroundColor: ACCENT6 }}>Rhabdomyolysis HALLMARK</span>
        <span className="badge" style={{ backgroundColor: ACCENT3 }}>NO Cardiomyopathy (Myopathic)</span>
        <span className="badge" style={{ backgroundColor: ACCENT4 }}>Renal Cysts = Lethal Neonatal</span>
        <span className="badge" style={{ backgroundColor: ACCENT5 }}>p.Ser113Leu Temperature-Sensitive</span>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active' : ''}`}
              style={tab === t ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'               && <OverviewTab    data={overview}   />}
      {tab === 'Patients & Biomarkers'  && <PatientsTab    data={breakdown}  />}
      {tab === 'Treatments & Genetics'  && <TreatmentsTab  data={breakdown}  />}
      {tab === 'Definitions'            && <DefinitionsTab data={defs}       />}
    </div>
  );
}
