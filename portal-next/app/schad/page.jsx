'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Treatments & Genetics', 'Definitions'];

// SCHAD colour scheme — warm amber/orange (hyperinsulinism / glucose crisis / GDH mechanism)
const ACCENT  = '#e65100';   // deep orange — hyperinsulinism / glucose crisis
const ACCENT2 = '#f57f17';   // amber — C4-OH NBS marker
const ACCENT3 = '#1b5e20';   // deep green — KEY POSITIVES / diazoxide response
const ACCENT4 = '#b71c1c';   // deep red — absolute CI / fasting / VPA
const ACCENT5 = '#4a148c';   // dark purple — genetics
const ACCENT6 = '#004d40';   // dark teal — GDH inhibition mechanism
const ACCENT7 = '#37474f';   // dark slate — NBS/epidemiology
const ACCENT8 = '#880e4f';   // dark pink — protein-sensitive component

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

// ── Overview tab ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  const b  = data.biomarkers || {};
  const cf = data.clinical_features || {};
  const pd = data.phenotype_distribution || {};

  return (
    <div>
      {/* KPIs */}
      <div className="row g-3 mb-4">
        <KPI label="Patients"   value={data.n_patients}  color={ACCENT}  />
        <KPI label="Seed"       value={data.seed}         color={ACCENT7} />
        <KPI label="OMIM Gene"  value={data.omim_gene}    color={ACCENT5} />
        <KPI label="Locus"      value={data.locus}        color={ACCENT2} />
        <KPI label="Inheritance" value="AR"               color={ACCENT7} />
        <KPI label="Prevalence" value="~1:50K HI"         color={ACCENT8} />
      </div>

      {/* Primary banner */}
      <div className="alert mb-4" style={{ backgroundColor: '#fff3e0', borderLeft: `5px solid ${ACCENT}` }}>
        <h6 className="fw-bold mb-1" style={{ color: ACCENT }}>
          SCHAD (HADH) — UNIQUE MECHANISM: GDH Disinhibition → Hyperinsulinism (NOT FAO Energy Crisis)
        </h6>
        <p className="mb-0 small">
          HADH normally <strong>physically inhibits GDH (GLUD1)</strong> in pancreatic beta-cell
          mitochondria. HADH LOF → GDH uninhibited → excess glutamate oxidation → excess ATP →
          KATP channels CLOSE → <strong>excess insulin secreted</strong>. This is{' '}
          <strong>congenital hyperinsulinism (HI)</strong>, not a typical FAO energy-crisis disorder.
          {' '}<strong>Ammonia NORMAL</strong> (KEY NEGATIVE vs GLUD1/HHS where 100-500 μmol/L).
          {' '}<strong>Diazoxide 70-80% responsive</strong>. Protein-sensitive HI present
          (leucine activates uninhibited GDH). <strong>FASTING ABSOLUTE CI</strong>.
        </p>
      </div>

      {/* Key biomarkers */}
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <InfoBox title="Avg Glucose Nadir — Critically Low" color={ACCENT}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT }}>{b.avg_glucose_nadir_mmol} mmol/L</div>
            <div className="text-muted small">Neonatal: &lt;1.5 mmol/L profound; Infantile: &lt;2.6 mmol/L; ALL due to excess insulin</div>
          </InfoBox>
        </div>
        <div className="col-md-6">
          <InfoBox title="Avg Insulin — Elevated (Inappropriate)" color={ACCENT8}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT8 }}>{b.avg_insulin_mU_L} mU/L</div>
            <div className="text-muted small">&gt;2 mU/L during hypoglycaemia = inappropriate HI · Insulin:glucose ratio elevated</div>
          </InfoBox>
        </div>
        <div className="col-md-6">
          <InfoBox title="Ammonia — NORMAL (KEY NEGATIVE vs GLUD1)" color={ACCENT3}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT3 }}>{b.avg_ammonia_umol} μmol/L</div>
            <div className="text-muted small">CRITICAL EXAM DISCRIMINATOR: HADH ammonia NORMAL vs GLUD1/HHS where 100-500 μmol/L</div>
          </InfoBox>
        </div>
        <div className="col-md-6">
          <InfoBox title="C4-OH (NBS Marker) — Mildly Elevated" color={ACCENT2}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT2 }}>{b.avg_c4oh_umol} μmol/L</div>
            <div className="text-muted small">Mildly elevated on NBS · Nonspecific (also IBD) · Clinical context + insulin essential</div>
          </InfoBox>
        </div>
      </div>

      {/* Phenotype distribution */}
      <InfoBox title="Phenotype Distribution (40-Patient Cohort)" color={ACCENT7}>
        <div className="row">
          {Object.entries(pd).map(([grp, n]) => (
            <div key={grp} className="col-md-4 mb-2">
              <PctBar
                label={grp}
                pct={Math.round(n / data.n_patients * 100)}
                color={
                  grp.includes('Neonatal') ? ACCENT :
                  grp.includes('Late')     ? ACCENT8 : ACCENT3
                }
              />
              <div className="text-muted small">{n} patients</div>
            </div>
          ))}
        </div>
        <div className="mt-2">
          <Badge text="Classic Neonatal HI 50%" color={ACCENT} />
          <Badge text="Late Infantile HI 35%" color={ACCENT8} />
          <Badge text="Mild Attenuated 15%" color={ACCENT3} />
        </div>
      </InfoBox>

      {/* Clinical features */}
      <InfoBox title="Clinical Features (40-Patient Cohort)" color={ACCENT7}>
        <div className="row">
          {Object.entries(cf).map(([feat, n]) => (
            <div key={feat} className="col-6 col-md-4 mb-2">
              <div className="d-flex justify-content-between small">
                <span style={{
                  color: feat === 'diazoxide_responsive' ? ACCENT3 :
                         feat === 'protein_sensitive' ? ACCENT8 :
                         feat.includes('seizure') || feat.includes('gtcs') || feat.includes('focal') ||
                         feat.includes('status') || feat.includes('myoclonic') || feat.includes('infantile') ? ACCENT4 : 'inherit'
                }}>
                  {feat.replace(/_/g, ' ')}
                </span>
                <span className="fw-bold" style={{ color: ACCENT }}>
                  {n}/{data.n_patients}
                </span>
              </div>
              <div className="progress" style={{ height: 6 }}>
                <div className="progress-bar" style={{
                  width: `${n / data.n_patients * 100}%`,
                  backgroundColor: feat === 'diazoxide_responsive' ? ACCENT3 :
                                   feat === 'protein_sensitive' ? ACCENT8 : ACCENT,
                }} />
              </div>
            </div>
          ))}
        </div>
        <div className="mt-2">
          <Badge text="Diazoxide Level A FIRST LINE ~80% response" color={ACCENT3} />
          <Badge text="Protein-sensitive HI (leucine → GDH activation)" color={ACCENT8} />
          <Badge text="GTCS secondary to hypoglycaemia" color={ACCENT4} />
          <Badge text="Ammonia NORMAL — KEY vs GLUD1/HHS" color={ACCENT6} />
        </div>
      </InfoBox>

      {/* Key exam facts */}
      <InfoBox title="Highest-Yield Exam Facts" color={ACCENT}>
        <ol className="mb-0 small">
          {(data.key_exam_facts || []).map((f, i) => (
            <li key={i} className="mb-1"
              style={{
                color: f.includes('ABSOLUTE') ? ACCENT4 :
                       f.includes('NORMAL') && f.includes('NEGATIVE') ? ACCENT3 :
                       f.includes('GDH') || f.includes('KATP') || f.includes('MECHANISM') ? ACCENT6 :
                       f.includes('p.His') || f.includes('GENETICS') ? ACCENT5 :
                       f.includes('PROTEIN') || f.includes('leucine') ? ACCENT8 :
                       f.includes('HIGH RISK') || f.includes('VPA') ? ACCENT4 :
                       f.includes('HYPERINSULINISM') || f.includes('UNIQUE') ? ACCENT :
                       'inherit'
              }}>
              {f}
            </li>
          ))}
        </ol>
      </InfoBox>

      {/* Mechanism diagram */}
      <InfoBox title="SCHAD-GDH Mechanism — Why HADH LOF Causes Hyperinsulinism" color={ACCENT6}>
        <div className="font-monospace small p-2 rounded" style={{ backgroundColor: '#e8f5e9' }}>
          <div style={{ color: ACCENT6 }}>Normal state: HADH binds GDH inside beta-cell mitochondria → GDH INHIBITED</div>
          <div className="text-muted">  Glutamate oxidation → moderate ATP → KATP partially open → basal insulin</div>
          <div className="mt-2" style={{ color: ACCENT }}><strong>SCHAD Deficiency: HADH ABSENT → GDH UNINHIBITED</strong></div>
          <div style={{ color: ACCENT }}>  Excess glutamate oxidised → EXCESS ATP → KATP CLOSE → Ca2+ influx → EXCESS INSULIN</div>
          <div className="mt-2" style={{ color: ACCENT8 }}>Protein meal → leucine → allosteric GDH activation → amplified HI (protein-sensitive)</div>
          <div className="mt-2" style={{ color: ACCENT3 }}>Diazoxide → OPENS KATP → K+ efflux → repolarisation → less Ca2+ → less insulin (THERAPEUTIC)</div>
          <div className="mt-2" style={{ color: ACCENT4 }}>Ammonia: NORMAL (GDH generates NH4+ but buffered; UNLIKE GLUD1/HHS gain-of-function where NH4+ overproduced → hyperammonaemia)</div>
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
  const nbs      = data.nbs_summary || {};

  return (
    <div>
      {/* By phenotype summary */}
      <InfoBox title="By Phenotype — Biomarker + Clinical Summary" color={ACCENT}>
        <div className="row">
          {Object.entries(byPheno).map(([grp, s]) => (
            <div key={grp} className="col-md-4 mb-3">
              <div className="card h-100 shadow-sm" style={{
                borderTop: `3px solid ${grp.includes('Neonatal') ? ACCENT : grp.includes('Late') ? ACCENT8 : ACCENT3}`
              }}>
                <div className="card-body p-2">
                  <div className="small fw-bold mb-1" style={{
                    color: grp.includes('Neonatal') ? ACCENT : grp.includes('Late') ? ACCENT8 : ACCENT3
                  }}>{grp}</div>
                  <table className="table table-sm small mb-0">
                    <tbody>
                      <tr><td>n</td><td className="fw-bold">{s.n}</td></tr>
                      <tr><td>avg glucose nadir (mmol/L)</td><td className="fw-bold" style={{ color: ACCENT }}>{s.avg_glucose_nadir}</td></tr>
                      <tr><td>avg insulin (mU/L)</td><td className="fw-bold" style={{ color: ACCENT8 }}>{s.avg_insulin}</td></tr>
                      <tr><td>avg C4-OH (μmol/L)</td><td className="fw-bold" style={{ color: ACCENT2 }}>{s.avg_c4oh}</td></tr>
                      <tr><td>avg ammonia (μmol/L)</td><td className="fw-bold" style={{ color: ACCENT3 }}>{s.avg_ammonia}</td></tr>
                      <tr><td>avg BHBA (mmol/L)</td><td className="fw-bold">{s.avg_bhba}</td></tr>
                      <tr><td>diazoxide responsive %</td>
                        <td className="fw-bold" style={{ color: ACCENT3 }}>{s.diazoxide_responsive_pct}%</td>
                      </tr>
                      <tr><td>protein sensitive %</td>
                        <td className="fw-bold" style={{ color: ACCENT8 }}>{s.protein_sensitive_pct}%</td>
                      </tr>
                      <tr><td>severe hypo &lt;1.5 %</td>
                        <td className="fw-bold" style={{ color: ACCENT4 }}>{s['severe_hypo_lt1.5_pct']}%</td>
                      </tr>
                      <tr><td>GTCS %</td>
                        <td className="fw-bold" style={{ color: ACCENT4 }}>{s.gtcs_pct}%</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ))}
        </div>
      </InfoBox>

      {/* NBS summary */}
      <InfoBox title="NBS / Biomarker Profile Summary" color={ACCENT7}>
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
          C4-OH nonspecific — always confirm with plasma insulin, glucose, ammonia during hypoglycaemia
          + HADH gene sequencing. Ammonia NORMAL = distinguishes SCHAD from GLUD1/HHS.
        </div>
      </InfoBox>

      {/* Patient table */}
      <InfoBox title={`Individual Patients (40-Patient Cohort, seed=${285})`} color={ACCENT7}>
        <div style={{ overflowX: 'auto' }}>
          <table className="table table-sm small mb-0">
            <thead>
              <tr>
                <th>ID</th><th>Phenotype</th><th>Onset (mo)</th>
                <th>Glucose (mmol)</th><th>Insulin (mU/L)</th><th>C4-OH (μmol)</th>
                <th>NH₃</th><th>BHBA</th>
                <th>Prot.Sens</th><th>Diaz.Resp</th>
                <th>Seizures</th><th>Allele 1</th><th>Allele 2</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.id}>
                  <td className="font-monospace text-muted" style={{ fontSize: '0.65rem' }}>{p.id}</td>
                  <td style={{
                    color: p.phenotype === 'Classic Neonatal HI' ? ACCENT :
                           p.phenotype === 'Late Infantile HI' ? ACCENT8 : ACCENT3,
                    fontSize: '0.7rem'
                  }}>{p.phenotype}</td>
                  <td>{p.onset_age_months}</td>
                  <td style={{ color: p.glucose_nadir_mmol < 1.5 ? ACCENT4 : p.glucose_nadir_mmol < 2.6 ? ACCENT : 'inherit' }}>
                    {p.glucose_nadir_mmol}
                  </td>
                  <td style={{ color: p.insulin_mU_L > 10 ? ACCENT8 : p.insulin_mU_L > 2 ? ACCENT : 'inherit' }}>
                    {p.insulin_mU_L}
                  </td>
                  <td style={{ color: p.c4oh_umol >= 0.35 ? ACCENT2 : 'inherit' }}>{p.c4oh_umol}</td>
                  <td style={{ color: ACCENT3 }}>{p.ammonia_umol}</td>
                  <td>{p.bhba_mmol}</td>
                  <td style={{ color: p.protein_sensitive ? ACCENT8 : 'inherit' }}>
                    {p.protein_sensitive ? 'YES' : 'no'}
                  </td>
                  <td style={{ color: p.diazoxide_responsive ? ACCENT3 : ACCENT4 }}>
                    {p.diazoxide_responsive ? 'YES' : 'NO'}
                  </td>
                  <td style={{ fontSize: '0.65rem', color: p.seizure_types && p.seizure_types.length > 0 ? ACCENT4 : 'inherit' }}>
                    {(p.seizure_types || []).join(', ') || '—'}
                  </td>
                  <td className="font-monospace" style={{ fontSize: '0.65rem', color: ACCENT5 }}>{p.variant_allele1}</td>
                  <td className="font-monospace" style={{ fontSize: '0.65rem', color: ACCENT5 }}>{p.variant_allele2}</td>
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
      <InfoBox title="Treatment Summary (40-Patient Cohort)" color={ACCENT3}>
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

      {/* Treatment evidence */}
      <InfoBox title="Treatment Evidence Levels — SCHAD/HHF4" color={ACCENT}>
        <div className="row">
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: ACCENT3 }}>Level A — Established</div>
            <ul className="small mb-0">
              <li><strong>Diazoxide (5-15 mg/kg/day)</strong> — Level A; KATP channel opener; 70-80% response; FIRST LINE</li>
              <li><strong>Frequent feeds / cornstarch</strong> — Level A; nocturnal hypoglycaemia prevention</li>
              <li><strong>Continuous glucose monitoring (CGM)</strong> — Level A; all HI disorders</li>
            </ul>
            <div className="fw-bold small mb-2 mt-3" style={{ color: ACCENT8 }}>Level B — Evidence Supported</div>
            <ul className="small mb-0">
              <li><strong>Protein moderation</strong> — Level B; NOT strict restriction (reduces leucine → GDH activation)</li>
              <li><strong>Octreotide</strong> — Level B; if diazoxide fails; somatostatin analogue; suppresses insulin</li>
              <li><strong>Partial pancreatectomy</strong> — Level B; diffuse, refractory only; risk of diabetes</li>
              <li><strong>LEV for seizures</strong> — Level B; secondary seizure management (avoid VPA)</li>
            </ul>
          </div>
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: ACCENT5 }}>Level C — Investigational</div>
            <ul className="small mb-0">
              <li><strong>Sirolimus (mTOR inhibitor)</strong> — Level C; refractory HI; inhibits beta-cell mTOR pathway</li>
            </ul>
            <div className="fw-bold small mb-2 mt-3" style={{ color: ACCENT4 }}>ABSOLUTE CONTRAINDICATIONS</div>
            <ul className="small mb-0">
              <li style={{ color: ACCENT4 }}><strong>FASTING — ABSOLUTE CI</strong> (all HI disorders; worsens hypoglycaemia)</li>
              <li style={{ color: ACCENT4 }}><strong>VPA — HIGH RISK</strong> (insulin secretagogue; opposite of needed; use LEV instead)</li>
              <li style={{ color: ACCENT4 }}><strong>Ketogenic diet — avoid</strong> (fasting-equivalent; catastrophic in HI)</li>
              <li style={{ color: ACCENT4 }}><strong>High-protein diet — avoid</strong> (not absolute CI; protein moderation = Level B)</li>
            </ul>
          </div>
        </div>
        <div className="mt-3">
          <Badge text="Diazoxide Level A — 70-80% response" color={ACCENT3} />
          <Badge text="FASTING ABSOLUTE CI" color={ACCENT4} />
          <Badge text="VPA HIGH RISK — insulin secretagogue" color={ACCENT4} />
          <Badge text="Protein MODERATION (NOT restriction)" color={ACCENT8} />
          <Badge text="Sirolimus Level C refractory" color={ACCENT5} />
        </div>
      </InfoBox>

      {/* CI box */}
      <InfoBox title="Contraindications — SCHAD/HHF4" color={ACCENT4}>
        <ul className="small mb-0">
          <li style={{ color: ACCENT4 }}><strong>FASTING — ABSOLUTE CONTRAINDICATION</strong>: all HI disorders; fasting worsens hypoglycaemia crisis; no fasting protocol</li>
          <li style={{ color: ACCENT4 }}><strong>VPA (valproate) — HIGH RISK</strong>: acts as insulin secretagogue (stimulates insulin secretion); completely opposite to therapeutic goal; use LEV for seizures</li>
          <li style={{ color: ACCENT4 }}><strong>Ketogenic diet — AVOID</strong>: equivalent to prolonged fasting in HI; catastrophic glucose lowering</li>
          <li><strong>High-protein diet — avoid</strong>: protein moderation (Level B) to reduce leucine-mediated GDH activation; not strict restriction</li>
          <li><strong>N2O (nitrous oxide) — caution</strong>: metabolic concern; prefer alternative anaesthesia agents</li>
        </ul>
      </InfoBox>

      {/* Variant counts */}
      <InfoBox title="Variant Distribution (Cohort Allele Counts)" color={ACCENT5}>
        <div className="row">
          {Object.entries(vc).map(([v, n]) => (
            <div key={v} className="col-md-6 mb-2">
              <div className="d-flex justify-content-between small">
                <span className="font-monospace text-muted" style={{ fontSize: '0.72rem' }}>{v}</span>
                <span className="fw-bold" style={{ color: ACCENT5 }}>{n} alleles</span>
              </div>
              <div className="progress" style={{ height: 6 }}>
                <div className="progress-bar" style={{ width: `${n / 80 * 100}%`, backgroundColor: ACCENT5 }} />
              </div>
            </div>
          ))}
        </div>
        <div className="mt-3">
          <table className="table table-sm small mb-0">
            <thead>
              <tr style={{ backgroundColor: '#f3e5f5' }}>
                <th style={{ color: ACCENT5 }}>Variant</th>
                <th style={{ color: ACCENT5 }}>Pop Freq</th>
                <th style={{ color: ACCENT5 }}>Domain</th>
                <th style={{ color: ACCENT5 }}>Phenotype Note</th>
              </tr>
            </thead>
            <tbody>
              {[
                { v: 'p.His170Arg (c.509A>G)', f: '35%', d: 'HADH-GDH interface', n: 'MOST COMMON; disrupts GDH inhibitory contact; neonatal; diazoxide responsive' },
                { v: 'p.Leu147Pro (c.440T>C)', f: '20%', d: 'Dimer interface', n: 'Destabilises homodimer; null-equivalent; neonatal severe' },
                { v: 'p.Arg236Gln (c.707G>A)', f: '15%', d: 'Catalytic / NAD-binding', n: 'Abolishes catalysis; complete LOF; neonatal' },
                { v: 'p.Glu96Ter (c.286G>T)', f: '12%', d: 'Premature stop (null)', n: 'NMD; no HADH protein; neonatal severe' },
                { v: 'c.636+1G>A',             f: '10%', d: 'Splice donor site', n: 'Null; aberrant splicing; neonatal' },
                { v: 'p.Val96Met (c.286G>A)',   f: '8%',  d: 'Catalytic core', n: 'Partial LOF; mild attenuated; fasting-only' },
              ].map(row => (
                <tr key={row.v}>
                  <td className="font-monospace" style={{ fontSize: '0.72rem', color: ACCENT5 }}>{row.v}</td>
                  <td style={{ color: ACCENT }}>{row.f}</td>
                  <td className="text-muted small">{row.d}</td>
                  <td className="text-muted small">{row.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-2 small text-muted">
          p.His170Arg is the most common variant (35%), located at the HADH-GDH protein-protein
          interaction interface. Biallelic LOF required for disease (AR). SCHAD/HHF4 is always
          diffuse HI — no focal lesion on 18F-DOPA PET.
        </div>
      </InfoBox>

      {/* Differential table */}
      <InfoBox title="Key Differentials — Protein-Sensitive / Hyperinsulinism" color={ACCENT6}>
        <table className="table table-sm small mb-0">
          <thead>
            <tr style={{ backgroundColor: '#e0f2f1' }}>
              <th style={{ color: ACCENT6 }}>Disorder</th>
              <th style={{ color: ACCENT6 }}>Ammonia</th>
              <th style={{ color: ACCENT6 }}>Protein-sensitive</th>
              <th style={{ color: ACCENT6 }}>C4-OH NBS</th>
              <th style={{ color: ACCENT6 }}>Key Discriminator</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ backgroundColor: '#fff3e0' }}>
              <td className="fw-bold" style={{ color: ACCENT }}>SCHAD/HHF4 (HADH)</td>
              <td style={{ color: ACCENT3 }}><strong>NORMAL</strong></td>
              <td style={{ color: ACCENT8 }}>YES</td>
              <td>Mildly elevated</td>
              <td>Ammonia NORMAL; HADH LOF → GDH uninhibited</td>
            </tr>
            <tr>
              <td className="fw-bold" style={{ color: ACCENT5 }}>GLUD1/HHS</td>
              <td style={{ color: ACCENT4 }}><strong>HIGH 100-500 μmol/L</strong></td>
              <td style={{ color: ACCENT8 }}>YES</td>
              <td>Normal</td>
              <td>Hyperammonaemia = KEY POSITIVE for GLUD1 (absent in SCHAD)</td>
            </tr>
            <tr>
              <td className="fw-bold" style={{ color: ACCENT7 }}>ABCC8/KCNJ11</td>
              <td style={{ color: ACCENT3 }}>NORMAL</td>
              <td>NO</td>
              <td>Normal</td>
              <td>NOT protein-sensitive; focal possible on 18F-DOPA PET</td>
            </tr>
            <tr>
              <td className="fw-bold" style={{ color: ACCENT7 }}>IBD (isobutyryl-CoA)</td>
              <td style={{ color: ACCENT3 }}>NORMAL</td>
              <td>NO</td>
              <td>Elevated C4-OH</td>
              <td>No HI; benign; metabolic not endocrine</td>
            </tr>
          </tbody>
        </table>
      </InfoBox>
    </div>
  );
}

// ── Definitions tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  const terms = data.terms || {};

  return (
    <div>
      <InfoBox title="Disease Overview" color={ACCENT}>
        <dl className="small mb-0">
          <dt>Disease</dt><dd>{data.disease_name}</dd>
          <dt>Gene</dt><dd>{data.gene}</dd>
          <dt>OMIM Gene</dt><dd>{data.omim_gene}</dd>
          <dt>OMIM Disease</dt><dd>{data.omim_disease}</dd>
          <dt>Inheritance</dt><dd>{data.inheritance}</dd>
        </dl>
      </InfoBox>

      <InfoBox title="Key Terms — SCHAD/HHF4 (15 Definitions)" color={ACCENT6}>
        {Object.entries(terms).map(([termKey, termVal]) => (
          <div key={termKey} className="mb-3">
            <div className="small fw-bold" style={{
              color: termKey.includes('KATP') || termKey.includes('GDH') || termKey.includes('interaction') ? ACCENT6 :
                     termKey.includes('Diazoxide') || termKey.includes('sirolimus') || termKey.includes('Pancreat') ? ACCENT3 :
                     termKey.includes('VPA') || termKey.includes('FASTING') ? ACCENT4 :
                     termKey.includes('p_His') || termKey.includes('genetics') ? ACCENT5 :
                     termKey.includes('Protein') || termKey.includes('leucine') ? ACCENT8 :
                     termKey.includes('C4_OH') || termKey.includes('NBS') ? ACCENT2 :
                     ACCENT
            }}>{termKey.replace(/_/g, ' ')}</div>
            <div className="small text-muted">{termVal}</div>
            <hr className="my-1" />
          </div>
        ))}
      </InfoBox>

      <InfoBox title="Comparison — SCHAD vs GLUD1 vs ABCC8 (Key Exam)" color={ACCENT}>
        <div className="row">
          <div className="col-md-4">
            <div className="fw-bold small mb-1" style={{ color: ACCENT }}>SCHAD/HHF4 (HADH)</div>
            <ul className="small mb-0">
              <li>Ammonia: NORMAL (KEY NEGATIVE)</li>
              <li>Protein-sensitive: YES (leucine → uninhibited GDH)</li>
              <li>C4-OH: mildly elevated (NBS)</li>
              <li>Always diffuse HI</li>
              <li>Diazoxide responsive ~80%</li>
              <li>Mechanism: GDH disinhibition (NOT FAO crisis)</li>
            </ul>
          </div>
          <div className="col-md-4">
            <div className="fw-bold small mb-1" style={{ color: ACCENT5 }}>GLUD1/HHS (gain-of-function)</div>
            <ul className="small mb-0">
              <li>Ammonia: HIGH 100-500 μmol/L</li>
              <li>Protein-sensitive: YES</li>
              <li>C4-OH: Normal</li>
              <li>Always diffuse HI</li>
              <li>Diazoxide responsive variable</li>
              <li>Mechanism: GDH GoF → excess glutamate oxidation</li>
            </ul>
          </div>
          <div className="col-md-4">
            <div className="fw-bold small mb-1" style={{ color: ACCENT7 }}>ABCC8/KCNJ11 (KATP defects)</div>
            <ul className="small mb-0">
              <li>Ammonia: Normal</li>
              <li>Protein-sensitive: NO</li>
              <li>C4-OH: Normal</li>
              <li>Focal possible (18F-DOPA PET)</li>
              <li>Diazoxide: often resistant (KATP abolished)</li>
              <li>Mechanism: KATP LOF → constitutive depolarisation</li>
            </ul>
          </div>
        </div>
      </InfoBox>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function SHADPage() {
  const [tab, setTab]            = useState('Overview');
  const [overview, setOverview]  = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]          = useState(null);
  const [error, setError]        = useState(null);

  useEffect(() => {
    fetch(`${API}/api/schad/overview`).then(r => r.json()).then(setOverview).catch(() => setError('Backend offline'));
    fetch(`${API}/api/schad/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/schad/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3 d-flex align-items-center gap-2 flex-wrap">
        <Link href="/" className="btn btn-sm btn-outline-secondary">Home</Link>
        <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
          SCHAD Epilepsy Dashboard
        </h4>
        <span className="badge" style={{ backgroundColor: ACCENT }}>SCHAD / HHF4</span>
        <span className="badge" style={{ backgroundColor: ACCENT5 }}>4q22.1</span>
        <span className="badge" style={{ backgroundColor: ACCENT7 }}>AR</span>
        <span className="badge" style={{ backgroundColor: ACCENT6 }}>GDH Disinhibition</span>
        <span className="badge" style={{ backgroundColor: ACCENT8 }}>Protein-Sensitive HI</span>
        <span className="badge" style={{ backgroundColor: ACCENT3 }}>Ammonia NORMAL</span>
        <span className="badge" style={{ backgroundColor: ACCENT4 }}>Fasting ABSOLUTE CI</span>
        <span className="badge" style={{ backgroundColor: ACCENT2 }}>C4-OH NBS</span>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active' : ''}`}
              style={tab === t ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(t)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'              && <OverviewTab    data={overview}  />}
      {tab === 'Patients & Biomarkers' && <PatientsTab    data={breakdown} />}
      {tab === 'Treatments & Genetics' && <TreatmentsTab  data={breakdown} />}
      {tab === 'Definitions'           && <DefinitionsTab data={defs}      />}
    </div>
  );
}
