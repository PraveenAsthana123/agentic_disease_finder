'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Iron & Imaging', 'Treatments', 'Definitions'];
const COLOR = '#4a148c';   // deep purple — DCAF17/WSS (nucleolar/ribosome biology, rare)
const LIGHT = '#f3e5f5';

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

function Bar({ label, value, max, color = COLOR }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ variant, text }) {
  const bg = variant === 'danger' ? '#ffebee' : variant === 'warning' ? '#fff8e1' : variant === 'success' ? '#e8f5e9' : '#e8eaf6';
  const border = variant === 'danger' ? '#c62828' : variant === 'warning' ? '#f57f17' : variant === 'success' ? '#2e7d32' : COLOR;
  return (
    <div className="mb-2 p-2 rounded small" style={{ background: bg, borderLeft: `4px solid ${border}` }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview...</div>;
  const kpis = data.kpis || {};
  const phenoDist = data.phenotype_distribution || [];
  const highlights = data.clinical_highlights || [];
  const cis = data.contraindications || [];
  const thresholds = data.thresholds || [];

  return (
    <div>
      <div className="alert py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
        <strong>DCAF17/C2orf37 (2q31.1) — 263aa WD-repeat nucleolar protein · OMIM Gene 612515 / WSS 241080 · AR Biallelic LOF:</strong>{' '}
        CUL4A-DDB1 E3 ubiquitin ligase substrate receptor; nucleolar ribosome biogenesis regulator.
        DCAF17 LOF → rRNA processing defect → ribosome output failure → tissue-selective vulnerability
        (gonads, neurons, hair follicles, β-cells — all high protein synthesis demand).{' '}
        <strong className="text-danger">Cardinal Pentad: Hypergonadotropic Hypogonadism + Diffuse Alopecia + Diabetes + Extrapyramidal Features + Sensorineural Hearing Loss.</strong>{' '}
        <span className="fw-bold" style={{ color: COLOR }}>
          Saudi/ME founder c.436delC (p.Gln146Lysfs*48) ~60% of cases. PHT/CBZ AVOID (worsen extrapyramidal + HRT failure via CYP3A4).
          LEV PREFERRED (no CYP induction, HRT-safe). HRT (oestrogen/testosterone) MANDATORY.
          GP iron MILD — NO Eye-of-Tiger (DDx PKAN). NO cortical iron (DDx Aceruloplasminemia/CP).
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Classic WSS" value={kpis.n_classic} color="#c62828" />
        <KPI label="Neuro Predominant" value={kpis.n_neurodegen} color="#e65100" />
        <KPI label="Endocrine Predominant" value={kpis.n_endocrine} color="#1565c0" />
        <KPI label="Mild/Late-onset" value={kpis.n_mild} color="#388e3c" />
        <KPI label="Hypogonadism" value={`${kpis.hypogonadism_pct}%`} color="#c62828" />
        <KPI label="Alopecia" value={`${kpis.alopecia_pct}%`} color="#c62828" />
        <KPI label="Diabetes" value={`${kpis.diabetes_pct}%`} color="#e65100" />
        <KPI label="Insulin-Dependent" value={`${kpis.insulin_dependent_pct}%`} color="#e65100" />
        <KPI label="Chorea" value={`${kpis.chorea_pct}%`} color={COLOR} />
        <KPI label="Dystonia" value={`${kpis.dystonia_pct}%`} color={COLOR} />
        <KPI label="Dysarthria" value={`${kpis.dysarthria_pct}%`} color={COLOR} />
        <KPI label="Hearing Loss" value={`${kpis.hearing_loss_pct}%`} color="#1565c0" />
        <KPI label="Seizures" value={`${kpis.seizures_pct}%`} color="#e65100" />
        <KPI label="Cognitive Decline" value={`${kpis.cognitive_decline_pct}%`} color={COLOR} />
        <KPI label="WM T2 Changes" value={`${kpis.wm_changes_pct}%`} color="#1565c0" />
        <KPI label="GP Iron (SWI)" value={`${kpis.gp_iron_pct}%`} color="#4a148c" />
        <KPI label="Mean Onset (yr)" value={kpis.mean_onset_yr} color={COLOR} />
      </div>

      {/* Phenotype Distribution */}
      <SectionCard title="Phenotype Distribution — 4 Clinical Subtypes">
        <div className="row">
          {phenoDist.map(ph => (
            <div key={ph.phenotype} className="col-md-3 mb-2">
              <div className="card border-0 shadow-sm h-100 text-center">
                <div className="card-body py-3">
                  <div className="fw-bold" style={{ color: COLOR }}>{ph.phenotype}</div>
                  <div className="display-6 fw-bold">{ph.n}</div>
                  <div className="text-muted small">{ph.pct}% of cohort</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Clinical Highlights */}
      <SectionCard title="Cardinal Feature Frequency — 40 Patients">
        {highlights.map(h => (
          <div key={h.finding} className="mb-3">
            <Bar label={h.finding} value={h.pct} max={100} />
            <div className="text-muted" style={{ fontSize: '0.78rem', marginLeft: 4 }}>{h.note}</div>
          </div>
        ))}
      </SectionCard>

      {/* Contraindications */}
      <SectionCard title="Drug Contraindications, Cautions & Preferred Agents">
        {cis.map(ci => (
          <Alert
            key={ci.drug}
            variant={ci.severity === 'AVOID' ? 'danger' : ci.severity.includes('PREFERRED') ? 'success' : ci.severity.includes('MANDATORY') ? 'success' : 'warning'}
            text={
              <>
                <strong>{ci.drug}</strong> — <strong>{ci.severity}</strong>: {ci.reason}.
                {ci.alternative && <em> Alternative: {ci.alternative}</em>}
              </>
            }
          />
        ))}
      </SectionCard>

      {/* Clinical Thresholds */}
      <SectionCard title="Clinical Thresholds & Action Points">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-light">
              <tr><th>Metric</th><th>Threshold</th><th>Action</th></tr>
            </thead>
            <tbody>
              {thresholds.map(t => (
                <tr key={t.metric}>
                  <td className="fw-bold">{t.metric}</td>
                  <td><span className="badge" style={{ background: COLOR }}>{t.threshold}</span></td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const phenoBreakdown = data.phenotype_breakdown || [];
  const variantBreakdown = data.variant_breakdown || [];
  const treatBreakdown = data.treatment_breakdown || [];
  const patientTable = data.patient_table || [];

  return (
    <div>
      <SectionCard title="Phenotype Breakdown — 4 Clinical Subtypes">
        <div className="row">
          {phenoBreakdown.map(ph => (
            <div key={ph.phenotype} className="col-md-6 mb-3">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-body small">
                  <div className="fw-bold mb-2" style={{ color: COLOR }}>{ph.phenotype} — {ph.n} pts ({ph.pct}%)</div>
                  <div>Mean onset: <strong>{ph.mean_onset_yr} yr</strong></div>
                  <div>Hypogonadism: <strong>{ph.hypogonadism_pct}%</strong></div>
                  <div>Alopecia: <strong>{ph.alopecia_pct}%</strong></div>
                  <div>Diabetes: <strong>{ph.diabetes_pct}%</strong></div>
                  <div>Chorea: <strong>{ph.chorea_pct}%</strong></div>
                  <div>Dystonia: <strong>{ph.dystonia_pct}%</strong></div>
                  <div>Dysarthria: <strong>{ph.dysarthria_pct}%</strong></div>
                  <div>Seizures: <strong>{ph.seizures_pct}%</strong></div>
                  <div>WM changes: <strong>{ph.wm_changes_pct}%</strong></div>
                  <div>GP iron: <strong>{ph.gp_iron_pct}%</strong></div>
                  <div>Cognitive decline: <strong>{ph.cognitive_decline_pct}%</strong></div>
                  <div>Hearing loss: <strong>{ph.hearing_loss_pct}%</strong></div>
                  <div>Neuropathy: <strong>{ph.neuropathy_pct}%</strong></div>
                  <div>Mean FSH: <strong>{ph.mean_fsh} IU/L</strong></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="DCAF17 Variant Distribution (biallelic LOF)">
        {variantBreakdown.map(v => (
          <Bar key={v.variant} label={`${v.variant} (n=${v.n})`} value={v.pct} max={100} />
        ))}
        <div className="card mt-3" style={{ background: LIGHT }}>
          <div className="card-body small">
            <strong>Variant Context:</strong> Saudi founder c.436delC (p.Gln146Lysfs*48) disrupts WD-repeat 2 domain
            (DDB1-binding interface) → complete CUL4A-DDB1 substrate receptor loss → severe Classic WSS phenotype.
            European/other variants: diverse missense at WD-propeller → partial LOF → milder or late-onset phenotypes.
            Consanguinity in ~70% of reported families worldwide.
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Treatment Distribution">
        {treatBreakdown.map(t => (
          <Bar key={t.treatment} label={`${t.treatment} (n=${t.n})`} value={t.pct} max={100} color="#1565c0" />
        ))}
      </SectionCard>

      <SectionCard title="Patient Table — Top 25 by Onset Age">
        <div className="table-responsive">
          <table className="table table-sm table-striped small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Phenotype</th><th>Sex</th><th>Onset</th><th>Dur</th>
                <th>Hypogon.</th><th>Alopecia</th><th>DM</th>
                <th>Chorea</th><th>Dystonia</th><th>Seizures</th>
                <th>FSH (IU/L)</th><th>Treatment</th>
              </tr>
            </thead>
            <tbody>
              {patientTable.map(p => (
                <tr key={p.id}>
                  <td className="fw-bold">{p.id}</td>
                  <td className="small">{p.phenotype.replace(' Predominant', '')}</td>
                  <td>{p.sex === 'Female' ? '♀' : '♂'}</td>
                  <td>{p.onset_yr}y</td>
                  <td>{p.disease_dur_yr}y</td>
                  <td>{p.hypogonadism ? '✓' : '—'}</td>
                  <td>{p.alopecia ? '✓' : '—'}</td>
                  <td>{p.diabetes ? '✓' : '—'}</td>
                  <td>{p.chorea ? '〜' : '—'}</td>
                  <td>{p.dystonia ? '◈' : '—'}</td>
                  <td>{p.seizures ? '⚡' : '—'}</td>
                  <td><strong style={{ color: p.fsh_iu_l > 30 ? '#c62828' : '#388e3c' }}>{p.fsh_iu_l}</strong></td>
                  <td className="small">{p.treatment}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function IronTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const ironRegions = data.iron_regions || [];

  return (
    <div>
      <SectionCard title="Iron & Imaging Pattern — Key DDx from Classic NBIA">
        <div className="alert alert-info small mb-3">
          <strong>WSS brain iron is MILD and NBIA-adjacent — NOT Classic NBIA:</strong>{' '}
          GP SWI hypointensity present in ~60% Classic WSS but lacks Eye-of-Tiger (DDx PKAN).
          White matter T2 changes (frontal periventricular) appear BEFORE iron accumulation — earliest MRI marker.
          NO cortical iron (DDx Aceruloplasminemia/CP). NO leukodystrophy (DDx FAHN/NBIA3).
          NO cavitations (DDx FTL/NBIA7).
        </div>
        {ironRegions.map(r => (
          <div key={r.region} className="mb-3">
            {r.pct > 0
              ? <Bar label={r.region} value={r.pct} max={100} color={r.region.includes('NO') ? '#388e3c' : COLOR} />
              : <div className="d-flex justify-content-between small mb-1">
                  <span style={{ color: '#388e3c' }}><strong>✓ {r.region}</strong></span>
                  <span className="badge bg-success">0%</span>
                </div>
            }
            <div className="text-muted" style={{ fontSize: '0.78rem', marginLeft: 4 }}>{r.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="MRI Comparison — WSS vs NBIA Subtypes">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-light">
              <tr>
                <th>MRI Feature</th>
                <th style={{ color: COLOR }}>WSS (DCAF17)</th>
                <th>PKAN (PANK2)</th>
                <th>CP (Aceruloplasminemia)</th>
                <th>FTL (NBIA7)</th>
                <th>FAHN (FA2H)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="fw-bold">Eye-of-Tiger (GP)</td>
                <td><span className="badge bg-success">ABSENT</span></td>
                <td><span className="badge bg-danger">PATHOGNOMONIC</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-success">Absent</span></td>
              </tr>
              <tr>
                <td className="fw-bold">Cortical Iron</td>
                <td><span className="badge bg-success">ABSENT</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-danger">UNIQUE/PATHOGNOMONIC</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-success">Absent</span></td>
              </tr>
              <tr>
                <td className="fw-bold">WM T2 changes</td>
                <td><span className="badge bg-warning text-dark">MILD periventricular</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-danger">Leukodystrophy (prominent)</span></td>
              </tr>
              <tr>
                <td className="fw-bold">GP Iron (SWI)</td>
                <td><span className="badge bg-warning text-dark">Mild</span></td>
                <td><span className="badge bg-danger">Severe</span></td>
                <td><span className="badge bg-warning text-dark">Moderate</span></td>
                <td><span className="badge bg-warning text-dark">Moderate-Severe</span></td>
                <td><span className="badge bg-warning text-dark">Mild</span></td>
              </tr>
              <tr>
                <td className="fw-bold">Cavitations (BG)</td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-danger">PATHOGNOMONIC (advanced)</span></td>
                <td><span className="badge bg-success">Absent</span></td>
              </tr>
              <tr>
                <td className="fw-bold">Serum Ferritin</td>
                <td><span className="badge bg-success">Normal</span></td>
                <td><span className="badge bg-success">Normal</span></td>
                <td><span className="badge bg-danger">HIGH &gt;500 ng/mL</span></td>
                <td><span className="badge bg-warning text-dark">LOW &lt;30 ng/mL</span></td>
                <td><span className="badge bg-success">Normal</span></td>
              </tr>
              <tr>
                <td className="fw-bold">Ceruloplasmin</td>
                <td><span className="badge bg-success">Normal</span></td>
                <td><span className="badge bg-success">Normal</span></td>
                <td><span className="badge bg-danger">Undetectable</span></td>
                <td><span className="badge bg-success">Normal</span></td>
                <td><span className="badge bg-success">Normal</span></td>
              </tr>
              <tr>
                <td className="fw-bold">Hypogonadism</td>
                <td><span className="badge bg-danger">CARDINAL FEATURE</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-success">Absent</span></td>
              </tr>
              <tr>
                <td className="fw-bold">Alopecia</td>
                <td><span className="badge bg-danger">CARDINAL FEATURE</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-success">Absent</span></td>
                <td><span className="badge bg-success">Absent</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;

  const treatments = [
    {
      name: "HRT (Oestrogen + Progestogen) — Females",
      level: "MANDATORY (primary ovarian failure; bone + cardiovascular protection)",
      dose: "17β-oestradiol 2 mg/day orally or patch equivalent; micronised progesterone 100-200 mg days 1-14 of cycle",
      mechanism: "Replaces absent endogenous oestrogen from primary gonadal failure; protects bone mineral density + cardiovascular health",
      note: "Continue until average menopause age (51yr); annual review. CRITICAL: avoid PHT/CBZ/OXC (CYP3A4 inducers reduce oestrogen levels → HRT failure).",
      color: "#c62828",
    },
    {
      name: "HRT (Testosterone) — Males",
      level: "MANDATORY (primary testicular failure; bone + cardiovascular + metabolic protection)",
      dose: "Testosterone undecanoate IM 1000 mg/12 wk; or testosterone gel 50 mg/day topically",
      mechanism: "Replaces absent testosterone from primary testicular failure; prevents osteoporosis, metabolic syndrome, fatigue",
      note: "Monitor haematocrit (secondary polycythaemia risk); testosterone levels; LH/FSH. PHT/CBZ must be avoided — CYP3A4 induction markedly reduces testosterone bioavailability.",
      color: "#c62828",
    },
    {
      name: "Levetiracetam (LEV) — AED First-Line",
      level: "PREFERRED first-line (Level B extrapolated; no CYP induction; HRT-safe)",
      dose: "500 mg BD, titrate to 1000-3000 mg/day (renal dose adjustment if eGFR <80)",
      mechanism: "SV2A modulator; renal excretion (66% unchanged); NO CYP450 induction → sex hormone replacement levels unaffected",
      note: "Broad-spectrum: effective against myoclonic + focal seizures (both relevant in WSS). Monitor mood/irritability (10-15%). Safe with testosterone/oestrogen.",
      color: "#1565c0",
    },
    {
      name: "Tetrabenazine / Deutetrabenazine — Chorea",
      level: "Level D (extrapolated from Huntington disease; <5 WSS cases reported)",
      dose: "Tetrabenazine: 12.5 mg/day TID, titrate to max 100 mg/day. Deutetrabenazine: 6 mg BD, titrate to max 48 mg/day",
      mechanism: "VMAT2 inhibitor → presynaptic dopamine depletion → reduced striatal dopamine tone → chorea reduction",
      note: "NOT effective for dystonia. Monitor depression (15% risk), sedation, QTc. Deutetrabenazine preferred (better tolerability). No HRT interaction.",
      color: "#4a148c",
    },
    {
      name: "GPi-DBS (Globus Pallidus internus Deep Brain Stimulation)",
      level: "Investigational / Level D (extrapolated; <5 WSS cases; NBIA-expert centre required)",
      dose: "Standard GPi-DBS coordinates; programming individualised for chorea/dystonia balance",
      mechanism: "GPi inhibition → thalamo-cortical circuit normalisation → reduced chorea + dystonia",
      note: "Best candidates: predominantly dystonic + severe functional impairment + stable cognition. Optimise HRT before DBS programming. Multidisciplinary movement disorder team required.",
      color: "#4a148c",
    },
    {
      name: "Metformin / SGLT2 Inhibitor — Diabetes",
      level: "Standard DM management (first-line type 2 protocol; insulin if β-cell exhaustion)",
      dose: "Metformin 500 mg BD → titrate to 2000 mg/day; insulin if HbA1c >8% uncontrolled",
      mechanism: "Metformin: AMPK activation → hepatic glucose output reduction. SGLT2: glucosuria. Insulin: β-cell failure replacement",
      note: "Endocrinology co-management. ~30% progress to insulin-dependence. Avoid rosiglitazone (hepatic CYP concern). Regular HbA1c/eGFR monitoring.",
      color: "#e65100",
    },
    {
      name: "Botulinum Toxin Type A — Oromandibular Dystonia/Dysarthria",
      level: "Level B (established for focal dystonia; extrapolated to WSS oromandibular dystonia)",
      dose: "Oromandibular: 20-40 U per masseter + pterygoid; repeat every 12 wk; speech therapy adjunct",
      mechanism: "SNARE complex inhibition → NMJ block → oromandibular muscle relaxation → improved speech",
      note: "Effective for local dystonia component even when generalised chorea persists. Combine with VMAT2 inhibitor for best chorea+dystonia combination.",
      color: "#2e7d32",
    },
  ];

  return (
    <div>
      <SectionCard title="Priority: Drug Contraindications & HRT Interaction">
        <Alert variant="danger" text={<><strong>PHT (Phenytoin) AVOID</strong> — CYP2C9/3A4 inducer: markedly reduces oestrogen + testosterone plasma levels → HRT failure → osteoporosis + cardiovascular risk + testosterone deficiency untreated. Also worsens extrapyramidal features (chorea/dystonia aggravation).</>} />
        <Alert variant="danger" text={<><strong>CBZ (Carbamazepine) AVOID</strong> — Strong CYP3A4 inducer: same HRT failure mechanism as PHT, even more potent. Exacerbates extrapyramidal features. First-choice AED replacement: LEV.</>} />
        <Alert variant="warning" text={<><strong>OXC (Oxcarbazepine) CAUTION</strong> — Moderate CYP3A4 inducer: reduces HRT efficacy; monitor sex hormone levels if used; extrapyramidal overlap.</>} />
        <Alert variant="warning" text={<><strong>VPA CAUTION — POLG1 screen MANDATORY first</strong> — POLG exclusion required before any VPA use; hepatotoxic in POLG-positive + ribosome pathway overlap risk.</>} />
        <Alert variant="success" text={<><strong>LEV PREFERRED FIRST-LINE</strong> — No CYP450 induction; sex hormone levels unaffected; renal excretion; broad-spectrum (myoclonic + focal). Safe with HRT in all WSS patients.</>} />
        <Alert variant="success" text={<><strong>HRT MANDATORY</strong> — Oestrogen/testosterone replacement regardless of neurological stage; protects bone, cardiovascular, metabolic health; does NOT treat neurodegeneration but prevents HRT-deprivation complications.</>} />
      </SectionCard>

      {treatments.map(t => (
        <div key={t.name} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${t.color}` }}>
          <div className="card-body small">
            <div className="fw-bold mb-1" style={{ color: t.color }}>{t.name}</div>
            <div><strong>Evidence:</strong> {t.level}</div>
            <div><strong>Dose:</strong> {t.dose}</div>
            <div><strong>Mechanism:</strong> {t.mechanism}</div>
            <div className="text-muted mt-1">{t.note}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const defs = data.definitions || [];
  return (
    <div>
      {defs.map(d => (
        <div key={d.term} className="card mb-3 shadow-sm">
          <div className="card-body">
            <div className="fw-bold small mb-1" style={{ color: COLOR }}>{d.term.replace(/-/g, ' ')}</div>
            <div className="text-muted small mb-1 fst-italic">{d.full}</div>
            <div className="small" style={{ whiteSpace: 'pre-wrap' }}>{d.detail}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function DCAF17Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/dcaf17/overview`).then(r => r.json()),
      fetch(`${API}/api/dcaf17/breakdown`).then(r => r.json()),
      fetch(`${API}/api/dcaf17/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: COLOR }} /></div>;
  if (error) return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  const panels = [
    <OverviewTab key="ov" data={overview} />,
    <PatientsTab key="pt" data={breakdown} />,
    <IronTab key="ir" data={breakdown} />,
    <TreatmentsTab key="tr" data={breakdown} />,
    <DefinitionsTab key="df" data={definitions} />,
  ];

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
          🧬 DCAF17 Woodhouse-Sakati Syndrome (WSS — OMIM 241080)
        </h4>
        <div className="text-muted small">
          DCAF17/C2orf37 (2q31.1) · 263aa WD-repeat Nucleolar Protein · AR Biallelic LOF · ~60-70 families worldwide 2026 ·
          Cardinal Pentad: Hypogonadism + Alopecia + Diabetes + Extrapyramidal + Hearing Loss ·
          Saudi founder c.436delC ~60% · GP iron mild (no Eye-of-Tiger) · PHT/CBZ AVOID (HRT failure) ·
          LEV PREFERRED · HRT MANDATORY · 40-patient cohort seed-531 · First described Woodhouse &amp; Sakati 1983
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {panels[tab]}
    </div>
  );
}
