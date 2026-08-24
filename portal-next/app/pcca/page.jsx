'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4527a0';   // deep purple — PCC complex / propionyl-CoA carboxylase
const ACCENT2 = '#b71c1c';   // dark red — metabolic crisis / hyperammonemia DANGER
const ACCENT3 = '#e65100';   // deep orange — propionyl-CoA accumulation / organic acids
const ACCENT4 = '#1565c0';   // deep blue — carnitine / treatment / level A
const ACCENT5 = '#880e4f';   // dark magenta — cardiomyopathy / BG infarct / complications
const ACCENT6 = '#1b5e20';   // dark green — KEY NEGATIVES / absent methylmalonate / no biotin
const ACCENT7 = '#37474f';   // dark slate — variant data / gene card
const ACCENT8 = '#4e342e';   // dark brown — methylcitrate / propionylglycine pathognomonic

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

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      {text}
    </div>
  );
}

function Section({ title, children, color = ACCENT }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

export default function PCCAPage() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/pcca/overview`).then(r => r.json()),
      fetch(`${API}/api/pcca/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pcca/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov)  return <div className="text-center p-5 text-muted">Loading PCCA Dashboard…</div>;

  const kpis = ov.kpis || {};

  return (
    <div className="container-fluid px-3 py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🧬 PCCA Epilepsy — Propionic Acidemia Type A (Propionyl-CoA Carboxylase Alpha Subunit Deficiency)
        </h4>
        <div className="text-muted small">
          PCCA · 13q32.3 · AR · OMIM *232000 / #606054 · 728 aa · BC + BCCP domains ·
          Lys669 = biotin attachment site · (αβ)₆ dodecamer with PCCB ·
          NBS: C3 (propionylcarnitine) elevated · Methylcitrate PATHOGNOMONIC ·
          NO methylmalonate (vs MMA KEY NEG) · Biotin NOT effective · VPA ABSOLUTE CI · n={ov.cohort_n}
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          <Alert
            variant="danger"
            text="⚠️ PROPIONIC ACIDEMIA — ACUTE DECOMPENSATION: Hyperammonemia (NH₃ > 300 µmol/L = EMERGENT), metabolic acidosis + ketoacidosis, hypoglycemia. FIRST response: IV glucose GIR 8-12 + STOP PROTEIN + ammonia scavengers. VPA ABSOLUTE CI — valproyl-CoA inhibits PCC directly + depletes carnitine: FATAL. Fasting = EXTREME HAZARD. Always give IV glucose in intercurrent illness. Biotin does NOT help (PA is not a biotin disorder)."
          />

          {/* Gene & Disease Overview */}
          <Section title="Gene & Disease Overview" color={ACCENT}>
            <div className="row g-2 mb-2">
              {[
                ['Gene',          ov.gene || 'PCCA'],
                ['Full Name',     ov.full_name || 'Propionyl-CoA Carboxylase Alpha'],
                ['Chromosome',    ov.chromosome || '13q32.3'],
                ['Inheritance',   ov.inheritance || 'AR'],
                ['OMIM Gene',     ov.omim_gene || '*232000'],
                ['OMIM Disease',  ov.omim_disease || '#606054'],
                ['Protein',       ov.protein_size || '728 aa; BC+BCCP; Lys669-biotin'],
                ['Prevalence',    ov.prevalence || '1:100,000–150,000 (EU)'],
                ['NBS Primary',   ov.nbs_primary || 'C3 (propionylcarnitine) elevated'],
                ['NBS Secondary', ov.nbs_secondary || 'C3/C2 ratio; urine OA (methylcitrate)'],
              ].map(([k, v]) => (
                <div key={k} className="col-12 col-md-6 col-lg-4">
                  <span className="fw-semibold text-muted small">{k}: </span>
                  <span className="small">{v}</span>
                </div>
              ))}
            </div>
            <div className="alert alert-info py-2" style={{ fontSize: 12 }}>
              <strong>Function:</strong> {ov.function}<br />
              <strong>Mechanism:</strong> {ov.mechanism}<br />
              <strong>Key Negative:</strong> {ov.key_negative}
            </div>
          </Section>

          {/* KPIs */}
          <Section title={`Cohort KPIs — ${ov.cohort_n} Patients`} color={ACCENT3}>
            <div className="row g-2">
              <KPI label="Avg C3 (µmol/L)" value={kpis.avg_c3_umol_l} color={ACCENT3} />
              <KPI label="Avg Methylcitrate" value={kpis.avg_methylcitrate} color={ACCENT8} />
              <KPI label="Avg NH₃ (µmol/L)" value={kpis.avg_ammonia_umol_l} color={ACCENT2} />
              <KPI label="Avg Carnitine" value={kpis.avg_free_carnitine} color={ACCENT4} />
              <KPI label="Seizures %" value={kpis.seizure_pct + '%'} color={ACCENT2} />
              <KPI label="Cardiomyopathy %" value={kpis.cardiomyopathy_pct + '%'} color={ACCENT5} />
              <KPI label="QT Prolonged %" value={kpis.qt_prolonged_pct + '%'} color={ACCENT5} />
              <KPI label="BG Infarct %" value={kpis.bg_infarct_pct + '%'} color={ACCENT5} />
              <KPI label="Neutropenia %" value={kpis.neutropenia_pct + '%'} color={ACCENT7} />
              <KPI label="NBS Detected %" value={kpis.nbs_detected_pct + '%'} color={ACCENT4} />
              <KPI label="Hypoglycemia %" value={kpis.hypoglycemia_pct + '%'} color={ACCENT3} />
              <KPI label="Liver Tx %" value={kpis.liver_transplant_pct + '%'} color={ACCENT6} />
            </div>
          </Section>

          {/* Phenotype Distribution */}
          <Section title="Phenotype Distribution" color={ACCENT2}>
            {(ov.phenotype_distribution || []).map((p, i) => (
              <PctBar key={i} label={p.phenotype} pct={p.pct}
                color={i === 0 ? ACCENT2 : i === 1 ? ACCENT5 : i === 2 ? ACCENT3 : ACCENT4} />
            ))}
          </Section>

          {/* PCC Pathway */}
          <Section title="PCC Two-Step Reaction Mechanism (PCCA performs Step 1)" color={ACCENT8}>
            <div className="row g-2">
              {(ov.pcc_pathway || []).map((s, i) => (
                <div key={i} className="col-12 col-md-6">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body py-2 px-3">
                      <div className="fw-bold small" style={{ color: ACCENT8 }}>{s.step}</div>
                      <div className="font-monospace" style={{ fontSize: 11, color: ACCENT }}>{s.reaction}</div>
                      <div className="text-muted" style={{ fontSize: 11 }}>Subunit: {s.enzyme_subunit}</div>
                      {s.cofactor && <div className="text-muted" style={{ fontSize: 11 }}>Cofactor: {s.cofactor}</div>}
                      <div className="mt-1 small" style={{ color: ACCENT2 }}>
                        <strong>PCCA LOF consequence:</strong> {s.loss_when_PCCA_mutant}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* PCCA vs PCCB */}
          {ov.pcca_vs_pccb && (
            <Section title={ov.pcca_vs_pccb.title} color={ACCENT6}>
              <div className="text-muted small mb-2">{ov.pcca_vs_pccb.note}</div>
              <div className="table-responsive">
                <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>Feature</th>
                      <th style={{ color: '#ce93d8' }}>PCCA (this disease)</th>
                      <th style={{ color: '#80cbc4' }}>PCCB (comparator)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.pcca_vs_pccb.comparison || []).map((row, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{row.feature}</td>
                        <td>{row.PCCA}</td>
                        <td>{row.PCCB}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          )}

          {/* High-Risk Situations */}
          <Section title="High-Risk Situations" color={ACCENT2}>
            {(ov.high_risk_situations || []).map((r, i) => (
              <div key={i} className="d-flex align-items-start mb-2">
                <span className={`badge me-2 ${
                  r.risk === 'ABSOLUTE CI' ? 'bg-danger' :
                  r.risk === 'EXTREME HAZARD' ? 'bg-warning text-dark' :
                  'bg-secondary'
                }`} style={{ minWidth: 120, fontSize: 10 }}>
                  {r.risk}
                </span>
                <div className="small"><strong>{r.situation}:</strong> {r.detail}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── TAB 1: PATIENTS & BIOMARKERS ── */}
      {tab === 1 && bk && (
        <div>
          {/* Biomarkers */}
          <Section title="Biomarkers (11 key measures)" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr>
                    <th>Biomarker</th><th>Normal Range</th><th>PA Range</th>
                    <th>Significance</th><th>Method</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.biomarkers || []).map((b, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{b.name}</td>
                      <td>{b.normal}</td>
                      <td style={{ color: b.pa_range && b.pa_range.includes('NORMAL') ? ACCENT6 : ACCENT3 }}>
                        {b.pa_range}
                      </td>
                      <td>{b.significance}</td>
                      <td className="text-muted">{b.method}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Key Variants */}
          <Section title="Key Pathogenic Variants in PCCA" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Variant</th><th>cDNA / Type</th><th>Domain</th><th>Severity</th><th>Note</th></tr>
                </thead>
                <tbody>
                  {(bk.key_variants || []).map((v, i) => (
                    <tr key={i}>
                      <td className="fw-semibold font-monospace">{v.variant}</td>
                      <td className="font-monospace small">{v.cdna}</td>
                      <td>{v.domain}</td>
                      <td>
                        <span className={`badge ${v.severity === 'Severe' ? 'bg-danger' : v.severity === 'Moderate' ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                          {v.severity}
                        </span>
                      </td>
                      <td className="text-muted small">{v.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Patient Cohort Sample */}
          <Section title="Patient Cohort Sample (15 of 40)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 11 }}>
                <thead className="table-dark">
                  <tr>
                    <th>ID</th><th>Sex</th><th>Phenotype</th><th>Onset (mo)</th>
                    <th>C3 (µmol/L)</th><th>Methylcitrate</th><th>3-OH-Prop</th>
                    <th>NH₃ (µmol/L)</th><th>Carnitine</th>
                    <th>Seizures</th><th>Cardio</th><th>BG Infarct</th>
                    <th>NBS</th><th>Genotype</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.patient_sample || []).map((p, i) => (
                    <tr key={i}>
                      <td className="font-monospace">{p.id}</td>
                      <td>{p.sex}</td>
                      <td style={{ fontSize: 10 }}>{p.phenotype}</td>
                      <td>{p.onset_age_months}</td>
                      <td style={{ color: p.c3_umol_l > 5 ? ACCENT3 : 'inherit' }}>{p.c3_umol_l}</td>
                      <td style={{ color: p.methylcitrate_umol_mmolCr > 50 ? ACCENT8 : 'inherit' }}>{p.methylcitrate_umol_mmolCr}</td>
                      <td>{p.three_oh_propionate_umol_mmolCr}</td>
                      <td style={{ color: p.ammonia_umol_l > 150 ? ACCENT2 : 'inherit' }}>{p.ammonia_umol_l}</td>
                      <td style={{ color: p.free_carnitine_umol_l < 20 ? ACCENT5 : 'inherit' }}>{p.free_carnitine_umol_l}</td>
                      <td>{p.seizures ? '✓' : '—'}</td>
                      <td style={{ color: p.cardiomyopathy ? ACCENT5 : 'inherit' }}>{p.cardiomyopathy ? '✓ DCM' : '—'}</td>
                      <td style={{ color: p.bg_infarct ? ACCENT5 : 'inherit' }}>{p.bg_infarct ? '✓ BG' : '—'}</td>
                      <td>{p.nbs_detected ? '✓' : '✗'}</td>
                      <td className="font-monospace" style={{ fontSize: 10 }}>{p.genotype}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 2: SEIZURES & TRIGGERS ── */}
      {tab === 2 && bk && (
        <div>
          {/* Seizure Types */}
          <Section title="Seizure Types in Propionic Acidemia" color={ACCENT2}>
            <div className="row g-2">
              {(bk.seizure_types || []).map((s, i) => (
                <div key={i} className="col-12 col-md-6">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body py-2 px-3">
                      <div className="d-flex justify-content-between align-items-center mb-1">
                        <span className="fw-semibold small">{s.type}</span>
                        <span className="badge" style={{ backgroundColor: ACCENT2 }}>{s.pct}%</span>
                      </div>
                      <div className="progress mb-1" style={{ height: 6 }}>
                        <div className="progress-bar" style={{ width: `${s.pct}%`, backgroundColor: ACCENT2 }} />
                      </div>
                      <div className="text-muted" style={{ fontSize: 11 }}>{s.note}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Metabolic Triggers */}
          <Section title="Metabolic Crisis Triggers" color={ACCENT3}>
            {(bk.metabolic_triggers || []).map((t, i) => (
              <div key={i} className="mb-2 d-flex align-items-start">
                <span className="badge me-2 bg-warning text-dark" style={{ minWidth: 40, fontSize: 10 }}>
                  {t.pct}%
                </span>
                <div className="small"><strong>{t.trigger}:</strong> {t.mechanism}</div>
              </div>
            ))}
          </Section>

          {/* High-Risk Drugs */}
          <Section title="High-Risk Drugs / Substances" color={ACCENT2}>
            {(bk.high_risk_drugs || []).map((d, i) => (
              <div key={i} className="d-flex align-items-start mb-2">
                <span className={`badge me-2 ${
                  d.risk === 'ABSOLUTE CI' ? 'bg-danger' :
                  d.risk === 'EXTREME HAZARD' ? 'bg-warning text-dark' :
                  'bg-secondary'
                }`} style={{ minWidth: 120, fontSize: 10 }}>{d.risk}</span>
                <div className="small"><strong>{d.drug}:</strong> {d.mechanism}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── TAB 3: TREATMENTS ── */}
      {tab === 3 && bk && (
        <div>
          <Alert
            variant="info"
            text="ℹ️ PCCA TREATMENT KEY: L-Carnitine (LEVEL A) — conjugates propionyl-CoA to C3 for renal excretion; secondary carnitine depletion is a major PA complication. Liver transplant (LEVEL B) corrects ~75% hepatic PCC but does NOT prevent cardiac complications (extra-hepatic PCC remains deficient). Metronidazole reduces gut bacterial propionate load. Biotin is NOT effective in PA."
          />
          <div className="row g-2">
            {(bk.treatments || []).map((t, i) => (
              <div key={i} className="col-12 col-md-6">
                <div className="card border-0 shadow-sm h-100">
                  <div className="card-body py-2 px-3">
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <span className="fw-semibold small">{t.treatment}</span>
                      <span className={`badge ${
                        t.evidence === 'Level A' ? 'bg-success' :
                        t.evidence === 'Level B' ? 'bg-primary' :
                        t.evidence === 'NOT EFFECTIVE' ? 'bg-secondary' :
                        t.evidence === 'ABSOLUTE CI' ? 'bg-danger' : 'bg-secondary'
                      }`}>{t.evidence}</span>
                    </div>
                    {t.response_pct > 0 && (
                      <div className="mb-1">
                        <div className="progress" style={{ height: 6 }}>
                          <div className="progress-bar bg-success" style={{ width: `${t.response_pct}%` }} />
                        </div>
                        <div className="text-muted" style={{ fontSize: 10 }}>{t.response_pct}% response</div>
                      </div>
                    )}
                    <div className="text-muted" style={{ fontSize: 11 }}>{t.note}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── TAB 4: DEFINITIONS ── */}
      {tab === 4 && df && (
        <div>
          {/* Gene Card */}
          <Section title="Gene Card — PCCA" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <tbody>
                  {Object.entries(df.gene_card || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-semibold text-muted" style={{ width: '30%' }}>{k}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Key Concepts */}
          <Section title="Key Concepts" color={ACCENT}>
            {(df.key_concepts || []).map((c, i) => (
              <div key={i} className="card border-0 shadow-sm mb-2">
                <div className="card-body py-2 px-3">
                  <div className="fw-semibold small mb-1" style={{ color: ACCENT }}>{c.concept}</div>
                  <div className="text-muted" style={{ fontSize: 12 }}>{c.explanation}</div>
                </div>
              </div>
            ))}
          </Section>

          {/* Diagnostic Thresholds */}
          <Section title="Diagnostic Thresholds" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
                </thead>
                <tbody>
                  {(df.diagnostic_thresholds || []).map((d, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{d.parameter}</td>
                      <td style={{ color: ACCENT3 }}>{d.threshold}</td>
                      <td>{d.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Differential Diagnosis */}
          <Section title="Differential Diagnosis" color={ACCENT6}>
            <div className="row g-2">
              {(df.differential_diagnosis || []).map((d, i) => (
                <div key={i} className="col-12 col-md-6">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body py-2 px-3">
                      <div className="fw-semibold small mb-1" style={{ color: ACCENT6 }}>{d.disease}</div>
                      <div className="text-muted" style={{ fontSize: 11 }}>{d.distinguishing}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}
    </div>
  );
}
