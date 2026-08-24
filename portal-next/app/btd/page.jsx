'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#00838f';   // deep cyan — biotinidase / recycling enzyme / biotin liberation
const ACCENT2 = '#880e4f';   // dark magenta — SNHL / optic atrophy / delayed diagnosis hazard
const ACCENT3 = '#e65100';   // deep orange — metabolic acidosis / organic acid accumulation
const ACCENT4 = '#1565c0';   // deep blue — biotin treatment / dramatic response
const ACCENT5 = '#ad1457';   // deep pink — biotinidase DEFICIENT (KEY diagnostic finding in BTD)
const ACCENT6 = '#1b5e20';   // dark green — KEY NEGATIVES / HLCS normal / differential
const ACCENT7 = '#37474f';   // dark slate — variant data / gene card
const ACCENT8 = '#4e342e';   // dark brown — biocytin accumulation / inhibition mechanism

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

export default function BTDPage() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/btd/overview`).then(r => r.json()),
      fetch(`${API}/api/btd/breakdown`).then(r => r.json()),
      fetch(`${API}/api/btd/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov)  return <div className="text-center p-5 text-muted">Loading BTD Dashboard…</div>;

  const kpis = ov.kpis || {};

  return (
    <div className="container-fluid px-3 py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🧬 BTD Epilepsy — Biotinidase Deficiency (Multiple Carboxylase Deficiency — Late-onset/Infantile)
        </h4>
        <div className="text-muted small">
          BTD · 3p25.1 · AR · OMIM *609019 / #253260 · 543 aa · 60-kDa glycoprotein ·
          Biotin RECYCLING enzyme (biocytin → free biotin) · Biotinidase DEFICIENT — KEY diagnostic ·
          Biotin LOW (actual depletion) · NBS primary: BTD enzyme assay · SNHL 75% if late-diagnosed ·
          Biotin 5–10 mg/day profound / 2–5 mg/day partial · n={ov.cohort_n}
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
            text="⚠️ BIOTINIDASE DEFICIENT — DIAGNOSTIC KEY: Serum BTD activity <10% = PROFOUND; 10-30% = PARTIAL. Biotin plasma LEVEL LOW (depletion) — distinguishes from HLCS where biotin NORMAL. SNHL (sensorineural hearing loss) 75% if diagnosis delayed >3-6 months — may be IRREVERSIBLE. Start biotin SAME DAY as NBS positive. Raw egg white / avidin ABSOLUTE CI — blocks all biotin absorption."
          />

          {/* Gene & Disease Overview */}
          <Section title="Gene & Disease Overview" color={ACCENT}>
            <div className="row g-2 mb-2">
              {[
                ['Gene',         ov.gene || 'BTD'],
                ['Full Name',    ov.full_name || 'Biotinidase'],
                ['Chromosome',   ov.chromosome || '3p25.1'],
                ['Inheritance',  ov.inheritance || 'AR'],
                ['OMIM Gene',    ov.omim_gene || '*609019'],
                ['OMIM Disease', ov.omim_disease || '#253260'],
                ['Protein',      ov.protein_size || '543 aa, 60-kDa glycoprotein'],
                ['Prevalence',   ov.prevalence || '1:61,000 combined'],
                ['NBS Primary',  ov.nbs_primary || 'BTD enzyme assay (fluorometric)'],
                ['NBS Secondary',ov.nbs_secondary || 'C5-OH + C3 acylcarnitines'],
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
              <KPI label="BTD Activity (avg %)" value={kpis.avg_btd_activity_pct + '%'} color={ACCENT5} />
              <KPI label="C5-OH NBS (avg µmol/L)" value={kpis.avg_c5oh_umol_l} color={ACCENT3} />
              <KPI label="Lactate (avg mmol/L)" value={kpis.avg_lactate_mmol_l} color={ACCENT3} />
              <KPI label="Seizures %" value={kpis.seizure_pct + '%'} color={ACCENT2} />
              <KPI label="SNHL %" value={kpis.snhl_pct + '%'} color={ACCENT2} />
              <KPI label="Optic Atrophy %" value={kpis.optic_atrophy_pct + '%'} color={ACCENT2} />
              <KPI label="Alopecia %" value={kpis.alopecia_pct + '%'} color={ACCENT8} />
              <KPI label="Skin Rash %" value={kpis.skin_rash_pct + '%'} color={ACCENT8} />
              <KPI label="Hypotonia %" value={kpis.hypotonia_pct + '%'} color={ACCENT7} />
              <KPI label="NBS Detected %" value={kpis.nbs_detected_pct + '%'} color={ACCENT4} />
              <KPI label="Biotin Response %" value={kpis.biotin_responsive_pct + '%'} color={ACCENT4} />
              <KPI label="Candida Infxn %" value={kpis.candida_infections_pct + '%'} color={ACCENT7} />
            </div>
          </Section>

          {/* Phenotype Distribution */}
          <Section title="Phenotype Distribution" color={ACCENT2}>
            {(ov.phenotype_distribution || []).map((p, i) => (
              <PctBar key={i} label={p.phenotype} pct={p.pct}
                color={i === 0 ? ACCENT2 : i === 1 ? ACCENT5 : i === 2 ? ACCENT3 : ACCENT4} />
            ))}
          </Section>

          {/* Four Carboxylases */}
          <Section title="Four Biotin-Dependent Carboxylases (all fail in BTD via biotin depletion)" color={ACCENT8}>
            <div className="row g-2">
              {(ov.four_carboxylases || []).map((c, i) => (
                <div key={i} className="col-12 col-md-6">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body py-2 px-3">
                      <div className="fw-bold small" style={{ color: ACCENT8 }}>{c.enzyme}</div>
                      <div className="text-muted" style={{ fontSize: 11 }}>Gene: {c.gene} · Chr: {c.chromosome}</div>
                      <div className="small mt-1"><strong>Role:</strong> {c.role}</div>
                      <div className="small mt-1"><strong>BTD block:</strong> {c.btd_block_consequence}</div>
                      <div className="small mt-1 text-muted"><strong>Biomarker:</strong> {c.biomarker}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* BTD vs HLCS */}
          {ov.btd_vs_hlcs && (
            <Section title={ov.btd_vs_hlcs.title} color={ACCENT6}>
              <div className="text-muted small mb-2">{ov.btd_vs_hlcs.note}</div>
              <div className="table-responsive">
                <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>Feature</th>
                      <th style={{ color: '#80cbc4' }}>BTD (this disease)</th>
                      <th style={{ color: '#ffcc80' }}>HLCS (comparator)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.btd_vs_hlcs.comparison || []).map((row, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{row.feature}</td>
                        <td>{row.BTD}</td>
                        <td>{row.HLCS}</td>
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
                <span className={`badge me-2 ${r.risk === 'ABSOLUTE CI' ? 'bg-danger' : r.risk === 'EXTREME HAZARD' ? 'bg-warning text-dark' : r.risk === 'HIGH HAZARD' ? 'bg-orange text-dark' : 'bg-secondary'}`}
                  style={{ minWidth: 110, fontSize: 10,
                    backgroundColor: r.risk === 'HIGH HAZARD' ? '#e65100' : undefined,
                    color: r.risk === 'HIGH HAZARD' ? '#fff' : undefined }}>
                  {r.risk}
                </span>
                <div className="small"><strong>{r.situation}:</strong> {r.detail}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── TAB 1: PATIENTS & PHENOTYPE ── */}
      {tab === 1 && bk && (
        <div>
          {/* Biomarkers */}
          <Section title="Biomarkers" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr>
                    <th>Biomarker</th><th>Normal Range</th><th>BTD Range</th>
                    <th>Significance</th><th>Method</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.biomarkers || []).map((b, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{b.name}</td>
                      <td>{b.normal}</td>
                      <td style={{ color: ACCENT3 }}>{b.btd_range}</td>
                      <td>{b.significance}</td>
                      <td className="text-muted">{b.method}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Key Variants */}
          <Section title="Key Pathogenic Variants in BTD" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Variant</th><th>cDNA</th><th>Domain</th><th>Severity</th><th>Note</th></tr>
                </thead>
                <tbody>
                  {(bk.key_variants || []).map((v, i) => (
                    <tr key={i}>
                      <td className="fw-semibold font-monospace">{v.variant}</td>
                      <td className="font-monospace small">{v.cdna}</td>
                      <td>{v.domain}</td>
                      <td>
                        <span className={`badge ${v.severity === 'Profound' ? 'bg-danger' : 'bg-warning text-dark'}`}>
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
                    <th>BTD Act%</th><th>C5-OH</th><th>C3</th><th>3-OH-IsoVal</th>
                    <th>Lactate</th><th>Biotin (pmol/L)</th>
                    <th>Alopecia</th><th>SNHL</th><th>Seizures</th>
                    <th>NBS</th><th>Resp</th><th>Variant</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.patient_sample || []).map((p, i) => (
                    <tr key={i}>
                      <td className="font-monospace">{p.id}</td>
                      <td>{p.sex}</td>
                      <td style={{ fontSize: 10 }}>{p.phenotype}</td>
                      <td>{p.onset_age_months}</td>
                      <td style={{ color: p.btd_activity_pct < 10 ? ACCENT2 : ACCENT3 }}>
                        {p.btd_activity_pct}%
                      </td>
                      <td>{p.c5oh_umol_l}</td>
                      <td>{p.c3_umol_l}</td>
                      <td>{p.three_oh_isovalerate_umol_mmolCr}</td>
                      <td>{p.lactate_mmol_l}</td>
                      <td style={{ color: p.biotin_plasma_pmol_l < 450 ? ACCENT2 : 'inherit' }}>
                        {p.biotin_plasma_pmol_l}
                      </td>
                      <td>{p.alopecia ? '✓' : '—'}</td>
                      <td style={{ color: p.hearing_loss_snhl ? ACCENT2 : 'inherit' }}>
                        {p.hearing_loss_snhl ? '✓ SNHL' : '—'}
                      </td>
                      <td>{p.seizures ? '✓' : '—'}</td>
                      <td>{p.nbs_detected ? '✓' : '✗'}</td>
                      <td>{p.biotin_responsive ? '✓' : '✗'}</td>
                      <td className="font-monospace" style={{ fontSize: 10 }}>{p.variant_genotype}</td>
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
          <Section title="Seizure Types in BTD" color={ACCENT2}>
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
          <Section title="High-Risk Drugs / Substances" color={ACCENT5}>
            {(bk.high_risk_drugs || []).map((d, i) => (
              <div key={i} className="d-flex align-items-start mb-2">
                <span className={`badge me-2 ${d.risk === 'ABSOLUTE CI' ? 'bg-danger' : 'bg-warning text-dark'}`}
                  style={{ minWidth: 90, fontSize: 10 }}>{d.risk}</span>
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
            variant="success"
            text="✅ BIOTIN IS DRAMATICALLY EFFECTIVE: 5-10 mg/day (profound) / 2-5 mg/day (partial). Lower dose than HLCS (10-40 mg) because HLCS enzyme is INTACT in BTD. Start SAME DAY as diagnosis — do not wait for DNA confirmation. Seizures resolve 24-72h. Hearing loss MAY NOT REVERSE if delayed. Lifelong treatment required."
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
                        t.evidence === 'CAUTION' ? 'bg-warning text-dark' :
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
          <Section title="Gene Card — BTD" color={ACCENT7}>
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
