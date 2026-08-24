'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4a148c';   // deep purple — MMACHC / cblC combined disorder
const ACCENT2 = '#006064';   // dark cyan — OHCbl + betaine / HHcy treatment
const ACCENT3 = '#b71c1c';   // deep red — MMA + HHcy accumulation / PATHOGNOMONIC
const ACCENT4 = '#1565c0';   // blue — treatment / Level A
const ACCENT5 = '#e65100';   // deep orange — maculopathy / retinal / unique cblC
const ACCENT6 = '#1b5e20';   // dark green — KEY NEGATIVES / isolated MMA distinctions
const ACCENT7 = '#37474f';   // slate — variant data / gene card
const ACCENT8 = '#880e4f';   // deep magenta — cobalamin pathway / branch point mechanism

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

export default function MMACHCPage() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mmachc/overview`).then(r => r.json()),
      fetch(`${API}/api/mmachc/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mmachc/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov)  return <div className="text-center p-5 text-muted">Loading MMACHC Dashboard…</div>;

  const kpis = ov.kpis || {};

  return (
    <div className="container-fluid px-3 py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🧬 MMACHC Epilepsy — Methylmalonic Acidemia + Homocystinuria (cblC — Combined MMA+HHcy)
        </h4>
        <div className="text-muted small">
          MMACHC · 1p34.1 · AR · OMIM *609831 / #277400 · 282 aa · cytoplasmic bifunctional decyanase/reductase ·
          MOST UPSTREAM intracellular Cbl processor · BOTH MMA ↑↑ AND HHcy ↑↑ (COMBINED — KEY DISTINCTION from isolated MMA) ·
          Maculopathy ~80% early-onset (PATHOGNOMONIC cblC) · OHCbl Level A + Betaine Level A MANDATORY ·
          N2O ABSOLUTE CI · n={ov.cohort_n}
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
            text="⚠️ MMACHC (cblC) HALLMARK: COMBINED MMA ↑↑ + Homocysteine ↑↑ (HHcy >30 µmol/L) — this is the PATHOGNOMONIC distinguishing feature from isolated MMA (MMUT/MMAA/MMAB where Hcy is NORMAL). FIRST response to crisis: IV glucose + OHCbl IM + Betaine. N2O (nitrous oxide) ABSOLUTE CI — irreversibly inactivates methionine synthase → acute HHcy crisis: LIFE-THREATENING. Betaine (Level A) is MANDATORY co-treatment with OHCbl. Maculopathy in ~80% early-onset — ophthalmology referral mandatory at diagnosis. Late-onset: psychiatric/schizophrenia-like with c.482G>A — always check tHcy+MMA in psychiatric patients."
          />

          {/* Gene & Disease Overview */}
          <Section title="Gene & Disease Overview" color={ACCENT}>
            <div className="row g-2 mb-2">
              {[
                ['Gene',          ov.gene || 'MMACHC'],
                ['Full Name',     ov.full_name || 'Methylmalonyl-CoA/Homocysteine Cobalamin Processing Chaperone'],
                ['Chromosome',    ov.chromosome || '1p34.1'],
                ['Inheritance',   ov.inheritance || 'AR'],
                ['OMIM Gene',     ov.omim_gene || '*609831'],
                ['OMIM Disease',  ov.omim_disease || '#277400'],
                ['Protein',       ov.protein_size || '282 aa; cytoplasmic; bifunctional decyanase/reductase'],
                ['Prevalence',    ov.prevalence || '~1:50,000–100,000 (most common intracellular Cbl disorder)'],
                ['NBS Primary',   ov.nbs_primary || 'C3 elevated — identical to isolated MMA; MMA+HHcy workup mandatory'],
                ['NBS Secondary', ov.nbs_secondary || 'Urine MMA + tHcy (BOTH elevated — PATHOGNOMONIC cblC); methionine low'],
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
              <strong>Key Positive / Negative:</strong> {ov.key_negative}
            </div>
          </Section>

          {/* KPIs */}
          <Section title={`Cohort KPIs — ${ov.cohort_n} Patients (cblC / MMACHC)`} color={ACCENT3}>
            <div className="row g-2">
              <KPI label="Avg MMA (mmol/mol Cr)" value={kpis.avg_mma_urine?.toLocaleString()} color={ACCENT3} />
              <KPI label="Avg tHcy (µmol/L) ↑" value={kpis.avg_homocysteine_umol_l} color={ACCENT8} />
              <KPI label="Avg Methionine (µmol/L) ↓" value={kpis.avg_methionine_umol_l} color={ACCENT5} />
              <KPI label="Avg C3 (µmol/L)" value={kpis.avg_c3_umol_l} color={ACCENT3} />
              <KPI label="Avg NH₃ (µmol/L)" value={kpis.avg_ammonia_umol_l} color={ACCENT2} />
              <KPI label="Avg Carnitine" value={kpis.avg_free_carnitine} color={ACCENT4} />
              <KPI label="Seizures %" value={kpis.seizure_pct + '%'} color={ACCENT3} />
              <KPI label="Maculopathy %" value={kpis.maculopathy_pct + '%'} color={ACCENT5} />
              <KPI label="Psychiatric %" value={kpis.psychiatric_pct + '%'} color={ACCENT} />
              <KPI label="NBS Detected %" value={kpis.nbs_detected_pct + '%'} color={ACCENT4} />
              <KPI label="OHCbl Response %" value={kpis.ohcbl_response_pct + '%'} color={ACCENT2} />
            </div>
            <div className="alert py-2 mt-2" style={{ backgroundColor: '#f3e5f5', fontSize: 12, borderLeft: `4px solid ${ACCENT8}` }}>
              <strong style={{ color: ACCENT8 }}>KEY cblC SIGNATURE:</strong> Avg tHcy {kpis.avg_homocysteine_umol_l} µmol/L (ELEVATED — PATHOGNOMONIC; NORMAL in MMUT/MMAA/MMAB isolated MMA) +
              Avg Methionine {kpis.avg_methionine_umol_l} µmol/L (LOW — methionine synthase blocked). This MMA+HHcy COMBINATION = cblC hallmark.
              Maculopathy {kpis.maculopathy_pct}% (absent in all isolated MMA — unique cblC retinal sign).
            </div>
          </Section>

          {/* Phenotype Distribution */}
          <Section title="Phenotype Distribution (cblC Types)" color={ACCENT3}>
            {(ov.phenotype_distribution || []).map((p, i) => (
              <PctBar key={i} label={p.phenotype} pct={p.pct}
                color={i === 0 ? ACCENT3 : i === 1 ? ACCENT5 : i === 2 ? ACCENT : ACCENT6} />
            ))}
          </Section>

          {/* MMACHC Pathway */}
          <Section title="MMACHC — Cobalamin Branch Point (5 Steps; MMACHC Step 1 UPSTREAM Block → BOTH MMA+HHcy Arms Fail)" color={ACCENT8}>
            <div className="row g-2">
              {(ov.mmachc_pathway || []).map((s, i) => (
                <div key={i} className="col-12 col-md-6">
                  <div className="card border-0 shadow-sm h-100" style={{ borderLeft: i === 0 ? `3px solid #b71c1c` : `3px solid ${ACCENT8}` }}>
                    <div className="card-body py-2 px-3">
                      <div className="fw-bold small" style={{ color: i === 0 ? '#b71c1c' : i >= 3 ? ACCENT3 : ACCENT8 }}>{s.step}</div>
                      <div className="font-monospace" style={{ fontSize: 11, color: ACCENT }}>{s.reaction}</div>
                      <div className="text-muted" style={{ fontSize: 11 }}>Enzyme: {s.enzyme}</div>
                      {s.cofactor && <div className="text-muted" style={{ fontSize: 11 }}>Cofactor: {s.cofactor}</div>}
                      <div className="mt-1 small" style={{ color: i === 0 ? '#b71c1c' : i >= 3 ? ACCENT3 : ACCENT6 }}>
                        <strong>cblC LOF:</strong> {s.consequence_lof}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* MMACHC vs Isolated MMA Comparison */}
          {ov.mmachc_vs_isolated_mma && (
            <Section title={ov.mmachc_vs_isolated_mma.title} color={ACCENT6}>
              <div className="text-muted small mb-2">{ov.mmachc_vs_isolated_mma.note}</div>
              <div className="table-responsive">
                <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>Feature</th>
                      <th style={{ color: '#ce93d8' }}>MMACHC — cblC (this disease)</th>
                      <th style={{ color: '#80cbc4' }}>Isolated MMA (MMUT / MMAA / MMAB)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.mmachc_vs_isolated_mma.comparison || []).map((row, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{row.feature}</td>
                        <td style={{
                          color: row.MMACHC_cblC?.includes('ELEVATED') || row.MMACHC_cblC?.includes('PATHOGNOMONIC') ? '#b71c1c' :
                                 row.MMACHC_cblC?.includes('ABSENT') ? '#b71c1c' :
                                 row.MMACHC_cblC?.includes('LOW') ? ACCENT5 :
                                 row.MMACHC_cblC?.includes('~75') || row.MMACHC_cblC?.includes('~80') ? ACCENT2 :
                                 row.MMACHC_cblC?.includes('ABSOLUTE CI') ? '#b71c1c' :
                                 'inherit'
                        }}>{row.MMACHC_cblC}</td>
                        <td style={{
                          color: row.Isolated_MMA?.includes('NORMAL') ? ACCENT6 :
                                 row.Isolated_MMA?.includes('NOT') ? ACCENT6 :
                                 'inherit'
                        }}>{row.Isolated_MMA}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          )}

          {/* High-Risk Situations */}
          <Section title="High-Risk Situations" color="#b71c1c">
            {(ov.high_risk_situations || []).map((r, i) => (
              <div key={i} className="d-flex align-items-start mb-2">
                <span className={`badge me-2 ${
                  r.risk === 'ABSOLUTE CI' ? 'bg-danger' :
                  r.risk === 'HIGH RISK' ? 'bg-danger' :
                  r.risk === 'HIGH RISK / AVOID' ? 'bg-warning text-dark' :
                  r.risk === 'CAUTION' ? 'bg-secondary' :
                  'bg-secondary'
                }`} style={{ minWidth: 140, fontSize: 10 }}>
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
          <Section title="Biomarkers (12 key measures — MMACHC / cblC Combined MMA+HHcy)" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr>
                    <th>Biomarker</th><th>Normal Range</th><th>cblC Range</th>
                    <th>Significance</th><th>Method</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.biomarkers || []).map((b, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{b.name}</td>
                      <td>{b.normal}</td>
                      <td style={{
                        color: b.mmachc_range?.includes('NORMAL') ? ACCENT6 :
                               b.mmachc_range?.includes('PATHOGNOMONIC') ? '#b71c1c' :
                               b.mmachc_range?.includes('ABSENT') ? '#b71c1c' :
                               b.mmachc_range?.includes('LOW') ? ACCENT5 :
                               b.mmachc_range?.includes('ELEVATED') ? '#b71c1c' :
                               ACCENT3
                      }}>
                        {b.mmachc_range}
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
          <Section title="Key Pathogenic Variants in MMACHC (cblC — Cytoplasmic Cobalamin Processor)" color={ACCENT7}>
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
                        <span className={`badge ${
                          v.severity?.includes('null') || v.severity?.includes('Severe') ? 'bg-danger' :
                          v.severity?.includes('Intermediate') ? 'bg-warning text-dark' :
                          v.severity?.includes('Mild') ? 'bg-secondary' :
                          v.severity?.includes('partial') ? 'bg-info text-dark' :
                          'bg-secondary'
                        }`}>
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
          <Section title="Patient Cohort Sample (15 of 40 — MMACHC cblC)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 11 }}>
                <thead className="table-dark">
                  <tr>
                    <th>ID</th><th>Sex</th><th>Phenotype</th><th>Diag (mo)</th>
                    <th>MMA Urine (mmol/mol Cr)</th><th>tHcy (µmol/L) ↑</th>
                    <th>Methionine (µmol/L) ↓</th><th>C3 (µmol/L)</th>
                    <th>NH₃ (µmol/L)</th><th>Carnitine</th>
                    <th>Seizures</th><th>Maculopathy</th><th>Psychiatric</th>
                    <th>NBS</th><th>OHCbl Resp</th><th>AED</th><th>Genotype</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.patient_sample || []).map((p, i) => (
                    <tr key={i}>
                      <td className="font-monospace">{p.patient_id}</td>
                      <td>{p.sex}</td>
                      <td style={{ fontSize: 10 }}>{p.phenotype}</td>
                      <td>{p.age_at_diagnosis_months}</td>
                      <td style={{ color: p.mma_urine_mmol_molCr > 500 ? ACCENT3 : 'inherit', fontWeight: 'bold' }}>{p.mma_urine_mmol_molCr?.toLocaleString()}</td>
                      <td style={{ color: p.total_homocysteine_umol_l > 50 ? '#b71c1c' : ACCENT5, fontWeight: 'bold' }}>{p.total_homocysteine_umol_l}</td>
                      <td style={{ color: p.methionine_umol_l < 15 ? ACCENT5 : 'inherit' }}>{p.methionine_umol_l}</td>
                      <td style={{ color: p.c3_umol_l > 6 ? ACCENT3 : 'inherit' }}>{p.c3_umol_l}</td>
                      <td style={{ color: p.ammonia_umol_l > 200 ? '#b71c1c' : 'inherit' }}>{p.ammonia_umol_l}</td>
                      <td style={{ color: p.free_carnitine_umol_l < 20 ? ACCENT : 'inherit' }}>{p.free_carnitine_umol_l}</td>
                      <td>{p.seizures ? '✓' : '—'}</td>
                      <td style={{ color: p.maculopathy ? ACCENT5 : 'inherit', fontWeight: p.maculopathy ? 'bold' : 'normal' }}>{p.maculopathy ? '✓ MAC' : '—'}</td>
                      <td style={{ color: p.psychiatric_features ? ACCENT : 'inherit' }}>{p.psychiatric_features ? '✓ PSY' : '—'}</td>
                      <td>{p.nbs_detected ? '✓' : '✗'}</td>
                      <td style={{ color: p.ohcbl_response ? ACCENT2 : 'inherit', fontWeight: p.ohcbl_response ? 'bold' : 'normal' }}>{p.ohcbl_response ? '✓ Resp' : '—'}</td>
                      <td className="text-muted" style={{ fontSize: 10 }}>{p.aed}</td>
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
          <Section title="Seizure Types in MMACHC (cblC — Combined MMA+HHcy Epilepsy)" color="#b71c1c">
            <div className="row g-2">
              {(bk.seizure_types || []).map((s, i) => (
                <div key={i} className="col-12 col-md-6">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body py-2 px-3">
                      <div className="d-flex justify-content-between align-items-center mb-1">
                        <span className="fw-semibold small">{s.type}</span>
                        <span className="badge" style={{ backgroundColor: '#b71c1c' }}>{s.pct}%</span>
                      </div>
                      <div className="progress mb-1" style={{ height: 6 }}>
                        <div className="progress-bar" style={{ width: `${s.pct}%`, backgroundColor: '#b71c1c' }} />
                      </div>
                      <div className="text-muted" style={{ fontSize: 11 }}>{s.note}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Metabolic Triggers */}
          <Section title="Metabolic Crisis Triggers in MMACHC (cblC)" color={ACCENT3}>
            {(bk.metabolic_triggers || []).map((t, i) => (
              <div key={i} className="mb-2 d-flex align-items-start">
                <span className={`badge me-2 ${t.pct >= 100 ? 'bg-danger' : 'bg-warning text-dark'}`} style={{ minWidth: 60, fontSize: 10 }}>
                  {t.pct >= 100 ? 'ABS CI' : t.pct + '%'}
                </span>
                <div className="small"><strong>{t.trigger}:</strong> {t.mechanism}</div>
              </div>
            ))}
          </Section>

          {/* High-Risk Drugs */}
          <Section title="High-Risk Drugs / Substances in MMACHC (cblC)" color="#b71c1c">
            {(bk.high_risk_drugs || []).map((d, i) => (
              <div key={i} className="d-flex align-items-start mb-2">
                <span className={`badge me-2 ${
                  d.risk === 'ABSOLUTE CI' ? 'bg-danger' :
                  d.risk === 'HIGH RISK / AVOID' ? 'bg-warning text-dark' :
                  d.risk === 'HIGH RISK' ? 'bg-danger' :
                  d.risk === 'CAUTION' ? 'bg-secondary' :
                  'bg-secondary'
                }`} style={{ minWidth: 150, fontSize: 10 }}>{d.risk}</span>
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
            text="ℹ️ MMACHC (cblC) TREATMENT CORNERSTONE: Two Level A co-treatments MANDATORY — (1) Hydroxocobalamin (OHCbl) 1–2 mg/day IM: ~75% HHcy response + ~55–60% MMA response; bypasses MMACHC upstream block. (2) Betaine (trimethylglycine) 100–200 mg/kg/day orally: cobalamin-INDEPENDENT remethylation via BHMT; reduces HHcy even if OHCbl response incomplete. N2O (nitrous oxide) ABSOLUTE CI — irreversibly inactivates methionine synthase; NEVER use N2O in cblC patients. VPA HIGH RISK / AVOID — use LEV. Maculopathy monitoring mandatory."
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
                        t.evidence === 'Level C/Supportive' || t.evidence === 'Level C' ? 'bg-info text-dark' :
                        t.evidence === 'HIGH RISK / AVOID' ? 'bg-warning text-dark' :
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
          <Section title="Gene Card — MMACHC (cblC — Cytoplasmic Cobalamin Branch-Point Processor)" color={ACCENT7}>
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
          <Section title="Key Concepts — MMACHC / cblC Combined MMA+HHcy (7 Core Concepts)" color={ACCENT}>
            {(df.key_concepts || []).map((c, i) => (
              <div key={i} className="card border-0 shadow-sm mb-2">
                <div className="card-body py-2 px-3">
                  <div className="fw-semibold small mb-1" style={{
                    color: i === 0 ? ACCENT8 : i === 1 ? '#b71c1c' : i === 3 ? ACCENT5 :
                           i === 4 ? ACCENT : i === 6 ? '#b71c1c' : ACCENT
                  }}>{c.concept}</div>
                  <div className="text-muted" style={{ fontSize: 12 }}>{c.explanation}</div>
                </div>
              </div>
            ))}
          </Section>

          {/* Diagnostic Thresholds */}
          <Section title="Diagnostic Thresholds — MMACHC (cblC Combined MMA+HHcy)" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
                </thead>
                <tbody>
                  {(df.diagnostic_thresholds || []).map((d, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{d.parameter}</td>
                      <td style={{
                        color: d.parameter?.includes('Homocysteine') || d.parameter?.includes('MMA') ? ACCENT3 :
                               d.parameter?.includes('OHCbl') ? ACCENT2 :
                               ACCENT3
                      }}>{d.threshold}</td>
                      <td>{d.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Differential Diagnosis */}
          <Section title="Differential Diagnosis — vs Isolated MMA, CBS Homocystinuria, PA, DLD, cblD" color={ACCENT6}>
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
