'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4a148c';   // deep purple — X-linked / HCFC1 transcriptional master
const ACCENT2 = '#880e4f';   // deep pink — HHcy arm / betaine critical
const ACCENT3 = '#b71c1c';   // deep red — MMA accumulation / severe encephalopathy
const ACCENT4 = '#1565c0';   // blue — treatment / Level A
const ACCENT5 = '#37474f';   // slate — KEY NEGATIVES / cblX vs cblC distinction
const ACCENT6 = '#1b5e20';   // dark green — THAP11 axis / transcriptional pathway
const ACCENT7 = '#e65100';   // deep orange — Ohtahara / neonatal severe
const ACCENT8 = '#006064';   // teal — X-linked / hemizygous males

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

function Section({ title, color = ACCENT, children }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-3 pb-1" style={{ color, borderBottom: `2px solid ${color}` }}>{title}</h6>
      {children}
    </div>
  );
}

export default function HCFC1Page() {
  const [tab, setTab]         = useState('Overview');
  const [overview, setOv]     = useState(null);
  const [breakdown, setBd]    = useState(null);
  const [definitions, setDf]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/hcfc1/overview`).then(r => r.json()),
      fetch(`${API}/api/hcfc1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hcfc1/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOv(ov); setBd(bd); setDf(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-4 text-center text-muted">Loading HCFC1 (cblX) dashboard…</div>;
  if (error)   return <div className="p-4 text-danger">Error: {error}</div>;
  if (!overview) return null;

  const ov = overview;
  const bd = breakdown || {};
  const df = definitions || {};

  return (
    <div className="container-fluid py-3 px-3" style={{ maxWidth: 1100 }}>

      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT2} 100%)`, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">&#x1f9ec; HCFC1 (cblX) Epilepsy Dashboard</h4>
        <div style={{ fontSize: 13, opacity: 0.9 }}>
          Methylmalonic Aciduria &amp; Homocystinuria — cblX type · X-Linked Transcriptional Regulator of MMACHC ·
          Xq28 · X-LINKED · OMIM {ov.omim_gene} / {ov.omim_disease}
        </div>
        <div className="mt-1" style={{ fontSize: 12, opacity: 0.8 }}>
          {ov.protein_size} · {ov.prevalence}
        </div>
      </div>

      {/* X-linked critical alert */}
      <div className="alert alert-danger py-2 mb-3" style={{ fontSize: 13, border: `2px solid ${ACCENT3}` }}>
        <strong>&#x274c; X-LINKED — ONLY X-linked intracellular cobalamin disorder</strong> ·
        All other cobalamin disorders (cblA–cblG, cblJ) are autosomal recessive.
        Males hemizygous → severely affected. Carrier females typically unaffected. ·
        <strong> N2O ABSOLUTE CI</strong> · <strong>VPA HIGH RISK (LEV first-line)</strong>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t}>
            <button className={`nav-link${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && (
        <div>
          <Section title="Cohort KPIs" color={ACCENT}>
            <div className="row g-2">
              <KPI label="Avg MMA Urine (mmol/mol Cr)" value={ov.kpis?.avg_mma_urine} color={ACCENT3} />
              <KPI label="Avg tHcy (µmol/L)" value={ov.kpis?.avg_homocysteine_umol_l} color={ACCENT2} />
              <KPI label="Avg Methionine (µmol/L)" value={ov.kpis?.avg_methionine_umol_l} color={ACCENT} />
              <KPI label="Avg C3 (µmol/L)" value={ov.kpis?.avg_c3_umol_l} color={ACCENT4} />
              <KPI label="Avg NH3 (µmol/L)" value={ov.kpis?.avg_ammonia_umol_l} color={ACCENT7} />
              <KPI label="% Seizures" value={`${ov.kpis?.pct_seizures}%`} color={ACCENT3} />
              <KPI label="% OHCbl Response" value={`${ov.kpis?.pct_ohcbl_response}%`} color={ACCENT4} />
              <KPI label="% NBS Detected" value={`${ov.kpis?.pct_nbs_detected}%`} color={ACCENT6} />
              <KPI label="% Male (X-linked)" value={`${ov.kpis?.pct_male}%`} color={ACCENT8} />
              <KPI label="% Maculopathy" value="0% (ABSENT)" color={ACCENT5} />
              <KPI label="% MMACHC Protein Absent" value="100%" color={ACCENT3} />
              <KPI label="Cohort N" value={ov.cohort_n} color={ACCENT} />
            </div>
          </Section>

          <div className="row">
            <div className="col-md-6">
              <Section title="Phenotype Distribution" color={ACCENT7}>
                {ov.phenotype_distribution && Object.entries(ov.phenotype_distribution).map(([k, v]) => (
                  <PctBar key={k} label={v.n ? `${k.replace(/_/g,' ')} (n=${v.n})` : k.replace(/_/g,' ')} pct={v.pct} color={k.includes('ohtahara') ? ACCENT7 : k.includes('west') ? ACCENT3 : ACCENT6} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Gene Card" color={ACCENT}>
                <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>{ov.gene} — Host Cell Factor C1</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td style={{ color: ACCENT }}><strong>{ov.inheritance}</strong></td></tr>
                    <tr><td className="fw-bold">Chromosome</td><td>{ov.chromosome}</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>{ov.omim_gene}</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>{ov.omim_disease}</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>{ov.prevalence}</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
          </div>

          <Section title="Function & Mechanism" color={ACCENT6}>
            <div className="row">
              <div className="col-md-6">
                <div className="card border-0 bg-light p-3 mb-3">
                  <h6 className="fw-bold" style={{ color: ACCENT6 }}>HCFC1 Function</h6>
                  <p style={{ fontSize: 13 }}>{ov.function}</p>
                </div>
              </div>
              <div className="col-md-6">
                <div className="card border-0 bg-light p-3 mb-3">
                  <h6 className="fw-bold" style={{ color: ACCENT3 }}>Disease Mechanism (cblX)</h6>
                  <p style={{ fontSize: 13 }}>{ov.mechanism}</p>
                </div>
              </div>
            </div>
          </Section>

          <Section title="NBS &amp; Biomarker Overview" color={ACCENT4}>
            <div className="card border-0 bg-light p-3 mb-2">
              <div className="small mb-1"><strong>NBS Primary:</strong> {ov.nbs_primary}</div>
              <div className="small"><strong>NBS Secondary &amp; Diagnostic:</strong> {ov.nbs_secondary}</div>
            </div>
          </Section>

          <Section title="Key Negatives / Distinguishing Features" color={ACCENT5}>
            <div className="alert alert-secondary py-2" style={{ fontSize: 13 }}>
              {ov.key_negative}
            </div>
          </Section>

          <Section title="Cobalamin Pathway — HCFC1 Position" color={ACCENT8}>
            <div className="card border-0 p-3" style={{ background: '#f3e5f5', fontSize: 12 }}>
              <div className="mb-1"><strong style={{ color: ACCENT8 }}>Step 0:</strong> Dietary cobalamin + TC2 → CD320 receptor → endocytosis → <em>lysosome</em></div>
              <div className="mb-1"><strong style={{ color: ACCENT8 }}>Step 1:</strong> LMBRD1 (cblF) pore + ABCD4 (cblJ) ATPase motor → exports cobalamin from lysosome → cytoplasm</div>
              <div className="mb-1 ps-3" style={{ borderLeft: `3px solid ${ACCENT6}` }}>
                <strong style={{ color: ACCENT6 }}>HCFC1-THAP11 axis:</strong> HCFC1 recruits THAP11 → activates <em>MMACHC promoter transcription</em>
                <br />← <strong style={{ color: ACCENT }}>cblX BLOCK</strong>: HCFC1 LOF → MMACHC not transcribed → MMACHC protein ABSENT
              </div>
              <div className="mb-1"><strong style={{ color: ACCENT8 }}>Step 2:</strong> MMACHC (cblC) converts cobalamin → cob(I)alamin <em>[ABSENT in cblX — no MMACHC]</em></div>
              <div className="mb-1"><strong style={{ color: ACCENT8 }}>Step 3:</strong> MMADHC distributes → MMAB (AdoCbl arm) + MTR (MeCbl arm)</div>
              <div><strong style={{ color: ACCENT8 }}>Step 4:</strong> AdoCbl → MMUT (MMA catabolism) · MeCbl → MTR (Hcy→Met remethylation)</div>
            </div>
          </Section>
        </div>
      )}

      {/* ── PATIENTS & BIOMARKERS ── */}
      {tab === 'Patients & Biomarkers' && (
        <div>
          <Alert text="All patients: MeCbl fibroblasts ABSENT · AdoCbl fibroblasts ABSENT · MMACHC protein ABSENT · X-linked hemizygous (males)" variant="info" />
          <Alert text="KEY NEGATIVE: Maculopathy 0% in cblX (KEY NEGATIVE vs cblC ~80%) · NO vacuolated lymphocytes (unlike cblF/LMBRD1)" variant="warning" />

          <Section title="Biomarker Ranges by Phenotype" color={ACCENT3}>
            <div className="row">
              <div className="col-md-4">
                <div className="card border-danger mb-3">
                  <div className="card-header text-white fw-bold" style={{ background: ACCENT7 }}>Ohtahara Neonatal Severe (~30%)</div>
                  <div className="card-body" style={{ fontSize: 12 }}>
                    <div>MMA urine: <strong>500–2,500 mmol/mol Cr</strong></div>
                    <div>tHcy: <strong>80–300 µmol/L</strong></div>
                    <div>Methionine: <strong>6–12 µmol/L</strong> (very low)</div>
                    <div>C3: <strong>8–25 µmol/L</strong></div>
                    <div>NH3: <strong>100–600 µmol/L</strong></div>
                    <div>Variants: Arg986Cys, IVS3-1G&gt;A splice null, Gly130Asp</div>
                    <div>Seizures: <strong>~97%</strong> (burst-suppression → IS)</div>
                  </div>
                </div>
              </div>
              <div className="col-md-4">
                <div className="card border-warning mb-3">
                  <div className="card-header text-white fw-bold" style={{ background: ACCENT3 }}>West Syndrome Infantile Classic (~55%)</div>
                  <div className="card-body" style={{ fontSize: 12 }}>
                    <div>MMA urine: <strong>200–1,200 mmol/mol Cr</strong></div>
                    <div>tHcy: <strong>40–180 µmol/L</strong></div>
                    <div>Methionine: <strong>8–16 µmol/L</strong></div>
                    <div>C3: <strong>4–16 µmol/L</strong></div>
                    <div>NH3: <strong>40–180 µmol/L</strong></div>
                    <div>Variants: Pro190Leu (most common), Ala115Val</div>
                    <div>Seizures: <strong>~90%</strong> (infantile spasms)</div>
                  </div>
                </div>
              </div>
              <div className="col-md-4">
                <div className="card border-success mb-3">
                  <div className="card-header text-white fw-bold" style={{ background: ACCENT6 }}>Childhood Attenuated (~15%)</div>
                  <div className="card-body" style={{ fontSize: 12 }}>
                    <div>MMA urine: <strong>80–400 mmol/mol Cr</strong></div>
                    <div>tHcy: <strong>20–80 µmol/L</strong></div>
                    <div>Methionine: <strong>12–22 µmol/L</strong></div>
                    <div>C3: <strong>3–9 µmol/L</strong></div>
                    <div>NH3: <strong>20–70 µmol/L</strong></div>
                    <div>Variants: Arg583Gln, Ala603Thr (Basic domain)</div>
                    <div>Seizures: <strong>~62%</strong></div>
                  </div>
                </div>
              </div>
            </div>
          </Section>

          <Section title="Key Variants in HCFC1" color={ACCENT}>
            <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
              <thead className="table-dark">
                <tr><th>Variant</th><th>Domain</th><th>Severity</th><th>Notes</th></tr>
              </thead>
              <tbody>
                <tr><td>p.Pro190Leu</td><td>β-propeller (THAP11 binding)</td><td>Severe</td><td>Most commonly reported; disrupts THAP11 binding → no MMACHC activation; infantile spasms</td></tr>
                <tr><td>p.Arg986Cys</td><td>HCF-Pro repeat</td><td>Severe neonatal</td><td>Disrupts cleavage; Ohtahara syndrome; HCF-Pro repeat essential for N/C fragment association</td></tr>
                <tr><td>p.Ala115Val</td><td>β-propeller</td><td>Moderate-severe</td><td>Partial THAP11 recruitment; residual MMACHC ~15–20%; infantile spasms</td></tr>
                <tr><td>p.Gly130Asp</td><td>β-propeller</td><td>Severe</td><td>Neonatal encephalopathy; THAP11-binding interface disrupted</td></tr>
                <tr><td>c.340-1G&gt;A</td><td>IVS3-1 splice (null)</td><td>Severe</td><td>Canonical splice acceptor null; Ohtahara + West syndrome transition; no residual HCFC1</td></tr>
                <tr><td>p.Arg583Gln</td><td>Basic domain</td><td>Moderate</td><td>Basic domain; partial THAP11 recruitment retained; childhood onset; better prognosis</td></tr>
                <tr><td>p.Ala603Thr</td><td>Basic domain</td><td>Attenuated</td><td>Basic domain; adolescent onset (rare); mildest reported cblX allele</td></tr>
              </tbody>
            </table>
          </Section>

          {bd.patient_sample && (
            <Section title="Patient Sample (6 cases)" color={ACCENT8}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered table-striped" style={{ fontSize: 11 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>ID</th><th>Sex</th><th>Phenotype</th><th>Genotype</th>
                      <th>Onset (mo)</th><th>Diag (mo)</th>
                      <th>MMA</th><th>tHcy</th><th>Met</th><th>C3</th><th>NH3</th><th>Car</th>
                      <th>Sz</th><th>NBS</th><th>OHCbl</th><th>Macula</th><th>MMACHC?</th><th>AED</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd.patient_sample.map(p => (
                      <tr key={p.id}>
                        <td>{p.id}</td><td>{p.sex}</td><td>{p.phenotype}</td><td style={{ fontSize: 10 }}>{p.genotype}</td>
                        <td>{p.onset_mo}</td><td>{p.diag_mo}</td>
                        <td>{p.mma}</td><td>{p.hcy}</td><td>{p.met}</td><td>{p.c3}</td><td>{p.nh3}</td><td>{p.car}</td>
                        <td>{p.sz ? '✓' : '—'}</td><td>{p.nbs ? '✓' : '—'}</td><td>{p.ohcbl_resp ? '✓' : '—'}</td>
                        <td style={{ color: ACCENT5 }}>ABSENT</td>
                        <td style={{ color: ACCENT3 }}>ABSENT</td>
                        <td style={{ fontSize: 10 }}>{p.aed}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          )}
        </div>
      )}

      {/* ── SEIZURES & TRIGGERS ── */}
      {tab === 'Seizures & Triggers' && (
        <div>
          <Alert text="cblX epilepsy is DUAL: (1) metabolic — MMA+HHcy driven; AND (2) primary NDD — HCFC1 has non-MMACHC transcriptional targets in brain. OHCbl+betaine alone insufficient for seizure control." variant="warning" />

          <Section title="Seizure Type Distribution" color={ACCENT7}>
            {(bd.seizure_type_distribution || []).map(s => (
              <PctBar key={s.type} label={s.type} pct={s.pct} color={s.type.includes('Ohtahara') ? ACCENT7 : s.type.includes('West') ? ACCENT3 : s.type.includes('Focal') ? ACCENT6 : ACCENT} />
            ))}
          </Section>

          <Section title="Metabolic Seizure Triggers" color={ACCENT3}>
            {(bd.metabolic_triggers || []).map(t => (
              <div key={t.trigger} className="card mb-2 border-0 bg-light">
                <div className="card-body py-2">
                  <div className="d-flex justify-content-between align-items-center mb-1">
                    <span className="fw-bold small">{t.trigger}</span>
                    <span className="badge" style={{ background: t.pct === 100 ? '#b71c1c' : ACCENT3, fontSize: 11 }}>{t.pct}%</span>
                  </div>
                  <div className="text-muted" style={{ fontSize: 12 }}>{t.mechanism}</div>
                </div>
              </div>
            ))}
          </Section>

          <Section title="High-Risk Drugs / Exposures" color={ACCENT3}>
            {(bd.high_risk_drugs || []).map(d => (
              <div key={d.drug} className={`alert ${d.risk === 'ABSOLUTE CI' ? 'alert-danger' : 'alert-warning'} py-2 mb-2`} style={{ fontSize: 12 }}>
                <strong>{d.drug}</strong> — <span className="badge bg-danger">{d.risk}</span>
                <div className="mt-1 text-muted">{d.mechanism}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── TREATMENTS ── */}
      {tab === 'Treatments' && (
        <div>
          <Alert text="OHCbl response LESS COMPLETE than cblC (MMACHC protein absent, not just dysfunctional). Betaine is ESPECIALLY CRITICAL. Combination OHCbl + betaine + protein restriction is standard care." variant="info" />
          <Section title="Treatments" color={ACCENT4}>
            {(bd.treatments || []).map(t => (
              <div key={t.treatment} className="card mb-3 border-0 shadow-sm">
                <div className="card-body py-2">
                  <div className="d-flex justify-content-between align-items-start mb-1">
                    <span className="fw-bold" style={{ fontSize: 13 }}>{t.treatment}</span>
                    <div className="d-flex gap-1">
                      <span className="badge" style={{ background: t.evidence === 'AVOID' ? '#b71c1c' : t.evidence.includes('A') ? ACCENT4 : '#6c757d', fontSize: 11 }}>{t.evidence}</span>
                      {t.response_pct > 0 && <span className="badge bg-success" style={{ fontSize: 11 }}>{t.response_pct}% response</span>}
                    </div>
                  </div>
                  <div className="text-muted" style={{ fontSize: 12 }}>{t.note}</div>
                  {t.response_pct > 0 && (
                    <div className="progress mt-2" style={{ height: 6 }}>
                      <div className="progress-bar" style={{ width: `${t.response_pct}%`, backgroundColor: t.evidence === 'AVOID' ? '#b71c1c' : ACCENT4 }} />
                    </div>
                  )}
                </div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && df.gene_card && (
        <div>
          <Section title="Gene Card" color={ACCENT}>
            <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
              <tbody>
                {Object.entries(df.gene_card).map(([k, v]) => (
                  <tr key={k}><td className="fw-bold" style={{ width: '30%' }}>{k}</td><td>{v}</td></tr>
                ))}
              </tbody>
            </table>
          </Section>

          {df.key_concepts && (
            <Section title="Key Concepts" color={ACCENT6}>
              {df.key_concepts.map((c, i) => (
                <div key={i} className="card mb-2 border-0 bg-light">
                  <div className="card-body py-2">
                    <div className="fw-bold mb-1" style={{ fontSize: 13, color: ACCENT6 }}>{c.concept}</div>
                    <div className="text-muted" style={{ fontSize: 12 }}>{c.explanation}</div>
                  </div>
                </div>
              ))}
            </Section>
          )}

          {df.diagnostic_thresholds && (
            <Section title="Diagnostic Thresholds" color={ACCENT4}>
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
                </thead>
                <tbody>
                  {df.diagnostic_thresholds.map((t, i) => (
                    <tr key={i}><td className="fw-bold">{t.parameter}</td><td style={{ color: ACCENT3 }}>{t.threshold}</td><td>{t.action}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
          )}

          {df.differential_diagnosis && (
            <Section title="Differential Diagnosis" color={ACCENT5}>
              {df.differential_diagnosis.map((d, i) => (
                <div key={i} className="card mb-2 border-0 shadow-sm">
                  <div className="card-body py-2">
                    <div className="fw-bold mb-1" style={{ fontSize: 13 }}>{d.disease}</div>
                    <div className="text-muted" style={{ fontSize: 12 }}>{d.distinguishing}</div>
                  </div>
                </div>
              ))}
            </Section>
          )}
        </div>
      )}
    </div>
  );
}
