'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — ABCD4 / ABC transporter / lysosomal
const ACCENT2 = '#004d40';   // dark teal — betaine / HHcy arm treatment
const ACCENT3 = '#b71c1c';   // deep red — MMA accumulation / lysosomal trap
const ACCENT4 = '#1565c0';   // blue — treatment / Level A
const ACCENT5 = '#37474f';   // slate — KEY NEGATIVES / cblJ vs cblF distinction
const ACCENT6 = '#6a1b9a';   // deep purple — NBD / ATPase / Walker motifs
const ACCENT7 = '#4e342e';   // dark brown — variant data / gene card
const ACCENT8 = '#0d47a1';   // dark blue — pathway / upstream

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

export default function ABCD4Page() {
  const [tab, setTab]         = useState('Overview');
  const [overview, setOv]     = useState(null);
  const [breakdown, setBd]    = useState(null);
  const [definitions, setDf]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/abcd4/overview`).then(r => r.json()),
      fetch(`${API}/api/abcd4/breakdown`).then(r => r.json()),
      fetch(`${API}/api/abcd4/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOv(ov); setBd(bd); setDf(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-4 text-center text-muted">Loading ABCD4 (cblJ) dashboard…</div>;
  if (error)   return <div className="p-4 text-danger">Error: {error}</div>;
  if (!overview) return null;

  const ov = overview;
  const bd = breakdown || {};
  const df = definitions || {};

  return (
    <div className="container-fluid py-3 px-3" style={{ maxWidth: 1100 }}>

      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT6} 100%)`, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">&#x1f9ec; ABCD4 (cblJ) Epilepsy Dashboard</h4>
        <div style={{ fontSize: 13, opacity: 0.9 }}>
          Methylmalonic Aciduria &amp; Homocystinuria — cblJ type · Lysosomal Cobalamin ATPase Motor ·
          14q24.3 · AR · OMIM {ov.omim_gene} / {ov.omim_disease}
        </div>
        <div className="mt-1" style={{ fontSize: 12, opacity: 0.8 }}>
          {ov.protein_size} · {ov.prevalence}
        </div>
      </div>

      {/* Critical alerts */}
      <Alert variant="danger" text="⚠️ N2O (Nitrous Oxide) — ABSOLUTE CI: irreversibly inactivates Methionine Synthase (MTR) → acute HHcy crisis. LIFE-THREATENING. Notify anesthesiology at every surgery." />
      <Alert variant="warning" text="⚠️ VPA (Valproate) — HIGH RISK: carnitine depletion + metabolic stress → MMA surge. Use LEV first-line for ALL seizure types in ABCD4/cblJ." />
      <Alert variant="warning" text="⚠️ Betaine MANDATORY alongside OHCbl — BHMT cobalamin-independent remethylation. Neither alone is sufficient." />
      <Alert variant="info" text="ℹ️ ABCD4 (cblJ): ATPase MOTOR of ABCD4–LMBRD1 lysosomal export complex. Biochemically IDENTICAL to cblF (LMBRD1) — NO vacuolated lymphocytes, NO stomatitis in cblJ. Gene panel ABCD4+LMBRD1 MANDATORY — cannot distinguish biochemically. VLCFA NORMAL (unlike ABCD1/X-ALD)." />

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === t ? ' active fw-bold' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'Overview' && (
        <div>
          <Section title="Key Performance Indicators" color={ACCENT}>
            <div className="row g-2">
              <KPI label="Cohort (n)" value={ov.cohort_n} color={ACCENT} />
              <KPI label="Avg MMA (mmol/mol Cr)" value={ov.kpis.avg_mma_urine} color={ACCENT3} />
              <KPI label="Avg tHcy (µmol/L)" value={ov.kpis.avg_homocysteine_umol_l} color={ACCENT2} />
              <KPI label="Avg Methionine (µmol/L)" value={ov.kpis.avg_methionine_umol_l} color={ACCENT6} />
              <KPI label="Seizures (%)" value={`${ov.kpis.seizure_pct}%`} color={ACCENT3} />
              <KPI label="Vacuolated Lymph (%)" value={`0% (ABSENT)`} color={ACCENT5} />
              <KPI label="Stomatitis (%)" value={`0% (ABSENT)`} color={ACCENT5} />
              <KPI label="NBS Detected (%)" value={`${ov.kpis.nbs_detected_pct}%`} color={ACCENT4} />
              <KPI label="OHCbl Response (%)" value={`${ov.kpis.ohcbl_response_pct}%`} color={ACCENT} />
              <KPI label="Avg NH3 (µmol/L)" value={ov.kpis.avg_ammonia_umol_l} color={ACCENT7} />
              <KPI label="Avg Carnitine (µmol/L)" value={ov.kpis.avg_free_carnitine} color={ACCENT4} />
              <KPI label="Avg C3 (µmol/L)" value={ov.kpis.avg_c3_umol_l} color={ACCENT8} />
            </div>
          </Section>

          {/* cblJ vs cblF distinction panel */}
          <Section title="cblJ (ABCD4) vs cblF (LMBRD1) — KEY DISTINCTIONS" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Feature</th><th>cblJ (ABCD4)</th><th>cblF (LMBRD1)</th></tr>
                </thead>
                <tbody>
                  <tr><td className="fw-bold">Molecular role</td><td style={{ color: ACCENT6 }}>ATPase MOTOR (NBD; Walker A/B)</td><td>Transporter PORE (9 TM helices)</td></tr>
                  <tr><td className="fw-bold">Gene / location</td><td style={{ color: ACCENT }}>ABCD4 · 14q24.3</td><td>LMBRD1 · 6q13</td></tr>
                  <tr><td className="fw-bold">Vacuolated lymphocytes</td><td style={{ color: ACCENT5, fontWeight: 'bold' }}>ABSENT (0%) — KEY NEGATIVE</td><td style={{ color: ACCENT3 }}>Present ~25–45% — KEY POSITIVE</td></tr>
                  <tr><td className="fw-bold">Stomatitis</td><td style={{ color: ACCENT5, fontWeight: 'bold' }}>ABSENT (0%) — KEY NEGATIVE</td><td style={{ color: ACCENT3 }}>Present ~65% — pathognomonic cblF</td></tr>
                  <tr><td className="fw-bold">MMA range</td><td>150–1,200 mmol/mol Cr</td><td>200–1,500 mmol/mol Cr</td></tr>
                  <tr><td className="fw-bold">Biochemistry</td><td colSpan={2} style={{ textAlign: 'center', color: ACCENT3, fontWeight: 'bold' }}>IDENTICAL — combined MMA + HHcy; both MeCbl/AdoCbl absent; lysosomal Cbl on EM</td></tr>
                  <tr><td className="fw-bold">Gene panel</td><td colSpan={2} style={{ textAlign: 'center', fontWeight: 'bold' }}>ABCD4 + LMBRD1 MANDATORY — cannot distinguish by biochemistry alone</td></tr>
                  <tr><td className="fw-bold">ABCD family</td><td style={{ color: ACCENT6 }}>LYSOSOMAL (unique); VLCFA NORMAL</td><td>Not ABCD family (LMBR1 domain)</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Phenotype Distribution" color={ACCENT8}>
            {(ov.phenotype_distribution || []).map((ph, i) => (
              <div key={i} className="mb-3 p-3 border rounded" style={{ borderLeft: `4px solid ${ACCENT8}` }}>
                <div className="fw-bold mb-1" style={{ color: ACCENT8 }}>{ph.phenotype}</div>
                <PctBar label="Prevalence" pct={ph.pct} color={ACCENT8} />
                <div className="row mt-2" style={{ fontSize: 12 }}>
                  <div className="col-md-3"><strong>Onset:</strong> {ph.onset}</div>
                  <div className="col-md-3"><strong>Vacuolated Lymph:</strong> <span style={{ color: ACCENT5 }}>{ph.vacuolated_lymphocytes}</span></div>
                  <div className="col-md-3"><strong>Stomatitis:</strong> <span style={{ color: ACCENT5 }}>{ph.stomatitis}</span></div>
                  <div className="col-md-3"><strong>MMA:</strong> {ph.mma}</div>
                </div>
                <div style={{ fontSize: 12, marginTop: 4, color: '#555' }}>{ph.note}</div>
              </div>
            ))}
          </Section>

          <Section title="Function &amp; Mechanism" color={ACCENT}>
            <div className="p-3 rounded mb-3" style={{ background: '#e8eaf6', fontSize: 13 }}>
              <strong>Function:</strong> {ov.function}
            </div>
            <div className="p-3 rounded mb-3" style={{ background: '#fff3e0', fontSize: 13 }}>
              <strong>Mechanism (LOF):</strong> {ov.mechanism}
            </div>
            <div className="p-3 rounded" style={{ background: '#f3e5f5', fontSize: 13 }}>
              <strong>Key Negatives:</strong> {ov.key_negative}
            </div>
          </Section>

          <Section title="ABCD4 Cobalamin Pathway (Step-by-Step)" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Step</th><th>Reaction</th><th>Enzyme / Gene</th><th>Consequence of cblJ LOF</th></tr>
                </thead>
                <tbody>
                  {(ov.abcd4_pathway || []).map((s, i) => (
                    <tr key={i} style={{ background: i === 1 ? '#fff3cd' : 'inherit' }}>
                      <td className="fw-bold" style={{ color: i === 1 ? ACCENT3 : ACCENT, width: '22%', fontSize: 11 }}>{s.step}</td>
                      <td style={{ fontSize: 11 }}>{s.reaction}</td>
                      <td style={{ color: ACCENT7, fontSize: 11 }}>{s.enzyme}</td>
                      <td style={{ color: i === 1 ? ACCENT3 : '#555', fontSize: 11, fontWeight: i === 1 ? 'bold' : 'normal' }}>{s.consequence_lof}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="High-Risk Situations" color={ACCENT3}>
            {(ov.high_risk_situations || []).map((s, i) => (
              <div key={i} className="mb-2 p-2 border rounded" style={{ borderLeft: `3px solid ${ACCENT3}` }}>
                <div className="d-flex justify-content-between">
                  <span className="fw-bold" style={{ color: ACCENT3, fontSize: 13 }}>{s.situation}</span>
                  <span className="badge" style={{ background: s.risk.includes('ABSOLUTE') ? '#b71c1c' : s.risk.includes('HIGH') ? '#e65100' : ACCENT4, color: '#fff', fontSize: 11 }}>{s.risk}</span>
                </div>
                <div style={{ fontSize: 12, marginTop: 4 }}>{s.detail}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── Patients & Biomarkers ── */}
      {tab === 'Patients & Biomarkers' && (
        <div>
          <Section title="Biomarker Reference — cblJ (ABCD4)" color={ACCENT2}>
            {(bd.biomarkers || []).map((b, i) => (
              <div key={i} className="mb-3 p-3 border rounded" style={{ borderLeft: `3px solid ${ACCENT2}` }}>
                <div className="fw-bold" style={{ color: ACCENT2, fontSize: 13 }}>{b.name}</div>
                <div className="row mt-1" style={{ fontSize: 12 }}>
                  <div className="col-md-2"><strong>Normal:</strong> {b.normal}</div>
                  <div className="col-md-4" style={{ color: ACCENT3 }}><strong>cblJ Range:</strong> {b.abcd4_range}</div>
                  <div className="col-md-6">{b.significance}</div>
                </div>
                <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>Method: {b.method}</div>
              </div>
            ))}
          </Section>

          <Section title="Key Variants (ABCD4)" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Variant</th><th>cDNA</th><th>Domain</th><th>Severity</th><th>Phenotype</th><th>Clinical Note</th></tr>
                </thead>
                <tbody>
                  {(bd.key_variants || []).map((v, i) => (
                    <tr key={i}>
                      <td className="fw-bold" style={{ color: ACCENT7 }}>{v.variant}</td>
                      <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{v.cdna}</td>
                      <td style={{ fontSize: 11, color: ACCENT6 }}>{v.domain}</td>
                      <td style={{ color: ACCENT3, fontSize: 11 }}>{v.severity}</td>
                      <td style={{ fontSize: 11 }}>{v.phenotype}</td>
                      <td style={{ fontSize: 11 }}>{v.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Patient Sample (n=15)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 11 }}>
                <thead className="table-dark">
                  <tr>
                    <th>ID</th><th>Sex</th><th>Phenotype</th><th>Onset (mo)</th>
                    <th>MMA↑</th><th>tHcy↑</th><th>Met↓</th><th>C3↑</th>
                    <th>NH3↑</th><th>Vacuol.</th><th>Stom.</th><th>NBS</th><th>OHCbl Resp.</th><th>AED</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.patient_sample || []).map((p, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{p.patient_id}</td>
                      <td>{p.sex}</td>
                      <td style={{ fontSize: 10 }}>{p.phenotype}</td>
                      <td>{p.onset_months}</td>
                      <td style={{ color: ACCENT3 }}>{p.mma_urine_mmol_molCr}</td>
                      <td style={{ color: ACCENT2 }}>{p.total_homocysteine_umol_l}</td>
                      <td style={{ color: ACCENT6 }}>{p.methionine_umol_l}</td>
                      <td style={{ color: ACCENT8 }}>{p.c3_umol_l}</td>
                      <td>{p.ammonia_umol_l}</td>
                      <td style={{ color: ACCENT5 }}>— (0%)</td>
                      <td style={{ color: ACCENT5 }}>— (0%)</td>
                      <td style={{ color: p.nbs_detected ? ACCENT : '#b71c1c' }}>{p.nbs_detected ? 'Detected' : 'Missed'}</td>
                      <td style={{ color: p.ohcbl_response ? ACCENT : '#b71c1c' }}>{p.ohcbl_response ? 'Yes' : 'No'}</td>
                      <td style={{ fontSize: 10 }}>{p.aed}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── Seizures & Triggers ── */}
      {tab === 'Seizures & Triggers' && (
        <div>
          <Section title="Seizure Types — ABCD4 (cblJ)" color={ACCENT3}>
            {(bd.seizure_types || []).map((s, i) => (
              <div key={i} className="mb-3">
                <PctBar label={s.type} pct={s.pct} color={ACCENT3} />
                <div style={{ fontSize: 12, color: '#555', marginLeft: 4 }}>{s.note}</div>
              </div>
            ))}
          </Section>

          <Section title="Metabolic Triggers" color={ACCENT8}>
            {(bd.metabolic_triggers || []).map((t, i) => (
              <div key={i} className="mb-3 p-2 border rounded" style={{ borderLeft: `3px solid ${ACCENT8}` }}>
                <div className="d-flex justify-content-between mb-1">
                  <span className="fw-bold" style={{ color: ACCENT8, fontSize: 13 }}>{t.trigger}</span>
                  <span className="text-muted small">{t.pct}% of patients</span>
                </div>
                <div style={{ fontSize: 12 }}>{t.mechanism}</div>
              </div>
            ))}
          </Section>

          <Section title="High-Risk Drugs — cblJ" color={ACCENT3}>
            {(bd.high_risk_drugs || []).map((d, i) => (
              <div key={i} className="mb-2 p-2 border rounded" style={{ borderLeft: `3px solid ${ACCENT3}` }}>
                <div className="d-flex justify-content-between">
                  <span className="fw-bold" style={{ color: ACCENT3, fontSize: 13 }}>{d.drug}</span>
                  <span className="badge" style={{ background: d.risk.includes('ABSOLUTE') ? '#b71c1c' : '#e65100', color: '#fff', fontSize: 11 }}>{d.risk}</span>
                </div>
                <div style={{ fontSize: 12, marginTop: 4 }}>{d.mechanism}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── Treatments ── */}
      {tab === 'Treatments' && (
        <div>
          <Alert variant="danger" text="N2O — ABSOLUTE CI: notify anesthesiology. Betaine MANDATORY alongside OHCbl." />
          <Section title="Treatments — cblJ (ABCD4)" color={ACCENT4}>
            {(bd.treatments || []).map((t, i) => (
              <div key={i} className="mb-3 p-3 border rounded" style={{ borderLeft: `4px solid ${t.evidence === 'Level A' ? ACCENT : t.evidence === 'Level B' ? ACCENT4 : '#b71c1c'}` }}>
                <div className="d-flex justify-content-between mb-1">
                  <span className="fw-bold" style={{ color: ACCENT, fontSize: 13 }}>{t.treatment}</span>
                  <span className="badge" style={{ background: t.evidence === 'Level A' ? ACCENT : t.evidence === 'Level B' ? ACCENT4 : '#b71c1c', color: '#fff' }}>{t.evidence}</span>
                </div>
                {t.response_pct > 0 && (
                  <PctBar label="Response rate" pct={t.response_pct} color={ACCENT} />
                )}
                <div style={{ fontSize: 12, color: '#555' }}>{t.note}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'Definitions' && (
        <div>
          <Section title="Gene Card — ABCD4 (cblJ)" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <tbody>
                  {Object.entries(df.gene_card || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold" style={{ color: ACCENT7, width: '30%' }}>{k}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Key Concepts" color={ACCENT}>
            {(df.key_concepts || []).map((c, i) => (
              <div key={i} className="mb-3 p-3 border rounded" style={{ borderLeft: `3px solid ${ACCENT4}` }}>
                <div className="fw-bold mb-1" style={{ color: ACCENT4, fontSize: 13 }}>{c.concept}</div>
                <div style={{ fontSize: 12, lineHeight: 1.6 }}>{c.explanation}</div>
              </div>
            ))}
          </Section>

          <Section title="Diagnostic Thresholds &amp; Actions" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
                </thead>
                <tbody>
                  {(df.diagnostic_thresholds || []).map((d, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{d.parameter}</td>
                      <td style={{ color: ACCENT3 }}>{d.threshold}</td>
                      <td style={{ fontSize: 11 }}>{d.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Differential Diagnosis" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Disease</th><th>Key Distinguishing Feature</th></tr>
                </thead>
                <tbody>
                  {(df.differential_diagnosis || []).map((d, i) => (
                    <tr key={i}>
                      <td className="fw-bold" style={{ width: '28%' }}>{d.disease}</td>
                      <td style={{ color: ACCENT6, fontSize: 11 }}>{d.distinguishing}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}
    </div>
  );
}
