'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Variants', 'GI & Neuro Profile', 'Treatments', 'Definitions'];
const COLOR = '#1565c0';   // deep blue — TYMP/MNGIE (GI + neuro; adult onset; HSCT curative)
const LIGHT = '#e3f2fd';

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

function Alert({ variant, text }) {
  const bg = variant === 'danger' ? '#ffebee' : variant === 'warning' ? '#fff8e1' : variant === 'success' ? '#e8f5e9' : LIGHT;
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

function Spinner() {
  return <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /></div>;
}

export default function TympPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/tymp/overview`).then(r => r.json()),
      fetch(`${API}/api/tymp/breakdown`).then(r => r.json()),
      fetch(`${API}/api/tymp/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefs(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <Spinner />;
  if (error) return <div className="alert alert-danger m-4">Failed to load: {error}</div>;

  return (
    <div className="container-fluid py-3 px-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: COLOR, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">⛔ TYMP — MNGIE (Mitochondrial Neurogastrointestinal Encephalomyopathy)</h4>
        <div className="small opacity-75">
          MDDS1 · Thymidine Phosphorylase Deficiency · 482 aa · 22q13.32 · OMIM Gene 131222 · Disease 603041 · AR biallelic LOF
        </div>
        <div className="mt-2 d-flex flex-wrap gap-2">
          <span className="badge" style={{ background: '#c62828' }}>VPA — CONTRAINDICATED</span>
          <span className="badge" style={{ background: '#c62828' }}>KD — CONTRAINDICATED</span>
          <span className="badge" style={{ background: '#c62828' }}>Propofol — AVOID (PRIS)</span>
          <span className="badge" style={{ background: '#2e7d32' }}>HSCT — Only Curative Tx</span>
          <span className="badge" style={{ background: '#1565c0' }}>Plasma Thymidine &gt;3 µmol/L — PATHOGNOMONIC</span>
          <span className="badge" style={{ background: '#37474f' }}>LEV — AED of Choice</span>
          <span className="badge" style={{ background: '#1565c0' }}>Adult / Adolescent Onset (15-40 yrs)</span>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link ${tab === t ? 'active' : ''}`}
              onClick={() => setTab(t)}
              style={tab === t ? { color: COLOR, borderBottom: `2px solid ${COLOR}` } : {}}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'Overview' && overview && (
        <>
          <div className="row mb-3">
            {(overview.kpis || []).map(k => (
              <KPI key={k.label} label={k.label} value={k.value} color={COLOR} />
            ))}
          </div>

          <div className="row">
            <div className="col-md-6">
              <SectionCard title="Gene & Disease Summary">
                <table className="table table-sm table-borderless small">
                  <tbody>
                    <tr><td className="fw-bold w-40">Gene</td><td>{overview.gene} — {overview.protein}</td></tr>
                    <tr><td className="fw-bold">Protein Length</td><td>{overview.protein_length_aa} aa · cytoplasmic homodimer · no MTS</td></tr>
                    <tr><td className="fw-bold">Location</td><td>{overview.chromosomal_location}</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>{overview.omim_gene}</td></tr>
                    <tr><td className="fw-bold">Disease OMIM</td><td>{overview.omim_disease} (MNGIE = MDDS1)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>{overview.inheritance}</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~{overview.prevalence_per_million} in 1,000,000 (very rare)</td></tr>
                    <tr><td className="fw-bold">Onset</td><td>Typically 15–40 years (adolescent / young adult) — DISTINCTIVE vs paediatric MDDS</td></tr>
                    <tr><td className="fw-bold">Cohort (simulated)</td><td>{overview.cohort_n} patients · seed {overview.seed}</td></tr>
                    <tr><td className="fw-bold">Mean Onset Age</td><td>{overview.mean_onset_age_yrs} years</td></tr>
                    <tr><td className="fw-bold">Mean BMI</td><td>{overview.mean_bmi} kg/m² (cachexia is cardinal feature)</td></tr>
                  </tbody>
                </table>
              </SectionCard>
            </div>

            <div className="col-md-6">
              <SectionCard title="Prescribing Safety">
                <Alert variant="danger" text="⛔ VPA — CONTRAINDICATED: thymidine pool imbalance worsens mtDNA multiple deletions; hepatotoxicity risk in mitochondrial disease." />
                <Alert variant="danger" text="⛔ KD — CONTRAINDICATED: OXPHOS-dependent beta-oxidation is impaired in mtDNA deletion disease; high-fat diet worsens energy failure." />
                <Alert variant="danger" text="⛔ Propofol — AVOID: Propofol Infusion Syndrome (PRIS) risk in all mitochondrial disease; use sevoflurane or ketamine for anaesthesia." />
                <Alert variant="success" text="✅ HSCT (allogeneic) — ONLY PROVEN CURATIVE THERAPY: donor TYMP normalises plasma thymidine; early HSCT before severe neurological disease gives best outcomes." />
                <Alert variant="success" text="✅ LEV (Levetiracetam) — AED of choice if seizures: renal excretion; no CoA sequestration; no ETC inhibition; IV formulation available." />
                <Alert variant="warning" text="⚠ Peritoneal Dialysis — BRIDGE to HSCT: reduces plasma thymidine 50-70% but does NOT fully normalise; use while awaiting HSCT." />
              </SectionCard>
            </div>
          </div>

          <SectionCard title="🔬 Pathognomonic Diagnostic Test — Plasma Thymidine">
            {overview.key_diagnostic_test && (
              <div className="row">
                <div className="col-md-6">
                  <table className="table table-sm small">
                    <tbody>
                      <tr><td className="fw-bold">Test</td><td>{overview.key_diagnostic_test.test}</td></tr>
                      <tr><td className="fw-bold">Normal range</td><td className="text-success">{overview.key_diagnostic_test.normal}</td></tr>
                      <tr><td className="fw-bold">MNGIE range</td><td className="text-danger fw-bold">{overview.key_diagnostic_test.mngie_range}</td></tr>
                      <tr><td className="fw-bold">Also order</td><td>{overview.key_diagnostic_test.also_order}</td></tr>
                      <tr><td className="fw-bold">Confirmation</td><td>{overview.key_diagnostic_test.confirmation}</td></tr>
                      <tr><td className="fw-bold">mtDNA result</td><td>{overview.key_diagnostic_test.mtdna}</td></tr>
                    </tbody>
                  </table>
                </div>
                <div className="col-md-6">
                  <div className="p-2 rounded small" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
                    {overview.key_diagnostic_test.note}
                  </div>
                </div>
              </div>
            )}
          </SectionCard>

          {overview.prescribing_summary && (
            <SectionCard title="Prescribing Quick-Reference">
              <div className="row">
                {Object.entries(overview.prescribing_summary).map(([k, v]) => (
                  <div key={k} className="col-md-6 mb-2">
                    <div className="small">
                      <strong className="text-capitalize">{k.replace(/_/g, ' ')}:</strong>{' '}
                      <span className={k === 'vpa' || k === 'kd' || k === 'propofol' ? 'text-danger' : k === 'curative' ? 'text-success' : ''}>{v}</span>
                    </div>
                  </div>
                ))}
              </div>
            </SectionCard>
          )}
        </>
      )}

      {/* ── Patients & Variants ── */}
      {tab === 'Patients & Variants' && breakdown && (
        <>
          <SectionCard title="Genotype Class Distribution">
            <div className="row mb-3">
              {(breakdown.genotype_distribution || []).map(g => (
                <div key={g.class} className="col-md-4 mb-3">
                  <div className="card h-100 shadow-sm">
                    <div className="card-body">
                      <div className="fw-bold" style={{ color: COLOR }}>{g.class}</div>
                      <div className="text-muted small">{g.n} patients ({g.pct}%)</div>
                      <div className="mt-1 small">{g.onset_note}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="Feature Prevalence (n=40 cohort)">
            {(breakdown.feature_prevalence || []).map(f => (
              <div key={f.feature} className="mb-4">
                <Bar label={f.feature} value={f.pct} />
                <div className="text-muted small ms-1">{f.note}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Sample Patients (first 8)">
            <div className="table-responsive">
              <table className="table table-sm table-hover small">
                <thead style={{ background: LIGHT }}>
                  <tr>
                    <th>ID</th><th>Sex</th><th>Onset</th><th>BMI</th>
                    <th>dThd µmol/L</th><th>dU µmol/L</th><th>TYMP%</th>
                    <th>Gastroparesis</th><th>PEO</th><th>Neuropathy</th>
                    <th>Leuko</th><th>HSCT</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.patients_sample || []).map(p => (
                    <tr key={p.id}>
                      <td>{p.id}</td>
                      <td>{p.sex}</td>
                      <td>{p.onset_age_yrs} yr</td>
                      <td className={p.bmi < 16 ? 'text-danger fw-bold' : p.bmi < 18.5 ? 'text-warning' : ''}>{p.bmi}</td>
                      <td className={p.plasma_thymidine_umolL > 15 ? 'text-danger fw-bold' : 'text-warning'}>{p.plasma_thymidine_umolL}</td>
                      <td>{p.plasma_dU_umolL}</td>
                      <td>{p.tymp_activity_pct_control}%</td>
                      <td>{p.gastroparesis ? '✓' : '—'}</td>
                      <td>{p.peo ? '✓' : '—'}</td>
                      <td>{p.peripheral_neuropathy ? '✓' : '—'}</td>
                      <td>{p.leukoencephalopathy_mri ? '✓' : '—'}</td>
                      <td>{p.hsct_completed ? '✅' : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="Disease Timeline">
            {(breakdown.disease_timeline || []).map(t => (
              <div key={t.phase} className="mb-3">
                <div className="fw-bold small" style={{ color: COLOR }}>{t.phase}</div>
                <div className="text-muted small">{t.events}</div>
              </div>
            ))}
          </SectionCard>
        </>
      )}

      {/* ── GI & Neuro Profile ── */}
      {tab === 'GI & Neuro Profile' && breakdown && (
        <>
          <SectionCard title="GI Dysmotility Profile">
            {breakdown.gi_profile && Object.entries(breakdown.gi_profile).map(([key, val]) => (
              <div key={key} className="mb-3 p-2 rounded" style={{ background: LIGHT }}>
                <div className="fw-bold small text-capitalize">{key.replace(/_/g, ' ')}</div>
                {Object.entries(val).map(([k2, v2]) => (
                  <div key={k2} className="small text-muted">
                    <strong>{k2.replace(/_/g, ' ')}: </strong>{String(v2)}
                  </div>
                ))}
              </div>
            ))}
          </SectionCard>

          <SectionCard title="MRI — Leukoencephalopathy Profile">
            {breakdown.mri_leukoencephalopathy_profile && (
              <div className="row">
                <div className="col-md-6">
                  <table className="table table-sm small">
                    <tbody>
                      {Object.entries(breakdown.mri_leukoencephalopathy_profile).filter(([k]) => k !== 'ddx_mri').map(([k, v]) => (
                        <tr key={k}>
                          <td className="fw-bold text-capitalize w-40">{k.replace(/_/g, ' ')}</td>
                          <td>{String(v)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="col-md-6">
                  <div className="fw-bold small mb-2" style={{ color: COLOR }}>DDx on MRI</div>
                  {breakdown.mri_leukoencephalopathy_profile.ddx_mri && Object.entries(breakdown.mri_leukoencephalopathy_profile.ddx_mri).map(([k, v]) => (
                    <Alert key={k} variant="info" text={`${k.replace(/_/g, ' ')}: ${v}`} />
                  ))}
                </div>
              </div>
            )}
          </SectionCard>

          <SectionCard title="Plasma Thymidine — Diagnostic Standards">
            {breakdown.plasma_thymidine_diagnostic && (
              <div className="row">
                <div className="col-md-6">
                  <table className="table table-sm small">
                    <tbody>
                      {Object.entries(breakdown.plasma_thymidine_diagnostic).filter(([k]) => k !== 'interpretation' && k !== 'key_note').map(([k, v]) => (
                        <tr key={k}>
                          <td className="fw-bold text-capitalize w-40">{k.replace(/_/g, ' ')}</td>
                          <td>{String(v)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="col-md-6">
                  {breakdown.plasma_thymidine_diagnostic.interpretation && (
                    <>
                      <div className="fw-bold small mb-2" style={{ color: COLOR }}>Severity by Thymidine Level</div>
                      {Object.entries(breakdown.plasma_thymidine_diagnostic.interpretation).map(([k, v]) => (
                        <Alert key={k} variant={k === 'severe' ? 'danger' : k === 'moderate' ? 'warning' : 'success'} text={`${k.toUpperCase()}: ${v}`} />
                      ))}
                    </>
                  )}
                  {breakdown.plasma_thymidine_diagnostic.key_note && (
                    <div className="p-2 rounded small mt-2" style={{ background: '#e8eaf6', borderLeft: `4px solid ${COLOR}` }}>
                      {breakdown.plasma_thymidine_diagnostic.key_note}
                    </div>
                  )}
                </div>
              </div>
            )}
          </SectionCard>

          <SectionCard title="DDx — Key Differentials">
            {breakdown.ddx_summary && (
              <div className="row">
                {Object.entries(breakdown.ddx_summary).map(([k, v]) => (
                  <div key={k} className="col-md-6 mb-2">
                    <div className="p-2 rounded small" style={{ background: '#f3e5f5', borderLeft: '4px solid #7b1fa2' }}>
                      <strong className="text-capitalize">{k.replace(/_/g, ' ')}: </strong>{v}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </SectionCard>
        </>
      )}

      {/* ── Treatments ── */}
      {tab === 'Treatments' && breakdown && (
        <SectionCard title="Treatment Ladder — TYMP MNGIE">
          {(breakdown.treatments || []).map(tx => {
            const lvl = tx.level || '';
            const bg = lvl.startsWith('A') ? '#e8f5e9' : lvl.startsWith('B') ? '#fff8e1' : '#fce4ec';
            const border = lvl.startsWith('A') ? '#2e7d32' : lvl.startsWith('B') ? '#f57f17' : '#c62828';
            return (
              <div key={tx.tx} className="mb-3 p-3 rounded" style={{ background: bg, borderLeft: `4px solid ${border}` }}>
                <div className="fw-bold small">{tx.tx}</div>
                <div className="small text-muted">{tx.level}</div>
                <div className="small mt-1">{tx.note}</div>
              </div>
            );
          })}
        </SectionCard>
      )}

      {/* ── Definitions ── */}
      {tab === 'Definitions' && defs && (
        <SectionCard title="Clinical Definitions — TYMP MNGIE">
          {(defs.terms || []).map(t => (
            <div key={t.term} className="mb-3">
              <div className="fw-bold small" style={{ color: COLOR }}>{t.term}</div>
              <div className="text-muted small">{t.definition}</div>
              <hr className="my-2" />
            </div>
          ))}
        </SectionCard>
      )}

      <div className="text-muted small text-end mt-2">
        TYMP MNGIE (MDDS1) · seed-565 · {overview?.generated}
      </div>
    </div>
  );
}
