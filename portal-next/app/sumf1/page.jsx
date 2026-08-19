'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — SUMF1 multi-sulfatase master regulator
const ACCENT2 = '#b71c1c';   // dark-red — HIGH RISK / all-sulfatases-low / lethal
const ACCENT3 = '#e65100';   // deep-orange — PATHOGNOMONIC / combined urine / ichthyosis
const ACCENT4 = '#1b5e20';   // dark-green — safe treatments / ACTH Level A / KD
const ACCENT5 = '#4a148c';   // dark-purple — molecular / FGE / formylglycine
const ACCENT6 = '#01579b';   // dark-blue — biomarkers / sulfatases / thresholds

const SUBTYPE_COLORS = {
  'Type A': '#b71c1c',       // red — neonatal lethal
  'Type B': '#e65100',       // orange — severe infantile
  'Type C': '#2e7d32',       // green — attenuated, ichthyosis
  'Attenuated': '#1565c0',   // blue — adult onset
  'Compound': '#6a1b9a',     // purple — compound het
};

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
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

function Badge({ text, color = ACCENT }) {
  return (
    <span className="badge me-1 mb-1" style={{ background: color, fontSize: '0.72rem' }}>{text}</span>
  );
}

function SectionCard({ title, color = ACCENT, children }) {
  return (
    <div className="card shadow-sm mb-4">
      <div className="card-header text-white fw-bold" style={{ background: color }}>{title}</div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function CICard({ item }) {
  const riskColor = item.risk?.includes('HIGH RISK') ? ACCENT2
    : item.risk?.includes('RELATIVE-CI') ? ACCENT3
    : item.risk?.includes('CAUTION') ? '#f57c00'
    : '#546e7a';
  return (
    <div className="col-md-6 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-header text-white small fw-bold" style={{ background: riskColor }}>
          {item.drug} — {item.risk}
        </div>
        <div className="card-body small">
          <p className="mb-1"><strong>Mechanism:</strong> {item.mechanism}</p>
          <p className="mb-1 text-success"><strong>Alternative:</strong> {item.alternative}</p>
          <p className="mb-0 text-muted"><em>{item.evidence}</em></p>
        </div>
      </div>
    </div>
  );
}

function TreatmentCard({ item }) {
  const lvl = item.level || '';
  const lvlColor = lvl.includes('A') ? ACCENT4
    : lvl.includes('B') ? '#1565c0'
    : lvl.includes('C') ? '#6a1b9a'
    : '#607d8b';
  return (
    <div className="col-md-6 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-header text-white small fw-bold" style={{ background: lvlColor }}>
          {item.treatment} — Level {item.level}
        </div>
        <div className="card-body small">
          <p className="mb-1"><strong>Indication:</strong> {item.indication}</p>
          <p className="mb-1"><strong>Mechanism:</strong> {item.mechanism}</p>
          <p className="mb-1 text-warning-emphasis"><strong>Monitoring:</strong> {item.monitoring}</p>
          {item.caution && <p className="mb-0 text-danger"><strong>Caution:</strong> {item.caution}</p>}
        </div>
      </div>
    </div>
  );
}

export default function SUMF1Dashboard() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/sumf1/overview`).then(r => r.json()),
      fetch(`${API}/api/sumf1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/sumf1/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  const ov = overview, br = breakdown, df = definitions;

  return (
    <div className="container-fluid py-3" style={{ background: '#f3f4f8', minHeight: '100vh' }}>
      {/* Header */}
      <div className="card shadow mb-4" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT5} 100%)` }}>
        <div className="card-body text-white py-4">
          <h2 className="mb-1 fw-bold">🧬 SUMF1 Epilepsy — Multiple Sulfatase Deficiency (Austin Disease)</h2>
          <p className="mb-1 opacity-75">
            Formylglycine-Generating Enzyme (FGE) · {ov.locus} · {ov.inheritance}
          </p>
          <p className="mb-0 small opacity-75">{ov.omim}</p>
          <div className="mt-2 d-flex flex-wrap gap-2">
            <Badge text="ALL 17 SULFATASES SIMULTANEOUSLY LOW — PATHOGNOMONIC" color={ACCENT2} />
            <Badge text="Combined Sulfatiduria + GAGuria — PATHOGNOMONIC" color={ACCENT3} />
            <Badge text="Ichthyosis + Leukodystrophy — Type C PATHOGNOMONIC" color={ACCENT3} />
            <Badge text="No Approved ERT" color="#37474f" />
            <Badge text="ACTH Level A — IS" color={ACCENT4} />
            <Badge text="CBZ/OXC HIGH RISK — MLD Neuropathy" color={ACCENT2} />
            <Badge text="VGB HIGH RISK — Types A+B" color={ACCENT2} />
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Cohort Size" value={ov.cohort_size} color={ACCENT} />
        <KPI label="Seizures Overall" value={`${ov.seizure_pct_overall}%`} color={ACCENT2} />
        <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Infantile Spasms" value={`${ov.infantile_spasms_pct}%`} color={ACCENT3} />
        <KPI label="Peripheral Neuropathy" value={`${ov.peripheral_neuropathy_pct}%`} color={ACCENT5} />
        <KPI label="All Sulfatases Low" value={`${ov.all_sulfatases_low_pct}%`} color={ACCENT3} />
        <KPI label="Ichthyosis" value={`${ov.ichthyosis_pct}%`} color={ACCENT3} />
        <KPI label="Leukodystrophy" value={`${ov.leukodystrophy_pct}%`} color={ACCENT5} />
        <KPI label="Dysostosis Multiplex" value={`${ov.dysostosis_multiplex_pct}%`} color={ACCENT5} />
        <KPI label="Hepatosplenomegaly" value={`${ov.hepatosplenomegaly_pct}%`} color={ACCENT6} />
        <KPI label="Diag. Delay (yr)" value={ov.mean_diagnosis_delay_years} color={ACCENT3} />
        <KPI label="On ACTH" value={`${ov.on_acth_pct}%`} color={ACCENT4} />
      </div>

      {/* Pathognomonic Alert */}
      <div className="alert border-0 shadow-sm mb-4" style={{ background: '#fff3e0', borderLeft: `5px solid ${ACCENT3}` }}>
        <strong style={{ color: ACCENT3 }}>⚠️ PATHOGNOMONIC — Multiple Sulfatase Simultaneous Deficiency:</strong>{' '}
        <span className="small">{ov.pathognomonic_note}</span>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && (
        <>
          <SectionCard title="🧬 Disease Overview — Multiple Sulfatase Deficiency (SUMF1)" color={ACCENT}>
            <p className="small">{ov.disease}</p>
            <div className="row mt-3">
              <div className="col-md-6">
                <p><strong>Gene:</strong> <code>{ov.gene}</code></p>
                <p><strong>Locus:</strong> {ov.locus}</p>
                <p><strong>Inheritance:</strong> {ov.inheritance}</p>
                <p><strong>OMIM:</strong> <code>{ov.omim}</code></p>
              </div>
              <div className="col-md-6">
                <p><strong>Protein:</strong> {ov.protein}</p>
                <p><strong>Mechanism:</strong> {ov.mechanism}</p>
              </div>
            </div>
          </SectionCard>

          {/* Sulfatase Affected Matrix */}
          <SectionCard title="🔬 SUMF1 → 17 Sulfatases Affected — Key Components" color={ACCENT5}>
            <div className="row">
              {Object.entries(br?.sumf1_sulfatase_affected || {}).map(([enzyme, info]) => (
                <div key={enzyme} className="col-md-6 col-lg-4 mb-3">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-header text-white fw-bold small" style={{ background: ACCENT5 }}>
                      {enzyme} — {info.disease_component}
                    </div>
                    <div className="card-body small">
                      <p className="mb-1"><strong>Substrate:</strong> {info.substrate}</p>
                      <p className="mb-1 text-primary"><strong>Biomarker:</strong> {info.biomarker}</p>
                      <p className="mb-0 text-warning-emphasis"><strong>Epilepsy link:</strong> {info.epilepsy_link}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="alert alert-info mt-2 small mb-0">
              <strong>Key insight:</strong> SUMF1/FGE activates ALL 17 sulfatases simultaneously.
              One gene mutation → all pathways blocked. This is why MSD presents as a "combined LSD" —
              MLD + MPS + ichthyosis in a single patient.
            </div>
          </SectionCard>

          {/* Diagnostic Hierarchy */}
          <SectionCard title="🔍 Diagnostic Hierarchy — When to Suspect MSD / SUMF1" color={ACCENT3}>
            <p className="small">{ov.diagnostic_hierarchy}</p>
          </SectionCard>

          {/* Treatment Hierarchy */}
          <SectionCard title="💊 Treatment Hierarchy" color={ACCENT4}>
            <ol className="small mb-0">
              {br?.treatment_hierarchy?.map((step, i) => <li key={i} className="mb-1">{step}</li>)}
            </ol>
          </SectionCard>

          {/* Subtype Severity Matrix */}
          <SectionCard title="📊 Subtype Severity by FGE Residual Activity" color={ACCENT6}>
            <div className="row">
              {Object.entries(br?.subtype_severity_matrix || {}).map(([sub, desc]) => (
                <div key={sub} className="col-md-6 mb-2">
                  <div className="card border-0 bg-light h-100">
                    <div className="card-body py-2 px-3 small">
                      <strong style={{ color: SUBTYPE_COLORS[sub] || ACCENT }}>{sub}:</strong> {desc}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          {/* Standards */}
          <SectionCard title="📚 Key Standards & References" color={ACCENT}>
            <ul className="small mb-0">
              {ov.standards?.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </SectionCard>
        </>
      )}

      {/* ── Tab 1: Patients & Etiology ── */}
      {tab === 1 && (
        <>
          <SectionCard title="👥 Cohort — 40 SUMF1 / MSD Patients" color={ACCENT}>
            <div className="row">
              <div className="col-md-4">
                <PctBar label="Seizures — Type A" pct={ov.seizure_pct_type_a} color={SUBTYPE_COLORS['Type A']} />
                <PctBar label="Seizures — Type B" pct={ov.seizure_pct_type_b} color={SUBTYPE_COLORS['Type B']} />
                <PctBar label="Seizures — Type C" pct={ov.seizure_pct_type_c} color={SUBTYPE_COLORS['Type C']} />
                <PctBar label="Seizures — Attenuated" pct={ov.seizure_pct_attenuated} color={SUBTYPE_COLORS['Attenuated']} />
              </div>
              <div className="col-md-4">
                <PctBar label="Drug-Resistant Epilepsy" pct={ov.drug_resistant_pct} color={ACCENT2} />
                <PctBar label="Peripheral Neuropathy" pct={ov.peripheral_neuropathy_pct} color={ACCENT5} />
                <PctBar label="Leukodystrophy" pct={ov.leukodystrophy_pct} color={ACCENT5} />
                <PctBar label="Dysostosis Multiplex" pct={ov.dysostosis_multiplex_pct} color={ACCENT5} />
              </div>
              <div className="col-md-4">
                <PctBar label="On ACTH" pct={ov.on_acth_pct} color={ACCENT4} />
                <PctBar label="On LEV" pct={ov.on_lev_pct} color={ACCENT4} />
                <PctBar label="On VPA" pct={ov.on_vpa_pct} color={ACCENT4} />
                <PctBar label="Ichthyosis" pct={ov.ichthyosis_pct} color={ACCENT3} />
              </div>
            </div>
            <div className="alert alert-info mt-3 small mb-0">
              <strong>Mean onset:</strong> {ov.mean_onset_months} months ·{' '}
              <strong>Diagnosis delay:</strong> {ov.mean_diagnosis_delay_years} years ·{' '}
              <strong>All sulfatases low:</strong> {ov.all_sulfatases_low_pct}% of cohort ·{' '}
              <strong>Hepatosplenomegaly:</strong> {ov.hepatosplenomegaly_pct}%
            </div>
          </SectionCard>

          <SectionCard title="🧬 Disease Subtypes — MSD Etiology Breakdown" color={ACCENT5}>
            <div className="row">
              {br?.etiologies?.map((et, i) => (
                <div key={i} className="col-md-6 mb-3">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-header text-white small fw-bold"
                      style={{ background: Object.values(SUBTYPE_COLORS)[i] || ACCENT }}>
                      {et.name} — {et.pct}%
                    </div>
                    <div className="card-body small">
                      <div className="mb-2">
                        <div className="progress" style={{ height: 10 }}>
                          <div className="progress-bar" style={{
                            width: `${et.pct * 2}%`,
                            background: Object.values(SUBTYPE_COLORS)[i] || ACCENT
                          }} />
                        </div>
                      </div>
                      <p className="mb-1"><strong>Onset:</strong> {et.onset}</p>
                      <p className="mb-1 small">{et.notes}</p>
                      <div className="alert alert-warning py-1 px-2 small mb-0">
                        <strong>Key Finding:</strong> {et.key_finding}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Tab 2: Seizures & Triggers ── */}
      {tab === 2 && (
        <>
          <SectionCard title="⚡ Seizure Types — SUMF1 / MSD" color={ACCENT2}>
            {br?.seizure_types?.map((st, i) => (
              <div key={i} className="mb-3 p-3 rounded" style={{ background: i % 2 === 0 ? '#fafafa' : '#fff' }}>
                <div className="d-flex justify-content-between align-items-center mb-1">
                  <strong className="small">{st.type}</strong>
                  <Badge text={`${st.pct}%`} color={ACCENT2} />
                </div>
                <div className="progress mb-1" style={{ height: 8 }}>
                  <div className="progress-bar" style={{ width: `${st.pct}%`, background: ACCENT2 }} />
                </div>
                <div className="text-muted small">{st.subtype}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="⚠️ Seizure Triggers" color={ACCENT3}>
            <div className="row">
              {br?.triggers?.map((tr, i) => (
                <div key={i} className="col-md-6 mb-3">
                  <div className="card border-0 bg-light h-100">
                    <div className="card-body small py-2">
                      <div className="d-flex justify-content-between">
                        <strong>{tr.trigger}</strong>
                        <Badge text={`${tr.pct}%`} color={ACCENT3} />
                      </div>
                      <div className="progress my-1" style={{ height: 6 }}>
                        <div className="progress-bar" style={{ width: `${tr.pct}%`, background: ACCENT3 }} />
                      </div>
                      <div className="text-muted small">{tr.notes}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="🚫 Contraindications & Drug Risks" color={ACCENT2}>
            <div className="row">
              {br?.contraindications?.map((ci, i) => (
                <CICard key={i} item={ci} />
              ))}
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Tab 3: Treatments ── */}
      {tab === 3 && (
        <>
          <SectionCard title="💊 Treatment Options — SUMF1 / MSD" color={ACCENT4}>
            <div className="row">
              {br?.treatments?.map((tx, i) => (
                <TreatmentCard key={i} item={tx} />
              ))}
            </div>
          </SectionCard>

          <SectionCard title="📊 Diagnostic Thresholds" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small mb-0">
                <thead>
                  <tr>
                    <th>Threshold</th>
                    <th>Value</th>
                    <th>Clinical Significance</th>
                  </tr>
                </thead>
                <tbody>
                  {br?.thresholds?.map((th, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{th.name}</td>
                      <td><code>{th.value}</code></td>
                      <td className="text-muted">{th.significance}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Tab 4: Definitions ── */}
      {tab === 4 && (
        <>
          <SectionCard title="📖 Diagnostic Algorithm — SUMF1 / Multiple Sulfatase Deficiency" color={ACCENT}>
            <ol className="small mb-0">
              {df?.diagnostic_algorithm?.map((step, i) => (
                <li key={i} className="mb-2">{step}</li>
              ))}
            </ol>
          </SectionCard>

          <SectionCard title="🔍 Differential Diagnosis" color={ACCENT3}>
            <div className="row">
              {Object.entries(df?.differential_diagnosis || {}).map(([dx, desc]) => (
                <div key={dx} className="col-md-6 mb-3">
                  <div className="card border-0 bg-light h-100">
                    <div className="card-body small py-2">
                      <strong style={{ color: ACCENT3 }}>{dx.replace(/_/g, ' ')}:</strong> {desc}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="🔬 SUMF1 / MSD Glossary" color={ACCENT5}>
            <div className="row">
              {Object.entries(df?.sumf1_glossary || {}).map(([term, def]) => (
                <div key={term} className="col-md-6 mb-3">
                  <div className="card border-0 bg-light h-100">
                    <div className="card-body small py-2">
                      <strong style={{ color: ACCENT5 }}>{term}:</strong> {def}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="📚 Key Concepts" color={ACCENT}>
            {df?.key_concepts?.map((kc, i) => (
              <div key={i} className="mb-3 p-3 rounded" style={{ background: i % 2 === 0 ? '#f3f4f8' : '#fff' }}>
                <strong style={{ color: ACCENT }}>{kc.term}</strong>
                <p className="small text-muted mt-1 mb-0">{kc.definition}</p>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="📚 Full References" color={ACCENT}>
            <ul className="small mb-0">
              {df?.standards?.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </SectionCard>
        </>
      )}

      <div className="text-center text-muted small mt-4 pb-3">
        SUMF1 Epilepsy Dashboard · {ov.cohort_size} patients · {ov.locus} · AR Biallelic LOF
        · Multiple Sulfatase Deficiency / Austin Disease
        · Pathognomonic: ALL Sulfatases Simultaneously Low · Combined Sulfatiduria + GAGuria · Ichthyosis Type C
      </div>
    </div>
  );
}
