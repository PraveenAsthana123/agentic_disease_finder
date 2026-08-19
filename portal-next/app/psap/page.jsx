'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — prosaposin / saposin multi-activator
const ACCENT2 = '#b71c1c';   // dark-red — HIGH RISK / false-negative enzyme / lethal
const ACCENT3 = '#e65100';   // deep-orange — CAUTION / PATHOGNOMONIC / false-negative
const ACCENT4 = '#1b5e20';   // dark-green — safe treatments / HSCT / Level A-B
const ACCENT5 = '#4a148c';   // dark-purple — saposin molecular / pathway
const ACCENT6 = '#01579b';   // dark-blue — biomarkers / enzyme activity

const SAP_COLORS = {
  SapA: '#b71c1c',   // red — Krabbe-phenocopy
  SapB: '#e65100',   // orange — MLD-phenocopy
  SapC: '#2e7d32',   // green — Gaucher-phenocopy
  SapD: '#4a148c',   // purple — Farber-phenocopy
  Complete: '#37474f', // dark-grey — neonatal lethal
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

export default function PSAPDashboard() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/psap/overview`).then(r => r.json()),
      fetch(`${API}/api/psap/breakdown`).then(r => r.json()),
      fetch(`${API}/api/psap/definitions`).then(r => r.json()),
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
          <h2 className="mb-1 fw-bold">🧬 PSAP Epilepsy — Prosaposin Deficiency</h2>
          <p className="mb-1 opacity-75">
            Saposin A / B / C / D Deficiencies · {ov.locus} · {ov.inheritance}
          </p>
          <p className="mb-0 small opacity-75">{ov.omim}</p>
          <div className="mt-2 d-flex flex-wrap gap-2">
            <Badge text="ENZYME FALSE NEGATIVE — PATHOGNOMONIC" color={ACCENT2} />
            <Badge text="SapA → Krabbe-Phenocopy" color={SAP_COLORS.SapA} />
            <Badge text="SapB → MLD-Phenocopy" color={SAP_COLORS.SapB} />
            <Badge text="SapC → Gaucher-Phenocopy" color={SAP_COLORS.SapC} />
            <Badge text="SapD → Farber-Type7" color={SAP_COLORS.SapD} />
            <Badge text="No Approved ERT" color="#37474f" />
            <Badge text="ACTH Level A — IS" color={ACCENT4} />
            <Badge text="VGB HIGH RISK — SapA Optic Atrophy" color={ACCENT2} />
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
        <KPI label="False-Neg Enzyme Assay" value={`${ov.false_negative_enzyme_assay_pct}%`} color={ACCENT3} />
        <KPI label="Leukodystrophy" value={`${ov.leukodystrophy_pct}%`} color={ACCENT5} />
        <KPI label="Optic Atrophy" value={`${ov.optic_atrophy_pct}%`} color={ACCENT6} />
        <KPI label="Hepatosplenomegaly" value={`${ov.hepatosplenomegaly_pct}%`} color={ACCENT6} />
        <KPI label="Diag. Delay (yr)" value={ov.mean_diagnosis_delay_years} color={ACCENT3} />
        <KPI label="On ACTH" value={`${ov.on_acth_pct}%`} color={ACCENT4} />
        <KPI label="On LEV" value={`${ov.on_lev_pct}%`} color={ACCENT4} />
      </div>

      {/* Pathognomonic Alert */}
      <div className="alert border-0 shadow-sm mb-4" style={{ background: '#fff3e0', borderLeft: `5px solid ${ACCENT3}` }}>
        <strong style={{ color: ACCENT3 }}>⚠️ PATHOGNOMONIC — Enzyme False Negative:</strong>{' '}
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
          <SectionCard title="🧬 Disease Overview — Prosaposin (PSAP) Deficiency" color={ACCENT}>
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

          {/* Saposin→Enzyme Matrix */}
          <SectionCard title="🔬 Saposin→Enzyme Activation Matrix" color={ACCENT5}>
            <div className="row">
              {Object.entries(br?.saposin_enzyme_matrix || {}).map(([sap, info]) => (
                <div key={sap} className="col-md-6 mb-3">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-header text-white fw-bold small" style={{ background: SAP_COLORS[sap] || ACCENT }}>
                      {sap} Deficiency — {info.disease_phenocopy}
                    </div>
                    <div className="card-body small">
                      <p className="mb-1"><strong>Enzyme:</strong> {info.enzyme}</p>
                      <p className="mb-1"><strong>Substrate:</strong> {info.substrate}</p>
                      <p className="mb-1 text-danger fw-bold"><strong>Assay:</strong> {info.enzyme_assay}</p>
                      <p className="mb-1 text-primary"><strong>Biomarker:</strong> {info.biomarker}</p>
                      <p className="mb-0 text-warning-emphasis small"><strong>Key CI:</strong> {info.specific_CI}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          {/* Diagnostic Hierarchy */}
          <SectionCard title="🔍 Diagnostic Hierarchy — When to Suspect PSAP" color={ACCENT3}>
            <p className="small">{ov.diagnostic_hierarchy}</p>
          </SectionCard>

          {/* Treatment Hierarchy */}
          <SectionCard title="💊 Treatment Hierarchy" color={ACCENT4}>
            <ol className="small mb-0">
              {br?.treatment_hierarchy?.map((step, i) => <li key={i} className="mb-1">{step}</li>)}
            </ol>
          </SectionCard>

          {/* Subtype Seizure Summary */}
          <SectionCard title="⚡ Seizure Prevalence by Subtype" color={ACCENT6}>
            <div className="row">
              {Object.entries(br?.subtype_seizure_summary || {}).map(([sub, desc]) => (
                <div key={sub} className="col-md-6 mb-2">
                  <div className="card border-0 bg-light h-100">
                    <div className="card-body py-2 px-3 small">
                      <strong style={{ color: SAP_COLORS[sub] || ACCENT }}>{sub}:</strong> {desc}
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
          <SectionCard title="👥 Cohort — 40 PSAP Patients" color={ACCENT}>
            <div className="row">
              <div className="col-md-4">
                <PctBar label="Seizure Prevalence (SapA)" pct={ov.seizure_pct_sapa} color={SAP_COLORS.SapA} />
                <PctBar label="Seizure Prevalence (SapB)" pct={ov.seizure_pct_sapb} color={SAP_COLORS.SapB} />
                <PctBar label="Seizure Prevalence (SapC)" pct={ov.seizure_pct_sapc} color={SAP_COLORS.SapC} />
                <PctBar label="Seizure Prevalence (SapD)" pct={ov.seizure_pct_sapd} color={SAP_COLORS.SapD} />
              </div>
              <div className="col-md-4">
                <PctBar label="Drug-Resistant Epilepsy" pct={ov.drug_resistant_pct} color={ACCENT2} />
                <PctBar label="Peripheral Neuropathy" pct={ov.peripheral_neuropathy_pct} color={ACCENT5} />
                <PctBar label="Leukodystrophy" pct={ov.leukodystrophy_pct} color={ACCENT5} />
                <PctBar label="Optic Atrophy" pct={ov.optic_atrophy_pct} color={ACCENT6} />
              </div>
              <div className="col-md-4">
                <PctBar label="On ACTH" pct={ov.on_acth_pct} color={ACCENT4} />
                <PctBar label="On LEV" pct={ov.on_lev_pct} color={ACCENT4} />
                <PctBar label="On VPA" pct={ov.on_vpa_pct} color={ACCENT4} />
                <PctBar label="On Ketogenic Diet" pct={ov.on_kd_pct} color={ACCENT4} />
              </div>
            </div>
            <div className="alert alert-info mt-3 small mb-0">
              <strong>Mean onset:</strong> {ov.mean_onset_months} months ·{' '}
              <strong>Diagnosis delay:</strong> {ov.mean_diagnosis_delay_years} years ·{' '}
              <strong>Enzyme false-negative assay:</strong> {ov.false_negative_enzyme_assay_pct}% of cohort
            </div>
          </SectionCard>

          <SectionCard title="🧬 Disease Subtypes — Etiology Breakdown" color={ACCENT5}>
            <div className="row">
              {br?.etiologies?.map((et, i) => (
                <div key={i} className="col-md-6 mb-3">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-header text-white small fw-bold"
                      style={{ background: Object.values(SAP_COLORS)[i] || ACCENT }}>
                      {et.name} — {et.pct}%
                    </div>
                    <div className="card-body small">
                      <div className="mb-2">
                        <div className="progress" style={{ height: 10 }}>
                          <div className="progress-bar" style={{
                            width: `${et.pct * 2}%`,
                            background: Object.values(SAP_COLORS)[i] || ACCENT
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
          <SectionCard title="⚡ Seizure Types — PSAP Subtypes" color={ACCENT2}>
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
          <SectionCard title="💊 Treatment Options — PSAP Subtypes" color={ACCENT4}>
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
          <SectionCard title="📖 Diagnostic Algorithm — PSAP / Saposin Deficiency" color={ACCENT}>
            <ol className="small mb-0">
              {df?.diagnostic_algorithm?.map((step, i) => (
                <li key={i} className="mb-2">{step}</li>
              ))}
            </ol>
          </SectionCard>

          <SectionCard title="🔬 Saposin Pathway Glossary" color={ACCENT5}>
            <div className="row">
              {Object.entries(df?.saposin_pathway_glossary || {}).map(([term, def]) => (
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
        PSAP Epilepsy Dashboard · {ov.cohort_size} patients · {ov.locus} · AR Biallelic LOF
        · Saposin A (Krabbe-phenocopy) · Saposin B (MLD-phenocopy)
        · Saposin C (Gaucher-phenocopy) · Saposin D (Farber-Type7)
        · Pathognomonic: Enzyme False-Negative Assay
      </div>
    </div>
  );
}
