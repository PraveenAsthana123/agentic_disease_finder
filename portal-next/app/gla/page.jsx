'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#7b1fa2';   // deep purple — X-linked, unique in series
const ACCENT2 = '#c62828';   // dark red — HIGH RISK / QTc / posterior-circulation-stroke
const ACCENT3 = '#e65100';   // deep orange — CAUTION / amenable-mutations / neuropathy
const ACCENT4 = '#1b5e20';   // dark green — SAFE / ERT approved / Level A
const ACCENT5 = '#0d47a1';   // dark blue — Gb3/Lyso-Gb3 biomarker / X-linked pathway
const ACCENT6 = '#004d40';   // dark teal — dual-use GBP / chaperone / migalastat

const SUBTYPE_COLORS = {
  'Classic Male': '#c62828',       // red — full phenotype
  'Late-Onset Male': '#e65100',    // orange — partial phenotype
  'Classic Female': '#6a1b9a',     // purple — variable lyonization
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
    : item.risk?.includes('NO BENEFIT') ? '#37474f'
    : item.risk?.includes('PROTOCOL') ? '#4a148c'
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

export default function GLADashboard() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/gla/overview`).then(r => r.json()),
      fetch(`${API}/api/gla/breakdown`).then(r => r.json()),
      fetch(`${API}/api/gla/definitions`).then(r => r.json()),
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
          <h2 className="mb-1 fw-bold">🧬 GLA Epilepsy — Fabry Disease / Anderson-Fabry Disease</h2>
          <p className="mb-1 opacity-75">
            Alpha-Galactosidase A (α-Gal A) · {ov.locus} · {ov.inheritance}
          </p>
          <p className="mb-0 small opacity-75">{ov.omim}</p>
          <div className="mt-2 d-flex flex-wrap gap-2">
            <Badge text="X-LINKED — UNIQUE IN LYSOSOMAL SERIES (not AR)" color={ACCENT} />
            <Badge text="Corneal Verticillata — PATHOGNOMONIC (95% males, 70% females)" color={ACCENT3} />
            <Badge text="Posterior Circulation Stroke — PATHOGNOMONIC (88%)" color={ACCENT2} />
            <Badge text="Angiokeratoma Buttocks/Genitalia — PATHOGNOMONIC (66%)" color={ACCENT3} />
            <Badge text="GBP/PGB DUAL USE — Neuropathic Pain + Antiseizure" color={ACCENT6} />
            <Badge text="Typical Antipsychotics HIGH RISK — QTc + HCM" color={ACCENT2} />
            <Badge text="Migalastat AMENABLE MUTATIONS ONLY" color={ACCENT6} />
            <Badge text="Agalsidase-alfa EMA 2001 / Agalsidase-beta FDA 2003 — Level A" color={ACCENT4} />
            <Badge text="Pegunigalsidase-alfa (Elfabrio) FDA/EMA 2023" color={ACCENT4} />
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Cohort Size" value={ov.cohort_size} color={ACCENT} />
        <KPI label="Seizures Overall" value={`${ov.seizure_pct_overall}%`} color={ACCENT2} />
        <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Posterior Stroke" value={`${ov.posterior_circulation_stroke_pct}%`} color={ACCENT2} />
        <KPI label="Corneal Verticillata (M)" value={`${ov.corneal_verticillata_males_pct}%`} color={ACCENT3} />
        <KPI label="Corneal Verticillata (F)" value={`${ov.corneal_verticillata_females_pct}%`} color={ACCENT3} />
        <KPI label="Angiokeratoma" value={`${ov.angiokeratoma_pct}%`} color={ACCENT3} />
        <KPI label="HCM" value={`${ov.hcm_pct}%`} color={ACCENT5} />
        <KPI label="DRG Neuropathy" value={`${ov.drg_neuropathy_pct}%`} color={ACCENT5} />
        <KPI label="Psychiatric Misdiag." value={`${ov.psychiatric_misdiagnosis_pct}%`} color={ACCENT6} />
        <KPI label="On ERT" value={`${ov.on_ert_pct}%`} color={ACCENT4} />
        <KPI label="On GBP/PGB" value={`${ov.on_gbp_pgb_pct}%`} color={ACCENT6} />
      </div>

      {/* Pathognomonic Alert */}
      <div className="alert border-0 shadow-sm mb-4" style={{ background: '#fff3e0', borderLeft: `5px solid ${ACCENT3}` }}>
        <strong style={{ color: ACCENT3 }}>⚠️ PATHOGNOMONIC — Fabry Disease / GLA (Xq22.1):</strong>{' '}
        <span className="small">{ov.pathognomonic_note}</span>
      </div>

      {/* X-linked Unique Alert */}
      <div className="alert border-0 shadow-sm mb-4" style={{ background: '#f3e5f5', borderLeft: `5px solid ${ACCENT}` }}>
        <strong style={{ color: ACCENT }}>🧬 X-LINKED — UNIQUE IN LYSOSOMAL SERIES:</strong>{' '}
        <span className="small">
          {ov.inheritance} · All heterozygous females are at risk (NO carrier state).
          GBP/PGB DUAL USE (neuropathic pain + antiseizure) — unique in lysosomal series.
          Migalastat: AMENABLE MUTATIONS ONLY (fabry-database.org mandatory).
          Typical antipsychotics HIGH RISK (QTc prolongation + Fabry HCM → Torsades).
        </span>
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
          <SectionCard title="🧬 Disease Overview — Fabry Disease / Anderson-Fabry Disease (GLA)" color={ACCENT}>
            <p className="small">{ov.disease}</p>
            <div className="row mt-3">
              <div className="col-md-6">
                <p><strong>Gene:</strong> <code>{ov.gene}</code></p>
                <p><strong>Locus:</strong> {ov.locus}</p>
                <p><strong>Inheritance:</strong> {ov.inheritance}</p>
                <p><strong>OMIM:</strong> <code>{ov.omim}</code></p>
                <p><strong>Discovery:</strong> {ov.discovery}</p>
              </div>
              <div className="col-md-6">
                <p><strong>Protein:</strong> {ov.protein}</p>
                <p><strong>Mechanism:</strong> {ov.mechanism}</p>
              </div>
            </div>
          </SectionCard>

          {/* Pathognomonic Features */}
          <SectionCard title="🔬 Pathognomonic Features — GLA / Fabry Disease" color={ACCENT3}>
            <div className="row">
              {ov.pathognomonic_features?.map((feat, i) => (
                <div key={i} className="col-md-6 mb-2">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body py-2 px-3 small">
                      <span style={{ color: ACCENT3 }}>⚠️</span>{' '}{feat}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          {/* Unique Features */}
          <SectionCard title="✨ Unique Features — GLA vs Other Lysosomal Diseases" color={ACCENT}>
            <div className="alert mb-3" style={{ background: '#ede7f6', borderLeft: `4px solid ${ACCENT}` }}>
              <span className="small">{ov.unique_features}</span>
            </div>
          </SectionCard>

          {/* Key Pharmacological Distinctions */}
          <SectionCard title="💊 Key Pharmacological Distinctions — GLA / Fabry" color={ACCENT6}>
            <div className="row">
              {Object.entries(ov.key_pharmacological_distinctions || {}).map(([key, val]) => (
                <div key={key} className="col-md-6 mb-3">
                  <div className="card border-0 bg-light h-100">
                    <div className="card-body small py-2">
                      <strong style={{ color: ACCENT6 }}>{key.replace(/_/g, ' ')}:</strong>{' '}{val}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          {/* Diagnostic Hierarchy */}
          <SectionCard title="🔍 Diagnostic Hierarchy — When to Suspect Fabry / GLA" color={ACCENT3}>
            <p className="small">{ov.diagnostic_hierarchy}</p>
          </SectionCard>

          {/* Treatment Highlights */}
          <SectionCard title="💊 Treatment Highlights" color={ACCENT4}>
            <ol className="small mb-0">
              {ov.treatment_highlights?.map((step, i) => <li key={i} className="mb-1">{step}</li>)}
            </ol>
          </SectionCard>

          {/* Subtype Severity Matrix */}
          <SectionCard title="📊 Subtype Severity by α-Gal A Residual Activity" color={ACCENT5}>
            <div className="row">
              {Object.entries(br?.subtype_severity_matrix || {}).map(([sub, desc]) => (
                <div key={sub} className="col-md-4 mb-2">
                  <div className="card border-0 bg-light h-100">
                    <div className="card-body py-2 px-3 small">
                      <strong style={{ color: Object.values(SUBTYPE_COLORS)[Object.keys(br?.subtype_severity_matrix || {}).indexOf(sub)] || ACCENT }}>{sub}:</strong>{' '}{desc}
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
          <SectionCard title="👥 Cohort — 40 GLA / Fabry Disease Patients" color={ACCENT}>
            <div className="row">
              <div className="col-md-4">
                <PctBar label="Seizures — Classic Male (55%)" pct={ov.seizure_pct_classic_male} color={SUBTYPE_COLORS['Classic Male']} />
                <PctBar label="Seizures — Late-Onset Male (15%)" pct={ov.seizure_pct_late_onset_male} color={SUBTYPE_COLORS['Late-Onset Male']} />
                <PctBar label="Seizures — Classic Female (30%)" pct={ov.seizure_pct_classic_female} color={SUBTYPE_COLORS['Classic Female']} />
                <PctBar label="Drug-Resistant Epilepsy" pct={ov.drug_resistant_pct} color={ACCENT2} />
              </div>
              <div className="col-md-4">
                <PctBar label="Posterior Circulation Stroke" pct={ov.posterior_circulation_stroke_pct} color={ACCENT2} />
                <PctBar label="Corneal Verticillata (Males)" pct={ov.corneal_verticillata_males_pct} color={ACCENT3} />
                <PctBar label="Corneal Verticillata (Females)" pct={ov.corneal_verticillata_females_pct} color={ACCENT3} />
                <PctBar label="Angiokeratoma" pct={ov.angiokeratoma_pct} color={ACCENT3} />
              </div>
              <div className="col-md-4">
                <PctBar label="On ERT (agalsidase/pegunigalsidase)" pct={ov.on_ert_pct} color={ACCENT4} />
                <PctBar label="On Migalastat" pct={ov.on_migalastat_pct} color={ACCENT6} />
                <PctBar label="On LEV" pct={ov.on_lev_pct} color={ACCENT4} />
                <PctBar label="On GBP/PGB (dual use)" pct={ov.on_gbp_pgb_pct} color={ACCENT6} />
              </div>
            </div>
            <div className="alert alert-info mt-3 small mb-0">
              <strong>HCM:</strong> {ov.hcm_pct}% ·{' '}
              <strong>DRG Neuropathy:</strong> {ov.drg_neuropathy_pct}% ·{' '}
              <strong>Psychiatric Misdiagnosis:</strong> {ov.psychiatric_misdiagnosis_pct}% ·{' '}
              <strong>Cohort:</strong> Classic Male 22 · Late-Onset Male 6 · Classic Female 12
            </div>
          </SectionCard>

          <SectionCard title="🧬 Disease Subtypes — GLA / Fabry Etiology Breakdown" color={ACCENT5}>
            <div className="row">
              {br?.etiologies?.map((et, i) => (
                <div key={i} className="col-md-4 mb-3">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-header text-white small fw-bold"
                      style={{ background: Object.values(SUBTYPE_COLORS)[i] || ACCENT }}>
                      {et.name} — {et.pct}%
                    </div>
                    <div className="card-body small">
                      <div className="mb-2">
                        <div className="progress" style={{ height: 10 }}>
                          <div className="progress-bar" style={{
                            width: `${et.pct}%`,
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

          {/* Biomarker Summary */}
          <SectionCard title="🔬 GLA / Fabry Biomarker Summary (Lyso-Gb3 Primary)" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small mb-0">
                <thead>
                  <tr>
                    <th>Biomarker</th>
                    <th>Threshold</th>
                    <th>Sensitivity</th>
                    <th>Specificity</th>
                    <th>Use</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(br?.gla_biomarker_summary || {}).map(([bm, info]) => (
                    <tr key={bm}>
                      <td className="fw-semibold">{bm.replace(/_/g, ' ')}</td>
                      <td><code>{info.threshold}</code></td>
                      <td>{info.sensitivity}</td>
                      <td>{info.specificity}</td>
                      <td className="text-muted">{info.use}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Tab 2: Seizures & Triggers ── */}
      {tab === 2 && (
        <>
          <div className="alert border-0 shadow-sm mb-3" style={{ background: '#ffebee', borderLeft: `5px solid ${ACCENT2}` }}>
            <strong style={{ color: ACCENT2 }}>⚡ Fabry Cerebrovascular Context:</strong>{' '}
            <span className="small">
              ALL seizures in Fabry disease are SECONDARY to cerebrovascular disease —
              there is NO primary Fabry epilepsy. Posterior circulation (vertebrobasilar) strokes
              PATHOGNOMONIC (88% of Fabry strokes). AED selection must account for stroke
              prevention anticoagulation drug interactions.
            </span>
          </div>

          <SectionCard title="⚡ Seizure Types — GLA / Fabry Disease" color={ACCENT2}>
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

          <SectionCard title="⚠️ Seizure Triggers — Fabry / GLA" color={ACCENT3}>
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

          <SectionCard title="🚫 Contraindications & Drug Risks — Fabry / GLA" color={ACCENT2}>
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
          <div className="alert border-0 shadow-sm mb-3" style={{ background: '#e8f5e9', borderLeft: `5px solid ${ACCENT4}` }}>
            <strong style={{ color: ACCENT4 }}>💊 GLA Treatment Priority:</strong>{' '}
            <span className="small">
              4 approved ERT/chaperone options (unique in lysosomal series).
              Migalastat: AMENABLE MUTATIONS ONLY — check fabry-database.org MANDATORY before prescribing.
              GBP/PGB: DUAL USE (neuropathic pain + antiseizure) — UNIQUE IN LYSOSOMAL SERIES.
              Typical antipsychotics: HIGH RISK (QTc + HCM → Torsades).
            </span>
          </div>

          <SectionCard title="💊 Treatment Options — GLA / Fabry Disease" color={ACCENT4}>
            <div className="row">
              {br?.treatments?.map((tx, i) => (
                <TreatmentCard key={i} item={tx} />
              ))}
            </div>
          </SectionCard>

          {/* Treatment Hierarchy */}
          <SectionCard title="📋 Treatment Hierarchy" color={ACCENT6}>
            <ol className="small mb-0">
              {br?.treatment_hierarchy?.map((step, i) => <li key={i} className="mb-1">{step}</li>)}
            </ol>
          </SectionCard>

          <SectionCard title="📊 Diagnostic Thresholds" color={ACCENT5}>
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
          <SectionCard title="📖 Diagnostic Algorithm — GLA / Fabry Disease (7 Steps)" color={ACCENT}>
            <ol className="small mb-0">
              {df?.diagnostic_algorithm?.map((step, i) => (
                <li key={i} className="mb-2">{step}</li>
              ))}
            </ol>
          </SectionCard>

          <SectionCard title="🔍 Differential Diagnosis — Fabry vs Other Conditions" color={ACCENT3}>
            <div className="row">
              {Object.entries(df?.differential_diagnosis || {}).map(([dx, desc]) => (
                <div key={dx} className="col-md-6 mb-3">
                  <div className="card border-0 bg-light h-100">
                    <div className="card-body small py-2">
                      <strong style={{ color: ACCENT3 }}>{dx.replace(/_/g, ' ')}:</strong>{' '}{desc}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="🔬 GLA / Fabry Glossary" color={ACCENT5}>
            <div className="row">
              {Object.entries(df?.gla_glossary || {}).map(([term, def]) => (
                <div key={term} className="col-md-6 mb-3">
                  <div className="card border-0 bg-light h-100">
                    <div className="card-body small py-2">
                      <strong style={{ color: ACCENT5 }}>{term}:</strong>{' '}{def}
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
        GLA Epilepsy Dashboard · {ov.cohort_size} patients · {ov.locus} · X-Linked (XL)
        · Fabry Disease / Anderson-Fabry Disease
        · Pathognomonic: Corneal Verticillata · Angiokeratoma · Posterior Circulation Stroke
        · GBP/PGB Dual Use (Neuropathic Pain + Antiseizure) — UNIQUE IN LYSOSOMAL SERIES
      </div>
    </div>
  );
}
