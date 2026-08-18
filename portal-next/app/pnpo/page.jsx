'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#880e4f';   // deep burgundy — PNPO / PLP-dependent metabolic epilepsy
const ACCENT2 = '#b71c1c';   // dark red — contraindications / danger
const ACCENT3 = '#1b5e20';   // dark green — PLP response / treatable

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
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
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

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header fw-bold" style={{ backgroundColor: '#fce4ec', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function TabBtn({ label, active, onClick }) {
  return (
    <button
      className={`btn btn-sm me-1 mb-1 ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
      style={active ? { backgroundColor: ACCENT, borderColor: ACCENT } : {}}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

export default function PNPOPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/pnpo/overview`).then(r => r.json()),
      fetch(`${API}/api/pnpo/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pnpo/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error)   return <div className="alert alert-danger m-4">Error: {error}</div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="p-3 mb-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #4a148c 100%)` }}>
        <h4 className="mb-1">🧬 PNPO Epilepsy — Pyridoxamine-5'-phosphate Oxidase Deficiency</h4>
        <div className="small opacity-90">
          PLP-Dependent Neonatal Epileptic Encephalopathy · 17q21.32 · AR-LOF · OMIM #610090
        </div>
        <div className="mt-2 small fw-bold" style={{ color: '#ffeb3b' }}>
          ⚠️ PNPO requires PLP (NOT pyridoxine) — pyridoxine trial will be NEGATIVE in true PNPO LOF
        </div>
      </div>

      {/* Critical Pearl */}
      <div className="alert alert-danger mb-3 fw-bold" style={{ fontSize: 14, borderLeft: `5px solid ${ACCENT2}` }}>
        🚨 CRITICAL: If IV pyridoxine 30 mg/kg gives no EEG response in 60 min →
        IMMEDIATELY trial PLP 30 mg/kg oral/NG (do NOT conclude "not B6-dependent").
        PNPO enzyme is absent → pyridoxine CANNOT be converted to PLP.
        Every hour without PLP = irreversible brain injury.
      </div>

      {/* Tabs */}
      <div className="mb-3">
        {TABS.map((t, i) => <TabBtn key={t} label={t} active={tab === i} onClick={() => setTab(i)} />)}
      </div>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && overview && (
        <>
          <div className="row g-2 mb-3">
            <KPI label="Patients"          value={overview.total_patients}           color={ACCENT} />
            <KPI label="Premature Birth"   value={`${overview.premature_birth_pct}%`} color="#5d4037" />
            <KPI label="In Utero Seizures" value={`${overview.in_utero_seizures_pct}%`} color={ACCENT} />
            <KPI label="PLP Seizure-Free"  value={`${overview.plp_seizure_free_pct}%`} color={ACCENT3} />
            <KPI label="Pyridoxine Fails"  value={`${overview.pyridoxine_ineffective_pct}%`} color={ACCENT2} />
            <KPI label="Inheritance"       value="AR"                                color="#4527a0" />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <SectionCard title="Etiology Distribution">
                {(overview.etiology_breakdown || []).map(e => (
                  <PctBar key={e.label} label={e.label.replace(/-/g, ' ')} pct={e.pct} color={ACCENT} />
                ))}
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="Key Biomarkers" borderColor="#b71c1c">
                <ul className="list-unstyled mb-0">
                  {(overview.key_biomarkers || []).map((b, i) => (
                    <li key={i} className="mb-1"><span className="badge me-2" style={{ backgroundColor: '#b71c1c' }}>↑↑</span>{b}</li>
                  ))}
                </ul>
                <hr />
                <div className="small text-muted">
                  <strong>Collect BEFORE pyridoxine trial</strong> — PMP/PNP normalise within hours of treatment (Clayton 2011 JIMD)
                </div>
              </SectionCard>

              <SectionCard title="Absolute Contraindications" borderColor={ACCENT2}>
                {(overview.absolute_CIs || []).map((ci, i) => (
                  <div key={i} className="badge me-1 mb-1 p-2" style={{ backgroundColor: ACCENT2, fontSize: 13 }}>
                    🚫 {ci}
                  </div>
                ))}
                <div className="mt-2 small text-danger fw-bold">
                  VPA inhibits GAD + pyridoxal kinase → compounds PLP deficiency.<br />
                  INH forms covalent PLP hydrazone adducts → inactivates PLP permanently.
                </div>
              </SectionCard>
            </div>
          </div>

          <SectionCard title="Treatment Hierarchy">
            <div className="d-flex flex-wrap gap-2">
              {(overview.treatment_hierarchy || []).map((t, i) => (
                <div key={i} className="badge p-2" style={{ backgroundColor: ACCENT3, fontSize: 13 }}>
                  {i + 1}. {t}
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="Critical Clinical Pearl" borderColor={ACCENT3}>
            <p className="mb-0 fw-bold" style={{ color: ACCENT3 }}>{overview.critical_pearl}</p>
          </SectionCard>
        </>
      )}

      {/* ── Tab 1: Patients & Etiology ── */}
      {tab === 1 && breakdown && (
        <>
          <SectionCard title={`Etiology Catalog (${breakdown.etiologies?.length || 0} classes)`}>
            {(breakdown.etiologies || []).map((e, i) => (
              <div key={i} className="mb-4 p-3 rounded" style={{ background: '#fce4ec', borderLeft: `3px solid ${ACCENT}` }}>
                <div className="fw-bold" style={{ color: ACCENT }}>{e.etiology}</div>
                <div className="small text-muted mb-1">N={e.n} · {e.pct}% · Class: {e.category}</div>
                <div className="small">{e.mechanism}</div>
                {e.response_to_pyridoxine && (
                  <div className="mt-2 small">
                    <span className="badge me-2" style={{ backgroundColor: ACCENT2 }}>Pyridoxine</span>{e.response_to_pyridoxine}
                  </div>
                )}
                {e.response_to_plp && (
                  <div className="mt-1 small">
                    <span className="badge me-2" style={{ backgroundColor: ACCENT3 }}>PLP</span>{e.response_to_plp}
                  </div>
                )}
              </div>
            ))}
          </SectionCard>

          <SectionCard title={`Patient Registry (N=${breakdown.patients?.length || 0})`}>
            <div className="table-responsive">
              <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                <thead style={{ backgroundColor: ACCENT, color: '#fff' }}>
                  <tr>
                    <th>ID</th><th>Name</th><th>Sex</th><th>Age(mo)</th>
                    <th>Onset</th><th>Etiology</th><th>Variant</th><th>Response</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.patients || []).map(p => (
                    <tr key={p.id}>
                      <td>{p.id}</td>
                      <td>{p.name}</td>
                      <td>{p.sex}</td>
                      <td>{p.age_months}</td>
                      <td>{p.onset}</td>
                      <td><span className="badge" style={{ backgroundColor: ACCENT }}>{p.etiology_short}</span></td>
                      <td><code style={{ fontSize: 11 }}>{p.pnpo_variant}</code></td>
                      <td style={{ fontSize: 11 }}>{p.plp_response}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Tab 2: Seizures & Triggers ── */}
      {tab === 2 && breakdown && (
        <>
          <SectionCard title="Seizure Types">
            {(breakdown.seizure_types || []).map((s, i) => (
              <div key={i} className="mb-4 p-3 rounded" style={{ background: '#fce4ec', borderLeft: `3px solid ${ACCENT}` }}>
                <div className="d-flex justify-content-between">
                  <span className="fw-bold" style={{ color: ACCENT }}>{s.type}</span>
                  <span className="badge" style={{ backgroundColor: ACCENT }}>{s.pct}%</span>
                </div>
                <PctBar label="" pct={s.pct} color={ACCENT} />
                <div className="small mb-1"><strong>EEG:</strong> {s.eeg}</div>
                <div className="small mb-1"><strong>Semiology:</strong> {s.semiology}</div>
                <div className="small text-success"><strong>Clinical tip:</strong> {s.clinical_tip}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Seizure Triggers">
            {(breakdown.triggers || []).map((t, i) => (
              <div key={i} className="mb-3 p-3 rounded" style={{ background: '#fff3e0', borderLeft: `3px solid #e65100` }}>
                <div className="d-flex justify-content-between">
                  <span className="fw-bold">{t.trigger}</span>
                  <span className="badge bg-warning text-dark">{t.pct}%</span>
                </div>
                <PctBar label="" pct={t.pct} color="#e65100" />
                <div className="small mb-1">{t.mechanism}</div>
                <div className="small text-primary"><strong>Note:</strong> {t.clinical_note}</div>
              </div>
            ))}
          </SectionCard>
        </>
      )}

      {/* ── Tab 3: Treatments ── */}
      {tab === 3 && breakdown && (
        <>
          <SectionCard title="Treatments">
            {(breakdown.treatments || []).map((t, i) => (
              <div key={i} className="mb-4 p-3 rounded" style={{ background: '#e8f5e9', borderLeft: `3px solid ${ACCENT3}` }}>
                <div className="d-flex justify-content-between align-items-start mb-1">
                  <span className="fw-bold" style={{ color: ACCENT3 }}>{t.treatment}</span>
                  <span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.level}</span>
                </div>
                <div className="small mb-1"><strong>Role:</strong> {t.role}</div>
                <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
                <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>
                <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
                <div className="small mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
                <div className="small text-primary"><strong>PNPO notes:</strong> {t.pnpo_specific_notes}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Absolute Contraindications" borderColor={ACCENT2}>
            {(breakdown.contraindications || []).map((ci, i) => (
              <div key={i} className="mb-3 p-3 rounded" style={{ background: '#ffebee', borderLeft: `3px solid ${ACCENT2}` }}>
                <div className="d-flex justify-content-between mb-1">
                  <span className="fw-bold text-danger">{ci.drug}</span>
                  <span className="badge bg-danger">{ci.severity}</span>
                </div>
                <div className="small mb-1">{ci.reason}</div>
                <div className="small text-success"><strong>Alternative:</strong> {ci.alternative}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Monitoring (14 items)" borderColor="#5d4037">
            <ol className="mb-0">
              {(breakdown.monitoring || []).map((m, i) => (
                <li key={i} className="small mb-1">{m}</li>
              ))}
            </ol>
          </SectionCard>

          <SectionCard title="Lifecycle Stages" borderColor="#4527a0">
            {(breakdown.lifecycle || []).map((l, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ background: '#ede7f6' }}>
                <div className="fw-bold" style={{ color: '#4527a0' }}>{l.stage} <span className="fw-normal text-muted small">({l.age_range})</span></div>
                <div className="small">{l.description}</div>
              </div>
            ))}
          </SectionCard>
        </>
      )}

      {/* ── Tab 4: Definitions ── */}
      {tab === 4 && definitions && (
        <>
          <SectionCard title="Gene & Protein">
            <div className="row g-2 small">
              <div className="col-md-6"><strong>Gene:</strong> {definitions.gene} ({definitions.full_name})</div>
              <div className="col-md-6"><strong>Locus:</strong> {definitions.locus}</div>
              <div className="col-md-6"><strong>OMIM Gene:</strong> {definitions.omim_gene}</div>
              <div className="col-md-6"><strong>OMIM Phenotype:</strong> {definitions.omim_phenotype}</div>
              <div className="col-md-12"><strong>Protein:</strong> {definitions.protein}</div>
              <div className="col-md-12"><strong>Enzyme class:</strong> {definitions.enzyme_class}</div>
            </div>
            {definitions.syndrome && (
              <div className="mt-2">
                {Object.entries(definitions.syndrome).map(([k, v]) => (
                  <div key={k} className="small mb-1">
                    <strong style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}:</strong> {v}
                  </div>
                ))}
              </div>
            )}
          </SectionCard>

          <SectionCard title="Key Pharmacological Distinctions">
            <ol className="mb-0">
              {(definitions.key_pharmacological_distinctions || []).map((d, i) => (
                <li key={i} className="small mb-2">{d}</li>
              ))}
            </ol>
          </SectionCard>

          <SectionCard title="Key Concepts (15)">
            {definitions.concepts && Object.entries(definitions.concepts).map(([k, v]) => (
              <div key={k} className="mb-3">
                <div className="fw-bold small" style={{ color: ACCENT }}>{k.replace(/-/g, ' ')}</div>
                <div className="small text-muted">{v}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Thresholds">
            <div className="row g-2">
              {definitions.thresholds && Object.entries(definitions.thresholds).map(([k, v]) => (
                <div key={k} className="col-md-4 col-sm-6">
                  <div className="border rounded p-2 small">
                    <div className="text-muted" style={{ fontSize: 11 }}>{k.replace(/_/g, ' ')}</div>
                    <div className="fw-bold">{v}</div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="Evidence Standards">
            <ol className="mb-0">
              {(definitions.evidence_standards || []).map((s, i) => (
                <li key={i} className="small mb-1">{s}</li>
              ))}
            </ol>
          </SectionCard>
        </>
      )}
    </div>
  );
}
