'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#311b92';   // deep indigo — Kv3.2 Shaw subfamily / KCNC2 GOF-LOF dual phenotype (same family as KCNC1 violet but distinct)
const ACCENT2 = '#b71c1c';   // dark red — CI / danger / contraindications
const ACCENT3 = '#e65100';   // deep orange — alerts / triggers / HIGH RISK
const ACCENT4 = '#1b5e20';   // deep green — safe / seizure freedom / Level A/B safe treatments
const ACCENT5 = '#004d40';   // dark teal — TRN / thalamocortical / monitoring

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

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header py-2" style={{ backgroundColor: '#f8f9fa', borderBottom: `1px solid ${borderColor}20` }}>
        <strong style={{ color: borderColor }}>{title}</strong>
      </div>
      <div className="card-body py-3">{children}</div>
    </div>
  );
}

function AlertBox({ color, children }) {
  return (
    <div className="p-3 rounded mb-3" style={{ backgroundColor: color + '15', border: `1px solid ${color}40` }}>
      <span style={{ color }}>{children}</span>
    </div>
  );
}

// ── Tab: Overview ───────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="KCNC2 Gene Overview — Kv3.2 Shaw K⁺ / PV+ Fast-Spiking Interneuron + TRN / 12q21.32" borderColor={ACCENT}>
        <div className="row g-2 mb-3">
          <div className="col-md-6"><strong>Gene / Locus:</strong> {data.gene} / {data.locus}</div>
          <div className="col-md-6"><strong>OMIM:</strong> {data.omim_gene} (gene) · {data.omim_disease}</div>
          <div className="col-md-6"><strong>Inheritance:</strong> {data.inheritance}</div>
          <div className="col-md-6"><strong>Precision Therapy:</strong> {data.precision_therapy}</div>
          <div className="col-12 mt-2"><strong>Protein:</strong> {data.protein}</div>
          <div className="col-12 mt-2"><strong>Mechanism:</strong> <span className="text-muted small">{data.mechanism}</span></div>
        </div>
      </SectionCard>

      <AlertBox color={ACCENT2}>
        ⚡ CRITICAL AHA: {data.key_aha}
      </AlertBox>

      <SectionCard title="KCNC2 vs KCNC1 — Essential Distinction" borderColor={ACCENT5}>
        <div className="p-3 rounded" style={{ backgroundColor: ACCENT5 + '10', border: `1px solid ${ACCENT5}30` }}>
          <div className="small" style={{ color: ACCENT5 }}>{data.kcnc2_vs_kcnc1}</div>
        </div>
      </SectionCard>

      <SectionCard title={`Cohort KPIs — ${data.n_patients} KCNC2 Patients`} borderColor={ACCENT}>
        <div className="row g-2">
          <KPI label="GOF Subtype" value={`${data.gof_pct}%`} color={ACCENT} />
          <KPI label="LOF DEE" value={`${data.lof_pct}%`} color={ACCENT2} />
          <KPI label="Haploinsufficiency" value={`${data.haplo_pct}%`} color={ACCENT} />
          <KPI label="Seizure-Free" value={`${data.seizure_free_pct}%`} color={ACCENT4} />
          <KPI label="Drug Resistant" value={`${data.drug_resistant_pct}%`} color={ACCENT2} />
          <KPI label="Photosensitive" value={`${data.photosensitive_pct}%`} color={ACCENT3} />
          <KPI label="On LEV" value={`${data.on_lev_pct}%`} color={ACCENT4} />
          <KPI label="On VPA" value={`${data.on_vpa_pct}%`} color={ACCENT} />
          <KPI label="On LTG" value={`${data.on_ltg_pct}%`} color={ACCENT} />
          <KPI label="On CBZ" value={`${data.on_cbz_pct}%`} color={ACCENT3} />
          <KPI label="POLG Done" value={`${data.polg_done_pct}%`} color={ACCENT4} />
          <KPI label="Avg Onset" value={`${data.avg_onset_years}Y`} color={ACCENT} />
        </div>
      </SectionCard>

      <SectionCard title="⚠ Safety Alerts — KCNC2 Specific" borderColor={ACCENT2}>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT2 }}>ABSOLUTE CI</span>
          <strong>Tiagabine (TGB):</strong> NCSE risk — all KCNC2 phenotypes
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT2 }}>ABSOLUTE CI</span>
          <strong>VPA + POLG1 mutation:</strong> {data.polg_alert}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT3 }}>HIGH RISK</span>
          <strong>CBZ / OXC in LOF:</strong> {data.cbz_risk_alert}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT3 }}>HIGH RISK</span>
          <strong>LTG in LOF DEE:</strong> {data.ltg_risk_alert}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT3 }}>NOT RECOMMENDED</span>
          <strong>ETX Monotherapy:</strong> {data.etx_risk_alert}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ backgroundColor: ACCENT3 }}>MANDATORY</span>
          <strong>VPPP females ≥12y on VPA:</strong> {data.vppp_alert}
        </div>
      </SectionCard>

      <SectionCard title="Key Pharmacological Note — KCNC2 GOF vs LOF" borderColor={ACCENT}>
        <div className="row g-3">
          <div className="col-md-6">
            <div className="p-3 rounded" style={{ backgroundColor: ACCENT4 + '10', border: `1px solid ${ACCENT4}30` }}>
              <strong style={{ color: ACCENT4 }}>GOF Phenotype — Safe AEDs</strong>
              <div className="small mt-2">
                <div>✓ LEV (Level B — first-line)</div>
                <div>✓ LTG (Level B — NaV block safe in GOF)</div>
                <div>✓ CBZ/OXC (Level C — EEG monitor at 6 weeks)</div>
                <div>✓ VPA (Level B — POLG1 first)</div>
                <div>✓ CLB (Level B adjunct)</div>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-3 rounded" style={{ backgroundColor: ACCENT2 + '08', border: `1px solid ${ACCENT2}30` }}>
              <strong style={{ color: ACCENT2 }}>LOF Phenotype — Avoid / Caution</strong>
              <div className="small mt-2">
                <div>⛔ TGB — ABSOLUTE CI (NCSE)</div>
                <div>⚠ CBZ/OXC — HIGH RISK (EEG mandatory)</div>
                <div>⚠ LTG — HIGH RISK (PV+ NaV1.1 second-hit)</div>
                <div>⛔ VPA + POLG1 — ABSOLUTE CI</div>
                <div>✓ LEV + VPA + CLB (if POLG1 clear)</div>
              </div>
            </div>
          </div>
        </div>
        <AlertBox color={ACCENT5}>
          ⚡ CLASSIFY GOF vs LOF (functional electrophysiology / genotype-phenotype) BEFORE selecting AED.
          If uncertain → start LEV + VPA (POLG1 first) — broadest safe coverage pending characterisation.
        </AlertBox>
      </SectionCard>
    </div>
  );
}

// ── Tab: Patients & Etiology ─────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { summary, etiologies, patients } = data;
  return (
    <div>
      <SectionCard title="Cohort Summary" borderColor={ACCENT}>
        <div className="row g-2">
          <KPI label="Total" value={summary?.total ?? 40} color={ACCENT} />
          <KPI label="GOF" value={`${summary?.gof_pct ?? 38}%`} color={ACCENT} />
          <KPI label="LOF DEE" value={`${summary?.lof_pct ?? 28}%`} color={ACCENT2} />
          <KPI label="Haploinsuff" value={`${summary?.haplo_pct ?? 18}%`} color={ACCENT} />
          <KPI label="Drug Resistant" value={`${summary?.drug_resistant_pct ?? 32}%`} color={ACCENT2} />
          <KPI label="POLG Done" value={`${summary?.polg_done_pct ?? 68}%`} color={ACCENT4} />
        </div>
        <AlertBox color={ACCENT2}>
          ⛔ Key CI: {summary?.key_ci ?? 'TGB ABSOLUTE CI; VPA+POLG1 ABSOLUTE CI; CBZ/LTG HIGH RISK in LOF'}
        </AlertBox>
      </SectionCard>

      <SectionCard title="Etiology Catalog — KCNC2 Variant Classes" borderColor={ACCENT}>
        {(etiologies || []).map((e, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ backgroundColor: '#f8f9fa', border: `1px solid ${ACCENT}20` }}>
            <div className="d-flex justify-content-between align-items-start mb-2">
              <strong style={{ color: ACCENT }}>{e.etiology}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{e.pct}% (n={e.n})</span>
            </div>
            <div className="small text-muted mb-2"><strong>Mechanism:</strong> {e.mechanism}</div>
            <div className="small mb-1"><strong>EEG:</strong> {e.eeg_correlate}</div>
            <div className="small mb-1"><strong>MRI:</strong> {e.mri_finding}</div>
            <div className="small"><strong>Semiology:</strong> {e.semiology}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Sample (first 15)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead style={{ backgroundColor: ACCENT + '15' }}>
              <tr>
                <th>ID</th><th>Phenotype</th><th>Onset (Y)</th>
                <th>LEV</th><th>LEV Resp</th><th>VPA</th><th>LTG</th><th>CLB</th><th>CBZ</th>
                <th>POLG</th><th>Photo</th><th>DRE</th>
              </tr>
            </thead>
            <tbody>
              {(patients || []).slice(0, 15).map((p, i) => (
                <tr key={i}>
                  <td className="small">{p.id}</td>
                  <td className="small" style={{ maxWidth: 130 }}>{p.category}</td>
                  <td className="small">{p.onset_years}</td>
                  <td><span style={{ color: p.on_lev ? ACCENT4 : '#999' }}>{p.on_lev ? '✓' : '–'}</span></td>
                  <td><span style={{ color: p.lev_responder ? ACCENT4 : ACCENT2 }}>{p.lev_responder ? '✓' : '✗'}</span></td>
                  <td><span style={{ color: p.on_vpa ? ACCENT : '#999' }}>{p.on_vpa ? '✓' : '–'}</span></td>
                  <td><span style={{ color: p.on_ltg ? ACCENT : '#999' }}>{p.on_ltg ? '✓' : '–'}</span></td>
                  <td><span style={{ color: p.on_clb ? ACCENT : '#999' }}>{p.on_clb ? '✓' : '–'}</span></td>
                  <td><span style={{ color: p.on_cbz ? ACCENT3 : '#999' }}>{p.on_cbz ? '⚠' : '–'}</span></td>
                  <td><span style={{ color: p.polg_done ? ACCENT4 : ACCENT2 }}>{p.polg_done ? 'Y' : '?'}</span></td>
                  <td><span style={{ color: p.photosensitive ? ACCENT3 : '#999' }}>{p.photosensitive ? '⚡' : '–'}</span></td>
                  <td><span style={{ color: p.drug_resistant ? ACCENT2 : ACCENT4 }}>{p.drug_resistant ? '✓' : '–'}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Seizure Types & Triggers ────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types, triggers } = data;
  return (
    <div>
      <SectionCard title="Seizure / Symptom Type Distribution" borderColor={ACCENT}>
        {(seizure_types || []).map((s, i) => (
          <PctBar key={i} label={s.type} pct={s.prevalence_pct} color={ACCENT} />
        ))}
      </SectionCard>

      <SectionCard title="Seizure Types — EEG + Semiology + Clinical Tips" borderColor={ACCENT}>
        {(seizure_types || []).map((s, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ backgroundColor: '#f8f9fa', border: `1px solid ${ACCENT}20` }}>
            <div className="d-flex justify-content-between mb-2">
              <strong style={{ color: ACCENT }}>{s.type}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{s.prevalence_pct}%</span>
            </div>
            <div className="small mb-1"><strong>EEG:</strong> {s.eeg}</div>
            <div className="small mb-1"><strong>Semiology:</strong> {s.semiology}</div>
            <div className="small p-2 rounded" style={{ backgroundColor: ACCENT + '10', color: ACCENT }}>
              <strong>💡 Clinical Tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Trigger Distribution" borderColor={ACCENT3}>
        {(triggers || []).map((t, i) => (
          <PctBar key={i} label={t.trigger} pct={t.prevalence_pct} color={ACCENT3} />
        ))}
      </SectionCard>

      <SectionCard title="Trigger Detail — Mechanism + Management" borderColor={ACCENT3}>
        {(triggers || []).map((t, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ backgroundColor: '#fff8f0', border: `1px solid ${ACCENT3}30` }}>
            <div className="d-flex justify-content-between mb-2">
              <strong style={{ color: ACCENT3 }}>{t.trigger}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.prevalence_pct}%</span>
            </div>
            <div className="small mb-1"><strong>Mechanism:</strong> {t.mechanism}</div>
            <div className="small p-2 rounded" style={{ backgroundColor: ACCENT3 + '10' }}>
              <strong>Management:</strong> {t.management}
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Treatments ──────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatments, contraindications, monitoring, lifecycle } = data;
  return (
    <div>
      <SectionCard title="Treatment Protocols — KCNC2 (GOF-LOF Phenotype-Guided)" borderColor={ACCENT4}>
        {(treatments || []).map((t, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ backgroundColor: '#f0fdf4', border: `1px solid ${ACCENT4}30` }}>
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ color: ACCENT4 }}>{t.drug}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.evidence}</span>
            </div>
            <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>
            <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small mb-1"><strong>Safety:</strong> {t.safety}</div>
            {t.kcnc2_note && (
              <div className="small p-2 rounded" style={{ backgroundColor: ACCENT + '10', color: ACCENT }}>
                <strong>🧬 KCNC2 Note:</strong> {t.kcnc2_note}
              </div>
            )}
            {(t.monitoring || []).length > 0 && (
              <div className="small mt-2">
                <strong>Monitoring:</strong>{' '}
                {t.monitoring.map((m, j) => (
                  <span key={j} className="badge me-1" style={{ backgroundColor: ACCENT + '25', color: ACCENT }}>{m}</span>
                ))}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications — KCNC2 Phenotype-Specific" borderColor={ACCENT2}>
        {(contraindications || []).map((c, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ backgroundColor: '#fff0f0', border: `1px solid ${ACCENT2}30` }}>
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ color: ACCENT2 }}>⛔ {c.drug}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT2 }}>{c.severity}</span>
            </div>
            <div className="small mb-1">{c.mechanism}</div>
            {c.alternative && (
              <div className="small p-2 rounded" style={{ backgroundColor: ACCENT4 + '10', color: ACCENT4 }}>
                <strong>Alternative:</strong> {c.alternative}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Protocol" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead style={{ backgroundColor: ACCENT + '15' }}>
              <tr><th>Monitoring Item</th><th>Frequency</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{m.item}</td>
                  <td className="small">{m.frequency}</td>
                  <td className="small text-muted">{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Disease Lifecycle — KCNC2 GOF-LOF Spectrum" borderColor={ACCENT5}>
        {(lifecycle || []).map((l, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ backgroundColor: '#e0f2f1', border: `1px solid ${ACCENT5}20` }}>
            <strong style={{ color: ACCENT5 }}>{l.stage}</strong>
            <div className="small mt-1 text-muted">{l.description}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { concepts, thresholds, standards, references } = data;
  return (
    <div>
      <SectionCard title="Key Concepts (15)" borderColor={ACCENT}>
        {(concepts || []).map((c, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ backgroundColor: '#f3f0ff', border: `1px solid ${ACCENT}20` }}>
            <strong style={{ color: ACCENT }}>{c.term}</strong>
            <div className="small text-muted mt-1">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead style={{ backgroundColor: ACCENT3 + '15' }}>
              <tr><th>Parameter</th><th>Normal</th><th>Action Threshold</th><th>Action</th></tr>
            </thead>
            <tbody>
              {(thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{t.parameter}</td>
                  <td className="small text-muted">{t.normal}</td>
                  <td className="small" style={{ color: ACCENT3 }}>{t.action_threshold}</td>
                  <td className="small">{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Clinical Standards" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead style={{ backgroundColor: ACCENT + '15' }}>
              <tr><th>Standard</th><th>Relevance</th></tr>
            </thead>
            <tbody>
              {(standards || []).map((s, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{s.standard}</td>
                  <td className="small text-muted">{s.relevance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="References" borderColor={ACCENT}>
        <ol className="small mb-0">
          {(references || []).map((r, i) => (
            <li key={i} className="mb-1">{r.citation}</li>
          ))}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function KCNC2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/kcnc2/overview`).then(r => r.json()),
      fetch(`${API}/api/kcnc2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/kcnc2/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(String(e)));
  }, []);

  if (error) return <div className="alert alert-danger m-4">{error}</div>;

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4 p-4 rounded text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT5} 100%)` }}>
        <h2 className="mb-1">🧬 KCNC2 Epilepsy — GGE / Focal / DEE (Kv3.2 Shaw K⁺)</h2>
        <div className="opacity-90 small">
          Kv3.2 Shaw K⁺ Channel · GOF-LOF Dual Phenotype · PV+ Fast-Spiking Interneuron + TRN · 12q21.32
          &nbsp;·&nbsp;OMIM *176262&nbsp;·&nbsp;GGE / Focal Epilepsy / DEE — NO progressive myoclonic epilepsy
          &nbsp;·&nbsp;<strong>CLASSIFY GOF vs LOF before AED</strong>
          &nbsp;·&nbsp;CBZ/LTG HIGH RISK in LOF (NOT absolute CI — distinct from KCNC1/EPM7)
          &nbsp;·&nbsp;TGB ABSOLUTE CI · POLG1 before VPA · VPPP females ≥12y
          &nbsp;·&nbsp;Photosensitivity 35% (vs KCNC1 EPM7 75%) · No Piracetam · No Giant SEPs
        </div>
      </div>

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { borderBottomColor: ACCENT, color: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
