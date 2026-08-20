'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments & CIs', 'Definitions'];

const ACCENT  = '#e65100';   // deep orange — GABA catabolism / SSA / GHB accumulation / mitochondrial
const ACCENT2 = '#b71c1c';   // dark red — VGB ABSOLUTE CI / GHB toxic / drug hazard
const ACCENT3 = '#f57f17';   // amber — GHB elevated / SSA / relative CI / caution
const ACCENT4 = '#1565c0';   // deep blue — safe treatments / LEV / VPA / KD
const ACCENT5 = '#1b5e20';   // dark green — KD response / seizure free / normal range
const ACCENT6 = '#4a148c';   // deep purple — GABA pathway / GABA-B / GHB receptor / MRS

function KPI({ label, value, color = ACCENT, sub }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function PctBar({ label, pct, color = ACCENT, extra }) {
  const numPct = typeof pct === 'string' ? parseFloat(pct) : (pct ?? 0);
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}{extra && <span className="text-muted ms-1">{extra}</span>}</span>
        <span className="text-muted">{numPct.toFixed(0)}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${Math.min(numPct, 100)}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, color: '#fff', fontSize: 11 }}>{text}</span>
  );
}

function CICard({ drug, level, reason, ssadh_note }) {
  const color = level?.includes('ABSOLUTE') ? ACCENT2
    : level?.includes('HIGH RISK') || level?.includes('RELATIVE CI') ? ACCENT3
    : ACCENT4;
  return (
    <div className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold small">{drug}</span>
          <Badge text={level?.split(' — ')[0]} color={color} />
        </div>
        <div className="small text-muted mb-1">{reason}</div>
        {ssadh_note && (
          <div className="small" style={{ color: ACCENT6, fontStyle: 'italic' }}>SSADH: {ssadh_note}</div>
        )}
      </div>
    </div>
  );
}

// ─── TAB: Overview ──────────────────────────────────────────────────────────
function TabOverview({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  return (
    <>
      <div className="alert alert-danger py-2 mb-3" style={{ fontSize: 13 }}>
        <strong>VGB (Vigabatrin) ABSOLUTE CONTRAINDICATION — UNIQUE METABOLIC MECHANISM:</strong>{' '}
        VGB inhibits GABA-T → MORE succinic semialdehyde (SSA) accumulates → SSAR diverts MORE SSA to GHB →
        acute seizure worsening within 48–72h. This is NOT the retinal toxicity CI (different from NCL diseases).
        Medical alert MANDATORY. Emergency note: "VGB ABSOLUTE CI — SSADH deficiency."
      </div>
      <div className="alert alert-warning py-2 mb-3" style={{ fontSize: 13 }}>
        <strong>Pathognomonic Biomarker:</strong> Urine GHB (4-hydroxybutyric acid) by GC-MS — 100–1000 mmol/mol creatinine
        (10–100× normal). Brain MRS GHB peak at 2.41 ppm (in vivo, non-invasive).
        MRI: bilateral globus pallidus T2 hyperintensity (80%).
      </div>

      <div className="row mb-2">
        <KPI label="Patients" value={ov.cohort_size} color={ACCENT} />
        <KPI label="Epilepsy" value={`${ov.n_epilepsy} (${ov.epilepsy_pct}%)`} color={ACCENT3} sub="~50% overall" />
        <KPI label="Drug-Resistant" value={`${ov.n_dre} (${ov.dre_pct}%)`} color={ACCENT2} sub="of those w/ epilepsy" />
        <KPI label="Ataxia" value={`${ov.n_ataxia} (${ov.ataxia_pct}%)`} color={ACCENT3} />
        <KPI label="ASD Features" value={`${ov.n_asd} (${ov.asd_pct}%)`} color={ACCENT6} sub="50% ASD" />
        <KPI label="ADHD-Like" value={`${ov.n_adhd_like} (${ov.adhd_like_pct}%)`} color={ACCENT6} sub="75%" />
      </div>
      <div className="row mb-2">
        <KPI label="On Ketogenic Diet" value={`${ov.n_on_kd} (${ov.on_kd_pct}%)`} color={ACCENT5} sub="precision Rx" />
        <KPI label="POLG1 Screened" value={`${ov.n_polg1_screened} (${ov.polg1_screened_pct}%)`} color={ACCENT4} sub="pre-VPA CPIC-A" />
        <KPI label="Avg Urine GHB" value={`${ov.avg_urine_ghb}`} color={ACCENT2} sub="mmol/mol Cr" />
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Feature Distribution" borderColor={ACCENT3}>
            <PctBar label="Intellectual Disability (ALL)" pct={100} color={ACCENT} />
            <PctBar label="Hypotonia" pct={70} color={ACCENT3} />
            <PctBar label="ADHD-Like Behaviour" pct={75} color={ACCENT6} />
            <PctBar label="Cerebellar Ataxia" pct={60} color={ACCENT3} />
            <PctBar label="Epilepsy" pct={50} color={ACCENT2} />
            <PctBar label="ASD Features" pct={50} color={ACCENT6} />
            <PctBar label="Sleep Disturbance" pct={65} color={ACCENT6} />
            <PctBar label="OCD Traits" pct={40} color={ACCENT6} />
            <PctBar label="Aggression/SIB" pct={35} color={ACCENT2} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Etiology Distribution" borderColor={ACCENT}>
            {(ov.etiologies || []).map((e, i) => (
              <PctBar key={i} label={e.etiology} pct={e.pct} color={ACCENT} extra={`(n=${e.n})`} />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Top Patients by Urine GHB Burden" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0" style={{ fontSize: 12 }}>
            <thead className="table-dark">
              <tr>
                <th>Patient</th><th>Sex</th><th>Onset</th><th>Urine GHB</th>
                <th>Epilepsy</th><th>DRE</th><th>KD</th><th>POLG1 Screened</th>
              </tr>
            </thead>
            <tbody>
              {(ov.per_patient_kpis || []).slice(0, 10).map((p, i) => (
                <tr key={i}>
                  <td className="fw-bold">{p.patient_id}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_age_months} mo</td>
                  <td>
                    <span className="fw-bold" style={{ color: p.urine_ghb_mmol_cr > 500 ? ACCENT2 : ACCENT3 }}>
                      {p.urine_ghb_mmol_cr.toFixed(0)}
                    </span>
                    <span className="text-muted ms-1" style={{ fontSize: 10 }}>mmol/mol Cr</span>
                  </td>
                  <td>{p.has_epilepsy ? <Badge text="Yes" color={ACCENT3} /> : <span className="text-muted small">No</span>}</td>
                  <td>{p.drug_resistant ? <Badge text="DRE" color={ACCENT2} /> : <span className="text-muted small">—</span>}</td>
                  <td>{p.on_kd ? <Badge text="KD" color={ACCENT5} /> : <span className="text-muted small">—</span>}</td>
                  <td>
                    {p.polg1_screened
                      ? <Badge text="✓ Screened" color={ACCENT4} />
                      : <Badge text="⚠ Not Done" color={ACCENT2} />}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ─── TAB: Patients & Etiology ────────────────────────────────────────────────
function TabPatients({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  return (
    <>
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="ID Severity Distribution" borderColor={ACCENT}>
            {(bk.id_severity_distribution || []).map((d, i) => (
              <PctBar key={i} label={`${d.severity} ID`} pct={d.pct} color={ACCENT} extra={`(n=${d.n})`} />
            ))}
          </SectionCard>
          <SectionCard title="Urine GHB Distribution (mmol/mol Cr)" borderColor={ACCENT2}>
            {(bk.urine_ghb_histogram || []).map((b, i) => (
              <PctBar key={i} label={b.range} pct={b.pct} color={ACCENT2} extra={`(n=${b.n})`} />
            ))}
            <div className="small text-muted mt-2">Normal: &lt;1 mmol/mol Cr. SSADH: 100–1000×.</div>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Age of Onset Distribution" borderColor={ACCENT3}>
            {(bk.onset_age_histogram || []).map((b, i) => (
              <PctBar key={i} label={b.range} pct={b.pct} color={ACCENT3} extra={`(n=${b.n})`} />
            ))}
            <div className="small text-muted mt-2">
              Language delay is usually the first red flag (18–24 mo vs 12 mo normal).
            </div>
          </SectionCard>
          <SectionCard title="Treatment Coverage (cohort)" borderColor={ACCENT4}>
            {Object.entries(bk.treatment_counts || {}).map(([drug, n], i) => (
              <PctBar
                key={i}
                label={drug}
                pct={Math.round(100 * n / 40)}
                color={drug === 'KD' ? ACCENT5 : ACCENT4}
                extra={`(n=${n})`}
              />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Lifecycle Stages" borderColor={ACCENT6}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
            <thead className="table-dark">
              <tr><th>Stage</th><th>Age</th><th>Description</th></tr>
            </thead>
            <tbody>
              {(bk.lifecycle || []).map((s, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ color: ACCENT6 }}>{s.stage}</td>
                  <td className="text-muted small">{s.age}</td>
                  <td className="small">{s.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ─── TAB: Seizures & Triggers ────────────────────────────────────────────────
function TabSeizures({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  return (
    <>
      <SectionCard title="Seizure Types in SSADH Deficiency" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
            <thead className="table-dark">
              <tr><th>Seizure Type</th><th>%</th><th>EEG</th><th>Semiology</th><th>Tips</th></tr>
            </thead>
            <tbody>
              {(bk.seizure_type_distribution || []).map((s, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ color: ACCENT2 }}>{s.type}</td>
                  <td><Badge text={`${s.pct}%`} color={ACCENT2} /></td>
                  <td className="small">{s.eeg}</td>
                  <td className="small">{s.semiology}</td>
                  <td className="small" style={{ color: ACCENT6 }}>{s.tips}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Seizure Triggers" borderColor={ACCENT3}>
        {(bk.trigger_distribution || []).map((t, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="fw-bold small">{t.trigger}</span>
              <Badge
                text={t.pct === 100 ? '100% — ABSOLUTE CI' : `${t.pct}%`}
                color={t.pct === 100 ? ACCENT2 : ACCENT3}
              />
            </div>
            <div className="small text-muted">{t.mechanism}</div>
            {t.pct === 100 && (
              <div className="alert alert-danger py-1 mt-1" style={{ fontSize: 11 }}>
                ABSOLUTE CONTRAINDICATION — Inform all prescribers; medical alert mandatory.
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Protocol" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0" style={{ fontSize: 11 }}>
            <thead className="table-light">
              <tr><th>Parameter</th><th>Frequency</th><th>Target</th></tr>
            </thead>
            <tbody>
              {(bk.monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold small">{m.parameter}</td>
                  <td className="text-muted small">{m.frequency}</td>
                  <td className="small">{m.target}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ─── TAB: Treatments & CIs ──────────────────────────────────────────────────
function TabTreatments({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;

  const treatments = (bk.treatments || []).filter(t => !t.level?.includes('ABSOLUTE CI'));
  const vgb = (bk.treatments || []).find(t => t.drug?.includes('Vigabatrin'));

  return (
    <>
      <div className="alert alert-danger py-2 mb-3" style={{ fontSize: 13 }}>
        <strong>VGB (VIGABATRIN) ABSOLUTE CONTRAINDICATION IN SSADH:</strong>{' '}
        Mechanism: VGB → GABA-T block → MORE SSA → SSAR diverts MORE SSA → GHB surge.
        Acute seizure worsening within 48–72h. NOT retinal CI. Different from NCL/retinal disease VGB CI.
        {vgb && <div className="mt-1 small">{vgb.ssadh_note}</div>}
      </div>

      {treatments.map((t, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{
          borderLeft: `4px solid ${t.level?.includes('Level A') ? ACCENT4 : t.level?.includes('Level B') ? ACCENT5 : ACCENT3}`
        }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-2">
              <h6 className="fw-bold mb-0">{t.drug}</h6>
              <Badge
                text={t.level?.split(' — ')[0]}
                color={t.level?.includes('Level A') ? ACCENT4 : t.level?.includes('Level B') ? ACCENT5 : ACCENT3}
              />
            </div>
            <div className="table-responsive">
              <table className="table table-sm table-bordered mb-2" style={{ fontSize: 12 }}>
                <tbody>
                  <tr><th style={{ width: '15%' }}>Dose</th><td>{t.dose}</td></tr>
                  <tr><th>MOA</th><td>{t.moa}</td></tr>
                  <tr><th>Efficacy</th><td>{t.efficacy}</td></tr>
                  <tr><th>Monitoring</th><td>{t.monitoring}</td></tr>
                </tbody>
              </table>
            </div>
            <div className="small" style={{ color: ACCENT6, fontStyle: 'italic' }}>
              <strong>SSADH note:</strong> {t.ssadh_note}
            </div>
          </div>
        </div>
      ))}

      <SectionCard title="Contraindications" borderColor={ACCENT2}>
        {(bk.contraindications || []).map((c, i) => (
          <CICard key={i} drug={c.drug} level={c.level} reason={c.reason} />
        ))}
      </SectionCard>
    </>
  );
}

// ─── TAB: Definitions ────────────────────────────────────────────────────────
function TabDefinitions({ def }) {
  if (!def) return <div className="text-muted">Loading…</div>;

  return (
    <>
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>{def.title}</h6>

      {def.gene_card && (
        <SectionCard title="Gene / Protein Card" borderColor={ACCENT}>
          <div className="table-responsive">
            <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
              <tbody>
                {Object.entries(def.gene_card).map(([k, v], i) => (
                  <tr key={i}><th style={{ width: '25%' }}>{k}</th><td>{v}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {def.pathway && (
        <SectionCard title={`Pathway: ${def.pathway.name}`} borderColor={ACCENT6}>
          <div className="table-responsive">
            <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
              <thead className="table-dark">
                <tr><th>Step</th><th>Enzyme</th><th>Gene</th><th>Reaction</th><th>Clinical Note</th></tr>
              </thead>
              <tbody>
                {(def.pathway.steps || []).map((s, i) => (
                  <tr key={i} style={{ background: s.gene === 'ALDH5A1' ? '#fff3e0' : undefined }}>
                    <td className="text-center fw-bold" style={{ color: ACCENT6 }}>{s.step}</td>
                    <td className="fw-bold" style={{ color: s.gene === 'ALDH5A1' ? ACCENT : undefined }}>
                      {s.enzyme}{s.gene === 'ALDH5A1' && <Badge text="THIS ENZYME" color={ACCENT} />}
                    </td>
                    <td><code>{s.gene}</code></td>
                    <td className="small">{s.reaction}</td>
                    <td className="small" style={{ color: ACCENT6 }}>{s.clinical}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {def.biomarkers && (
        <SectionCard title="Biomarkers" borderColor={ACCENT2}>
          <div className="table-responsive">
            <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
              <thead className="table-light">
                <tr><th>Biomarker</th><th>Method</th><th>Normal</th><th>SSADH Range</th><th>Notes</th></tr>
              </thead>
              <tbody>
                {def.biomarkers.map((b, i) => (
                  <tr key={i}>
                    <td className="fw-bold">{b.marker}</td>
                    <td className="small">{b.method}</td>
                    <td className="small text-success">{b.reference_range}</td>
                    <td className="small fw-bold" style={{ color: ACCENT2 }}>{b.ssadh_range}</td>
                    <td className="small">{b.notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      <SectionCard title="Key Concepts" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
            <thead className="table-light">
              <tr><th style={{ width: '30%' }}>Term</th><th>Definition</th></tr>
            </thead>
            <tbody>
              {(def.key_concepts || []).map((c, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ color: ACCENT }}>{c.term}</td>
                  <td className="small">{c.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {def.differential_diagnosis && (
        <SectionCard title="Differential Diagnosis" borderColor={ACCENT3}>
          <div className="table-responsive">
            <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
              <thead className="table-light">
                <tr><th>Condition</th><th>Key Distinction from SSADH</th></tr>
              </thead>
              <tbody>
                {def.differential_diagnosis.map((d, i) => (
                  <tr key={i}>
                    <td className="fw-bold">{d.condition}</td>
                    <td className="small">{d.distinction}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {def.thresholds && (
        <SectionCard title="Clinical Thresholds" borderColor={ACCENT5}>
          <div className="table-responsive">
            <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
              <thead className="table-light">
                <tr><th>Parameter</th><th>Value</th><th>Clinical Implication</th></tr>
              </thead>
              <tbody>
                {def.thresholds.map((t, i) => (
                  <tr key={i}>
                    <td className="fw-bold small">{t.parameter}</td>
                    <td className="fw-bold" style={{ color: ACCENT2 }}>{t.value}</td>
                    <td className="small">{t.clinical}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {def.references && (
        <SectionCard title="References" borderColor={ACCENT6}>
          <ol className="mb-0 small">
            {def.references.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
          </ol>
        </SectionCard>
      )}
    </>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────
export default function ALDH5A1Page() {
  const [tab, setTab] = useState(0);
  const [ov,  setOv]  = useState(null);
  const [bk,  setBk]  = useState(null);
  const [def, setDef] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    const base = `${API}/api/aldh5a1`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDef(d); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-3 gap-3">
        <div style={{ fontSize: 36 }}>🧬</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>ALDH5A1 Epilepsy — SSADH Deficiency</h4>
          <div className="text-muted small">
            Succinic Semialdehyde Dehydrogenase Deficiency · 4-Hydroxybutyric Aciduria ·
            GABA Catabolism Final Step · AR Biallelic LOF · 6p22.3
          </div>
          <div className="mt-1">
            <Badge text="GABA Catabolism" color={ACCENT6} />
            <Badge text="GHB Accumulation" color={ACCENT2} />
            <Badge text="VGB ABSOLUTE CI" color={ACCENT2} />
            <Badge text="KD Precision Rx" color={ACCENT5} />
            <Badge text="Globus Pallidus T2" color={ACCENT} />
            <Badge text="Brain MRS GHB 2.41ppm" color={ACCENT3} />
          </div>
        </div>
      </div>

      {err && <div className="alert alert-danger py-2">{err}</div>}

      {/* Summary banner */}
      <div className="row mb-3">
        <div className="col-md-3">
          <div className="card shadow-sm text-center p-2" style={{ borderTop: `3px solid ${ACCENT}` }}>
            <div className="fw-bold" style={{ color: ACCENT, fontSize: 22 }}>{ov?.cohort_size ?? '…'}</div>
            <div className="text-muted small">Patients · AR 6p22.3</div>
          </div>
        </div>
        <div className="col-md-3">
          <div className="card shadow-sm text-center p-2" style={{ borderTop: `3px solid ${ACCENT2}` }}>
            <div className="fw-bold" style={{ color: ACCENT2, fontSize: 22 }}>{ov?.avg_urine_ghb ?? '…'}</div>
            <div className="text-muted small">Avg Urine GHB (mmol/mol Cr)</div>
          </div>
        </div>
        <div className="col-md-3">
          <div className="card shadow-sm text-center p-2" style={{ borderTop: `3px solid ${ACCENT3}` }}>
            <div className="fw-bold" style={{ color: ACCENT3, fontSize: 22 }}>{ov?.epilepsy_pct ?? '…'}%</div>
            <div className="text-muted small">Epilepsy (all types)</div>
          </div>
        </div>
        <div className="col-md-3">
          <div className="card shadow-sm text-center p-2" style={{ borderTop: `3px solid ${ACCENT5}` }}>
            <div className="fw-bold" style={{ color: ACCENT5, fontSize: 22 }}>{ov?.on_kd_pct ?? '…'}%</div>
            <div className="text-muted small">On Ketogenic Diet</div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              onClick={() => setTab(i)}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <TabOverview ov={ov} />}
      {tab === 1 && <TabPatients bk={bk} />}
      {tab === 2 && <TabSeizures bk={bk} />}
      {tab === 3 && <TabTreatments bk={bk} />}
      {tab === 4 && <TabDefinitions def={def} />}
    </div>
  );
}
