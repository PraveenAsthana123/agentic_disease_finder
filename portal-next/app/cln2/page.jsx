'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4a148c';   // deep violet-purple — CLN2 / NCL lysosomal storage disorder
const ACCENT2 = '#b71c1c';   // dark red — ABSOLUTE CI / danger / fatal disease
const ACCENT3 = '#e65100';   // deep orange — urgent alerts / triggers / warnings
const ACCENT4 = '#1565c0';   // deep blue — safe treatments / cerliponase

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#f3e5f5', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

// ── TAB 1: Overview ──────────────────────────────────────────────────────────
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Alert
        variant="danger"
        text="⛔ ABSOLUTE CI: Vigabatrin / VGB (irreversible retinal toxicity — CLN2 has progressive retinal degeneration; VGB superimposes vigabatrin-associated retinopathy → catastrophic visual loss). DELAY IN CERLIPONASE ALFA = clinical urgency — start within 4-6 weeks of diagnosis. IV LEV 60 mg/kg = SE rescue (not IV fosphenytoin)."
      />
      <Alert
        variant="warning"
        text="⚠ HIGH RISK: CBZ / OXC (may worsen myoclonic component). GBP / Pregabalin (α2δ — myoclonus worsening; Crespel 1999). Fosphenytoin IV (avoid — LEV is first-line SE rescue)."
      />
      <Alert
        variant="success"
        text="✅ ONLY NCL WITH FDA/EMA-APPROVED TREATMENT: Cerliponase alfa (Brineura, BioMarin) — intracerebroventricular (ICV) enzyme replacement therapy — FDA-approved April 2017. Start immediately after diagnosis. Every week of delay = irreversible neuronal loss. VPA IS SAFE in CLN2 (lysosomal disorder, NOT mitochondrial — contrast MERRF/POLG where VPA is ABSOLUTE CI)."
      />
      <Alert
        variant="info"
        text="🔵 CLN2 DIAGNOSTIC PRIORITY: Giant SSPS occipital response at 1-3 Hz = PATHOGNOMONIC EEG finding. TPP1 enzyme assay (DBS/leukocyte) gives diagnosis in DAYS — must be FIRST test when curvilinear/fingerprint EM bodies seen (before KCTD7/EPM3 WES). CLN2 is FATAL without treatment — earlier cerliponase = more neurons preserved."
      />

      <div className="row mb-4">
        <KPI label="Gene / Locus" value="CLN2/TPP1 / 11p15.4" color={ACCENT} />
        <KPI label="Protein" value="TPP1 Lysosomal Protease" color={ACCENT} />
        <KPI label="Inheritance" value="Autosomal Recessive" color={ACCENT} />
        <KPI label="Cohort" value={`${ov.cohort_size} patients`} color={ACCENT} />
        <KPI label="Mean Onset" value={`${ov.mean_onset_years}y`} color={ACCENT2} />
        <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
      </div>

      <SectionCard title="Molecular Mechanism — Lysosomal TPP1 Deficiency & SCMAS Accumulation">
        <p style={{ fontSize: 14 }}>{ov.mechanism}</p>
        <div className="row mt-3">
          <div className="col-md-6">
            <PctBar label="Cognitive Regression (100%)" pct={ov.cognitive_regression_pct} color={ACCENT2} />
            <PctBar label="Retinal Involvement" pct={ov.retinal_involvement_pct} color={ACCENT2} />
            <PctBar label="Giant SSPS 1-3Hz Positive" pct={ov.giant_ssps_positive_pct} color={ACCENT} />
            <PctBar label="EM Fingerprint Profile Positive" pct={ov.eem_fingerprint_positive_pct} color={ACCENT} />
          </div>
          <div className="col-md-6">
            <PctBar label="Photosensitivity (IPS+)" pct={ov.photosensitivity_pct} color={ACCENT3} />
            <PctBar label="NCSE Events" pct={ov.ncse_pct} color={ACCENT3} />
            <PctBar label="On Cerliponase Alfa" pct={ov.on_cerliponase_alfa_pct} color={ACCENT4} />
            <PctBar label="On VPA (Safe — Lysosomal Not Mitochondrial)" pct={ov.on_vpa_pct} color={ACCENT4} />
          </div>
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Cerliponase Alfa — ICV Enzyme Replacement" borderColor={ACCENT4}>
            <p style={{ fontSize: 13 }}><strong>FDA-approved April 2017 · EMA-approved May 2017 · FIRST-EVER NCL treatment</strong></p>
            <p style={{ fontSize: 13 }}>{ov.approved_treatment}</p>
            <div className="alert alert-success py-1 mt-2" style={{ fontSize: 12 }}>
              ✅ Pivotal trial (Schulz 2018 NEJM N=23): ~2× slower CLN2-CRS motor+language decline vs. natural history. Earlier start = better outcome.
            </div>
            <PctBar label="Cerliponase within 6mo diagnosis" pct={ov.cerliponase_started_within_6mo_diagnosis_pct} color={ACCENT4} />
            <p className="mt-2 text-muted" style={{ fontSize: 12 }}>Mean diagnostic delay: <strong>{ov.mean_delay_diagnosis_years} years</strong> — reducing this delay is the most impactful intervention.</p>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Ambulation Outcome — Natural History vs. Treated" borderColor={ACCENT3}>
            <PctBar label="Ambulatory at diagnosis (all)" pct={ov.ambulatory_at_diagnosis_pct} color={ACCENT4} />
            <PctBar label="Ambulatory 5y after diagnosis — UNTREATED" pct={ov.ambulatory_5y_after_diagnosis_natural_pct} color={ACCENT2} />
            <PctBar label="Ambulatory 5y after diagnosis — CERLIPONASE" pct={ov.ambulatory_5y_after_diagnosis_treated_pct} color={ACCENT4} />
            <div className="alert alert-warning py-1 mt-2" style={{ fontSize: 12 }}>
              ⚠ Without cerliponase: 78% lose ambulation within 5y of diagnosis. With treatment: 65% maintain ambulation at 5y.
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="AED Profile — ABSOLUTE & HIGH-RISK Contraindications">
        <div className="row">
          <div className="col-md-6">
            <h6 className="fw-bold text-danger">ABSOLUTE CI</h6>
            {(ov.absolute_ci || []).map((ci, i) => (
              <div key={i} className="alert alert-danger py-1 mb-2" style={{ fontSize: 12 }}>⛔ {ci}</div>
            ))}
          </div>
          <div className="col-md-6">
            <h6 className="fw-bold text-warning">HIGH RISK</h6>
            {(ov.high_risk_ci || []).map((ci, i) => (
              <div key={i} className="alert alert-warning py-1 mb-2" style={{ fontSize: 12 }}>⚠ {ci}</div>
            ))}
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Discovery & Key References" borderColor="#666">
        <p style={{ fontSize: 13 }}><strong>Gene discovery:</strong> {ov.discovery}</p>
        <p style={{ fontSize: 13 }}><strong>Unique feature:</strong> {ov.unique_feature}</p>
      </SectionCard>
    </div>
  );
}

// ── TAB 2: Patients & Etiology ───────────────────────────────────────────────
function PatientsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="Etiology Classes — 6 Genotypic Categories (40 patients)">
        {(bk.etiologies || []).map((e, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong style={{ color: ACCENT }}>{e.class.replace(/-/g, ' ')}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{e.pct}% (n={e.count})</span>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT }} />
            </div>
            <p style={{ fontSize: 13 }} className="mb-1">{e.description}</p>
            <p style={{ fontSize: 12 }} className="text-muted mb-1"><strong>Mechanism:</strong> {e.gene_mechanism}</p>
            <p style={{ fontSize: 12 }} className="text-muted mb-0">
              <strong>Key variants:</strong> {(e.key_variants || []).join(' · ')}
            </p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Lifecycle Stages — CLN2 Disease Trajectory">
        {(bk.lifecycle || []).map((l, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ color: ACCENT }}>{l.stage.replace(/-/g, ' ')}</strong>
              <span className="badge bg-secondary">{l.age}</span>
            </div>
            <p style={{ fontSize: 13 }} className="mb-0">{l.description}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 3: Seizures & Triggers ───────────────────────────────────────────────
function SeizuresTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="Seizure Types — CLN2 Multi-Seizure Disorder">
        {(bk.seizure_types || []).map((s, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong style={{ color: ACCENT }}>{s.type.replace(/-/g, ' ')}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{s.pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${s.pct}%`, backgroundColor: ACCENT }} />
            </div>
            <p style={{ fontSize: 13 }} className="mb-1">{s.description}</p>
            {s.eeg && <p style={{ fontSize: 12 }} className="text-info mb-1"><strong>EEG:</strong> {s.eeg}</p>}
            {s.semiology && <p style={{ fontSize: 12 }} className="text-muted mb-1"><strong>Semiology:</strong> {s.semiology}</p>}
            {s.clinical_tip && (
              <div className="alert alert-warning py-1 mt-1" style={{ fontSize: 12 }}>
                💡 {s.clinical_tip}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers — Management Strategies">
        {(bk.triggers || []).map((t, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong style={{ color: ACCENT3 }}>{t.trigger.replace(/-/g, ' ')}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: ACCENT3 }} />
            </div>
            <p style={{ fontSize: 13 }} className="mb-1">{t.description}</p>
            {t.management && (
              <p style={{ fontSize: 12 }} className="text-success mb-0"><strong>Management:</strong> {t.management}</p>
            )}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 4: Treatments ────────────────────────────────────────────────────────
function TreatmentsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const levelColor = (lv) => {
    if (lv?.includes('A')) return '#1b5e20';
    if (lv?.includes('B')) return ACCENT4;
    return '#5d4037';
  };
  return (
    <div>
      <SectionCard title="Treatments — Evidence Levels">
        {(bk.treatments || []).map((t, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong style={{ color: ACCENT4 }}>{t.drug}</strong>
              <span className="badge" style={{ backgroundColor: levelColor(t.level) }}>{t.level}</span>
            </div>
            <p style={{ fontSize: 12 }} className="text-muted mb-1"><strong>Dose:</strong> {t.dose}</p>
            <p style={{ fontSize: 13 }} className="mb-1"><strong>Mechanism:</strong> {t.moa}</p>
            <p style={{ fontSize: 13 }} className="mb-1"><strong>Efficacy:</strong> {t.efficacy}</p>
            <p style={{ fontSize: 12 }} className="text-muted mb-1"><strong>Monitoring:</strong> {t.monitoring}</p>
            {t.cln2_note && (
              <div className="alert alert-info py-1 mt-1" style={{ fontSize: 12 }}>
                🔵 <strong>CLN2 note:</strong> {t.cln2_note}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications — Absolute & High-Risk" borderColor={ACCENT2}>
        {(bk.contraindications || []).map((c, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong style={{ color: ACCENT2 }}>{c.drug.replace(/-/g, ' ')}</strong>
              <span className={`badge ${c.severity === 'ABSOLUTE' ? 'bg-danger' : 'bg-warning text-dark'}`}>
                {c.severity}
              </span>
            </div>
            <p style={{ fontSize: 13 }} className="mb-1">{c.reason}</p>
            {c.alternative && (
              <p style={{ fontSize: 12 }} className="text-success mb-0"><strong>Alternative:</strong> {c.alternative}</p>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring — 14 Essential Parameters" borderColor="#5d4037">
        <div className="row">
          {(bk.monitoring || []).map((m, i) => (
            <div key={i} className="col-md-6 mb-2">
              <div className="p-2 border rounded h-100">
                <strong style={{ fontSize: 12, color: ACCENT }}>{m.item.replace(/-/g, ' ')}</strong>
                <p style={{ fontSize: 11 }} className="text-muted mb-0 mt-1">{m.rationale}</p>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

// ── TAB 5: Definitions ───────────────────────────────────────────────────────
function DefinitionsTab({ df }) {
  if (!df) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="Core Concepts — 15 CLN2 Clinical Principles">
        {(df.concepts || []).map((c, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <strong style={{ color: ACCENT, fontSize: 13 }}>{c.concept.replace(/-/g, ' ')}</strong>
            <p style={{ fontSize: 13 }} className="mt-2 mb-1">{c.definition}</p>
            <p style={{ fontSize: 11 }} className="text-muted mb-0"><strong>Standards:</strong> {c.standard}</p>
          </div>
        ))}
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Thresholds" borderColor={ACCENT3}>
            {(df.thresholds || []).map((t, i) => (
              <div key={i} className="mb-2 p-2 border rounded">
                <strong style={{ fontSize: 12 }}>{t.threshold}</strong>
                <p style={{ fontSize: 11 }} className="text-muted mb-0">{t.standard}</p>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Standards & Guidelines" borderColor={ACCENT4}>
            {(df.standards || []).map((s, i) => (
              <div key={i} className="mb-2 p-2 border rounded">
                <strong style={{ fontSize: 12 }}>{s.standard}</strong>
                <p style={{ fontSize: 11 }} className="text-muted mb-0">{s.detail}</p>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="References" borderColor="#666">
        {(df.references || []).map((r, i) => (
          <div key={i} className="mb-2 p-2 border rounded">
            <strong style={{ fontSize: 12 }}>{r.ref}</strong>
            <p style={{ fontSize: 12 }} className="text-muted mb-0">{r.citation}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Root ─────────────────────────────────────────────────────────────────────
export default function CLN2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/cln2/overview`).then(r => r.json()).then(setOverview).catch(() => setErr('Overview fetch failed'));
    fetch(`${API}/api/cln2/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setErr('Breakdown fetch failed'));
    fetch(`${API}/api/cln2/definitions`).then(r => r.json()).then(setDefinitions).catch(() => setErr('Definitions fetch failed'));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3" style={{ borderBottom: `3px solid ${ACCENT}`, paddingBottom: 8 }}>
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          CLN2 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 2
        </h4>
        <div className="text-muted small">
          Late-Infantile Batten Disease · TPP1 / CLN2 (11p15.4) · Lysosomal Serine Protease · SCMAS Storage ·
          Giant SSPS 1-3Hz Pathognomonic · Cerliponase Alfa FDA 2017 (ONLY NCL ERT) ·
          VGB ABSOLUTE CI (Retinal) · VPA SAFE (Lysosomal — NOT Mitochondrial) ·
          ICV Ommaya Every 2 Weeks · Fatal Without Treatment
        </div>
      </div>

      {err && <div className="alert alert-danger">{err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      <div>
        {tab === 0 && <OverviewTab ov={overview} />}
        {tab === 1 && <PatientsTab bk={breakdown} />}
        {tab === 2 && <SeizuresTab bk={breakdown} />}
        {tab === 3 && <TreatmentsTab bk={breakdown} />}
        {tab === 4 && <DefinitionsTab df={definitions} />}
      </div>
    </div>
  );
}
