'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#880e4f';   // deep magenta-burgundy — MERRF mitochondrial disease
const ACCENT2 = '#b71c1c';   // deep red — ABSOLUTE CI
const ACCENT3 = '#e65100';   // deep orange — warnings / high risk
const ACCENT4 = '#1b5e20';   // deep forest green — safe / mitochondrial cofactors

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

// ── TAB 1: Overview ──────────────────────────────────────────────────────────
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Alert
        text="🚨 VPA ABSOLUTE CI — MOST CRITICAL MERRF DISTINCTION: Valproate NEVER in MERRF (mitochondrial hepatotoxicity / Alpers-like). UNLIKE ALL OTHER PMEs where VPA is backbone. IV LEV 60 mg/kg = SOLE SE rescue. Document in ED, anaesthesia, and allergy alert."
        variant="danger"
      />
      <Alert
        text="⚡ ABSOLUTE CIs: CBZ / OXC (Na-channel + myoclonus) · PHT / Fosphenytoin-IV (Na-channel + Complex I inhibition) · TGB (NCSE + Purkinje loss) · HIGH RISK: GBP/PGB (α2δ myoclonic worsening) · Statins (CoQ10 depletion → OXPHOS worsening)"
        variant="danger"
      />
      <Alert
        text="🧬 MT-TK (mtDNA 8344) — m.8344A>G → defective mt-tRNA-Lys → Complex I/IV deficiency → ATP failure. Maternal inheritance (heteroplasmy). Ragged-red fibers (Gomori trichrome) + SNHL (90%) = first clue. Standard Rx: LEV + CLB + Piracetam + CoQ10 + Riboflavin + L-Carnitine."
        variant="info"
      />
      <Alert
        text="❤️ WPW / HCM in ~40% MERRF — annual ECG mandatory; Holter if WPW; EP study + ablation if symptomatic WPW (sudden cardiac death risk). m.8363G>A variant: highest cardiac risk."
        variant="warning"
      />

      <SectionCard title="Gene & Protein Summary" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Gene:</strong> {ov.gene}</p>
        <p className="small mb-1"><strong>Protein:</strong> {ov.protein}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {ov.inheritance}</p>
        <p className="small mb-1"><strong>OMIM:</strong> {ov.omim}</p>
        <p className="small mb-1"><strong>Disease:</strong> {ov.disease}</p>
        <p className="small mb-0"><strong>Mechanism:</strong> {ov.mechanism}</p>
      </SectionCard>

      <SectionCard title="Cohort Statistics" borderColor={ACCENT}>
        <div className="row">
          <KPI label="Patients" value={ov.cohort_size} color={ACCENT} />
          <KPI label="Female" value={`${ov.female_pct}%`} color={ACCENT} />
          <KPI label="Mean Onset" value={`${ov.mean_onset_years}y`} color={ACCENT4} />
          <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT3} />
          <KPI label="Photosensitive" value={`${ov.photosensitivity_pct}%`} color={ACCENT3} />
          <KPI label="Ambulatory" value={`${ov.ambulatory_pct}%`} color="#2e7d32" />
        </div>
        <div className="row mt-2">
          <KPI label="On LEV" value={`${ov.on_lev_pct}%`} color={ACCENT} />
          <KPI label="On CLB" value={`${ov.on_clb_pct}%`} color={ACCENT} />
          <KPI label="On Piracetam" value={`${ov.on_piracetam_pct}%`} color={ACCENT} />
          <KPI label="On CoQ10" value={`${ov.on_coq10_pct}%`} color={ACCENT4} />
          <KPI label="Ragged-Red ✓" value={`${ov.ragged_red_fibers_pct}%`} color={ACCENT2} />
          <KPI label="SNHL" value={`${ov.sensorineural_hl_pct}%`} color={ACCENT3} />
        </div>
        <div className="row mt-2">
          <KPI label="Cardiac (WPW/HCM)" value={`${ov.cardiac_abnormality_pct}%`} color={ACCENT2} />
        </div>
      </SectionCard>

      <SectionCard title="Key Thresholds" borderColor={ACCENT3}>
        <div className="row">
          <div className="col-md-6">
            <p className="small mb-1"><strong>Heteroplasmy threshold (muscle):</strong> &gt;{ov.heteroplasmy_clinical_threshold_pct}% mutant → biochemical deficit</p>
            <p className="small mb-1"><strong>Plasma lactate diagnostic:</strong> &gt;{ov.lactate_diagnostic_threshold_mmol} mmol/L (fasting)</p>
            <p className="small mb-1"><strong>CSF L:P ratio:</strong> &gt;{ov.lp_ratio_threshold} → Complex I/IV deficiency</p>
            <p className="small mb-1"><strong>Giant SEP diagnostic:</strong> N20/P25 &gt;{ov.sep_amplitude_threshold_uv} µV (pathognomonic)</p>
            <p className="small mb-0"><strong>Hearing loss referral:</strong> &gt;{ov.hearing_loss_referral_db} dB HL</p>
          </div>
          <div className="col-md-6">
            <p className="small mb-1"><strong>LVEF cardiology referral:</strong> &lt;{ov.lvef_cardiology_threshold_pct}%</p>
            <p className="small mb-1"><strong>CoQ10 supplementation:</strong> &lt;{ov.coq10_supplementation_threshold} µg/mL serum</p>
            <p className="small mb-1"><strong>IV LEV for SE:</strong> {ov.lev_iv_se_dose_mgkg} mg/kg (SOLE rescue — NO VPA/PHT)</p>
            <p className="small mb-1"><strong>Discovery:</strong> {ov.discovery}</p>
            <p className="small mb-0"><strong>Unique feature:</strong> {ov.unique_feature}</p>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Absolute & High-Risk Contraindications" borderColor={ACCENT2}>
        <p className="small fw-bold mb-1" style={{ color: ACCENT2 }}>ABSOLUTE CI (NEVER prescribe):</p>
        <p className="small mb-2">{(ov.absolute_ci || []).join(' · ')}</p>
        <p className="small fw-bold mb-1" style={{ color: ACCENT3 }}>HIGH RISK — AVOID:</p>
        <p className="small mb-0">{(ov.high_risk_ci || []).join(' · ')}</p>
      </SectionCard>
    </div>
  );
}

// ── TAB 2: Patients & Etiology ───────────────────────────────────────────────
function PatientsTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const { etiologies = [], patients = [] } = bd;
  return (
    <div>
      <SectionCard title="Etiology Distribution (5 classes)" borderColor={ACCENT}>
        {etiologies.map((e, i) => (
          <div key={i} className="mb-3">
            <PctBar label={e.class.replace(/-/g, ' ')} pct={e.pct} color={ACCENT} />
            <p className="small text-muted mb-1">{e.detail}</p>
            <p className="small text-muted"><strong>Testing:</strong> {e.testing}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title={`Patient Cohort (n=${patients.length})`} borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th>ID</th><th>Onset(y)</th><th>Sex</th><th>Mutation</th>
                <th>Heteroplasmy</th><th>RRF</th><th>SNHL</th><th>Cardiac</th>
                <th>Drug-R</th><th>Current Rx</th>
              </tr>
            </thead>
            <tbody>
              {patients.slice(0, 20).map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td>{p.age_onset}</td>
                  <td>{p.sex}</td>
                  <td style={{ fontSize: 11 }}>{p.mutation}</td>
                  <td><span className="badge bg-secondary">{p.heteroplasmy_pct}%</span></td>
                  <td><span className={`badge bg-${p.ragged_red_fibers === 'Yes' ? 'danger' : 'success'}`}>{p.ragged_red_fibers}</span></td>
                  <td><span className={`badge bg-${p.hearing_loss === 'Yes' ? 'warning text-dark' : 'success'}`}>{p.hearing_loss}</span></td>
                  <td><span className={`badge bg-${p.cardiac !== 'Normal' ? 'danger' : 'success'}`} style={{ fontSize: 10 }}>{p.cardiac}</span></td>
                  <td><span className={`badge bg-${p.drug_resistant ? 'warning text-dark' : 'success'}`}>{p.drug_resistant ? 'Yes' : 'No'}</span></td>
                  <td style={{ fontSize: 10 }}>{p.current_rx}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-muted small">Showing first 20 of {patients.length} patients. VPA never prescribed in any MERRF patient.</p>
      </SectionCard>
    </div>
  );
}

// ── TAB 3: Seizures & Triggers ───────────────────────────────────────────────
function SeizuresTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const { seizure_types = [], triggers = [] } = bd;
  return (
    <div>
      <SectionCard title="Seizure Types (5 types)" borderColor={ACCENT}>
        {seizure_types.map((s, i) => (
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: '1px solid #e0e0e0' }}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong className="small">{s.type}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{s.pct}%</span>
            </div>
            <PctBar label="" pct={s.pct} color={ACCENT} />
            <p className="small text-muted mb-1"><strong>EEG:</strong> {s.eeg}</p>
            <p className="small text-muted mb-1"><strong>Semiology:</strong> {s.semiology}</p>
            <p className="small" style={{ color: ACCENT4 }}><strong>Clinical tip:</strong> {s.clinical_tip}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers (8 triggers)" borderColor={ACCENT3}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-2">
            <PctBar label={t.trigger} pct={t.pct} color={ACCENT3} />
            <p className="small text-muted">{t.advice}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 4: Treatments ────────────────────────────────────────────────────────
function TreatmentsTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const { treatments = [], contraindications = [], monitoring = [], lifecycle = [] } = bd;
  return (
    <div>
      <SectionCard title="Treatments (8 interventions) — VPA ABSENT (absolute CI)" borderColor={ACCENT4}>
        {treatments.map((t, i) => (
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: '1px solid #e8f5e9' }}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong className="small">{t.drug}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.level}</span>
            </div>
            <p className="small text-muted mb-1"><strong>Dose:</strong> {t.dose}</p>
            <p className="small text-muted mb-1"><strong>MOA:</strong> {t.moa}</p>
            <p className="small text-muted mb-1"><strong>Efficacy:</strong> {t.efficacy}</p>
            <p className="small text-muted mb-1"><strong>Monitoring:</strong> {t.monitoring}</p>
            <p className="small" style={{ color: ACCENT4 }}><strong>MERRF note:</strong> {t.merrf_note}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications" borderColor={ACCENT2}>
        {contraindications.map((c, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ backgroundColor: c.risk.includes('ABSOLUTE') ? '#ffebee' : '#fff8e1' }}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong className="small">{c.drug}</strong>
              <span className="badge" style={{ backgroundColor: c.risk.includes('ABSOLUTE') ? ACCENT2 : ACCENT3 }}>{c.risk}</span>
            </div>
            <p className="small text-muted mb-0">{c.reason}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Checklist (14 items)" borderColor={ACCENT}>
        {monitoring.map((m, i) => (
          <div key={i} className="mb-2 pb-2" style={{ borderBottom: '1px solid #fce4ec' }}>
            <div className="d-flex justify-content-between align-items-center">
              <strong className="small">{m.item.replace(/-/g, ' ')}</strong>
              <span className="badge bg-info text-dark" style={{ fontSize: 10 }}>{m.frequency}</span>
            </div>
            <p className="small text-muted mb-0">{m.purpose}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Lifecycle Stages (6 stages)" borderColor={ACCENT}>
        {lifecycle.map((l, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ backgroundColor: '#fce4ec' }}>
            <strong className="small d-block mb-1" style={{ color: ACCENT }}>{l.stage}</strong>
            <p className="small text-muted mb-1">{l.note}</p>
            <p className="small mb-0"><strong>Management:</strong> {l.management}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 5: Definitions ───────────────────────────────────────────────────────
function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  const { concepts = [], thresholds = [], standards = [], references = [] } = defs;
  return (
    <div>
      <SectionCard title="Key Concepts (15)" borderColor={ACCENT}>
        {concepts.map((c, i) => (
          <div key={i} className="mb-3 pb-2" style={{ borderBottom: '1px solid #fce4ec' }}>
            <strong className="small d-block mb-1" style={{ color: ACCENT }}>{c.concept.replace(/-/g, ' ')}</strong>
            <p className="small text-muted mb-1">{c.definition}</p>
            <p className="small mb-0" style={{ color: '#546e7a' }}><em>Standard: {c.standard}</em></p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds (12)" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm" style={{ fontSize: 12 }}>
            <thead><tr><th>Threshold</th><th>Value</th><th>Rationale</th></tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td><strong>{t.name}</strong></td>
                  <td><span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.value}</span></td>
                  <td className="text-muted">{t.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Standards & Guidelines (12)" borderColor={ACCENT4}>
        <div className="row">
          {standards.map((s, i) => (
            <div key={i} className="col-md-6 mb-2">
              <strong className="small">{s.name}</strong>
              <p className="small text-muted mb-0">{s.scope}</p>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="References (6)" borderColor={ACCENT}>
        {references.map((r, i) => (
          <p key={i} className="small text-muted mb-2">
            <strong>[{r.ref}]</strong> {r.citation}
          </p>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────────
export default function MERRFPage() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr]   = useState('');

  useEffect(() => {
    fetch(`${API}/api/merrf/overview`).then(r => r.json()).then(setOv).catch(e => setErr(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 3) {
      if (!bd) fetch(`${API}/api/merrf/breakdown`).then(r => r.json()).then(setBd).catch(e => setErr(e.message));
    }
    if (tab === 2) {
      if (!bd) fetch(`${API}/api/merrf/breakdown`).then(r => r.json()).then(setBd).catch(e => setErr(e.message));
    }
    if (tab === 4) {
      if (!defs) fetch(`${API}/api/merrf/definitions`).then(r => r.json()).then(setDefs).catch(e => setErr(e.message));
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          🔋 MERRF Epilepsy
        </h4>
        <p className="text-muted small mb-0">
          Myoclonic Epilepsy with Ragged-Red Fibers · MT-TK (m.8344A&gt;G) · Maternal Inheritance · Complex I/IV Deficiency · VPA ABSOLUTE CI · 40-patient cohort
        </p>
      </div>

      {err && <div className="alert alert-danger py-2 small">{err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <PatientsTab bd={bd} />}
      {tab === 2 && <SeizuresTab bd={bd} />}
      {tab === 3 && <TreatmentsTab bd={bd} />}
      {tab === 4 && <DefinitionsTab defs={defs} />}
    </div>
  );
}
