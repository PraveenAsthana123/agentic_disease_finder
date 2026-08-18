'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4e342e';   // dark espresso-brown — KCTD7 BTB/CUL3 ubiquitin proteasome degradation pathway
const ACCENT2 = '#b71c1c';   // dark red — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep orange — alerts / triggers / high-risk
const ACCENT4 = '#1b5e20';   // deep green — safe treatments / antioxidant selenium

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#efebe9', color: borderColor }}>
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
        text="⚠ ABSOLUTE CIs: CBZ / OXC / PHT / Fosphenytoin-IV (Na-channel — worsen action myoclonus; IV fosphenytoin ABSOLUTE CI for SE → IV LEV 60 mg/kg) · TGB (GAT-1 — NCSE + progressive neuronal loss) · VPA without POLG1 screen (Alpers-Huttenlocher risk)"
        variant="danger"
      />
      <Alert
        text="🧬 KCTD7 (7q11.21) — BTB/POZ domain CUL3 E3 ubiquitin ligase adaptor · EPM3 · AR LOF · Onset 14 months–3 years (ULTRA-EARLY PME) · NCL-like EM curvilinear bodies WITHOUT CLN gene mutations · Rapid severe cognitive regression within 3–5 years · Selenium supplementation UNIQUE treatment"
        variant="info"
      />
      <Alert
        text="⚡ CRITICAL DISTINCTION: (1) NCL-like EM → MUST exclude CLN2 first (TPP1 enzyme assay); (2) Selenium supplementation (GPx antioxidant — UNIQUE to EPM3 among PMEs); (3) Metformin/KD are NOT disease-modifying in EPM3 (ubiquitin mechanism ≠ glycogen); (4) No scoliosis (unlike GOSR2); (5) No VPA mitochondrial CI (unlike MERRF)"
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
          <KPI label="Mean Onset" value={`${ov.mean_onset_years}y`} color={ACCENT3} />
          <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
          <KPI label="Photosensitive" value={`${ov.photosensitivity_pct}%`} color={ACCENT3} />
          <KPI label="Ambulatory" value={`${ov.ambulatory_pct}%`} color={ACCENT4} />
        </div>
        <div className="row mt-2">
          <KPI label="On VPA" value={`${ov.on_vpa_pct}%`} color={ACCENT} />
          <KPI label="On LEV" value={`${ov.on_lev_pct}%`} color={ACCENT} />
          <KPI label="On Piracetam" value={`${ov.on_piracetam_pct}%`} color={ACCENT} />
          <KPI label="On CLB" value={`${ov.on_clb_pct}%`} color={ACCENT} />
          <KPI label="On Selenium 🌿" value={`${ov.on_selenium_pct}%`} color={ACCENT4} />
          <KPI label="On KD" value={`${ov.on_kd_pct}%`} color={ACCENT4} />
        </div>
        <div className="row mt-2">
          <KPI label="Severe ID" value={`${ov.severe_id_pct}%`} color={ACCENT2} />
          <KPI label="NCL-like EM ✓" value={`${ov.ncl_like_em_pct}%`} color={ACCENT3} />
          <KPI label="Cerebellar Atrophy" value={`${ov.cerebellar_atrophy_pct}%`} color={ACCENT3} />
          <KPI label="Giant SEP ✓" value={`${ov.giant_sep_confirmed_pct}%`} color={ACCENT} />
          <KPI label="POLG1 Screened" value={`${ov.polg1_screened_pct}%`} color={ACCENT4} />
        </div>
      </SectionCard>

      <SectionCard title="Key Thresholds & Discovery" borderColor={ACCENT3}>
        <div className="row">
          <div className="col-md-6">
            <p className="small mb-1"><strong>Giant SEP threshold:</strong> N20/P25 &gt;{ov.sep_amplitude_threshold_uv} µV (pathognomonic cortical myoclonus)</p>
            <p className="small mb-1"><strong>Selenium target:</strong> {ov.selenium_target_ugL} µg/L serum (quarterly monitoring)</p>
            <p className="small mb-1"><strong>VPA trough target:</strong> {ov.vpa_trough_target_ugml} µg/mL</p>
            <p className="small mb-0"><strong>IV LEV for SE:</strong> {ov.lev_iv_se_dose_mgkg} mg/kg (ABSOLUTE FIRST-LINE)</p>
          </div>
          <div className="col-md-6">
            <p className="small mb-1"><strong>Discovery:</strong> {ov.discovery}</p>
            <p className="small mb-0"><strong>Unique feature:</strong> {ov.unique_feature}</p>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Absolute & High-Risk Contraindications" borderColor={ACCENT2}>
        <p className="small fw-bold mb-1" style={{ color: ACCENT2 }}>ABSOLUTE CI:</p>
        <p className="small mb-2">{(ov.absolute_ci || []).join(' · ')}</p>
        <p className="small fw-bold mb-1" style={{ color: ACCENT3 }}>HIGH RISK / NOT INDICATED — AVOID:</p>
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
                <th>NCL-like EM</th><th>Ambulatory</th><th>Drug-R</th><th>Severe ID</th><th>Selenium</th><th>Current Rx</th>
              </tr>
            </thead>
            <tbody>
              {patients.slice(0, 20).map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td>{p.age_onset}</td>
                  <td>{p.sex}</td>
                  <td style={{ fontSize: 11 }}>{p.mutation}</td>
                  <td><span className={`badge bg-${p.ncl_like_em === 'Yes' ? 'warning text-dark' : 'light text-dark'}`}>{p.ncl_like_em}</span></td>
                  <td><span className={`badge bg-${p.ambulatory ? 'success' : 'secondary'}`}>{p.ambulatory ? 'Yes' : 'No'}</span></td>
                  <td><span className={`badge bg-${p.drug_resistant ? 'danger' : 'success'}`}>{p.drug_resistant ? 'Yes' : 'No'}</span></td>
                  <td><span className={`badge bg-${p.severe_id ? 'danger' : 'warning text-dark'}`}>{p.severe_id ? 'Yes' : 'Mild'}</span></td>
                  <td><span className={`badge bg-${p.on_selenium === 'Yes' ? 'success' : 'light text-dark'}`}>{p.on_selenium}</span></td>
                  <td style={{ fontSize: 11 }}>{p.current_rx}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-muted small">Showing first 20 of {patients.length} patients.</p>
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
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: '1px solid #efebe9' }}>
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
      <SectionCard title="Treatments (8 interventions)" borderColor={ACCENT4}>
        {treatments.map((t, i) => (
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: '1px solid #e8f5e9' }}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong className="small">{t.drug}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.level}</span>
            </div>
            <p className="small mb-1"><strong>Dose:</strong> {t.dose}</p>
            <p className="small mb-1"><strong>MOA:</strong> {t.moa}</p>
            <p className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</p>
            <p className="small mb-1"><strong>Monitoring:</strong> {t.monitoring}</p>
            <p className="small mb-0" style={{ color: ACCENT }}><strong>EPM3 note:</strong> {t.epm3_note}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications (6)" borderColor={ACCENT2}>
        {contraindications.map((c, i) => (
          <div key={i} className="mb-3 pb-2" style={{ borderBottom: '1px solid #ffcdd2' }}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong className="small">{c.drug}</strong>
              <span className={`badge bg-${c.risk.includes('ABSOLUTE') ? 'danger' : 'warning text-dark'}`}>{c.risk}</span>
            </div>
            <p className="small mb-0 text-muted">{c.reason}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring (14 items)" borderColor={ACCENT}>
        {monitoring.map((m, i) => (
          <div key={i} className="mb-2 pb-1" style={{ borderBottom: '1px solid #efebe9' }}>
            <strong className="small">{m.item}</strong>
            <span className="badge ms-2" style={{ backgroundColor: '#795548', fontSize: 10 }}>{m.frequency}</span>
            <p className="small text-muted mb-0">{m.purpose}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Lifecycle Stages (6)" borderColor={ACCENT3}>
        {lifecycle.map((lc, i) => (
          <div key={i} className="mb-3 pb-2" style={{ borderBottom: '1px solid #fff3e0' }}>
            <strong className="small" style={{ color: ACCENT3 }}>{lc.stage}</strong>
            <p className="small text-muted mb-1 mt-1">{lc.note}</p>
            <p className="small mb-0"><strong>Management:</strong> {lc.management}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 5: Definitions ───────────────────────────────────────────────────────
function DefinitionsTab({ df }) {
  if (!df) return <div className="text-muted">Loading…</div>;
  const { concepts = [], thresholds = [], standards = [], references = [] } = df;
  return (
    <div>
      <SectionCard title="Key Concepts (15)" borderColor={ACCENT}>
        {concepts.map((c, i) => (
          <div key={i} className="mb-3 pb-2" style={{ borderBottom: '1px solid #efebe9' }}>
            <strong className="small" style={{ color: ACCENT }}>{c.concept.replace(/-/g, ' ')}</strong>
            <p className="small text-muted mb-1 mt-1">{c.definition}</p>
            <p className="small mb-0"><span className="badge" style={{ backgroundColor: '#a1887f', fontSize: 10 }}>{c.standard}</span></p>
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
                  <td className="fw-bold">{t.name}</td>
                  <td><span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.value}</span></td>
                  <td className="text-muted">{t.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Standards (12)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm" style={{ fontSize: 12 }}>
            <thead><tr><th>Standard</th><th>Scope</th></tr></thead>
            <tbody>
              {standards.map((s, i) => (
                <tr key={i}>
                  <td className="fw-bold">{s.name}</td>
                  <td className="text-muted">{s.scope}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Key References (6)" borderColor={ACCENT}>
        {references.map((r, i) => (
          <div key={i} className="mb-2">
            <span className="badge me-2" style={{ backgroundColor: ACCENT }}>{r.ref}</span>
            <span className="small text-muted">{r.citation}</span>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── MAIN PAGE ────────────────────────────────────────────────────────────────
export default function KCTD7Page() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [df, setDf] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/kctd7/overview`).then(r => r.json()).then(setOv).catch(() => setOv({}));
    fetch(`${API}/api/kctd7/breakdown`).then(r => r.json()).then(setBd).catch(() => setBd({}));
    fetch(`${API}/api/kctd7/definitions`).then(r => r.json()).then(setDf).catch(() => setDf({}));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🧬 KCTD7 Epilepsy Dashboard
        </h4>
        <p className="text-muted small mb-0">
          Progressive Myoclonic Epilepsy Type 3 (EPM3) · BTB/CUL3 Ubiquitin E3 Ligase Adaptor LOF ·
          7q11.21 · AR · Onset 14 months–3 years · NCL-like EM (curvilinear bodies, CLN enzymes normal) ·
          Selenium supplementation (UNIQUE antioxidant — GPx pathway) · 40-patient cohort ·
          Discovery: Van Bogaert et al. 2007
        </p>
      </div>

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

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <PatientsTab bd={bd} />}
      {tab === 2 && <SeizuresTab bd={bd} />}
      {tab === 3 && <TreatmentsTab bd={bd} />}
      {tab === 4 && <DefinitionsTab df={df} />}
    </div>
  );
}
