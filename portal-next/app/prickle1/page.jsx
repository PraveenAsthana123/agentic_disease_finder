'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#33691e';   // dark olive-green — PRICKLE1 WNT/PCP planar cell polarity pathway
const ACCENT2 = '#b71c1c';   // dark red — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep orange — alerts / triggers / high-risk
const ACCENT4 = '#1565c0';   // deep blue — LEV SV2A rational / safe treatments

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#f1f8e9', color: borderColor }}>
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
        text="⛔ ABSOLUTE CI: CBZ / OXC / PHT / Fosphenytoin-IV (Na-channel — worsen action myoclonus in ALL PME). IV LEV 60 mg/kg = SOLE SE rescue. TGB ABSOLUTE CI (NCSE + PCP Purkinje vulnerability)."
      />
      <Alert
        variant="warning"
        text="⚠ HIGH RISK: GBP / PGB (α2δ — paradoxical myoclonic worsening; Crespel 1999; EPM5 misdiagnosis trap in cognitively-intact adolescents). Vigabatrin AVOID. LTG monotherapy HIGH RISK."
      />
      <Alert
        variant="info"
        text="🔵 EPM5 DISTINCTION: Prominent cerebellar ataxia often precedes myoclonus. Preserved cognition throughout (unlike EPM3 severe ID). Non-fatal — rehabilitation is core treatment. LEV SV2A mechanism rationally complements PRICKLE1-SYNAPSIN disruption. PER (AMPA): start 2 mg/night — SARA monitoring mandatory (ataxia amplification risk)."
      />

      <div className="row mb-4">
        <KPI label="Gene / Locus" value="PRICKLE1 / 12q12" color={ACCENT} />
        <KPI label="Disease" value="EPM5" color={ACCENT} />
        <KPI label="Inheritance" value="AR (severe) / AD (mild)" color={ACCENT} />
        <KPI label="Cohort" value={`${ov.cohort_size} patients`} color={ACCENT} />
        <KPI label="Mean Onset" value={`${ov.mean_onset_years} yrs`} color={ACCENT} />
        <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT3} />
      </div>

      <SectionCard title="Gene & Protein — PRICKLE1 WNT/PCP Pathway">
        <p className="small mb-1"><strong>Gene:</strong> {ov.gene}</p>
        <p className="small mb-1"><strong>Protein:</strong> {ov.protein}</p>
        <p className="small mb-1"><strong>Mechanism:</strong> {ov.mechanism}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {ov.inheritance}</p>
        <p className="small mb-1"><strong>OMIM:</strong> {ov.omim}</p>
        <p className="small mb-0"><strong>Discovery:</strong> {ov.discovery}</p>
      </SectionCard>

      <SectionCard title="EPM5 Clinical Signature — PRICKLE1">
        <p className="small mb-2">{ov.disease}</p>
        <p className="small mb-2"><strong>Unique feature:</strong> {ov.unique_feature}</p>
      </SectionCard>

      <SectionCard title="Cohort Summary">
        <div className="row">
          <div className="col-md-6">
            <PctBar label="Drug-Resistant" pct={ov.drug_resistant_pct} color={ACCENT3} />
            <PctBar label="Ambulatory" pct={ov.ambulatory_pct} color={ACCENT} />
            <PctBar label="Preserved Cognition" pct={ov.preserved_cognition_pct} color={ACCENT4} />
            <PctBar label="Cerebellar Ataxia" pct={ov.cerebellar_ataxia_pct} color={ACCENT3} />
            <PctBar label="Photosensitivity (IPS)" pct={ov.photosensitivity_pct} color={ACCENT3} />
            <PctBar label="Giant SEP Confirmed" pct={ov.giant_sep_confirmed_pct} color={ACCENT} />
          </div>
          <div className="col-md-6">
            <PctBar label="On VPA (backbone)" pct={ov.on_vpa_pct} color={ACCENT4} />
            <PctBar label="On LEV (SV2A rational)" pct={ov.on_lev_pct} color={ACCENT4} />
            <PctBar label="On Piracetam" pct={ov.on_piracetam_pct} color={ACCENT4} />
            <PctBar label="On CLB" pct={ov.on_clb_pct} color={ACCENT4} />
            <PctBar label="On Perampanel" pct={ov.on_perampanel_pct} color={ACCENT4} />
            <PctBar label="POLG1 Screened Pre-VPA" pct={ov.polg1_screened_pct} color={ACCENT} />
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Key Thresholds" borderColor={ACCENT4}>
        <div className="row small">
          <div className="col-md-6">
            <p><strong>Giant SEP threshold:</strong> &gt;8 µV (N20/P25) — cortical myoclonus</p>
            <p><strong>IV LEV for SE:</strong> {ov.lev_iv_se_dose_mgkg} mg/kg IV (SOLE rescue)</p>
            <p><strong>VPA trough target:</strong> {ov.vpa_trough_target_ugml} µg/mL</p>
          </div>
          <div className="col-md-6">
            <p><strong>PER start dose:</strong> 2 mg at night (increase 2 mg/4-6 wk; SARA monitored)</p>
            <p><strong>SARA walking aid:</strong> &gt;10 → walking aid; &gt;25 → wheelchair</p>
            <p><strong>POLG1 bridge pre-VPA:</strong> 7–14 days LEV + piracetam</p>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Absolute & High-Risk Contraindications" borderColor={ACCENT2}>
        <p className="small fw-bold text-danger mb-1">ABSOLUTE CI:</p>
        {(ov.absolute_ci || []).map((ci, i) => (
          <p key={i} className="small text-danger mb-1">⛔ {ci}</p>
        ))}
        <p className="small fw-bold mt-2" style={{ color: ACCENT3 }}>HIGH RISK / AVOID:</p>
        {(ov.high_risk_ci || []).map((ci, i) => (
          <p key={i} className="small mb-1" style={{ color: ACCENT3 }}>⚠ {ci}</p>
        ))}
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
      <SectionCard title="Etiology Classes — PRICKLE1 / EPM5 (5 classes)">
        {etiologies.map((e, i) => (
          <div key={i} className="mb-3 pb-3 border-bottom">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="fw-bold small">{e.class}</span>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{e.pct}%</span>
            </div>
            <PctBar label="" pct={e.pct} color={ACCENT} />
            <p className="small text-muted mb-1">{e.detail}</p>
            <p className="small"><strong>Testing:</strong> {e.testing}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title={`Patient Cohort — ${patients.length} EPM5 Patients`}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr style={{ backgroundColor: '#f1f8e9' }}>
                <th>ID</th><th>Onset</th><th>Sex</th><th>Mutation</th>
                <th>Ambul.</th><th>Cognition</th><th>Giant SEP</th><th>Ataxia</th><th>DRE</th><th>Regimen</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ color: ACCENT }}>{p.id}</td>
                  <td>{p.age_onset}y</td>
                  <td>{p.sex}</td>
                  <td><small>{p.mutation}</small></td>
                  <td>{p.ambulatory ? '✅' : '🦽'}</td>
                  <td>{p.preserved_cognition}</td>
                  <td>{p.giant_sep}</td>
                  <td>{p.cerebellar_ataxia}</td>
                  <td>{p.drug_resistant ? <span className="text-danger">DRE</span> : <span className="text-success">Ctrl</span>}</td>
                  <td><small>{p.current_rx}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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
      <SectionCard title="Seizure / Symptom Types — PRICKLE1 EPM5 (5 types)">
        {seizure_types.map((s, i) => (
          <div key={i} className="mb-4 pb-3 border-bottom">
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-bold small">{s.type}</span>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{s.pct}%</span>
            </div>
            <PctBar label="" pct={s.pct} color={ACCENT} />
            <p className="small mb-1"><strong>EEG:</strong> {s.eeg}</p>
            <p className="small mb-1"><strong>Semiology:</strong> {s.semiology}</p>
            <div className="alert alert-light py-1 mb-0" style={{ fontSize: 12 }}>
              <strong>Clinical tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers — PRICKLE1 EPM5 (8 triggers)" borderColor={ACCENT3}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-bold small" style={{ color: ACCENT3 }}>{t.trigger}</span>
              <span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.pct}%</span>
            </div>
            <PctBar label="" pct={t.pct} color={ACCENT3} />
            <p className="small text-muted mb-0">{t.advice}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 4: Treatments ─────────────────────────────────────────────────────────
function TreatmentsTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const { treatments = [], contraindications = [], monitoring = [], lifecycle = [] } = bd;
  return (
    <div>
      <Alert
        variant="danger"
        text="⛔ SE PROTOCOL: IV LEV 60 mg/kg SOLE rescue. NEVER fosphenytoin. NEVER CBZ IV. Document in ED, anaesthesia, school protocol."
      />
      <Alert
        variant="info"
        text="🔵 LEV SV2A: PRICKLE1 regulates SYNAPSIN-dependent synaptic vesicle trafficking. LEV SV2A modulation is the most mechanistically rational co-backbone in EPM5. PER (AMPA antagonist): start 2 mg/night — monitor SARA at each increment (cerebellar ataxia amplification risk)."
      />

      <SectionCard title="Treatments — PRICKLE1 EPM5 (8 agents)">
        {treatments.map((t, i) => (
          <div key={i} className="mb-4 pb-3 border-bottom">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold">{t.drug}</span>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.level}</span>
            </div>
            <p className="small mb-1"><strong>Dose:</strong> {t.dose}</p>
            <p className="small mb-1"><strong>MOA:</strong> {t.moa}</p>
            <p className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</p>
            <p className="small mb-1"><strong>Monitoring:</strong> {t.monitoring}</p>
            <div className="alert alert-light py-1 mb-0" style={{ fontSize: 12 }}>
              <strong>EPM5 note:</strong> {t.epm5_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications — PRICKLE1 EPM5" borderColor={ACCENT2}>
        {contraindications.map((c, i) => (
          <div key={i} className="mb-2 pb-2 border-bottom">
            <div className="d-flex justify-content-between">
              <span className="fw-bold small text-danger">{c.drug}</span>
              <span className={`badge ${c.risk.includes('ABSOLUTE') ? 'bg-danger' : 'bg-warning text-dark'}`}>{c.risk}</span>
            </div>
            <p className="small text-muted mb-0">{c.reason}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring — PRICKLE1 EPM5 (14 items)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead>
              <tr style={{ backgroundColor: '#e8f5e9' }}>
                <th>Item</th><th>Frequency</th><th>Purpose</th>
              </tr>
            </thead>
            <tbody>
              {monitoring.map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold small" style={{ color: ACCENT4 }}>{m.item}</td>
                  <td className="small">{m.frequency}</td>
                  <td className="small text-muted">{m.purpose}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Lifecycle Stages — PRICKLE1 EPM5 (6 stages)">
        {lifecycle.map((l, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <p className="fw-bold small mb-1" style={{ color: ACCENT }}>{l.stage}</p>
            <p className="small mb-1 text-muted">{l.note}</p>
            <p className="small mb-0"><strong>Management:</strong> {l.management}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 5: Definitions ───────────────────────────────────────────────────────
function DefinitionsTab({ def }) {
  if (!def) return <div className="text-muted">Loading…</div>;
  const { concepts = [], thresholds = [], standards = [], references = [] } = def;
  return (
    <div>
      <SectionCard title="Key Concepts — PRICKLE1 EPM5 (15 concepts)">
        {concepts.map((c, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <p className="fw-bold small mb-1" style={{ color: ACCENT }}>{c.concept}</p>
            <p className="small mb-1">{c.definition}</p>
            <p className="small text-muted mb-0"><strong>Standard:</strong> {c.standard}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds — PRICKLE1 EPM5" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead>
              <tr style={{ backgroundColor: '#e8f5e9' }}>
                <th>Threshold</th><th>Value</th><th>Rationale</th>
              </tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ color: ACCENT4 }}>{t.name}</td>
                  <td className="fw-bold text-danger">{t.value}</td>
                  <td className="small text-muted">{t.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Standards & Guidelines" borderColor={ACCENT}>
        {standards.map((s, i) => (
          <div key={i} className="d-flex justify-content-between mb-1 small">
            <span className="fw-bold" style={{ color: ACCENT }}>{s.name}</span>
            <span className="text-muted">{s.scope}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Key References" borderColor={ACCENT4}>
        {references.map((r, i) => (
          <p key={i} className="small mb-1">
            <strong style={{ color: ACCENT4 }}>[{r.ref}]</strong> {r.citation}
          </p>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function PRICKLE1Page() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [def, setDef] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/prickle1/overview`).then(r => r.json()).then(setOv).catch(() => setErr('API error'));
    fetch(`${API}/api/prickle1/breakdown`).then(r => r.json()).then(setBd).catch(() => setErr('API error'));
    fetch(`${API}/api/prickle1/definitions`).then(r => r.json()).then(setDef).catch(() => setErr('API error'));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-4" style={{ borderLeft: `6px solid ${ACCENT}`, paddingLeft: 16 }}>
        <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
          PRICKLE1 Epilepsy — EPM5
        </h2>
        <p className="text-muted small mb-0">
          Progressive Myoclonic Epilepsy Type 5 · WNT/PCP Planar Cell Polarity Pathway ·
          PRICKLE1-SYNAPSIN Synaptic Vesicle Trafficking · 12q12 · AR (severe EPM5) / AD (mild NP-ME) ·
          Prominent Early Cerebellar Ataxia · Preserved Cognition · Non-Fatal PME ·
          LEV SV2A Mechanistically Rational · PER AMPA Ataxia Amplification Risk ·
          CBZ/OXC/PHT ABSOLUTE CI · TGB ABSOLUTE CI · GBP/PGB Misdiagnosis Trap (Cognitively-Intact Adolescent) ·
          No Storage Material · SARA Primary Outcome · OMIM #613832 · Bassuk 2008 / Tao 2011
        </p>
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

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <PatientsTab bd={bd} />}
      {tab === 2 && <SeizuresTab bd={bd} />}
      {tab === 3 && <TreatmentsTab bd={bd} />}
      {tab === 4 && <DefinitionsTab def={def} />}
    </div>
  );
}
