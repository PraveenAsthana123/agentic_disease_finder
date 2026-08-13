'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep navy — GABRG2 / GABA-A
const ACCENT2 = '#b71c1c';   // dark red — CI / danger
const ACCENT3 = '#1b5e20';   // dark green — KD / success
const ACCENT4 = '#e65100';   // deep orange — fever / temperature alert
const ACCENT5 = '#4a148c';   // purple — BDZ / rescue dose

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#e8eaf6', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function TabBtn({ label, active, onClick }) {
  return (
    <button
      className={`btn btn-sm me-2 mb-2 ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
      style={active ? { backgroundColor: ACCENT, borderColor: ACCENT } : {}}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

// ── Tab 1: Overview ────────────────────────────────────────────────────────────
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <div className="alert alert-danger fw-bold py-2 mb-3" style={{ fontSize: 13 }}>
        🧬 <strong>KEY AHA:</strong> {ov.key_aha}
      </div>

      <SectionCard title="🧬 GABRG2 Gene & Syndrome" borderColor={ACCENT}>
        <div className="row g-2">
          {[
            ['Gene', ov.gene],
            ['Locus', ov.locus],
            ['Inheritance', ov.inheritance],
            ['Protein', ov.protein],
          ].map(([k, v]) => (
            <div className="col-12 col-md-6" key={k}>
              <strong>{k}:</strong> <span className="text-secondary">{v}</span>
            </div>
          ))}
          <div className="col-12 mt-1">
            <strong>Mechanism:</strong> <span className="text-secondary">{ov.mechanism}</span>
          </div>
        </div>
      </SectionCard>

      <div className="row g-3 mb-4">
        <KPI label="Patients" value={ov.n_patients} color={ACCENT} />
        <KPI label="Febrile SE" value={`${ov.fse_pct}%`} color={ACCENT4} />
        <KPI label="Myoclonic" value={`${ov.myoclonic_pct}%`} color={ACCENT2} />
        <KPI label="West Synd." value={`${ov.west_syndrome_pct}%`} color={ACCENT} />
        <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="On KD" value={`${ov.on_kd_pct}%`} color={ACCENT3} />
        <KPI label="G-Tube" value={`${ov.gtube_pct}%`} color={ACCENT5} />
        <KPI label="POLG Done" value={`${ov.polg_done_pct}%`} color={ACCENT3} />
        <KPI label="BDZ ↑ Dose" value={`${ov.bdz_higher_dose_pct}%`} color={ACCENT5} />
        <KPI label="Temp Plan" value={`${ov.temperature_action_plan_pct}%`} color={ACCENT4} />
        <KPI label="Avg Onset" value={`${ov.avg_onset_months}m`} color={ACCENT} />
      </div>

      <div className="alert alert-warning py-2 mb-3" style={{ fontSize: 13 }}>
        🌡️ <strong>BDZ RESCUE — GABRG2-ADJUSTED DOSE:</strong> γ2 LOF → 30-40% fewer BDZ receptor sites at synapse → standard midazolam 0.2 mg/kg OFTEN FAILS.
        Use <strong>midazolam 0.3-0.4 mg/kg buccal</strong>. Document in ALL Emergency Seizure Plans (school, ambulance, ED).
      </div>

      <SectionCard title="⚠️ Contraindications Summary" borderColor={ACCENT2}>
        <ul className="mb-0">
          {(ov.contraindications_summary || []).map((ci, i) => (
            <li key={i} className="text-danger fw-semibold small">{ci}</li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="📐 Key Thresholds" borderColor={ACCENT}>
        <ul className="mb-0 small">
          {(ov.thresholds || []).map((t, i) => <li key={i}>{t}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="📚 References" borderColor={ACCENT}>
        <ul className="mb-0 small">
          {(ov.references || []).map((r, i) => <li key={i} className="text-muted">{r}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Etiology ─────────────────────────────────────────────────
function PatientsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const s = bk.summary || {};
  return (
    <div>
      <Alert variant="info" text={`41-patient GABRG2 cohort · FSE ${s.fse_pct}% · Myoclonic ${s.myoclonic_pct}% · West ${s.west_syndrome_pct}% · DRE ${s.drug_resistant_pct}% · KD ${s.kd_pct}% · G-Tube ${s.gtube_pct}% · BDZ-higher-dose documented ${s.bdz_higher_dose_pct}% · POLG-without-screen on VPA: ${s.vpa_without_polg} patients ⚠ · LTG prescribed in myoclonic: ${s.ltg_prescribed_total} (${s.ltg_switched_off} switched off)`} />

      <SectionCard title="🧬 Etiology Distribution (5-class catalog)" borderColor={ACCENT}>
        {(bk.etiology_distribution || []).map((e, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="d-flex justify-content-between">
              <strong className="small">{e.etiology}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{e.pct}% (n={e.n})</span>
            </div>
            <div className="text-muted" style={{ fontSize: 12 }}>{e.mechanism_short}…</div>
            <div className="text-secondary" style={{ fontSize: 11 }}>EEG: {e.eeg_signature_short}…</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="👥 Patient Sample" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th>ID</th><th>Category</th><th>Onset (m)</th><th>Age (m)</th>
                <th>West</th><th>Myoclonic</th><th>DRE</th><th>KD</th><th>BDZ↑</th><th>Temp Plan</th><th>POLG</th>
              </tr>
            </thead>
            <tbody>
              {(bk.patients_sample || []).map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.category}</td>
                  <td>{p.onset_months}</td>
                  <td>{p.age_months}</td>
                  <td>{p.west_syndrome ? '✓' : ''}</td>
                  <td className={p.myoclonic_seizures ? 'text-warning fw-bold' : ''}>{p.myoclonic_seizures ? 'Y' : 'N'}</td>
                  <td className={p.drug_resistant ? 'text-danger fw-bold' : ''}>{p.drug_resistant ? 'Y' : 'N'}</td>
                  <td className={p.on_kd ? 'text-success' : ''}>{p.on_kd ? 'Y' : 'N'}</td>
                  <td className={p.bdz_higher_dose ? 'text-primary' : 'text-danger'}>{p.bdz_higher_dose ? 'Y' : 'N'}</td>
                  <td className={p.temperature_action_plan ? 'text-success' : 'text-danger'}>{p.temperature_action_plan ? 'Y' : 'N'}</td>
                  <td className={p.polg_tested === 'Y' ? 'text-success' : 'text-danger'}>{p.polg_tested}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Seizure Types & Triggers ───────────────────────────────────────────
function SeizureTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <div className="alert alert-warning fw-bold py-2 mb-3" style={{ fontSize: 13 }}>
        🌡️ <strong>TEMPERATURE ALERT:</strong> K289M variant — channel inactivates at 40°C → acute synaptic disinhibition.
        Action threshold <strong>37.5°C</strong> (not 38°C standard). Fever Action Plan with GABRG2-adjusted BDZ dose mandatory.
        <br/>🚫 <strong>MYOCLONIC COMPONENT = LTG ABSOLUTE CI</strong> — NaV1.1 block in interneurons → myoclonus aggravation.
      </div>

      <SectionCard title="⚡ Seizure Type Prevalence" borderColor={ACCENT}>
        {(bk.seizure_types || []).map((s, i) => (
          <PctBar key={i} label={s.type} pct={s.prevalence_pct}
            color={s.type.includes('Febrile') ? ACCENT4 : ACCENT} />
        ))}
      </SectionCard>

      <SectionCard title="⚡ Seizure Type Detail" borderColor={ACCENT}>
        {(bk.seizure_detail || []).map((s, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="fw-bold small" style={{ color: s.type.includes('Febrile') ? ACCENT4 : ACCENT }}>
              {s.type} — {s.prevalence_pct}%
            </div>
            <div className="small text-muted">{s.semiology}</div>
            <div className="small"><strong>EEG:</strong> {s.eeg_pattern}</div>
            <div className="small text-success"><strong>Tip:</strong> {s.clinical_tip}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🌡️ Triggers" borderColor={ACCENT4}>
        {(bk.triggers || []).map((t, i) => (
          <PctBar key={i} label={t.trigger} pct={t.prevalence_pct} color={ACCENT4} />
        ))}
      </SectionCard>

      <SectionCard title="🌡️ Trigger Detail" borderColor={ACCENT4}>
        {(bk.trigger_detail || []).map((t, i) => (
          <div key={i} className="mb-2 small">
            <strong>{t.trigger} ({t.prevalence_pct}%):</strong>{' '}
            <span className="text-muted">{t.details}</span>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Treatments ──────────────────────────────────────────────────────────
function TreatmentsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <div className="alert alert-danger fw-bold py-2 mb-3" style={{ fontSize: 13 }}>
        🚫 <strong>ABSOLUTE CI:</strong> LTG (if any myoclonic component — NaV1.1 interneuron block) | Tiagabine in DEE11 (NCSE) | CBZ/OXC/PHT (generalised phenotype).<br/>
        💊 <strong>BDZ RESCUE DOSE:</strong> Standard doses FAIL in GABRG2 (γ2 LOF → 30-40% fewer BDZ sites). Use midazolam <strong>0.3-0.4 mg/kg</strong> (not 0.2 mg/kg). Document in ALL Emergency Seizure Plans.<br/>
        🔬 <strong>CLB ADVANTAGE:</strong> Clobazam (1,5-BDZ, α2/α3-preferring) retains partial activity even with γ2 LOF — preferred BDZ adjunct in GABRG2.
      </div>

      <SectionCard title="💊 Treatments" borderColor={ACCENT3}>
        {(bk.treatment_detail || []).map((t, i) => (
          <div key={i} className="mb-4 pb-3 border-bottom">
            <div className="d-flex justify-content-between align-items-start">
              <strong style={{ color: ACCENT }}>{t.drug}</strong>
              <span className="badge bg-secondary ms-2" style={{ fontSize: 10 }}>{t.level}</span>
            </div>
            <div className="small mt-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small"><strong>MOA:</strong> {t.moa}</div>
            <div className="small"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small text-warning"><strong>Safety:</strong> {t.safety}</div>
            <div className="small text-muted"><strong>Monitor:</strong> {t.monitoring}</div>
            {t.gabrg2_note && (
              <div className="small mt-1 p-1 rounded" style={{ backgroundColor: '#e8eaf6', color: ACCENT }}>
                🧬 <strong>GABRG2-Specific:</strong> {t.gabrg2_note}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications" borderColor={ACCENT2}>
        {(bk.contraindication_detail || []).map((c, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="fw-bold text-danger">{c.drug}</div>
            <div className="small"><strong>Risk:</strong> {c.risk}</div>
            <div className="small text-muted">{c.reason}</div>
            {c.alternative && (
              <div className="small text-success"><strong>Alternative:</strong> {c.alternative}</div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔍 Monitoring" borderColor={ACCENT}>
        {(bk.monitoring || []).map((m, i) => (
          <div key={i} className="mb-2 small pb-2 border-bottom">
            <strong>{m.item}</strong> — <span className="text-primary">{m.frequency}</span>
            <div className="text-muted">{m.details}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📅 Lifecycle Windows" borderColor={ACCENT5}>
        {(bk.lifecycle || []).map((w, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="fw-bold small" style={{ color: ACCENT5 }}>{w.window}</div>
            <div className="small text-muted">{w.focus}</div>
            <ul className="mb-0 small">
              {(w.key_actions || []).map((a, j) => <li key={j}>{a}</li>)}
            </ul>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 5: Definitions ─────────────────────────────────────────────────────────
function DefinitionsTab({ df }) {
  if (!df) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="📖 Key Concepts" borderColor={ACCENT}>
        {(df.concepts || []).map((c, i) => (
          <div key={i} className="mb-2 pb-1 border-bottom small">
            <strong style={{ color: ACCENT }}>{c.term}:</strong>{' '}
            <span className="text-muted">{c.definition}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Full Contraindications" borderColor={ACCENT2}>
        {(df.contraindications_full || []).map((c, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="fw-bold text-danger">{c.drug}</div>
            <div className="small"><strong>Risk:</strong> {c.risk}</div>
            <div className="small text-muted">{c.reason}</div>
            {c.alternative && (
              <div className="small text-success"><strong>Alternative:</strong> {c.alternative}</div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📐 Thresholds" borderColor={ACCENT}>
        <ul className="mb-0 small">
          {(df.thresholds || []).map((t, i) => <li key={i}>{t}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="📋 Standards" borderColor={ACCENT}>
        <ul className="mb-0 small">
          {(df.standards || []).map((s, i) => <li key={i} className="text-muted">{s}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="📚 References" borderColor={ACCENT}>
        <ul className="mb-0 small">
          {(df.references || []).map((r, i) => <li key={i} className="text-muted">{r}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function GABRG2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/gabrg2/overview`).then(r => r.json()),
      fetch(`${API}/api/gabrg2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/gabrg2/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <div className="container mt-4 alert alert-danger">Error: {error}</div>;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          🧬 GABRG2 Epilepsy
        </h4>
        <div className="text-muted small">
          DEE11 · GEFS+ Spectrum · GABA-A γ2 Subunit · Fever-Sensitive · BDZ Hyposensitivity · 5q34
        </div>
        <div className="text-muted small mt-1">
          <strong>γ2 LOF:</strong> Fewer BDZ-sensitive pentamers → <strong style={{ color: ACCENT2 }}>standard rescue doses FAIL</strong> → use midazolam 0.3-0.4 mg/kg &nbsp;|&nbsp;
          <strong style={{ color: ACCENT4 }}>K289M: temperature-sensitive gating</strong> (channel inactivates at 40°C) → action at 37.5°C &nbsp;|&nbsp;
          <strong style={{ color: ACCENT2 }}>LTG ABSOLUTE CI if myoclonic seizures</strong>
        </div>
      </div>

      <div className="mb-3">
        {TABS.map((t, i) => <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />)}
      </div>

      {tab === 0 && <OverviewTab ov={overview} />}
      {tab === 1 && <PatientsTab bk={breakdown} />}
      {tab === 2 && <SeizureTab bk={breakdown} />}
      {tab === 3 && <TreatmentsTab bk={breakdown} />}
      {tab === 4 && <DefinitionsTab df={definitions} />}
    </div>
  );
}
