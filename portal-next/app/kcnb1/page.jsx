'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1565c0';   // deep blue — KCNB1 / potassium channel
const ACCENT2 = '#b71c1c';   // dark red — CI / danger
const ACCENT3 = '#2e7d32';   // dark green — KD / success
const ACCENT4 = '#e65100';   // deep orange — GOF / alert
const ACCENT5 = '#4a148c';   // purple — CSWS/ESES / GOF clustering

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#e3f2fd', color: borderColor }}>
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

// ── Tab 1: Overview ───────────────────────────────────────────────────────────
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <div className="alert alert-danger fw-bold py-2 mb-3" style={{ fontSize: 13 }}>
        ⚡ <strong>KEY AHA:</strong> {ov.key_aha}
      </div>

      <SectionCard title="🧬 KCNB1 Gene & Syndrome" borderColor={ACCENT}>
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
        <KPI label="West Syndrome" value={`${ov.west_syndrome_pct}%`} color={ACCENT2} />
        <KPI label="GOF Variants" value={`${ov.gof_pct}%`} color={ACCENT4} />
        <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="CSWS/ESES" value={`${ov.csws_eses_pct}%`} color={ACCENT5} />
        <KPI label="On KD" value={`${ov.on_kd_pct}%`} color={ACCENT3} />
        <KPI label="LGS Evolution" value={`${ov.lgs_evolution_pct}%`} color={ACCENT4} />
        <KPI label="POLG Done" value={`${ov.polg_done_pct}%`} color={ACCENT3} />
        <KPI label="Avg Onset" value={`${ov.avg_onset_months}m`} color={ACCENT} />
        <KPI label="VPA w/o POLG" value={ov.vpa_without_polg} color={ACCENT2} />
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

// ── Tab 2: Patients & Etiology ────────────────────────────────────────────────
function PatientsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const s = bk.summary || {};
  return (
    <div>
      <Alert variant="info" text={
        `41-patient KCNB1 DEE26 cohort · West ${s.west_pct}% · GOF ${s.gof_pct}% · DRE ${s.drug_resistant_pct}% · ` +
        `CSWS/ESES ${s.csws_pct}% · KD ${s.kd_pct}% · LGS-evolution ${s.lgs_pct}% · VPA without POLG screen: ${s.vpa_without_polg} patients ⚠`
      } />

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
                <th>West</th><th>GOF</th><th>DRE</th><th>CSWS</th><th>KD</th><th>LGS</th><th>POLG</th>
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
                  <td className={p.gof ? 'text-warning fw-bold' : ''}>{p.gof ? 'Y' : 'N'}</td>
                  <td className={p.drug_resistant ? 'text-danger fw-bold' : ''}>{p.drug_resistant ? 'Y' : 'N'}</td>
                  <td className={p.csws_eses ? 'text-purple fw-bold' : ''}>{p.csws_eses ? 'Y' : 'N'}</td>
                  <td className={p.on_kd ? 'text-success' : ''}>{p.on_kd ? 'Y' : 'N'}</td>
                  <td className={p.lgse_evolution ? 'text-warning' : ''}>{p.lgse_evolution ? 'Y' : 'N'}</td>
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

// ── Tab 3: Seizure Types & Triggers ──────────────────────────────────────────
function SeizureTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <div className="alert alert-warning fw-bold py-2 mb-3" style={{ fontSize: 13 }}>
        🌡️ <strong>FEVER THRESHOLD 37.5°C</strong> (not 38°C) — Kv2.1 Q10~2.5: temperature shifts channel activation. &nbsp;|&nbsp;
        🚫 <strong>CBZ/OXC/PHT DURING WEST:</strong> NaV1.1 PV-interneuron block → spasm aggravation. &nbsp;|&nbsp;
        ⚡ <strong>CSWS/ESES (GOF):</strong> NREM-activated spike burst — steroid + CLB + melatonin.
      </div>

      <SectionCard title="⚡ Seizure Type Prevalence" borderColor={ACCENT}>
        {(bk.seizure_types || []).map((s, i) => (
          <PctBar key={i} label={s.type} pct={s.prevalence_pct} color={ACCENT} />
        ))}
      </SectionCard>

      <SectionCard title="⚡ Seizure Type Detail" borderColor={ACCENT}>
        {(bk.seizure_detail || []).map((s, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="fw-bold small" style={{ color: ACCENT }}>
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
            <span className="text-muted">{t.detail}</span>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Treatments ─────────────────────────────────────────────────────────
function TreatmentsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <div className="alert alert-danger fw-bold py-2 mb-3" style={{ fontSize: 13 }}>
        🚫 <strong>ABSOLUTE CI:</strong> Tiagabine (NCSE in KCNB1 dysmature cortex).<br/>
        ⛔ <strong>HIGH RISK during West:</strong> CBZ / OXC / PHT (NaV1.1 PV-interneuron block → spasm aggravation) | LTG if myoclonic component.<br/>
        🔬 <strong>NO APPROVED PRECISION THERAPY</strong> for KCNB1 DEE26 — KD (KATP-independent of Kv2.1) is the preferred DRE escalation.<br/>
        🧬 <strong>POLG MANDATORY before VPA</strong> in every KCNB1 infant.
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
            {t.kcnb1_specific && (
              <div className="small mt-1 p-1 rounded" style={{ backgroundColor: '#e3f2fd', color: ACCENT }}>
                ⚡ <strong>KCNB1-Specific:</strong> {t.kcnb1_specific}
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
            <div className="small"><strong>Mechanism:</strong> {c.mechanism}</div>
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
            <div className="text-muted">{m.rationale}</div>
            <div className="text-success">{m.action}</div>
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

// ── Tab 5: Definitions ────────────────────────────────────────────────────────
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
            <div className="small"><strong>Mechanism:</strong> {c.mechanism}</div>
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

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function KCNB1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/kcnb1/overview`).then(r => r.json()),
      fetch(`${API}/api/kcnb1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/kcnb1/definitions`).then(r => r.json()),
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
          ⚡ KCNB1 Epilepsy (DEE26)
        </h4>
        <div className="text-muted small">
          DEE26 · Kv2.1 Channelopathy · Voltage-Gated K+ Delayed-Rectifier I_Kslow · GOF-LOF Dual · 20q13.13
        </div>
        <div className="text-muted small mt-1">
          <strong>LOF (55%):</strong> Haploinsufficiency → impaired AP repolarisation → burst firing → West syndrome / infantile spasms &nbsp;|&nbsp;
          <strong>GOF (45%):</strong> Left-shifted activation / PRC clustering defect → CSWS/ESES + focal refractory DEE &nbsp;|&nbsp;
          <strong style={{ color: ACCENT2 }}>POLG mandatory before VPA · CBZ/OXC/PHT HIGH RISK during West</strong>
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
