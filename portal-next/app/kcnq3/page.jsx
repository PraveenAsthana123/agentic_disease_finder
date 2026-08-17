'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a5276';   // deep blue — neonatal channelopathy / M-current
const ACCENT2 = '#b71c1c';   // dark red — CI / danger / withdrawn
const ACCENT3 = '#e65100';   // deep orange — fever / alerts
const ACCENT4 = '#1b5e20';   // deep green — BFNS remission / safe
const ACCENT5 = '#4a148c';   // purple — DEE spectrum / mechanisms

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

      <div className="row g-2 mb-3">
        <div className="col-12 col-md-6">
          <div className="alert py-2 mb-0" style={{ fontSize: 12, backgroundColor: '#fce4ec', borderColor: ACCENT2, color: '#880e4f' }}>
            🚨 <strong>TIAGABINE:</strong> {ov.tiagabine_alert}
          </div>
        </div>
        <div className="col-12 col-md-6">
          <div className="alert py-2 mb-0" style={{ fontSize: 12, backgroundColor: '#fff3e0', borderColor: ACCENT3, color: '#bf360c' }}>
            ⚠️ <strong>CBZ/OXC:</strong> {ov.cbz_alert}
          </div>
        </div>
      </div>

      <div className="row g-2 mb-3">
        <div className="col-12 col-md-6">
          <div className="alert py-2 mb-0" style={{ fontSize: 12, backgroundColor: '#f3e5f5', borderColor: ACCENT5, color: '#4a148c' }}>
            🏥 <strong>POLG ALERT:</strong> {ov.polg_alert}
          </div>
        </div>
        <div className="col-12 col-md-6">
          <div className="alert py-2 mb-0" style={{ fontSize: 12, backgroundColor: '#fce4ec', borderColor: ACCENT2, color: '#880e4f' }}>
            ⛔ <strong>RETIGABINE:</strong> {ov.retigabine_alert}
          </div>
        </div>
      </div>

      <SectionCard title="🧬 KCNQ3 Gene & M-Current" borderColor={ACCENT}>
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

      <SectionCard title={`📊 Cohort KPIs (n=${ov.n_patients})`} borderColor={ACCENT4}>
        <div className="row g-2">
          <KPI label="BFNS-3 %" value={`${ov.bfns_pct}%`} color={ACCENT4} />
          <KPI label="Seizure-free %" value={`${ov.seizure_free_pct}%`} color={ACCENT4} />
          <KPI label="DRE %" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
          <KPI label="West Syndrome" value={`${ov.west_pct}%`} color={ACCENT5} />
          <KPI label="On CBZ/OXC" value={`${ov.on_cbz_oxc_pct}%`} color={ACCENT} />
          <KPI label="On PB (acute)" value={`${ov.on_pb_pct}%`} color={ACCENT3} />
          <KPI label="On KD" value={`${ov.on_kd_pct}%`} color={ACCENT5} />
          <KPI label="POLG Done" value={`${ov.polg_done_pct}%`} color={ov.polg_done_pct >= 80 ? ACCENT4 : ACCENT2} />
          <KPI label="HLA-B*15:02 Done" value={`${ov.hla_done_pct}%`} color={ov.hla_done_pct >= 75 ? ACCENT4 : ACCENT2} />
          <KPI label="OXC-SIADH Na↓" value={`${ov.hyponatraemia_pct}%`} color={ov.hyponatraemia_pct > 10 ? ACCENT2 : ACCENT4} />
          <KPI label="Avg Onset" value={`Day ${ov.avg_onset_days}`} color={ACCENT} />
        </div>
      </SectionCard>

      <SectionCard title="⚠️ Contraindications Summary" borderColor={ACCENT2}>
        {(ov.contraindications_summary || []).map((c, i) => (
          <div key={i} className="alert alert-danger py-1 mb-1" style={{ fontSize: 12 }}>{c}</div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Key Thresholds" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm mb-0" style={{ fontSize: 12 }}>
            <thead className="table-light"><tr><th>Parameter</th><th>Threshold / Action</th></tr></thead>
            <tbody>
              {Object.entries(ov.thresholds || {}).map(([k, v], i) => (
                <tr key={i}>
                  <td><strong>{k.replace(/_/g, ' ')}</strong></td>
                  <td style={{ color: ACCENT3 }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Etiology ────────────────────────────────────────────────
function PatientsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const { summary, etiology_distribution, patients_sample } = bk;
  return (
    <div>
      <SectionCard title="📊 Cohort Summary" borderColor={ACCENT}>
        <div className="row g-2">
          {[
            ['Total Patients', summary.n, ACCENT],
            ['Seizure-Free', `${summary.seizure_free_pct}%`, ACCENT4],
            ['DRE', `${summary.drug_resistant_pct}%`, ACCENT2],
            ['On CBZ/OXC', `${summary.on_cbz_oxc_pct}%`, ACCENT],
            ['On KD', `${summary.on_kd_pct}%`, ACCENT5],
            ['POLG Done', `${summary.polg_done_pct}%`, summary.polg_done_pct >= 80 ? ACCENT4 : ACCENT2],
            ['HLA-B*15:02', `${summary.hla_done_pct}%`, summary.hla_done_pct >= 75 ? ACCENT4 : ACCENT2],
            ['Hyponatraemia', `${summary.hyponatraemia_pct}%`, summary.hyponatraemia_pct > 10 ? ACCENT2 : ACCENT4],
            ['VPA w/o POLG', summary.vpa_without_polg, summary.vpa_without_polg > 0 ? ACCENT2 : ACCENT4],
          ].map(([k, v, c]) => (
            <div className="col-6 col-md-3" key={k}>
              <div className="card text-center py-2 shadow-sm">
                <div className="fw-bold" style={{ color: c }}>{v}</div>
                <div className="small text-muted">{k}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🧬 Etiology Distribution (5 Classes — KCNQ3)" borderColor={ACCENT5}>
        {(etiology_distribution || []).map((e, i) => (
          <div key={i} className="mb-3 p-2 border rounded" style={{ borderLeft: `4px solid ${ACCENT5}` }}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong style={{ fontSize: 13 }}>{e.etiology}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT5 }}>{e.pct}% (n≈{e.n})</span>
            </div>
            <div style={{ fontSize: 12, color: '#555' }}>{e.mechanism_short}…</div>
            <div className="mt-1" style={{ fontSize: 12, color: '#777' }}>EEG: {e.eeg_signature_short}…</div>
            <div className="progress mt-1" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT5 }} />
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="👥 Patient Sample (first 15)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-hover" style={{ fontSize: 11 }}>
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Name</th><th>Age(m)</th><th>Onset(day)</th>
                <th>Etiology</th><th>DRE</th><th>Seizure-Free</th>
                <th>West</th><th>CBZ/OXC</th><th>KD</th><th>POLG</th><th>HLA</th>
              </tr>
            </thead>
            <tbody>
              {(patients_sample || []).map(p => (
                <tr key={p.id}>
                  <td><code>{p.id}</code></td>
                  <td>{p.name}</td>
                  <td>{p.age_months}</td>
                  <td>Day {p.onset_days}</td>
                  <td style={{ fontSize: 10 }}>{p.etiology.split(' ')[0]}</td>
                  <td>{p.drug_resistant ? <span style={{ color: ACCENT2 }}>Yes</span> : 'No'}</td>
                  <td>{p.seizure_free ? <span style={{ color: ACCENT4 }}>Yes</span> : '—'}</td>
                  <td>{p.west_syndrome ? '🟡 West' : '—'}</td>
                  <td>{p.on_cbz_oxc ? '✓' : '—'}</td>
                  <td>{p.on_kd ? '✓' : '—'}</td>
                  <td style={{ color: p.polg_tested === 'Y' ? ACCENT4 : ACCENT2 }}>{p.polg_tested}</td>
                  <td>{p.hla_b1502_tested ? <span style={{ color: ACCENT4 }}>✓</span> : <span style={{ color: ACCENT2 }}>!</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Seizures & Triggers ────────────────────────────────────────────────
function SeizuresTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const { seizure_detail, trigger_detail } = bk;
  return (
    <div>
      <SectionCard title="⚡ Seizure Types (KCNQ3 — BFNS-3 + DEE spectrum)" borderColor={ACCENT}>
        {(seizure_detail || []).map((s, i) => (
          <div key={i} className="mb-3 p-2 border rounded">
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ color: ACCENT, fontSize: 13 }}>{s.type}</strong>
              <span className="badge bg-primary">{s.prevalence_pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${s.prevalence_pct}%`, backgroundColor: ACCENT }} />
            </div>
            <div style={{ fontSize: 12 }}><strong>EEG:</strong> {s.eeg_correlate}</div>
            <div className="alert alert-info py-1 mt-1 mb-0" style={{ fontSize: 12 }}>
              💡 <strong>Clinical Tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🌡️ Seizure Triggers (KCNQ3)" borderColor={ACCENT3}>
        {(trigger_detail || []).map((t, i) => (
          <div key={i} className="mb-2 p-2 border rounded">
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ fontSize: 13 }}>{t.trigger}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.prevalence_pct}%</span>
            </div>
            <div className="progress mb-1" style={{ height: 5 }}>
              <div className="progress-bar" style={{ width: `${t.prevalence_pct}%`, backgroundColor: ACCENT3 }} />
            </div>
            <div style={{ fontSize: 12, color: '#555' }}>{t.mechanism}</div>
            <div className="mt-1" style={{ fontSize: 12, color: ACCENT4 }}>
              <strong>Action:</strong> {t.action}
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Treatments ─────────────────────────────────────────────────────────
function TreatmentsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const { treatment_detail, contraindication_detail, monitoring, lifecycle } = bk;
  return (
    <div>
      <SectionCard title="💊 Treatments (KCNQ3 — Evidence-Based)" borderColor={ACCENT4}>
        {(treatment_detail || []).map((t, i) => (
          <div key={i} className="mb-3 p-2 border rounded" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong style={{ color: ACCENT4, fontSize: 13 }}>{t.drug}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT4, fontSize: 10 }}>{t.level}</span>
            </div>
            <div style={{ fontSize: 12 }}><strong>Dose:</strong> {t.dose}</div>
            <div style={{ fontSize: 12 }}><strong>MOA:</strong> {t.moa}</div>
            <div style={{ fontSize: 12 }}><strong>Efficacy:</strong> {t.efficacy}</div>
            <div style={{ fontSize: 12 }}><strong>Safety:</strong> {t.safety}</div>
            <div style={{ fontSize: 12 }}><strong>Monitoring:</strong> {t.monitoring}</div>
            {t.kcnq3_note && (
              <div className="alert py-1 mt-1 mb-0" style={{ fontSize: 12, backgroundColor: '#e8f5e9', borderColor: ACCENT4, color: '#1b5e20' }}>
                🧬 <strong>KCNQ3-specific:</strong> {t.kcnq3_note}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications (KCNQ3)" borderColor={ACCENT2}>
        {(contraindication_detail || []).map((c, i) => (
          <div key={i} className="mb-3 alert alert-danger py-2" style={{ fontSize: 12 }}>
            <div className="fw-bold mb-1">
              <span className="badge bg-danger me-2">{c.severity}</span>{c.drug}
            </div>
            <div><strong>Risk:</strong> {c.risk}</div>
            <div className="mt-1"><strong>Mechanism:</strong> {c.mechanism}</div>
            <div className="mt-1"><strong>Action:</strong> {c.action}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📋 Monitoring Protocol (KCNQ3)" borderColor={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm mb-0" style={{ fontSize: 12 }}>
            <thead className="table-light">
              <tr><th>Item</th><th>Frequency</th><th>Threshold</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td><strong>{m.item}</strong></td>
                  <td style={{ color: ACCENT5 }}>{m.frequency}</td>
                  <td style={{ color: ACCENT3, fontSize: 11 }}>{m.threshold}</td>
                  <td style={{ fontSize: 11 }}>{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📅 Disease Lifecycle (KCNQ3 — 6 Windows)" borderColor={ACCENT}>
        {(lifecycle || []).map((l, i) => (
          <div key={i} className="mb-2 p-2 border rounded">
            <div className="fw-bold" style={{ color: ACCENT, fontSize: 13 }}>{l.phase}</div>
            <div style={{ fontSize: 12, color: '#444' }}>{l.description}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 5: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ def }) {
  if (!def) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="📖 Key Concepts (KCNQ3 / M-Current / Kv7.3)" borderColor={ACCENT}>
        {(def.concepts || []).map((c, i) => (
          <div key={i} className="mb-2 p-2 border-bottom">
            <div className="fw-bold" style={{ color: ACCENT, fontSize: 13 }}>{c.term}</div>
            <div style={{ fontSize: 12, color: '#444' }}>{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications (Full)" borderColor={ACCENT2}>
        {(def.contraindications_full || []).map((c, i) => (
          <div key={i} className="mb-2 p-2 border rounded" style={{ borderLeft: `3px solid ${ACCENT2}` }}>
            <div className="fw-bold" style={{ color: ACCENT2, fontSize: 12 }}>{c.severity} — {c.drug}</div>
            <div style={{ fontSize: 12 }}>{c.risk}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Key Thresholds" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm mb-0" style={{ fontSize: 12 }}>
            <thead className="table-light"><tr><th>Parameter</th><th>Threshold / Action</th></tr></thead>
            <tbody>
              {Object.entries(def.thresholds || {}).map(([k, v], i) => (
                <tr key={i}>
                  <td><strong>{k.replace(/_/g, ' ')}</strong></td>
                  <td style={{ color: ACCENT3 }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📚 Standards & References" borderColor={ACCENT5}>
        <div className="mb-3">
          <strong>Standards:</strong>
          <ul className="mb-0 mt-1" style={{ fontSize: 12 }}>
            {(def.standards || []).map((s, i) => <li key={i}>{s}</li>)}
          </ul>
        </div>
        <div>
          <strong>References:</strong>
          <ol className="mb-0 mt-1" style={{ fontSize: 12 }}>
            {(def.references || []).map((r, i) => <li key={i}>{r}</li>)}
          </ol>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function KCNQ3Page() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [def, setDef] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/kcnq3/overview`).then(r => r.json()),
      fetch(`${API}/api/kcnq3/breakdown`).then(r => r.json()),
      fetch(`${API}/api/kcnq3/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDef(d); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          ⚡ KCNQ3 Epilepsy (BFNS-3 / DEE-KCNQ3 / Kv7.3 M-Current Partner / 11q23.3)
        </h4>
        <div className="text-muted small">
          KCNQ3 — Kv7.3 Voltage-Gated K⁺ Channel · Obligate Kv7.2/Kv7.3 M-Current Heteromer ·
          11q23.3 · Benign Familial Neonatal Seizures type 3 / DEE spectrum ·
          CBZ/OXC KCNQ3-specific (Kv7 M-current enhancement)
        </div>
      </div>

      {err && <div className="alert alert-danger">Error: {err}</div>}

      <div className="mb-3 border-bottom pb-2">
        {TABS.map((t, i) => (
          <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />
        ))}
      </div>

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <PatientsTab bk={bk} />}
      {tab === 2 && <SeizuresTab bk={bk} />}
      {tab === 3 && <TreatmentsTab bk={bk} />}
      {tab === 4 && <DefinitionsTab def={def} />}
    </div>
  );
}
