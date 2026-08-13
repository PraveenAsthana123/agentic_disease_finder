'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4a148c';   // deep purple — FOXG1 / forebrain
const ACCENT2 = '#b71c1c';   // dark red — CI / danger / tiagabine
const ACCENT3 = '#1b5e20';   // dark green — KD / success
const ACCENT4 = '#e65100';   // deep orange — dyskinesias / baclofen alert

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
        🧬 <strong>KEY AHA:</strong> {ov.key_aha}
      </div>

      {/* Gene summary */}
      <SectionCard title="🧬 FOXG1 Gene & Syndrome" borderColor={ACCENT}>
        <div className="row small">
          {[
            ['Gene', ov.gene],
            ['Locus', ov.locus],
            ['Inheritance', ov.inheritance],
            ['Protein', ov.protein],
          ].map(([k, v]) => (
            <div key={k} className="col-md-6 mb-2">
              <strong>{k}:</strong> {v}
            </div>
          ))}
          <div className="col-12 mt-1">
            <strong>Mechanism:</strong> {ov.mechanism}
          </div>
        </div>
      </SectionCard>

      {/* KPIs */}
      <SectionCard title="📊 Cohort KPIs (N=41)" borderColor={ACCENT}>
        <div className="row">
          <KPI label="West Syndrome" value={`${ov.west_syndrome_pct}%`} color={ACCENT} />
          <KPI label="Drug Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
          <KPI label="On KD" value={`${ov.on_kd_pct}%`} color={ACCENT3} />
          <KPI label="G-Tube" value={`${ov.gtube_pct}%`} color={ACCENT4} />
          <KPI label="Baclofen (Dyskinesias)" value={`${ov.baclofen_pct}%`} color='#7b1fa2' />
          <KPI label="POLG Done" value={`${ov.polg_done_pct}%`} color='#0277bd' />
          <KPI label="On VPA" value={`${ov.on_vpa_pct}%`} color='#558b2f' />
          <KPI label="Avg Onset" value={`${ov.avg_onset_months}m`} color='#4e342e' />
        </div>
      </SectionCard>

      {/* Contraindications summary */}
      <SectionCard title="🚫 Contraindications Summary" borderColor={ACCENT2}>
        {(ov.contraindications_summary || []).map((drug, i) => (
          <div key={i} className="alert alert-danger py-1 mb-2 small">
            ⛔ <strong>AVOID:</strong> {drug}
          </div>
        ))}
      </SectionCard>

      {/* Standards */}
      <SectionCard title="📚 Standards & References" borderColor={ACCENT3}>
        <div className="row">
          <div className="col-md-6">
            <strong className="small">Standards:</strong>
            <ul className="small mb-0 mt-1">
              {(ov.standards || []).map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </div>
          <div className="col-md-6">
            <strong className="small">References:</strong>
            <ul className="small mb-0 mt-1">
              {(ov.references || []).map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Etiology ────────────────────────────────────────────────
function EtiologyTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const { etiology_distribution: etiol, patients_sample: pats, summary } = bd;
  return (
    <div>
      {/* Summary pills */}
      <SectionCard title="📊 Cohort Summary" borderColor={ACCENT}>
        <div className="row small">
          {[
            ['West Syndrome', `${summary.west_syndrome_pct}%`],
            ['Drug Resistant', `${summary.drug_resistant_pct}%`],
            ['On KD', `${summary.kd_pct}%`],
            ['G-Tube', `${summary.gtube_pct}%`],
            ['Baclofen Use', `${summary.baclofen_pct}%`],
            ['ACTH Complete Response', `${summary.acth_complete_pct}%`],
            ['ACTH Partial Response', `${summary.acth_partial_pct}%`],
            ['ACTH No Response', `${summary.acth_none_pct}%`],
          ].map(([k, v]) => (
            <div key={k} className="col-6 col-md-3 mb-2">
              <div className="border rounded p-2 text-center">
                <div className="fw-bold" style={{ color: ACCENT }}>{v}</div>
                <div className="text-muted" style={{ fontSize: 11 }}>{k}</div>
              </div>
            </div>
          ))}
        </div>
        {summary.vpa_without_polg > 0 && (
          <div className="alert alert-danger mt-2 py-1 small">
            ⚠️ <strong>{summary.vpa_without_polg} patients on VPA WITHOUT confirmed POLG exclusion</strong> — URGENT: complete POLG sequencing.
          </div>
        )}
      </SectionCard>

      {/* Etiology catalog */}
      <SectionCard title="🧬 Etiology Catalog (5 classes)" borderColor={ACCENT}>
        {(etiol || []).map((e, i) => (
          <div key={i} className="card mb-3 shadow-sm">
            <div className="card-header d-flex justify-content-between align-items-center py-1"
                 style={{ backgroundColor: '#ede7f6' }}>
              <span className="fw-bold small" style={{ color: ACCENT }}>{e.category}</span>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{e.pct}% · n={e.n}</span>
            </div>
            <div className="card-body py-2 small">
              <div className="mb-1"><strong>Etiology:</strong> {e.etiology}</div>
              <div className="mb-1 text-muted"><strong>Mechanism:</strong> {e.mechanism_short}…</div>
              <div className="text-muted"><strong>EEG signature:</strong> {e.eeg_signature_short}…</div>
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Patient table */}
      <SectionCard title={`🧑‍⚕️ Patient Cohort (N=${(pats || []).length})`} borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>ID</th><th>Category</th><th>Onset (m)</th><th>Age (m)</th>
                <th>West</th><th>DRE</th><th>KD</th><th>G-Tube</th><th>Baclofen</th><th>POLG</th>
              </tr>
            </thead>
            <tbody>
              {(pats || []).map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {p.category.replace('FOXG1-', '').replace(/-/g, ' ')}
                  </td>
                  <td>{p.onset_months}</td>
                  <td>{p.age_months}</td>
                  <td>{p.west_syndrome ? '✅' : '—'}</td>
                  <td style={{ color: p.drug_resistant ? ACCENT2 : 'inherit' }}>{p.drug_resistant ? 'DRE' : '—'}</td>
                  <td style={{ color: p.on_kd ? ACCENT3 : 'inherit' }}>{p.on_kd ? 'KD' : '—'}</td>
                  <td>{p.gtube ? '🔩' : '—'}</td>
                  <td style={{ color: p.baclofen_on ? '#7b1fa2' : 'inherit' }}>{p.baclofen_on ? 'BAC' : '—'}</td>
                  <td style={{ color: p.polg_tested === 'N' ? ACCENT2 : ACCENT3 }}>
                    {p.polg_tested === 'Y' ? '✅' : p.polg_tested === 'N' ? '⚠️ TODO' : p.polg_tested}
                  </td>
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
function SeizureTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const { seizure_detail: seizures, trigger_detail: triggers } = bd;
  return (
    <div>
      <SectionCard title="⚡ Seizure Types (prevalence in FOXG1 cohort)" borderColor={ACCENT}>
        {(seizures || []).map((s, i) => (
          <div key={i} className="card mb-3 shadow-sm">
            <div className="card-header d-flex justify-content-between py-1" style={{ backgroundColor: '#f3e5f5' }}>
              <span className="fw-bold small" style={{ color: ACCENT }}>{s.type}</span>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{s.prevalence_pct}%</span>
            </div>
            <div className="card-body py-2 small">
              <div className="mb-2">
                <div className="progress mb-1" style={{ height: 8 }}>
                  <div className="progress-bar" style={{ width: `${s.prevalence_pct}%`, backgroundColor: ACCENT }} />
                </div>
              </div>
              <div className="mb-1"><strong>Semiology:</strong> {s.semiology}</div>
              <div className="mb-1 text-muted"><strong>EEG Pattern:</strong> {s.eeg_pattern}</div>
              <div className="alert alert-info py-1 mb-0" style={{ fontSize: 12 }}>
                💡 <strong>Clinical Tip:</strong> {s.clinical_tip}
              </div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="⚠️ Seizure Triggers" borderColor={ACCENT4}>
        {(triggers || []).map((t, i) => (
          <div key={i} className="card mb-2 shadow-sm">
            <div className="card-header d-flex justify-content-between py-1" style={{ backgroundColor: '#fff3e0' }}>
              <span className="fw-bold small">{t.trigger}</span>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.prevalence_pct}%</span>
            </div>
            <div className="card-body py-2 small">
              <div className="mb-1">
                <div className="progress mb-1" style={{ height: 6 }}>
                  <div className="progress-bar" style={{ width: `${t.prevalence_pct}%`, backgroundColor: ACCENT4 }} />
                </div>
              </div>
              {t.mechanism}
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Treatments ─────────────────────────────────────────────────────────
function TreatmentsTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const { treatment_detail: treatments, contraindication_detail: cis } = bd;
  return (
    <div>
      <Alert text="⛔ TIAGABINE: ABSOLUTE CI — NCSE risk in FOXG1 diffuse cortical dysmaturation. | CBZ/OXC/PHT: AVOID — worsen myoclonus + reduce VPA levels. | POLG MANDATORY before VPA." variant="danger" />
      <Alert text="🔵 VPA first-line (POLG done). ACTH+VGB for West. KD for DRE. CLB adjunct. Baclofen for dyskinesias — NOT seizures." variant="info" />

      <SectionCard title="💊 Treatments" borderColor={ACCENT3}>
        {(treatments || []).map((t, i) => {
          const isCI = t.level?.includes('CI') || t.level?.includes('ABSOLUTE') || t.level?.includes('WITHDRAWN');
          const isDyskinesia = t.drug?.includes('Baclofen');
          return (
            <div key={i} className="card mb-3 shadow-sm"
                 style={{ borderLeft: `4px solid ${isCI ? ACCENT2 : isDyskinesia ? '#7b1fa2' : ACCENT3}` }}>
              <div className="card-header py-1"
                   style={{ backgroundColor: isCI ? '#ffebee' : isDyskinesia ? '#f3e5f5' : '#e8f5e9' }}>
                <div className="d-flex justify-content-between align-items-start">
                  <span className="fw-bold small" style={{ color: isCI ? ACCENT2 : isDyskinesia ? '#7b1fa2' : ACCENT3 }}>
                    {t.drug}
                  </span>
                  <span className={`badge ${isCI ? 'bg-danger' : isDyskinesia ? '' : 'bg-success'}`}
                        style={isDyskinesia ? { backgroundColor: '#7b1fa2' } : {}}>
                    {t.level?.split(' ')[0]} {t.level?.split(' ')[1]}
                  </span>
                </div>
              </div>
              <div className="card-body py-2 small">
                <div className="mb-1"><strong>Dose:</strong> {t.dose}</div>
                <div className="mb-1"><strong>MOA:</strong> {t.moa}</div>
                <div className="mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
                <div className="mb-1 text-danger"><strong>Safety:</strong> {t.safety}</div>
                <div className="mb-1 text-muted"><strong>Monitoring:</strong> {t.monitoring}</div>
                {t.foxg1_specific && (
                  <div className={`alert py-1 mt-1 mb-0 alert-${isDyskinesia ? 'secondary' : 'warning'}`}
                       style={{ fontSize: 11 }}>
                    <strong>FOXG1-specific:</strong> {t.foxg1_specific}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </SectionCard>

      <SectionCard title="🚫 Contraindications" borderColor={ACCENT2}>
        {(cis || []).map((c, i) => (
          <div key={i} className="alert alert-danger mb-3" style={{ fontSize: 13 }}>
            <div className="fw-bold">⛔ {c.drug} — Risk: {c.risk}</div>
            <div className="mt-1">{c.reason}</div>
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
      <SectionCard title="📖 Key Concepts (14)" borderColor={ACCENT}>
        {(def.concepts || []).map((c, i) => (
          <div key={i} className="mb-3">
            <div className="fw-bold small" style={{ color: ACCENT }}>{c.term}</div>
            <div className="text-muted small">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📋 Monitoring Protocol" borderColor='#0277bd'>
        {(def.monitoring_full || []).map((m, i) => (
          <div key={i} className="card mb-2 shadow-sm">
            <div className="card-header d-flex justify-content-between py-1" style={{ backgroundColor: '#e3f2fd' }}>
              <span className="fw-bold small" style={{ color: '#0277bd' }}>{m.item}</span>
              <span className="badge bg-info text-dark">{m.frequency}</span>
            </div>
            <div className="card-body py-1 small text-muted">{m.rationale}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="⏳ Patient Lifecycle" borderColor={ACCENT4}>
        {(def.lifecycle || []).map((w, i) => (
          <div key={i} className="card mb-2 shadow-sm">
            <div className="card-header py-1" style={{ backgroundColor: '#fff3e0' }}>
              <span className="fw-bold small" style={{ color: ACCENT4 }}>{w.window}</span>
            </div>
            <div className="card-body py-2 small">
              <div><strong>Key events:</strong> {w.key_events}</div>
              <div className="mt-1 text-muted"><strong>Priority actions:</strong> {w.priority_actions}</div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Clinical Thresholds" borderColor={ACCENT2}>
        <ul className="small mb-0">
          {(def.thresholds || []).map((t, i) => <li key={i} className="mb-1">{t}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="📚 Standards & References" borderColor={ACCENT3}>
        <div className="row">
          <div className="col-md-6">
            <strong className="small">Standards:</strong>
            <ul className="small mt-1 mb-0">
              {(def.standards || []).map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </div>
          <div className="col-md-6">
            <strong className="small">References:</strong>
            <ul className="small mt-1 mb-0">
              {(def.references || []).map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function FOXG1Page() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [def, setDef] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/foxg1/overview`).then(r => r.json()),
      fetch(`${API}/api/foxg1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/foxg1/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1100 }}>
      {/* Header */}
      <div className="card mb-4 shadow" style={{ borderTop: `5px solid ${ACCENT}` }}>
        <div className="card-body py-3">
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>🧠 FOXG1 Syndrome (Congenital Rett Variant / FOXG1-Related DEE)</h4>
          <div className="small opacity-75 mt-1">
            Congenital-onset DEE · Forkhead Box G1 · 14q12 · De novo LOF / 14q12 deletion ·
            Hyperkinetic dyskinesias (NOT stereotypies) · Frontal hypoplasia · Equal sex ratio ·
            Tiagabine ABSOLUTE CI · CBZ/OXC/PHT avoid (myoclonus) · POLG mandatory before VPA · Baclofen for dyskinesias
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="mb-3">
        {TABS.map((t, i) => (
          <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />
        ))}
      </div>

      {/* Tab content */}
      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <EtiologyTab bd={bd} />}
      {tab === 2 && <SeizureTab bd={bd} />}
      {tab === 3 && <TreatmentsTab bd={bd} />}
      {tab === 4 && <DefinitionsTab def={def} />}

      <div className="text-muted small mt-3">
        FOXG1 Syndrome Dashboard · N=41 cohort · Sources: Ariani 2008 · Kortüm 2011 · Marwan 2012 ·
        Vegas 2018 · UKISS 2004 Lancet Neurol · ILAE-2022 · NICE-NG217 · FDA-SHARE-REMS · MHRA-PREVENT
      </div>
    </div>
  );
}
