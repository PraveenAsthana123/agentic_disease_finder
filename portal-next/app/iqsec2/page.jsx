'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1565c0';   // deep blue — IQSEC2 X-linked ArfGEF (Xp11.22, synaptic AMPAR trafficking)
const ACCENT2 = '#b71c1c';   // deep red — ABSOLUTE CI / HIGH RISK
const ACCENT3 = '#2e7d32';   // forest green — precision / emerging therapy
const ACCENT4 = '#6a1b9a';   // deep purple — X-linked / genetics

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

function Badge({ text, color }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: 11 }}>{text}</span>
  );
}

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading overview…</div>;
  const d = data;
  return (
    <>
      <Alert
        text="⚠️ IQSEC2 X-Linked DEE: PHT/CBZ/OXC HIGH RISK (myoclonic worsening; IV PHT ABSOLUTE CI for SE — use IV LEV). TGB ABSOLUTE CI (NCSE — non-verbal patients cannot report symptoms). LTG monotherapy HIGH RISK. LEV: behavioural toxicity caution in XLID/ASD. VGB: ERG mandatory every 3 months (non-verbal — cannot report visual loss). POLG1 before VPA."
        variant="danger"
      />
      <Alert
        text="🧬 IQSEC2 (Xp11.22) = ArfGEF BRAG1; activates Arf1/Arf3 GTPases → AMPA receptor (GluA1/GluA2) trafficking to postsynaptic density; pLI=1.00. X-linked dominant (de novo females ~75%). Myoclonic encephalopathy HALLMARK (90%). IS/West syndrome 65%. DRE 80%+. No single precision therapy approved (2024); perampanel emerging Level C."
        variant="info"
      />

      <div className="row g-2 mb-4">
        <KPI label="Cohort" value={d.cohort_size} color={ACCENT} />
        <KPI label="Females" value={`${d.female_pct}%`} color={ACCENT4} />
        <KPI label="DRE" value={`${d.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="IS/West" value={`${d.infantile_spasms_pct}%`} color={ACCENT} />
        <KPI label="Myoclonic" value={`${d.myoclonic_pct}%`} color={ACCENT2} />
        <KPI label="Absent Speech" value={`${d.absent_speech_pct}%`} color="#e65100" />
        <KPI label="ASD Features" value={`${d.asd_features_pct}%`} color={ACCENT4} />
        <KPI label="CSWS" value={`${d.csws_pct}%`} color="#37474f" />
        <KPI label="On KD" value={`${d.on_kd_pct}%`} color={ACCENT3} />
        <KPI label="On VPA" value={`${d.on_vpa_pct}%`} color={ACCENT3} />
        <KPI label="On LEV" value={`${d.on_lev_pct}%`} color="#0277bd" />
        <KPI label="Onset (mo)" value={d.mean_onset_months} color={ACCENT} />
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Gene Summary — IQSEC2 (Xp11.22)" borderColor={ACCENT4}>
            <table className="table table-sm small mb-0">
              <tbody>
                <tr><td className="fw-semibold">Gene</td><td>{d.gene}</td></tr>
                <tr><td className="fw-semibold">Inheritance</td><td>{d.inheritance}</td></tr>
                <tr><td className="fw-semibold">OMIM</td><td style={{ fontFamily: 'monospace' }}>{d.omim}</td></tr>
                <tr><td className="fw-semibold">X-Linked Note</td><td style={{ fontSize: 12 }}>{d.x_linked_clinical_note}</td></tr>
                <tr><td className="fw-semibold">Precision Rx</td><td>{d.precision_therapy}</td></tr>
              </tbody>
            </table>
          </SectionCard>

          <SectionCard title="Etiology Distribution" borderColor={ACCENT}>
            {Object.entries(d.etiology_distribution || {}).map(([k, v]) => (
              <PctBar key={k} label={`${k} (n=${v.n})`} pct={v.pct} color={ACCENT} />
            ))}
          </SectionCard>
        </div>

        <div className="col-md-6">
          <SectionCard title="Key Contraindications" borderColor={ACCENT2}>
            {(d.key_contraindications || []).map((ci, i) => (
              <div key={i} className="alert alert-danger py-1 mb-1" style={{ fontSize: 12 }}>
                🚫 {ci}
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Seizure Type Distribution" borderColor={ACCENT}>
            {(d.seizure_type_distribution || []).map(s => (
              <PctBar key={s.type} label={s.type} pct={s.pct} color={ACCENT} />
            ))}
          </SectionCard>

          <SectionCard title="Trigger Distribution" borderColor="#f57f17">
            {(d.trigger_distribution || []).map(t => (
              <PctBar key={t.trigger} label={t.trigger} pct={t.pct} color="#f57f17" />
            ))}
          </SectionCard>
        </div>
      </div>
    </>
  );
}

// ── Tab: Patients & Etiology ──────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { etiology_catalog, patient_sample } = data;
  return (
    <>
      <SectionCard title="Etiology Catalog — 5 Classes" borderColor={ACCENT}>
        {(etiology_catalog || []).map((e, i) => (
          <div key={i} className="card mb-3 shadow-sm border-0" style={{ borderLeft: `4px solid ${ACCENT}` }}>
            <div className="card-header py-1" style={{ backgroundColor: '#e3f2fd', fontSize: 13 }}>
              <strong>{e.etiology}</strong>
              <span className="badge ms-2" style={{ backgroundColor: ACCENT }}>{e.pct}% · n={e.n}</span>
            </div>
            <div className="card-body py-2" style={{ fontSize: 12 }}>
              <div><strong>Mechanism:</strong> {e.mechanism}</div>
              <div className="mt-1"><strong>EEG:</strong> {e.eeg_correlate}</div>
              <div className="mt-1"><strong>Onset:</strong> {e.typical_age_onset}</div>
              <div className="mt-1"><strong>DRE:</strong> {e.drug_resistance}</div>
              {e.x_inactivation_note && (
                <div className="mt-1 text-primary"><strong>X-Inactivation:</strong> {e.x_inactivation_note}</div>
              )}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Sample (n=15 of 40)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ backgroundColor: '#e3f2fd' }}>
              <tr>
                <th>ID</th><th>Name</th><th>Sex</th><th>Onset(mo)</th><th>Age</th>
                <th>DRE</th><th>IS</th><th>Myocl</th><th>Speech</th><th>ASD</th>
                <th>VPA</th><th>KD</th>
              </tr>
            </thead>
            <tbody>
              {(patient_sample || []).map(p => (
                <tr key={p.id}>
                  <td style={{ fontFamily: 'monospace' }}>{p.id}</td>
                  <td>{p.name}</td>
                  <td><Badge text={p.sex} color={p.sex === 'F' ? ACCENT4 : ACCENT} /></td>
                  <td>{p.age_onset_months}m</td>
                  <td>{p.current_age}y</td>
                  <td>{p.drug_resistant ? <Badge text="DRE" color={ACCENT2} /> : <span className="text-muted">–</span>}</td>
                  <td>{p.infantile_spasms ? <Badge text="IS" color={ACCENT} /> : '–'}</td>
                  <td>{p.myoclonic ? <Badge text="Myocl" color="#e65100" /> : '–'}</td>
                  <td>{p.absent_speech ? <Badge text="Absent" color="#37474f" /> : <span className="text-success small">Partial</span>}</td>
                  <td>{p.asd_features ? <Badge text="ASD" color={ACCENT4} /> : '–'}</td>
                  <td>{p.on_vpa ? <Badge text="VPA" color={ACCENT3} /> : '–'}</td>
                  <td>{p.on_kd ? <Badge text="KD" color="#f57f17" /> : '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Seizures & Triggers ──────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types, triggers } = data;
  return (
    <>
      <SectionCard title="Seizure Types — 5 Types (IQSEC2)" borderColor={ACCENT}>
        {(seizure_types || []).map((s, i) => (
          <div key={i} className="card mb-3 shadow-sm border-0">
            <div className="card-header py-1" style={{ backgroundColor: '#e3f2fd', fontSize: 13 }}>
              <strong>{s.type}</strong>
              <span className="badge ms-2" style={{ backgroundColor: ACCENT }}>{s.frequency_pct}%</span>
            </div>
            <div className="card-body py-2" style={{ fontSize: 12 }}>
              <div><strong>Mechanism:</strong> {s.mechanism}</div>
              <div className="mt-1"><strong>EEG:</strong> {s.eeg_pattern}</div>
              <div className="mt-1"><strong>Semiology:</strong> {s.semiology}</div>
              <div className="mt-1 text-warning-emphasis"><strong>Clinical Tips:</strong> {s.clinical_tips}</div>
              <div className="mt-1 text-success"><strong>Treatment:</strong> {s.treatment_note}</div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers — 8 Triggers" borderColor="#f57f17">
        {(triggers || []).map((t, i) => (
          <div key={i} className="d-flex align-items-start mb-3 p-2 rounded" style={{ background: '#fff8e1' }}>
            <div className="me-3 text-center" style={{ minWidth: 54 }}>
              <div className="fw-bold" style={{ color: '#f57f17', fontSize: 18 }}>{t.pct}%</div>
              <div style={{ fontSize: 10, color: '#888' }}>prevalence</div>
            </div>
            <div>
              <div className="fw-semibold" style={{ fontSize: 13 }}>{t.trigger}</div>
              <div className="text-muted" style={{ fontSize: 12 }}>{t.mechanism}</div>
            </div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments ───────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatments, contraindications, monitoring, lifecycle } = data;

  const ciColor = (level) => {
    if (level === 'ABSOLUTE CI') return ACCENT2;
    if (level === 'HIGH RISK') return '#e65100';
    if (level?.includes('CAUTION')) return '#f9a825';
    return '#37474f';
  };

  return (
    <>
      <SectionCard title="Treatments — 8 Agents (IQSEC2)" borderColor={ACCENT3}>
        {(treatments || []).map((t, i) => (
          <div key={i} className="card mb-3 border-0 shadow-sm">
            <div className="card-header py-1" style={{ backgroundColor: '#e8f5e9', fontSize: 13 }}>
              <strong>{t.drug}</strong>
              <span className="badge ms-2" style={{ backgroundColor: ACCENT3 }}>{t.level}</span>
            </div>
            <div className="card-body py-2" style={{ fontSize: 12 }}>
              <div><strong>Dose:</strong> {t.dose}</div>
              <div className="mt-1"><strong>MOA:</strong> {t.moa}</div>
              <div className="mt-1"><strong>Efficacy:</strong> {t.efficacy}</div>
              <div className="mt-1"><strong>Monitoring:</strong> {t.monitoring}</div>
              <div className="mt-1 text-primary"><strong>IQSEC2 Note:</strong> {t.iqsec2_note}</div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications — 6 CIs (IQSEC2)" borderColor={ACCENT2}>
        {(contraindications || []).map((c, i) => (
          <div key={i} className="card mb-3 border-0 shadow-sm" style={{ borderLeft: `4px solid ${ciColor(c.level)}` }}>
            <div className="card-header py-1" style={{ backgroundColor: '#ffebee', fontSize: 13 }}>
              <strong>🚫 {c.drug}</strong>
              <span className="badge ms-2" style={{ backgroundColor: ciColor(c.level) }}>{c.level}</span>
            </div>
            <div className="card-body py-2" style={{ fontSize: 12 }}>
              <div><strong>Reason:</strong> {c.reason}</div>
              <div className="mt-1 text-success"><strong>Alternative:</strong> {c.alternative}</div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring — 14 Items" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm small table-hover">
            <thead style={{ backgroundColor: '#e3f2fd' }}>
              <tr><th>Item</th><th>Frequency</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ fontSize: 12 }}>{m.item}</td>
                  <td style={{ fontSize: 11 }}>{m.frequency}</td>
                  <td style={{ fontSize: 11 }}>{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Lifecycle — 6 Stages" borderColor={ACCENT4}>
        {(lifecycle || []).map((l, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: '#f3e5f5' }}>
            <div className="fw-bold mb-1" style={{ color: ACCENT4, fontSize: 13 }}>{l.stage}</div>
            <div style={{ fontSize: 12 }}><strong>Key Issues:</strong> {l.key_issues}</div>
            <div className="mt-1" style={{ fontSize: 12 }}><strong>Action:</strong> {l.action}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { concepts, thresholds, standards, references } = data;
  return (
    <>
      <SectionCard title="Key Concepts — 15 Terms" borderColor={ACCENT}>
        <div className="row">
          {(concepts || []).map((c, i) => (
            <div key={i} className="col-md-6 mb-3">
              <div className="p-2 rounded" style={{ background: '#e3f2fd', fontSize: 12 }}>
                <div className="fw-bold mb-1" style={{ color: ACCENT }}>{c.term}</div>
                <div>{c.definition}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Clinical Thresholds — 12 Parameters" borderColor="#f57f17">
        <div className="table-responsive">
          <table className="table table-sm small table-hover">
            <thead style={{ backgroundColor: '#fff8e1' }}>
              <tr><th>Parameter</th><th>Value / Threshold</th><th>Action</th></tr>
            </thead>
            <tbody>
              {(thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ fontSize: 12 }}>{t.parameter}</td>
                  <td><Badge text={t.value} color="#f57f17" /></td>
                  <td style={{ fontSize: 11 }}>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Evidence Standards — 12 Standards" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm small table-hover">
            <thead style={{ backgroundColor: '#e8f5e9' }}>
              <tr><th>Code</th><th>Title</th><th>IQSEC2 Relevance</th></tr>
            </thead>
            <tbody>
              {(standards || []).map((s, i) => (
                <tr key={i}>
                  <td><Badge text={s.code} color={ACCENT3} /></td>
                  <td style={{ fontSize: 11 }}>{s.title}</td>
                  <td style={{ fontSize: 11 }}>{s.relevance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Key References — 6 Papers" borderColor={ACCENT4}>
        {(references || []).map((r, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f3e5f5', fontSize: 12 }}>
            <div className="fw-bold mb-1" style={{ color: ACCENT4 }}>{r.id}</div>
            <div className="mb-1">{r.citation}</div>
            <div className="text-muted"><strong>Key finding:</strong> {r.key_finding}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function IQSEC2Page() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('Overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/iqsec2/overview`).then(r => r.json()),
      fetch(`${API}/api/iqsec2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/iqsec2/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOverview(o); setBreakdown(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!overview) return <div className="text-muted p-3">Loading IQSEC2 dashboard…</div>;

  return (
    <div className="p-3">
      <h3 style={{ color: ACCENT }}>
        🧬 IQSEC2 Epilepsy Dashboard
      </h3>
      <p className="text-muted small mb-3">
        X-Linked DEE / ArfGEF BRAG1 / AMPAR Trafficking / Myoclonic Encephalopathy / IS-West / Xp11.22
        — {overview.cohort_size} patients · DRE {overview.drug_resistant_pct}% · Myoclonic {overview.myoclonic_pct}% · {overview.female_pct}% female
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === t ? ' active' : ''}`}
              style={tab === t ? { color: ACCENT, borderBottomColor: ACCENT, fontWeight: 600 } : {}}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'            && <OverviewTab   data={overview} />}
      {tab === 'Patients & Etiology' && <PatientsTab   data={breakdown} />}
      {tab === 'Seizures & Triggers' && <SeizuresTab   data={breakdown} />}
      {tab === 'Treatments'          && <TreatmentsTab data={breakdown} />}
      {tab === 'Definitions'         && <DefinitionsTab data={defs} />}
    </div>
  );
}
