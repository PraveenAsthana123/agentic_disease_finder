'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4a148c';   // deep purple — MEF2C master transcription factor / neuronal gene regulation
const ACCENT2 = '#b71c1c';   // deep red — ABSOLUTE CI / HIGH RISK
const ACCENT3 = '#1b5e20';   // deep green — VPA HDAC rationale / precision
const ACCENT4 = '#e65100';   // deep orange — photosensitivity / CSWS warning

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

function Badge({ text, color }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: 11 }}>{text}</span>
  );
}

// ── Tab 1: Overview ───────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Alert
        text="⚠️ MEF2C GENERALISED EPILEPSY: PHT/CBZ/OXC Na-channel blockers HIGH RISK — paradoxically worsen myoclonic seizures in generalised epilepsy (55% MEF2C have myoclonic); IV PHT/fosphenytoin CONTRAINDICATED for SE — use IV LEV. LTG monotherapy HIGH RISK for myoclonic aggravation. TGB ABSOLUTE CI (NCSE)."
        variant="danger"
      />
      <Alert
        text="🧬 VPA HDAC EPIGENETIC RATIONALE: VPA inhibits HDAC4/5 → dissociation from MEF2C C-terminal → partial restoration of MEF2C transcriptional targets (GABAergic interneuron genes, Arc/PSD-95 synaptic pruning genes); unique MEF2C-specific mechanism beyond anticonvulsant effect alone."
        variant="info"
      />
      <Alert
        text="📸 PHOTOSENSITIVITY 35%: MEF2C GABAergic interneuron LOF → visual cortex hyperexcitability → photoparoxysmal response in 35% (higher than most DEE). IPS EEG at diagnosis + annually. Blue-light 450 nm filter glasses. CSWS screen (annual overnight EEG mandatory — 8% CSWS, non-convulsive regression risk)."
        variant="warning"
      />

      <div className="row g-3 mb-4">
        <KPI label="Cohort" value={data.cohort_size} color={ACCENT} />
        <KPI label="Seizures" value={`${data.seizure_prevalence_pct}%`} color={ACCENT2} />
        <KPI label="Drug-Resistant" value={`${data.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Infantile Spasms" value={`${data.infantile_spasms_pct}%`} color={ACCENT4} />
        <KPI label="Myoclonic" value={`${data.myoclonic_pct}%`} color={ACCENT4} />
        <KPI label="Photosensitive" value={`${data.photosensitive_pct}%`} color={ACCENT4} />
        <KPI label="CSWS +" value={`${data.csws_positive_pct}%`} color={ACCENT} />
        <KPI label="On KD" value={`${data.on_kd_pct}%`} color={ACCENT3} />
        <KPI label="VPPP Applicable" value={`${data.vppp_applicable_pct}%`} color={ACCENT} />
        <KPI label="Mean AEDs" value={data.mean_aed_count} color={ACCENT} />
        <KPI label="Etiology Classes" value={data.etiology_classes} color={ACCENT} />
        <KPI label="Concepts" value={data.concepts} color={ACCENT} />
      </div>

      <SectionCard title="Gene Summary" borderColor={ACCENT}>
        <table className="table table-sm table-borderless mb-0" style={{ fontSize: 13 }}>
          <tbody>
            <tr><td className="fw-bold text-muted" style={{ width: 200 }}>Gene</td><td>{data.gene} ({data.locus})</td></tr>
            <tr><td className="fw-bold text-muted">Protein</td><td>{data.protein}</td></tr>
            <tr><td className="fw-bold text-muted">OMIM Syndrome</td><td>{data.omim_syndrome}</td></tr>
            <tr><td className="fw-bold text-muted">OMIM Gene</td><td>{data.omim_gene}</td></tr>
            <tr><td className="fw-bold text-muted">Interneuron Mechanism</td><td style={{ color: ACCENT4 }}><strong>{data.interneuron_mechanism}</strong></td></tr>
            <tr><td className="fw-bold text-muted">Rett-Like Phenotype</td><td style={{ color: ACCENT }}>{data.rett_like_phenotype}</td></tr>
            <tr><td className="fw-bold text-muted">VPA HDAC Rationale</td><td style={{ color: ACCENT3 }}>{data.vpa_hdac_rationale}</td></tr>
            <tr><td className="fw-bold text-muted">Photosensitivity Note</td><td style={{ color: ACCENT4 }}>{data.photosensitivity_note}</td></tr>
            <tr><td className="fw-bold text-muted">Key CIs</td><td><span style={{ color: ACCENT2 }}>{data.key_ci}</span></td></tr>
          </tbody>
        </table>
      </SectionCard>

      <SectionCard title="Standards & References" borderColor={ACCENT}>
        <div className="d-flex flex-wrap gap-1">
          {(data.standards || []).map(s => <Badge key={s} text={s} color={ACCENT} />)}
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Etiology ────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { etiologies = [], patients = [] } = data;
  return (
    <div>
      <SectionCard title="Etiology Classes (5 classes — 40 patients)" borderColor={ACCENT}>
        {etiologies.map((e, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ background: '#f3e5f5', borderLeft: `4px solid ${ACCENT}` }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <strong style={{ color: ACCENT }}>{e.label}</strong>
              <span className="badge ms-2" style={{ background: ACCENT }}>{e.pct}%</span>
            </div>
            <PctBar label="" pct={e.pct} color={ACCENT} />
            <p className="small mb-2">{e.mechanism}</p>
            <div className="row g-2" style={{ fontSize: 12 }}>
              <div className="col-md-4"><strong>Inheritance:</strong> {e.inheritance}</div>
              <div className="col-md-4"><strong>Diagnostic Clock:</strong> {e.diagnostic_clock}</div>
              <div className="col-md-4"><strong>Management:</strong> {e.management_note}</div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead style={{ background: ACCENT, color: '#fff' }}>
              <tr>
                <th>ID</th><th>Name</th><th>Age Dx (y)</th><th>Etiology</th>
                <th>Seizures</th><th>IS</th><th>DRE</th><th>Myoclonic</th>
                <th>Photo</th><th>CSWS</th><th>AEDs</th><th>KD</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.id}>
                  <td className="fw-bold">{p.id}</td>
                  <td>{p.name}</td>
                  <td>{p.age_dx}</td>
                  <td style={{ maxWidth: 160, fontSize: 11 }}>{p.etiology}</td>
                  <td>{p.has_seizures ? '✓' : '—'}</td>
                  <td>{p.is_onset ? <span style={{ color: ACCENT2 }}>IS</span> : '—'}</td>
                  <td>{p.drug_resistant ? <span style={{ color: ACCENT2 }}>DRE</span> : '—'}</td>
                  <td>{p.myoclonic ? <span style={{ color: ACCENT4 }}>Myo</span> : '—'}</td>
                  <td>{p.photosensitive ? <span style={{ color: ACCENT4 }}>PS</span> : '—'}</td>
                  <td>{p.csws_positive ? <span style={{ color: ACCENT2 }}>CSWS</span> : '—'}</td>
                  <td>{p.n_aed}</td>
                  <td>{p.on_kd ? <span style={{ color: ACCENT3 }}>KD</span> : '—'}</td>
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
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types = [], triggers = [] } = data;
  return (
    <div>
      <SectionCard title="Seizure Types (5 types)" borderColor={ACCENT}>
        {seizure_types.map((s, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ background: '#f3e5f5', borderLeft: `4px solid ${ACCENT}` }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <strong style={{ color: ACCENT }}>{s.type}</strong>
              <span className="badge" style={{ background: ACCENT2 }}>{s.pct}%</span>
            </div>
            <PctBar label="" pct={s.pct} color={ACCENT} />
            <div className="row g-2 mb-2" style={{ fontSize: 12 }}>
              <div className="col-md-12"><strong>EEG:</strong> {s.eeg}</div>
            </div>
            <p className="small mb-2">{s.semiology}</p>
            <div className="alert alert-info py-1 mb-0" style={{ fontSize: 12 }}>
              <strong>Clinical Tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers (8 triggers)" borderColor={ACCENT4}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between mb-1">
              <strong style={{ fontSize: 13 }}>{t.trigger}</strong>
              <span className="badge" style={{ background: ACCENT4 }}>{t.pct}%</span>
            </div>
            <PctBar label="" pct={t.pct} color={ACCENT4} />
            <p className="small text-muted mb-0">{t.management}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Treatments ─────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatments = [], contraindications = [], monitoring = [] } = data;
  return (
    <div>
      <Alert
        text="⛔ ABSOLUTE CI: Tiagabine (TGB) — NCSE risk (GABAergic desensitisation amplified by interneuron deficit); VPA in POLG1 biallelic — Alpers-Huttenlocher fatal. ⚠️ HIGH RISK: PHT/CBZ/OXC — myoclonic worsening in generalised epilepsy; LTG monotherapy — myoclonic aggravation; Abrupt AED withdrawal — cluster/SE trigger; VGB without ERG monitoring (SHARE REMS mandatory)."
        variant="danger"
      />

      <SectionCard title="Treatments (8 agents)" borderColor={ACCENT3}>
        {treatments.map((t, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ background: '#e8f5e9', borderLeft: `4px solid ${ACCENT3}` }}>
            <div className="d-flex justify-content-between align-items-start mb-2">
              <strong style={{ color: ACCENT3, fontSize: 15 }}>{t.drug}</strong>
              <span className="badge" style={{ background: ACCENT }}>{t.level}</span>
            </div>
            <div className="row g-2 mb-2" style={{ fontSize: 12 }}>
              <div className="col-md-6"><strong>Indication:</strong> {t.indication}</div>
              <div className="col-md-6"><strong>Dose:</strong> {t.dose}</div>
              <div className="col-md-12"><strong>MOA:</strong> {t.moa}</div>
              <div className="col-md-12"><strong>Monitoring:</strong> {t.monitoring}</div>
            </div>
            <div className="alert alert-primary py-1 mb-0" style={{ fontSize: 12 }}>
              <strong>MEF2C-Specific Note:</strong> {t.mef2c_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications (6 items)" borderColor={ACCENT2}>
        {contraindications.map((ci, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: '#ffebee', borderLeft: `4px solid ${ACCENT2}` }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <strong style={{ color: ACCENT2 }}>{ci.drug}</strong>
              <span className="badge" style={{ background: ci.level.includes('ABSOLUTE') ? '#b71c1c' : '#e65100' }}>
                {ci.level}
              </span>
            </div>
            <p className="small mb-1">{ci.reason}</p>
            <div className="text-success small"><strong>Alternative:</strong> {ci.alternative}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Protocol (14 items)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead style={{ background: ACCENT, color: '#fff' }}>
              <tr><th>Monitoring Item</th><th>Frequency</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {monitoring.map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ minWidth: 200 }}>{m.item}</td>
                  <td style={{ minWidth: 160 }}>{m.frequency}</td>
                  <td style={{ fontSize: 11 }}>{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 5: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { concepts = [], thresholds = [], references = [], standards = [] } = data;
  return (
    <div>
      <SectionCard title="Key Concepts (15)" borderColor={ACCENT}>
        {concepts.map((c, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: '#f3e5f5', borderLeft: `4px solid ${ACCENT}` }}>
            <strong style={{ color: ACCENT }}>{c.concept}</strong>
            <p className="small mb-0 mt-1">{c.definition}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds (12)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead style={{ background: ACCENT4, color: '#fff' }}>
              <tr><th>Threshold</th><th>Value</th><th>Action</th></tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.name}</td>
                  <td style={{ color: ACCENT4 }}>{t.value}</td>
                  <td style={{ fontSize: 11 }}>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Evidence References (6)" borderColor={ACCENT3}>
        {references.map((r, i) => (
          <div key={i} className="mb-3">
            <strong style={{ color: ACCENT3 }}>{r.ref}</strong>
            <p className="small mb-1">{r.citation}</p>
            <p className="small text-muted mb-0"><strong>Key Finding:</strong> {r.key_finding}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Standards (12)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead style={{ background: ACCENT, color: '#fff' }}>
              <tr><th>Standard</th><th>Scope</th></tr>
            </thead>
            <tbody>
              {standards.map((s, i) => (
                <tr key={i}>
                  <td className="fw-bold">{s.standard}</td>
                  <td>{s.scope}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Lifecycle sidebar ─────────────────────────────────────────────────────────
function LifecycleSidebar({ lifecycle }) {
  if (!lifecycle || !lifecycle.length) return null;
  return (
    <div className="card shadow-sm mb-4" style={{ borderLeft: `4px solid ${ACCENT}` }}>
      <div className="card-header fw-bold" style={{ backgroundColor: '#f3e5f5', color: ACCENT }}>
        Lifecycle Stages (6)
      </div>
      <div className="card-body p-2">
        {lifecycle.map((stage, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: i % 2 === 0 ? '#f3e5f5' : '#fff', fontSize: 12 }}>
            <div className="fw-bold" style={{ color: ACCENT }}>{stage.stage}</div>
            <div className="mb-1"><strong>Key issues:</strong> {(stage.key_issues || []).join(' · ')}</div>
            <div className="mb-1"><strong>Actions:</strong> {(stage.actions || []).join(' · ')}</div>
            <div><strong>AED:</strong> <em>{stage.aed}</em></div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MEF2CDashboard() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/mef2c/overview`)
      .then(r => r.json()).then(setOverview).catch(() => setError('Overview fetch failed'));
    fetch(`${API}/api/mef2c/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => setError('Breakdown fetch failed'));
    fetch(`${API}/api/mef2c/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => setError('Definitions fetch failed'));
  }, []);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1400 }}>
      {/* Header */}
      <div className="p-4 mb-4 rounded shadow-sm text-white"
        style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #7b1fa2 100%)` }}>
        <h1 className="h4 fw-bold mb-1">MEF2C Epilepsy Dashboard</h1>
        <div style={{ fontSize: 13, opacity: 0.92 }}>
          MEF2C Haploinsufficiency Syndrome (MHS) · 5q14.3 · De novo dominant LOF · OMIM #613443
        </div>
        <div style={{ fontSize: 12, opacity: 0.80 }}>
          Master Neuronal Transcription Factor · MADS-Box · GABAergic Interneuron Specification ·
          Rett-Like Phenotype Without MECP2 · Photosensitivity 35% · CSWS 8% · VPA HDAC Epigenetic Rationale
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {/* Nav tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      <div className="row">
        {/* Main content */}
        <div className="col-lg-9">
          {tab === 0 && <OverviewTab data={overview} />}
          {tab === 1 && <PatientsTab data={breakdown} />}
          {tab === 2 && <SeizuresTab data={breakdown} />}
          {tab === 3 && <TreatmentsTab data={breakdown} />}
          {tab === 4 && <DefinitionsTab data={definitions} />}
        </div>

        {/* Sidebar */}
        <div className="col-lg-3">
          <LifecycleSidebar lifecycle={breakdown?.lifecycle} />

          <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
            <div className="card-header fw-bold" style={{ background: '#ffebee', color: ACCENT2, fontSize: 13 }}>
              ⛔ Critical CI Summary
            </div>
            <div className="card-body p-2" style={{ fontSize: 12 }}>
              <div className="mb-2"><strong style={{ color: '#b71c1c' }}>ABSOLUTE CI:</strong><br />
                TGB (NCSE — GABAergic desensitisation)<br />
                VPA + POLG1 biallelic (Alpers-Huttenlocher fatal)
              </div>
              <div className="mb-2"><strong style={{ color: ACCENT4 }}>HIGH RISK:</strong><br />
                PHT / CBZ / OXC (myoclonic worsening)<br />
                LTG monotherapy (myoclonic aggravation)<br />
                Abrupt AED withdrawal (cluster → SE)<br />
                VGB without ERG/REMS monitoring
              </div>
              <div><strong style={{ color: ACCENT3 }}>PRECISION:</strong><br />
                VPA HDAC rationale (MEF2C-specific)<br />
                KD early at AED failure #2<br />
                ACTH + VGB simultaneous for IS<br />
                POLG1 screen MANDATORY before VPA
              </div>
            </div>
          </div>

          <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
            <div className="card-header fw-bold" style={{ background: '#fff3e0', color: ACCENT4, fontSize: 13 }}>
              📸 Photosensitivity Protocol
            </div>
            <div className="card-body p-2" style={{ fontSize: 12 }}>
              <p className="mb-1">35% MEF2C → photoparoxysmal response (higher than most DEE)</p>
              <ul className="ps-3 mb-0">
                <li>IPS EEG at diagnosis + annually</li>
                <li>Blue-light 450 nm filter glasses</li>
                <li>No stroboscopic environments</li>
                <li>VPA preferred over LTG (anti-photic)</li>
                <li>Screen time limit 1-2 h/day</li>
              </ul>
            </div>
          </div>

          <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT}` }}>
            <div className="card-header fw-bold" style={{ background: '#f3e5f5', color: ACCENT, fontSize: 13 }}>
              🧬 MEF2C at a Glance
            </div>
            <div className="card-body p-2" style={{ fontSize: 12 }}>
              <p className="mb-1"><strong>Gene:</strong> MEF2C (5q14.3)</p>
              <p className="mb-1"><strong>Mechanism:</strong> GABAergic interneuron LOF → E/I imbalance</p>
              <p className="mb-1"><strong>Rett-like:</strong> Stereotypy without MECP2</p>
              <p className="mb-1"><strong>CSWS:</strong> Annual sleep EEG (8% — silent regression)</p>
              <p className="mb-0"><strong>POLG1:</strong> Screen BEFORE VPA (mandatory)</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
