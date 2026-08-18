'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#33691e';   // deep olive-green — GABA-B metabotropic receptor
const ACCENT2 = '#8a0000';   // crimson — ABSOLUTE CI / HIGH RISK
const ACCENT3 = '#1a5276';   // navy — precision therapy / Baclofen (LOF only)
const ACCENT4 = '#4a148c';   // deep purple — GOF constitutive signalling

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
        text="⚠️ GABBR2 DEE-59: GOF/LOF functional assay MANDATORY before any Baclofen decision — Baclofen is PRECISION Rx in LOF but CONTRAINDICATED in GOF. TGB ABSOLUTE CI (NCSE). PHT/CBZ/OXC HIGH RISK (worsen IS). NEVER stop Baclofen abruptly (medical emergency)."
        variant="danger"
      />
      <Alert
        text="🔬 GABA-B Receptor = GABBR1 (ligand-binding, 6p22.1) + GABBR2 (Gi-effector, 22q12.2) obligatory heterodimer. GABBR2 LOF → presynaptic autoreceptor loss → disinhibited GABA release → GABAA desensitisation → paradoxical excitation. GOF → constitutive Gi → excess tonic inhibition during development → compensatory excitatory upregulation → DEE."
        variant="info"
      />

      <div className="row g-2 mb-4">
        <KPI label="Cohort" value={d.cohort_size} color={ACCENT} />
        <KPI label="GOF Patients" value={d.gof_patients} color={ACCENT4} />
        <KPI label="LOF Patients" value={d.lof_patients} color={ACCENT3} />
        <KPI label="Drug-Resistant" value={`${d.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="West Syndrome (IS)" value={`${d.west_syndrome_pct}%`} color={ACCENT2} />
        <KPI label="LGS Evolution" value={`${d.lgs_evolution_pct}%`} color={ACCENT4} />
        <KPI label="On Baclofen (LOF)" value={`${d.on_baclofen_pct}%`} color={ACCENT3} />
        <KPI label="On KD" value={`${d.on_kd_pct}%`} color={ACCENT} />
        <KPI label="On VGB" value={`${d.on_vgb_pct}%`} color={ACCENT} />
        <KPI label="Functional Assay Done" value={`${d.functional_assay_done_pct}%`} color="#e65100" />
        <KPI label="Mean Onset (y)" value={d.mean_age_onset_years} color={ACCENT} />
      </div>

      <SectionCard title="GABA-B Receptor Biology — GABBR1 + GABBR2 Heterodimer" borderColor={ACCENT}>
        <div className="row">
          <div className="col-md-6">
            <h6 style={{ color: ACCENT4 }}>GOF Mechanism (55% — Severe DEE-59)</h6>
            <ul className="small">
              <li><strong>TM4-TM5 or VFTM missense</strong> → constitutive Gi activation</li>
              <li>Excess tonic inhibition during cortical development</li>
              <li>Compensatory excitatory synapse upregulation → net hyperexcitability</li>
              <li>West syndrome (IS) → LGS evolution in 60%</li>
              <li><strong style={{ color: ACCENT2 }}>Baclofen CONTRAINDICATED in GOF</strong></li>
            </ul>
          </div>
          <div className="col-md-6">
            <h6 style={{ color: ACCENT3 }}>LOF Mechanism (45% — GEFS+/Milder DEE)</h6>
            <ul className="small">
              <li><strong>Truncating/frameshift</strong> → haploinsufficiency</li>
              <li>Reduced GABA-B surface density → presynaptic autoreceptor loss</li>
              <li>Disinhibited GABA release → GABAA desensitisation → excitation</li>
              <li>GEFS+/generalised epilepsy; milder than GOF</li>
              <li><strong style={{ color: ACCENT3 }}>Baclofen PRECISION Rx in LOF (40-60% response)</strong></li>
            </ul>
          </div>
        </div>
        <div className="mt-3">
          <h6>Gi/Go Signal Transduction (GABBR2 effector module)</h6>
          <div className="row small">
            <div className="col-md-4">
              <Badge text="Gi → ↓cAMP → ↓PKA" color={ACCENT} />
              <div>Inhibits adenylyl cyclase → ↓protein synthesis/glutamate release</div>
            </div>
            <div className="col-md-4">
              <Badge text="Gβγ → GIRK K⁺ channels" color={ACCENT3} />
              <div>K⁺ efflux → slow IPSP 150-500 ms (postsynaptic hyperpolarisation)</div>
            </div>
            <div className="col-md-4">
              <Badge text="Gβγ → ↓Cav2.1/2.2" color={ACCENT} />
              <div>↓Presynaptic Ca²⁺ → ↓GABA/glutamate release (auto + heteroreceptors)</div>
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Etiology Distribution" borderColor={ACCENT4}>
        {d.etiology_distribution && Object.entries(d.etiology_distribution).map(([cat, v]) => (
          <PctBar key={cat} label={cat.replace(/-/g, ' ')} pct={v.pct}
            color={cat.includes('GOF') ? ACCENT4 : ACCENT3} />
        ))}
      </SectionCard>

      <SectionCard title="Seizure Type Frequency" borderColor={ACCENT}>
        {d.seizure_type_distribution?.map(s => (
          <PctBar key={s.type} label={s.type} pct={s.pct} color={ACCENT} />
        ))}
      </SectionCard>

      <SectionCard title="Trigger Frequency" borderColor="#e65100">
        {d.trigger_distribution?.map(t => (
          <PctBar key={t.trigger} label={t.trigger} pct={t.pct} color="#e65100" />
        ))}
      </SectionCard>

      <SectionCard title="Key Contraindications" borderColor={ACCENT2}>
        {d.key_contraindications?.map((ci, i) => (
          <Alert key={i} text={`🚫 ${ci}`} variant="danger" />
        ))}
      </SectionCard>

      <div className="card shadow-sm mb-4" style={{ borderLeft: `4px solid ${ACCENT3}` }}>
        <div className="card-body">
          <h6 className="fw-bold" style={{ color: ACCENT3 }}>Precision Therapy</h6>
          <p className="mb-0 small">{d.precision_therapy}</p>
          <div className="mt-2 small text-muted">
            Gene: {d.gene} · Receptor: {d.receptor} · Inheritance: {d.inheritance}
          </div>
          <div className="small text-muted">OMIM: {d.omim}</div>
        </div>
      </div>
    </>
  );
}

// ── Tab: Patients & Etiology ──────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { etiology_catalog = [], patient_sample = [] } = data;
  return (
    <>
      <Alert
        text="⚠️ GOF/LOF FUNCTIONAL ASSAY mandatory before ANY Baclofen decision. GOF (TM4-TM5/VFTM) = Baclofen CI. LOF (truncating/missense) = Baclofen precision Rx. Cannot determine from variant type alone — functional assay in Xenopus/HEK293."
        variant="danger"
      />
      <SectionCard title="Etiology Catalog (5 classes)" borderColor={ACCENT4}>
        {etiology_catalog.map((e, i) => (
          <div key={i} className="mb-4 pb-3 border-bottom">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <strong style={{ color: e.category.includes('GOF') ? ACCENT4 : ACCENT3 }}>
                {e.etiology}
              </strong>
              <span className="badge ms-2" style={{ backgroundColor: e.category.includes('GOF') ? ACCENT4 : ACCENT3 }}>
                {e.pct}% (n={e.n})
              </span>
            </div>
            <p className="small text-muted mb-1">{e.mechanism}</p>
            <div className="small"><strong>EEG:</strong> {e.eeg_correlate}</div>
            <div className="small"><strong>Onset:</strong> {e.typical_age_onset} · <strong>DRE:</strong> {e.drug_resistance}</div>
            <div className="small"><strong>Baclofen:</strong> <span style={{ color: e.baclofen_role?.includes('CI') || e.baclofen_role?.includes('CONTRA') ? ACCENT2 : ACCENT3 }}>{e.baclofen_role}</span></div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Sample (first 15 of 40)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Name</th><th>Sex</th><th>GOF/LOF</th>
                <th>Onset (y)</th><th>Age (y)</th><th>DRE</th>
                <th>West</th><th>LGS</th><th>Baclofen</th><th>KD</th><th>Assay</th>
              </tr>
            </thead>
            <tbody>
              {patient_sample.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.name}</td>
                  <td>{p.sex}</td>
                  <td>
                    <span className="badge" style={{ backgroundColor: p.gof_lof === 'GOF' ? ACCENT4 : ACCENT3 }}>
                      {p.gof_lof}
                    </span>
                  </td>
                  <td>{p.age_onset}</td>
                  <td>{p.current_age}</td>
                  <td>{p.drug_resistant ? <span className="text-danger fw-bold">DRE</span> : '—'}</td>
                  <td>{p.west_syndrome ? '✓' : '—'}</td>
                  <td>{p.lgs_evolution ? '✓' : '—'}</td>
                  <td>{p.on_baclofen ? <span style={{ color: ACCENT3 }}>✓</span> : '—'}</td>
                  <td>{p.on_kd ? '✓' : '—'}</td>
                  <td>{p.functional_assay_done ? <span style={{ color: ACCENT }}>✓</span> : <span className="text-danger">PENDING</span>}</td>
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
  const { seizure_types = [], triggers = [] } = data;
  return (
    <>
      <Alert
        text="IS = EMERGENCY: Start ACTH within 2 weeks of onset — every day of delay worsens cognitive outcome. AVOID PHT/CBZ/OXC — worsen IS and myoclonus. TGB ABSOLUTE CI."
        variant="danger"
      />
      <SectionCard title="Seizure Types (5 types)" borderColor={ACCENT}>
        {seizure_types.map((s, i) => (
          <div key={i} className="mb-4 pb-3 border-bottom">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong style={{ color: ACCENT }}>{s.type}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{s.frequency_pct}%</span>
            </div>
            <PctBar label="" pct={s.frequency_pct} color={ACCENT} />
            <div className="small mb-1"><strong>Semiology:</strong> {s.semiology}</div>
            <div className="small mb-1"><strong>EEG tip:</strong> {s.eeg_tip}</div>
            <div className="small text-warning bg-dark p-1 rounded">
              <strong>Clinical tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Triggers (8 triggers)" borderColor="#e65100">
        {triggers.map((t, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong>{t.trigger}</strong>
              <span className="badge" style={{ backgroundColor: '#e65100' }}>{t.pct}%</span>
            </div>
            <PctBar label="" pct={t.pct} color="#e65100" />
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments ───────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatments = [], contraindications = [], monitoring = [], lifecycle = [] } = data;

  const levelColor = (lvl) => {
    if (lvl?.includes('A')) return '#1a5276';
    if (lvl?.includes('B')) return '#1e8449';
    return '#7d6608';
  };

  return (
    <>
      <Alert
        text="🎯 BACLOFEN PRECISION: ONLY in LOF — functional assay mandatory first. ACTH/VGB = Level A for IS. NEVER stop Baclofen abruptly. TGB ABSOLUTE CI. PHT/CBZ HIGH RISK (worsen IS)."
        variant="danger"
      />

      <SectionCard title="Treatments (8)" borderColor={ACCENT3}>
        {treatments.map((t, i) => (
          <div key={i} className="mb-4 pb-3 border-bottom">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <strong style={{ color: levelColor(t.level) }}>{t.name}</strong>
              <span className="badge ms-2" style={{ backgroundColor: levelColor(t.level) }}>{t.level}</span>
            </div>
            <div className="small mb-1"><strong>Indication:</strong> {t.indication}</div>
            <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>
            <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
            <div className="small p-2 rounded" style={{ backgroundColor: '#f1f8e9', borderLeft: `3px solid ${ACCENT}` }}>
              <strong>GABBR2 note:</strong> {t.gabbr2_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications (6)" borderColor={ACCENT2}>
        {contraindications.map((ci, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <strong style={{ color: ACCENT2 }}>🚫 {ci.drug}</strong>
              <span className="badge ms-2" style={{ backgroundColor: ci.level.includes('ABSOLUTE') ? ACCENT2 : '#e65100' }}>
                {ci.level}
              </span>
            </div>
            <div className="small mb-1">{ci.reason}</div>
            <div className="small text-success"><strong>Alternative:</strong> {ci.alternative}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring (14 items)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead className="table-light">
              <tr><th>Item</th><th>Frequency</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {monitoring.map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold">{m.item}</td>
                  <td className="text-muted">{m.frequency}</td>
                  <td>{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Lifecycle Stages (7)" borderColor={ACCENT4}>
        {lifecycle.map((l, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="fw-bold" style={{ color: ACCENT4 }}>{l.stage}</div>
            <div className="small mb-1"><strong>Issues:</strong> {l.key_issues}</div>
            <div className="small text-muted"><strong>Action:</strong> {l.action}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { concepts = [], thresholds = [], standards = [], references = [] } = data;
  return (
    <>
      <SectionCard title="15 Core Concepts" borderColor={ACCENT}>
        {concepts.map((c, i) => (
          <div key={i} className="mb-2 pb-2 border-bottom">
            <span className="fw-bold me-2" style={{ color: ACCENT }}>{c.term}:</span>
            <span className="small">{c.definition}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="12 Clinical Thresholds" borderColor="#e65100">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead className="table-light">
              <tr><th>Parameter</th><th>Value</th><th>Action</th></tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.parameter}</td>
                  <td><Badge text={t.value} color="#e65100" /></td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="12 Evidence Standards" borderColor={ACCENT3}>
        {standards.map((s, i) => (
          <div key={i} className="mb-2 pb-1 border-bottom small">
            <Badge text={s.code} color={ACCENT3} />
            <strong> {s.title}</strong>
            <span className="text-muted ms-2">— {s.relevance}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="6 Key References" borderColor={ACCENT4}>
        {references.map((r, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom small">
            <div className="fw-bold" style={{ color: ACCENT4 }}>{r.id}</div>
            <div className="fst-italic text-muted">{r.citation}</div>
            <div><strong>Key finding:</strong> {r.key_finding}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function GABBR2Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/gabbr2/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (activeTab === 1 || activeTab === 2 || activeTab === 3) {
      if (!breakdown) {
        fetch(`${API}/api/gabbr2/breakdown`)
          .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
      }
    }
    if (activeTab === 4 && !definitions) {
      fetch(`${API}/api/gabbr2/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
    }
  }, [activeTab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3" style={{ borderLeft: `6px solid ${ACCENT}`, paddingLeft: 16 }}>
        <h3 className="fw-bold mb-0" style={{ color: ACCENT }}>
          🧬 GABBR2 Epilepsy
        </h3>
        <div className="text-muted small">
          DEE-59 · GABA-B Receptor Subunit 2 · GOF/LOF · Baclofen Precision (LOF) ·
          TGB ABSOLUTE CI · West Syndrome → LGS · 22q12.2 · OMIM #617137
        </div>
        <div className="mt-1">
          <Badge text="22q12.2" color={ACCENT} />
          <Badge text="De novo dominant" color={ACCENT} />
          <Badge text="GOF 55% — Severe DEE" color={ACCENT4} />
          <Badge text="LOF 45% — GEFS+" color={ACCENT3} />
          <Badge text="Baclofen LOF only" color={ACCENT3} />
          <Badge text="TGB ABSOLUTE CI" color={ACCENT2} />
          <Badge text="OMIM #617137" color="#555" />
        </div>
      </div>

      {error && <div className="alert alert-danger">API error: {error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((tab, i) => (
          <li key={tab} className="nav-item">
            <button
              className={`nav-link ${activeTab === i ? 'active' : ''}`}
              onClick={() => setActiveTab(i)}
            >{tab}</button>
          </li>
        ))}
      </ul>

      <div>
        {activeTab === 0 && <OverviewTab data={overview} />}
        {activeTab === 1 && <PatientsTab data={breakdown} />}
        {activeTab === 2 && <SeizuresTab data={breakdown} />}
        {activeTab === 3 && <TreatmentsTab data={breakdown} />}
        {activeTab === 4 && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
