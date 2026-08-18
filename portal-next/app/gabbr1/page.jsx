'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#880e4f';   // deep burgundy — GABBR1 ligand-binding (GABA-B subunit 1, Venus flytrap)
const ACCENT2 = '#8a0000';   // crimson — ABSOLUTE CI / HIGH RISK
const ACCENT3 = '#1a5276';   // navy — precision therapy / Baclofen (LOF only)
const ACCENT4 = '#6a1520';   // dark rose — GOF / GABBR1a isoform

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
        text="⚠️ GABBR1 GEFS+/Focal: Functional assay (VFTM GABA dose-response) MANDATORY before any Baclofen decision — LOF = Precision Rx (Level C); GOF = ABSOLUTE CI. TGB ABSOLUTE CI (NCSE). PHT/CBZ/OXC HIGH RISK. LTG monotherapy HIGH RISK if myoclonic. NEVER stop Baclofen abruptly."
        variant="danger"
      />
      <Alert
        text="🔬 GABBR1 (6p22.1) = GABA-B ligand-binding subunit (Venus flytrap domain). GABBR1a (sushi repeat, presynaptic autoreceptor) vs GABBR1b (postsynaptic GIRK). MILDER than GABBR2 DEE-59: GEFS+/focal; no West syndrome typical; cognition usually preserved. Both TGB ABSOLUTE CI."
        variant="info"
      />

      <div className="row g-2 mb-4">
        <KPI label="Cohort" value={d.cohort_size} color={ACCENT} />
        <KPI label="LOF Patients" value={d.lof_patients} color={ACCENT3} />
        <KPI label="GOF Patients" value={d.gof_patients} color={ACCENT4} />
        <KPI label="Drug-Resistant" value={`${d.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="GEFS+" value={`${d.gefs_plus_pct}%`} color={ACCENT} />
        <KPI label="Absence Seizures" value={`${d.absence_seizures_pct}%`} color={ACCENT} />
        <KPI label="Focal Seizures" value={`${d.focal_seizures_pct}%`} color={ACCENT4} />
        <KPI label="Cognition Preserved" value={`${d.cognitive_preserved_pct}%`} color="#2e7d32" />
        <KPI label="On Baclofen (LOF)" value={`${d.on_baclofen_pct}%`} color={ACCENT3} />
        <KPI label="On VPA" value={`${d.on_vpa_pct}%`} color={ACCENT} />
        <KPI label="Functional Assay Done" value={`${d.functional_assay_done_pct}%`} color="#e65100" />
        <KPI label="Mean Onset (y)" value={d.mean_age_onset_years} color={ACCENT} />
      </div>

      <SectionCard title="GABBR1 Isoforms — GABBR1a (Presynaptic) vs GABBR1b (Postsynaptic)" borderColor={ACCENT4}>
        <div className="row">
          <div className="col-md-6">
            <h6 style={{ color: ACCENT4 }}>GABBR1a (Sushi Repeat+, Exon 1a)</h6>
            <ul className="small">
              <li><strong>38-aa sushi repeat</strong> binds fibronectin → presynaptic targeting</li>
              <li>Presynaptic <strong>autoreceptor</strong> on GABAergic terminals (GABA feedback)</li>
              <li>Presynaptic <strong>heteroreceptor</strong> on glutamatergic terminals (suppresses Glu)</li>
              <li>LOF → focal epilepsy (presynaptic autoreceptor loss)</li>
              <li><strong>Best baclofen target</strong> (intact GABBR1a on normal allele)</li>
            </ul>
          </div>
          <div className="col-md-6">
            <h6 style={{ color: ACCENT3 }}>GABBR1b (No Sushi, Postsynaptic)</h6>
            <ul className="small">
              <li>Dendritic/postsynaptic localisation (no sushi repeat)</li>
              <li>Activates <strong>GIRK K⁺ channels</strong> → slow IPSP 150-500 ms</li>
              <li>Modulates thalamo-cortical excitability (absence / GEFS+)</li>
              <li>LOF → generalised epilepsy, absence (thalamo-cortical resonance)</li>
              <li>ESM directly complements GABBR1b LOF (T-type Ca²⁺ block)</li>
            </ul>
          </div>
        </div>
        <div className="mt-2 small text-muted">
          Both isoforms require GABBR2 co-assembly (obligatory heterodimer). GABA binds ONLY to GABBR1 VFT domain — GABBR2 cannot bind GABA alone.
        </div>
      </SectionCard>

      <SectionCard title="GABBR1 vs GABBR2 — Severity Comparison" borderColor={ACCENT}>
        <div className="row small">
          <div className="col-md-6">
            <div className="fw-bold mb-1" style={{ color: ACCENT }}>GABBR1 (6p22.1) — THIS DASHBOARD</div>
            <ul>
              <li>MILDER: GEFS+, absence, focal epilepsy</li>
              <li>Mean onset: 3-8 years (school age)</li>
              <li>DR: ~30% (vs GABBR2 GOF 80-90%)</li>
              <li>West syndrome: NOT typical</li>
              <li>Cognition: usually preserved</li>
              <li>Baclofen Level C LOF (0.5-1.5 mg/kg/day)</li>
              <li>OMIM gene *603540 (no dedicated DEE OMIM#)</li>
            </ul>
          </div>
          <div className="col-md-6">
            <div className="fw-bold mb-1" style={{ color: ACCENT2 }}>GABBR2 (22q12.2) — DEE-59</div>
            <ul>
              <li>SEVERE: DEE-59, IS (West) → LGS</li>
              <li>IS onset 4-12 months (infancy)</li>
              <li>DR: ~75-90% (GOF severe)</li>
              <li>West syndrome: 85% GOF</li>
              <li>Profound ID (GOF)</li>
              <li>Baclofen Level C LOF (1-2 mg/kg/day)</li>
              <li>OMIM DEE-59 #617137</li>
            </ul>
          </div>
        </div>
        <div className="alert alert-warning py-1 mt-2 small">
          BOTH GABBR1 and GABBR2: TGB ABSOLUTE CI · Baclofen abrupt withdrawal = medical emergency · POLG1 screen before VPA · PHT/CBZ HIGH RISK
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
          <p className="mt-1 mb-0 small"><strong>vs GABBR2:</strong> {d.vs_gabbr2}</p>
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
        text="⚠️ GABBR1 GOF/LOF FUNCTIONAL ASSAY mandatory before ANY Baclofen decision. Xenopus oocyte or HEK293: measure GABA dose-response (EC50 shift). GOF = Baclofen ABSOLUTE CI. LOF = Baclofen Level C Precision Rx. GABBR1a-selective sushi domain mutations = best baclofen candidates."
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
            <div className="small">
              <strong>Baclofen:</strong>{' '}
              <span style={{ color: e.baclofen_role?.toUpperCase().includes('ABSOLUTE') || e.baclofen_role?.toUpperCase().includes('CONTRA') ? ACCENT2 : ACCENT3 }}>
                {e.baclofen_role}
              </span>
            </div>
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
                <th>GEFS+</th><th>Absence</th><th>Focal</th>
                <th>Baclofen</th><th>VPA</th><th>Assay</th><th>Cog Preserved</th>
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
                  <td>{p.gefs_plus ? '✓' : '—'}</td>
                  <td>{p.absence_seizures ? '✓' : '—'}</td>
                  <td>{p.focal_seizures ? '✓' : '—'}</td>
                  <td>{p.on_baclofen ? <span style={{ color: ACCENT3 }}>✓</span> : '—'}</td>
                  <td>{p.on_vpa ? '✓' : '—'}</td>
                  <td>{p.functional_assay_done ? <span style={{ color: ACCENT }}>✓</span> : <span className="text-danger">PENDING</span>}</td>
                  <td>{p.cognitive_preserved ? <span className="text-success">✓</span> : <span className="text-danger">↓</span>}</td>
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
        text="GABBR1 GEFS+ key insight: Fever → increased GABBR1 receptor endocytosis → acute surface GABA-B loss → febrile seizure. Fever management protocol (paracetamol + early BDZ) is primary prevention. AVOID PHT/CBZ — worsen generalised epilepsy. TGB ABSOLUTE CI. ESM Level A for absence."
        variant="info"
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
        text="🎯 GABBR1 Treatment: VPA/LTG first-line GEFS+; ESM Level A for absence. Baclofen Level C LOF ONLY — functional assay mandatory. LTG AVOID monotherapy if myoclonic component. NEVER stop Baclofen abruptly. PHT/CBZ HIGH RISK. TGB ABSOLUTE CI."
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
            <div className="small p-2 rounded" style={{ backgroundColor: '#fce4ec', borderLeft: `3px solid ${ACCENT}` }}>
              <strong>GABBR1 note:</strong> {t.gabbr1_note}
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

      <SectionCard title="Lifecycle Stages (6)" borderColor={ACCENT4}>
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
export default function GABBR1Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/gabbr1/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (activeTab === 1 || activeTab === 2 || activeTab === 3) {
      if (!breakdown) {
        fetch(`${API}/api/gabbr1/breakdown`)
          .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
      }
    }
    if (activeTab === 4 && !definitions) {
      fetch(`${API}/api/gabbr1/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
    }
  }, [activeTab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3" style={{ borderLeft: `6px solid ${ACCENT}`, paddingLeft: 16 }}>
        <h3 className="fw-bold mb-0" style={{ color: ACCENT }}>
          🧬 GABBR1 Epilepsy
        </h3>
        <div className="text-muted small">
          GABA-B Receptor Subunit 1 · Venus Flytrap Ligand-Binding · GABBR1a (Presynaptic-Sushi) / GABBR1b (Postsynaptic-GIRK) ·
          GEFS+ / Focal / Absence · Baclofen Precision (LOF only) · TGB-ABSOLUTE-NCSE · PHT-CBZ-HIGH-RISK ·
          6p22.1 · GABA-B Heterodimer Partner to GABBR2
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${activeTab === i ? 'active fw-bold' : ''}`}
              style={activeTab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setActiveTab(i)}
            >
              {t}
            </button>
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
