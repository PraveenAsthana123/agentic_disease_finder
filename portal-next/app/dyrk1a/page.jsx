'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#0d47a1';   // deep navy blue — DYRK1A kinase signalling
const ACCENT2 = '#b71c1c';   // deep red — ABSOLUTE CI / HIGH RISK
const ACCENT3 = '#1b5e20';   // deep green — precision therapy / folinic acid
const ACCENT4 = '#e65100';   // deep orange — NRSF pathway warning

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

// ── Tab 1: Overview ───────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Alert
        text="⚠️ DYRK1A NRSF/Nav1.1 PATHWAY: PHT/CBZ/OXC Na-channel blockers HIGH RISK — suppress Nav1.1 in GABAergic interneurons (NRSF-stabilised LOF) → disinhibition → seizure worsening. DYRK1A Inhibitors (harmine/leucettine/EGCG) ABSOLUTE CI — used in Down syndrome (3× DYRK1A) but catastrophic in haploinsufficiency (1× DYRK1A). NEVER confuse trisomy 21 and LOF syndrome pharmacology."
        variant="danger"
      />
      <Alert
        text="🧬 NRSF/REST MECHANISM: DYRK1A LOF → NRSF not phosphorylated → NRSF accumulates → represses SCN1A (Nav1.1), GABRB3 → functional Nav1.1 deficiency analogous to Dravet. VPA HDAC inhibition may epigenetically de-repress NRSF-silenced epilepsy genes — rational first-line choice."
        variant="info"
      />
      <Alert
        text="🌿 FOLINIC ACID PRECISION: 25-35% DYRK1A patients anti-FRA1 antibody positive → cerebral folate deficiency. Screen ALL patients. Folinic acid 0.5-1 mg/kg/day if positive. POLG1 screen MANDATORY before VPA."
        variant="success"
      />

      <div className="row g-3 mb-4">
        <KPI label="Cohort" value={data.cohort_size} color={ACCENT} />
        <KPI label="Seizures" value={`${data.seizure_prevalence_pct}%`} color={ACCENT2} />
        <KPI label="Drug-Resistant" value={`${data.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Infantile Spasms" value={`${data.infantile_spasms_pct}%`} color={ACCENT4} />
        <KPI label="Autism" value={`${data.autism_pct}%`} color={ACCENT} />
        <KPI label="Mean HC SD" value={data.mean_hc_sd} color={ACCENT4} />
        <KPI label="MRI Abnormal" value={`${data.mri_abnormal_pct}%`} color={ACCENT} />
        <KPI label="Anti-FRA1 +" value={`${data.anti_fra1_positive_pct}%`} color={ACCENT3} />
        <KPI label="Folinic Acid" value={`${data.folinic_acid_pct}%`} color={ACCENT3} />
        <KPI label="On KD" value={`${data.on_kd_pct}%`} color={ACCENT} />
        <KPI label="Mean AEDs" value={data.mean_aed_count} color={ACCENT} />
        <KPI label="Etiology Classes" value={data.etiology_classes} color={ACCENT} />
      </div>

      <SectionCard title="Gene Summary" borderColor={ACCENT}>
        <table className="table table-sm table-borderless mb-0" style={{ fontSize: 13 }}>
          <tbody>
            <tr><td className="fw-bold text-muted" style={{ width: 180 }}>Gene</td><td>{data.gene} ({data.locus})</td></tr>
            <tr><td className="fw-bold text-muted">Protein</td><td>{data.protein}</td></tr>
            <tr><td className="fw-bold text-muted">OMIM Syndrome</td><td>{data.omim_syndrome}</td></tr>
            <tr><td className="fw-bold text-muted">OMIM Gene</td><td>{data.omim_gene}</td></tr>
            <tr><td className="fw-bold text-muted">NRSF Pathway</td><td style={{ color: ACCENT4 }}><strong>{data.nrsf_pathway}</strong></td></tr>
            <tr><td className="fw-bold text-muted">Trisomy 21 Distinction</td><td style={{ color: ACCENT2 }}>{data.trisomy_21_distinction}</td></tr>
            <tr><td className="fw-bold text-muted">Precision Adjunct</td><td style={{ color: ACCENT3 }}>{data.precision_adjunct}</td></tr>
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
          <div key={i} className="mb-4 p-3 rounded" style={{ background: '#e3f2fd', borderLeft: `4px solid ${ACCENT}` }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <strong style={{ color: ACCENT }}>{e.etiology}</strong>
              <span className="badge ms-2" style={{ background: ACCENT }}>{e.pct}% (n={e.n})</span>
            </div>
            <PctBar label="" pct={e.pct} color={ACCENT} />
            <p className="small mb-2">{e.mechanism}</p>
            <div className="row g-2" style={{ fontSize: 12 }}>
              <div className="col-md-4"><strong>Seizure Types:</strong> {(e.seizure_types || []).join(', ')}</div>
              <div className="col-md-4"><strong>Onset Range:</strong> {e.age_onset_range}</div>
              <div className="col-md-4"><strong>Drug Response:</strong> {e.drug_response}</div>
              <div className="col-md-4"><strong>Typical AED:</strong> {e.typical_aed}</div>
              <div className="col-md-4"><strong>Microcephaly:</strong> {e.microcephaly_severity}</div>
              <div className="col-md-4"><strong>NRSF Impact:</strong> {e.nrsf_impact}</div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead style={{ background: ACCENT, color: '#fff' }}>
              <tr>
                <th>ID</th><th>Name</th><th>Age Dx (mo)</th><th>Etiology</th>
                <th>HC SD</th><th>Seizures</th><th>IS</th><th>DRE</th>
                <th>AEDs</th><th>KD</th><th>Anti-FRA1+</th><th>MRI Abn</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.id}>
                  <td className="fw-bold">{p.id}</td>
                  <td>{p.name}</td>
                  <td>{p.age_dx}</td>
                  <td style={{ maxWidth: 160, fontSize: 11 }}>{p.etiology}</td>
                  <td style={{ color: p.hc_sd < -3 ? ACCENT2 : ACCENT4 }}>{p.hc_sd}</td>
                  <td>{p.has_seizures ? '✓' : '—'}</td>
                  <td>{p.is_onset ? <span style={{ color: ACCENT2 }}>IS</span> : '—'}</td>
                  <td>{p.drug_resistant ? <span style={{ color: ACCENT2 }}>DRE</span> : '—'}</td>
                  <td>{p.n_aed}</td>
                  <td>{p.kd ? '✓' : '—'}</td>
                  <td>{p.anti_fra1_positive ? <span style={{ color: ACCENT3 }}>+</span> : '—'}</td>
                  <td>{p.mri_abnormal ? <span style={{ color: ACCENT4 }}>Abn</span> : '—'}</td>
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
          <div key={i} className="mb-4 p-3 rounded" style={{ background: '#e3f2fd', borderLeft: `4px solid ${ACCENT}` }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <strong style={{ color: ACCENT }}>{s.type}</strong>
              <span className="badge" style={{ background: ACCENT2 }}>{s.prevalence_pct}%</span>
            </div>
            <PctBar label="" pct={s.prevalence_pct} color={ACCENT} />
            <div className="row g-2 mb-2" style={{ fontSize: 12 }}>
              <div className="col-md-3"><strong>Onset:</strong> {s.onset}</div>
              <div className="col-md-3"><strong>Duration:</strong> {s.duration}</div>
              <div className="col-md-6"><strong>EEG:</strong> {s.eeg}</div>
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
              <span className="badge" style={{ background: ACCENT4 }}>{t.prevalence_pct}%</span>
            </div>
            <PctBar label="" pct={t.prevalence_pct} color={ACCENT4} />
            <p className="small text-muted mb-0">{t.mechanism}</p>
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
        text="⛔ ABSOLUTE CI: DYRK1A Inhibitors (harmine / leucettine-41 / EGCG/green tea extract) — used in Down syndrome (trisomy 21, 3× DYRK1A) but CATASTROPHIC in LOF (1× DYRK1A). Tiagabine (TGB) — NCSE risk (GABRB3 reduction via NRSF). VPA in POLG1+ — Alpers-Huttenlocher fatal. ⚠️ HIGH RISK: PHT/CBZ/OXC — NRSF/Nav1.1 pathway worsening (Dravet-like)."
        variant="danger"
      />

      <SectionCard title="Treatments (8 agents)" borderColor={ACCENT3}>
        {treatments.map((t, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ background: '#e8f5e9', borderLeft: `4px solid ${ACCENT3}` }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <strong style={{ color: ACCENT3 }}>{t.drug}</strong>
              <span className="badge" style={{ background: ACCENT3 }}>{t.level}</span>
            </div>
            <div className="row g-2 mb-2" style={{ fontSize: 12 }}>
              <div className="col-md-6"><strong>Indication:</strong> {t.indication}</div>
              <div className="col-md-6"><strong>Dose:</strong> {t.dose}</div>
              <div className="col-12"><strong>MOA:</strong> {t.moa}</div>
              <div className="col-md-6"><strong>Efficacy:</strong> {t.efficacy}</div>
              <div className="col-md-6"><strong>Monitoring:</strong> {t.monitoring}</div>
            </div>
            <div className="alert alert-primary py-1 mb-0" style={{ fontSize: 12 }}>
              <strong>DYRK1A Note:</strong> {t.dyrk1a_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications (6 CIs)" borderColor={ACCENT2}>
        {contraindications.map((c, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{
            background: c.severity.includes('ABSOLUTE') ? '#ffebee' : '#fff8e1',
            borderLeft: `4px solid ${c.severity.includes('ABSOLUTE') ? ACCENT2 : '#f57f17'}`
          }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <strong style={{ color: ACCENT2 }}>{c.drug}</strong>
              <span className="badge" style={{
                background: c.severity.includes('ABSOLUTE') ? ACCENT2 : '#f57f17'
              }}>{c.severity}</span>
            </div>
            <p className="small mb-1">{c.mechanism}</p>
            <p className="small mb-1"><strong>Action:</strong> {c.action}</p>
            <p className="small text-muted mb-0"><strong>Evidence:</strong> {c.evidence}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Checklist (14 items)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead style={{ background: ACCENT, color: '#fff' }}>
              <tr><th>Item</th><th>Frequency</th><th>Purpose</th></tr>
            </thead>
            <tbody>
              {monitoring.map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold">{m.item}</td>
                  <td>{m.frequency}</td>
                  <td>{m.purpose}</td>
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
          <div key={i} className="mb-3 pb-3 border-bottom">
            <strong style={{ color: ACCENT }}>{c.concept}</strong>
            <p className="small mb-0 mt-1">{c.definition}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds (12)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead style={{ background: ACCENT4, color: '#fff' }}>
              <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.parameter}</td>
                  <td style={{ color: ACCENT2 }}>{t.threshold}</td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="References (6)" borderColor={ACCENT3}>
        {references.map((r, i) => (
          <div key={i} className="mb-2 pb-2 border-bottom">
            <strong style={{ fontSize: 12, color: ACCENT3 }}>{r.ref}</strong>
            <p className="small mb-0">{r.citation}</p>
            <p className="small text-muted mb-0"><em>{r.impact}</em></p>
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

// ── Lifecycle Panel ────────────────────────────────────────────────────────────
function LifecyclePanel({ stages = [] }) {
  return (
    <SectionCard title="Lifecycle Stages (6)" borderColor={ACCENT}>
      <div className="row g-3">
        {stages.map((s, i) => (
          <div key={i} className="col-md-6 col-lg-4">
            <div className="card h-100 shadow-sm" style={{ borderTop: `3px solid ${ACCENT}` }}>
              <div className="card-header fw-bold small py-2" style={{ color: ACCENT }}>
                {s.stage}
              </div>
              <div className="card-body py-2">
                <ul className="list-unstyled mb-2" style={{ fontSize: 12 }}>
                  {(s.key_features || []).map((f, j) => (
                    <li key={j} className="mb-1">• {f}</li>
                  ))}
                </ul>
                <div className="alert alert-light py-1 mb-0" style={{ fontSize: 11 }}>
                  {s.action}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </SectionCard>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function DYRK1APage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [ov, br, df] = await Promise.all([
          fetch(`${API}/api/dyrk1a/overview`).then(r => r.json()),
          fetch(`${API}/api/dyrk1a/breakdown`).then(r => r.json()),
          fetch(`${API}/api/dyrk1a/definitions`).then(r => r.json()),
        ]);
        setOverview(ov);
        setBreakdown(br);
        setDefinitions(df);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return (
    <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}>
      <div className="spinner-border" style={{ color: ACCENT }} />
    </div>
  );
  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  return (
    <div className="container-fluid px-3 py-3">
      {/* Header */}
      <div className="p-3 mb-4 rounded text-white" style={{ background: ACCENT }}>
        <h4 className="mb-1">🧬 DYRK1A Epilepsy — DYRK1A Syndrome / 21q22.13 Haploinsufficiency</h4>
        <div style={{ fontSize: 13, opacity: 0.9 }}>
          DEE + Primary Microcephaly + Severe ID + Autism · NRSF/Nav1.1 Pathway ·
          PHT/CBZ/OXC HIGH RISK · DYRK1A Inhibitors ABSOLUTE CI in LOF ·
          Folinic Acid for Anti-FRA1+ (25-35%) · VPA HDAC Epigenetic Rationale ·
          OMIM #614721 · 40-patient cohort · pLI=0.99
        </div>
      </div>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT, borderBottomWidth: 2 } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* Tab content */}
      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && (
        <div>
          <SeizuresTab data={breakdown} />
          {breakdown && <LifecyclePanel stages={breakdown.lifecycle || []} />}
        </div>
      )}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
