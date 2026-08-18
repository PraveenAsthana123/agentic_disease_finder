'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#7b1fa2';   // deep purple — CERS1 sphingolipid/ceramide biosynthesis pathway
const ACCENT2 = '#b71c1c';   // dark red — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep orange — alerts / triggers / high-risk
const ACCENT4 = '#1565c0';   // deep blue — safe treatments

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

// ── TAB 1: Overview ──────────────────────────────────────────────────────────
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Alert
        variant="danger"
        text="⛔ ABSOLUTE CI: CBZ / OXC / PHT / Fosphenytoin-IV (Na-channel — worsen action myoclonus in ALL PME). IV LEV 60 mg/kg = SOLE SE rescue. TGB ABSOLUTE CI (NCSE + Purkinje GABA depletion by C18-ceramide deficiency)."
      />
      <Alert
        variant="warning"
        text="⚠ HIGH RISK: GBP / PGB (α2δ — paradoxical myoclonic worsening; Crespel 1999; multi-specialty prescribing trap). Vigabatrin AVOID. LTG monotherapy HIGH RISK."
      />
      <Alert
        variant="info"
        text="🔵 CERS1-PMEA DISTINCTION: Selective Purkinje cell degeneration via C18-ceramide deficiency. Prominent cerebellar ataxia often the FIRST symptom. Preserved cognition throughout (100% of cohort). Non-fatal — rehabilitation is core. NO DISEASE-MODIFYING THERAPY. SARA + UMRS dual tracking mandatory (two independent disease axes)."
      />

      <div className="row mb-4">
        <KPI label="Gene / Locus" value="CERS1 / 19p13.2" color={ACCENT} />
        <KPI label="Protein" value="Ceramide Synthase 1" color={ACCENT} />
        <KPI label="Inheritance" value="Autosomal Recessive" color={ACCENT} />
        <KPI label="Cohort" value={`${ov.cohort_size} patients`} color={ACCENT} />
        <KPI label="Mean Onset" value={`${ov.mean_onset_years}y`} color={ACCENT} />
        <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
      </div>

      <SectionCard title="Molecular Mechanism — C18-Ceramide Deficiency & Purkinje Cell Death">
        <p style={{ fontSize: 14 }}>{ov.mechanism}</p>
        <div className="row mt-3">
          <div className="col-md-6">
            <PctBar label="Cerebellar Ataxia Present" pct={ov.cerebellar_ataxia_pct} color={ACCENT} />
            <PctBar label="Preserved Cognition" pct={ov.preserved_cognition_pct} color={ACCENT4} />
            <PctBar label="Giant SEP Confirmed" pct={ov.giant_sep_confirmed_pct} color={ACCENT} />
            <PctBar label="Cerebellar MRI Atrophy" pct={ov.cerebellar_mri_atrophy_pct} color={ACCENT} />
          </div>
          <div className="col-md-6">
            <PctBar label="Photosensitivity (IPS+)" pct={ov.photosensitivity_pct} color={ACCENT3} />
            <PctBar label="Ambulatory (unaided)" pct={ov.ambulatory_pct} color={ACCENT4} />
            <PctBar label="On VPA" pct={ov.on_vpa_pct} color={ACCENT4} />
            <PctBar label="On Piracetam" pct={ov.on_piracetam_pct} color={ACCENT4} />
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Gene / Protein Biology" borderColor={ACCENT4}>
        <p style={{ fontSize: 13 }}><strong>Gene:</strong> {ov.gene}</p>
        <p style={{ fontSize: 13 }}><strong>Protein:</strong> {ov.protein}</p>
        <p style={{ fontSize: 13 }}><strong>OMIM:</strong> {ov.omim}</p>
        <p style={{ fontSize: 13 }}><strong>Discovery:</strong> {ov.discovery}</p>
        <p style={{ fontSize: 13 }}><strong>Unique Feature:</strong> {ov.unique_feature}</p>
      </SectionCard>

      <SectionCard title="Absolute Contraindications" borderColor={ACCENT2}>
        {(ov.absolute_ci || []).map((ci, i) => (
          <div key={i} className="alert alert-danger py-1 mb-2" style={{ fontSize: 13 }}>
            ⛔ {ci}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="High-Risk Drugs" borderColor={ACCENT3}>
        {(ov.high_risk_ci || []).map((ci, i) => (
          <div key={i} className="alert alert-warning py-1 mb-2" style={{ fontSize: 13 }}>
            ⚠ {ci}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Key Clinical Parameters" borderColor={ACCENT4}>
        <div className="row">
          <div className="col-md-6">
            <p style={{ fontSize: 13 }}><strong>Inheritance:</strong> {ov.inheritance}</p>
            <p style={{ fontSize: 13 }}><strong>Disease:</strong> {ov.disease}</p>
            <p style={{ fontSize: 13 }}><strong>Giant SEP threshold:</strong> {ov.sep_amplitude_threshold_uv} µV (N20/P25)</p>
          </div>
          <div className="col-md-6">
            <p style={{ fontSize: 13 }}><strong>IV LEV SE dose:</strong> {ov.lev_iv_se_dose_mgkg} mg/kg</p>
            <p style={{ fontSize: 13 }}><strong>VPA trough target:</strong> {ov.vpa_trough_target_ugml} µg/mL</p>
            <p style={{ fontSize: 13 }}><strong>POLG1 screen:</strong> Mandatory {ov.polg1_screened_pct}% screened in cohort</p>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}

// ── TAB 2: Patients & Etiology ────────────────────────────────────────────────
function PatientsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const { etiologies = [], patients = [] } = bk;
  return (
    <div>
      <SectionCard title="Etiology Distribution — CERS1 Variant Classes">
        {etiologies.map((e, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <div className="d-flex justify-content-between">
              <strong style={{ color: ACCENT }}>{e.class.replace(/-/g, ' ')}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT, color: '#fff' }}>{e.pct}% (n={e.count})</span>
            </div>
            <div style={{ fontSize: 13 }} className="mt-1">{e.description}</div>
            <div style={{ fontSize: 12, color: '#555' }} className="mt-1"><em>Mechanism: {e.gene_mechanism}</em></div>
            {e.key_variants && (
              <div className="mt-1">
                {e.key_variants.map((v, j) => (
                  <span key={j} className="badge bg-secondary me-1 mb-1" style={{ fontSize: 11 }}>{v}</span>
                ))}
              </div>
            )}
            <div className="progress mt-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT }} />
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title={`Patient Cohort (n=${patients.length})`}>
        <div style={{ overflowX: 'auto' }}>
          <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th>ID</th><th>Age</th><th>Sex</th><th>Onset</th><th>Etiology</th>
                <th>Seizure Type</th><th>AEDs</th><th>DRE</th>
                <th>SARA</th><th>Giant SEP</th><th>MRI Atrophy</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.age}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_age}y</td>
                  <td><span className="badge" style={{ backgroundColor: ACCENT, color: '#fff', fontSize: 10 }}>{p.etiology.replace(/-/g, ' ')}</span></td>
                  <td style={{ maxWidth: 160 }}>{p.seizure_type.replace(/-/g, ' ')}</td>
                  <td>{(p.current_aeds || []).join(', ')}</td>
                  <td>{p.drug_resistant ? <span className="badge bg-danger">DRE</span> : <span className="badge bg-success">Ctrl</span>}</td>
                  <td><span className={`badge ${p.sara_score > 25 ? 'bg-danger' : p.sara_score > 15 ? 'bg-warning text-dark' : 'bg-success'}`}>{p.sara_score}</span></td>
                  <td>{p.giant_sep ? '✓' : '–'}</td>
                  <td>{p.cerebellar_mri_atrophy ? '✓' : '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── TAB 3: Seizures & Triggers ────────────────────────────────────────────────
function SeizuresTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const { seizure_types = [], triggers = [], lifecycle = [] } = bk;
  return (
    <div>
      <SectionCard title="Seizure Types in CERS1-PMEA">
        {seizure_types.map((s, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <div className="d-flex justify-content-between">
              <strong style={{ color: ACCENT }}>{s.type.replace(/-/g, ' ')}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT, color: '#fff' }}>{s.pct}%</span>
            </div>
            <div className="progress mb-2 mt-1" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${s.pct}%`, backgroundColor: ACCENT }} />
            </div>
            <div style={{ fontSize: 13 }}>{s.description}</div>
            <div style={{ fontSize: 12, color: '#555', marginTop: 4 }}><strong>EEG:</strong> {s.eeg_finding}</div>
            <div style={{ fontSize: 12, color: '#555' }}><strong>Semiology:</strong> {s.semiology}</div>
            <div className="alert alert-info py-1 mt-2" style={{ fontSize: 12 }}>
              💡 <strong>Clinical Tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers" borderColor={ACCENT3}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3 p-3 border rounded" style={{ borderColor: ACCENT3 }}>
            <div className="d-flex justify-content-between">
              <strong style={{ color: ACCENT3 }}>{t.trigger.replace(/-/g, ' ')}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT3, color: '#fff' }}>{t.pct}%</span>
            </div>
            <div className="progress mb-2 mt-1" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: ACCENT3 }} />
            </div>
            <div style={{ fontSize: 13 }}><strong>Mechanism:</strong> {t.mechanism}</div>
            <div style={{ fontSize: 13 }}><strong>Management:</strong> {t.management}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Disease Lifecycle Stages" borderColor={ACCENT4}>
        {lifecycle.map((s, i) => (
          <div key={i} className="mb-3 p-3 border rounded" style={{ borderColor: ACCENT4 }}>
            <strong style={{ color: ACCENT4 }}>{s.stage}</strong>
            <div style={{ fontSize: 13 }} className="mt-1">{s.description}</div>
            <div className="alert alert-primary py-1 mt-2" style={{ fontSize: 12 }}>
              🎯 <strong>Key Action:</strong> {s.key_action}
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 4: Treatments ────────────────────────────────────────────────────────
function TreatmentsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const { treatments = [], contraindications = [], monitoring = [] } = bk;
  return (
    <div>
      <Alert variant="danger" text="⛔ NO DISEASE-MODIFYING THERAPY EXISTS for CERS1-PMEA — management is purely symptomatic. CBZ/OXC/PHT/Fosphenytoin ABSOLUTE CI. TGB ABSOLUTE CI. IV LEV 60 mg/kg = SOLE SE rescue." />

      <SectionCard title="Treatments (Ranked by Evidence Level)">
        {treatments.map((t, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <div className="d-flex justify-content-between align-items-start">
              <strong style={{ color: ACCENT }}>{t.drug}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT4, color: '#fff', fontSize: 11 }}>{t.level}</span>
            </div>
            <div style={{ fontSize: 13 }} className="mt-1"><strong>Dose:</strong> {t.dose}</div>
            <div style={{ fontSize: 13 }}><strong>Mechanism:</strong> {t.mechanism}</div>
            <div style={{ fontSize: 13 }}><strong>Efficacy:</strong> {t.efficacy}</div>
            <div style={{ fontSize: 13 }}><strong>Monitoring:</strong> {t.monitoring}</div>
            <div className="alert alert-secondary py-1 mt-2" style={{ fontSize: 12 }}>
              🔬 <strong>CERS1 Note:</strong> {t.cers1_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications" borderColor={ACCENT2}>
        {contraindications.map((c, i) => (
          <div key={i} className="mb-3 p-3 border rounded" style={{ borderColor: c.risk === 'ABSOLUTE-CI' ? ACCENT2 : ACCENT3 }}>
            <div className="d-flex justify-content-between">
              <strong style={{ color: c.risk === 'ABSOLUTE-CI' ? ACCENT2 : ACCENT3 }}>{c.drug}</strong>
              <span className={`badge ${c.risk === 'ABSOLUTE-CI' ? 'bg-danger' : 'bg-warning text-dark'}`}>{c.risk}</span>
            </div>
            <div style={{ fontSize: 13 }} className="mt-1"><strong>Mechanism:</strong> {c.mechanism}</div>
            <div className="alert alert-warning py-1 mt-2" style={{ fontSize: 12 }}>
              ⚠ <strong>CERS1-PMEA specific:</strong> {c.cers1_specific}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Protocol" borderColor={ACCENT4}>
        <table className="table table-sm" style={{ fontSize: 12 }}>
          <thead>
            <tr><th>Monitoring Item</th><th>Frequency</th><th>Rationale</th></tr>
          </thead>
          <tbody>
            {monitoring.map((m, i) => (
              <tr key={i}>
                <td><strong>{m.item.replace(/-/g, ' ')}</strong></td>
                <td><span className="badge bg-secondary">{m.frequency}</span></td>
                <td>{m.rationale}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </SectionCard>
    </div>
  );
}

// ── TAB 5: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ df }) {
  if (!df) return <div className="text-muted">Loading…</div>;
  const { concepts = [], thresholds = [], standards = [], references = [] } = df;
  return (
    <div>
      <SectionCard title="Key Concepts (15)">
        {concepts.map((c, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <strong style={{ color: ACCENT }}>{c.concept.replace(/-/g, ' ')}</strong>
            <div style={{ fontSize: 13 }} className="mt-1">{c.definition}</div>
            <div style={{ fontSize: 11, color: '#777' }} className="mt-1"><em>Standards: {c.standard}</em></div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds (12)" borderColor={ACCENT4}>
        <table className="table table-sm" style={{ fontSize: 12 }}>
          <thead><tr><th>Threshold</th><th>Value</th><th>Rationale</th></tr></thead>
          <tbody>
            {thresholds.map((t, i) => (
              <tr key={i}>
                <td><strong>{t.name}</strong></td>
                <td><span className="badge" style={{ backgroundColor: ACCENT }}>{t.value}</span></td>
                <td>{t.rationale}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </SectionCard>

      <SectionCard title="Clinical Standards (12)" borderColor={ACCENT3}>
        {standards.map((s, i) => (
          <div key={i} className="mb-1 d-flex align-items-start gap-2">
            <span className="badge" style={{ backgroundColor: ACCENT3, minWidth: 140 }}>{s.name}</span>
            <span style={{ fontSize: 13 }}>{s.scope}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="References (6)" borderColor={ACCENT4}>
        {references.map((r, i) => (
          <div key={i} className="mb-2 p-2 border rounded" style={{ fontSize: 12 }}>
            <strong>[{r.ref}]</strong> {r.citation}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function CERS1Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bk, setBk]       = useState(null);
  const [df, setDf]       = useState(null);
  const [err, setErr]     = useState('');

  useEffect(() => {
    fetch(`${API}/api/cers1/overview`)
      .then(r => r.json()).then(setOv).catch(() => setErr('overview failed'));
    fetch(`${API}/api/cers1/breakdown`)
      .then(r => r.json()).then(setBk).catch(() => setErr('breakdown failed'));
    fetch(`${API}/api/cers1/definitions`)
      .then(r => r.json()).then(setDf).catch(() => setErr('definitions failed'));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3 p-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${ACCENT}, #4a148c)` }}>
        <h4 className="mb-1">🧬 CERS1 Epilepsy — Progressive Myoclonic Epilepsy with Cerebellar Ataxia (CERS1-PMEA)</h4>
        <div style={{ fontSize: 13, opacity: 0.9 }}>
          Ceramide Synthase 1 / LASS1 / C18-Ceramide Deficiency / Selective Purkinje Cell Degeneration / 19p13.2<br />
          CBZ-OXC-PHT <strong>ABSOLUTE CI</strong> · TGB <strong>ABSOLUTE CI</strong> · GBP-PGB HIGH RISK · IV LEV 60 mg/kg SE ·
          SARA + UMRS Dual Tracking · No Disease-Modifying Therapy · Preserved Cognition · Non-Fatal PME
        </div>
      </div>

      {err && <div className="alert alert-danger">{err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT, borderBottomWidth: 2, fontWeight: 600 } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <PatientsTab bk={bk} />}
      {tab === 2 && <SeizuresTab bk={bk} />}
      {tab === 3 && <TreatmentsTab bk={bk} />}
      {tab === 4 && <DefinitionsTab df={df} />}
    </div>
  );
}
