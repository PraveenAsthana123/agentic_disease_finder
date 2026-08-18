'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#bf360c';   // deep burnt-orange — SCARB2 lysosomal/renal
const ACCENT2 = '#b71c1c';   // deep red — ABSOLUTE CI
const ACCENT3 = '#e65100';   // deep orange — warnings / renal
const ACCENT4 = '#1a237e';   // deep indigo — AED/neuro

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#fbe9e7', color: borderColor }}>
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
        text="⚠ ABSOLUTE CIs: CBZ / OXC / PHT / Fosphenytoin (worsen action myoclonus) · TGB (NCSE + Purkinje loss) · VPA without POLG1 screen (Alpers) · GBP/PGB — DOUBLE CI in AMRF: paradoxical myoclonic worsening + 100% renally cleared → accumulation in CKD (nephrologists prescribe GBP for uraemic pruritus — document in renal team records)"
        variant="danger"
      />
      <Alert
        text="🫘 SCARB2 (4q21.1) — LIMP-2 · GBA1 lysosomal transport LOF · Action Myoclonus-Renal Failure Syndrome (AMRF / EPM4) · ONLY PME with renal involvement · FSGS + proteinuria PATHOGNOMONIC · Test urine ACR at every visit · Later onset 15-35y · ACE/ARB mandatory"
        variant="info"
      />
      <Alert
        text="🔬 Renal transplant corrects the kidney — neurological disease CONTINUES post-transplant (SCARB2 LOF persists in CNS neurons). Counsel patients: transplant is NOT a neurological cure."
        variant="warning"
      />

      <SectionCard title="Gene & Protein Summary" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Gene:</strong> {ov.gene}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {ov.inheritance}</p>
        <p className="small mb-0"><strong>OMIM:</strong> {ov.omim}</p>
      </SectionCard>

      <SectionCard title="Cohort Statistics" borderColor={ACCENT}>
        <div className="row">
          <KPI label="Patients" value={ov.cohort_size} color={ACCENT} />
          <KPI label="Female" value={`${ov.female_pct}%`} color={ACCENT} />
          <KPI label="Mean Onset" value={`${ov.mean_onset_years}y`} color={ACCENT4} />
          <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT3} />
          <KPI label="Photosensitive" value={`${ov.photosensitivity_pct}%`} color={ACCENT3} />
          <KPI label="Ambulatory" value={`${ov.ambulatory_pct}%`} color="#2e7d32" />
        </div>
        <div className="row mt-2">
          <KPI label="On VPA" value={`${ov.on_vpa_pct}%`} color={ACCENT} />
          <KPI label="On LEV" value={`${ov.on_lev_pct}%`} color={ACCENT} />
          <KPI label="On Piracetam" value={`${ov.on_piracetam_pct}%`} color={ACCENT4} />
          <KPI label="On CLB" value={`${ov.on_clb_pct}%`} color={ACCENT} />
          <KPI label="ACE/ARB" value={`${ov.ace_arb_pct}%`} color="#2e7d32" />
          <KPI label="Giant SEP ✓" value={`${ov.giant_sep_confirmed_pct}%`} color={ACCENT4} />
        </div>
        <div className="row mt-2">
          <KPI label="POLG1 Screened" value={`${ov.polg1_screened_pct}%`} color="#2e7d32" />
          <KPI label="Consanguineous" value={`${ov.consanguineous_pct}%`} color={ACCENT3} />
        </div>
      </SectionCard>

      {ov.egfr_group_distribution && (
        <SectionCard title="eGFR / Renal Stage Distribution (AMRF-Specific)" borderColor={ACCENT3}>
          <div className="row">
            {Object.entries(ov.egfr_group_distribution).map(([k, v], i) => (
              <div key={i} className="col-6 col-md-3 mb-2 text-center">
                <div className="fw-bold" style={{ color: ACCENT3 }}>{v}</div>
                <div className="text-muted small">{k.replace(/_/g, ' ')}</div>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Seizure Type Distribution" borderColor={ACCENT4}>
            {ov.seizure_type_distribution?.map((s, i) => (
              <PctBar key={i} label={s.type} pct={s.pct} color={ACCENT4} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Trigger Distribution" borderColor={ACCENT3}>
            {ov.trigger_distribution?.map((t, i) => (
              <PctBar key={i} label={t.trigger} pct={t.pct} color={ACCENT3} />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Key Contraindications" borderColor={ACCENT2}>
        {ov.key_contraindications?.map((ci, i) => (
          <Alert key={i} text={ci} variant="danger" />
        ))}
      </SectionCard>

      <SectionCard title="AMRF Pathognomonic Feature (Renal)" borderColor={ACCENT3}>
        <p className="small mb-0">{ov.amrf_pathognomonic_note}</p>
      </SectionCard>

      <SectionCard title="AED–Renal Drug Interaction Note" borderColor={ACCENT4}>
        <p className="small mb-0">{ov.renal_drug_note}</p>
      </SectionCard>

      <SectionCard title="Transplant Counselling Note" borderColor="#2e7d32">
        <p className="small mb-0">{ov.transplant_counselling_note}</p>
      </SectionCard>
    </div>
  );
}

// ── TAB 2: Patients & Etiology ────────────────────────────────────────────────
function PatientsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="Etiology Catalog" borderColor={ACCENT}>
        {bk.etiology_catalog?.map((e, i) => (
          <div key={i} className="mb-3 pb-3 border-bottom">
            <div className="d-flex justify-content-between align-items-center">
              <strong className="small">{e.category}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{e.pct}% (n={e.n})</span>
            </div>
            <div className="small text-muted mt-1">{e.molecular_class}</div>
            <div className="small mt-1"><em>Pathomechanism:</em> {e.pathomechanism}</div>
            {e.representative_variants && (
              <div className="small mt-1"><em>Variants:</em> {e.representative_variants?.join(' · ')}</div>
            )}
            <div className="small mt-1 text-info"><em>Note:</em> {e.clinical_note}</div>
            {e.founder && <div className="small text-warning"><em>Founder:</em> {e.founder}</div>}
            {e.renal_onset_pct !== undefined && (
              <div className="small"><em>Renal-first presentation:</em> {e.renal_onset_pct}%</div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Sample (first 15)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead>
              <tr style={{ backgroundColor: ACCENT, color: '#fff' }}>
                <th>ID</th><th>Name</th><th>Sex</th><th>Onset (y)</th>
                <th>eGFR</th><th>Renal Status</th><th>VPA</th><th>LEV</th>
                <th>Piracetam</th><th>ACE/ARB</th><th>Giant SEP</th>
              </tr>
            </thead>
            <tbody>
              {bk.patient_sample?.map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td>{p.name}</td>
                  <td>{p.sex}</td>
                  <td>{(p.age_onset_months / 12).toFixed(1)}</td>
                  <td><span style={{ color: p.egfr < 30 ? ACCENT2 : p.egfr < 60 ? ACCENT3 : '#2e7d32' }}>{p.egfr}</span></td>
                  <td><small>{p.renal_status?.replace(/_/g, ' ')}</small></td>
                  <td>{p.on_vpa ? '✓' : '—'}</td>
                  <td>{p.on_lev ? '✓' : '—'}</td>
                  <td>{p.on_piracetam ? '✓' : '—'}</td>
                  <td>{p.ace_arb ? '✓' : '—'}</td>
                  <td>{p.giant_sep ? '⚡' : '—'}</td>
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
  return (
    <div>
      <SectionCard title="Seizure Types" borderColor={ACCENT4}>
        {bk.seizure_types?.map((s, i) => (
          <div key={i} className="mb-4 pb-3 border-bottom">
            <div className="d-flex justify-content-between">
              <strong className="small">{s.type}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{s.frequency_pct}%</span>
            </div>
            <PctBar label="" pct={s.frequency_pct} color={ACCENT4} />
            <div className="small text-muted"><em>EEG:</em> {s.eeg_pattern}</div>
            <div className="small mt-1"><em>Semiology:</em> {s.semiology}</div>
            <div className="small mt-1 text-success"><em>Clinical tip:</em> {s.clinical_tip}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Triggers" borderColor={ACCENT3}>
        {bk.triggers?.map((t, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="d-flex justify-content-between">
              <strong className="small">{t.trigger}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.pct}%</span>
            </div>
            <PctBar label="" pct={t.pct} color={ACCENT3} />
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 4: Treatments ────────────────────────────────────────────────────────
function TreatmentsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Alert
        text="AMRF TREATMENT PRIORITY: (1) ACE inhibitor/ARB — Level A mandatory for proteinuria/FSGS; (2) VPA backbone + piracetam (Level B — RENAL DOSE ADJUST: piracetam 100% renally cleared); (3) LEV adjunct — RENAL DOSE ADJUST (65% renally cleared); (4) CLB (metabolite partial renal clearance); (5) Renal replacement therapy when eGFR <15. NEVER: CBZ/PHT/GBP/PGB/TGB. GBP/PGB double-CI in AMRF (worsens myoclonus + accumulates in renal failure)."
        variant="info"
      />

      <SectionCard title="Pharmacological & Renal Treatments" borderColor={ACCENT}>
        {bk.treatments?.map((t, i) => (
          <div key={i} className="mb-4 pb-3 border-bottom">
            <div className="d-flex justify-content-between align-items-start">
              <strong>{t.treatment}</strong>
              <span
                className="badge ms-2"
                style={{ backgroundColor: t.evidence_level?.includes('A') ? '#2e7d32' : t.evidence_level?.includes('B') ? ACCENT : '#795548' }}
              >
                {t.evidence_level}
              </span>
            </div>
            <div className="small text-muted mt-1"><em>Dose:</em> {t.dose}</div>
            <div className="small mt-1"><em>MOA:</em> {t.moa}</div>
            <div className="small mt-1"><em>Efficacy:</em> {t.efficacy}</div>
            <div className="small mt-1"><em>Monitoring:</em> {t.monitoring}</div>
            {t.amrf_specific_notes && (
              <div className="small mt-1 text-primary"><em>AMRF-specific:</em> {t.amrf_specific_notes}</div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications" borderColor={ACCENT2}>
        {bk.contraindications?.map((ci, i) => (
          <div key={i} className="mb-3 pb-3 border-bottom">
            <div className="d-flex justify-content-between align-items-start">
              <strong className="text-danger">{ci.drug}</strong>
              <span className={`badge ms-2 ${ci.risk_level?.includes('ABSOLUTE') ? 'bg-danger' : 'bg-warning text-dark'}`}>
                {ci.risk_level}
              </span>
            </div>
            <div className="small mt-1"><em>Mechanism:</em> {ci.mechanism}</div>
            <div className="small mt-1 text-danger"><em>Consequence:</em> {ci.clinical_consequence}</div>
            {ci.amrf_specific && <div className="small mt-1 text-warning"><em>AMRF trap:</em> {ci.amrf_specific}</div>}
            <div className="small mt-1 text-success"><em>Alternative:</em> {ci.alternative}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead>
              <tr style={{ backgroundColor: ACCENT, color: '#fff' }}>
                <th>Item</th><th>Frequency</th><th>Target</th>
              </tr>
            </thead>
            <tbody>
              {bk.monitoring?.map((m, i) => (
                <tr key={i}>
                  <td><strong><small>{m.item}</small></strong></td>
                  <td><small>{m.frequency}</small></td>
                  <td><small>{m.target}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Lifecycle Stages" borderColor={ACCENT4}>
        {bk.lifecycle?.map((s, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="fw-bold small" style={{ color: ACCENT4 }}>
              {s.stage} <em className="text-muted">({s.timing})</em>
            </div>
            <div className="small mt-1 text-muted"><em>Clinical focus:</em> {s.clinical_focus}</div>
            {s.key_events && (
              <ul className="small mt-1 mb-0">
                {s.key_events.map((ev, j) => <li key={j}>{ev}</li>)}
              </ul>
            )}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 5: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ df }) {
  if (!df) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="Key Concepts" borderColor={ACCENT}>
        {df.concepts?.map((c, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="fw-bold small" style={{ color: ACCENT }}>{c.concept}</div>
            <div className="small text-muted mt-1">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead>
              <tr style={{ backgroundColor: ACCENT4, color: '#fff' }}>
                <th>Parameter</th><th>Threshold</th><th>Action</th>
              </tr>
            </thead>
            <tbody>
              {df.thresholds?.map((t, i) => (
                <tr key={i}>
                  <td><strong><small>{t.parameter}</small></strong></td>
                  <td><small>{t.threshold}</small></td>
                  <td><small>{t.action}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Clinical Standards" borderColor={ACCENT3}>
        {df.standards?.map((s, i) => (
          <div key={i} className="mb-2 pb-2 border-bottom">
            <strong className="small" style={{ color: ACCENT3 }}>{s.standard}</strong>
            <div className="small text-muted">{s.relevance}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Key References" borderColor={ACCENT}>
        {df.references?.map((r, i) => (
          <div key={i} className="mb-2 pb-2 border-bottom">
            <div className="small" style={{ color: ACCENT }}>{r.ref}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── MAIN PAGE ─────────────────────────────────────────────────────────────────
export default function SCARB2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/scarb2/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2 || tab === 3) {
      if (!breakdown) {
        fetch(`${API}/api/scarb2/breakdown`)
          .then(r => r.json()).then(setBreakdown).catch(e => setError(String(e)));
      }
    }
    if (tab === 4) {
      if (!definitions) {
        fetch(`${API}/api/scarb2/definitions`)
          .then(r => r.json()).then(setDefinitions).catch(e => setError(String(e)));
      }
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3" style={{ borderLeft: `6px solid ${ACCENT}`, paddingLeft: 12 }}>
        <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
          SCARB2 Epilepsy — Action Myoclonus-Renal Failure Syndrome (EPM4 / AMRF)
        </h4>
        <div className="text-muted small">
          LIMP-2 · Lysosomal GBA1 Transport LOF · ONLY PME with Renal Involvement ·
          FSGS Proteinuria Pathognomonic · GBP/PGB Double-CI (Myoclonus + Renal Accumulation) ·
          ACE/ARB Mandatory · Transplant Does NOT Halt CNS Disease · 4q21.1
        </div>
      </div>

      {error && <div className="alert alert-danger">Error: {error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { borderBottomColor: ACCENT, color: ACCENT, fontWeight: 700 } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab ov={overview} />}
      {tab === 1 && <PatientsTab bk={breakdown} />}
      {tab === 2 && <SeizuresTab bk={breakdown} />}
      {tab === 3 && <TreatmentsTab bk={breakdown} />}
      {tab === 4 && <DefinitionsTab df={definitions} />}
    </div>
  );
}
