'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#00695c';   // deep teal — CSTB cystatin B cathepsin inhibitor
const ACCENT2 = '#b71c1c';   // deep red — ABSOLUTE CI
const ACCENT3 = '#e65100';   // deep orange — warnings
const ACCENT4 = '#1a237e';   // deep indigo — piracetam Level A

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#e0f2f1', color: borderColor }}>
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
        text="⚠ ABSOLUTE CIs: CBZ / OXC / PHT / Fosphenytoin (worsen action myoclonus — PRIMARY disability) · TGB (NCSE) · VPA without POLG1 screen (Alpers-Huttenlocher) · GBP/PGB (paradoxical myoclonic worsening α2δ)"
        variant="danger"
      />
      <Alert
        text="🧬 CSTB (21q22.3) — Cystatin B · ~90% dodecamer repeat expansion · Unverricht-Lundborg Disease (EPM1) — MOST COMMON PME worldwide · NOT uniformly fatal (unlike Lafora) · Action myoclonus = primary disability · Piracetam Level A"
        variant="info"
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
          <KPI label="POLG1 Screened" value={`${ov.polg1_screened_pct}%`} color="#2e7d32" />
          <KPI label="Consanguineous" value={`${ov.consanguineous_pct}%`} color={ACCENT3} />
        </div>
      </SectionCard>

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

      <SectionCard title="Piracetam Note" borderColor={ACCENT4}>
        <p className="small mb-0">{ov.piracetam_note}</p>
      </SectionCard>

      <SectionCard title="Prognosis Note (Non-Fatal PME)" borderColor="#2e7d32">
        <p className="small mb-0">{ov.prognosis_note}</p>
      </SectionCard>

      <SectionCard title="Founder Populations Note" borderColor={ACCENT}>
        <p className="small mb-0">{ov.founder_note}</p>
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
            <div className="small text-muted mt-1">{e.mechanism}</div>
            <div className="small mt-1"><em>EEG:</em> {e.eeg_pattern}</div>
            <div className="small"><em>Onset:</em> {e.onset_typical} · <em>Progression:</em> {e.progression_rate}</div>
            {e.notable && <div className="small text-info"><em>Notable:</em> {e.notable}</div>}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Sample (first 15)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead>
              <tr style={{ backgroundColor: ACCENT, color: '#fff' }}>
                <th>ID</th><th>Name</th><th>Sex</th><th>Onset (y)</th>
                <th>Age</th><th>Genotype</th><th>VPA</th><th>LEV</th>
                <th>Piracetam</th><th>Photosens</th><th>Ambulatory</th>
              </tr>
            </thead>
            <tbody>
              {bk.patient_sample?.map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td>{p.name}</td>
                  <td>{p.sex}</td>
                  <td>{p.age_onset_years}</td>
                  <td>{p.age_now}</td>
                  <td><small>{p.genotype}</small></td>
                  <td>{p.on_vpa ? '✓' : '—'}</td>
                  <td>{p.on_lev ? '✓' : '—'}</td>
                  <td>{p.on_piracetam ? '✓' : '—'}</td>
                  <td>{p.photosensitivity ? '⚡' : '—'}</td>
                  <td>{p.ambulatory ? '✓' : '—'}</td>
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
            <div className="small text-muted"><em>EEG:</em> {s.eeg_correlate}</div>
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
        text="TREATMENT HIERARCHY: (1) Piracetam — Level A (action myoclonus, most effective in ULD); (2) VPA backbone (Level A); (3) LEV adjunct (Level B); (4) CLB (Level B, nocturnal dose for morning myoclonus); (5) Physiotherapy + OT (core non-pharmacological); (6) ZNS / KBr rescue. NEVER: CBZ / PHT / OXC / TGB / GBP / PGB. VPA requires POLG1 screen first."
        variant="info"
      />

      <SectionCard title="Pharmacological Treatments" borderColor={ACCENT}>
        {bk.treatments?.filter(t => !t.drug.includes('Physiotherapy')).map((t, i) => (
          <div key={i} className="mb-4 pb-3 border-bottom">
            <div className="d-flex justify-content-between align-items-start">
              <strong>{t.drug}</strong>
              <span
                className="badge ms-2"
                style={{ backgroundColor: t.level.includes('A') ? '#2e7d32' : t.level.includes('B') ? ACCENT : '#795548' }}
              >
                {t.level}
              </span>
            </div>
            <div className="small text-muted mt-1"><em>Dose:</em> {t.dose}</div>
            <div className="small mt-1"><em>MOA:</em> {t.moa}</div>
            <div className="small mt-1"><em>Efficacy:</em> {t.efficacy}</div>
            <div className="small mt-1"><em>Monitoring:</em> {t.monitoring}</div>
            <div className="small mt-1 text-primary"><em>ULD note:</em> {t.uld_note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Non-Pharmacological Treatment" borderColor="#2e7d32">
        {bk.treatments?.filter(t => t.drug.includes('Physiotherapy')).map((t, i) => (
          <div key={i}>
            <div className="d-flex justify-content-between align-items-start">
              <strong>{t.drug}</strong>
              <span className="badge ms-2" style={{ backgroundColor: '#2e7d32' }}>{t.level}</span>
            </div>
            <div className="small text-muted mt-1"><em>Protocol:</em> {t.dose}</div>
            <div className="small mt-1"><em>MOA:</em> {t.moa}</div>
            <div className="small mt-1"><em>Efficacy:</em> {t.efficacy}</div>
            <div className="small mt-1"><em>Monitoring:</em> {t.monitoring}</div>
            <div className="small mt-1 text-success"><em>ULD note:</em> {t.uld_note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications" borderColor={ACCENT2}>
        {bk.contraindications?.map((ci, i) => (
          <div key={i} className="mb-3 pb-3 border-bottom">
            <div className="d-flex justify-content-between align-items-start">
              <strong className="text-danger">{ci.drug_class}</strong>
              <span className="badge bg-danger ms-2">{ci.level}</span>
            </div>
            <div className="small mt-1"><em>Mechanism:</em> {ci.mechanism}</div>
            <div className="small mt-1 text-danger"><em>Emergency:</em> {ci.emergency_note}</div>
            <div className="small mt-1 text-success"><em>Alternative:</em> {ci.alternative}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead>
              <tr style={{ backgroundColor: ACCENT, color: '#fff' }}>
                <th>Item</th><th>Frequency / Notes</th>
              </tr>
            </thead>
            <tbody>
              {bk.monitoring?.map((m, i) => (
                <tr key={i}>
                  <td><strong><small>{m.item}</small></strong></td>
                  <td><small>{m.frequency}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Lifecycle Stages" borderColor={ACCENT4}>
        {bk.lifecycle?.map((s, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="fw-bold small" style={{ color: ACCENT4 }}>{s.stage} <em className="text-muted">({s.age})</em></div>
            <div className="small mt-1">{s.focus}</div>
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
            <div className="small text-muted mt-1">{c.explanation}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead>
              <tr style={{ backgroundColor: ACCENT4, color: '#fff' }}>
                <th>Threshold</th><th>Value</th>
              </tr>
            </thead>
            <tbody>
              {df.thresholds?.map((t, i) => (
                <tr key={i}>
                  <td><strong><small>{t.threshold}</small></strong></td>
                  <td><small>{t.value}</small></td>
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
            <div className="small text-muted">{s.description}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Key References" borderColor={ACCENT}>
        {df.references?.map((r, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="fw-bold small" style={{ color: ACCENT }}>{r.ref}</div>
            <div className="small text-muted">{r.full}</div>
            <div className="small mt-1"><em>Key finding:</em> {r.key_finding}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── MAIN PAGE ─────────────────────────────────────────────────────────────────
export default function CSTBPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/cstb/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2 || tab === 3) {
      if (!breakdown) {
        fetch(`${API}/api/cstb/breakdown`)
          .then(r => r.json()).then(setBreakdown).catch(e => setError(String(e)));
      }
    }
    if (tab === 4) {
      if (!definitions) {
        fetch(`${API}/api/cstb/definitions`)
          .then(r => r.json()).then(setDefinitions).catch(e => setError(String(e)));
      }
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3" style={{ borderLeft: `6px solid ${ACCENT}`, paddingLeft: 12 }}>
        <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
          CSTB Epilepsy — Unverricht-Lundborg Disease (EPM1)
        </h4>
        <div className="text-muted small">
          Cystatin B · Dodecamer Repeat Expansion · Most Common PME Worldwide ·
          Piracetam Level A · CBZ/OXC/PHT ABSOLUTE CI · Non-Fatal PME · 21q22.3
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
