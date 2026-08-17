'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a3a6e';   // navy blue — GATOR1/mTOR pathway
const ACCENT2 = '#8a0000';   // crimson — absolute CI / high-risk
const ACCENT3 = '#005040';   // teal — precision therapy / Everolimus
const ACCENT4 = '#6b3600';   // amber-brown — FCD IIb / surgery

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#e8f0ff', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: 11 }}>
      {text}
    </span>
  );
}

export default function NPRL2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/nprl2/overview`).then(r => r.json()),
      fetch(`${API}/api/nprl2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nprl2/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border text-primary" /><p className="mt-2">Loading NPRL2/NPRL3 GATOR1 data…</p></div>;
  if (error) return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1400 }}>
      {/* Header */}
      <div className="card mb-4 shadow" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #2d5a8e 60%, ${ACCENT3} 100%)` }}>
        <div className="card-body text-white py-4">
          <h2 className="mb-1 fw-bold">🧬 NPRL2 / NPRL3 Epilepsy — GATOR1 Complex</h2>
          <p className="mb-1 opacity-90" style={{ fontSize: 15 }}>
            Familial Focal Epilepsy with Variable Foci (FFEVF) · Focal Cortical Dysplasia IIb · mTOR Pathway Precision Therapy
          </p>
          <div className="d-flex flex-wrap gap-2 mt-2">
            <Badge text="NPRL2 · 3p24.3" color={ACCENT3} />
            <Badge text="NPRL3 · 8q24.22" color={ACCENT3} />
            <Badge text="GATOR1 Trimer = DEPDC5/NPRL2/NPRL3" color="#4a7acc" />
            <Badge text="AD (85%) · AR biallelic rare" color="#6a4a9e" />
            <Badge text="FCD IIb 52%" color={ACCENT4} />
            <Badge text="Everolimus Level B" color="#007755" />
            <Badge text="40 Patients" color="#555" />
          </div>
        </div>
      </div>

      {/* GATOR1 Mechanism Banner */}
      <div className="alert alert-info mb-3 py-2" style={{ fontSize: 13, borderLeft: `4px solid ${ACCENT}` }}>
        <strong>🔬 GATOR1 Mechanism:</strong> NPRL2 + NPRL3 + DEPDC5 form the GATOR1 heterotrimer. GATOR1 = GAP for Rag GTPases (RagA/B) →
        converts RagA-GTP → RagA-GDP → mTORC1 detaches from lysosome → mTOR off. NPRL2/3 LOF → constitutive mTORC1 hyperactivation →
        excess S6K1/4EBP1 phosphorylation → aberrant cortical growth → <strong>Focal Cortical Dysplasia IIb</strong> → FFEVF focal epilepsy.
        <strong> Precision Rx: Everolimus</strong> (mTORC1 allosteric inhibitor; trough 3–7 ng/mL).
      </div>

      {/* Absolute CI Banner */}
      <div className="alert alert-danger py-2 mb-3" style={{ fontSize: 13 }}>
        <strong>⛔ ABSOLUTE CI:</strong> Tiagabine → NCSE in focal epilepsy. &nbsp;|&nbsp;
        <strong>⚠️ HIGH RISK:</strong> VGB (not indicated focal epilepsy); VPA without POLG screen (Alpers); CBZ without HLA-B*15:02 (SJS/TEN); VPA females without VPPP (MHRA 2021). &nbsp;|&nbsp;
        <strong>⚙️ DRUG INTERACTION:</strong> CBZ (CYP3A4 inducer) reduces Everolimus trough by 50–70% — increase Everolimus dose if co-prescribed.
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ─────────────────────────────────────────────────── */}
      {tab === 0 && overview && (
        <>
          {/* KPIs */}
          <div className="row mb-2">
            <KPI label="Total Patients" value={overview.cohort_size} color={ACCENT} />
            <KPI label="NPRL2 Patients" value={overview.nprl2_patients} color={ACCENT} />
            <KPI label="NPRL3 Patients" value={overview.nprl3_patients} color="#4a7acc" />
            <KPI label="Drug Resistant" value={`${overview.drug_resistant_pct}%`} color={ACCENT2} />
            <KPI label="FCD MRI+" value={`${overview.fcd_mri_positive_pct}%`} color={ACCENT4} />
            <KPI label="Surgery" value={`${overview.surgery_pct}%`} color={ACCENT4} />
          </div>
          <div className="row mb-4">
            <KPI label="Engel I Post-Surgery" value={`${overview.surgery_engel_I_pct}%`} color={ACCENT3} />
            <KPI label="On Everolimus" value={`${overview.everolimus_pct}%`} color={ACCENT3} />
            <KPI label="Mean Onset Age" value={`${overview.mean_age_onset_years}y`} color="#555" />
            <KPI label="Inheritance" value="AD 85%" color="#6a4a9e" />
            <KPI label="GATOR1 Role" value="Rag GTPase GAP" color={ACCENT} />
            <KPI label="mTOR Target" value="Everolimus L-B" color={ACCENT3} />
          </div>

          <div className="row">
            <div className="col-md-6">
              <SectionCard title="Etiology Distribution" borderColor={ACCENT}>
                {overview.etiology_distribution && Object.entries(overview.etiology_distribution).map(([cat, info]) => (
                  <PctBar key={cat} label={cat} pct={info.pct} color={ACCENT} />
                ))}
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="Seizure Type Prevalence" borderColor={ACCENT4}>
                {overview.seizure_type_distribution && overview.seizure_type_distribution.map((s, i) => (
                  <PctBar key={i} label={s.type} pct={s.pct} color={ACCENT4} />
                ))}
              </SectionCard>
            </div>
          </div>

          <div className="row">
            <div className="col-md-6">
              <SectionCard title="Common Triggers" borderColor="#c07000">
                {overview.trigger_distribution && overview.trigger_distribution.map((t, i) => (
                  <PctBar key={i} label={t.trigger} pct={t.pct} color="#c07000" />
                ))}
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="GATOR1 Complex & Precision Therapy" borderColor={ACCENT3}>
                <div className="mb-3">
                  <strong className="text-info">GATOR1 Trimer:</strong>
                  <div className="mt-2 p-2" style={{ background: '#f0f8ff', borderRadius: 6, fontSize: 13 }}>
                    <div>🔲 <strong>DEPDC5</strong> (22q12.3) — Scaffold + GAP catalytic subunit (~70% of GATOR1 epilepsy)</div>
                    <div>🔲 <strong>NPRL2</strong> (3p24.3) — Bridges DEPDC5 ↔ NPRL3 (~20%)</div>
                    <div>🔲 <strong>NPRL3</strong> (8q24.22) — Anchors GATOR1 to Ragulator / lysosome (~10%)</div>
                    <div className="mt-1 text-muted">All three: LOF → mTORC1 hyperactivation → FCD IIb + FFEVF</div>
                  </div>
                </div>
                <div className="mb-2">
                  <strong>Precision Therapy:</strong>
                  <div style={{ fontSize: 13 }}>{overview.precision_therapy}</div>
                </div>
                <div>
                  <strong>Surgical Outcomes:</strong>
                  <div style={{ fontSize: 13 }}>{overview.surgical_outcome}</div>
                </div>
              </SectionCard>
            </div>
          </div>

          <SectionCard title="Key Contraindications Summary" borderColor={ACCENT2}>
            <div className="row">
              {overview.key_contraindications && overview.key_contraindications.map((ci, i) => (
                <div key={i} className="col-md-6 mb-2">
                  <div className="alert alert-danger py-1 mb-0" style={{ fontSize: 12 }}>⛔ {ci}</div>
                </div>
              ))}
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Tab 1: Patients & Etiology ──────────────────────────────────────── */}
      {tab === 1 && breakdown && (
        <>
          <SectionCard title="5-Class Etiology Catalog" borderColor={ACCENT}>
            {breakdown.etiology_catalog && breakdown.etiology_catalog.map((et, i) => (
              <div key={i} className="card mb-3 shadow-sm">
                <div className="card-header d-flex justify-content-between" style={{ backgroundColor: '#e8f0ff' }}>
                  <strong style={{ color: ACCENT }}>{et.etiology}</strong>
                  <span className="badge" style={{ backgroundColor: ACCENT }}>{et.pct}% (n={et.n})</span>
                </div>
                <div className="card-body" style={{ fontSize: 13 }}>
                  <div className="mb-2"><strong>Mechanism:</strong> {et.mechanism}</div>
                  <div className="mb-2"><strong>EEG Correlate:</strong> {et.eeg_correlate}</div>
                  <div className="row">
                    <div className="col-4"><Badge text={`Onset: ${et.typical_age_onset}`} color={ACCENT} /></div>
                    <div className="col-4"><Badge text={`DRE: ${et.drug_resistance}`} color={ACCENT2} /></div>
                    <div className="col-4"><Badge text={`FCD MRI+: ${et.fcd_mri}`} color={ACCENT4} /></div>
                  </div>
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Patient Sample (15 of 40)" borderColor={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm table-striped table-hover" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr>
                    <th>ID</th><th>Name</th><th>Age</th><th>Onset</th><th>Gene</th>
                    <th>Drug Resistant</th><th>Surgery</th><th>FCD MRI+</th><th>Everolimus</th><th>Sz/Month</th>
                  </tr>
                </thead>
                <tbody>
                  {breakdown.patient_sample && breakdown.patient_sample.map(p => (
                    <tr key={p.id}>
                      <td><strong>{p.id}</strong></td>
                      <td>{p.name}</td>
                      <td>{p.age}y</td>
                      <td>{p.age_onset}y</td>
                      <td><Badge text={p.gene} color={p.gene === 'NPRL2' ? ACCENT : '#4a7acc'} /></td>
                      <td>{p.drug_resistant ? <span className="text-danger fw-bold">Yes</span> : <span className="text-success">No</span>}</td>
                      <td>{p.surgery ? (p.seizure_free_post_surgery ? '✅ Free' : '⚠️ Partial') : '—'}</td>
                      <td>{p.fcd_mri ? <span className="text-warning fw-bold">+</span> : '—'}</td>
                      <td>{p.on_everolimus ? <span className="text-success fw-bold">Yes</span> : '—'}</td>
                      <td>{p.seizures_per_month}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}

      {/* ── Tab 2: Seizures & Triggers ──────────────────────────────────────── */}
      {tab === 2 && breakdown && (
        <>
          <SectionCard title="5 Seizure Types — NPRL2/NPRL3 GATOR1" borderColor={ACCENT4}>
            {breakdown.seizure_types && breakdown.seizure_types.map((s, i) => (
              <div key={i} className="card mb-3 shadow-sm">
                <div className="card-header d-flex justify-content-between" style={{ backgroundColor: '#fff5e6' }}>
                  <strong style={{ color: ACCENT4 }}>{s.type}</strong>
                  <Badge text={`${s.frequency_pct}%`} color={ACCENT4} />
                </div>
                <div className="card-body" style={{ fontSize: 13 }}>
                  <div className="mb-2"><strong>🧠 EEG:</strong> {s.eeg}</div>
                  <div className="mb-2"><strong>🎭 Semiology:</strong> {s.semiology}</div>
                  <div className="alert alert-info py-1 mb-0" style={{ fontSize: 12 }}>
                    <strong>💡 Clinical Tip:</strong> {s.clinical_tip}
                  </div>
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="8 Seizure Triggers" borderColor="#c07000">
            {breakdown.triggers && breakdown.triggers.map((t, i) => (
              <div key={i} className="mb-3">
                <div className="d-flex justify-content-between small mb-1">
                  <strong>{t.trigger}</strong>
                  <span className="text-muted">{t.pct}%</span>
                </div>
                <div className="progress mb-1" style={{ height: 10 }}>
                  <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: '#c07000' }} />
                </div>
                <div style={{ fontSize: 12, color: '#555' }}>{t.note}</div>
              </div>
            ))}
          </SectionCard>
        </>
      )}

      {/* ── Tab 3: Treatments ──────────────────────────────────────────────── */}
      {tab === 3 && breakdown && (
        <>
          {/* Contraindications first */}
          <SectionCard title="Contraindications" borderColor={ACCENT2}>
            {breakdown.contraindications && breakdown.contraindications.map((ci, i) => (
              <div key={i} className={`alert ${ci.level.includes('ABSOLUTE') ? 'alert-danger' : 'alert-warning'} mb-2 py-2`} style={{ fontSize: 13 }}>
                <div className="d-flex justify-content-between">
                  <strong>{ci.drug}</strong>
                  <Badge text={ci.level} color={ci.level.includes('ABSOLUTE') ? '#8a0000' : '#856404'} />
                </div>
                <div>{ci.reason}</div>
                <div className="mt-1 text-muted small"><strong>Action:</strong> {ci.action}</div>
              </div>
            ))}
          </SectionCard>

          {/* Treatments */}
          <SectionCard title="7 Treatments — NPRL2/NPRL3 GATOR1 Epilepsy" borderColor={ACCENT3}>
            {breakdown.treatments && breakdown.treatments.map((t, i) => (
              <div key={i} className="card mb-3 shadow-sm">
                <div className="card-header d-flex justify-content-between" style={{ backgroundColor: '#e8fff5' }}>
                  <strong style={{ color: ACCENT3 }}>{t.drug}</strong>
                  <Badge text={t.level} color={t.level.includes('Level A') ? ACCENT3 : (t.level.includes('Level B') ? '#4a7acc' : '#888')} />
                </div>
                <div className="card-body" style={{ fontSize: 13 }}>
                  <div className="row mb-2">
                    <div className="col-md-6">
                      <div><strong>Indication:</strong> {t.indication}</div>
                      <div><strong>Dose:</strong> {t.dose}</div>
                      <div><strong>MOA:</strong> {t.moa}</div>
                    </div>
                    <div className="col-md-6">
                      <div><strong>Efficacy:</strong> {t.efficacy}</div>
                      <div><strong>Safety:</strong> {t.safety}</div>
                      <div><strong>Monitoring:</strong> {t.monitoring}</div>
                    </div>
                  </div>
                  <div className="alert alert-secondary py-1 mb-0" style={{ fontSize: 12 }}>
                    <strong>🧬 GATOR1/NPRL2 Note:</strong> {t.gator1_note}
                  </div>
                </div>
              </div>
            ))}
          </SectionCard>

          {/* Monitoring */}
          <SectionCard title="Monitoring Protocol (14 items)" borderColor={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 13 }}>
                <thead className="table-dark">
                  <tr><th>Item</th><th>Interval</th><th>Rationale</th></tr>
                </thead>
                <tbody>
                  {breakdown.monitoring && breakdown.monitoring.map((m, i) => (
                    <tr key={i}>
                      <td><strong>{m.item}</strong></td>
                      <td>{m.interval}</td>
                      <td>{m.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          {/* Lifecycle */}
          <SectionCard title="6-Window Lifecycle" borderColor={ACCENT4}>
            {breakdown.lifecycle && breakdown.lifecycle.map((lc, i) => (
              <div key={i} className="mb-3">
                <div className="d-flex align-items-center mb-1">
                  <span className="badge me-2" style={{ backgroundColor: ACCENT4 }}>{i + 1}</span>
                  <strong style={{ color: ACCENT4 }}>{lc.window}</strong>
                </div>
                <div style={{ fontSize: 13, paddingLeft: 28 }}>{lc.description}</div>
              </div>
            ))}
          </SectionCard>
        </>
      )}

      {/* ── Tab 4: Definitions ─────────────────────────────────────────────── */}
      {tab === 4 && definitions && (
        <>
          <SectionCard title="15 Key Concepts — NPRL2/NPRL3 GATOR1" borderColor={ACCENT}>
            <div className="row">
              {definitions.concepts && definitions.concepts.map((c, i) => (
                <div key={i} className="col-md-6 mb-3">
                  <div className="card h-100 shadow-sm">
                    <div className="card-header py-1 fw-bold" style={{ backgroundColor: '#e8f0ff', color: ACCENT, fontSize: 13 }}>
                      {c.term}
                    </div>
                    <div className="card-body py-2" style={{ fontSize: 12 }}>{c.definition}</div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="12 Clinical Thresholds" borderColor={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm table-striped" style={{ fontSize: 13 }}>
                <thead className="table-dark">
                  <tr><th>Parameter</th><th>Value</th><th>Action</th></tr>
                </thead>
                <tbody>
                  {definitions.thresholds && definitions.thresholds.map((t, i) => (
                    <tr key={i}>
                      <td><strong>{t.parameter}</strong></td>
                      <td><Badge text={t.value} color={ACCENT4} /></td>
                      <td style={{ fontSize: 12 }}>{t.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <div className="row">
            <div className="col-md-6">
              <SectionCard title="12 Standards & Guidelines" borderColor={ACCENT3}>
                {definitions.standards && definitions.standards.map((s, i) => (
                  <div key={i} className="mb-2">
                    <Badge text={s.std} color={ACCENT3} />
                    <div style={{ fontSize: 12, color: '#555' }}>{s.domain}</div>
                  </div>
                ))}
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="6 Key References" borderColor={ACCENT}>
                {definitions.references && definitions.references.map((r, i) => (
                  <div key={i} className="mb-3">
                    <strong className="d-block" style={{ color: ACCENT }}>{r.ref}</strong>
                    <div style={{ fontSize: 12, color: '#444' }}>{r.citation}</div>
                  </div>
                ))}
              </SectionCard>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
