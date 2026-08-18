'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#5d4037'; // warm chestnut brown — nAChR β2 / cholinergic (distinct from CHRNA4 #6d4c41 darker)
const DANGER = '#b71c1c';
const SUCCESS = '#2e7d32';
const WARN = '#e65100';
const PSYCH_COLOR = '#6a1b9a'; // purple for psychiatric comorbidity (distinct)
const COG_COLOR = '#1565c0';   // blue for cognitive impairment

function KPI({ label, value, color = COLOR }) {
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

function Bar({ label, value, max = 100, color = COLOR }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card shadow-sm mb-3">
      <div className="card-header fw-semibold text-white py-2" style={{ background: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const cohort = data.cohort || {};
  const etiologies = data.etiologies || [];

  return (
    <div>
      {/* Key alert banner */}
      <div className="alert py-2 small mb-3 border" style={{ borderColor: COLOR, borderLeftWidth: 5, background: '#efebe9' }}>
        <strong>🧬 CHRNB2 (1q21.3) — nAChR β2 Subunit — (α4)₂(β2)₃ Heteropentamer — ADNFLE3 (OMIM #605375):</strong>{' '}
        CHRNB2 encodes the <strong>β2 subunit</strong> of the neuronal nicotinic acetylcholine receptor.{' '}
        Partners with CHRNA4-encoded α4 subunits to form the <strong>high-sensitivity (α4)₂(β2)₃</strong> isoform (EC50 ~1 µM ACh).{' '}
        GOF mutations (V287M/V287L in TM2) → delayed desensitisation → excessive NREM cholinergic current → nocturnal frontal lobe seizures.{' '}
        <span style={{ color: DANGER }} className="fw-bold">
          ⚠️ HLA-B*15:02: CBZ/OXC ABSOLUTE CI in SE Asian (SJS/TEN fatal) — use LCM. {' '}
          ⚠️ BUPROPION ABSOLUTE CI (lowers seizure threshold — dangerous with 40% psychiatric comorbidity). {' '}
          ⚠️ TGB ABSOLUTE CI (NCSE). {' '}
          ⚠️ VARENICLINE HIGH RISK (partial α4β2 agonist). {' '}
          ⚠️ V287M: 30% cognitive impairment — prefer OXC over CBZ. {' '}
          ⚠️ CBZ autoinduction: re-check TDM at 6-8 weeks.
        </span>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={cohort.total || 40} />
        <KPI label="Seizure-Free ≥6mo" value={cohort.seizure_free_6mo || '—'} color={SUCCESS} />
        <KPI label="Psychiatric Comorbidity" value={cohort.psychiatric_comorbidity || '—'} color={PSYCH_COLOR} />
        <KPI label="Cognitive (V287M)" value={cohort.cognitive_impairment_v287m || '—'} color={COG_COLOR} />
        <KPI label="HLA-B*15:02 Tested" value={cohort.hla_b1502_tested || '—'} color={WARN} />
        <KPI label="Avg CBZ TDM (µg/mL)" value={cohort.avg_cbz_tdm_ug_ml || '—'} color={COLOR} />
      </div>

      {/* Gene info */}
      <div className="row g-3 mb-3">
        <div className="col-md-6">
          <SectionCard title="🔬 Gene & Channel">
            <table className="table table-sm table-borderless mb-0 small">
              <tbody>
                <tr><td className="fw-semibold">Gene</td><td>{data.gene} ({data.locus})</td></tr>
                <tr><td className="fw-semibold">Full name</td><td>{data.full_name}</td></tr>
                <tr><td className="fw-semibold">Protein</td><td>{data.protein}</td></tr>
                <tr><td className="fw-semibold">Channel</td><td>{data.channel}</td></tr>
                <tr><td className="fw-semibold">Companion gene</td><td><span style={{color:COLOR}}>{data.companion_gene}</span></td></tr>
                <tr><td className="fw-semibold">Syndrome</td><td>{data.syndrome}</td></tr>
                <tr><td className="fw-semibold">Inheritance</td><td>{data.inheritance}</td></tr>
                <tr><td className="fw-semibold">OMIM (gene)</td><td>{data.omim_gene}</td></tr>
                <tr><td className="fw-semibold">OMIM (ADNFLE3)</td><td>{data.omim_adnfle3}</td></tr>
                <tr><td className="fw-semibold">First mutation</td><td>{data.first_mutation}</td></tr>
              </tbody>
            </table>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="💊 Precision Pharmacology">
            <p className="small mb-2">{data.precision_pharmacology}</p>
            <p className="small mb-2 fw-semibold" style={{color:PSYCH_COLOR}}>
              Key distinction from CHRNA4: {data.key_distinction_from_chrna4}
            </p>
            <p className="small mb-0 fw-semibold text-danger">Hallmark misdiagnosis: {data.hallmark_misdiagnosis}</p>
          </SectionCard>
          <SectionCard title="🚨 Top Contraindications" borderColor={DANGER}>
            <ul className="small mb-0 ps-3">
              {(data.key_contraindications || []).map((ci, i) => (
                <li key={i} className="text-danger fw-semibold mb-1">{ci}</li>
              ))}
            </ul>
          </SectionCard>
        </div>
      </div>

      {/* Key mutations */}
      <SectionCard title="🧬 Key Pathogenic Variants (TM2 Cluster)">
        <div className="row g-2">
          {Object.entries(data.key_mutations || {}).map(([mut, desc]) => (
            <div key={mut} className="col-md-4">
              <div className="card border-0 h-100" style={{background:'#efebe9'}}>
                <div className="card-body py-2 px-3">
                  <div className="fw-bold small" style={{color:COLOR}}>{mut}</div>
                  <div className="small text-muted">{desc}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Etiology distribution */}
      <SectionCard title="📊 Etiology Distribution (40 Patients)">
        {etiologies.map((et, i) => (
          <Bar key={i} label={et.class} value={et.pct} color={COLOR} />
        ))}
      </SectionCard>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const patients = data.patients || [];
  const etiologies = data.etiologies || [];

  return (
    <div>
      <SectionCard title="🧬 Etiology Catalog — 5 Classes">
        {etiologies.map((et, i) => (
          <div key={i} className="mb-3 p-2 border-start border-3" style={{ borderColor: COLOR }}>
            <div className="fw-semibold small" style={{ color: COLOR }}>
              {et.category} — {et.pct}%
            </div>
            <div className="small text-muted mb-1">{et.mechanism}</div>
            <div className="row g-1 small">
              <div className="col-md-4"><span className="fw-semibold">EEG: </span>{et.eeg}</div>
              <div className="col-md-4"><span className="fw-semibold">Onset: </span>{et.onset_months}</div>
              <div className="col-md-4"><span className="fw-semibold">Severity: </span>{et.severity}</div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="👥 Patient Cohort (first 15 shown)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Etiology</th><th>Onset (y)</th><th>CBZ dose (mg)</th>
                <th>TDM (µg/mL)</th><th>SF 6mo</th><th>Psych</th><th>Cognitive</th>
              </tr>
            </thead>
            <tbody>
              {patients.slice(0, 15).map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td className="small">{p.etiology.replace(/-/g, ' ')}</td>
                  <td>{p.age_onset_years}</td>
                  <td>{p.cbz_xr_dose_mg}</td>
                  <td style={{color: p.cbz_tdm_ug_ml >= 8 ? SUCCESS : DANGER}}>{p.cbz_tdm_ug_ml}</td>
                  <td>{p.seizure_free_6mo ? <span style={{color:SUCCESS}}>✓</span> : <span style={{color:DANGER}}>✗</span>}</td>
                  <td>{p.psychiatric_comorbidity ? <span style={{color:PSYCH_COLOR}}>✓</span> : '—'}</td>
                  <td>{p.cognitive_impairment_v287m ? <span style={{color:COG_COLOR}}>✓</span> : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const seizures = data.seizure_types || [];
  const triggers = data.triggers || [];

  return (
    <div>
      <SectionCard title="⚡ Seizure Types — 5 Types">
        {seizures.map((sz, i) => (
          <div key={i} className="mb-3 p-2 border-start border-3" style={{ borderColor: COLOR }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-semibold small" style={{ color: COLOR }}>{sz.type}</span>
              <span className="badge" style={{ background: COLOR }}>{sz.pct}%</span>
            </div>
            <div className="small text-muted mb-1">{sz.semiology}</div>
            <div className="small"><span className="fw-semibold">EEG: </span>{sz.eeg}</div>
            <div className="small mt-1 p-1 rounded" style={{background:'#efebe9'}}>
              <span className="fw-semibold">💡 Tip: </span>{sz.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔥 Triggers — 8 Triggers">
        {triggers.map((tr, i) => (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span className="fw-semibold">{tr.trigger}</span>
              <span>{tr.pct}%</span>
            </div>
            <div className="progress mb-1" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${tr.pct}%`, backgroundColor: COLOR }} />
            </div>
            <div className="small text-muted">{tr.mechanism}</div>
            <div className="small fw-semibold mt-1">{tr.management}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const treatments = data.treatments || [];
  const cis = data.contraindications || [];
  const monitoring = data.monitoring || [];
  const lifecycle = data.lifecycle || [];

  return (
    <div>
      <SectionCard title="💊 Treatments — 8 AEDs (CHRNB2-specific)">
        {treatments.map((tx, i) => (
          <div key={i} className="mb-3 p-2 border-start border-3" style={{ borderColor: COLOR }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-semibold" style={{ color: COLOR }}>{tx.drug}</span>
              <span className="badge text-white" style={{ background: WARN }}>{tx.level}</span>
            </div>
            <div className="row g-2 small">
              <div className="col-md-6"><span className="fw-semibold">Mechanism: </span>{tx.mechanism}</div>
              <div className="col-md-6"><span className="fw-semibold">Dose: </span>{tx.dose}</div>
              <div className="col-md-6"><span className="fw-semibold">Efficacy: </span>{tx.efficacy}</div>
              <div className="col-md-6"><span className="fw-semibold">Monitoring: </span>{tx.monitoring}</div>
            </div>
            <div className="small mt-1 p-1 rounded" style={{ background: '#efebe9' }}>
              <span className="fw-semibold">🧬 CHRNB2-specific: </span>{tx.chrnb2_specific}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications — 6 Items" borderColor={DANGER}>
        {cis.map((ci, i) => (
          <div key={i} className="mb-2 p-2 border-start border-3 border-danger">
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-semibold text-danger small">{ci.drug}</span>
              <span className="badge bg-danger">{ci.severity}</span>
            </div>
            <div className="small text-muted mb-1">{ci.reason}</div>
            {ci.alternative && <div className="small fw-semibold" style={{color:SUCCESS}}>Alternative: {ci.alternative}</div>}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔍 Monitoring — 14 Items">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead className="table-light"><tr><th>Item</th><th>Frequency</th><th>Rationale</th></tr></thead>
            <tbody>
              {monitoring.map((m, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{m.item}</td>
                  <td>{m.frequency}</td>
                  <td className="text-muted">{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🗓️ Lifecycle — 6 Stages">
        {lifecycle.map((lc, i) => (
          <div key={i} className="mb-2 p-2 border rounded">
            <div className="fw-semibold small" style={{ color: COLOR }}>{lc.stage} <span className="text-muted">({lc.age})</span></div>
            <div className="small text-muted mb-1">{lc.description}</div>
            <div className="small">
              <span className="fw-semibold">Priority: </span>
              {(lc.priority_actions || []).join(' · ')}
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const concepts = data.concepts || [];
  const thresholds = data.thresholds || [];
  const standards = data.evidence_standards || [];
  const distinctions = data.key_pharmacological_distinctions || [];

  return (
    <div>
      <SectionCard title="🔑 Key Pharmacological Distinctions" borderColor={DANGER}>
        <ul className="small mb-0">
          {distinctions.map((d, i) => (
            <li key={i} className="mb-2 fw-semibold" style={{color: i < 3 ? DANGER : COLOR}}>{d}</li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="📖 15 Key Concepts">
        {concepts.map((c, i) => (
          <div key={i} className="mb-2 p-2 border-start border-2" style={{ borderColor: COLOR }}>
            <div className="fw-semibold small" style={{ color: COLOR }}>{c.concept}</div>
            <div className="small text-muted">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Thresholds — 12 Items">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead className="table-light"><tr><th>Parameter</th><th>Value</th><th>Context</th></tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{t.parameter}</td>
                  <td style={{ color: COLOR }}>{t.value}</td>
                  <td className="text-muted">{t.context}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📚 Evidence Standards — 12 Items">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead className="table-light"><tr><th>Standard</th><th>Applies to</th></tr></thead>
            <tbody>
              {standards.map((s, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{s.standard}</td>
                  <td className="text-muted">{s.applies_to}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
          🧬 Gene Identity
        </div>
        <div className="card-body">
          <table className="table table-sm table-borderless mb-0 small">
            <tbody>
              <tr><td className="fw-semibold">Gene</td><td>{data.gene}</td></tr>
              <tr><td className="fw-semibold">Full name</td><td>{data.full_name}</td></tr>
              <tr><td className="fw-semibold">Locus</td><td>{data.locus}</td></tr>
              <tr><td className="fw-semibold">OMIM</td><td>{data.omim}</td></tr>
              <tr><td className="fw-semibold">Protein</td><td>{data.protein}</td></tr>
              <tr><td className="fw-semibold">Channel family</td><td>{data.channel_family}</td></tr>
              <tr><td className="fw-semibold">Syndrome (ADNFLE3)</td><td>{data.syndrome?.ADNFLE3}</td></tr>
              <tr><td className="fw-semibold">Companion</td><td>{data.syndrome?.Companion}</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function CHRNB2Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchAll() {
      setLoading(true);
      setError(null);
      try {
        const [ov, bk, df] = await Promise.all([
          fetch(`${API}/api/chrnb2/overview`).then(r => r.json()),
          fetch(`${API}/api/chrnb2/breakdown`).then(r => r.json()),
          fetch(`${API}/api/chrnb2/definitions`).then(r => r.json()),
        ]);
        setOverview(ov);
        setBreakdown(bk);
        setDefinitions(df);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchAll();
  }, []);

  const renderTab = () => {
    if (loading) return <div className="text-center py-5"><div className="spinner-border" style={{color:COLOR}} /></div>;
    if (error) return <div className="alert alert-danger">Error: {error}</div>;
    switch (activeTab) {
      case 0: return <OverviewTab data={overview} />;
      case 1: return <PatientsTab data={breakdown} />;
      case 2: return <SeizuresTab data={breakdown} />;
      case 3: return <TreatmentsTab data={breakdown} />;
      case 4: return <DefinitionsTab data={definitions} />;
      default: return null;
    }
  };

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <div className="rounded-circle d-flex align-items-center justify-content-center text-white fw-bold"
          style={{ width: 48, height: 48, background: COLOR, fontSize: 18 }}>β2</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            CHRNB2 Epilepsy — ADNFLE Type 3
          </h4>
          <div className="text-muted small">
            nAChR β2 Subunit · (α4)₂(β2)₃ Heteropentamer · GOF Delayed Desensitisation ·
            CBZ-XR First-Line · HLA-B*15:02 · Psychiatric Comorbidity 40% · 1q21.3
          </div>
        </div>
      </div>

      {/* Companion gene alert */}
      <div className="alert alert-info py-2 small mb-3">
        <strong>🔗 Companion gene:</strong> CHRNB2 (β2, 1q21.3) + <strong>CHRNA4</strong> (α4, 20q13.33) form the{' '}
        <strong>(α4)₂(β2)₃ nAChR</strong> — mutations in either subunit cause clinically identical ADNFLE.{' '}
        Key CHRNB2 distinction: <strong>40% psychiatric comorbidity</strong> (vs 20% CHRNA4) and{' '}
        <strong>V287M → 30% cognitive impairment</strong>.{' '}
        <a href="/chrna4" style={{color:COLOR}}>→ CHRNA4 dashboard</a>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((tab, i) => (
          <li key={tab} className="nav-item">
            <button
              className={`nav-link ${activeTab === i ? 'active fw-semibold' : ''}`}
              style={activeTab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setActiveTab(i)}
            >{tab}</button>
          </li>
        ))}
      </ul>

      {renderTab()}
    </div>
  );
}
