'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#4e342e'; // deep brown — nAChR α2 / habenular (distinct from CHRNA4 #6d4c41 and CHRNB2 #5d4037)
const DANGER = '#b71c1c';
const SUCCESS = '#2e7d32';
const WARN = '#e65100';
const HABENULA_COLOR = '#6a1b9a'; // purple for habenular/nicotine pathway
const DRE_COLOR = '#1565c0';

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
        <strong>🧬 CHRNA2 (8p21.2) — nAChR α2 Subunit — ADNFLE2 (OMIM #610353) — RAREST ADNFLE Gene:</strong>{' '}
        CHRNA2 encodes the <strong>α2 subunit</strong> of the neuronal nAChR, assembling as (α2)₂(β4)₃.{' '}
        Highest expression in the <strong>habenulo-interpeduncular pathway</strong> (nicotine reward circuit).{' '}
        GOF mutations (I279N/I304N in TM2/TM3) → delayed desensitisation → NREM cholinergic surges → frontal seizures.{' '}
        Completes the ADNFLE nAChR triad: CHRNA4 (ADNFLE1) + CHRNB2 (ADNFLE3) + <strong>CHRNA2 (ADNFLE2)</strong>.{' '}
        <span style={{ color: DANGER }} className="fw-bold">
          ⚠️ HLA-B*15:02: CBZ/OXC ABSOLUTE CI in SE Asian (SJS/TEN fatal) — use LCM. {' '}
          ⚠️ BUPROPION ABSOLUTE CI (lowers seizure threshold — both for depression AND smoking cessation). {' '}
          ⚠️ TGB ABSOLUTE CI (NCSE). {' '}
          ⚠️ VARENICLINE HIGH RISK (partial α2β4 agonist — directly activates GOF habenular receptor). {' '}
          ⚠️ High-dose nicotine (&gt;7 mg patch) HIGH RISK. {' '}
          ⚠️ CBZ autoinduction: re-check TDM at 6-8 weeks.
        </span>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={cohort.total || 40} />
        <KPI label="Seizure-Free ≥6mo" value={cohort.seizure_free_6mo || '—'} color={SUCCESS} />
        <KPI label="Drug-Resistant (DRE)" value={cohort.dre_patients || '—'} color={DRE_COLOR} />
        <KPI label="VPSG Performed" value={cohort.vpsg_performed || '—'} color={COLOR} />
        <KPI label="HLA-B*15:02 Tested" value={cohort.hla_b1502_tested || '—'} color={WARN} />
        <KPI label="Misdiag. Parasomnia" value={cohort.misdiagnosed_parasomnia || '—'} color={HABENULA_COLOR} />
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
                <tr><td className="fw-semibold">Companion genes</td><td><span style={{color:COLOR}}>{data.companion_genes}</span></td></tr>
                <tr><td className="fw-semibold">Syndrome</td><td>{data.syndrome}</td></tr>
                <tr><td className="fw-semibold">Triad position</td><td><span className="badge text-bg-warning">{data.adnfle_triad_position}</span></td></tr>
                <tr><td className="fw-semibold">Inheritance</td><td>{data.inheritance}</td></tr>
                <tr><td className="fw-semibold">OMIM (gene)</td><td>{data.omim_gene}</td></tr>
                <tr><td className="fw-semibold">OMIM (ADNFLE2)</td><td>{data.omim_adnfle2}</td></tr>
                <tr><td className="fw-semibold">First variant</td><td>{data.first_mutation}</td></tr>
                <tr><td className="fw-semibold">Misdiagnosis</td><td><span style={{color:WARN}}>{data.hallmark_misdiagnosis}</span></td></tr>
              </tbody>
            </table>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="⚗️ Key Mutations">
            {data.key_mutations && Object.entries(data.key_mutations).map(([mut, desc]) => (
              <div key={mut} className="mb-2 p-2 rounded" style={{background:'#f5f5f5'}}>
                <span className="badge me-2" style={{background:COLOR}}>{mut}</span>
                <span className="small">{desc}</span>
              </div>
            ))}
          </SectionCard>
          <SectionCard title="🟣 Habenular Significance" borderColor={HABENULA_COLOR}>
            <p className="small mb-0">{data.habenular_significance}</p>
          </SectionCard>
        </div>
      </div>

      {/* Etiology distribution */}
      <SectionCard title="📊 Etiology Distribution">
        {etiologies.map(e => (
          <Bar key={e.class} label={e.class.replace(/-/g, ' ')} value={e.pct} color={COLOR} />
        ))}
      </SectionCard>

      {/* Precision pharmacology */}
      <SectionCard title="💊 Precision Pharmacology">
        <p className="small mb-2">{data.precision_pharmacology}</p>
        <ul className="small mb-0 ps-3">
          {(data.key_contraindications || []).map((ci, i) => (
            <li key={i} className="text-danger fw-semibold mb-1">{ci}</li>
          ))}
        </ul>
      </SectionCard>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading patients…</div>;
  const patients = data.patients || [];
  const etiologies = data.etiologies || [];

  return (
    <div>
      <SectionCard title="🧬 Etiology Profiles">
        {etiologies.map(e => (
          <div key={e.category} className="mb-3 p-3 rounded" style={{background:'#efebe9', borderLeft:`4px solid ${COLOR}`}}>
            <div className="fw-semibold mb-1">{e.category} — <span className="text-muted">{e.pct}%</span></div>
            <div className="small text-muted">{e.mechanism}</div>
            {e.eeg && <div className="small mt-1"><strong>EEG:</strong> {e.eeg}</div>}
            {e.onset_months && <div className="small"><strong>Onset:</strong> {e.onset_months}</div>}
            {e.severity && <div className="small"><strong>Severity:</strong> {e.severity}</div>}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="👥 Patient Cohort (n=40)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead style={{background:COLOR, color:'white'}}>
              <tr>
                <th>ID</th><th>Name</th><th>Age</th><th>Sex</th><th>Onset</th>
                <th>Variant</th><th>Etiology</th><th>AED</th>
                <th>SF 6mo</th><th>DRE</th><th>VPSG</th><th>Misdiag</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.name}</td>
                  <td>{p.age}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_age}y</td>
                  <td><code className="small">{p.variant}</code></td>
                  <td className="small">{p.etiology?.split('-').slice(0,2).join('-')}</td>
                  <td><span className="badge text-bg-secondary small">{p.current_aed}</span></td>
                  <td>{p.seizure_free_6mo ? <span className="text-success">✓</span> : <span className="text-danger">✗</span>}</td>
                  <td>{p.dre ? <span className="text-danger fw-bold">DRE</span> : '—'}</td>
                  <td>{p.vpsg_performed ? <span className="text-success">✓</span> : '—'}</td>
                  <td>{p.misdiagnosed_parasomnia ? <span className="text-warning">Para</span> : '—'}</td>
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
  if (!data) return <div className="text-center py-4 text-muted">Loading seizures…</div>;
  const seizures = data.seizure_types || [];
  const triggers = data.triggers || [];

  return (
    <div>
      <SectionCard title="⚡ Seizure Types">
        {seizures.map(s => (
          <div key={s.type} className="mb-3 p-3 rounded" style={{background:'#efebe9', borderLeft:`4px solid ${s.emergency ? DANGER : COLOR}`}}>
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-semibold">{s.type.replace(/-/g, ' ')}</span>
              <span className="badge" style={{background: s.pct >= 80 ? DANGER : COLOR}}>{s.pct}%</span>
            </div>
            <div className="small text-muted">{s.description}</div>
            {s.eeg && <div className="small mt-1"><strong>EEG:</strong> {s.eeg}</div>}
            {s.semiology_tips && <div className="small mt-1 text-info"><strong>Tips:</strong> {s.semiology_tips}</div>}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="⚠️ Seizure Triggers">
        {triggers.map(t => (
          <div key={t.trigger} className="mb-2 p-2 rounded d-flex gap-2 align-items-start" style={{background:'#fff3e0'}}>
            <span className="badge flex-shrink-0" style={{background:WARN}}>{t.pct}%</span>
            <div>
              <div className="fw-semibold small">{t.trigger.replace(/-/g, ' ')}</div>
              <div className="text-muted small">{t.mechanism}</div>
              <div className="small text-success"><strong>Management:</strong> {t.management}</div>
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading treatments…</div>;
  const treatments = data.treatments || [];
  const cis = data.contraindications || [];
  const monitoring = data.monitoring || [];
  const lifecycle = data.lifecycle || [];

  return (
    <div>
      <SectionCard title="💊 Treatments">
        {treatments.map(t => (
          <div key={t.drug} className="mb-3 p-3 rounded" style={{background:'#efebe9', borderLeft:`4px solid ${COLOR}`}}>
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-semibold">{t.drug}</span>
              <span className="badge me-1" style={{background:COLOR}}>{t.evidence}</span>
              <span className="badge text-bg-secondary">{t.role}</span>
            </div>
            <div className="small text-muted mb-1">{t.mechanism}</div>
            <div className="small"><strong>Dose:</strong> {t.dose}</div>
            <div className="small"><strong>Monitoring:</strong> {t.monitoring}</div>
            {t.chrna2_specific && (
              <div className="small mt-1 p-1 rounded" style={{background:'#e8f5e9'}}>
                <strong style={{color:SUCCESS}}>CHRNA2-specific:</strong> {t.chrna2_specific}
              </div>
            )}
            {t.sf_pct && (
              <div className="mt-2">
                <div className="small mb-1">Seizure-free rate: {t.sf_pct}%</div>
                <div className="progress" style={{height:8}}>
                  <div className="progress-bar" style={{width:`${t.sf_pct}%`, backgroundColor: SUCCESS}} />
                </div>
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications" borderColor={DANGER}>
        {cis.map(ci => (
          <div key={ci.drug} className="mb-2 p-2 rounded" style={{background:'#ffebee'}}>
            <div className="d-flex gap-2 align-items-start">
              <span className={`badge flex-shrink-0 ${ci.level.includes('ABSOLUTE') ? 'text-bg-danger' : 'text-bg-warning'}`}>{ci.level}</span>
              <div>
                <div className="fw-semibold small">{ci.drug}</div>
                <div className="text-muted small">{ci.reason}</div>
                {ci.alternative && <div className="small text-success"><strong>Alternative:</strong> {ci.alternative}</div>}
              </div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔍 Monitoring">
        <div className="table-responsive">
          <table className="table table-sm small mb-0">
            <thead><tr><th>Item</th><th>Frequency</th><th>Rationale</th></tr></thead>
            <tbody>
              {monitoring.map(m => (
                <tr key={m.item}>
                  <td className="fw-semibold">{m.item}</td>
                  <td>{m.frequency}</td>
                  <td className="text-muted">{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🔄 Lifecycle Stages">
        {lifecycle.map(l => (
          <div key={l.stage} className="mb-3 p-3 rounded" style={{background:'#efebe9', borderLeft:`4px solid ${COLOR}`}}>
            <div className="fw-semibold mb-2">{l.stage}</div>
            <ul className="small mb-1 ps-3">
              {(l.key_issues || []).map((issue, i) => <li key={i}>{issue}</li>)}
            </ul>
            <div className="small text-success"><strong>Treatment focus:</strong> {l.treatment_focus}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions…</div>;
  const concepts = data.concepts || [];
  const thresholds = data.thresholds || [];
  const standards = data.evidence_standards || [];
  const distinctions = data.key_pharmacological_distinctions || [];

  return (
    <div>
      <SectionCard title="📚 Key Concepts (15)">
        {concepts.map(c => (
          <div key={c.concept} className="mb-2 p-2 rounded" style={{background:'#efebe9', borderLeft:`3px solid ${COLOR}`}}>
            <div className="fw-semibold small">{c.concept}</div>
            <div className="text-muted small">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="⚗️ Key Pharmacological Distinctions" borderColor={HABENULA_COLOR}>
        {distinctions.map((d, i) => (
          <div key={i} className="mb-2 p-2 rounded small" style={{background:'#f3e5f5', borderLeft:`3px solid ${HABENULA_COLOR}`}}>
            {d}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Thresholds">
        <div className="table-responsive">
          <table className="table table-sm small mb-0">
            <thead><tr><th>Threshold</th><th>Value</th><th>Rationale</th></tr></thead>
            <tbody>
              {thresholds.map(t => (
                <tr key={t.threshold}>
                  <td className="fw-semibold">{t.threshold}</td>
                  <td><code className="small">{t.value}</code></td>
                  <td className="text-muted">{t.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📋 Evidence Standards">
        <div className="table-responsive">
          <table className="table table-sm small mb-0">
            <thead><tr><th>Standard</th><th>Relevance</th></tr></thead>
            <tbody>
              {standards.map(s => (
                <tr key={s.standard}>
                  <td className="fw-semibold">{s.standard}</td>
                  <td className="text-muted">{s.relevance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

export default function CHRNA2Dashboard() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/chrna2/overview`).then(r => r.json()),
      fetch(`${API}/api/chrna2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/chrna2/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return (
    <div className="container py-5 text-center">
      <div className="spinner-border" style={{ color: COLOR }} />
      <p className="mt-3 text-muted">Loading CHRNA2 dashboard…</p>
    </div>
  );
  if (error) return (
    <div className="container py-5">
      <div className="alert alert-danger">Error: {error}</div>
    </div>
  );

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-3 mb-3">
        <div className="rounded-circle d-flex align-items-center justify-content-center text-white fw-bold"
          style={{ width: 56, height: 56, background: COLOR, fontSize: 20 }}>α2</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>CHRNA2 Epilepsy — ADNFLE2</h4>
          <div className="text-muted small">
            nAChR α2 Subunit · 8p21.2 · Habenulo-Interpeduncular Pathway ·
            Rarest ADNFLE Gene (&lt;10 families) · Completing ADNFLE nAChR Triad
          </div>
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${activeTab === i ? 'active fw-semibold' : ''}`}
              style={activeTab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setActiveTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {activeTab === 0 && <OverviewTab data={overview} />}
      {activeTab === 1 && <PatientsTab data={breakdown} />}
      {activeTab === 2 && <SeizuresTab data={breakdown} />}
      {activeTab === 3 && <TreatmentsTab data={breakdown} />}
      {activeTab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
