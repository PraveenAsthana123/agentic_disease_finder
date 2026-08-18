'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR  = '#e64a19'; // deep burnt-orange — mTOR apex kinase / rapamycin target (distinct from all others)
const DANGER = '#b71c1c';
const SUCCESS = '#1b5e20';
const WARN   = '#f57f17';
const SOMATIC_COLOR = '#4a148c'; // purple for somatic/mosaic
const PRECISION_COLOR = '#00695c'; // teal for everolimus precision

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
  const alerts = data.key_alerts || [];

  return (
    <div>
      {/* Critical alerts */}
      <div className="alert py-2 small mb-3 border" style={{ borderColor: COLOR, borderLeftWidth: 5, background: '#fbe9e7' }}>
        <strong style={{ color: COLOR }}>MTOR (1p36.22) — mTOR Kinase / mTORopathy Apex</strong>
        <div className="mt-1 text-muted">
          GOF somatic mosaic (75%) or germline Smith-Kingsmore (25%) · FCD IIb / HME / MCAP ·
          Everolimus direct mTORC1 inhibitor (FKBP12-FRB binding) · Somatic VAF often &lt;5% in blood →
          deep sequencing mandatory · Apex of mTOR pathway: TSC1/TSC2 → DEPDC5/NPRL2/NPRL3 → <strong>MTOR</strong>
        </div>
      </div>

      {/* Absolute CI banner */}
      <div className="alert alert-danger py-2 small mb-3">
        <strong>ABSOLUTE CI:</strong> Tiagabine (TGB → NCSE in FCD IIb) ·
        VPA without POLG1 screen (Alpers overlap — fatal hepatotoxicity) ·
        Live vaccines on everolimus (immunosuppression — disseminated vaccine infection)
      </div>

      {/* Somatic + everolimus banner */}
      <div className="alert py-2 small mb-3" style={{ background: '#e8f5e9', borderColor: SUCCESS, borderLeftWidth: 5, border: '1px solid' }}>
        <strong style={{ color: SUCCESS }}>PRECISION:</strong>{' '}
        Everolimus 2.5–5 mg/day · trough 3–7 ng/mL (neurological dose) ·
        FKBP12-rapamycin binds FRB domain of mutant mTOR protein directly ·
        CYP3A4: CBZ/PHT/OXC reduce everolimus 70–80% → increase dose 2–3× + intensive TDM ·
        Azoles increase &gt;10× → dose reduce 90%
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-3">
        <KPI label="Cohort (N)" value={cohort.n ?? 40} color={COLOR} />
        <KPI label="Seizure-free %" value={`${cohort.seizure_free_pct ?? '--'}%`} color={SUCCESS} />
        <KPI label="On Everolimus" value={`${cohort.on_everolimus_pct ?? '--'}%`} color={PRECISION_COLOR} />
        <KPI label="Post-Surgery" value={`${cohort.post_surgery_pct ?? '--'}%`} color={WARN} />
        <KPI label="Somatic mutation" value={`${cohort.somatic_mutation_pct ?? '--'}%`} color={SOMATIC_COLOR} />
        <KPI label="Locus" value="1p36.22" color="#37474f" />
      </div>

      {/* Etiology breakdown */}
      <SectionCard title="5-Class Etiology Spectrum (40 patients)" borderColor={COLOR}>
        <div className="row">
          <div className="col-md-6">
            {etiologies.map(e => (
              <Bar key={e.category} label={e.category} value={e.pct} color={COLOR} />
            ))}
          </div>
          <div className="col-md-6">
            {etiologies.map(e => (
              <div key={e.category} className="mb-2 small border-start ps-2" style={{ borderColor: COLOR, borderLeftWidth: 3 }}>
                <strong>{e.category}</strong>
                <div className="text-muted">{e.severity}</div>
              </div>
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Key alerts */}
      <SectionCard title="Key Clinical Alerts" borderColor={DANGER}>
        {alerts.map((a, i) => (
          <div key={i} className="d-flex align-items-start mb-2 small">
            <span className="me-2 fw-bold" style={{ color: DANGER }}>⚠</span>
            <span>{a}</span>
          </div>
        ))}
      </SectionCard>

      {/* mTOR pathway context */}
      <SectionCard title="mTORopathy Pathway — MTOR is the Apex" borderColor={PRECISION_COLOR}>
        <div className="small text-muted mb-2">
          MTOR completes the mTOR pathway trilogy. All upstream mTORopathy genes converge here:
        </div>
        <div className="row g-2">
          {[
            { arm: 'PI3K-AKT Arm (Growth Factors)', genes: 'TSC1 / TSC2', desc: 'TSC2 inhibits RHEB → RHEB activates mTORC1. Mutations: tubers, bilateral.', color: '#6a1b9a' },
            { arm: 'GATOR1 Arm (Amino Acids)', genes: 'DEPDC5 / NPRL2 / NPRL3', desc: 'GATOR1 inhibits Rag GTPases → Rags activate mTORC1. Mutations: focal FCD.', color: '#1a3a6e' },
            { arm: 'APEX — Direct Target', genes: 'MTOR (this dashboard)', desc: 'Catalytic core. Somatic GOF → FCD IIb / HME / MCAP. Direct rapamycin target.', color: COLOR },
          ].map(r => (
            <div key={r.arm} className="col-md-4">
              <div className="card h-100 shadow-sm">
                <div className="card-header small fw-bold text-white" style={{ background: r.color }}>{r.arm}</div>
                <div className="card-body small">
                  <div className="fw-bold mb-1">{r.genes}</div>
                  <div className="text-muted">{r.desc}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
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
      <SectionCard title="Etiology Catalog — 5 Classes" borderColor={COLOR}>
        {etiologies.map(e => (
          <div key={e.category} className="mb-3 border-start ps-3" style={{ borderColor: COLOR, borderLeftWidth: 3 }}>
            <div className="fw-bold small">{e.category} ({e.pct}%)</div>
            <div className="text-muted small">{e.mechanism}</div>
            <div className="mt-1 small"><strong>EEG:</strong> {e.eeg}</div>
            <div className="small"><strong>Onset:</strong> {e.onset_months}</div>
          </div>
        ))}
      </SectionCard>
      <SectionCard title="Patient Cohort (40 synthetic patients)" borderColor={COLOR}>
        <div style={{ overflowX: 'auto' }}>
          <table className="table table-sm table-striped small">
            <thead>
              <tr>
                <th>ID</th><th>Age</th><th>Sex</th><th>Etiology</th>
                <th>MRI</th><th>Medications</th><th>Ev</th><th>Surgery</th><th>SF</th><th>Sz/mo</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.age_y}y</td>
                  <td>{p.sex}</td>
                  <td className="small" style={{ maxWidth: 160, fontSize: '0.75rem' }}>{p.etiology}</td>
                  <td className="small" style={{ maxWidth: 140, fontSize: '0.72rem' }}>{p.mri}</td>
                  <td className="small" style={{ maxWidth: 160, fontSize: '0.75rem' }}>{p.drugs}</td>
                  <td>{p.everolimus ? <span style={{ color: PRECISION_COLOR }}>✓</span> : '–'}</td>
                  <td>{p.surgery ? <span style={{ color: WARN }}>✓</span> : '–'}</td>
                  <td>{p.seizure_free ? <span style={{ color: SUCCESS }}>✓</span> : '–'}</td>
                  <td>{p.seizure_free ? 0 : p.seizure_freq_monthly}</td>
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
      <SectionCard title="Seizure Type Catalog — 5 Types" borderColor={COLOR}>
        {seizures.map(s => (
          <div key={s.type} className="mb-3 border-start ps-3" style={{ borderColor: COLOR, borderLeftWidth: 3 }}>
            <div className="d-flex justify-content-between align-items-center">
              <div className="fw-bold small">{s.type}</div>
              <span className="badge" style={{ background: COLOR }}>{s.pct}%</span>
            </div>
            <Bar label="" value={s.pct} color={COLOR} />
            <div className="text-muted small"><strong>EEG:</strong> {s.eeg}</div>
            <div className="text-muted small"><strong>Semiology:</strong> {s.semiology}</div>
            <div className="small text-info"><strong>Clinical Tip:</strong> {s.clinical_tip}</div>
          </div>
        ))}
      </SectionCard>
      <SectionCard title="Trigger Catalog — 8 Triggers" borderColor={WARN}>
        {triggers.map(t => (
          <div key={t.trigger} className="mb-3 border-start ps-3" style={{ borderColor: WARN, borderLeftWidth: 3 }}>
            <div className="d-flex justify-content-between align-items-center">
              <div className="fw-bold small">{t.trigger}</div>
              <span className="badge" style={{ background: WARN }}>{t.pct}%</span>
            </div>
            <Bar label="" value={t.pct} color={WARN} />
            <div className="text-muted small">{t.mechanism}</div>
            <div className="small" style={{ color: SUCCESS }}><strong>Management:</strong> {t.management}</div>
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
      <SectionCard title="Treatment Catalog — 8 Options" borderColor={SUCCESS}>
        {treatments.map(t => (
          <div key={t.drug} className="mb-3 border-start ps-3" style={{ borderColor: SUCCESS, borderLeftWidth: 3 }}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <div className="fw-bold small">{t.drug}</div>
              <span className="badge bg-secondary">{t.level}</span>
            </div>
            <div className="text-muted small"><strong>Mechanism:</strong> {t.mechanism}</div>
            <div className="text-muted small"><strong>Dose:</strong> {t.dose}</div>
            <div className="text-muted small"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="text-muted small"><strong>Monitoring:</strong> {t.monitoring}</div>
            <div className="small" style={{ color: PRECISION_COLOR }}><strong>MTOR-Specific:</strong> {t.mtor_specific}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications — 6 Absolute/High Risk" borderColor={DANGER}>
        {cis.map(c => (
          <div key={c.drug} className="mb-2 border-start ps-3" style={{ borderColor: DANGER, borderLeftWidth: 3 }}>
            <div className="d-flex justify-content-between align-items-center">
              <span className="fw-bold small">{c.drug}</span>
              <span className="badge" style={{ background: c.risk === 'ABSOLUTE' ? DANGER : WARN }}>{c.risk}</span>
            </div>
            <div className="text-muted small">{c.reason}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring — 14 Items" borderColor={WARN}>
        <div className="row g-2">
          {monitoring.map(m => (
            <div key={m.item} className="col-md-6">
              <div className="card h-100 small p-2" style={{ borderLeft: `3px solid ${WARN}` }}>
                <div className="fw-bold">{m.item}</div>
                <div className="text-muted" style={{ fontSize: '0.75rem' }}>{m.frequency}</div>
                <div style={{ fontSize: '0.75rem' }}>{m.rationale}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Lifecycle — 6 Stages" borderColor={COLOR}>
        {lifecycle.map(l => (
          <div key={l.stage} className="mb-2 border-start ps-3" style={{ borderColor: COLOR, borderLeftWidth: 3 }}>
            <div className="fw-bold small">{l.stage} <span className="text-muted fw-normal">({l.age})</span></div>
            <div className="text-muted small">{l.notes}</div>
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
  const standards = data.standards || [];
  const references = data.references || [];
  const trilogy = data.gator1_trilogy || {};

  return (
    <div>
      {/* mTOR pathway trilogy */}
      <SectionCard title="mTORopathy Trilogy — Pathway Context" borderColor={PRECISION_COLOR}>
        <div className="small mb-2 fw-bold">{trilogy.note}</div>
        <div className="mb-2">
          <div className="fw-bold small mb-1">Upstream:</div>
          {(trilogy.upstream || []).map((u, i) => (
            <div key={i} className="text-muted small mb-1 border-start ps-2" style={{ borderColor: '#6a1b9a', borderLeftWidth: 2 }}>{u}</div>
          ))}
        </div>
        <div className="mb-2">
          <span className="badge me-2" style={{ background: COLOR }}>APEX</span>
          <span className="small fw-bold">{trilogy.apex}</span>
        </div>
        <div>
          <div className="fw-bold small mb-1">Downstream targets:</div>
          {(trilogy.downstream || []).map((d, i) => (
            <div key={i} className="text-muted small mb-1 border-start ps-2" style={{ borderColor: PRECISION_COLOR, borderLeftWidth: 2 }}>{d}</div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Key Concepts — 15 Definitions" borderColor={COLOR}>
        {concepts.map(c => (
          <div key={c.term} className="mb-2 border-start ps-3" style={{ borderColor: COLOR, borderLeftWidth: 3 }}>
            <div className="fw-bold small">{c.term}</div>
            <div className="text-muted small">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds — 12" borderColor={WARN}>
        <div className="row g-2">
          {thresholds.map(t => (
            <div key={t.param} className="col-md-6">
              <div className="card small p-2" style={{ borderLeft: `3px solid ${WARN}` }}>
                <div className="fw-bold">{t.param}: <span style={{ color: COLOR }}>{t.value}</span></div>
                <div className="text-muted" style={{ fontSize: '0.75rem' }}>{t.note}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Standards — 12" borderColor="#455a64">
        <div className="row g-2">
          {standards.map(s => (
            <div key={s.name} className="col-md-6">
              <div className="card small p-2" style={{ borderLeft: '3px solid #455a64' }}>
                <div className="fw-bold">{s.name}</div>
                <div className="text-muted" style={{ fontSize: '0.75rem' }}>{s.scope}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="References — 6" borderColor="#546e7a">
        {references.map(r => (
          <div key={r.id} className="mb-2 small text-muted border-start ps-2" style={{ borderColor: '#546e7a', borderLeftWidth: 2 }}>
            <strong>{r.id}:</strong> {r.citation}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

export default function MTORPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState({});

  const load = async (idx) => {
    if (idx === 0 && !overview) {
      setLoading(l => ({ ...l, 0: true }));
      try { const r = await fetch(`${API}/api/mtor/overview`); setOverview(await r.json()); } catch {}
      setLoading(l => ({ ...l, 0: false }));
    }
    if ((idx === 1 || idx === 2 || idx === 3) && !breakdown) {
      setLoading(l => ({ ...l, bd: true }));
      try { const r = await fetch(`${API}/api/mtor/breakdown`); setBreakdown(await r.json()); } catch {}
      setLoading(l => ({ ...l, bd: false }));
    }
    if (idx === 4 && !definitions) {
      setLoading(l => ({ ...l, 4: true }));
      try { const r = await fetch(`${API}/api/mtor/definitions`); setDefinitions(await r.json()); } catch {}
      setLoading(l => ({ ...l, 4: false }));
    }
  };

  useEffect(() => { load(0); }, []);

  const handleTab = (idx) => { setTab(idx); load(idx); };

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          MTOR Epilepsy Dashboard
        </h4>
        <div className="text-muted small">
          Mechanistic Target of Rapamycin · mTOR Kinase · 1p36.22 · GOF Somatic/Germline ·
          FCD IIb / HME / MCAP / Smith-Kingsmore · Everolimus Precision (Direct mTORC1 Target) ·
          mTORopathy Apex — completes TSC1/TSC2 → DEPDC5/NPRL2/NPRL3 → <strong>MTOR</strong> trilogy
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => handleTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
