'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1b5e20';   // deep forest green — AGA / Finnish / progressive but born-normal
const ACCENT2 = '#b71c1c';   // dark red — HIGH RISK / ABSOLUTE CI / typical AP trap
const ACCENT3 = '#e65100';   // deep orange — RELATIVE CI / CAUTION / VGB behavioral CI
const ACCENT4 = '#4a148c';   // deep purple — gene therapy AAV-AGA / investigational / no ERT
const ACCENT5 = '#006064';   // teal — behavioral management / quetiapine / POLG1 mandatory
const ACCENT6 = '#1565c0';   // blue — LEV first-line / SAFE / EEG-guided decisions

const ETIOLOGY_COLORS = {
  'Finnish': '#1b5e20',
  'Non-Finnish': '#1565c0',
  'Null': '#b71c1c',
  'Severe': '#b71c1c',
  'Attenuated': '#2e7d32',
  'Compound': '#e65100',
  'Rare': '#555',
};

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-6" style={{ color }}>{value}</div>
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
        <span>{label}</span><span className="fw-bold">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

function Badge({ text, color = ACCENT }) {
  return (
    <span className="badge me-1 mb-1" style={{ background: color, fontSize: '0.72rem' }}>{text}</span>
  );
}

function SectionCard({ title, color = ACCENT, children }) {
  return (
    <div className="card shadow-sm mb-4">
      <div className="card-header text-white fw-bold" style={{ background: color }}>{title}</div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function CICard({ drug, severity, reason }) {
  const riskColor = severity?.includes('ABSOLUTE') ? ACCENT2 :
                    severity === 'HIGH RISK' ? '#c62828' :
                    severity?.includes('RELATIVE') ? ACCENT3 :
                    severity?.includes('AVOID') ? '#c62828' :
                    severity?.includes('MANDATORY') ? ACCENT2 : '#666';
  return (
    <div className="card mb-3 border-0 shadow-sm">
      <div className="card-body py-2">
        <div className="d-flex align-items-start gap-2 flex-wrap">
          <span className="badge fs-6" style={{ background: riskColor }}>{severity}</span>
          <strong>{drug}</strong>
        </div>
        <div className="small mt-1 text-muted">{reason}</div>
      </div>
    </div>
  );
}

function TreatmentCard({ drug, level, indication, ci }) {
  const levelColor = level?.startsWith('Level A') ? ACCENT6 :
                     level?.startsWith('Level B') ? ACCENT :
                     ACCENT3;
  return (
    <div className="card mb-2 border-0 shadow-sm">
      <div className="card-body py-2">
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <Badge text={level || '—'} color={levelColor} />
          <strong>{drug}</strong>
        </div>
        <div className="small mt-1 text-muted">{indication}</div>
        {ci && <div className="small mt-1 text-danger"><strong>CI:</strong> {ci}</div>}
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const kpis = data.kpis || {};
  const kpiList = Object.entries(kpis).map(([k, v]) => ({ label: k.replace(/_/g, ' '), value: v }));
  return (
    <>
      <SectionCard title="⚠️ AGA / AGU Unique Features — Aspartylglucosaminuria" color={ACCENT2}>
        <ul className="mb-0 small">
          <li><strong>BORN NORMAL → PROGRESSIVE</strong> — infants appear near-normal (0-5yr); intellectual regression starts age 5-10yr; behavioral problems PRECEDE seizures; seizures emerge age 10-30yr — UNIQUE developmental trajectory among LSDs</li>
          <li><strong>FINNISH FOUNDER p.Cys163Ser</strong> — ~98% Finnish alleles; 1:18,000 Finland (most common LSD in Finland); autocatalytic processing blocked → zero enzyme activity</li>
          <li><strong>BEHAVIORAL PHENOTYPE DOMINANT</strong> — aggression, hyperactivity, self-injury (SIB) — psychiatric misdiagnosis (ADHD/autism/conduct disorder) delays correct diagnosis 5-10yr; behavioral management = seizure prevention</li>
          <li><strong>TYPICAL ANTIPSYCHOTICS HIGH RISK</strong> — EPS sensitivity (basal ganglia oligosaccharide storage); use ATYPICAL AP (quetiapine preferred) for behavioral features</li>
          <li><strong>EEG-GUIDED CBZ/OXC DECISION</strong> — CBZ/OXC ACCEPTABLE if EEG confirms NO myoclonic component; RELATIVE CI if polyspike-wave/myoclonus present (AGA-specific EEG rule)</li>
          <li><strong>URINE ASPARTYLGLUCOSAMINE (NOT GAG)</strong> — standard MPS GAG screen NORMAL; order urine oligosaccharide screen (GlcNAc-Asn); POLG1 mandatory before VPA</li>
        </ul>
      </SectionCard>

      <SectionCard title="🧬 Disease Mechanism" color={ACCENT4}>
        <p className="small mb-0">{data.disease_mechanism}</p>
      </SectionCard>

      <SectionCard title="📊 Key Clinical Metrics" color={ACCENT}>
        <div className="row">
          <div className="col-md-6">
            <PctBar label="Epilepsy prevalence (60-70%)" pct={65} color={ACCENT} />
            <PctBar label="Drug-resistant epilepsy DRE (25-40%)" pct={33} color={ACCENT2} />
            <PctBar label="Behavioral phenotype (aggression/SIB)" pct={95} color={ACCENT2} />
          </div>
          <div className="col-md-6">
            <PctBar label="Finnish founder p.Cys163Ser alleles (~98%)" pct={98} color={ACCENT} />
            <PctBar label="Psychiatric misdiagnosis delay (5-10yr)" pct={75} color={ACCENT3} />
            <PctBar label="Macroorchidism (adult males, 50-70%)" pct={60} color={ACCENT5} />
          </div>
        </div>
      </SectionCard>

      <SectionCard title="🇫🇮 Finnish Population Data" color={ACCENT}>
        <div className="row">
          <div className="col-md-6">
            <div className="small mb-2"><strong>Prevalence:</strong> 1:18,000 births (Finland) — most common LSD in Finland</div>
            <div className="small mb-2"><strong>Founder allele:</strong> p.Cys163Ser (c.488G>C) ~98% Finnish alleles</div>
            <div className="small mb-2"><strong>Non-Finnish:</strong> Scattered worldwide; often compound-het; shorter diagnosis delay in era of WGS</div>
          </div>
          <div className="col-md-6">
            <div className="small mb-2"><strong>BMT experience:</strong> Largest AGU BMT cohort in Finland (1990s-2000s); somatic &gt; CNS benefit</div>
            <div className="small mb-2"><strong>Gene therapy:</strong> AAV-AGA (UX143, Ultragenyx) — Phase I intrathecal 2025</div>
            <div className="small mb-2"><strong>Survival:</strong> 4th-5th decade typical with supportive care</div>
          </div>
        </div>
      </SectionCard>

      <div className="row mb-4">
        {kpiList.slice(0, 12).map((k, i) => <KPI key={i} label={k.label} value={k.value} color={ACCENT} />)}
      </div>
    </>
  );
}

function EtiologyTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const etiologies = data.etiology_breakdown || [];
  const etiologiesFull = data.etiologies || [];
  const source = etiologiesFull.length > 0 ? etiologiesFull : etiologies;
  return (
    <>
      <SectionCard title="🧬 Variant Classes / Etiology Spectrum" color={ACCENT}>
        {source.map((e, i) => {
          const key = Object.keys(ETIOLOGY_COLORS).find(k => e.name?.includes(k)) || 'Rare';
          const color = ETIOLOGY_COLORS[key] || ACCENT;
          return (
            <div key={i} className="mb-4 pb-3 border-bottom">
              <div className="d-flex align-items-center gap-2 mb-1 flex-wrap">
                <span className="badge" style={{ background: color }}>{e.pct}% (n={e.n})</span>
                <strong>{e.name}</strong>
              </div>
              {e.seizure_risk && <div className="small text-muted mb-1"><strong>Seizure risk:</strong> {e.seizure_risk}</div>}
              {e.eeg && <div className="small text-muted mb-1"><strong>EEG:</strong> {e.eeg}</div>}
              {e.variant_detail && <div className="small text-muted"><strong>Variant detail:</strong> {e.variant_detail}</div>}
            </div>
          );
        })}
      </SectionCard>

      <SectionCard title="👥 Patient Sample (N=40)" color={ACCENT6}>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead>
              <tr>
                <th>ID</th><th>Etiology</th><th>Age Onset</th><th>Seizure Type</th>
                <th>Tx 1</th><th>CI Avoided</th><th>Prior Psych Dx</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).slice(0, 15).map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td style={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.etiology}</td>
                  <td>{p.age_onset_yr}yr</td>
                  <td>{p.seizure_type}</td>
                  <td>{p.treatment_1}</td>
                  <td className="text-danger">{p.ci_avoided}</td>
                  <td className="text-warning">{p.psychiatric_dx_before_agu}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="text-muted small">Showing 15 of 40 patients</div>
        </div>
      </SectionCard>
    </>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const seizures = data.seizure_summary || [];
  const triggers = data.trigger_summary || [];
  return (
    <>
      <SectionCard title="⚡ Seizure Types" color={ACCENT}>
        {seizures.map((s, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="d-flex gap-2 align-items-center mb-1 flex-wrap">
              <Badge text={`${s.pct}%`} color={ACCENT} />
              <strong>{s.type}</strong>
            </div>
            <PctBar label="" pct={s.pct} color={ACCENT} />
            <div className="small text-muted">{s.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔥 Seizure Triggers" color={ACCENT3}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <div className="d-flex gap-2 align-items-center mb-1 flex-wrap">
              <Badge text={`${t.pct}%`} color={ACCENT3} />
              <strong>{t.trigger}</strong>
            </div>
            <PctBar label="" pct={t.pct} color={ACCENT3} />
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const treatments = data.treatment_summary || [];
  const cis = data.contraindication_summary || [];
  return (
    <>
      <SectionCard title="💊 Treatments (Evidence Level)" color={ACCENT6}>
        {treatments.map((t, i) => (
          <TreatmentCard key={i} {...t} />
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications / Special Risks" color={ACCENT2}>
        {cis.map((c, i) => (
          <CICard key={i} {...c} />
        ))}
      </SectionCard>
    </>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const glossary = data.glossary || [];
  const algorithm = data.diagnostic_algorithm || [];
  const pharm = data.pharmacological_distinctions || [];
  const diff = data.differential_diagnosis || [];
  return (
    <>
      <SectionCard title="📖 Glossary (20 Terms)" color={ACCENT}>
        {glossary.map((g, i) => (
          <div key={i} className="mb-3 pb-2 border-bottom">
            <strong className="small" style={{ color: ACCENT }}>{g.term}</strong>
            <p className="small text-muted mb-0 mt-1">{g.definition}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔬 12-Step Diagnostic Algorithm" color={ACCENT5}>
        <ol className="small mb-0">
          {algorithm.map((step, i) => (
            <li key={i} className="mb-2">{step}</li>
          ))}
        </ol>
      </SectionCard>

      <SectionCard title="⚗️ Pharmacological Distinctions" color={ACCENT6}>
        <ul className="small mb-0">
          {pharm.map((p, i) => <li key={i} className="mb-2">{p}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="🔍 Differential Diagnosis" color={ACCENT3}>
        <ul className="small mb-0">
          {diff.map((d, i) => <li key={i} className="mb-2">{d}</li>)}
        </ul>
      </SectionCard>
    </>
  );
}

export default function AGAPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/aga/overview`).then(r => r.json()),
      fetch(`${API}/api/aga/breakdown`).then(r => r.json()),
      fetch(`${API}/api/aga/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
    }).catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-4">
      <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
        AGA Epilepsy Dashboard — Aspartylglucosaminuria (AGU)
      </h2>
      <p className="text-muted small mb-1">
        <strong>Aspartylglucosaminuria (AGU)</strong> · Aspartylglucosaminidase (AGA) Deficiency ·
        Oligosaccharidosis (NOT MPS) · Autosomal Recessive · 4q34.3 ·
        BORN NORMAL → PROGRESSIVE (regression age 5-10yr — unique LSD trajectory) ·
        Finnish founder p.Cys163Ser (~98% Finnish alleles / 1:18,000) ·
        BEHAVIORAL PHENOTYPE DOMINANT (aggression/SIB — psychiatric misdiagnosis 5-10yr delay) ·
        NO ERT (2026) · AAV-AGA Phase I (UX143) · EEG-guided CBZ/OXC decision ·
        Typical AP HIGH RISK (behavioral trap) · POLG1 mandatory before VPA
      </p>
      <div>
        <Badge text="AR — 4q34.3" color={ACCENT} />
        <Badge text="Oligosaccharidosis (NOT MPS)" color={ACCENT4} />
        <Badge text="Born Normal → Progressive" color={ACCENT3} />
        <Badge text="Finnish Founder p.Cys163Ser" color={ACCENT} />
        <Badge text="Behavioral Phenotype Dominant" color={ACCENT2} />
        <Badge text="Typical AP HIGH RISK" color={ACCENT2} />
        <Badge text="CBZ/OXC EEG-guided (AGA-specific)" color={ACCENT6} />
        <Badge text="POLG1 mandatory before VPA" color={ACCENT2} />
        <Badge text="No ERT (2026)" color={ACCENT4} />
        <Badge text="AAV-AGA Phase I (UX143)" color={ACCENT4} />
        <Badge text="Urine GlcNAc-Asn (NOT GAG screen)" color={ACCENT5} />
        <Badge text="Epilepsy 60-70%" color={ACCENT} />
      </div>

      {err && <div className="alert alert-danger small mt-2">API error: {err}</div>}

      <ul className="nav nav-tabs mb-4 mt-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <EtiologyTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
