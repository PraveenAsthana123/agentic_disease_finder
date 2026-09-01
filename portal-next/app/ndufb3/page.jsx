'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#1b5e20';   // deep green — PP-module membrane arm, first nuclear CI mutation
const LIGHT = '#e8f5e9';

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

function Bar({ label, value, color = COLOR }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ variant, text }) {
  const bg     = variant === 'danger'  ? '#ffebee' : variant === 'warning' ? '#fff8e1'
               : variant === 'success' ? '#e8f5e9' : LIGHT;
  const border = variant === 'danger'  ? '#c62828' : variant === 'warning' ? '#f57f17'
               : variant === 'success' ? '#2e7d32' : COLOR;
  return (
    <div className="mb-2 p-2 rounded small" style={{ background: bg, borderLeft: `4px solid ${border}` }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        {title && <h6 className="fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>}
        {children}
      </div>
    </div>
  );
}

// ── Tab: Overview ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ff = data.feature_frequencies_pct || {};
  const bf = data.biochemical_fingerprint || {};
  const pt = data.protein || {};
  const co = data.cohort || {};
  return (
    <>
      <div className="row g-2 mb-3">
        <KPI label="Gene"          value={data.gene}             color={COLOR} />
        <KPI label="Also Known As" value={data.also_known_as || 'B12'} color={COLOR} />
        <KPI label="Size"          value={`${pt.size_kda} kDa`}  color={COLOR} />
        <KPI label="Fe-S Cluster"  value="None (structural)"     color="#455a64" />
        <KPI label="Module"        value="PP-module"             color="#388e3c" />
        <KPI label="Inheritance"   value="AR biallelic"          color={COLOR} />
        <KPI label="Chromosome"    value={data.chromosome}       color={COLOR} />
        <KPI label="OMIM Gene"     value={`*${data.omim_gene}`}   color="#43a047" />
        <KPI label="OMIM Disease"  value={`#${data.omim_disease}`} color="#43a047" />
        <KPI label="Cohort N"      value={co.n}                  color={COLOR} />
        <KPI label="Mean CI Act"   value={`${co.ci_activity_mean_pct}%`} color="#e53935" />
        <KPI label="Seed"          value={co.seed}               color="#78909c" />
      </div>

      <div className="alert mb-4" style={{ background: '#e8f5e9', borderLeft: '5px solid #1b5e20' }}>
        <strong>Historical Significance:</strong> {data.historical_note}
      </div>

      <SectionCard title="Subunit Role — PP-Module (Proximal Pump) Membrane Arm">
        <p className="small mb-2" style={{ whiteSpace: 'pre-line' }}>{data.key_pathway_note}</p>
        <div className="row g-2">
          <div className="col-md-6">
            <div className="rounded p-2 small" style={{ background: '#f1f8e9' }}>
              <strong>Module:</strong> {pt.module}<br />
              <strong>Fe-S cluster:</strong> None (structural membrane arm subunit)<br />
              <strong>Function:</strong> {pt.function}
            </div>
          </div>
          <div className="col-md-6">
            <div className="rounded p-2 small" style={{ background: '#fff8e1' }}>
              <strong>BN-PAGE pattern:</strong> Absent CI (cleaner — PP-module scaffolding loss)<br />
              <em>vs N-module structural (NDUFA2/NDUFS3/NDUFS5): sub-assembly intermediates</em><br />
              <em>vs Fe-S relay block (NDUFS7/NDUFS8): also absent CI but direct relay failure</em>
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Biochemical Fingerprint — Isolated CI Deficiency">
        {Object.entries(bf).map(([k, v]) => (
          <div key={k} className="d-flex gap-2 mb-1 small">
            <span className="fw-semibold" style={{ minWidth: 120 }}>{k.replace(/_/g,' ')}</span>
            <span className={v.startsWith('NORMAL') ? 'text-success' : 'text-danger fw-bold'}>{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Feature Frequencies (40-patient cohort, seed-627)">
        <div className="row g-3">
          <div className="col-md-6">
            <Bar label={`Psychomotor Regression ${ff.psychomotor_regression}%`}  value={ff.psychomotor_regression} />
            <Bar label={`Hypotonia ${ff.hypotonia}%`}                             value={ff.hypotonia} />
            <Bar label={`Lactic Acidosis ${ff.lactic_acidosis}%`}                 value={ff.lactic_acidosis} />
            <Bar label={`Leigh-Syndrome MRI ${ff.leigh_mri}%`}                    value={ff.leigh_mri} />
            <Bar label={`Seizures ${ff.seizures}%`}                               value={ff.seizures} />
            <Bar label={`Ataxia ${ff.ataxia}%`}                                   value={ff.ataxia} />
          </div>
          <div className="col-md-6">
            <Bar label={`Respiratory Compromise ${ff.respiratory_compromise}%`}   value={ff.respiratory_compromise} color="#ef6c00" />
            <Bar label={`Dystonia ${ff.dystonia}%`}                               value={ff.dystonia} color="#7b1fa2" />
            <Bar label={`HCM ${ff.hcm}% (LOW — DDx NDUFV2 80%/SCO2 100%)`}       value={ff.hcm} color="#78909c" />
            <Bar label={`Peripheral Neuropathy ${ff.peripheral_neuropathy}% (DDx NDUFS1 50%)`} value={ff.peripheral_neuropathy} color="#78909c" />
            <Bar label={`Olfactory Bulb Lesions ${ff.olfactory_bulb_lesions}% (DDx NDUFS4 58%)`} value={ff.olfactory_bulb_lesions} color="#78909c" />
            <Bar label={`Leukodystrophy ${ff.leukodystrophy}% (DDx NDUFV1 45%)`} value={ff.leukodystrophy} color="#78909c" />
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Key Differential Diagnosis Points">
        {(data.key_ddx || []).map((d, i) => (
          <Alert
            key={i}
            variant={d.feature.startsWith('NO') ? 'success' : 'warning'}
            text={`${d.feature} — ${d.significance}`}
          />
        ))}
      </SectionCard>

      <SectionCard title="Key References">
        <ul className="small mb-0 ps-3">
          {(data.key_references || []).map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Features ────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const { patients = [], feature_frequencies = {}, outcome_distribution = {}, mutation_distribution = {}, region_distribution = {}, sex_distribution = {}, ci_activity_histogram = {} } = data;
  return (
    <>
      <SectionCard title="CI Activity Distribution (40-patient cohort)">
        <div className="row g-2">
          {(ci_activity_histogram.bins || []).map((bin, i) => (
            <div key={i} className="col-6 col-md-3">
              <div className="text-center p-2 rounded" style={{ background: LIGHT }}>
                <div className="fw-bold" style={{ color: COLOR }}>{ci_activity_histogram.counts?.[i] ?? 0}</div>
                <div className="small text-muted">CI {bin}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Outcome Distribution">
        {Object.entries(outcome_distribution).map(([k, v]) => (
          <div key={k} className="d-flex justify-content-between small border-bottom py-1">
            <span>{k}</span><span className="fw-bold" style={{ color: COLOR }}>{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Mutation Distribution">
        {Object.entries(mutation_distribution).sort((a,b) => b[1]-a[1]).map(([k, v]) => (
          <div key={k} className="d-flex justify-content-between small border-bottom py-1">
            <span className="font-monospace">{k}</span>
            <span className="fw-bold" style={{ color: COLOR }}>{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Feature Frequencies">
        {Object.entries(feature_frequencies).map(([k, v]) => (
          <Bar key={k} label={`${k.replace(/_/g,' ')} (${v.count}/${data.n})`} value={v.pct} />
        ))}
      </SectionCard>

      <SectionCard title="Patient Roster (first 20)">
        <div className="table-responsive">
          <table className="table table-sm table-striped small">
            <thead>
              <tr>
                <th>#</th><th>Sex</th><th>Onset (mo)</th><th>Region</th>
                <th>CI%</th><th>Leigh MRI</th><th>LA</th><th>Seizures</th><th>HCM</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {patients.slice(0, 20).map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.age_onset_months}</td>
                  <td>{p.region}</td>
                  <td style={{ color: '#e53935', fontWeight: 600 }}>{p.ci_activity_pct}%</td>
                  <td>{p.leigh_mri ? '✓' : '—'}</td>
                  <td>{p.lactic_acidosis ? '✓' : '—'}</td>
                  <td>{p.seizures ? '✓' : '—'}</td>
                  <td>{p.hcm ? '✓' : '—'}</td>
                  <td className="text-muted">{p.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments & DDx ──────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <p className="text-muted">Loading overview for treatments…</p>;
  const abs  = data.absolute_contraindications || [];
  const ci   = data.contraindicated || [];
  const pref = data.preferred_treatments || [];
  return (
    <>
      <SectionCard title="ABSOLUTE Contraindications (never use)" borderColor="#c62828">
        {abs.map((t, i) => <Alert key={i} variant="danger" text={t} />)}
      </SectionCard>

      <SectionCard title="Contraindicated (avoid)" borderColor="#ef6c00">
        {ci.map((t, i) => <Alert key={i} variant="warning" text={t} />)}
      </SectionCard>

      <SectionCard title="Preferred / Recommended Treatments" borderColor="#2e7d32">
        {pref.map((t, i) => <Alert key={i} variant="success" text={t} />)}
      </SectionCard>

      <SectionCard title="Prescribing Safety Pocket Summary">
        <div className="rounded p-3 small font-monospace" style={{ background: '#f8f9fa', whiteSpace: 'pre-wrap' }}>
{`ABSOLUTE CI  : Metformin · Valproate · Linezolid · Chloramphenicol
CONTRAINDICATED: Ketogenic diet
AVOID        : Propofol (PRIS + CIV block)
HIGH CAUTION : Phenobarbital (secondary CI inhibitor)
PREFERRED AED: LEV (levetiracetam) — renal, no mito toxicity
ANAESTHESIA  : Sevoflurane (NOT propofol)
GLUCOSE      : IV dextrose GIR 6–8 mg/kg/min — NEVER fast
COFACTORS    : Riboflavin B2 · CoQ10 · Thiamine B1* · Biotin* · Succinate · Carnitine
               (* = MANDATORY empiric before genetics: rules out SLC19A3/BTD)
HISTORICAL   : FIRST nuclear CI mutation (Andreu 1999) — PP-module B12 subunit`}
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ───────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const sections = [
    { key: 'pharmacology',       title: 'Pharmacology' },
    { key: 'gene_concepts',      title: 'Gene / Protein Concepts' },
    { key: 'disease_concepts',   title: 'Disease Concepts' },
    { key: 'prescribing_safety', title: 'Prescribing Safety' },
  ];
  return (
    <>
      {sections.map(({ key, title }) => (
        <SectionCard key={key} title={title}>
          {(data[key] || []).map((item, i) => (
            <div key={i} className="mb-3 border-bottom pb-2">
              <div className="fw-semibold small" style={{ color: COLOR }}>{item.term}</div>
              <div className="small text-muted mt-1" style={{ whiteSpace: 'pre-line' }}>{item.detail}</div>
            </div>
          ))}
        </SectionCard>
      ))}
    </>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────
export default function NDUFB3Page() {
  const [tab,  setTab]  = useState(0);
  const [ov,   setOv]   = useState(null);
  const [bk,   setBk]   = useState(null);
  const [def,  setDef]  = useState(null);
  const [err,  setErr]  = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ndufb3/overview`).then(r => r.json()),
      fetch(`${API}/api/ndufb3/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ndufb3/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDef(d); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="mb-4">
        <h2 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 NDUFB3 — Leigh Syndrome Isolated Complex I Deficiency
        </h2>
        <p className="text-muted small mb-1">
          PP-Module B12 Subunit · Proximal Pump Membrane Arm · No Fe-S Cluster · AR Biallelic · 2q31.3
        </p>
        <p className="text-muted small mb-1">
          OMIM Gene *603839 · Disease Leigh Syndrome #256000 · CI-Leigh series · 40-patient cohort seed-627
        </p>
        <p className="small fw-semibold" style={{ color: '#1b5e20' }}>
          ★ First nuclear-encoded CI subunit mutation identified in human disease (Andreu 1999 NatGenet)
        </p>
        {err && <div className="alert alert-danger small">{err}</div>}
      </div>

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab    data={ov}  />}
      {tab === 1 && <PatientsTab    data={bk}  />}
      {tab === 2 && <TreatmentsTab  data={ov}  />}
      {tab === 3 && <DefinitionsTab data={def} />}
    </div>
  );
}
