'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — GNPTAB / Mucolipidosis II-III
const ACCENT2 = '#b71c1c';   // dark red — ABSOLUTE CI / cardiac / gingival hyperplasia
const ACCENT3 = '#e65100';   // deep orange — RELATIVE CI / CAUTION
const ACCENT4 = '#4a148c';   // deep purple — no ERT / inverse enzyme pattern
const ACCENT5 = '#006064';   // teal — ML-IIIA carpal tunnel / attenuated
const ACCENT6 = '#1565c0';   // blue — LEV / ACTH / safe

const ETIOLOGY_COLORS = {
  'Null': '#b71c1c',
  'Missense': '#1565c0',
  'Severe': '#b71c1c',
  'Attenuated': '#2e7d32',
  'ML-II': '#b71c1c',
  'ML-IIIA': '#006064',
  'ML-IIIB': '#2e7d32',
  'Splice': '#e65100',
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
                    severity?.includes('CAUTION') ? ACCENT3 :
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
      <SectionCard title="⚠️ GNPTAB / Mucolipidosis II-III Unique Features" color={ACCENT2}>
        <ul className="mb-0 small">
          <li><strong>INVERSE PLASMA/LEUKOCYTE ENZYME PATTERN — PATHOGNOMONIC</strong> — plasma lysosomal enzymes 10-50x ELEVATED; leukocyte enzymes LOW (opposite of all other LSDs); due to M6P targeting failure — GlcNAc-1-phosphotransferase deficiency causes enzymes to be secreted into plasma instead of routed to lysosomes; confirms ML-II/III without biopsy</li>
          <li><strong>GINGIVAL HYPERPLASIA — PATHOGNOMONIC for ML-II</strong> — thick fibrotic hyperplastic gums present from birth; NOT seen in MPS-I Hurler, GALNS (MPS-IVA), or ARSB (MPS-VI); distinguishes ML-II from all other lysosomal storage disorders; complicates intubation and anesthesia</li>
          <li><strong>NO CORNEAL CLOUDING</strong> — distinguishes GNPTAB from MPS-I (Hurler), MPS-VI (Maroteaux-Lamy), MPS-VII (Sly); corneal clouding absent even in severe ML-II; key negative finding in differential diagnosis</li>
          <li><strong>ACTH Level A for Infantile Spasms</strong> — VGB ABSOLUTE CI in ML-II due to cardiomyopathy (vigabatrin retinal toxicity secondary concern; cardiac CI primary; ACTH/prednisolone first-line for IS in ML-II)</li>
          <li><strong>PHT/Fosphenytoin ABSOLUTE CI</strong> — cardiac contraindication (QTc/PR prolongation in ML-II cardiomegaly); IV fosphenytoin NEVER used in ML-II status epilepticus; IV LEV replaces as IV rescue AED</li>
          <li><strong>ANESTHESIA EXTREME HAZARD</strong> — difficult airway (gingival hyperplasia + restricted jaw), cardiomyopathy (reduced cardiac reserve), joint contractures (positioning risk); multi-specialist perioperative planning mandatory; elective procedures deferred when possible</li>
        </ul>
      </SectionCard>

      <SectionCard title="🧬 Disease Mechanism" color={ACCENT4}>
        <p className="small mb-0">{data.disease_mechanism}</p>
      </SectionCard>

      <SectionCard title="📊 Key Clinical Metrics" color={ACCENT}>
        <div className="row">
          <div className="col-md-6">
            <PctBar label="Epilepsy prevalence ML-II (40-65%)" pct={52} color={ACCENT} />
            <PctBar label="Infantile spasms in ML-II epilepsy" pct={55} color={ACCENT2} />
            <PctBar label="Cardiomegaly/cardiomyopathy ML-II (80%)" pct={80} color={ACCENT2} />
          </div>
          <div className="col-md-6">
            <PctBar label="Drug-resistant epilepsy ML-II (30-40%)" pct={35} color={ACCENT2} />
            <PctBar label="Epilepsy ML-IIIA (15-25%)" pct={20} color={ACCENT5} />
            <PctBar label="Carpal tunnel ML-IIIA by age 10 (85%)" pct={85} color={ACCENT5} />
          </div>
        </div>
      </SectionCard>

      <SectionCard title="🔬 GNPTAB vs Other LSDs" color={ACCENT5}>
        <div className="row">
          <div className="col-md-6">
            <div className="small mb-2"><strong>vs MPS-I Hurler:</strong> GNPTAB no corneal clouding (MPS-I universal); gingival hyperplasia ML-II pathognomonic; plasma enzymes ELEVATED in GNPTAB (MPS-I enzymes low in leukocytes, plasma normal to low); MPS-I has HSCT (GNPTAB none)</div>
            <div className="small mb-2"><strong>vs MPS-II Hunter:</strong> X-linked recessive (GNPTAB autosomal recessive); both have no corneal clouding; HS+DS urine elevated in Hunter (GNPTAB GAG normal or mild); idursulfase ERT available in Hunter (GNPTAB no ERT approved 2026)</div>
          </div>
          <div className="col-md-6">
            <div className="small mb-2"><strong>vs MCOLN1 (ML-IV):</strong> corneal clouding PRESENT in ML-IV (ABSENT in GNPTAB ML-II); plasma enzymes NORMAL in ML-IV (ELEVATED 10-50x in GNPTAB — pathognomonic distinction); ML-IV has achlorhydria; GNPTAB has cardiomyopathy</div>
            <div className="small mb-2"><strong>Enzyme assay:</strong> GlcNAc-1-phosphotransferase activity in leukocytes/fibroblasts; confirm with plasma lysosomal enzyme panel (10-50x elevation); urine GAG usually normal or mild elevation (NOT diagnostic screen)</div>
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
          const key = Object.keys(ETIOLOGY_COLORS).find(k => e.name?.includes(k)) || 'Missense';
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
                <th>ID</th><th>Etiology</th><th>ML Type</th><th>Age Onset</th><th>Seizure Type</th>
                <th>Gingival Hyperplasia</th><th>Cardiomyopathy</th><th>Treatment 1</th><th>CI Avoided</th><th>Plasma Enzymes</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).slice(0, 15).map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.etiology}</td>
                  <td>{p.ml_type}</td>
                  <td>{p.age_onset_yr}yr</td>
                  <td>{p.seizure_type}</td>
                  <td className={p.gingival_hyperplasia ? 'text-danger fw-bold' : 'text-muted'}>
                    {p.gingival_hyperplasia ? 'Yes' : 'No'}
                  </td>
                  <td className={p.cardiomyopathy ? 'text-danger fw-bold' : 'text-muted'}>
                    {p.cardiomyopathy ? 'Yes' : 'No'}
                  </td>
                  <td>{p.treatment_1}</td>
                  <td className="text-danger">{p.ci_avoided}</td>
                  <td className="text-warning">{p.plasma_enzymes}</td>
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
      <SectionCard title="📖 Glossary (15 Terms)" color={ACCENT}>
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

export default function GNPTABPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/gnptab/overview`).then(r => r.json()),
      fetch(`${API}/api/gnptab/breakdown`).then(r => r.json()),
      fetch(`${API}/api/gnptab/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
    }).catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-4">
      <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
        GNPTAB Epilepsy Dashboard — Mucolipidosis II / III
      </h2>
      <p className="text-muted small mb-1">
        <strong>Mucolipidosis II (I-Cell Disease)</strong> · Mucolipidosis IIIA/B (Pseudo-Hurler Polydystrophy) ·
        GlcNAc-1-Phosphotransferase Alpha/Beta Deficiency · M6P Targeting Failure · Autosomal Recessive · 12q23.2 ·
        INVERSE plasma/leukocyte enzyme pattern PATHOGNOMONIC · Gingival Hyperplasia ML-II ·
        No Corneal Clouding · PHT ABSOLUTE CI (Cardiac) · ACTH Level A IS · Anesthesia EXTREME HAZARD · No ERT 2026
      </p>
      <div>
        <Badge text="AR — 12q23.2" color={ACCENT} />
        <Badge text="Mucolipidosis II / III (NOT MPS)" color={ACCENT4} />
        <Badge text="Inverse Plasma/Leukocyte Enzyme PATHOGNOMONIC" color={ACCENT2} />
        <Badge text="Gingival Hyperplasia Pathognomonic (ML-II)" color={ACCENT2} />
        <Badge text="NO Corneal Clouding" color={ACCENT5} />
        <Badge text="PHT/Fosphenytoin ABSOLUTE CI (Cardiac)" color={ACCENT2} />
        <Badge text="VGB ABSOLUTE CI in ML-II IS (Cardiomyopathy)" color={ACCENT2} />
        <Badge text="ACTH Level A for Infantile Spasms" color={ACCENT6} />
        <Badge text="Anesthesia EXTREME HAZARD" color={ACCENT2} />
        <Badge text="POLG1 mandatory before VPA" color={ACCENT2} />
        <Badge text="No ERT Approved (2026)" color={ACCENT4} />
        <Badge text="Epilepsy 40-65% ML-II / 15-25% ML-IIIA" color={ACCENT} />
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
