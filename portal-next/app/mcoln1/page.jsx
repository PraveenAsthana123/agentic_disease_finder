'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — ML-IV primary
const ACCENT2 = '#880e4f';   // deep magenta — VGB RELATIVE CI / retinal degeneration
const ACCENT3 = '#e65100';   // deep orange — elevated gastrin / achlorhydria / caution
const ACCENT4 = '#4a148c';   // deep purple — no ERT / mucolipin channel biology
const ACCENT5 = '#006064';   // teal — Ashkenazi founder / iron anemia / safe
const ACCENT6 = '#1565c0';   // blue — LEV / LTG / safe AEDs / supportive

const ETIOLOGY_COLORS = {
  'Ashkenazi': '#1a237e',
  'Type I':    '#880e4f',
  'Type II':   '#1565c0',
  'Compound':  '#e65100',
  'Missense':  '#2e7d32',
  'Private':   '#4a148c',
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
  const riskColor = severity?.includes('ABSOLUTE') ? '#b71c1c' :
                    severity === 'HIGH RISK'        ? '#c62828' :
                    severity?.includes('RELATIVE')  ? ACCENT2  :
                    severity?.includes('CAUTION')   ? ACCENT3  :
                    severity?.includes('MANDATORY') ? '#b71c1c' : '#666';
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
                     level?.startsWith('Level B') ? ACCENT  :
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
      <SectionCard title="⚠️ MCOLN1 / ML-IV Unique Features" color={ACCENT2}>
        <ul className="mb-0 small">
          <li><strong>CORNEAL CLOUDING — PRESENT (bilateral, from infancy)</strong> — directly OPPOSITE to GNPTAB ML-II/III where corneal clouding is ABSENT; first sign often in year 1; progressive; slit-lamp examination mandatory in all LSD workups with psychomotor retardation; combined with retinal degeneration → compounding visual disability</li>
          <li><strong>ELEVATED SERUM GASTRIN + ACHLORHYDRIA — PATHOGNOMONIC</strong> — 100% of ML-IV patients; MCOLN1 required for V-ATPase H+ secretion in gastric parietal cells → achlorhydria → gastrin hyperproduction; fasting gastrin markedly elevated (>10x ULN); SPECIFIC to ML-IV among all LSDs; serum gastrin is the FIRST diagnostic blood test to order</li>
          <li><strong>NORMAL PLASMA LYSOSOMAL ENZYME PANEL</strong> — unlike GNPTAB ML-II where plasma enzymes 10-50x elevated; MCOLN1 deficiency does NOT affect M6P targeting; lysosomal enzymes trafficked normally; NORMAL panel + corneal clouding + elevated gastrin = ML-IV</li>
          <li><strong>VGB RELATIVE CI — Retinal Degeneration</strong> — vigabatrin causes irreversible visual field constriction; ML-IV has pre-existing progressive retinal degeneration (ERG abnormal 100%); VGB compounds retinal insult AND visual field monitoring impossible in severe ID; unlike GNPTAB ML-II (VGB ABSOLUTE CI for cardiac), this is RELATIVE in ML-IV</li>
          <li><strong>PHT NOT CONTRAINDICATED in ML-IV</strong> — no cardiac involvement in ML-IV; contrast with GNPTAB ML-II (PHT ABSOLUTE CI for cardiomyopathy); IV fosphenytoin IS an option in ML-IV SE; do NOT apply GNPTAB ML-II CI rules to ML-IV patients</li>
          <li><strong>NO BONE DISEASE / NO GAG ACCUMULATION</strong> — unlike MPS diseases; no dysostosis multiplex, no atlantoaxial instability, no carpal tunnel; urine GAG NORMAL (critical: negative GAG screen does NOT exclude LSD — must still consider ML-IV); no anesthesia concern for AAI positioning</li>
        </ul>
      </SectionCard>

      <SectionCard title="🧬 Disease Mechanism" color={ACCENT4}>
        <p className="small mb-0">{data.disease_mechanism}</p>
      </SectionCard>

      <SectionCard title="📊 Key Clinical Metrics" color={ACCENT}>
        <div className="row">
          <div className="col-md-6">
            <PctBar label="Corneal clouding (100% — bilateral from infancy)" pct={100} color={ACCENT2} />
            <PctBar label="Retinal degeneration / ERG abnormal (100%)" pct={100} color={ACCENT2} />
            <PctBar label="Achlorhydria + elevated gastrin (100% — PATHOGNOMONIC)" pct={100} color={ACCENT3} />
          </div>
          <div className="col-md-6">
            <PctBar label="Epilepsy prevalence (15-25%)" pct={20} color={ACCENT} />
            <PctBar label="Iron deficiency anemia (secondary to achlorhydria, ~80%)" pct={80} color={ACCENT5} />
            <PctBar label="Drug-resistant epilepsy (5-15%)" pct={10} color={ACCENT2} />
          </div>
        </div>
      </SectionCard>

      <SectionCard title="🔬 ML-IV vs Other Mucolipidoses / LSDs with Corneal Clouding" color={ACCENT5}>
        <div className="row">
          <div className="col-md-6">
            <div className="small mb-2"><strong>vs GNPTAB (ML-II/III):</strong> ML-IV corneal clouding PRESENT (GNPTAB absent); ML-IV plasma enzymes NORMAL (GNPTAB 10-50x elevated — PATHOGNOMONIC inverse pattern); ML-IV elevated gastrin (GNPTAB: normal gastrin, gingival hyperplasia pathognomonic); ML-IV no cardiac (GNPTAB cardiomyopathy 80%)</div>
            <div className="small mb-2"><strong>vs MPS-I Hurler (IDUA):</strong> MPS-I urine HS+DS elevated (ML-IV urine GAG NORMAL); IDUA enzyme low (ML-IV plasma/leukocyte enzymes all normal); MPS-I HSCT Level A before 2.5yr + laronidase ERT (ML-IV: no ERT, no HSCT)</div>
          </div>
          <div className="col-md-6">
            <div className="small mb-2"><strong>vs MPS-VII Sly (GUSB):</strong> MPS-VII triple GAG (HS+DS+CS) PATHOGNOMONIC (ML-IV urine GAG NORMAL); MPS-VII vestronidase alfa ERT; NIHF in MPS-VII (ML-IV no hydrops)</div>
            <div className="small mb-2"><strong>Gastrin + ERG screen:</strong> Elevated fasting serum gastrin + ERG abnormality = ML-IV fingerprint; no other LSD causes this combination; order before MCOLN1 sequencing when clinical picture suggests ML-IV</div>
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
  const etiologiesFull = data.etiologies || [];
  const source = etiologiesFull.length > 0 ? etiologiesFull : (data.etiology_breakdown || []);
  return (
    <>
      <SectionCard title="🧬 Variant Classes / Etiology Spectrum" color={ACCENT}>
        {source.map((e, i) => {
          const key = Object.keys(ETIOLOGY_COLORS).find(k => e.name?.includes(k)) || 'Ashkenazi';
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
                <th>Corneal Clouding</th><th>Retinal Degen.</th><th>Iron Anemia</th><th>Gastrin Level</th><th>Treatment 1</th><th>CI Avoided</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).slice(0, 15).map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.etiology}</td>
                  <td>{p.ml_type}</td>
                  <td>{p.age_onset_yr}yr</td>
                  <td>{p.seizure_type}</td>
                  <td className={p.corneal_clouding ? 'text-danger fw-bold' : 'text-muted'}>{p.corneal_clouding ? 'Yes' : 'No'}</td>
                  <td className={p.retinal_degeneration ? 'text-warning fw-bold' : 'text-muted'}>{p.retinal_degeneration ? 'Yes' : 'No'}</td>
                  <td className={p.iron_anemia ? 'text-warning' : 'text-muted'}>{p.iron_anemia ? 'Yes' : 'No'}</td>
                  <td className="text-danger small">{p.gastrin_level}</td>
                  <td>{p.treatment_1}</td>
                  <td className="text-danger">{p.ci_avoided}</td>
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

export default function MCOLN1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mcoln1/overview`).then(r => r.json()),
      fetch(`${API}/api/mcoln1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mcoln1/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
    }).catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-4">
      <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
        MCOLN1 Epilepsy Dashboard — Mucolipidosis IV (ML-IV)
      </h2>
      <p className="text-muted small mb-1">
        <strong>Mucolipidosis type IV</strong> · Mucolipin-1 (TRPML1) Lysosomal Ca²⁺ Channel Deficiency ·
        Biallelic LOF · AR · 19p13.2 ·
        Corneal Clouding PRESENT (vs ML-II absent) · Elevated Serum Gastrin PATHOGNOMONIC ·
        Normal Plasma Enzymes (vs ML-II 10-50x elevated) · ERG Abnormal 100% ·
        VGB RELATIVE CI (Retinal) · PHT NOT CI (No Cardiac) · No ERT 2026
      </p>
      <div>
        <Badge text="AR — 19p13.2" color={ACCENT} />
        <Badge text="Mucolipidosis IV (ML-IV — NOT MPS)" color={ACCENT4} />
        <Badge text="Corneal Clouding PRESENT (vs ML-II ABSENT)" color={ACCENT2} />
        <Badge text="Elevated Gastrin + Achlorhydria PATHOGNOMONIC" color={ACCENT3} />
        <Badge text="Normal Plasma Lysosomal Enzymes (vs ML-II 10-50x↑)" color={ACCENT5} />
        <Badge text="ERG Abnormal 100% — Retinal Degeneration" color={ACCENT2} />
        <Badge text="VGB RELATIVE CI (Retinal + Visual Monitoring Impossible)" color={ACCENT2} />
        <Badge text="PHT / Fosphenytoin NOT CI (No Cardiac)" color={ACCENT6} />
        <Badge text="POLG1 Mandatory Before VPA" color={ACCENT2} />
        <Badge text="Iron Supplementation MANDATORY (Achlorhydria)" color={ACCENT5} />
        <Badge text="No ERT Approved (2026)" color={ACCENT4} />
        <Badge text="Epilepsy 15-25% (Focal + GTCS; NO Infantile Spasms)" color={ACCENT} />
        <Badge text="Ashkenazi Founder IVS3-2A>G ~70% / p.Arg403Cys ~24%" color={ACCENT5} />
        <Badge text="No Bone Disease / Urine GAG NORMAL" color={ACCENT6} />
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
