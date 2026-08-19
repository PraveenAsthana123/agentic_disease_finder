'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#880e4f';   // deep pink/maroon — MPS-VI / ARSB / DS somatic dominant
const ACCENT2 = '#b71c1c';   // dark red — EXTREME HAZARD / anesthesia / PHT ABSOLUTE AVOID
const ACCENT3 = '#e65100';   // deep orange — CAUTION / relative CI / AAI
const ACCENT4 = '#1a237e';   // deep indigo — ERT / HSCT / investigational
const ACCENT5 = '#006064';   // teal — OSA / BiPAP / airway dominant trigger
const ACCENT6 = '#2e7d32';   // green — SAFE / LEV first-line / VP shunt / BiPAP

const ETIOLOGY_COLORS = {
  'Classic/Severe': '#b71c1c',
  'Attenuated': '#2e7d32',
  'Portuguese-Brazilian-Founder': '#e65100',
  'Intermediate': '#880e4f',
  'Rare-Private': '#555',
};

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

function CICard({ drug, risk, reason, alternative }) {
  const riskColor = risk?.includes('ABSOLUTE') ? ACCENT2 :
                    risk === 'EXTREME HAZARD' ? ACCENT2 :
                    risk === 'HIGH RISK' ? '#c62828' :
                    risk === 'ABSOLUTE/RELATIVE CI (STRONGEST MPS-VI contraindication)' ? ACCENT2 :
                    risk?.includes('AVOID') ? '#c62828' :
                    risk?.includes('RELATIVE CI') ? ACCENT3 :
                    risk?.includes('CAUTION') ? '#f57f17' : '#666';
  return (
    <div className="card mb-3 border-0 shadow-sm">
      <div className="card-body py-2">
        <div className="d-flex align-items-start gap-2 flex-wrap">
          <span className="badge fs-6" style={{ background: riskColor }}>{risk}</span>
          <strong>{drug}</strong>
        </div>
        <div className="small mt-1 text-muted">{reason}</div>
        {alternative && <div className="small mt-1"><span className="fw-bold text-success">Alternative:</span> {alternative}</div>}
      </div>
    </div>
  );
}

function TreatmentCard({ name, level, role, ci }) {
  const levelColor = level?.startsWith('Level A') ? ACCENT6 : level?.startsWith('Level B') ? ACCENT : ACCENT3;
  return (
    <div className="card mb-2 border-0 shadow-sm">
      <div className="card-body py-2">
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <Badge text={level || '—'} color={levelColor} />
          <strong>{name}</strong>
        </div>
        <div className="small mt-1 text-muted">{role}</div>
        {ci && <div className="small mt-1 text-danger"><strong>CI:</strong> {ci}</div>}
      </div>
    </div>
  );
}

// ── Tab components ──────────────────────────────────────────────────────────

function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading overview…</div>;
  return (
    <>
      <SectionCard title="Gene & Inheritance" color={ACCENT}>
        <div className="row">
          <div className="col-md-6">
            <table className="table table-sm table-borderless mb-0">
              <tbody>
                <tr><td className="fw-bold">Gene</td><td>{data.gene}</td></tr>
                <tr><td className="fw-bold">Locus</td><td>{data.locus}</td></tr>
                <tr><td className="fw-bold">OMIM</td><td>{data.omim}</td></tr>
                <tr><td className="fw-bold">Inheritance</td><td>{data.inheritance}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="col-md-6">
            <p className="small text-muted mb-0">{data.disease_mechanism}</p>
          </div>
        </div>
      </SectionCard>

      <div className="row g-2 mb-4">
        <KPI label="Cohort" value={data.cohort_size} color={ACCENT} />
        <KPI label="Epilepsy" value={data.epilepsy_prevalence_pct?.overall + '%'} color={ACCENT2} />
        <KPI label="DRE" value={data.drug_resistance_pct?.overall + '%'} color={ACCENT2} />
        <KPI label="OSA (Dominant)" value={`${data.osa_pct}%`} color={ACCENT5} />
        <KPI label="Hydrocephalus" value={`${data.communicating_hydrocephalus_pct}%`} color={ACCENT3} />
        <KPI label="AAI" value={`${data.atlantoaxial_instability_pct}%`} color={ACCENT3} />
        <KPI label="Corneal Clouding" value="100% (Universal)" color={ACCENT2} />
        <KPI label="Cardiac Valvulopathy" value="100% (Universal)" color={ACCENT2} />
        <KPI label="On ERT" value={`${data.on_ert_pct}%`} color={ACCENT4} />
        <KPI label="HSCT (Severe)" value={`${data.on_hsct_pct}%`} color={ACCENT4} />
        <KPI label="Hearing Loss" value={`${data.hearing_loss_pct}%`} color={ACCENT5} />
      </div>

      <SectionCard title="Variant Spectrum (Etiologies)" color={ACCENT4}>
        {data.etiologies?.map((e, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between align-items-center flex-wrap gap-1 mb-1">
              <strong style={{ color: ETIOLOGY_COLORS[e.name?.split(' ')[0]] || '#333' }}>{e.name}</strong>
              <Badge text={`${e.pct}% (n=${e.n})`} color={ETIOLOGY_COLORS[e.name?.split(' ')[0]] || ACCENT} />
            </div>
            <div className="small text-muted">{e.variant_detail}</div>
            <div className="small"><strong>Seizure risk:</strong> {e.seizure_risk} | <strong>EEG:</strong> {e.eeg}</div>
            <div className="small">
              <Badge text="Galsulfase ERT (weekly)" color={ACCENT4} />
              {e.hsct_eligible
                ? <Badge text="HSCT eligible (severe, <6-8yr)" color={ACCENT} />
                : <Badge text="ERT alone (attenuated)" color="#777" />}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Key Clinical Concepts" color={ACCENT}>
        <ul className="list-unstyled mb-0">
          {data.key_concepts?.map((c, i) => (
            <li key={i} className="mb-1 small">
              <span className="me-1" style={{ color: ACCENT }}>▸</span>{c}
            </li>
          ))}
        </ul>
      </SectionCard>
    </>
  );
}

function EtiologyTab({ data }) {
  if (!data) return <div className="text-muted">Loading breakdown…</div>;
  const { patients = [], etiologies = [] } = data;
  return (
    <>
      <div className="row g-2 mb-4">
        <KPI label="Cohort" value={data.cohort_size} color={ACCENT} />
        <KPI label="Seizures" value={`${data.seizure_n} (${data.seizure_pct}%)`} color={ACCENT2} />
        <KPI label="DRE" value={`${data.drug_resistant_n} (${data.drug_resistant_pct}%)`} color={ACCENT2} />
        <KPI label="OSA" value={`${data.osa_n} (${data.osa_pct}%)`} color={ACCENT5} />
        <KPI label="Hydrocephalus" value={`${data.hydrocephalus_n} (${data.hydrocephalus_pct}%)`} color={ACCENT3} />
        <KPI label="VP Shunt" value={`${data.vp_shunt_n} (${data.vp_shunt_pct}%)`} color={ACCENT6} />
        <KPI label="On ERT" value={`${data.on_ert_n} (${data.on_ert_pct}%)`} color={ACCENT4} />
        <KPI label="Post-HSCT" value={`${data.on_hsct_n} (${data.on_hsct_pct}%)`} color={ACCENT4} />
        <KPI label="AAI" value={`${data.atlantoaxial_n} (${data.atlantoaxial_pct}%)`} color={ACCENT3} />
      </div>

      <SectionCard title="Etiology Distribution" color={ACCENT4}>
        {etiologies.map((e, i) => (
          <PctBar key={i} label={e.name} pct={e.pct} color={ETIOLOGY_COLORS[e.name?.split(' ')[0]] || ACCENT} />
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients, seed=42)" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>ID</th><th>Phenotype</th><th>Seizures</th>
                <th>AED</th><th>Response</th><th>OSA</th>
                <th>Hydro</th><th>Shunt</th><th>ERT</th><th>HSCT</th><th>AAI</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i}>
                  <td><code>{p.patient_id}</code></td>
                  <td><Badge text={p.phenotype?.split(' (')[0]} color={
                    p.phenotype?.includes('Classic') ? ACCENT2 :
                    p.phenotype?.includes('Portuguese') ? ACCENT3 :
                    p.phenotype?.includes('Attenuated') ? ACCENT6 : '#555'} /></td>
                  <td>{p.has_seizures ? <span className="text-danger fw-bold">Yes</span> : <span className="text-muted">No</span>}</td>
                  <td>{p.primary_aed || '—'}</td>
                  <td>
                    {p.drug_response ? (
                      <Badge text={p.drug_response}
                        color={p.drug_response === 'Drug-resistant' ? ACCENT2 :
                               p.drug_response === 'Partially controlled' ? ACCENT3 : ACCENT6} />
                    ) : '—'}
                  </td>
                  <td>{p.osa ? <Badge text="OSA" color={ACCENT5} /> : '—'}</td>
                  <td>{p.hydrocephalus ? <Badge text="Hydro+" color={ACCENT3} /> : '—'}</td>
                  <td>{p.vp_shunt ? <Badge text="Shunt" color={ACCENT6} /> : '—'}</td>
                  <td>{p.on_ert ? <Badge text="ERT" color={ACCENT4} /> : '—'}</td>
                  <td>{p.post_hsct ? <Badge text="HSCT" color={ACCENT4} /> : '—'}</td>
                  <td>{p.atlantoaxial_instability ? <Badge text="AAI+" color={ACCENT3} /> : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading breakdown…</div>;
  const { seizure_types = [], triggers = [] } = data;
  return (
    <>
      <SectionCard title="Seizure Types" color={ACCENT2}>
        {seizure_types.map((st, i) => (
          <div key={i} className="mb-2">
            <PctBar label={st.type} pct={st.pct} color={ACCENT2} />
            <div className="small text-muted">{st.eeg}</div>
          </div>
        ))}
      </SectionCard>
      <SectionCard title="Seizure Triggers (MPS-VI Specific — OSA Dominant)" color={ACCENT5}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3">
            <PctBar label={t.trigger} pct={t.pct} color={ACCENT5} />
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </SectionCard>
      <SectionCard title="Monitoring Protocol" color={ACCENT4}>
        <ul className="list-unstyled mb-0">
          {data.monitoring?.map((m, i) => (
            <li key={i} className="mb-1 small">
              <span className="me-1" style={{ color: ACCENT4 }}>▸</span>{m}
            </li>
          ))}
        </ul>
      </SectionCard>
      <SectionCard title="Clinical Thresholds & Action Points" color={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr></thead>
            <tbody>
              {data.thresholds?.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold small">{t.parameter}</td>
                  <td><Badge text={t.threshold} color={ACCENT3} /></td>
                  <td className="small text-muted">{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
      <SectionCard title="Disease Lifecycle" color={ACCENT}>
        {data.lifecycle?.map((l, i) => (
          <div key={i} className="mb-3">
            <div className="fw-bold">{l.stage}</div>
            <div className="small text-muted">{l.features}</div>
            <div className="small mt-1"><strong>Action:</strong> {l.action}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading breakdown…</div>;
  return (
    <>
      <SectionCard title="Treatments (AEDs + ERT + HSCT + Surgical)" color={ACCENT6}>
        {data.treatments?.map((t, i) => (
          <TreatmentCard key={i} {...t} />
        ))}
      </SectionCard>
      <SectionCard title="Contraindications & Hazards" color={ACCENT2}>
        {data.contraindications?.map((c, i) => (
          <CICard key={i} {...c} />
        ))}
      </SectionCard>
    </>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading definitions…</div>;
  return (
    <>
      <SectionCard title="Glossary" color={ACCENT4}>
        <div className="row">
          {data.definitions?.map((d, i) => (
            <div key={i} className="col-md-6 mb-3">
              <strong>{d.term}</strong>
              <p className="small text-muted mb-0">{d.definition}</p>
            </div>
          ))}
        </div>
      </SectionCard>
      <SectionCard title="10-Step Diagnostic Algorithm" color={ACCENT}>
        <ol className="mb-0">
          {data.diagnostic_algorithm?.map((step, i) => (
            <li key={i} className="small mb-1">{step}</li>
          ))}
        </ol>
      </SectionCard>
      <SectionCard title="Pharmacological Distinctions (12 Points)" color={ACCENT2}>
        <ol className="mb-0">
          {data.pharmacological_distinctions?.map((p, i) => (
            <li key={i} className="small mb-1">{p}</li>
          ))}
        </ol>
      </SectionCard>
      <SectionCard title="Differential Diagnosis" color={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Condition</th><th>Distinguishing Features</th></tr></thead>
            <tbody>
              {data.differential_diagnosis?.map((d, i) => (
                <tr key={i}>
                  <td className="fw-bold small">{d.condition}</td>
                  <td className="small text-muted">{d.distinction}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
      <SectionCard title="Key Concepts" color={ACCENT}>
        <ul className="list-unstyled mb-0">
          {data.key_concepts?.map((c, i) => (
            <li key={i} className="mb-1 small">
              <span className="me-1" style={{ color: ACCENT }}>▸</span>{c}
            </li>
          ))}
        </ul>
      </SectionCard>
      <SectionCard title="Reference Standards" color={ACCENT4}>
        <ul className="mb-0">
          {data.standards?.map((s, i) => <li key={i} className="small">{s}</li>)}
        </ul>
      </SectionCard>
    </>
  );
}

// ── Main component ──────────────────────────────────────────────────────────

export default function ARSBPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/arsb/overview`).then(r => r.json()).then(setOverview).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3) {
      fetch(`${API}/api/arsb/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/arsb/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4">
        <h1 className="h3 fw-bold mb-1" style={{ color: ACCENT }}>
          🫀 ARSB Epilepsy Dashboard
        </h1>
        <p className="text-muted small mb-1">
          <strong>MPS-VI (Maroteaux-Lamy Syndrome)</strong> · Arylsulfatase B (N-Acetylgalactosamine-4-Sulfatase) Deficiency ·
          Autosomal Recessive · 5q14.1 · DS+C4S elevated (NOT HS/KS — GAG fingerprint) ·
          NORMAL INTELLIGENCE (key feature — contrasts MPS I/II/III) ·
          OSA DOMINANT trigger (macroglossia + laryngeal DS — more severe than any other MPS) ·
          Corneal clouding UNIVERSAL and MOST SEVERE (Goldman VF IMPOSSIBLE → VGB strongest CI) ·
          Cardiac valvulopathy UNIVERSAL (PHT ABSOLUTE AVOID) ·
          Communicating hydrocephalus 20-30% (DS arachnoid infiltration) ·
          ERT: Galsulfase (Naglazyme) FDA 2005 weekly · HSCT for severe &lt;6-8yr (somatic benefit) ·
          Epilepsy 15-25% · AR both sexes equally · p.R152W Portuguese-Brazilian founder
        </p>
        <div>
          <Badge text="AR — 5q14.1" color={ACCENT} />
          <Badge text="DS+C4S elevated (NOT HS/KS)" color={ACCENT4} />
          <Badge text="Normal Intelligence KEY" color={ACCENT6} />
          <Badge text="OSA DOMINANT trigger (macroglossia)" color={ACCENT5} />
          <Badge text="Corneal Clouding 100% (Goldman VF IMPOSSIBLE)" color={ACCENT2} />
          <Badge text="Cardiac 100% (PHT ABSOLUTE AVOID)" color={ACCENT2} />
          <Badge text="Hydrocephalus 20-30%" color={ACCENT3} />
          <Badge text="ERT: Galsulfase FDA 2005 weekly" color={ACCENT4} />
          <Badge text="HSCT severe <6-8yr (somatic)" color={ACCENT4} />
          <Badge text="VGB ABSOLUTE CI (corneal)" color={ACCENT2} />
          <Badge text="POLG1 mandatory" color={ACCENT2} />
        </div>
      </div>

      {err && <div className="alert alert-danger small">API error: {err}</div>}

      <ul className="nav nav-tabs mb-4">
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
