'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4a148c';   // deep purple — MPS-IVA / GALNS / KS+C6S skeletal
const ACCENT2 = '#b71c1c';   // dark red — EXTREME HAZARD / anesthesia / cord compression
const ACCENT3 = '#e65100';   // deep orange — CAUTION / relative CI / AAI
const ACCENT4 = '#1a237e';   // deep indigo — ERT / investigational
const ACCENT5 = '#006064';   // teal — OSA / respiratory / thoracic cage restriction
const ACCENT6 = '#2e7d32';   // green — SAFE / LEV first-line / cervical fusion

const ETIOLOGY_COLORS = {
  'Classic/Severe (null/null — biallelic truncating)': '#b71c1c',
  'Attenuated (biallelic missense — p.R386C)': '#1a237e',
  'Saudi Founder (p.I113F biallelic — consanguineous Middle East)': '#e65100',
  'Intermediate (null + missense — compound-het)': '#4a148c',
  'Rare/Private (deep intronic or novel biallelic)': '#555',
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
  const riskColor = risk === 'EXTREME HAZARD' ? ACCENT2 :
                    risk === 'AVOID' ? '#c62828' :
                    risk === 'HIGH RISK' ? '#c62828' :
                    risk === 'RELATIVE CI' ? ACCENT3 :
                    risk === 'CAUTION (not absolute CI)' ? '#f57f17' : '#666';
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
        <KPI label="Epilepsy" value={data.epilepsy_prevalence_pct} color={ACCENT2} />
        <KPI label="DRE" value={data.drug_resistance_pct} color={ACCENT2} />
        <KPI label="OSA" value={`${data.osa_pct}%`} color={ACCENT5} />
        <KPI label="AAI" value={`${data.atlantoaxial_pct}%`} color={ACCENT3} />
        <KPI label="Cervical Fusion" value={`${data.cervical_fusion_pct}%`} color={ACCENT6} />
        <KPI label="On ERT" value={`${data.on_ert_pct}%`} color={ACCENT4} />
        <KPI label="HSCT" value="0% (N/A)" color="#999" />
        <KPI label="Hearing Loss" value={`${data.hearing_loss_pct}%`} color={ACCENT5} />
      </div>

      <SectionCard title="Variant Spectrum (Etiologies)" color={ACCENT4}>
        {data.etiologies?.map((e, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between align-items-center flex-wrap gap-1 mb-1">
              <strong style={{ color: ETIOLOGY_COLORS[e.name] || '#333' }}>{e.name}</strong>
              <Badge text={`${e.pct}% (n=${e.n})`} color={ETIOLOGY_COLORS[e.name] || ACCENT} />
            </div>
            <div className="small text-muted">{e.variant_detail}</div>
            <div className="small"><strong>Seizure risk:</strong> {e.seizure_risk} | <strong>EEG:</strong> {e.eeg}</div>
            <div className="small">
              <Badge text="ERT (elosulfase alfa)" color={ACCENT4} />
              <Badge text="No HSCT (intelligence normal)" color="#777" />
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
        <KPI label="On ERT" value={`${data.on_ert_n} (${data.on_ert_pct}%)`} color={ACCENT4} />
        <KPI label="AAI" value={`${data.atlantoaxial_n} (${data.atlantoaxial_pct}%)`} color={ACCENT3} />
        <KPI label="Cervical Fusion" value={`${data.cervical_fusion_n} (${data.cervical_fusion_pct}%)`} color={ACCENT6} />
      </div>

      <SectionCard title="Etiology Distribution" color={ACCENT4}>
        {etiologies.map((e, i) => (
          <PctBar key={i} label={e.name} pct={e.pct} color={ETIOLOGY_COLORS[e.name] || ACCENT} />
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients, seed=42)" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>ID</th><th>Phenotype</th><th>Etiology</th><th>Seizures</th>
                <th>AED</th><th>Response</th><th>OSA</th><th>ERT</th><th>AAI</th>
                <th>Fusion</th><th>Hearing</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i}>
                  <td><code>{p.patient_id}</code></td>
                  <td><Badge text={p.phenotype?.split(' (')[0]} color={
                    p.phenotype?.includes('Classic') ? ACCENT2 :
                    p.phenotype?.includes('Saudi') ? ACCENT3 :
                    p.phenotype?.includes('Attenuated') ? ACCENT6 : '#555'} /></td>
                  <td className="text-muted" style={{ maxWidth: 160, whiteSpace: 'normal' }}>
                    {p.etiology?.split(' (')[0]}
                  </td>
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
                  <td>{p.on_ert ? <Badge text="ERT" color={ACCENT4} /> : '—'}</td>
                  <td>{p.atlantoaxial_instability ? <Badge text="AAI+" color={ACCENT3} /> : '—'}</td>
                  <td>{p.cervical_fusion ? <Badge text="Fused" color={ACCENT6} /> : '—'}</td>
                  <td>{p.hearing_loss ? <Badge text="HL+" color={ACCENT5} /> : '—'}</td>
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
      <SectionCard title="Seizure Triggers (MPS-IVA Specific)" color={ACCENT5}>
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
      <SectionCard title="Treatments (AEDs + ERT + Surgical)" color={ACCENT6}>
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

export default function GALNSPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/galns/overview`).then(r => r.json()).then(setOverview).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3) {
      fetch(`${API}/api/galns/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/galns/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4">
        <h1 className="h3 fw-bold mb-1" style={{ color: ACCENT }}>
          🦴 GALNS Epilepsy Dashboard
        </h1>
        <p className="text-muted small mb-1">
          <strong>MPS-IVA (Morquio A Syndrome)</strong> · N-acetylgalactosamine-6-sulfate Sulfatase Deficiency ·
          Autosomal Recessive · 16q24.3 · KS+C6S elevated (NOT HS/DS — fingerprint distinction) ·
          NORMAL INTELLIGENCE (key feature — contrasts MPS I/II/III) ·
          Atlantoaxial Instability MOST SEVERE among all MPS (odontoid hypoplasia) ·
          Anesthesia EXTREME HAZARD (neck hyperextension → cord transection) ·
          ERT: Elosulfase alfa (Vimizim) FDA 2014 every other week · No HSCT (intelligence preserved) ·
          Epilepsy LOW 10-15% (cord compression primary mechanism, not CNS GAG) ·
          OSA 50% (thoracic cage restriction + tracheomalacia) · Pectus carinatum universal ·
          AR both sexes equally affected
        </p>
        <div>
          <Badge text="AR — 16q24.3" color={ACCENT} />
          <Badge text="KS+C6S elevated (NOT HS/DS)" color={ACCENT4} />
          <Badge text="Normal Intelligence KEY" color={ACCENT6} />
          <Badge text="AAI MOST SEVERE (odontoid hypoplasia)" color={ACCENT2} />
          <Badge text="Anesthesia EXTREME HAZARD" color={ACCENT2} />
          <Badge text="ERT: Elosulfase alfa FDA 2014" color={ACCENT4} />
          <Badge text="No HSCT" color="#777" />
          <Badge text="Epilepsy LOW 10-15%" color={ACCENT3} />
          <Badge text="PHT AVOID cardiac" color={ACCENT2} />
          <Badge text="VGB RELATIVE CI" color={ACCENT3} />
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
