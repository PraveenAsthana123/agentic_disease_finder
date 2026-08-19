'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4a148c';   // deep purple — GUSB / MPS-VII / rare LSD
const ACCENT2 = '#b71c1c';   // dark red — EXTREME HAZARD / ABSOLUTE CI / hydrops
const ACCENT3 = '#e65100';   // deep orange — CAUTION / RELATIVE CI / AAI
const ACCENT4 = '#1a237e';   // deep indigo — ERT / HSCT / investigational
const ACCENT5 = '#006064';   // teal — OSA / BiPAP / hydrocephalus
const ACCENT6 = '#2e7d32';   // green — SAFE / LEV first-line / VPA (POLG1-cleared)

const ETIOLOGY_COLORS = {
  'Neonatal/Severe': '#b71c1c',
  'Belgian-Founder': '#4a148c',
  'Juvenile/Intermediate': '#e65100',
  'Attenuated': '#2e7d32',
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

function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const kpis = data.kpis || [];
  return (
    <>
      <div className="row mb-4">
        {kpis.map((k, i) => <KPI key={i} label={k.label} value={k.value} color={ACCENT} />)}
      </div>

      <SectionCard title="⚠️ MPS-VII Unique Features — GUSB / Sly Syndrome" color={ACCENT2}>
        <ul className="mb-0 small">
          <li><strong>NON-IMMUNE HYDROPS FETALIS (30-40% neonatal-severe)</strong> — ONLY MPS type with hydrops as primary presentation; NIHF + coarse facies + hepatosplenomegaly → GUSB enzyme assay IMMEDIATELY</li>
          <li><strong>HS + DS + CS TRIPLE GAG elevation</strong> — PATHOGNOMONIC; no other MPS type elevates all three simultaneously; CS distinguishes MPS-VII from MPS-I/II (HS+DS only)</li>
          <li><strong>Corneal clouding PRESENT</strong> — distinguishes from MPS-II Hunter (absent) at bedside; VGB relative CI (corneal + HS+DS+CS triple accumulation)</li>
          <li><strong>Vestronidase alfa (FDA 2017)</strong> — 4 mg/kg IV every 2 WEEKS (not weekly like galsulfase); DOES NOT cross BBB; somatic benefit only</li>
          <li><strong>POLG1 mandatory exclusion</strong> before VPA — HS+DS+CS mitochondrial stress amplifies hepatotoxicity; CPIC A</li>
        </ul>
      </SectionCard>

      <SectionCard title="🧬 Disease Mechanism" color={ACCENT4}>
        <p className="small mb-0">{data.disease_mechanism}</p>
      </SectionCard>

      <SectionCard title="📊 Key Clinical Metrics" color={ACCENT}>
        <div className="row">
          <div className="col-md-6">
            <PctBar label="Epilepsy prevalence" pct={data.epilepsy_prevalence_pct || 40} color={ACCENT} />
            <PctBar label="Drug-resistant epilepsy" pct={data.drug_resistance_pct || 18} color={ACCENT2} />
            <PctBar label="OSA / airway obstruction" pct={data.osA_pct || 52} color={ACCENT5} />
          </div>
          <div className="col-md-6">
            <PctBar label="Communicating hydrocephalus" pct={data.hydrocephalus_pct || 43} color={ACCENT3} />
            <PctBar label="Atlantoaxial instability" pct={data.aai_pct || 36} color={ACCENT3} />
            <PctBar label="Hydrops fetalis (neonatal-severe)" pct={data.hydrops_fetalis_pct_severe || 35} color={ACCENT2} />
          </div>
        </div>
      </SectionCard>

      {data.clinical_pearls?.length > 0 && (
        <SectionCard title="💎 Clinical Pearls — MPS-VII" color="#5d4037">
          {data.clinical_pearls.map((p, i) => (
            <div key={i} className="mb-3">
              <strong className="small">{p.pearl}</strong>
              <p className="small text-muted mb-0 mt-1">{p.detail}</p>
            </div>
          ))}
        </SectionCard>
      )}

      {data.monitoring_parameters?.length > 0 && (
        <SectionCard title="🩺 Monitoring Parameters" color={ACCENT5}>
          <ul className="small mb-0">
            {data.monitoring_parameters.map((m, i) => <li key={i} className="mb-1">{m}</li>)}
          </ul>
        </SectionCard>
      )}
    </>
  );
}

function EtiologyTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const etiologies = data.etiologies || [];
  return (
    <>
      <SectionCard title="🧬 Variant Classes / Etiology Spectrum" color={ACCENT}>
        {etiologies.map((e, i) => {
          const key = Object.keys(ETIOLOGY_COLORS).find(k => e.name?.includes(k)) || 'Rare-Private';
          const color = ETIOLOGY_COLORS[key] || ACCENT;
          return (
            <div key={i} className="mb-4 pb-3 border-bottom">
              <div className="d-flex align-items-center gap-2 mb-1 flex-wrap">
                <span className="badge" style={{ background: color }}>{e.pct}% (n={e.n})</span>
                <strong>{e.name}</strong>
                {e.hsct_eligible && <Badge text="HSCT eligible" color={ACCENT4} />}
              </div>
              <div className="small text-muted mb-1"><strong>Seizure risk:</strong> {e.seizure_risk}</div>
              <div className="small text-muted mb-1"><strong>EEG:</strong> {e.eeg}</div>
              <div className="small text-muted"><strong>Variant detail:</strong> {e.variant_detail}</div>
            </div>
          );
        })}
      </SectionCard>
    </>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const seizures = data.seizure_types || [];
  const triggers = data.triggers || [];
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
            <div className="small text-muted">{s.eeg}</div>
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
  const treatments = data.treatments || [];
  const cis = data.contraindications || [];
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
  const abbrev = data.abbreviations || {};
  const refs = data.references || [];
  return (
    <>
      <SectionCard title="📖 Disease Definitions" color={ACCENT}>
        <table className="table table-sm small mb-0">
          <tbody>
            {[
              ['Gene', data.gene],
              ['Full name', data.full_name],
              ['Disease', data.disease],
              ['OMIM', data.omim],
              ['Locus', data.locus],
              ['Inheritance', data.inheritance],
              ['Enzyme defect', data.enzyme_defect],
              ['GAG elevated', data.gag_elevated],
              ['ERT', data.ert],
              ['HSCT', data.hsct],
              ['Epilepsy prevalence', data.epilepsy_pct],
              ['DRE rate', data.dre_pct],
              ['Key distinguishing', data.key_distinguishing],
              ['Founder mutation', data.founder_mutation],
              ['POLG1 mandatory', data.polg1_mandatory ? 'YES — CPIC A before VPA' : 'No'],
            ].map(([k, v]) => (
              <tr key={k}>
                <td className="fw-bold text-nowrap pe-3" style={{ color: ACCENT }}>{k}</td>
                <td>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </SectionCard>

      <SectionCard title="🔤 Abbreviations" color={ACCENT4}>
        <div className="row">
          {Object.entries(abbrev).map(([k, v]) => (
            <div key={k} className="col-12 col-md-6 small mb-1">
              <strong>{k}:</strong> {v}
            </div>
          ))}
        </div>
      </SectionCard>

      {refs.length > 0 && (
        <SectionCard title="📚 References" color="#5d4037">
          <ul className="small mb-0">
            {refs.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        </SectionCard>
      )}
    </>
  );
}

export default function GUSBPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/gusb/overview`).then(r => r.json()),
      fetch(`${API}/api/gusb/breakdown`).then(r => r.json()),
      fetch(`${API}/api/gusb/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
    }).catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-4">
      <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
        GUSB Epilepsy Dashboard — MPS-VII (Sly Syndrome)
      </h2>
      <p className="text-muted small mb-1">
        <strong>MPS-VII (Sly Syndrome)</strong> · Beta-glucuronidase (GUSB) Deficiency ·
        Autosomal Recessive · 7q11.21 · HS+DS+CS ALL THREE elevated (PATHOGNOMONIC — unique among all MPS) ·
        NON-IMMUNE HYDROPS FETALIS (30-40% neonatal-severe — ONLY MPS type with hydrops) ·
        Corneal clouding PRESENT (distinguishes from MPS-II Hunter absent) ·
        ERT: Vestronidase alfa (Mepsevii) FDA 2017 — 4 mg/kg IV every 2 weeks ·
        HSCT: severe early-onset &lt; 3yr (data limited) ·
        Epilepsy 30-50% · POLG1 mandatory · p.L176F Belgian founder · AR both sexes equally
      </p>
      <div>
        <Badge text="AR — 7q11.21" color={ACCENT} />
        <Badge text="HS+DS+CS all three (PATHOGNOMONIC)" color={ACCENT4} />
        <Badge text="Non-immune Hydrops Fetalis (ONLY MPS)" color={ACCENT2} />
        <Badge text="Corneal clouding PRESENT (≠ MPS-II)" color={ACCENT3} />
        <Badge text="Vestronidase alfa FDA 2017 (q2 weeks)" color={ACCENT4} />
        <Badge text="HSCT severe <3yr (limited data)" color={ACCENT4} />
        <Badge text="POLG1 mandatory" color={ACCENT2} />
        <Badge text="VGB RELATIVE CI (corneal)" color={ACCENT3} />
        <Badge text="PHT AVOID (cardiac)" color={ACCENT2} />
        <Badge text="Typical-AP HIGH RISK (HS+DS+CS basal ganglia)" color={ACCENT2} />
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
