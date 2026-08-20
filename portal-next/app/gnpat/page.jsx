'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#00695c';   // teal — GNPAT enzyme / plasmalogen Step 1 / RCDP2 identity
const ACCENT2 = '#b71c1c';   // dark red — HIGH RISK / VGB cataracts / VPA hepatotoxicity
const ACCENT3 = '#e65100';   // deep orange — RELATIVE CI / caution / thresholds
const ACCENT4 = '#1565c0';   // deep blue — safe treatments / LEV / first-line
const ACCENT5 = '#1b5e20';   // deep green — NORMAL markers (VLCFA, phytanic) / positive findings
const ACCENT6 = '#6a1b9a';   // purple — alkylglycerol / experimental / plasmalogen precursor

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
  const numPct = typeof pct === 'string' ? parseInt(pct) : pct;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${numPct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span className="badge me-1" style={{ backgroundColor: color, color: '#fff', fontSize: 11 }}>{text}</span>
  );
}

function CICard({ drug, level, reason, alternative }) {
  const color = level?.includes('HIGH RISK') || level?.includes('HAZARD')
    ? ACCENT2 : ACCENT3;
  return (
    <div className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold small">{drug}</span>
          <Badge text={level?.split('(')[0]?.trim().split(' ').slice(0, 3).join(' ')} color={color} />
        </div>
        <p className="small text-danger mb-1">{reason}</p>
        {alternative && <p className="small text-muted mb-0"><strong>Alternative:</strong> {alternative}</p>}
      </div>
    </div>
  );
}

function TreatmentCard({ drug, class: cls, evidence, dose, moa, monitoring, ci }) {
  return (
    <div className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold small">{drug}</span>
          <Badge text={cls?.split('—')[0]?.trim()} color={ACCENT4} />
        </div>
        <p className="small mb-1"><strong>Evidence:</strong> {evidence}</p>
        {dose && <p className="small mb-1"><strong>Dose:</strong> {dose}</p>}
        {moa && <p className="small mb-1"><strong>MOA:</strong> {moa}</p>}
        {monitoring && <p className="small mb-1 text-muted"><strong>Monitor:</strong> {monitoring}</p>}
        {ci && <p className="small mb-0 text-warning"><strong>CI:</strong> {ci}</p>}
      </div>
    </div>
  );
}

// ── Overview Tab ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading overview…</div>;
  const d = data;
  return (
    <div>
      <Alert
        text="🚨 CRITICAL DISTINCTION FROM RCDP1/PEX7: Phytanic acid is NORMAL in RCDP2/GNPAT (PEX7 is intact → PHYH imported via PTS2 normally). RCDP1 phytanic ELEVATED; RCDP2 phytanic NORMAL. Both have plasmalogens severely low. VLCFA NORMAL in both (distinguishes from ZSD)."
        variant="danger"
      />
      <Alert
        text="🚨 VGB — HIGH RISK: Cataracts present in 60-70% of RCDP2 + VGB irreversible visual field constriction = additive visual impairment. NOT absolute CI (unlike ZSD where retinopathy universal/severe), but HIGH RISK. Monthly VF/VEP monitoring mandatory if VGB used."
        variant="danger"
      />
      <Alert
        text="⚠ VPA — RELATIVE CI: Hepatotoxicity risk. NOTE: Phytanic NORMAL in RCDP2 (less phytanic-driven hepatic stress vs RCDP1), but POLG1 exclusion remains MANDATORY (CPIC Grade A). LFT q3 months if VPA used."
        variant="warning"
      />
      <Alert
        text={`ℹ GNPAT = DHAPAT (enzyme, PTS1 import, NOT PTS2). Alkylglycerols MOST DIRECT bypass in RCDP2 (bypass the blocked GNPAT Step 1). PHT/CBZ/OXC CAN be used (no adrenal mechanism). No ERT / No HSCT. No founder mutation — full gene sequencing required.`}
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="Classic RCDP2 %" value={`${d.classic_rcdp_pct}%`} color={ACCENT2} />
        <KPI label="Intermediate %" value={`${d.intermediate_rcdp_pct}%`} color={ACCENT3} />
        <KPI label="Mild RCDP2 %" value={`${d.mild_rcdp_pct}%`} color={ACCENT5} />
        <KPI label="Drug Resistant %" value={`${d.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="Cataract %" value={`${d.cataract_pct}%`} color={ACCENT3} />
        <KPI label="Rhizomelia %" value={`${d.rhizomelia_pct}%`} color={ACCENT} />
        <KPI label="VLCFA NORMAL" value="100%" color={ACCENT5} />
        <KPI label="Plasmalogens LOW" value={`${d.plasmalogen_low_pct}%`} color={ACCENT2} />
        <KPI label="OMIM Gene" value={`*${d.omim_gene}`} color={ACCENT} />
        <KPI label="Locus" value={d.locus} color={ACCENT4} />
      </div>

      <SectionCard title="Disease Summary — GNPAT / RCDP2 (Rhizomelic Chondrodysplasia Punctata Type 2)" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-1"><strong>Common Variant:</strong> {d.common_variant}</p>
        <p className="small mb-0">{d.disease_mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Profile" borderColor={ACCENT4}>
            <PctBar label="Seizures (overall cohort)" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Drug-resistant seizures" pct={d.drug_resistance_pct} color={ACCENT2} />
            <PctBar label="Classic RCDP2 (severe)" pct={d.classic_rcdp_pct} color={ACCENT2} />
            <PctBar label="Intermediate RCDP2" pct={d.intermediate_rcdp_pct} color={ACCENT3} />
            <PctBar label="Mild RCDP2" pct={d.mild_rcdp_pct} color={ACCENT5} />
            <PctBar label="Cataracts (VGB risk additive)" pct={d.cataract_pct} color={ACCENT3} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Biochemical Profile — KEY DISTINCTIONS" borderColor={ACCENT4}>
            <PctBar label="VLCFA C26:0 NORMAL (PTS1 intact)" pct={d.vlcfa_normal_pct} color={ACCENT5} />
            <PctBar label="Plasmalogens (RBC) SEVERELY LOW" pct={d.plasmalogen_low_pct} color={ACCENT2} />
            <PctBar label="Phytanic acid NORMAL (PEX7 intact)" pct={d.phytanic_normal_pct} color={ACCENT5} />
            <PctBar label="Rhizomelia (proximal shortening)" pct={d.rhizomelia_pct} color={ACCENT} />
            <PctBar label="Stippled epiphyses (neonatal)" pct={d.stippling_pct} color={ACCENT3} />
            <div className="mt-2 small text-muted">
              <strong>NBS:</strong> {d.nbs_positive_rate}
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
        <Alert text="VGB — HIGH RISK (cataracts 60-70% + VF loss = additive). NOT absolute CI as in ZSD. Monthly VF/VEP monitoring mandatory if used." variant="danger" />
        <Alert text="VPA — RELATIVE CI (hepatotoxicity). Phytanic NORMAL in RCDP2 (less phytanic-hepatic stress vs RCDP1), but POLG1 MANDATORY (CPIC A). LFT q3 months." variant="warning" />
        <Alert text="PHT/CBZ/OXC — CAN BE USED (no adrenal insufficiency in RCDP2; no CYP3A4 cortisol mechanism). Contrast with ABCD1 where PHT = ABSOLUTE CI." variant="secondary" />
        <Alert text="LEV first-line all forms. ACTH Level B for IS. Alkylglycerols most direct bypass (Step 1 GNPAT block). DHA Level C. No ERT / No HSCT." variant="info" />
      </SectionCard>

      <SectionCard title="Key Concepts" borderColor={ACCENT}>
        <ul className="list-unstyled mb-0">
          {d.key_concepts?.map((c, i) => (
            <li key={i} className="mb-1 small">
              <span className="me-1" style={{ color: ACCENT }}>▸</span>{c}
            </li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="Reference Standards" borderColor={ACCENT4}>
        <ul className="mb-0">
          {d.standards?.map((s, i) => <li key={i} className="small">{s}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Patients & Etiology Tab ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { etiologies, patients } = data;
  const ETIOL_COLORS = {
    'Classic RCDP2': ACCENT2,
    'Intermediate RCDP2': ACCENT3,
    'Mild RCDP2': ACCENT5,
  };
  const getColor = (name) => {
    for (const [key, color] of Object.entries(ETIOL_COLORS)) {
      if (name?.includes(key.split(' ')[0]) && name?.includes(key.split(' ')[1])) return color;
    }
    return ACCENT;
  };
  return (
    <div>
      <Alert
        text="ℹ RCDP2 SPECTRUM (GNPAT): Classic (null/null → severe 40%) · Intermediate (null/hypomorphic → 40%) · Mild (hypomorphic/hypomorphic → 20%). No founder mutation — all private/rare variants. Phytanic NORMAL in all forms (PEX7 intact)."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>GNPAT-RCDP2 Phenotypic Classes — 3 Forms (40 Patients)</h6>
      {etiologies?.map((e, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${getColor(e.name)}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-2">
              <h6 className="fw-bold mb-0" style={{ color: getColor(e.name) }}>{e.name}</h6>
              <div>
                <Badge text={`${e.pct}%`} color={getColor(e.name)} />
                <Badge text={`n=${e.n}`} color="#555" />
                <Badge text={e.sex} color="#888" />
              </div>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: getColor(e.name) }} />
            </div>
            <div className="row small">
              <div className="col-md-6">
                <p className="mb-1"><strong>Onset:</strong> {e.onset_age}</p>
                <p className="mb-1"><strong>Seizure risk:</strong> {e.seizure_risk}</p>
                <p className="mb-1"><strong>EEG:</strong> {e.eeg}</p>
              </div>
              <div className="col-md-6">
                <p className="mb-1"><strong>MRI:</strong> {e.mri}</p>
                <div className="d-flex gap-2 flex-wrap mt-1">
                  {e.dha_supplement && <Badge text="DHA supplementation" color={ACCENT6} />}
                  {!e.hsct_eligible && <Badge text="HSCT not indicated" color="#888" />}
                  {!e.ert_available && <Badge text="No ERT (2026)" color="#888" />}
                </div>
              </div>
            </div>
            <p className="small text-muted mb-0 mt-2">{e.variant_detail}</p>
          </div>
        </div>
      ))}

      <SectionCard title="Patient Table (40 patients)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead>
              <tr>
                <th>ID</th><th>Phenotype</th><th>Sex</th><th>Genotype</th><th>Cataract</th>
                <th>Seizures</th><th>AED</th><th>Response</th><th>DHA</th><th>Alkylglycerol</th>
              </tr>
            </thead>
            <tbody>
              {patients?.map((p, i) => (
                <tr key={i}>
                  <td><code>{p.patient_id}</code></td>
                  <td><Badge text={p.phenotype?.split(' ')[0] + ' ' + (p.phenotype?.split(' ')[1] || '')} color={getColor(p.phenotype)} /></td>
                  <td>{p.sex}</td>
                  <td className="small">{p.genotype}</td>
                  <td>{p.cataract ? <span className="text-warning fw-bold">Yes</span> : <span className="text-muted">—</span>}</td>
                  <td>{p.has_seizures ? <span className="text-danger fw-bold">Yes</span> : <span className="text-muted">No</span>}</td>
                  <td className="small">{p.primary_aed || '—'}</td>
                  <td>
                    {p.drug_response ? (
                      <Badge text={p.drug_response}
                        color={p.drug_response === 'Drug-resistant' ? ACCENT2 :
                               p.drug_response === 'Partially controlled' ? ACCENT3 : ACCENT4} />
                    ) : '—'}
                  </td>
                  <td>{p.on_dha ? <Badge text="DHA" color={ACCENT6} /> : '—'}</td>
                  <td>{p.on_alkylglycerol ? <Badge text="AlkylG" color={ACCENT6} /> : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Seizures & Triggers Tab ───────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types = [], triggers = [], monitoring = [], thresholds = [], lifecycle = [] } = data;
  return (
    <>
      <SectionCard title="Seizure Types (RCDP2 — IS, Focal, Myoclonic, GTCS)" borderColor={ACCENT2}>
        {seizure_types.map((st, i) => (
          <div key={i} className="mb-3">
            <PctBar label={st.type} pct={st.pct} color={ACCENT2} />
            <div className="small text-muted">{st.eeg}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers (Fever + Metabolic Stress — Phytanic Less Prominent)" borderColor={ACCENT3}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3">
            <PctBar label={t.trigger} pct={t.pct} color={ACCENT3} />
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Protocol" borderColor={ACCENT4}>
        <ul className="list-unstyled mb-0">
          {monitoring.map((m, i) => (
            <li key={i} className="mb-1 small">
              <span className="me-1" style={{ color: ACCENT4 }}>▸</span>{m}
            </li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="Clinical Thresholds & Action Points" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
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

      <SectionCard title="Disease Lifecycle (6 Stages)" borderColor={ACCENT}>
        {lifecycle.map((l, i) => (
          <div key={i} className="mb-3 pb-3 border-bottom">
            <div className="fw-bold small" style={{ color: ACCENT }}>{l.stage}</div>
            <div className="small text-muted">{l.features}</div>
            <div className="small mt-1"><strong>Action:</strong> {l.action}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Treatments Tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <>
      <Alert
        text="ℹ LEV = first-line AED (all RCDP2 forms). ACTH Level B for IS (avoid VGB due to cataracts). Alkylglycerol supplementation is MOST PHARMACOLOGICALLY DIRECT in RCDP2 (bypasses blocked GNPAT Step 1). DHA Level C. Phytol-restricted diet less critical (phytanic usually normal). No ERT / No HSCT."
        variant="info"
      />
      <SectionCard title="Treatments (AEDs + Supplementation + Experimental)" borderColor={ACCENT4}>
        {data.treatments?.map((t, i) => (
          <TreatmentCard key={i} {...t} />
        ))}
      </SectionCard>
      <SectionCard title="Contraindications & Hazards" borderColor={ACCENT2}>
        {data.contraindications?.map((c, i) => (
          <CICard key={i} {...c} />
        ))}
      </SectionCard>
    </>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading definitions…</div>;
  return (
    <>
      <SectionCard title="Key Concepts (15)" borderColor={ACCENT}>
        <ul className="list-unstyled mb-0">
          {data.key_concepts?.map((c, i) => (
            <li key={i} className="mb-2 small">
              <span className="me-1" style={{ color: ACCENT }}>▸</span>{c}
            </li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="12-Step Diagnostic Algorithm" borderColor={ACCENT4}>
        <ol className="mb-0">
          {data.diagnostic_algorithm?.map((step, i) => (
            <li key={i} className="small mb-2">{step}</li>
          ))}
        </ol>
      </SectionCard>

      <SectionCard title="Pharmacological Distinctions (12 Points)" borderColor={ACCENT2}>
        <ol className="mb-0">
          {data.pharmacological_distinctions?.map((p, i) => (
            <li key={i} className="small mb-2">{p}</li>
          ))}
        </ol>
      </SectionCard>

      <SectionCard title="Differential Diagnosis (7 Conditions)" borderColor={ACCENT3}>
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

      <SectionCard title="Reference Standards" borderColor={ACCENT4}>
        <ul className="mb-0">
          {data.standards?.map((s, i) => <li key={i} className="small mb-1">{s}</li>)}
        </ul>
      </SectionCard>
    </>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function GNPATPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/gnpat/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3) {
      fetch(`${API}/api/gnpat/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/gnpat/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4">
        <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
          GNPAT Epilepsy — Rhizomelic Chondrodysplasia Punctata Type 2 (RCDP2)
        </h2>
        <p className="text-muted small mb-2">
          GNPAT / DHAPAT (1q42.2) · Glyceronephosphate O-Acyltransferase · Step 1 plasmalogen synthesis ·
          PTS1 import (-SKF-) NOT PTS2 · AR biallelic LOF · ~5% of RCDP ·
          Plasmalogens (RBC) SEVERELY LOW · Phytanic NORMAL (PEX7 intact — KEY DISTINCTION from RCDP1) ·
          VLCFA NORMAL · No founder mutation · Cataracts 65% · Rhizomelia 100% · Stippling 85% ·
          Seizures 60% classic · VGB HIGH RISK (cataracts + VF) ·
          VPA RELATIVE CI · LEV first-line · ACTH Level B IS ·
          Alkylglycerols MOST DIRECT bypass (Step 1) · No ERT · No HSCT · 40 patients
        </p>
        {err && <div className="alert alert-danger small">{err}</div>}
      </div>

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
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
