'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#2e7d32';   // deep green — FAR1 fatty-alcohol reductase / RCDP4 / rare
const ACCENT2 = '#b71c1c';   // dark red — HIGH RISK / VGB cataracts / VPA hepatotoxicity
const ACCENT3 = '#e65100';   // deep orange — RELATIVE CI / caution / thresholds
const ACCENT4 = '#1565c0';   // deep blue — safe treatments / LEV / first-line
const ACCENT5 = '#1b5e20';   // dark green — NORMAL markers (VLCFA, phytanic) / positive findings
const ACCENT6 = '#6a1b9a';   // purple — alkylglycerol / experimental / ether-bond bypass

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
    ? ACCENT2 : level?.includes('RELATIVE') ? ACCENT3
    : level?.includes('NOT APPLICABLE') ? '#616161' : ACCENT4;
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

// ── Overview Tab ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading overview…</div>;
  const d = data;
  return (
    <div>
      <Alert
        text="🚨 CRITICAL DISTINCTION FROM RCDP1/PEX7: Phytanic acid is NORMAL in RCDP4/FAR1 (PEX7 is intact → PHYH imported via PTS2 normally). RCDP1 phytanic ELEVATED; RCDP4 phytanic NORMAL. Both have plasmalogens severely low. VLCFA NORMAL in both (distinguishes from ZSD)."
        variant="danger"
      />
      <Alert
        text="🚨 VGB — HIGH RISK: Cataracts present in 55-62% of RCDP4 + VGB irreversible visual field constriction = additive visual impairment. NOT absolute CI (unlike ZSD where retinopathy universal/severe), but HIGH RISK. ACTH preferred for IS. Monthly VF/VEP mandatory if VGB used."
        variant="danger"
      />
      <Alert
        text="⚠ VPA — RELATIVE CI: Hepatotoxicity risk. POLG1 MANDATORY (CPIC Grade A). Phytanic NORMAL in RCDP4 (less phytanic-driven hepatic stress vs RCDP1), but POLG1/mitochondrial risk persists. LFT q3 months if VPA used."
        variant="warning"
      />
      <Alert
        text={`ℹ FAR1 = Fatty Acyl-CoA Reductase 1 (553 aa PMP; NADPH-dependent). Provides fatty alcohol substrate for AGPS Step 2. FAR1 LOF → no fatty alcohol → AGPS has no substrate → ether bond cannot form. Alkylglycerols bypass FAR1 block (pre-formed ether lipid). PHT/CBZ/OXC CAN be used (no adrenal). No ERT / No HSCT. Rarest RCDP: ~15–20 cases worldwide.`}
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="Classic RCDP4 %" value={`${d.classic_rcdp_pct}%`} color={ACCENT2} />
        <KPI label="Intermediate %" value={`${d.intermediate_rcdp_pct}%`} color={ACCENT3} />
        <KPI label="Mild RCDP4 %" value={`${d.mild_rcdp_pct}%`} color={ACCENT5} />
        <KPI label="Drug Resistant %" value={`${d.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="Cataract %" value={`${d.cataract_pct}%`} color={ACCENT3} />
        <KPI label="Rhizomelia %" value={`${d.rhizomelia_pct}%`} color={ACCENT} />
        <KPI label="VLCFA NORMAL" value="100%" color={ACCENT5} />
        <KPI label="Plasmalogens LOW" value={`${d.plasmalogen_low_pct}%`} color={ACCENT2} />
        <KPI label="OMIM Gene" value={`*${d.omim_gene}`} color={ACCENT} />
        <KPI label="Locus" value={d.locus} color={ACCENT4} />
      </div>

      <SectionCard title="Disease Summary — FAR1 / RCDP4 (Rhizomelic Chondrodysplasia Punctata Type 4)" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-1"><strong>Variant Spectrum:</strong> {d.common_variant}</p>
        <p className="small mb-0">{d.disease_mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Profile" borderColor={ACCENT4}>
            <PctBar label="Seizures (overall cohort)" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Drug-resistant seizures" pct={d.drug_resistance_pct} color={ACCENT2} />
            <PctBar label="Classic RCDP4 (severe)" pct={d.classic_rcdp_pct} color={ACCENT2} />
            <PctBar label="Intermediate RCDP4" pct={d.intermediate_rcdp_pct} color={ACCENT3} />
            <PctBar label="Mild RCDP4" pct={d.mild_rcdp_pct} color={ACCENT5} />
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
        <Alert text="VGB — HIGH RISK (cataracts 55-62% + VF loss = additive). NOT absolute CI as in ZSD. Monthly VF/VEP monitoring mandatory if used. Prefer ACTH for IS." variant="danger" />
        <Alert text="VPA — RELATIVE CI (hepatotoxicity). Phytanic NORMAL in RCDP4 (less phytanic-hepatic stress vs RCDP1), but POLG1 MANDATORY (CPIC A). LFT q3 months." variant="warning" />
        <Alert text="PHT/CBZ/OXC — CAN BE USED (no adrenal insufficiency in RCDP4; no CYP3A4 cortisol mechanism). Contrast with ABCD1 where PHT = ABSOLUTE CI." variant="secondary" />
        <Alert text="LEV first-line all forms. ACTH Level B for IS. Alkylglycerols bypass FAR1 block (pre-formed ether lipid; experimental Level C). DHA Level C. No ERT / No HSCT." variant="info" />
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
    'Classic RCDP4': ACCENT2,
    'Intermediate RCDP4': ACCENT3,
    'Mild RCDP4': ACCENT5,
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
        text="ℹ RCDP4 SPECTRUM (FAR1): Classic (null/null → severe 38%) · Intermediate (null/hypomorphic → 42%) · Mild (hypomorphic/hypomorphic → 20%). No founder mutation — all private/rare variants. Phytanic NORMAL in all forms (PEX7 intact). Rarest molecularly defined RCDP subtype (~15–20 cases worldwide)."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>FAR1-RCDP4 Phenotypic Classes — 3 Forms (40 Patients)</h6>
      {etiologies?.map((e, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${getColor(e.name)}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-2">
              <h6 className="fw-bold mb-0" style={{ color: getColor(e.name) }}>{e.name}</h6>
              <span className="badge" style={{ backgroundColor: getColor(e.name) }}>{e.pct}% · n={e.n}</span>
            </div>
            <div className="row small">
              <div className="col-md-6">
                <p className="mb-1"><strong>Sex:</strong> {e.sex}</p>
                <p className="mb-1"><strong>Onset:</strong> {e.onset_age}</p>
                <p className="mb-1"><strong>Seizure risk:</strong> {e.seizure_risk}</p>
                <p className="mb-1"><strong>EEG:</strong> {e.eeg}</p>
                <p className="mb-1"><strong>MRI:</strong> {e.mri}</p>
              </div>
              <div className="col-md-6">
                <p className="mb-1 text-muted">{e.variant_detail}</p>
                <div className="mt-1">
                  <Badge text={e.dha_supplement ? 'DHA: Yes' : 'DHA: No'} color={ACCENT5} />
                  <Badge text={e.hsct_eligible ? 'HSCT: Yes' : 'HSCT: No'} color={e.hsct_eligible ? ACCENT4 : '#9e9e9e'} />
                  <Badge text={e.ert_available ? 'ERT: Available' : 'ERT: None'} color={e.ert_available ? ACCENT4 : '#9e9e9e'} />
                </div>
              </div>
            </div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4" style={{ color: ACCENT }}>Individual Patients (40 Synthetic — RCDP4-01 to RCDP4-40)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr>
              <th>ID</th><th>Phenotype</th><th>Sex</th><th>Onset (mo)</th>
              <th>Seizures</th><th>Sz Type</th><th>Drug Resistant</th>
              <th>Cataracts</th><th>Phytanic</th><th>Genotype</th>
            </tr>
          </thead>
          <tbody>
            {patients?.map((p, i) => (
              <tr key={i}>
                <td className="small fw-bold" style={{ color: ACCENT }}>{p.id}</td>
                <td className="small">{p.phenotype}</td>
                <td className="small">{p.sex}</td>
                <td className="small">{p.onset_age_months}</td>
                <td className="small">
                  <Badge text={p.has_seizures ? 'Yes' : 'No'} color={p.has_seizures ? ACCENT2 : ACCENT5} />
                </td>
                <td className="small">{p.seizure_type || '—'}</td>
                <td className="small">
                  {p.drug_resistant && <Badge text="DRE" color={ACCENT2} />}
                  {!p.drug_resistant && p.has_seizures && <Badge text="Controlled" color={ACCENT5} />}
                  {!p.has_seizures && <Badge text="Sz-free" color={ACCENT5} />}
                </td>
                <td className="small">
                  <Badge text={p.cataracts ? 'Yes' : 'No'} color={p.cataracts ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text="NORMAL" color={ACCENT5} />
                </td>
                <td className="small text-muted">{p.genotype}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Seizures & Triggers Tab ───────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types, triggers, monitoring, lifecycle } = data;
  return (
    <div>
      <Alert
        text="⚠ SEIZURE PROFILE RCDP4: Infantile spasms 35% (ACTH Level B; avoid VGB due to cataracts). Focal 32%. Myoclonic 16%. GTCS 12%. SE 6%. Drug resistance 28%. Plasmalogen deficiency impairs neuronal membrane function — identical mechanism to RCDP1/2/3."
        variant="warning"
      />

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Seizure Types (5 Types)" borderColor={ACCENT2}>
            {seizure_types?.map((s, i) => (
              <div key={i} className="mb-3">
                <PctBar label={s.type} pct={s.pct} color={ACCENT2} />
                <div className="small text-muted">
                  <strong>Preferred Rx:</strong> {s.preferred_tx}
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Seizure Triggers (7)" borderColor={ACCENT3}>
            {triggers?.map((t, i) => (
              <PctBar key={i} label={t.trigger} pct={t.pct} color={ACCENT3} />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Monitoring Parameters (7)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Parameter</th><th>Threshold / Target</th><th>Frequency</th></tr></thead>
            <tbody>
              {monitoring?.map((m, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{m.parameter}</td>
                  <td className="small text-muted">{m.threshold}</td>
                  <td className="small">{m.frequency}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Disease Lifecycle (6 Stages)" borderColor={ACCENT}>
        {lifecycle?.map((l, i) => (
          <div key={i} className="mb-2">
            <div className="fw-bold small" style={{ color: ACCENT }}>{l.stage}</div>
            <div className="small text-muted">{l.features}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Treatments Tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatments, contraindications } = data;
  return (
    <div>
      <Alert
        text="ℹ TREATMENT PRINCIPLES RCDP4: LEV first-line. ACTH Level B for IS (preferred over VGB — avoid adding to cataract risk). Alkylglycerols bypass FAR1 block — pre-formed ether lipid bypasses need for fatty alcohol (FAR1) and ether-bond step (AGPS). DHA Level C. No ERT, No HSCT. POLG1 MANDATORY before VPA."
        variant="info"
      />

      <SectionCard title="Approved & Supportive Treatments (6)" borderColor={ACCENT4}>
        {treatments?.map((t, i) => (
          <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
            <div className="card-body py-2 px-3">
              <div className="d-flex justify-content-between align-items-start mb-1">
                <span className="fw-bold small">{t.drug}</span>
                <div>
                  <Badge text={t.class} color={ACCENT4} />
                  <Badge text={t.level} color={ACCENT} />
                </div>
              </div>
              <p className="small text-muted mb-1"><strong>Dose:</strong> {t.dose}</p>
              <p className="small text-muted mb-1">{t.notes}</p>
              {t.ci !== 'None' && t.ci !== 'None specific' && t.ci !== 'None specific to RCDP4' && (
                <p className="small text-danger mb-0"><strong>CI:</strong> {t.ci}</p>
              )}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications & High-Risk Drugs (5)" borderColor={ACCENT2}>
        {contraindications?.map((c, i) => (
          <CICard key={i} drug={c.drug} level={c.level} reason={c.reason} alternative={c.alternative} />
        ))}
      </SectionCard>

      <SectionCard title="Experimental Plasmalogen Restoration" borderColor={ACCENT6}>
        <Alert text="🧪 ALKYLGLYCEROLS (FAR1-block bypass — RCDP4 primary rationale): FAR1 deficiency means no fatty alcohol → AGPS has no substrate. Alkylglycerols (batyl/chimyl/selachyl alcohol) are pre-formed ether lipids that bypass BOTH the fatty-alcohol requirement (FAR1) AND the ether-bond step (AGPS). Same experimental Level C mechanism as RCDP2/RCDP3. Human data limited (n<20 total across all RCDP subtypes)." variant="secondary" />
        <Alert text="🧪 DHA (Docosahexaenoic acid) supplementation: Restores DHA-plasmalogen; 200 mg/day infants. Safe, no drug interactions. Level C." variant="secondary" />
        <Alert text="❌ NO ERT (2026): FAR1 is a peroxisomal membrane protein (PMP; not secreted; not lysosomal). ERT cannot replace intracellular peroxisomal membrane enzymes — systemic enzyme delivery cannot cross the peroxisomal membrane as functional FAR1." variant="light" />
        <Alert text="❌ NO HSCT: RCDP4 pathology is hypomyelination (plasmalogen deficiency), not neuroinflammation. HSCT is indicated for inflammatory demyelination (ABCD1-CCALD, Krabbe) — completely irrelevant mechanism." variant="light" />
      </SectionCard>
    </div>
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
export default function FAR1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/far1/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3) {
      fetch(`${API}/api/far1/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/far1/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4">
        <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
          FAR1 Epilepsy — Rhizomelic Chondrodysplasia Punctata Type 4 (RCDP4)
        </h2>
        <p className="text-muted small mb-2">
          FAR1 / Fatty Acyl-CoA Reductase 1 (11p15.3) · 553 aa peroxisomal membrane protein (PMP) ·
          NADPH-dependent · Produces fatty alcohol for AGPS Step 2 ·
          AR biallelic LOF · ~15–20 cases worldwide 2026 ·
          Plasmalogens (RBC) SEVERELY LOW · Phytanic NORMAL (PEX7 intact — KEY DISTINCTION from RCDP1) ·
          VLCFA NORMAL · No founder mutation · Cataracts 58% · Rhizomelia 100% · Stippling 80% ·
          Seizures 55% · VGB HIGH RISK (cataracts + VF) ·
          VPA RELATIVE CI · LEV first-line · ACTH Level B IS ·
          Alkylglycerols bypass FAR1 block · No ERT · No HSCT · 40 patients
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
