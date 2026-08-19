'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a6b3c';   // dark forest green — oligosaccharide storage / metabolic
const ACCENT2 = '#c62828';   // dark red — HIGH RISK / contraindications
const ACCENT3 = '#e65100';   // deep orange — CAUTION / monitor
const ACCENT4 = '#1565c0';   // dark blue — ERT approved / velmanase alfa / Level A
const ACCENT5 = '#4a148c';   // deep purple — HSCT / CNS treatment
const ACCENT6 = '#00695c';   // teal — SAFE / CBZ safe distinction / IVIG

const ETIOLOGY_COLORS = {
  'Compound Heterozygous — Missense + Missense': '#1a6b3c',
  'Homozygous Missense (Consanguineous / Founder)': '#2e7d32',
  'Compound Heterozygous — Null + Missense': '#e65100',
  'Homozygous Null / Biallelic Truncating': '#c62828',
  'Attenuated / Late-Onset (Mild Type I Phenotype)': '#00695c',
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

function CICard({ item }) {
  const riskColor = item.risk?.includes('HIGH RISK') ? ACCENT2
    : item.risk?.includes('RELATIVE CI') ? ACCENT3
    : item.risk?.includes('CAUTION') ? '#f57c00'
    : item.risk?.includes('MONITOR') ? ACCENT6
    : '#546e7a';
  return (
    <div className="col-md-6 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-header text-white small fw-bold" style={{ background: riskColor }}>
          {item.drug} — {item.risk}
        </div>
        <div className="card-body small">
          <p className="mb-1"><strong>Mechanism:</strong> {item.mechanism}</p>
          <p className="mb-1 text-success"><strong>Alternative:</strong> {item.alternative}</p>
          <p className="mb-0 text-muted"><em>{item.evidence}</em></p>
        </div>
      </div>
    </div>
  );
}

function TreatmentCard({ item }) {
  const lvlColor = item.level === 'A' ? ACCENT4 : item.level === 'B' ? ACCENT : ACCENT3;
  return (
    <div className="col-md-6 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-header text-white small fw-bold" style={{ background: lvlColor }}>
          Level {item.level} — {item.treatment}
        </div>
        <div className="card-body small">
          <p className="mb-1"><strong>Indication:</strong> {item.indication}</p>
          <p className="mb-1"><strong>Mechanism:</strong> {item.mechanism}</p>
          <p className="mb-1 text-info"><strong>Monitoring:</strong> {item.monitoring}</p>
          {item.caution && <p className="mb-0 text-warning"><strong>Caution:</strong> {item.caution}</p>}
        </div>
      </div>
    </div>
  );
}

// ── Overview Tab ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading…</div>;
  return (
    <>
      <div className="alert mb-4 text-white fw-bold" style={{ background: ACCENT }}>
        🧬 MAN2B1 / Alpha-Mannosidosis — Lysosomal Acid Alpha-Mannosidase Deficiency
        &nbsp;|&nbsp; {data.locus} &nbsp;|&nbsp; {data.inheritance}
      </div>

      {/* Pathognomonic box */}
      <div className="alert border-2 mb-4" style={{ borderColor: ACCENT4, background: '#e3f2fd' }}>
        <strong style={{ color: ACCENT4 }}>PATHOGNOMONIC FINDINGS:</strong>
        <div className="mt-1 small">{data.pathognomonic_note}</div>
      </div>

      {/* CBZ safe distinction box */}
      <div className="alert border-2 mb-4" style={{ borderColor: ACCENT6, background: '#e0f2f1' }}>
        <strong style={{ color: ACCENT6 }}>KEY PHARMACOLOGICAL DISTINCTION — CBZ/OXC SAFE:</strong>
        <div className="mt-1 small">
          Alpha-mannosidosis does NOT have demyelinating peripheral neuropathy → CBZ/OXC are SAFE
          (unlike MLD/ARSA/GALC/SUMF1 where CBZ/OXC = ABSOLUTE CIs). HLA-B*15:02 testing mandatory
          before CBZ/OXC in SE Asian patients (SJS/TEN). Standard monitoring applies.
        </div>
      </div>

      {/* KPI row */}
      <div className="row g-2 mb-4">
        <KPI label="Cohort" value={data.cohort_size} color={ACCENT} />
        <KPI label="Seizures overall" value={`${data.seizure_pct_overall}%`} color={ACCENT2} />
        <KPI label="Severe phenotype" value={`${data.seizure_pct_severe}%`} color={ACCENT2} />
        <KPI label="Drug-resistant" value={`${data.drug_resistant_pct}%`} color={ACCENT3} />
        <KPI label="Hearing loss" value={`${data.hearing_loss_pct}%`} color={ACCENT5} />
        <KPI label="Infections" value={`${data.infections_pct}%`} color={ACCENT3} />
        <KPI label="Psychiatric" value={`${data.psychiatric_features_pct}%`} color={ACCENT5} />
        <KPI label="Ataxia" value={`${data.ataxia_pct}%`} color={ACCENT3} />
        <KPI label="Vacuol. lymph." value={`${data.vacuolated_lymphocytes_pct}%`} color={ACCENT4} />
        <KPI label="On ERT" value={`${data.on_ert_pct}%`} color={ACCENT4} />
        <KPI label="Dx delay" value={`${data.mean_diagnosis_delay_years}y`} color={ACCENT2} />
        <KPI label="Febrile clusters" value={`${data.febrile_cluster_pct}%`} color={ACCENT3} />
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Disease Overview" color={ACCENT}>
            <p className="small mb-0">{data.disease}</p>
          </SectionCard>
          <SectionCard title="Protein & Mechanism" color={ACCENT4}>
            <p className="small mb-1"><strong>Protein:</strong> {data.protein}</p>
            <p className="small mb-0"><strong>Mechanism:</strong> {data.mechanism}</p>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Diagnostic Hierarchy" color={ACCENT5}>
            <p className="small mb-0" style={{ whiteSpace: 'pre-line' }}>{data.diagnostic_hierarchy}</p>
          </SectionCard>
          <SectionCard title="AED Distribution" color={ACCENT6}>
            <PctBar label="On LEV" pct={data.on_lev_pct} color={ACCENT} />
            <PctBar label="On VPA" pct={data.on_vpa_pct} color={ACCENT4} />
            <PctBar label="On CBZ (SAFE)" pct={data.on_cbz_pct} color={ACCENT6} />
            <PctBar label="On ERT (velmanase alfa)" pct={data.on_ert_pct} color={ACCENT4} />
            <PctBar label="On IVIG" pct={data.on_ivig_pct} color={ACCENT5} />
          </SectionCard>
        </div>
      </div>

      {data.standards && (
        <SectionCard title="Standards & References" color="#546e7a">
          <div className="d-flex flex-wrap gap-1">
            {data.standards.map((s, i) => <Badge key={i} text={s} color="#546e7a" />)}
          </div>
        </SectionCard>
      )}
    </>
  );
}

// ── Patients & Etiology Tab ───────────────────────────────────────────────────
function EtiologyTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading…</div>;
  const { etiologies = [] } = data;
  return (
    <>
      <SectionCard title="Genotype Distribution — 40 Patients" color={ACCENT}>
        {etiologies.map((e, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f9fbe7' }}>
            <div className="d-flex justify-content-between">
              <span className="fw-bold small" style={{ color: ETIOLOGY_COLORS[e.name] || ACCENT }}>{e.name}</span>
              <Badge text={`${e.pct}%`} color={ETIOLOGY_COLORS[e.name] || ACCENT} />
            </div>
            <div className="progress my-1" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, background: ETIOLOGY_COLORS[e.name] || ACCENT }} />
            </div>
            <p className="small mb-1 text-muted">{e.onset}</p>
            <p className="small mb-1">{e.notes}</p>
            <p className="small mb-0 fw-bold" style={{ color: ACCENT4 }}>Key finding: {e.key_finding}</p>
          </div>
        ))}
      </SectionCard>

      {data.subtype_severity_matrix && (
        <SectionCard title="Phenotype Severity Matrix" color={ACCENT5}>
          {Object.entries(data.subtype_severity_matrix || {}).map(([k, v], i) => (
            <div key={i} className="mb-2 small"><strong>{k}:</strong> {v}</div>
          ))}
        </SectionCard>
      )}

      {data.ert_vs_hsct_comparison && (
        <SectionCard title="ERT vs HSCT — Treatment Comparison" color={ACCENT4}>
          <div className="row">
            {Object.entries(data.ert_vs_hsct_comparison).map(([k, v], i) => (
              typeof v === 'object' && !Array.isArray(v) ? (
                <div key={i} className="col-md-6 mb-3">
                  <div className="card h-100 border-0 shadow-sm">
                    <div className="card-header fw-bold text-white small"
                      style={{ background: k === 'velmanase_alfa' ? ACCENT4 : k === 'hsct' ? ACCENT5 : ACCENT6 }}>
                      {k === 'velmanase_alfa' ? 'Velmanase Alfa (ERT)' : k === 'hsct' ? 'HSCT' : 'Combination'}
                    </div>
                    <div className="card-body small">
                      {Object.entries(v).map(([field, val], j) => (
                        <p key={j} className="mb-1"><strong>{field.replace(/_/g, ' ')}:</strong> {val}</p>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div key={i} className="col-12 small mb-2">
                  <strong>Combination:</strong> {v}
                </div>
              )
            ))}
          </div>
        </SectionCard>
      )}
    </>
  );
}

// ── Seizures & Triggers Tab ───────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading…</div>;
  const { seizure_types = [], triggers = [] } = data;
  return (
    <>
      <SectionCard title="Seizure Types" color={ACCENT2}>
        {seizure_types.map((s, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between align-items-center">
              <span className="fw-bold small">{s.type}</span>
              <Badge text={`${s.pct}%`} color={ACCENT2} />
            </div>
            <div className="progress my-1" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${s.pct}%`, background: ACCENT2 }} />
            </div>
            <p className="small text-muted mb-0">{s.subtype}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers" color={ACCENT3}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between align-items-center">
              <span className="fw-bold small">{t.trigger}</span>
              <Badge text={`${t.pct}%`} color={ACCENT3} />
            </div>
            <div className="progress my-1" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${t.pct}%`, background: ACCENT3 }} />
            </div>
            <p className="small text-muted mb-0">{t.notes}</p>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Treatments Tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading…</div>;
  const { treatments = [], contraindications = [], treatment_hierarchy = [] } = data;
  return (
    <>
      {treatment_hierarchy.length > 0 && (
        <SectionCard title="Treatment Hierarchy" color={ACCENT4}>
          <ol className="mb-0 small">
            {treatment_hierarchy.map((h, i) => <li key={i} className="mb-1">{h}</li>)}
          </ol>
        </SectionCard>
      )}

      <SectionCard title="Approved & Recommended Treatments" color={ACCENT}>
        <div className="row">
          {treatments.map((t, i) => <TreatmentCard key={i} item={t} />)}
        </div>
      </SectionCard>

      <SectionCard title="Drug Safety — Contraindications & Cautions" color={ACCENT2}>
        <div className="row">
          {contraindications.map((c, i) => <CICard key={i} item={c} />)}
        </div>
      </SectionCard>

      {data.thresholds && (
        <SectionCard title="Clinical Thresholds" color={ACCENT5}>
          <ul className="mb-0 small">
            {data.thresholds.map((t, i) => <li key={i} className="mb-1">{t}</li>)}
          </ul>
        </SectionCard>
      )}
    </>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading…</div>;
  const { diagnostic_algorithm = [], man2b1_glossary = {}, key_concepts = [], standards = [], differential_diagnosis = {} } = data;
  return (
    <>
      <SectionCard title="Diagnostic Algorithm (8 Steps)" color={ACCENT4}>
        <ol className="mb-0 small">
          {diagnostic_algorithm.map((step, i) => <li key={i} className="mb-2">{step}</li>)}
        </ol>
      </SectionCard>

      {key_concepts.length > 0 && (
        <SectionCard title="Key Concepts" color={ACCENT}>
          {key_concepts.map((c, i) => (
            <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f1f8e9' }}>
              <div className="fw-bold small" style={{ color: ACCENT }}>{c.term}</div>
              <p className="small mb-0 mt-1">{c.definition}</p>
            </div>
          ))}
        </SectionCard>
      )}

      {Object.keys(man2b1_glossary).length > 0 && (
        <SectionCard title="MAN2B1 / Alpha-Mannosidosis Glossary" color={ACCENT5}>
          <dl className="row small mb-0">
            {Object.entries(man2b1_glossary).map(([term, def], i) => (
              <div key={i} className="col-md-6 mb-2">
                <dt style={{ color: ACCENT5 }}>{term}</dt>
                <dd className="mb-0">{def}</dd>
              </div>
            ))}
          </dl>
        </SectionCard>
      )}

      {Object.keys(differential_diagnosis).length > 0 && (
        <SectionCard title="Differential Diagnosis" color={ACCENT3}>
          {Object.entries(differential_diagnosis).map(([k, v], i) => (
            <div key={i} className="mb-3 p-2 rounded" style={{ background: '#fff3e0' }}>
              <div className="fw-bold small" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}</div>
              <p className="small mb-0 mt-1">{v}</p>
            </div>
          ))}
        </SectionCard>
      )}

      {standards.length > 0 && (
        <SectionCard title="Standards & References" color="#546e7a">
          <div className="d-flex flex-wrap gap-1">
            {standards.map((s, i) => <Badge key={i} text={s} color="#546e7a" />)}
          </div>
        </SectionCard>
      )}
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function Man2b1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/man2b1/overview`).then(r => r.json()).then(setOverview).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2 || tab === 3) {
      fetch(`${API}/api/man2b1/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/man2b1/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
        🧬 MAN2B1 — Alpha-Mannosidosis (Lysosomal Acid Alpha-Mannosidase Deficiency)
      </h2>
      <p className="text-muted small mb-3">
        {OMIM_DISPLAY} · {LOCUS_DISPLAY} · AR biallelic LOF · Velmanase alfa (Lamzede) EMA 2018 / FDA 2023 · HSCT CNS-preferred · 40-patient cohort
      </p>

      {err && <div className="alert alert-danger">API error: {err}</div>}

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { borderBottomColor: ACCENT, color: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
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

const OMIM_DISPLAY = 'OMIM 248500 / 609458';
const LOCUS_DISPLAY = '19p13.2';
