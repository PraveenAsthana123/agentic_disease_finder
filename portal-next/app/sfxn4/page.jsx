'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Anemia', 'Hematology & Iron', 'Treatments', 'Definitions'];
const COLOR = '#880e4f';   // deep magenta — SFXN4/MDDS8B (sideroblastic anemia; hematological dominant)
const LIGHT = '#fce4ec';

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
  const bg = variant === 'danger' ? '#ffebee' : variant === 'warning' ? '#fff8e1' : variant === 'success' ? '#e8f5e9' : LIGHT;
  const border = variant === 'danger' ? '#c62828' : variant === 'warning' ? '#f57f17' : variant === 'success' ? '#2e7d32' : COLOR;
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
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function Spinner() {
  return <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /></div>;
}

// ── Tab 1: Overview ──────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="Gene & Disease Identity">
        <div className="row g-2 small">
          {[
            ['Gene', data.gene],
            ['Protein', data.protein],
            ['Disease', data.disease],
            ['OMIM Gene', data.omim_gene],
            ['OMIM Disease', data.omim_disease],
            ['Locus', data.chromosome],
            ['Inheritance', data.inheritance],
            ['Onset', data.onset],
          ].map(([k, v]) => (
            <div key={k} className="col-12 col-md-6">
              <span className="fw-semibold">{k}:</span> <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
        <div className="mt-3 p-2 rounded small" style={{ background: LIGHT }}>
          <strong>Cardinal Feature:</strong> Sideroblastic Anemia (ring sideroblasts on Perl&apos;s stain) — 100% PATHOGNOMONIC among MDDS series. The only MDDS gene causing sideroblastic anemia as a disease-defining feature.
        </div>
      </SectionCard>

      <SectionCard title="Contraindications — NEVER Use in SFXN4 / MDDS8B">
        <Alert variant="danger" text="⛔ VPA (Valproic Acid) — ABSOLUTE CI: CoA sequestration by valproyl-CoA + mtDNA depletion aggravation + hepatotoxicity in ALL mitochondrial disease; fatal liver failure documented across MDDS series" />
        <Alert variant="danger" text="⛔ KD (Ketogenic Diet) — CONTRAINDICATED: pan-OXPHOS deficiency (CI+CIII+CIV reduced); KD forces OXPHOS-dependent beta-oxidation that MDDS8B cannot sustain → energy failure in muscle/heart" />
        <Alert variant="danger" text="⛔ Propofol — AVOID (PRIS): inhibits Complex I + uncouples beta-oxidation → fatal lactic acidosis + cardiac failure in mitochondrial disease; use sevoflurane or ketamine instead" />
        <Alert variant="warning" text="⚠ Pyridoxine (B6) monotherapy — NOT effective in SFXN4 (B6-nonresponsive); empirical trial ≤4-6 weeks to exclude ALAS2/X-linked; do NOT delay diagnosis" />
        <Alert variant="warning" text="⚠ HSCT — Corrects hematological disease (anemia) but does NOT cure mitochondrial disease in muscle/brain/heart/liver (contrast with TYMP/MNGIE where HSCT cures enzyme deficiency)" />
      </SectionCard>

      <SectionCard title="Clinical KPIs — Synthetic Cohort (n=40, seed-569)">
        <div className="row">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Phenotype Frequency">
        {(data.phenotype_bars || []).map((b, i) => (
          <Bar key={i} label={b.label} value={b.pct} />
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers">
        <div className="row g-2">
          {(data.trigger_distribution || []).map((t, i) => (
            <div key={i} className="col-12 col-md-6">
              <Bar label={t.label} value={t.pct} />
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Key DDx — Distinguishing SFXN4/MDDS8B from Other Sideroblastic Anemias">
        <Alert variant="success" text="✅ SFXN4/MDDS8B = ONLY sideroblastic anemia + OXPHOS deficiency + mtDNA depletion + lactic acidosis — ring sideroblasts + RCE deficiency + high lactate = SFXN4 until proven otherwise" />
        <div className="small text-muted">Key differentials: ALAS2 (X-linked, B6-responsive, NO OXPHOS) · Pearson (mtDNA deletion, pancreatic exocrine insufficiency) · SFXN1 (isolated anemia, NO OXPHOS) · SLC25A38 (AR, glycine transporter) · GLRX5 (ISC defect) · RRM2B/MDDS8A (no sideroblasts, Fanconi 52%)</div>
      </SectionCard>

      <SectionCard title="Disease Lifecycle Stages">
        {(data.lifecycle || []).map((stage, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff', borderLeft: `3px solid ${COLOR}` }}>
            <div className="fw-bold small" style={{ color: COLOR }}>{stage.stage}</div>
            <ul className="small text-muted mb-0 mt-1 ps-3">
              {(stage.events || []).map((e, j) => <li key={j}>{e}</li>)}
            </ul>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Standards">
        <ol className="small ps-3 mb-0">
          {(data.standards || []).map((s, i) => <li key={i} className="mb-1">{s}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Anemia ─────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <Spinner />;
  const pts = data.all_patients || [];
  return (
    <div>
      <SectionCard title={`Patient Registry — All ${pts.length} Patients (seed-569)`}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ backgroundColor: COLOR, color: '#fff' }}>
              <tr>
                <th>ID</th>
                <th>Etiology</th>
                <th>Onset (mo)</th>
                <th>Hgb (g/dL)</th>
                <th>Transfusion Dep.</th>
                <th>Anemia Type</th>
                <th>Iron Overload</th>
                <th>Lactate (mmol/L)</th>
                <th>CK (×ULN)</th>
                <th>DCM</th>
                <th>Seizures</th>
                <th>Hepatopathy</th>
              </tr>
            </thead>
            <tbody>
              {pts.map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold" style={{ color: COLOR }}>{p.id}</td>
                  <td className="text-muted" style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.etiology}</td>
                  <td>{p.age_onset_months}</td>
                  <td className={p.hgb_at_diagnosis_gdl < 7 ? 'text-danger fw-bold' : ''}>{p.hgb_at_diagnosis_gdl}</td>
                  <td>{p.transfusion_dependent ? '✓' : '—'}</td>
                  <td>{p.anemia_type.replace('Sideroblastic', 'SA').replace('-', ' ')}</td>
                  <td>{p.iron_overload_secondary ? '⚠' : '—'}</td>
                  <td className={p.lactate_mmol > 5 ? 'text-danger' : ''}>{p.lactate_mmol}</td>
                  <td>{p.ck_x_uln}</td>
                  <td>{p.cardiomyopathy ? '✓' : '—'}</td>
                  <td>{p.seizures ? p.seizure_types.join(', ') : '—'}</td>
                  <td>{p.hepatopathy ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Etiology / Variant Class Distribution">
        {(data.etiology_distribution || (
          // fallback: compute from patients
          Object.entries(pts.reduce((acc, p) => { acc[p.etiology] = (acc[p.etiology] || 0) + 1; return acc; }, {}))
            .sort((a, b) => b[1] - a[1])
            .map(([label, count]) => ({ label, count, pct: Math.round(count / pts.length * 100) }))
        )).map((e, i) => (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span className="fw-semibold">{e.label}</span>
              <span className="text-muted">{e.count} / {pts.length} ({e.pct}%)</span>
            </div>
            <div className="progress" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: COLOR }} />
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Metabolic Summary">
        {data.metabolic_summary && (
          <div className="row g-2 small">
            {[
              ['Avg Lactate (lactic pts)', `${data.metabolic_summary.avg_lactate_lactic} mmol/L`],
              ['Avg Hgb at Dx', `${data.metabolic_summary.avg_hgb} g/dL`],
              ['Avg CK (myopathic pts)', `${data.metabolic_summary.avg_ck_myopathy}× ULN`],
              ['Normocytic SA', `${data.metabolic_summary.pct_anemia_normocytic}%`],
              ['Macrocytic SA', `${data.metabolic_summary.pct_anemia_macrocytic}%`],
              ['Microcytic SA', `${data.metabolic_summary.pct_anemia_microcytic}%`],
            ].map(([k, v]) => (
              <div key={k} className="col-12 col-md-4">
                <div className="p-2 rounded" style={{ background: LIGHT }}>
                  <div className="text-muted">{k}</div>
                  <div className="fw-bold" style={{ color: COLOR }}>{v}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Hematology & Iron ─────────────────────────────────────────────────
function HematologyTab({ data }) {
  if (!data) return <Spinner />;
  const { differentials = [] } = data;
  return (
    <div>
      <SectionCard title="Sideroblastic Anemia — Bone Marrow Findings">
        <Alert variant="danger" text="🔴 Ring Sideroblasts on Perl's Iron Stain (Prussian Blue) — PATHOGNOMONIC in SFXN4/MDDS8B; ≥15% of bone marrow erythroblasts; iron-laden perinuclear mitochondria; diagnostic before genetic confirmation" />
        <div className="small text-muted mt-2">
          <strong>Mechanism:</strong> SFXN4 LOF → serine import failure → glycine deficiency in mitochondria → ALAS cannot condense glycine + succinyl-CoA → δ-ALA (first heme synthesis step) → heme synthesis failure → iron accumulates in perinuclear mitochondria → ring sideroblasts
        </div>
        <div className="row g-3 mt-2">
          {[
            { label: 'Anemia Type', value: 'Sideroblastic — B6-Nonresponsive', color: '#880e4f' },
            { label: 'MCV', value: 'Variable (Normo/Macro/Micro)', color: '#ad1457' },
            { label: 'Serum Iron', value: 'HIGH (180-380 μg/dL)', color: '#c2185b' },
            { label: 'Transferrin Sat', value: 'HIGH (65-98%)', color: '#d81b60' },
            { label: 'Ferritin', value: 'HIGH (transfusion-related)', color: '#e91e63' },
            { label: 'Ring Sideroblasts', value: '≥15% (DIAGNOSTIC)', color: '#880e4f' },
          ].map(({ label, value, color }) => (
            <div key={label} className="col-12 col-md-4">
              <div className="p-2 rounded text-center" style={{ background: LIGHT }}>
                <div className="text-muted small">{label}</div>
                <div className="fw-bold" style={{ color }}>{value}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Iron Overload Management">
        <Alert variant="warning" text="⚠ Secondary iron overload from repeated transfusions — chelation mandatory once ferritin >1000 μg/L or LIC >3 mg/g dry weight" />
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Chelator</th><th>Route</th><th>Dose</th><th>Key Monitoring</th></tr></thead>
            <tbody>
              <tr>
                <td className="fw-semibold" style={{ color: COLOR }}>Deferasirox (Exjade/Jadenu)</td>
                <td>Oral</td>
                <td>10-30 mg/kg/day once daily</td>
                <td>Creatinine weekly×4 then monthly; LFTs; audiometry; ophthalmology annually</td>
              </tr>
              <tr>
                <td className="fw-semibold" style={{ color: COLOR }}>Deferoxamine (Desferal)</td>
                <td>SC/IV 8-12h nightly</td>
                <td>20-40 mg/kg/day (5-7 d/wk)</td>
                <td>Audiometry 6-monthly; ophthalmology annually; stop during febrile illness (Yersinia risk)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Cardiac Summary — SFXN4 vs SLC25A4 (MDDS2)">
        <Alert variant="success" text="KEY DDx: SFXN4/MDDS8B → Dilated Cardiomyopathy (DCM, systolic dysfunction) | SLC25A4/ANT1-MDDS2 → Hypertrophic Cardiomyopathy (HCM, 100%)" />
        {data.cardiac_summary && (
          <div className="row g-2 small">
            {[
              ['Cardiomyopathy Prevalence', `${data.cardiac_summary.pct_cardiomyopathy}%`],
              ['Dominant Type', data.cardiac_summary.dominant_type],
              ['Key Distinction', data.cardiac_summary.vs_mdds2_slc25a4],
            ].map(([k, v]) => (
              <div key={k} className="col-12">
                <span className="fw-semibold">{k}:</span> <span className="text-muted">{v}</span>
              </div>
            ))}
          </div>
        )}
      </SectionCard>

      <SectionCard title="Key Differentials — Sideroblastic Anemia">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ backgroundColor: COLOR, color: '#fff' }}>
              <tr>
                <th>Condition</th>
                <th>Gene</th>
                <th>Key Difference from SFXN4</th>
                <th>Distinguishing Test</th>
              </tr>
            </thead>
            <tbody>
              {differentials.map((d, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ color: COLOR }}>{d.condition}</td>
                  <td className="text-muted">{d.gene}</td>
                  <td>{d.key_difference}</td>
                  <td className="text-muted">{d.distinguishing_test}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Treatments ────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <Spinner />;
  const { treatments = [] } = data;
  return (
    <div>
      <Alert variant="danger" text="⛔ REMEMBER: VPA ABSOLUTE CI · KD CONTRAINDICATED · Propofol AVOID (PRIS) · Pyridoxine NOT effective in SFXN4 — apply universally in MDDS8B" />
      {treatments.map((t, i) => (
        <SectionCard key={i} title={`${t.name} — ${t.tier} [${t.evidence}]`}>
          <div className="small">
            <p><strong>Mechanism:</strong> {t.mechanism}</p>
            <p><strong>Dose:</strong> {t.dose}</p>
            <p><strong>Monitoring:</strong> {t.monitoring}</p>
            {t.caution && <Alert variant="warning" text={`⚠ ${t.caution}`} />}
          </div>
        </SectionCard>
      ))}
    </div>
  );
}

// ── Tab 5: Definitions ───────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Spinner />;
  const sections = [
    { key: 'gene_concepts', title: 'Gene & Protein Concepts' },
    { key: 'disease_concepts', title: 'Disease & Clinical Concepts' },
    { key: 'diagnostic_concepts', title: 'Diagnostic Concepts' },
    { key: 'pharmacology', title: 'Pharmacology' },
    { key: 'thresholds', title: 'Clinical Thresholds & Action Points' },
  ];
  return (
    <div>
      {sections.map(({ key, title }) => (
        <SectionCard key={key} title={title}>
          {key === 'thresholds' ? (
            <div className="table-responsive">
              <table className="table table-sm small">
                <thead><tr><th>Threshold</th><th>Action</th></tr></thead>
                <tbody>
                  {(data[key] || []).map((t, i) => (
                    <tr key={i}>
                      <td className="fw-semibold" style={{ color: COLOR }}>{t.threshold}</td>
                      <td>{t.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            (data[key] || []).map((item, i) => (
              <div key={i} className="mb-3">
                <div className="fw-semibold small" style={{ color: COLOR }}>{item.term}</div>
                <div className="small text-muted mt-1">{item.definition}</div>
                {i < (data[key].length - 1) && <hr className="my-2" />}
              </div>
            ))
          )}
        </SectionCard>
      ))}
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────────
export default function SFXN4Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/sfxn4/overview`).then(r => r.json()).then(setOverview).catch(() => setError('Overview load failed'));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2 || tab === 3) {
      fetch(`${API}/api/sfxn4/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setError('Breakdown load failed'));
    }
    if (tab === 4) {
      fetch(`${API}/api/sfxn4/definitions`).then(r => r.json()).then(setDefs).catch(() => setError('Definitions load failed'));
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          🩸 SFXN4 / Sideroflexin-4 — Sideroblastic Anemia Mitochondrial DNA Depletion Syndrome 8B (MDDS8B)
        </h4>
        <div className="text-muted small">
          OMIM Gene: 615564 · OMIM Disease: 615081 · 10q26.11 · 322 aa · 5-TM IMM Serine Transporter ·
          AR biallelic LOF → Sideroblastic Anemia 100% + OXPHOS deficiency + mtDNA depletion · Seed-569 · n=40 synthetic cohort
        </div>
        <div className="mt-2 p-2 rounded small fw-bold" style={{ background: '#fce4ec', border: '1px solid #880e4f', color: '#880e4f' }}>
          ⛔ VPA ABSOLUTE CI · ⛔ KD CONTRAINDICATED · ⛔ Propofol AVOID (PRIS) · ⚠ Pyridoxine B6 NOT effective (B6-nonresponsive) · ⚠ HSCT corrects anemia NOT mtDNA disease
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR, fontWeight: 600 } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <HematologyTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={defs} />}
    </div>
  );
}
