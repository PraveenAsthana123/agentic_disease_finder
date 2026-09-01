'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Variants', 'Cardiac & Metabolic', 'Treatments', 'Definitions'];
const COLOR = '#b71c1c';   // deep red — SLC25A4/ANT1 (cardiomyopathic MDDS; heart failure dominant)
const LIGHT = '#ffebee';

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
            ['Alias', data.alias],
            ['Full Name', data.full_name],
            ['Disease', data.disease],
            ['OMIM Gene', data.omim_gene],
            ['OMIM Disease', data.omim_disease],
            ['Locus', data.locus],
            ['Protein', `${data.protein_length_aa} aa`],
            ['Inheritance', data.inheritance],
          ].map(([k, v]) => (
            <div key={k} className="col-12 col-md-6">
              <span className="fw-semibold">{k}:</span> <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
        <div className="mt-3 p-2 rounded small" style={{ background: LIGHT }}>
          <strong>Mechanism:</strong> {data.mechanism}
        </div>
      </SectionCard>

      <SectionCard title="Contraindications — NEVER Use in MDDS2">
        <Alert variant="danger" text="⛔ VPA (Valproic Acid) — ABSOLUTE CI: CoA sequestration + mtDNA depletion aggravation + hepatotoxicity in ALL mitochondrial disease" />
        <Alert variant="danger" text="⛔ KD (Ketogenic Diet) — CONTRAINDICATED: OXPHOS-dependent beta-oxidation fails in pan-OXPHOS deficiency; fat diet → cardiac + muscle energy failure" />
        <Alert variant="danger" text="⛔ Propofol — AVOID (PRIS): inhibits Complex I + beta-oxidation → fatal lactic acidosis + cardiac failure in mitochondrial disease" />
        <Alert variant="warning" text="⚠ ACE Inhibitors / ARB — CONTRAINDICATED if LVOTO gradient >30 mmHg (vasodilation worsens dynamic obstruction)" />
        <Alert variant="warning" text="⚠ Sodium Channel Blockers (CBZ/OXC) — RELATIVE CI if myoclonic seizures; OK for focal seizures without myoclonus" />
      </SectionCard>

      <SectionCard title="Clinical KPIs — Synthetic Cohort (n=40, seed-567)">
        <div className="row">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Key DDx — Distinguishing MDDS2 from Other MDDS">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead><tr><th>DDx Disease</th><th>Distinguishing Feature</th></tr></thead>
            <tbody>
              {(data.key_ddx || []).map((d, i) => {
                const [disease, ...rest] = d.split(' — ');
                return (
                  <tr key={i}>
                    <td className="fw-semibold" style={{ color: COLOR }}>{disease}</td>
                    <td>{rest.join(' — ')}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="References">
        <ol className="small ps-3 mb-0">
          {(data.references || []).map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Variants ───────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <Spinner />;
  const { etiologies = [], lifecycle = [] } = data;
  return (
    <div>
      <SectionCard title="Variant / Etiology Classes (n=40 synthetic cohort, seed-567)">
        {etiologies.map((e, i) => (
          <div key={i} className="mb-4">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="fw-semibold small">{e.class}</span>
              <span className="badge" style={{ backgroundColor: COLOR }}>{e.n}/{40} ({e.pct}%)</span>
            </div>
            <div className="progress mb-2" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: COLOR }} />
            </div>
            <div className="small text-muted mb-1"><strong>Severity:</strong> {e.severity}</div>
            <div className="small text-muted mb-1"><strong>Examples:</strong> {(e.examples || []).join('; ')}</div>
            <div className="small" style={{ background: LIGHT, borderLeft: `3px solid ${COLOR}`, padding: '6px 10px', borderRadius: 4 }}>{e.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="ANT1 AR MDDS2 vs ANT1 AD PEO2 — Critical Clinical Distinction">
        {data.ant1_vs_peo2 && (
          <div className="table-responsive">
            <table className="table table-sm table-bordered small">
              <thead style={{ backgroundColor: COLOR, color: '#fff' }}>
                <tr><th>Feature</th><th>MDDS2 (AR LOF)</th><th>PEO2 (AD Dominant-Negative)</th></tr>
              </thead>
              <tbody>
                {Object.entries({ genetics: 'Genetics', mtdna: 'mtDNA Finding', onset: 'Onset', dominant_feature: 'Dominant Feature', lactic_acidosis: 'Lactic Acidosis', severity: 'Severity' }).map(([k, label]) => (
                  <tr key={k}>
                    <td className="fw-semibold">{label}</td>
                    <td style={{ color: COLOR }}>{data.ant1_vs_peo2.mdds2[k]}</td>
                    <td className="text-muted">{data.ant1_vs_peo2.peo2[k]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </SectionCard>

      <SectionCard title="Disease Lifecycle Stages">
        {lifecycle.map((s, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff', borderLeft: `3px solid ${COLOR}` }}>
            <div className="fw-bold small" style={{ color: COLOR }}>{s.stage}</div>
            <div className="small text-muted mt-1"><strong>Features:</strong> {s.features}</div>
            <div className="small mt-1"><strong>Management:</strong> {s.management}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Cardiac & Metabolic ───────────────────────────────────────────────
function CardiacTab({ data }) {
  if (!data) return <Spinner />;
  const { seizure_profiles = [], cardiac_features = [], metabolic_markers = [] } = data;
  return (
    <div>
      <SectionCard title="Cardiac Features (HCM Dominant — 100%)">
        <Alert variant="danger" text="⚠ Hypertrophic Cardiomyopathy (HCM) is 100% prevalent and the LEADING CAUSE OF DEATH in MDDS2 — dominant feature distinguishing MDDS2 from all other encephalomyopathic MDDS (SUCLA2/SUCLG1/RRM2B/FBXL4)" />
        {cardiac_features.map((c, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
            <div className="d-flex justify-content-between align-items-center">
              <span className="fw-semibold small">{c.feature}</span>
              <span className="badge" style={{ backgroundColor: COLOR }}>{c.prevalence_pct}%</span>
            </div>
            <div className="small text-muted mt-1">{c.notes}</div>
            <div className="small mt-1"><strong>Management:</strong> {c.management}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Profiles (35% prevalence — less prominent than encephalomyopathic MDDS)">
        <Alert variant="warning" text="⚠ Seizures occur in ~35% — less prominent than in SUCLA2/SUCLG1/FBXL4; when present, LEV first-line; VPA ABSOLUTE CI regardless of seizure type" />
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead><tr><th>Seizure Type</th><th>Prevalence</th><th>EEG</th><th>Clinical Tip</th></tr></thead>
            <tbody>
              {seizure_profiles.map((s, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ color: COLOR }}>{s.type}</td>
                  <td><Bar label="" value={s.prevalence_pct} /></td>
                  <td className="text-muted">{s.eeg}</td>
                  <td className="small">{s.tip}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Metabolic Markers & Thresholds">
        {metabolic_markers.map((m, i) => (
          <div key={i} className="mb-3 border rounded p-2 small">
            <div className="fw-bold" style={{ color: COLOR }}>{m.marker}</div>
            <div className="row mt-1">
              <div className="col-md-4"><span className="text-muted">Normal: </span>{m.normal}</div>
              <div className="col-md-4"><span className="text-muted">MDDS2: </span><strong>{m.mdds2_value}</strong></div>
              <div className="col-md-4"><span className="text-muted">Significance: </span>{m.significance}</div>
            </div>
            <div className="mt-1 p-1 rounded" style={{ background: LIGHT }}>
              <strong>Action:</strong> {m.action}
            </div>
          </div>
        ))}
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
      <Alert variant="danger" text="⛔ REMEMBER: VPA ABSOLUTE CI · KD CONTRAINDICATED · Propofol AVOID (PRIS) — apply universally in MDDS2 regardless of seizure type or clinical context" />
      {treatments.map((t, i) => (
        <SectionCard key={i} title={`${t.treatment} — ${t.level}`}>
          <div className="small">
            <p><strong>Dose / Detail:</strong> {t.dose_or_detail}</p>
            <p><strong>Mechanism:</strong> {t.mechanism}</p>
            {t.caveat && <Alert variant="warning" text={`⚠ Caveat: ${t.caveat}`} />}
            <p className="mb-0"><strong>Monitoring:</strong> {t.monitoring}</p>
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
    { key: 'gene_protein', title: 'Gene & Protein Concepts' },
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
export default function SLC25A4Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/slc25a4/overview`).then(r => r.json()).then(setOverview).catch(() => setError('Overview load failed'));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2) {
      fetch(`${API}/api/slc25a4/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setError('Breakdown load failed'));
    }
    if (tab === 3) {
      fetch(`${API}/api/slc25a4/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/slc25a4/definitions`).then(r => r.json()).then(setDefs).catch(() => setError('Definitions load failed'));
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          ♥ SLC25A4 / ANT1 — Cardiomyopathic mtDNA Depletion Syndrome 2 (MDDS2)
        </h4>
        <div className="text-muted small">
          OMIM Gene: 103220 · OMIM Disease: 615418 · 4q35.1 · 298 aa · AR biallelic LOF →
          HCM 100% + OXPHOS deficiency + mtDNA depletion · Seed-567 · n=40 synthetic cohort
        </div>
        <div className="mt-2 p-2 rounded small fw-bold" style={{ background: '#ffebee', border: '1px solid #c62828', color: '#b71c1c' }}>
          ⛔ VPA ABSOLUTE CI · ⛔ KD CONTRAINDICATED · ⛔ Propofol AVOID (PRIS) · ⚠ ACE/ARB CI if LVOTO &gt;30 mmHg
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
      {tab === 2 && <CardiacTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={defs} />}
    </div>
  );
}
