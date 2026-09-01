'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#1a237e';   // deep indigo — Leigh/basal ganglia / SURF1 Complex IV
const LIGHT = '#e8eaf6';

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
      </SectionCard>

      <SectionCard title="Mechanism — SURF1 / Complex IV Assembly / COX1 Cofactor Insertion Failure">
        <p className="small text-muted mb-0">{data.mechanism}</p>
      </SectionCard>

      <SectionCard title="No Hepatopathy, No Iron Overload, No 3-MGA — Critical Negative Features" borderColor="#1565c0">
        <p className="small text-muted mb-0">{data.no_hepatopathy_note}</p>
      </SectionCard>

      <SectionCard title={`KPIs — ${data.cohort}`}>
        <div className="row g-2">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Drug Contraindications & Safety" borderColor="#c62828">
        {(data.contraindications || []).map(c => (
          <Alert
            key={c.drug}
            variant={c.severity.startsWith('ABSOLUTE') ? 'danger' : 'warning'}
            text={<><strong>{c.drug}</strong> — {c.severity}: {c.mechanism}</>}
          />
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Features ───────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="Patient Cohort (40 patients · seed-593 · SURF1 biallelic Leigh/COX)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Genotype</th><th>Sex</th><th>Onset(mo)</th>
                <th>Lactate</th><th>COX%</th>
                <th>Features</th><th>Treatments</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id} style={{ background: p.outcome.startsWith('Died') ? '#fff3e0' : '#f3e5f5' }}>
                  <td className="fw-semibold">{p.id}</td>
                  <td className="small" style={{ maxWidth: 200, wordBreak: 'break-word' }}>{p.geno}</td>
                  <td>{p.sex}</td>
                  <td>
                    <span style={{ color: p.onset_mo < 6 ? '#c62828' : p.onset_mo < 12 ? '#e65100' : '#558b2f', fontWeight: 600 }}>
                      {p.onset_mo}
                    </span>
                  </td>
                  <td>
                    <span style={{
                      color: p.lactate >= 8 ? '#b71c1c' : p.lactate >= 5 ? '#c62828' : '#e65100',
                      fontWeight: 700,
                    }}>{p.lactate}</span>
                  </td>
                  <td>
                    <span style={{ color: p.cox_pct < 10 ? '#b71c1c' : p.cox_pct < 20 ? '#c62828' : '#e65100', fontWeight: 600 }}>
                      {p.cox_pct}%
                    </span>
                  </td>
                  <td className="small">{p.features}</td>
                  <td className="small">{p.treatments}</td>
                  <td style={{ color: p.outcome.startsWith('Died') ? '#c62828' : '#2e7d32', fontWeight: 600 }}>
                    {p.outcome}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Treatments & DDx ───────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="Clinical Feature Frequencies (Cohort)">
        {Object.entries(data.feature_frequencies || {}).map(([feat, pct]) => (
          <Bar
            key={feat}
            label={feat}
            value={pct}
            color={
              pct === 100 ? COLOR :
              feat.toLowerCase().includes('leigh') ? COLOR :
              feat.toLowerCase().includes('lactic') || feat.toLowerCase().includes('died') ? '#c62828' :
              feat.toLowerCase().includes('resp') ? '#b71c1c' :
              feat.toLowerCase().includes('seizure') ? '#6a1b9a' :
              feat.toLowerCase().includes('no hepat') || feat.toLowerCase().includes('no iron') ? '#2e7d32' :
              COLOR
            }
          />
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Spinner />;
  const sections = [
    { key: 'pharmacology',       label: 'Pharmacology & Molecular Biology' },
    { key: 'gene_concepts',      label: 'Gene & Genotype–Phenotype Concepts' },
    { key: 'disease_concepts',   label: 'Disease Concepts & DDx' },
    { key: 'prescribing_safety', label: 'Prescribing Safety (Extended)' },
  ];
  return (
    <div>
      {sections.map(sec => (
        data[sec.key] && (
          <SectionCard key={sec.key} title={sec.label}>
            {data[sec.key].map(d => (
              <div key={d.term} className="mb-4">
                <div className="fw-semibold small mb-1" style={{ color: COLOR }}>{d.term}</div>
                <p className="small text-muted mb-0" style={{ whiteSpace: 'pre-wrap' }}>{d.definition}</p>
              </div>
            ))}
          </SectionCard>
        )
      ))}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function SURF1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/surf1/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1) fetch(`${API}/api/surf1/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 2) fetch(`${API}/api/surf1/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 3) fetch(`${API}/api/surf1/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="d-flex align-items-center mb-3 gap-3">
        <div style={{
          width: 48, height: 48, borderRadius: '50%',
          background: COLOR, display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: '#fff', fontWeight: 'bold', fontSize: 22, flexShrink: 0,
        }}>🧠</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            SURF1 — Leigh Syndrome (Complex IV / COX Deficiency)
          </h4>
          <div className="text-muted small">
            SURF1-300aa · 9q34.2 · AR · OMIM #185620 ·
            Most common single-gene Leigh cause · Isolated COX deficiency ·
            Bilateral basal ganglia + brainstem Leigh MRI · No hepatopathy · No iron overload ·
            VPA / Metformin / Linezolid / KD ABSOLUTE CI · Propofol AVOID · LEV preferred AED
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <TreatmentsTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
