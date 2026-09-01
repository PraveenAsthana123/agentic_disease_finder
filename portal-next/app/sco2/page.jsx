'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#b71c1c';   // deep crimson — cardiac emergency / HCM / SCO2
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

      <SectionCard title="Mechanism — SCO2 / CuA Copper Delivery / Complex IV COX2 Assembly">
        <p className="small text-muted mb-0">{data.mechanism}</p>
      </SectionCard>

      <SectionCard title="HCM 100% — Most Severe Cardiac COX Assembly Disease: SCO2 vs SURF1 / SCO1 / COX10" borderColor="#c62828">
        <p className="small text-muted mb-0">{data.hcm_note}</p>
      </SectionCard>

      <SectionCard title={`KPIs — ${data.cohort}`}>
        <div className="row g-2">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Drug Contraindications & Safety" borderColor="#b71c1c">
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
      <SectionCard title="Patient Cohort (40 patients · seed-595 · SCO2 biallelic — HCM + COX deficiency)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Genotype</th><th>Sex</th><th>Onset(wk)</th>
                <th>Lactate</th><th>COX%</th><th>HCM Wall(mm)</th>
                <th>Features</th><th>Treatments</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id} style={{ background: p.outcome.startsWith('Died') ? '#fff3e0' : '#fce4ec' }}>
                  <td className="fw-semibold">{p.id}</td>
                  <td className="small" style={{ maxWidth: 200, wordBreak: 'break-word' }}>{p.geno}</td>
                  <td>{p.sex}</td>
                  <td>
                    <span style={{ color: p.onset_wk < 2 ? '#b71c1c' : p.onset_wk < 6 ? '#c62828' : '#e65100', fontWeight: 600 }}>
                      {p.onset_wk}
                    </span>
                  </td>
                  <td>
                    <span style={{
                      color: p.lactate >= 10 ? '#b71c1c' : p.lactate >= 6 ? '#c62828' : '#e65100',
                      fontWeight: 700,
                    }}>{p.lactate}</span>
                  </td>
                  <td>
                    <span style={{ color: p.cox_pct < 8 ? '#b71c1c' : p.cox_pct < 15 ? '#c62828' : '#e65100', fontWeight: 600 }}>
                      {p.cox_pct}%
                    </span>
                  </td>
                  <td>
                    <span style={{ color: p.hcm_wall >= 18 ? '#b71c1c' : p.hcm_wall >= 14 ? '#c62828' : '#e65100', fontWeight: 600 }}>
                      {p.hcm_wall}
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
              feat.toLowerCase().includes('hcm') ? COLOR :
              feat.toLowerCase().includes('lactic') ? '#c62828' :
              feat.toLowerCase().includes('resp') ? '#b71c1c' :
              feat.toLowerCase().includes('seizure') ? '#6a1b9a' :
              feat.toLowerCase().includes('no hepat') || feat.toLowerCase().includes('no 3') ? '#2e7d32' :
              feat.toLowerCase().includes('alive') ? '#2e7d32' :
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
export default function SCO2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/sco2/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1) fetch(`${API}/api/sco2/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 2) fetch(`${API}/api/sco2/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 3) fetch(`${API}/api/sco2/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="d-flex align-items-center mb-3 gap-3">
        <div style={{
          width: 48, height: 48, borderRadius: '50%',
          background: COLOR, display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: '#fff', fontWeight: 'bold', fontSize: 22, flexShrink: 0,
        }}>❤️</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            SCO2 — Fatal Infantile HCM (Complex IV / COX Deficiency)
          </h4>
          <div className="text-muted small">
            SCO2-266aa · 22q13.33 · AR · OMIM *604272 ·
            CuA copper chaperone for COX2 · HCM 100% CARDINAL ·
            Isolated COX deficiency · NO hepatopathy (DDx SCO1) ·
            VPA / Metformin / Linezolid / Propofol / KD ABSOLUTE CI · Digoxin AVOID · LEV preferred AED
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
