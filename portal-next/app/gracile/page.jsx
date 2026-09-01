'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#4e342e';   // deep brown — iron overload / hepatic / GRACILE
const LIGHT = '#efebe9';

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

      <SectionCard title="Mechanism — BCS1L / Complex III Assembly / Rieske FeS Insertion Failure">
        <p className="small text-muted mb-0">{data.mechanism}</p>
      </SectionCard>

      <SectionCard title="GRACILE vs Björnstad Syndrome — Same Gene, Opposite Prognosis">
        <p className="small text-muted mb-0">{data.bjnb_contrast}</p>
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
      <SectionCard title="Patient Cohort (40 patients · seed-591 · GRACILE BCS1L biallelic)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Genotype</th><th>Sex</th><th>BWT(kg)</th>
                <th>Lactate</th><th>pH</th><th>Ferritin</th>
                <th>Features</th><th>Treatments</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id} style={{ background: p.outcome.startsWith('Died') ? '#fff3e0' : '#f1f8e9' }}>
                  <td className="fw-semibold">{p.id}</td>
                  <td className="small" style={{ maxWidth: 200, wordBreak: 'break-word' }}>{p.geno}</td>
                  <td>{p.sex}</td>
                  <td>
                    <span style={{ color: p.bwt < 2.0 ? '#c62828' : '#558b2f', fontWeight: 600 }}>
                      {p.bwt}
                    </span>
                  </td>
                  <td>
                    <span style={{
                      color: p.lactate >= 15 ? '#b71c1c' : p.lactate >= 12 ? '#c62828' : '#e65100',
                      fontWeight: 700,
                    }}>{p.lactate}</span>
                  </td>
                  <td>
                    <span style={{ color: p.pH < 7.1 ? '#b71c1c' : '#c62828', fontWeight: 600 }}>
                      {p.pH}
                    </span>
                  </td>
                  <td>
                    <span style={{ color: p.ferritin > 3000 ? '#bf360c' : '#e64a19', fontWeight: 600 }}>
                      {p.ferritin}
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
              feat.toLowerCase().includes('death') || feat.toLowerCase().includes('mortality') ? '#c62828' :
              feat.toLowerCase().includes('liver') ? '#c62828' :
              feat.toLowerCase().includes('iron') ? '#bf360c' :
              feat.toLowerCase().includes('lactic') ? '#b71c1c' :
              feat.toLowerCase().includes('hypogly') ? '#e65100' :
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
    { key: 'pharmacology',      label: 'Pharmacology & Molecular Biology' },
    { key: 'gene_concepts',     label: 'Gene & Allele Concepts' },
    { key: 'disease_concepts',  label: 'Disease Concepts & DDx' },
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
export default function GRACILEPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/gracile/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1) fetch(`${API}/api/gracile/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 2) fetch(`${API}/api/gracile/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 3) fetch(`${API}/api/gracile/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="d-flex align-items-center mb-3 gap-3">
        <div style={{
          width: 48, height: 48, borderRadius: '50%',
          background: COLOR, display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: '#fff', fontWeight: 'bold', fontSize: 22, flexShrink: 0,
        }}>🫀</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            GRACILE Syndrome — BCS1L Complex III Assembly Deficiency
          </h4>
          <div className="text-muted small">
            BCS1L-478aa · 2q35 · AR · OMIM #603839 ·
            Growth Restriction + Aminoaciduria + Cholestasis + Iron Overload + Lactic Acidosis + Early Death ·
            Finnish founder p.Ser78Gly (1/36 carrier) ·
            VPA / Iron / Metformin / KD / Linezolid / Propofol ABSOLUTE CI ·
            Continuous IV Dextrose GIR 8–10 mandatory · Never fast
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
