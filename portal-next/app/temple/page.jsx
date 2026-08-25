'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Growth', 'Treatments & Genetics', 'Definitions'];

// Temple Syndrome colour scheme — teal/green (imprinting / growth / development)
const ACCENT  = '#004d40';   // deep teal — 14q32.3 imprinting / DLK1
const ACCENT2 = '#00695c';   // teal — paternal LOF mechanism
const ACCENT3 = '#1b5e20';   // deep green — KEY POSITIVES / CPP treatment
const ACCENT4 = '#b71c1c';   // deep red — misdiagnosis risk / VPA risk
const ACCENT5 = '#0d47a1';   // dark blue — genetics / methylation
const ACCENT6 = '#4a148c';   // purple — imprinting (contrast PWS/AS purple)
const ACCENT7 = '#37474f';   // dark slate — epidemiology
const ACCENT8 = '#e65100';   // orange — CPP / GnRH analog urgency

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

function Alert({ color, children }) {
  return (
    <div className="alert mb-2" style={{ background: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 6 }}>
      {children}
    </div>
  );
}

function Section({ title, color, children }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

export default function TemplePage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/temple/overview`).then(r => r.json()),
      fetch(`${API}/api/temple/breakdown`).then(r => r.json()),
      fetch(`${API}/api/temple/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error) return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  const kpi = overview?.kpis || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3" style={{ borderLeft: `6px solid ${ACCENT}`, paddingLeft: 14 }}>
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>🧬 Temple Syndrome (TS14)</h4>
        <div className="text-muted small">
          MEG3 / DLK1 · 14q32.3 · Genomic Imprinting (Paternal LOF) · OMIM #616222
          <span className="ms-3 badge" style={{ background: ACCENT6 }}>Imprinting Disorder</span>
          <span className="ms-1 badge" style={{ background: ACCENT2 }}>upd(14)mat · 70%</span>
        </div>
      </div>

      <Alert color={ACCENT4}>
        <strong>⚠ Misdiagnosis Risk:</strong> TS14 presents identically to PWS in neonates (hypotonia + SGA + NG tube). Standard 15q11-q13 methylation test = NORMAL in TS14. If PWS screen negative + neonatal hypotonia + SGA → test 14q32.3 (IG-DMR methylation) immediately.
      </Alert>

      <Alert color={ACCENT8}>
        <strong>⚡ CPP Alert:</strong> Central Precocious Puberty in ~50-60% females (onset 5.5-8.5y). DLK1 deficiency → premature GnRH activation. GnRH analog (leuprolide) Level A — improves adult height +4-7 cm. Treat promptly.
      </Alert>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Epilepsy" value={`${kpi.epilepsy_pct ?? '--'}%`} color={ACCENT4} />
            <KPI label="CPP (females)" value={`${kpi.cpp_pct ?? '--'}%`} color={ACCENT8} />
            <KPI label="On GnRH Analog" value={`${kpi.gnrh_analog_pct ?? '--'}%`} color={ACCENT3} />
            <KPI label="GH Therapy" value={`${kpi.gh_therapy_pct ?? '--'}%`} color={ACCENT2} />
            <KPI label="Neonatal Hypotonia" value={`${kpi.neonatal_hypotonia_pct ?? '--'}%`} color={ACCENT5} />
            <KPI label="Avg Height SDS" value={kpi.avg_height_sds ?? '--'} color={ACCENT} />
            <KPI label="Avg BMI SDS" value={kpi.avg_bmi_sds ?? '--'} color={ACCENT7} />
            <KPI label="Avg IQ" value={kpi.avg_iq ?? '--'} color={ACCENT6} />
            <KPI label="DLK1 Serum (%)" value={`${kpi.avg_dlk1_serum_pct ?? '--'}%`} color={ACCENT2} />
            <KPI label="Scoliosis" value={`${kpi.scoliosis_pct ?? '--'}%`} color={ACCENT7} />
            <KPI label="Polyhydramnios" value={`${kpi.polyhydramnios_pct ?? '--'}%`} color={ACCENT5} />
            <KPI label="Cohort (n)" value={overview?.cohort_n ?? '--'} color={ACCENT7} />
          </div>

          {/* Mechanism distribution */}
          <Section title="Genetic Mechanism Distribution" color={ACCENT5}>
            <div className="row">
              {Object.entries(overview?.mechanism_distribution || {}).map(([mech, n]) => (
                <div key={mech} className="col-md-6 mb-2">
                  <div className="d-flex justify-content-between align-items-center p-2 rounded" style={{ background: ACCENT5 + '12' }}>
                    <span className="small fw-semibold">{mech.replace(/ \(.*\)/, '')}</span>
                    <span className="badge" style={{ background: ACCENT5 }}>{n} pts</span>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Key facts */}
          <Section title="Key Clinical Facts" color={ACCENT}>
            <ul className="list-unstyled mb-0">
              {(overview?.key_facts || []).map((f, i) => (
                <li key={i} className="mb-1 small">
                  <span style={{ color: ACCENT }} className="me-1">▸</span>{f}
                </li>
              ))}
            </ul>
          </Section>

          {/* Seizure types */}
          {overview?.seizure_types && Object.keys(overview.seizure_types).length > 0 && (
            <Section title="Seizure Type Distribution (Epilepsy Subset)" color={ACCENT4}>
              <div className="row">
                {Object.entries(overview.seizure_types).map(([type, n]) => (
                  <div key={type} className="col-md-6 mb-2">
                    <div className="d-flex justify-content-between p-2 rounded" style={{ background: ACCENT4 + '12' }}>
                      <span className="small">{type}</span>
                      <span className="badge" style={{ background: ACCENT4 }}>{n}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* AED distribution */}
          {overview?.aed_distribution && Object.keys(overview.aed_distribution).length > 0 && (
            <Section title="AED Distribution (Epilepsy Subset)" color={ACCENT3}>
              <div className="row">
                {Object.entries(overview.aed_distribution).map(([aed, n]) => (
                  <div key={aed} className="col-6 col-md-4 mb-2">
                    <div className="d-flex justify-content-between p-2 rounded" style={{ background: ACCENT3 + '12' }}>
                      <span className="small fw-semibold">{aed}</span>
                      <span className="badge" style={{ background: ACCENT3 }}>{n}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          )}
        </div>
      )}

      {/* ── TAB 1: Patients & Growth ── */}
      {tab === 1 && breakdown && (
        <div>
          <Section title="Treatment Summary" color={ACCENT3}>
            <div className="row g-2 mb-3">
              {[
                ['Total Patients', breakdown.summary?.total, ACCENT7],
                ['Epilepsy', breakdown.summary?.with_epilepsy, ACCENT4],
                ['CPP', breakdown.summary?.with_cpp, ACCENT8],
                ['On GnRH Analog', breakdown.summary?.on_gnrh_analog, ACCENT3],
                ['On GH Therapy', breakdown.summary?.on_gh_therapy, ACCENT2],
                ['Neonatal Hypotonia', breakdown.summary?.neonatal_hypotonia, ACCENT5],
                ['Scoliosis', breakdown.summary?.scoliosis, ACCENT7],
              ].map(([label, val, color]) => (
                <div key={label} className="col-6 col-md-3 mb-2">
                  <div className="card shadow-sm text-center py-2">
                    <div className="fw-bold fs-5" style={{ color }}>{val ?? '--'}</div>
                    <div className="text-muted small">{label}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* EEG Patterns */}
          <Section title="EEG Patterns (TS14 Epilepsy Subset)" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead><tr style={{ background: ACCENT6 + '18' }}>
                  <th>EEG Pattern</th><th>Frequency (%)</th><th>Notes</th>
                </tr></thead>
                <tbody>
                  {(breakdown.eeg_patterns || []).map((ep, i) => (
                    <tr key={i}>
                      <td className="fw-semibold small">{ep.pattern}</td>
                      <td><span className="badge" style={{ background: ACCENT6 }}>{ep.pct}%</span></td>
                      <td className="small text-muted">{ep.notes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Patient table */}
          <Section title="Patient Cohort (n=40, seed=291)" color={ACCENT}>
            <div className="table-responsive" style={{ maxHeight: 480, overflowY: 'auto' }}>
              <table className="table table-sm table-hover align-middle mb-0">
                <thead className="sticky-top" style={{ background: ACCENT + '18' }}>
                  <tr>
                    <th>ID</th><th>Sex</th><th>Mechanism</th><th>Ht SDS</th>
                    <th>BMI SDS</th><th>IQ</th><th>CPP</th><th>GnRH</th>
                    <th>GH Rx</th><th>Epilepsy</th><th>Seizure</th><th>AED</th>
                    <th>DLK1%</th><th>Hypotonia</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.patients || []).map((p, i) => (
                    <tr key={i}>
                      <td className="fw-semibold small">{p.patient_id}</td>
                      <td className="small">{p.sex}</td>
                      <td className="small" style={{ maxWidth: 140, whiteSpace: 'normal' }}>
                        {p.mechanism.split(' (')[0].split('—')[0]}
                      </td>
                      <td className="small">
                        <span style={{ color: p.height_sds < -3 ? ACCENT4 : ACCENT7 }}>{p.height_sds}</span>
                      </td>
                      <td className="small">
                        <span style={{ color: p.bmi_sds > 2 ? ACCENT8 : ACCENT7 }}>{p.bmi_sds}</span>
                      </td>
                      <td className="small">{p.iq}</td>
                      <td className="small">
                        {p.has_cpp ? <span className="badge" style={{ background: ACCENT8 }}>CPP {p.age_puberty_onset_y}y</span> : '—'}
                      </td>
                      <td className="small">
                        {p.on_gnrh_analog ? <span className="badge" style={{ background: ACCENT3 }}>GnRH</span> : '—'}
                      </td>
                      <td className="small">
                        {p.on_gh_therapy ? <span className="badge" style={{ background: ACCENT2 }}>GH</span> : '—'}
                      </td>
                      <td className="small">
                        {p.has_epilepsy ? <span className="badge" style={{ background: ACCENT4 }}>Yes</span> : '—'}
                      </td>
                      <td className="small">{p.seizure_type ?? '—'}</td>
                      <td className="small">{p.current_aed ?? '—'}</td>
                      <td className="small">
                        <span style={{ color: p.dlk1_serum_pct < 40 ? ACCENT4 : ACCENT3 }}>{p.dlk1_serum_pct}%</span>
                      </td>
                      <td className="small">
                        {p.hypotonia_neonatal ? <span className="badge" style={{ background: ACCENT5 }}>Yes</span> : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 2: Treatments & Genetics ── */}
      {tab === 2 && breakdown && (
        <div>
          <Section title="Treatment Evidence Table" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead><tr style={{ background: ACCENT3 + '18' }}>
                  <th>Treatment</th><th>Level</th><th>Target</th><th>Notes</th>
                </tr></thead>
                <tbody>
                  {(breakdown.treatments || []).map((t, i) => (
                    <tr key={i}>
                      <td className="fw-semibold small">{t.name}</td>
                      <td>
                        <span className="badge" style={{ background: t.level === 'A' ? ACCENT3 : t.level === 'B' ? ACCENT5 : ACCENT7 }}>
                          Level {t.level}
                        </span>
                      </td>
                      <td className="small">{t.target}</td>
                      <td className="small text-muted">{t.notes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Genetic Mechanisms" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead><tr style={{ background: ACCENT5 + '18' }}>
                  <th>Mechanism</th><th>Freq (%)</th><th>Detection</th><th>Phenotype</th><th>Recurrence</th>
                </tr></thead>
                <tbody>
                  {(breakdown.mechanisms || []).map((m, i) => (
                    <tr key={i}>
                      <td className="fw-semibold small">{m.mechanism}</td>
                      <td><span className="badge" style={{ background: ACCENT5 }}>{m.freq}%</span></td>
                      <td className="small">{m.detection}</td>
                      <td className="small">{m.phenotype}</td>
                      <td className="small fw-semibold" style={{ color: m.recurrence.startsWith('50') ? ACCENT4 : ACCENT3 }}>
                        {m.recurrence}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Alert color={ACCENT4}>
            <strong>Parent-of-origin matters for recurrence:</strong> Paternal deletion inherited from father → 50% recurrence. Same deletion inherited from mother → NO TS14 (DLK1 not expressed maternally). Always study both parents with SNP array + methylation.
          </Alert>

          <Alert color={ACCENT8}>
            <strong>DLK1 vs GH:</strong> DLK1 inhibits adipogenesis AND GnRH. Loss of DLK1 → (1) truncal obesity + (2) CPP. GH secretion is usually NORMAL — GH therapy is off-label Level C for short stature, NOT a replacement (contrast PWS where GH is Level A for deficiency).
          </Alert>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && definitions && (
        <div>
          {Object.entries(definitions).map(([section, entries]) => (
            <Section key={section} title={section.replace(/_/g, ' ')} color={ACCENT5}>
              {Object.entries(entries).map(([term, text]) => (
                <div key={term} className="mb-3 p-3 rounded" style={{ background: ACCENT5 + '0a', border: `1px solid ${ACCENT5}30` }}>
                  <div className="fw-bold small mb-1" style={{ color: ACCENT5 }}>{term.replace(/_/g, ' ')}</div>
                  <div className="small text-muted" style={{ lineHeight: 1.6, whiteSpace: 'pre-line' }}>{text}</div>
                </div>
              ))}
            </Section>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="mt-4 pt-2 border-top text-muted small">
        <strong style={{ color: ACCENT }}>Temple Syndrome (TS14)</strong> — MEG3/DLK1 · 14q32.3 · Genomic Imprinting ·
        OMIM Disease #616222 · OMIM Gene MEG3 *601626 · DLK1 *176290 ·
        Cohort n=40 seed=291 · 3 endpoints /api/temple/overview|breakdown|definitions
        <span className="ms-2">|</span>
        <Link className="ms-2" href="/pws" style={{ color: ACCENT2 }}>← PWS (15q11-q13, Paternal)</Link>
        <span className="ms-2">|</span>
        <Link className="ms-2" href="/ube3a" style={{ color: ACCENT6 }}>Angelman (15q11-q13, Maternal) →</Link>
      </div>
    </div>
  );
}
