'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4e342e';   // deep brown — Farber / lipogranuloma / ceramide
const ACCENT2 = '#b71c1c';   // dark-red — HIGH RISK / danger / no-ERT
const ACCENT3 = '#e65100';   // deep-orange — CAUTION / PATHOGNOMONIC
const ACCENT4 = '#1b5e20';   // dark-green — safe treatments / HSCT / Level A-B
const ACCENT5 = '#4a148c';   // dark-purple — ceramide pathway / molecular
const ACCENT6 = '#01579b';   // dark-blue — biomarkers / enzyme activity

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
    : item.risk?.includes('RELATIVE-CI') ? ACCENT3
    : item.risk?.includes('CAUTION') ? '#f57c00'
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
  const lvl = item.level || '';
  const lvlColor = lvl.includes('A') ? ACCENT4
    : lvl.includes('B') ? '#1565c0'
    : lvl.includes('C') ? '#6a1b9a'
    : '#607d8b';
  return (
    <div className="col-md-6 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-header text-white small fw-bold" style={{ background: lvlColor }}>
          {item.treatment} — Level {item.level}
        </div>
        <div className="card-body small">
          <p className="mb-1"><strong>Indication:</strong> {item.indication}</p>
          <p className="mb-1"><strong>Mechanism:</strong> {item.mechanism}</p>
          <p className="mb-1 text-warning-emphasis"><strong>Monitoring:</strong> {item.monitoring}</p>
          {item.caution && <p className="mb-0 text-danger"><strong>Caution:</strong> {item.caution}</p>}
        </div>
      </div>
    </div>
  );
}

export default function ASAH1Dashboard() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/asah1/overview`).then(r => r.json()),
      fetch(`${API}/api/asah1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/asah1/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (err) return <div className="p-4 text-danger">Error: {err}</div>;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="card shadow mb-4 text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT5} 100%)` }}>
        <div className="card-body py-3">
          <h4 className="mb-1 fw-bold">&#x1f9e0; ASAH1 — Farber Disease (Acid Ceramidase Deficiency)</h4>
          <div className="small opacity-75">
            Farber Lipogranulomatosis · Ceramide Catabolism Block · Lipogranuloma Triad PATHOGNOMONIC ·
            Farber Bodies EM PATHOGNOMONIC · No Approved ERT (Critical Gap) · HSCT Level B Non-CNS ·
            ACTH Level A IS Type 5 · CBZ/OXC RELATIVE-CI PME · Typical Antipsychotics HIGH RISK Ceramide Additive ·
            AR Biallelic LOF · 8p22
          </div>
          <div className="mt-2 small">
            <Badge text="OMIM #228000" color={ACCENT6} />
            <Badge text="8p22" color={ACCENT5} />
            <Badge text="AR Biallelic LOF" color="#37474f" />
            <Badge text="Rarest LSD ~200 cases" color={ACCENT2} />
            <Badge text="No Approved ERT" color={ACCENT2} />
            <Badge text="HSCT Level B" color={ACCENT4} />
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Cohort" value={`${ov?.cohort_size ?? 40}pts`} color={ACCENT} />
        <KPI label="Seizures (overall)" value={`${ov?.seizure_pct_overall ?? 42}%`} color={ACCENT3} />
        <KPI label="Seizures Type 5" value={`${ov?.seizure_pct_type5 ?? 88}%`} color={ACCENT2} />
        <KPI label="IS (Type 5)" value={`${ov?.infantile_spasms_pct ?? 32}%`} color={ACCENT2} />
        <KPI label="Drug-Resistant" value={`${ov?.drug_resistant_pct ?? 70}%`} color={ACCENT2} />
        <KPI label="Dx Delay" value={`${ov?.mean_diagnosis_delay_years ?? 2.4}yr`} color="#78909c" />
        <KPI label="Lipogranuloma Triad" value={`${ov?.lipogranuloma_triad_pct ?? 90}%`} color={ACCENT3} />
        <KPI label="Hepatomegaly" value={`${ov?.hepatomegaly_pct ?? 72}%`} color={ACCENT5} />
        <KPI label="On HSCT" value={`${ov?.on_hsct_pct ?? 22}%`} color={ACCENT4} />
        <KPI label="On ACTH" value={`${ov?.on_acth_pct ?? 28}%`} color={ACCENT4} />
        <KPI label="On LEV" value={`${ov?.on_lev_pct ?? 60}%`} color={ACCENT4} />
        <KPI label="Global Cases" value={`~${ov?.global_cases_estimate ?? 200}`} color="#78909c" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* Tab 0: Overview */}
      {tab === 0 && ov && (
        <div>
          <SectionCard title="&#x1f9ec; Disease Overview — Farber Lipogranulomatosis (ASAH1)" color={ACCENT}>
            <p className="mb-2">{ov.disease}</p>
            <div className="row mt-3">
              <div className="col-md-6">
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><th>Gene</th><td>{ov.gene}</td></tr>
                    <tr><th>Locus</th><td>{ov.locus}</td></tr>
                    <tr><th>OMIM</th><td>{ov.omim}</td></tr>
                    <tr><th>Inheritance</th><td>{ov.inheritance}</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="col-md-6">
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><th>Protein</th><td>{ov.protein}</td></tr>
                    <tr><th>Mechanism</th><td className="small">{ov.mechanism?.substring(0, 200)}…</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </SectionCard>

          <div className="row">
            <div className="col-md-6">
              <SectionCard title="&#x1f7e0; Lipogranuloma Triad — PATHOGNOMONIC" color={ACCENT3}>
                <p className="small">{ov.pathognomonic_triad_note}</p>
              </SectionCard>
            </div>
            <div className="col-md-6">
              <SectionCard title="&#x1f52c; Farber Bodies on EM — PATHOGNOMONIC" color={ACCENT5}>
                <p className="small">{ov.farber_bodies_note}</p>
              </SectionCard>
            </div>
          </div>

          <SectionCard title="&#x26a0;&#xfe0f; No Approved ERT — Critical Management Gap" color={ACCENT2}>
            <p className="small">{ov.no_ert_note}</p>
          </SectionCard>

          <SectionCard title="&#x1f4ca; Ceramide Pathway Context" color={ACCENT5}>
            <div className="row">
              <div className="col-md-6">
                <PctBar label="Lipogranuloma Triad (Pathognomonic)" pct={90} color={ACCENT3} />
                <PctBar label="Seizures — Type 5 (Neurological)" pct={88} color={ACCENT2} />
                <PctBar label="Seizures — Overall Cohort" pct={42} color={ACCENT} />
                <PctBar label="Drug-Resistant (Type 5)" pct={70} color={ACCENT2} />
                <PctBar label="Farber Bodies on EM" pct={85} color={ACCENT5} />
              </div>
              <div className="col-md-6">
                <PctBar label="Hepatomegaly (Types 1+5)" pct={72} color={ACCENT5} />
                <PctBar label="On HSCT (Types 1–3)" pct={22} color={ACCENT4} />
                <PctBar label="On ACTH (Type 5 IS)" pct={28} color={ACCENT4} />
                <PctBar label="On LEV" pct={60} color={ACCENT4} />
                <PctBar label="Infantile Spasms (Type 5)" pct={32} color={ACCENT2} />
              </div>
            </div>
            <div className="mt-3 p-2 rounded small" style={{ background: '#fce4ec' }}>
              <strong>Ceramide pathway:</strong> Sphingomyelin →[SMPD1 (11p15.4)]→ <strong>Ceramide</strong> →[<strong>ASAH1 (8p22) BLOCK</strong>]→ Sphingosine + Fatty Acid.
              CERS1-6 (de novo synthesis) also produce ceramide. Ceramide accumulates in ASAH1 LOF (pro-apoptotic).
              Typical antipsychotics activate SMPD1 → more ceramide → additive toxicity (HIGH RISK).
            </div>
          </SectionCard>

          <SectionCard title="&#x1f4d6; Standards & References" color={ACCENT6}>
            <ol className="small mb-0">
              {(ov.standards || []).map((s, i) => <li key={i}>{s}</li>)}
            </ol>
          </SectionCard>
        </div>
      )}

      {/* Tab 1: Patients & Etiology */}
      {tab === 1 && bk && (
        <div>
          <SectionCard title="&#x1f9ec; Patient Cohort — 40 Patients, 6 ASAH1 Subtypes" color={ACCENT}>
            <div className="row">
              {(bk.etiologies || []).map((e, i) => (
                <div key={i} className="col-md-6 mb-3">
                  <div className="card h-100 border-0 shadow-sm">
                    <div className="card-header text-white small fw-bold" style={{ background: ACCENT }}>
                      {e.subtype} — {e.pct}% (n={e.n}) · AC activity ~{e.ac_activity_pct}%
                    </div>
                    <div className="card-body small">
                      <p className="mb-1 text-muted"><em>Onset: ~{e.onset_months} months · Alleles: {e.alleles}</em></p>
                      <ul className="mb-1 ps-3">
                        {(e.features || []).map((f, j) => <li key={j}>{f}</li>)}
                      </ul>
                      <div className="mt-1 p-1 rounded" style={{ background: '#fafafa' }}><em className="small text-muted">{e.note}</em></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="&#x1f9e0; Subtype Seizure Summary" color={ACCENT5}>
            <table className="table table-sm table-striped small">
              <thead><tr><th>Subtype</th><th>Seizure Profile</th></tr></thead>
              <tbody>
                {Object.entries(bk.subtype_seizure_summary || {}).map(([k, v], i) => (
                  <tr key={i}><td className="fw-bold">{k}</td><td>{v}</td></tr>
                ))}
              </tbody>
            </table>
          </SectionCard>

          <SectionCard title="&#x2697;&#xfe0f; Ceramide Pathway Diagram" color={ACCENT5}>
            {bk.ceramide_pathway && (
              <div className="small">
                <div className="p-2 mb-2 rounded text-white" style={{ background: ACCENT5 }}>
                  <strong>Upstream:</strong> {bk.ceramide_pathway.upstream_smpd1}
                </div>
                <div className="p-2 mb-2 rounded text-white" style={{ background: '#6a1b9a' }}>
                  <strong>De Novo:</strong> {bk.ceramide_pathway.cers_synthesis}
                </div>
                <div className="p-2 mb-2 rounded text-white fw-bold" style={{ background: ACCENT2 }}>
                  <strong>ASAH1 BLOCK (Farber):</strong> {bk.ceramide_pathway.asah1_catabolism}
                </div>
                <div className="p-2 mb-2 rounded" style={{ background: '#e8f5e9' }}>
                  <strong>Downstream S1P:</strong> {bk.ceramide_pathway.downstream_sphk}
                </div>
                <div className="p-2 mb-2 rounded" style={{ background: '#e3f2fd' }}>
                  <strong>Downstream GBA substrate:</strong> {bk.ceramide_pathway.downstream_ugcg}
                </div>
                <div className="p-2 mb-2 rounded" style={{ background: '#fff3e0' }}>
                  <strong>Saposin D (PSAP):</strong> {bk.ceramide_pathway.saposin_d}
                </div>
                <div className="p-2 rounded" style={{ background: '#fce4ec' }}>
                  <strong>Clinical implication:</strong> {bk.ceramide_pathway.clinical_implication}
                </div>
              </div>
            )}
          </SectionCard>
        </div>
      )}

      {/* Tab 2: Seizures & Triggers */}
      {tab === 2 && bk && (
        <div>
          <SectionCard title="&#x26a1; Seizure Types (ASAH1 Farber Disease)" color={ACCENT2}>
            <div className="row">
              {(bk.seizure_types || []).map((s, i) => (
                <div key={i} className="col-md-6 mb-3">
                  <div className="card h-100 border-0 shadow-sm">
                    <div className="card-header text-white small fw-bold" style={{ background: ACCENT2 }}>
                      {s.type} — {s.pct}% {s.subtype_restricted ? `(${s.subtype_restricted})` : ''}
                    </div>
                    <div className="card-body small">
                      <p className="mb-1"><strong>EEG:</strong> {s.eeg}</p>
                      <p className="mb-1 text-success"><strong>First-line:</strong> {s.first_line}</p>
                      <p className="mb-0 text-muted"><em>{s.notes}</em></p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="&#x1f525; Seizure Triggers" color={ACCENT3}>
            {(bk.triggers || []).map((t, i) => (
              <div key={i} className="mb-3">
                <div className="d-flex justify-content-between small mb-1">
                  <span className="fw-bold">{t.trigger}</span><span>{t.pct}%</span>
                </div>
                <div className="progress mb-1" style={{ height: 8 }}>
                  <div className="progress-bar" style={{ width: `${t.pct}%`, background: t.pct === 100 ? ACCENT2 : ACCENT3 }} />
                </div>
                <div className="small text-muted">{t.mechanism}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="&#x1f489; Treatment Hierarchy (Seizure Management)" color={ACCENT4}>
            <ol className="small mb-0">
              {(bk.treatment_hierarchy || []).map((h, i) => <li key={i} className="mb-1">{h}</li>)}
            </ol>
          </SectionCard>
        </div>
      )}

      {/* Tab 3: Treatments & Contraindications */}
      {tab === 3 && bk && (
        <div>
          <SectionCard title="&#x1f489; Treatments (ASAH1 Farber Disease)" color={ACCENT4}>
            <div className="row">
              {(bk.treatments || []).map((t, i) => <TreatmentCard key={i} item={t} />)}
            </div>
          </SectionCard>

          <SectionCard title="&#x26d4; Contraindications & HIGH RISK Drugs" color={ACCENT2}>
            <div className="row">
              {(bk.contraindications || []).map((c, i) => <CICard key={i} item={c} />)}
            </div>
          </SectionCard>

          <SectionCard title="&#x1f4cf; Clinical Thresholds" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small">
                <thead><tr><th>Threshold</th><th>Value</th><th>Context</th></tr></thead>
                <tbody>
                  {(bk.thresholds || []).map((t, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{t.name}</td>
                      <td><Badge text={t.value} color={ACCENT6} /></td>
                      <td className="text-muted">{t.context}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </div>
      )}

      {/* Tab 4: Definitions */}
      {tab === 4 && df && (
        <div>
          <SectionCard title="&#x1f4d6; Diagnostic Algorithm (7 Steps)" color={ACCENT}>
            <ol className="small mb-0">
              {(df.diagnostic_algorithm || []).map((s, i) => <li key={i} className="mb-2">{s}</li>)}
            </ol>
          </SectionCard>

          <SectionCard title="&#x1f9e0; Key Concepts (19 Terms)" color={ACCENT5}>
            <div className="accordion" id="conceptsAccordion">
              {(df.key_concepts || []).map((c, i) => (
                <div key={i} className="accordion-item">
                  <h2 className="accordion-header">
                    <button className="accordion-button collapsed small fw-bold py-2" type="button"
                      data-bs-toggle="collapse" data-bs-target={`#concept${i}`}>
                      {c.term}
                    </button>
                  </h2>
                  <div id={`concept${i}`} className="accordion-collapse collapse" data-bs-parent="#conceptsAccordion">
                    <div className="accordion-body small">{c.definition}</div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="&#x2697;&#xfe0f; Ceramide Pathway Glossary" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Term</th><th>Definition</th></tr></thead>
                <tbody>
                  {Object.entries(df.ceramide_pathway_glossary || {}).map(([k, v], i) => (
                    <tr key={i}><td className="fw-bold" style={{ color: ACCENT5 }}>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="&#x1f4cf; Clinical Thresholds" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small">
                <thead><tr><th>Threshold</th><th>Value</th><th>Context</th></tr></thead>
                <tbody>
                  {(df.thresholds || []).map((t, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{t.name}</td>
                      <td><Badge text={t.value} color={ACCENT6} /></td>
                      <td className="text-muted">{t.context}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="&#x1f4d6; Standards & References (12)" color={ACCENT6}>
            <ol className="small mb-0">
              {(df.standards || []).map((s, i) => <li key={i} className="mb-1">{s}</li>)}
            </ol>
          </SectionCard>
        </div>
      )}
    </div>
  );
}
