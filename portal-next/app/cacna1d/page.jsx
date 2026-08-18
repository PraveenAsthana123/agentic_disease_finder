'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#1a237e'; // deep navy — Cav1.3 low-threshold L-type; distinct from CACNA1C dark-cyan, CACNA1A/B/E blues
const DANGER = '#b71c1c'; // dark red — CI / cardiac danger
const SUCCESS = '#1b5e20'; // dark green — seizure freedom
const WARN = '#e65100';   // deep orange — SANDD / aldosteronism warning

function KPI({ label, value, color = COLOR }) {
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

function Bar({ label, value, max, color = COLOR }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card shadow-sm mb-3">
      <div className="card-header fw-semibold text-white py-2" style={{ background: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const etiologies = data.etiology_distribution || {};
  const treatments = data.treatments_summary || [];
  const monitoring = data.monitoring_summary || [];
  const lifecycle = data.lifecycle || [];
  const thresholds = data.thresholds || [];
  const ciSummary = data.contraindications_summary || [];
  const seizureSummary = data.seizure_summary || [];
  const etioEntries = Object.entries(etiologies).map(([k, v]) => ({ category: k, count: v }));

  return (
    <div>
      <div className="alert py-2 small mb-3 border" style={{ borderColor: COLOR, borderLeftWidth: 5, background: '#e8eaf6' }}>
        <strong>🧬 CACNA1D (3p14.3) — Cav1.3 Low-Threshold L-type HVA Ca²⁺ Channel — SANDD Syndrome [LOF] / DEE+Autism+Aldosteronism [GOF]:</strong>{' '}
        Cav1.3 activates at V1/2 ≈ <strong>−40 to −55 mV</strong> — uniquely low among L-types (Cav1.2 V1/2 −10 to −20 mV).{' '}
        LOF (biallelic): <strong>SANDD</strong> (Sinoatrial node Dysfunction and Deafness — cochlear IHC synaptic failure + SA node bradycardia/SSS) → pacemaker + cochlear implant.{' '}
        GOF (de novo dominant): <strong>DEE + autism + primary aldosteronism</strong> (~30%) + sinus tachycardia; NO LQTS8 (unlike CACNA1C Timothy Syndrome).{' '}
        <strong>Cav1 subfamily:</strong> Cav1.1/CACNA1S (skeletal) · Cav1.2/CACNA1C (cardiac/TS-LQTS8) · <strong>Cav1.3/CACNA1D (cochlear+pacemaker/SANDD+DEE)</strong> · Cav1.4/CACNA1F (retinal/CSNB2).{' '}
        <span style={{ color: DANGER }} className="fw-bold">
          PRECISION (GOF): Isradipine (DHP Cav1.3-preferential; NOT verapamil — verapamil is for CACNA1C/Cav1.2).{' '}
          SANDD LOF: Cochlear implant + Pacemaker (isradipine ABSOLUTE CI in LOF — worsens bradycardia).{' '}
          ABSOLUTE CI: TGB (NCSE) · VPA+POLG1 (Alpers) · Isradipine-in-LOF (cardioinhibitory) · VGB-long-term (VFD — devastating in deaf SANDD patient).
        </span>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={data.total_patients} />
        <KPI label="Seizure-Free" value={`${data.seizure_free_pct}%`} color={SUCCESS} />
        <KPI label="Drug-Resistant" value={`${data.dre_pct}%`} color={DANGER} />
        <KPI label="GOF-DEE" value={data.gof_count} color={COLOR} />
        <KPI label="SANDD (LOF)" value={data.sandd_count} color={WARN} />
        <KPI label="Aldosteronism" value={data.aldosteronism_count} color={WARN} />
        <KPI label="Pacemaker" value={data.pacemaker_count} color={DANGER} />
        <KPI label="Cochlear Implant" value={data.cochlear_implant_count} color={COLOR} />
        <KPI label="Isradipine Rx" value={data.isradipine_rx_count} color={COLOR} />
        <KPI label="ASD Dx" value={data.asd_count} color={COLOR} />
      </div>

      <div className="row g-3">
        {/* Etiology */}
        <div className="col-md-6">
          <SectionCard title="Etiology Distribution (GOF-LOF Spectrum)">
            {etioEntries.map((e, i) => (
              <div key={i} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{e.category.replace(/-/g, ' ')}</span>
                  <span className="text-muted">{e.count} pts</span>
                </div>
                <div className="progress" style={{ height: 12 }}>
                  <div className="progress-bar" style={{ width: `${Math.round(e.count / 40 * 100)}%`, backgroundColor: COLOR }} />
                </div>
              </div>
            ))}
          </SectionCard>
        </div>

        {/* Seizure Summary */}
        <div className="col-md-6">
          <SectionCard title="Seizure Types (% of cohort)">
            {seizureSummary.map((st, i) => (
              <Bar key={i} label={st.type.replace(/-/g, ' ')} value={st.frequency_pct} max={100} color={WARN} />
            ))}
          </SectionCard>
        </div>

        {/* Contraindications */}
        <div className="col-md-6">
          <SectionCard title="Contraindications (CACNA1D-Specific)" borderColor={DANGER}>
            <table className="table table-sm table-hover mb-0">
              <thead><tr><th>Drug/Class</th><th>Risk</th></tr></thead>
              <tbody>
                {ciSummary.map((ci, i) => (
                  <tr key={i}>
                    <td className="small">{ci.drug}</td>
                    <td><span className="badge" style={{ background: DANGER, fontSize: '0.7rem', whiteSpace: 'normal' }}>{ci.risk?.slice(0, 60)}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </SectionCard>
        </div>

        {/* Treatments */}
        <div className="col-md-6">
          <SectionCard title="Treatments (Evidence Levels)">
            <table className="table table-sm table-hover mb-0">
              <thead><tr><th>Drug</th><th>Level</th></tr></thead>
              <tbody>
                {treatments.map((t, i) => (
                  <tr key={i}>
                    <td className="small">{t.drug}</td>
                    <td><span className="badge" style={{ background: COLOR, fontSize: '0.7rem' }}>{t.level?.split(' (')[0]}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </SectionCard>
        </div>

        {/* Monitoring */}
        <div className="col-md-6">
          <SectionCard title="Monitoring Priorities (Top 8)">
            <ul className="list-unstyled mb-0 small">
              {monitoring.map((m, i) => (
                <li key={i} className="mb-1">
                  <span className="fw-semibold" style={{ color: COLOR }}>{m.item?.slice(0, 55)}</span>
                  <br /><span className="text-muted">{m.frequency?.slice(0, 60)}</span>
                </li>
              ))}
            </ul>
          </SectionCard>
        </div>

        {/* Lifecycle */}
        <div className="col-md-6">
          <SectionCard title="Lifecycle Windows">
            <ul className="list-unstyled mb-0 small">
              {lifecycle.map((lc, i) => (
                <li key={i} className="mb-1">
                  <span className="fw-semibold" style={{ color: COLOR }}>{lc.stage}</span>
                  <br /><span className="text-muted">{lc.key_action}</span>
                </li>
              ))}
            </ul>
          </SectionCard>
        </div>

        {/* Thresholds */}
        <div className="col-12">
          <SectionCard title="Clinical Thresholds">
            <div className="row row-cols-2 row-cols-md-3 g-2">
              {thresholds.map((th, i) => (
                <div key={i} className="col">
                  <div className="border rounded p-2 h-100">
                    <div className="fw-semibold small" style={{ color: WARN }}>{th.threshold}</div>
                    <div className="small text-muted">{th.action}</div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

function BreakdownTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;

  return (
    <div>
      {/* Etiology Catalog */}
      <SectionCard title="Etiology Catalog — 5-Class GOF-LOF Spectrum">
        <div className="row g-3">
          {(data.etiologies || []).map((e, i) => (
            <div key={i} className="col-md-6">
              <div className="border rounded p-3 h-100">
                <div className="fw-bold mb-1" style={{ color: COLOR }}>{e.category.replace(/-/g, ' ')} — {e.pct}%</div>
                <div className="small"><strong>Mechanism:</strong> {e.mechanism}</div>
                <div className="small mt-1"><strong>Phenotype:</strong> {e.phenotype}</div>
                <div className="small mt-1"><strong>EEG:</strong> {e.eeg_pattern}</div>
                <div className="small mt-1"><strong>Severity:</strong> {e.severity}</div>
                <div className="small mt-1 text-muted"><em>{e.reference}</em></div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Patients Table */}
      <SectionCard title="Patient Cohort (40 Patients — Synthetic)">
        <div className="table-responsive">
          <table className="table table-sm table-hover table-striped mb-0" style={{ fontSize: '0.78rem' }}>
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Etiology</th><th>Age</th><th>Variant</th>
                <th>GOF</th><th>SANDD</th><th>Aldosteronism</th><th>BP sys</th>
                <th>K⁺</th><th>Pacemaker</th><th>CI</th><th>Isradipine</th>
                <th>ASD</th><th>SeizFree</th><th>DRE</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map((p, i) => (
                <tr key={i}>
                  <td>{p.patient_id}</td>
                  <td style={{ maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.etiology?.replace(/-/g, ' ')}</td>
                  <td>{p.age_years}y</td>
                  <td>{p.variant_class}</td>
                  <td>{p.etiology?.includes('GOF') ? '✓' : '—'}</td>
                  <td>{p.sandd_deafness ? '✓' : '—'}</td>
                  <td style={{ color: p.aldosteronism ? WARN : 'inherit' }}>{p.aldosteronism ? '✓' : '—'}</td>
                  <td style={{ color: p.systolic_bp > 140 ? DANGER : 'inherit' }}>{p.systolic_bp}</td>
                  <td style={{ color: p.k_mmol_l < 3.5 ? WARN : 'inherit' }}>{p.k_mmol_l}</td>
                  <td>{p.pacemaker_implanted ? '✓' : '—'}</td>
                  <td>{p.cochlear_implant ? '✓' : '—'}</td>
                  <td>{p.isradipine_rx ? '✓' : '—'}</td>
                  <td>{p.asd_diagnosis ? '✓' : '—'}</td>
                  <td style={{ color: p.seizure_free ? SUCCESS : 'inherit' }}>{p.seizure_free ? '✓' : '—'}</td>
                  <td style={{ color: p.dre ? DANGER : 'inherit' }}>{p.dre ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function SeizuresTriggerTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;

  return (
    <div>
      {/* Seizure Types */}
      <SectionCard title="Seizure Types — 5-Type Profile">
        <div className="row g-3">
          {(data.seizure_types || []).map((st, i) => (
            <div key={i} className="col-md-6">
              <div className="border rounded p-3 h-100">
                <div className="d-flex justify-content-between align-items-center mb-1">
                  <span className="fw-bold" style={{ color: COLOR }}>{st.type.replace(/-/g, ' ')}</span>
                  <span className="badge" style={{ background: WARN }}>{st.frequency_pct}%</span>
                </div>
                <div className="small"><strong>Onset age:</strong> {st.onset_age}</div>
                <div className="small mt-1"><strong>EEG:</strong> {st.eeg}</div>
                <div className="small mt-1"><strong>Semiology:</strong> {st.semiology}</div>
                <div className="small mt-1 p-1 rounded" style={{ background: '#e8eaf6', color: COLOR }}><em>Tip: {st.clinical_tip}</em></div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Triggers */}
      <SectionCard title="Seizure Triggers — 8 Triggers">
        <div className="row g-3">
          {(data.triggers || []).map((tr, i) => (
            <div key={i} className="col-md-6">
              <div className="border rounded p-3 h-100">
                <div className="d-flex justify-content-between align-items-center mb-1">
                  <span className="fw-bold" style={{ color: WARN }}>{tr.trigger}</span>
                  <span className="badge" style={{ background: COLOR }}>{tr.rate_pct}%</span>
                </div>
                <div className="small text-muted">{tr.note}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const [expanded, setExpanded] = useState(null);

  return (
    <div>
      {/* Contraindications */}
      <SectionCard title="Contraindications — 6 CACNA1D-Specific CIs" borderColor={DANGER}>
        <div className="row g-3 mb-2">
          {(data.contraindications || []).map((ci, i) => (
            <div key={i} className="col-md-6">
              <div className="border border-danger rounded p-3 h-100">
                <div className="fw-bold mb-1" style={{ color: DANGER }}>{ci.drug.replace(/-/g, ' ')}</div>
                <span className="badge mb-2" style={{ background: DANGER }}>{ci.level?.split(' —')[0]}</span>
                <div className="small"><strong>Risk:</strong> {ci.risk}</div>
                <div className="small mt-1"><strong>Mechanism:</strong> {ci.mechanism}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Treatments */}
      <SectionCard title="Treatments — 8 Agents (Evidence + CACNA1D-Specific Notes)">
        {(data.treatments || []).map((t, i) => (
          <div key={i} className="border rounded mb-2">
            <button
              className="btn btn-link w-100 text-start fw-semibold py-2 px-3 d-flex justify-content-between"
              style={{ color: COLOR, textDecoration: 'none' }}
              onClick={() => setExpanded(expanded === i ? null : i)}
            >
              <span>{t.drug?.split(' (')[0]}</span>
              <span className="badge" style={{ background: COLOR, fontSize: '0.7rem' }}>{t.level?.split(' (')[0]?.split(' —')[0]}</span>
            </button>
            {expanded === i && (
              <div className="px-3 pb-3 small">
                <div><strong>Level of evidence:</strong> {t.level}</div>
                <div className="mt-1"><strong>MOA:</strong> {t.moa}</div>
                <div className="mt-1"><strong>Dose:</strong> {t.dose}</div>
                <div className="mt-1"><strong>Efficacy:</strong> {t.efficacy}</div>
                <div className="mt-1"><strong>Monitoring:</strong> {t.monitoring}</div>
                <div className="mt-1 p-2 rounded" style={{ background: '#e8eaf6', color: COLOR }}><strong>CACNA1D note:</strong> {t.cacna1d_note}</div>
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      {/* Monitoring */}
      <SectionCard title="Monitoring — 14 Items (CACNA1D + Cardiac + Aldosterone)">
        <div className="row g-3">
          {(data.monitoring || []).map((m, i) => (
            <div key={i} className="col-md-6">
              <div className="border rounded p-2 h-100">
                <div className="fw-semibold small" style={{ color: COLOR }}>{m.item}</div>
                <div className="small text-muted">{m.frequency}</div>
                <div className="small mt-1" style={{ color: WARN }}>{m.rationale?.slice(0, 80)}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const gs = data.gene_summary || {};

  return (
    <div>
      {/* Gene Summary Card */}
      <SectionCard title="CACNA1D — Gene Summary">
        <div className="row g-2 small">
          {Object.entries(gs).map(([k, v]) => (
            <div key={k} className="col-md-6">
              <span className="fw-semibold text-capitalize" style={{ color: COLOR }}>{k.replace(/_/g, ' ')}: </span>
              <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Key Concepts */}
      <SectionCard title="Key Concepts — 15 Definitions">
        <div className="row g-3">
          {(data.definitions || []).map((d, i) => (
            <div key={i} className="col-md-6">
              <div className="border rounded p-3 h-100">
                <div className="fw-bold mb-1" style={{ color: COLOR }}>{d.term}</div>
                <div className="small text-muted">{d.definition}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Thresholds */}
      <SectionCard title="Clinical Thresholds — 12">
        <div className="row g-2">
          {(data.thresholds || []).map((th, i) => (
            <div key={i} className="col-md-6">
              <div className="border rounded p-2 h-100">
                <div className="fw-semibold small" style={{ color: WARN }}>{th.threshold}</div>
                <div className="small text-muted">{th.action}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Standards + References */}
      <div className="row g-3">
        <div className="col-md-6">
          <SectionCard title="Evidence Standards — 12">
            <ul className="list-unstyled mb-0 small">
              {(data.standards || []).map((s, i) => (
                <li key={i} className="mb-1">
                  <span className="fw-semibold" style={{ color: COLOR }}>{s.name}:</span>{' '}
                  <span className="text-muted">{s.applies}</span>
                </li>
              ))}
            </ul>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="References — 6">
            <ul className="list-unstyled mb-0 small">
              {(data.references || []).map((r, i) => (
                <li key={i} className="mb-2">
                  <span className="fw-semibold" style={{ color: COLOR }}>{r.author} ({r.year})</span>{' '}
                  <span className="text-muted">{r.journal} — {r.title}</span>
                  {r.pmid && <span className="badge ms-1" style={{ background: COLOR, fontSize: '0.6rem' }}>PMID:{r.pmid}</span>}
                </li>
              ))}
            </ul>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

export default function CACNA1DPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/cacna1d/overview`).then(r => r.json()),
      fetch(`${API}/api/cacna1d/breakdown`).then(r => r.json()),
      fetch(`${API}/api/cacna1d/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border" style={{ color: COLOR }} /></div>;
  if (error) return <div className="alert alert-danger m-3">{error}</div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="rounded p-3 mb-3 text-white" style={{ background: COLOR }}>
        <h4 className="mb-0 fw-bold">🧬 CACNA1D Epilepsy — SANDD Syndrome / DEE + Autism + Primary Aldosteronism</h4>
        <div className="small opacity-75 mt-1">
          Cav1.3 (α1D) Low-Threshold L-type HVA Ca²⁺ Channel · V1/2 ≈ −40 to −55 mV · 3p14.3
          · OMIM #614896 SANDD · GOF: Isradipine-Precision · LOF: Cochlear Implant + Pacemaker · 40-Patient Cohort
        </div>
        <div className="small opacity-75">
          Cav1 Subfamily: Cav1.1/CACNA1S (skeletal) · Cav1.2/CACNA1C (cardiac/LQTS8) · <strong>Cav1.3/CACNA1D (cochlear+pacemaker/SANDD+DEE) ←</strong> · Cav1.4/CACNA1F (retinal/CSNB2)
        </div>
        <div className="small opacity-75">
          KEY: Cav1.3 GOF → isradipine (NOT verapamil — verapamil is Cav1.2/CACNA1C); LOF → NO QTc prolongation (distinct from TS LQTS8); deafness + bradycardia in SANDD
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* Tab Content */}
      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <BreakdownTab data={breakdown} />}
      {tab === 2 && <SeizuresTriggerTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
