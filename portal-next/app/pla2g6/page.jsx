'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Cerebellar & Neuropathy', 'Treatments', 'Definitions'];
const COLOR = '#1b5e20';   // deep green — PLAN/NBIA2 (phospholipid green)
const LIGHT = '#e8f5e9';

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

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview...</div>;
  const kpis = data.kpis || {};
  const etiologies = data.etiology_distribution || [];
  const treatments = data.treatments_summary || [];
  const monitoring = data.monitoring_summary || [];
  const lifecycle = data.lifecycle || [];
  const thresholds = data.thresholds || [];
  const cis = data.contraindications_summary || [];
  const ddx = data.ddx_table || [];
  const highlights = data.clinical_highlights || [];
  const tier = data.tier_summary || {};
  const maxEtio = Math.max(...etiologies.map(e => e.pct || 0), 1);

  return (
    <div>
      <div className="alert py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
        <strong>PLA2G6 (22q13.1) — 806aa · iPLA2β · Ankyrin repeats (aa 1-290) + Patatin-like domain (aa 461-799) · OMIM 603604/256600/610217/612953 · PLAN/NBIA2:</strong>{' '}
        3rd most common NBIA (~5-15%). AR biallelic PLA2G6 → phospholipid remodeling failure → mitochondrial membrane dysfunction → iron accumulation GP/SN (LATE).{' '}
        <strong className="text-danger">NO eye-of-tiger sign (key DDx PKAN). Cerebellar cortical atrophy EARLIEST + MOST PROMINENT MRI finding.</strong>{' '}
        Axonal neuropathy 100% INAD / 82% ANAD (NCS MANDATORY). Spheroid bodies on nerve biopsy PATHOGNOMONIC.{' '}
        <span className="fw-bold" style={{ color: COLOR }}>
          3 phenotypes: Classic INAD (onset 6mo-3yr) · Atypical NAD (1-5yr) · PARK14 adult parkinsonism.
          PHT ABSOLUTE CI. VGB ABSOLUTE CI (optic atrophy additive). POLG mandatory before VPA. GPi-DBS Level C. Deferiprone investigational.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Classic INAD" value={kpis.n_inad} color="#dc3545" />
        <KPI label="Atypical NAD" value={kpis.n_anad} color="#e65100" />
        <KPI label="PARK14" value={kpis.n_park14} color="#6f42c1" />
        <KPI label="Cerebellar Atrophy" value={`${kpis.cerebellar_atrophy_pct}%`} color="#dc3545" />
        <KPI label="Optic Atrophy" value={`${kpis.optic_atrophy_pct}%`} color="#dc3545" />
        <KPI label="Axonal Neuropathy" value={`${kpis.axonal_neuropathy_pct}%`} color="#e65100" />
        <KPI label="Spastic Paraparesis" value={`${kpis.spastic_paraparesis_pct}%`} color="#e65100" />
        <KPI label="Dystonia" value={`${kpis.dystonia_pct}%`} color={COLOR} />
        <KPI label="Has Seizures" value={`${kpis.has_seizures_pct}%`} color="#e65100" />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="Lost Ambulation" value={`${kpis.ambulation_lost_pct}%`} color="#dc3545" />
        <KPI label="Cognitive Decline" value={`${kpis.cognitive_decline_pct}%`} color={COLOR} />
        <KPI label="Baclofen" value={`${kpis.baclofen_pct}%`} color="#0d6efd" />
        <KPI label="GPi-DBS" value={`${kpis.dbs_pct}%`} color="#0d6efd" />
        <KPI label="POLG Tested" value={`${kpis.polg_tested_pct}%`} color="#0d6efd" />
        <KPI label="L-DOPA (PARK14)" value={`${kpis.levodopa_pct}%`} color="#6f42c1" />
        <KPI label="Seizure-Free" value={`${kpis.seizure_free_pct}%`} color="#198754" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Etiology Distribution (4 Categories)
            </div>
            <div className="card-body">
              {etiologies.map((e, i) => (
                <Bar key={i} label={`${e.etiology} (n=${e.n})`} value={e.pct} max={maxEtio} />
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Clinical Highlights
            </div>
            <div className="card-body p-2">
              <ul className="small mb-0 ps-3">
                {highlights.map((h, i) => <li key={i} className="mb-1">{h}</li>)}
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small text-danger" style={{ background: '#fff3cd' }}>
              Contraindications
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0 small">
                <thead><tr><th>Drug</th><th>Reason</th></tr></thead>
                <tbody>
                  {cis.map((c, i) => (
                    <tr key={i}>
                      <td className="fw-bold text-danger text-nowrap">{c.drug}</td>
                      <td>{c.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Monitoring Schedule
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0 small">
                <thead><tr><th>Item</th><th>Frequency</th></tr></thead>
                <tbody>
                  {monitoring.map((m, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{m.item}</td>
                      <td className="text-muted">{m.frequency}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-12">
          <div className="card shadow-sm">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Disease Lifecycle (INAD → ANAD → PARK14 trajectories)
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0 small">
                <thead><tr><th>Stage</th><th>Clinical Features</th></tr></thead>
                <tbody>
                  {lifecycle.map((l, i) => (
                    <tr key={i} style={i >= 4 ? { background: '#fce4ec' } : {}}>
                      <td className="fw-semibold text-nowrap">{l.stage}</td>
                      <td>{l.features}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-12">
          <div className="card shadow-sm">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              DDx — PLAN vs Other NBIA + Cerebellar/Neuropathic Syndromes
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0 small">
                <thead>
                  <tr><th>Condition</th><th>Distinguishing from PLAN</th><th>Shared</th></tr>
                </thead>
                <tbody>
                  {ddx.map((d, i) => (
                    <tr key={i}>
                      <td className="fw-semibold text-nowrap">{d.condition}</td>
                      <td>{d.distinguishing}</td>
                      <td className="text-muted">{d.shared}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-12">
          <div className="card shadow-sm">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Clinical Thresholds
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0 small">
                <thead><tr><th>Parameter</th><th>Threshold</th><th>Significance</th></tr></thead>
                <tbody>
                  {thresholds.map((t, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{t.parameter}</td>
                      <td><code>{t.threshold}</code></td>
                      <td className="text-muted">{t.significance}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Tier Summary
        </div>
        <div className="card-body p-2">
          <table className="table table-sm mb-0 small">
            <tbody>
              {Object.entries(tier).map(([k, v]) => (
                <tr key={k}>
                  <td className="fw-semibold text-capitalize">{k.replace(/_/g, ' ')}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function PhenotypeTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const etio = data.etiology_breakdown || [];
  const pheno = data.phenotype_breakdown || [];
  const pts = data.per_patient || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Etiology Breakdown — 4 Categories (n=40, seed-521)
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead>
              <tr>
                <th>Etiology</th><th>n</th><th>%</th>
                <th>INAD %</th><th>Cerebellar Atr.</th>
                <th>Axonal Neuro.</th><th>Seizures</th>
                <th>DBS</th><th>Mean Onset (yr)</th>
              </tr>
            </thead>
            <tbody>
              {etio.map((e, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{e.etiology}</td>
                  <td>{e.n}</td>
                  <td>{e.pct}%</td>
                  <td>{e.inad_pct !== undefined ? `${e.inad_pct}%` : '—'}</td>
                  <td>{e.cerebellar_atrophy_pct !== undefined ? `${e.cerebellar_atrophy_pct}%` : '100%'}</td>
                  <td>{e.axonal_neuropathy_pct !== undefined ? `${e.axonal_neuropathy_pct}%` : '—'}%</td>
                  <td>{e.has_seizures_pct !== undefined ? `${e.has_seizures_pct}%` : '—'}</td>
                  <td>{e.dbs_pct !== undefined ? `${e.dbs_pct}%` : '—'}</td>
                  <td>{e.mean_onset_yr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {pheno.length > 0 && (
        <div className="card shadow-sm mb-4">
          <div className="card-header fw-bold small" style={{ background: LIGHT }}>
            Phenotype Breakdown — INAD / ANAD / PARK14
          </div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0 small">
              <thead>
                <tr>
                  <th>Phenotype</th><th>n</th><th>%</th>
                  <th>Onset (yr)</th><th>Axonal Neuro.</th>
                  <th>Seizures</th><th>Optic Atr.</th><th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {pheno.map((p, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{p.phenotype}</td>
                    <td>{p.n}</td>
                    <td>{p.pct}%</td>
                    <td>{p.mean_onset_yr}</td>
                    <td>{p.axonal_neuropathy_pct}%</td>
                    <td>{p.seizures_pct}%</td>
                    <td>{p.optic_atrophy_pct}%</td>
                    <td className="text-muted">{p.notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Per-Patient Summary (n=40)
        </div>
        <div className="card-body p-0">
          <div style={{ maxHeight: 420, overflowY: 'auto' }}>
            <table className="table table-sm mb-0 small">
              <thead>
                <tr>
                  <th>ID</th><th>Phenotype</th><th>Etiology</th><th>Onset(yr)</th>
                  <th>Cereb.</th><th>Opt.Atr.</th><th>Ax.Neuro.</th>
                  <th>Spast.</th><th>Dystonia</th><th>Amb.Lost</th>
                  <th>Sz</th><th>DR</th><th>AEDs</th>
                  <th>Baclo.</th><th>L-DOPA</th><th>DBS</th>
                  <th>Cogn.</th><th>Psych.</th><th>POLG</th><th>SF</th>
                </tr>
              </thead>
              <tbody>
                {pts.map((p, i) => {
                  const phBg = p.phenotype === 'INAD' ? '#ffebee' : p.phenotype === 'ANAD' ? '#fff3e0' : '#f3e5f5';
                  return (
                    <tr key={i} style={{ background: phBg }}>
                      <td className="fw-semibold">{p.id}</td>
                      <td>
                        <span className="badge" style={{
                          background: p.phenotype === 'INAD' ? '#dc3545' : p.phenotype === 'ANAD' ? '#e65100' : '#6f42c1'
                        }}>
                          {p.phenotype}
                        </span>
                      </td>
                      <td>{p.etiology}</td>
                      <td>{p.onset_yr}</td>
                      <td>{p.cerebellar_atrophy ? '🔴' : '–'}</td>
                      <td>{p.optic_atrophy ? '🔴' : '–'}</td>
                      <td>{p.axonal_neuropathy ? '⚠️' : '–'}</td>
                      <td>{p.spastic_paraparesis ? '⚠️' : '–'}</td>
                      <td>{p.dystonia_severity || '–'}</td>
                      <td>{p.ambulation_lost ? '⚠️' : '–'}</td>
                      <td>{p.has_seizures ? '✓' : '–'}</td>
                      <td>{p.drug_resistant ? '🔴' : '–'}</td>
                      <td>{p.n_aeds}</td>
                      <td>{p.baclofen ? '✓' : '–'}</td>
                      <td>{p.levodopa ? '✓' : '–'}</td>
                      <td>{p.dbs ? '✓' : '–'}</td>
                      <td>{p.cognitive_decline ? '✓' : '–'}</td>
                      <td>{p.psychiatric ? '✓' : '–'}</td>
                      <td>{p.polg_tested ? '✓' : '–'}</td>
                      <td>{p.seizure_free ? '✓' : '–'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function CerebellarNeuropathyTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const sz = data.seizure_breakdown || [];
  const lc = data.lifecycle || [];
  const thr = data.thresholds || [];
  const refs = data.references || [];
  const stds = data.standards || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Seizure Type Breakdown (57% of PLAN patients have seizures — INAD highest; n=40)
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead>
              <tr><th>Type</th><th>n</th><th>%</th><th>Drug-Resistant (%)</th></tr>
            </thead>
            <tbody>
              {sz.map((s, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{s.type}</td>
                  <td>{s.n}</td>
                  <td>{s.pct}%</td>
                  <td>{s.drug_resistant_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="alert alert-info small mb-4">
        <strong>Cerebellar Atrophy &amp; Axonal Neuropathy Key Facts (PLAN):</strong>
        <ul className="mb-0 mt-1 ps-3">
          <li><strong>Cerebellar cortical atrophy (100%):</strong> EARLIEST + MOST PROMINENT MRI finding. Purkinje cell loss. Spinocerebellar tracts. Differentiates PLAN from PKAN (GP dominant).</li>
          <li><strong>GP iron (T2/SWI hypointensity):</strong> Appears LATE — may be ABSENT at onset of INAD. Key DDx from PKAN where GP iron is early and pathognomonic (eye-of-tiger).</li>
          <li><strong>NO eye-of-tiger sign:</strong> GP T2 central hyperintense + hypointense rim is ABSENT in PLAN. Presence of eye-of-tiger should redirect to PKAN (PANK2).</li>
          <li><strong>Axonal neuropathy (100% INAD, ~82% ANAD):</strong> NCS shows reduced motor + sensory amplitudes, NORMAL conduction velocity. EMG: chronic denervation distal muscles. NCS MANDATORY at diagnosis.</li>
          <li><strong>Spheroid bodies on nerve biopsy:</strong> Axonal spheroids in peripheral nerve = PATHOGNOMONIC for PLAN (pre-molecular era diagnostic). Still used when gene panel negative.</li>
          <li><strong>Optic atrophy (50% overall, 70% INAD):</strong> VEP prolonged. OCT RNFL thinning. Annual ophthalmology mandatory. VGB ABSOLUTE CI — additive risk to existing optic atrophy.</li>
        </ul>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: '#e8eaf6' }}>
          Clinical Lifecycle — INAD / ANAD / PARK14
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead><tr><th>Stage</th><th>Clinical Features</th></tr></thead>
            <tbody>
              {lc.map((l, i) => (
                <tr key={i} style={i >= 4 ? { background: '#fce4ec' } : {}}>
                  <td className="fw-semibold text-nowrap">{l.stage}</td>
                  <td>{l.features}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Standards &amp; References
        </div>
        <div className="card-body p-2 small">
          <p className="fw-semibold mb-1">Clinical Standards:</p>
          <ul className="ps-3 mb-2">
            {stds.map((s, i) => <li key={i}>{s}</li>)}
          </ul>
          <p className="fw-semibold mb-1">Key References:</p>
          <ol className="ps-3 mb-0">
            {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
          </ol>
        </div>
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const tx = data.treatment_summary || [];
  const cis = data.contraindications || [];
  const mon = data.monitoring || [];

  const levelColor = (lv) => ({
    'A': '#198754', 'B': '#0d6efd', 'C': '#fd7e14',
    'INV': '#6f42c1', 'CI': '#dc3545'
  }[lv] || '#6c757d');

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          AED &amp; Symptomatic Treatment Summary
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead>
              <tr><th>Drug</th><th>Level</th><th>Tried %</th><th>Responder %</th><th>Notes</th></tr>
            </thead>
            <tbody>
              {tx.map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{t.drug}</td>
                  <td>
                    <span className="badge" style={{ background: levelColor(t.level) }}>
                      {t.level}
                    </span>
                  </td>
                  <td>{t.tried_pct}%</td>
                  <td>{t.responder_pct}%</td>
                  <td className="text-muted">{t.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small text-danger" style={{ background: '#fff3cd' }}>
          Contraindications — ABSOLUTE CI &amp; AVOID
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead><tr><th>Drug</th><th>Reason</th></tr></thead>
            <tbody>
              {cis.map((c, i) => (
                <tr key={i}>
                  <td className="fw-bold text-danger text-nowrap">{c.drug}</td>
                  <td>{c.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Full Monitoring Schedule
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead><tr><th>Item</th><th>Frequency</th><th>Rationale</th></tr></thead>
            <tbody>
              {mon.map((m, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{m.item}</td>
                  <td className="text-nowrap">{m.frequency}</td>
                  <td className="text-muted">{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const defs = data.definitions || [];
  const refs = data.references || [];
  const stds = data.standards || [];
  const cis = data.contraindications || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Key Concepts — {defs.length} Definitions
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead><tr><th style={{ width: '28%' }}>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {defs.map((d, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ color: COLOR }}>{d.term}</td>
                  <td>{d.def}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small text-danger" style={{ background: '#fff3cd' }}>
          Contraindications Summary
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead><tr><th>Drug</th><th>Reason</th></tr></thead>
            <tbody>
              {cis.map((c, i) => (
                <tr key={i}>
                  <td className="fw-bold text-danger text-nowrap">{c.drug}</td>
                  <td>{c.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Standards &amp; References
        </div>
        <div className="card-body p-2 small">
          <p className="fw-semibold mb-1">Clinical Standards ({stds.length}):</p>
          <ul className="ps-3 mb-2">
            {stds.map((s, i) => <li key={i}>{s}</li>)}
          </ul>
          <p className="fw-semibold mb-1">References ({refs.length}):</p>
          <ol className="ps-3 mb-0">
            {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
          </ol>
        </div>
      </div>
    </div>
  );
}

export default function PLA2G6Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/pla2g6/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(() => setError('Backend offline — start: bash scripts/restart_backend.sh'));
    fetch(`${API}/api/pla2g6/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (breakdown) {
      fetch(`${API}/api/pla2g6/definitions`)
        .then(r => r.json())
        .then(d => {
          setBreakdown(prev => prev ? {
            ...prev,
            definitions: d.definitions || [],
            references: d.references || [],
            standards: d.standards || [],
            contraindications: d.contraindications || [],
          } : prev);
        })
        .catch(() => {});
    }
  }, [breakdown]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3" style={{ borderLeft: `6px solid ${COLOR}`, paddingLeft: 12 }}>
        <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
          PLA2G6 PLAN — PLA2G6-Associated Neurodegeneration (NBIA2 / INAD1 / PARK14)
        </h4>
        <div className="text-muted small">
          PLA2G6 (iPLA2β — calcium-independent phospholipase A2 beta) · 22q13.1 · OMIM 603604/256600/610217/612953 · Autosomal Recessive ·
          40-patient cohort (seed-521) · 3rd most common NBIA (~5-15%) ·
          Classic INAD (60%, onset 6mo-3yr) · Atypical NAD (28%, 1-5yr) · PARK14 (12%, 30-50yr) ·
          NO eye-of-tiger sign (key DDx PKAN) · Cerebellar atrophy EARLIEST MRI · Axonal neuropathy 100% INAD ·
          Spheroid bodies PATHOGNOMONIC · PHT ABSOLUTE CI · VGB ABSOLUTE CI · POLG mandatory ·
          GPi-DBS Level C · Deferiprone investigational
        </div>
      </div>

      {error && <div className="alert alert-danger small py-2">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PhenotypeTab data={breakdown} />}
      {tab === 2 && <CerebellarNeuropathyTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={breakdown} />}
    </div>
  );
}
