'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Neuropathy & Spasticity', 'Treatments', 'Definitions'];
const COLOR = '#4a148c';   // deep purple — MPAN/NBIA4
const LIGHT = '#f3e5f5';

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
        <strong>C19orf12 (19q12) — 152aa · 2 TM domains · ER-mito contact sites (MAMs) · OMIM 614297/614298 · MPAN/NBIA4:</strong>{' '}
        2nd most common NBIA (~20-35%). AR biallelic C19orf12 → MAM dysfunction + iron-sulfur cluster failure → GP+SN iron.{' '}
        <strong className="text-danger">NO eye-of-tiger sign (key DDx from PKAN). Optic atrophy 80% KEY distinguishing feature.</strong>{' '}
        Motor axonal neuropathy 60%. Pyramidal signs 100%. Cognitive decline 80%.{' '}
        <span className="fw-bold" style={{ color: COLOR }}>
          p.Gly69Arg: Polish/Slavic founder (~30-40% European alleles).
          PHT AVOID. POLG mandatory before VPA. GPi-DBS Level C.
          Deferiprone investigational (MPAN-specific trial 2024-2026).
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Juvenile MPAN" value={kpis.n_juvenile} color={COLOR} />
        <KPI label="Adult-onset MPAN" value={kpis.n_adult} color="#7b1fa2" />
        <KPI label="Optic Atrophy" value={`${kpis.optic_atrophy_pct}%`} color="#dc3545" />
        <KPI label="Axonal Neuropathy" value={`${kpis.axonal_neuropathy_pct}%`} color="#e65100" />
        <KPI label="Has Seizures" value={`${kpis.has_seizures_pct}%`} color="#e65100" />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="Severe Dystonia" value={`${kpis.dystonia_severe_pct}%`} color="#dc3545" />
        <KPI label="Lost Ambulation" value={`${kpis.ambulation_lost_pct}%`} color="#dc3545" />
        <KPI label="Cognitive Decline" value={`${kpis.cognitive_decline_pct}%`} color={COLOR} />
        <KPI label="Parkinsonism" value={`${kpis.parkinsonism_pct}%`} color="#6f42c1" />
        <KPI label="Baclofen" value={`${kpis.baclofen_pct}%`} color="#0d6efd" />
        <KPI label="GPi-DBS" value={`${kpis.dbs_pct}%`} color="#0d6efd" />
        <KPI label="POLG Tested" value={`${kpis.polg_tested_pct}%`} color="#0d6efd" />
        <KPI label="Seizure-Free" value={`${kpis.seizure_free_pct}%`} color="#198754" />
        <KPI label="Mean Onset (yr)" value={kpis.mean_onset_yr} color={COLOR} />
        <KPI label="Juvenile Onset (yr)" value={kpis.juvenile_mean_onset_yr} color="#dc3545" />
        <KPI label="Adult Onset (yr)" value={kpis.adult_mean_onset_yr} color="#7b1fa2" />
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
              Disease Lifecycle (Juvenile &rarr; Adult-onset timelines)
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
              DDx — MPAN vs Other NBIA + Spastic/Neuropathic Syndromes
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0 small">
                <thead>
                  <tr><th>Condition</th><th>Distinguishing from MPAN</th><th>Shared</th></tr>
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

function EtiologyTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const etio = data.etiology_breakdown || [];
  const pts = data.per_patient || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Etiology Breakdown — 4 Categories (n=40, seed-519)
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead>
              <tr>
                <th>Etiology</th><th>n</th><th>%</th>
                <th>Juvenile %</th><th>Optic Atrophy</th>
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
                  <td>{e.juvenile_pct}%</td>
                  <td>{e.optic_atrophy_pct}%</td>
                  <td>{e.axonal_neuropathy_pct}%</td>
                  <td>{e.has_seizures_pct}%</td>
                  <td>{e.dbs_pct}%</td>
                  <td>{e.mean_onset_yr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Per-Patient Summary (n=40)
        </div>
        <div className="card-body p-0">
          <div style={{ maxHeight: 420, overflowY: 'auto' }}>
            <table className="table table-sm mb-0 small">
              <thead>
                <tr>
                  <th>ID</th><th>Form</th><th>Etiology</th><th>Onset(yr)</th>
                  <th>Opt.Atr.</th><th>Ax.Neuro.</th><th>Spast.</th>
                  <th>Dystonia</th><th>Amb.Lost</th>
                  <th>Seizures</th><th>DR</th><th>AEDs</th>
                  <th>Baclofen</th><th>DBS</th>
                  <th>Cogn.</th><th>Psych</th><th>Park.</th><th>POLG</th><th>SF</th>
                </tr>
              </thead>
              <tbody>
                {pts.map((p, i) => (
                  <tr key={i} style={p.optic_atrophy ? { background: '#f3e5f5' } : {}}>
                    <td className="fw-semibold">{p.id}</td>
                    <td>
                      <span className="badge" style={{ background: p.form === 'Juvenile' ? '#dc3545' : '#7b1fa2' }}>
                        {p.form}
                      </span>
                    </td>
                    <td>{p.etiology}</td>
                    <td>{p.onset_yr}</td>
                    <td>{p.optic_atrophy ? '🔴' : '–'}</td>
                    <td>{p.axonal_neuropathy ? '⚠️' : '–'}</td>
                    <td>{p.spasticity_severe ? '⚠️' : '–'}</td>
                    <td>{p.dystonia_severity}</td>
                    <td>{p.ambulation_lost ? '⚠️' : '–'}</td>
                    <td>{p.has_seizures ? '✓' : '–'}</td>
                    <td>{p.drug_resistant ? '🔴' : '–'}</td>
                    <td>{p.n_aeds}</td>
                    <td>{p.baclofen ? '✓' : '–'}</td>
                    <td>{p.dbs ? '✓' : '–'}</td>
                    <td>{p.cognitive_decline ? '✓' : '–'}</td>
                    <td>{p.psychiatric ? '✓' : '–'}</td>
                    <td>{p.parkinsonism ? '✓' : '–'}</td>
                    <td>{p.polg_tested ? '✓' : '–'}</td>
                    <td>{p.seizure_free ? '✓' : '–'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function NeuropathySpasticityTab({ data }) {
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
          Seizure Type Breakdown (25% of MPAN patients have seizures — secondary epilepsy, n=40)
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
        <strong>Neuropathy &amp; Spasticity Key Facts (MPAN):</strong>
        <ul className="mb-0 mt-1 ps-3">
          <li><strong>Motor axonal neuropathy (60%):</strong> NCS — reduced motor amplitudes, NORMAL CV. EMG: denervation in distal muscles. Combined UMN + LMN signs.</li>
          <li><strong>Pyramidal signs (100%):</strong> Spastic paraparesis, hyperreflexia, Babinski. ALL MPAN patients. Baclofen (oral/intrathecal) first-line.</li>
          <li><strong>Intrathecal baclofen (ITB):</strong> Candidacy: Ashworth ≥3 bilateral lower limbs + oral baclofen intolerance.</li>
          <li><strong>Optic atrophy (80%):</strong> VEP latency prolonged. OCT RNFL thinning detects subclinical. Annual ophthalmology from diagnosis.</li>
          <li><strong>GPi-DBS Level C:</strong> ~30-40% BFMDRS improvement (case series). Spastic-dystonic combination limits predictability. Less evidence vs PKAN (Level B).</li>
        </ul>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: '#e8eaf6' }}>
          Clinical Lifecycle — Juvenile &amp; Adult-onset MPAN
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
    'INV': '#6f42c1'
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

export default function C19orf12Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/c19orf12/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(() => setError('Backend offline — start: bash scripts/restart_backend.sh'));
    fetch(`${API}/api/c19orf12/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (breakdown) {
      fetch(`${API}/api/c19orf12/definitions`)
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
          C19orf12 MPAN — Mitochondrial Membrane Protein-Associated Neurodegeneration (NBIA4)
        </h4>
        <div className="text-muted small">
          C19orf12 (Mitochondrial Membrane Protein) · 19q12 · OMIM 614297/614298 · Autosomal Recessive ·
          40-patient cohort (seed-519) · 2nd most common NBIA (~20-35%) ·
          Juvenile MPAN (75%, onset 8-14yr) vs Adult-onset (25%, 15-30yr) ·
          NO eye-of-tiger sign (key DDx PKAN) · Optic atrophy 80% KEY · Motor axonal neuropathy 60% ·
          Pyramidal signs 100% · p.Gly69Arg Polish/Slavic founder · PHT AVOID · POLG mandatory ·
          GPi-DBS Level C · Deferiprone investigational 2024-2026
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
      {tab === 1 && <EtiologyTab data={breakdown} />}
      {tab === 2 && <NeuropathySpasticityTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={breakdown} />}
    </div>
  );
}
