'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

/* ─── colour helpers ─────────────────────────────────────────────────────── */
const ZONE_COLORS = {
  'Generalized (bilateral)': '#3b82f6',
  'Temporal (mesial)': '#8b5cf6',
  'Temporal (lateral/neocortical)': '#a78bfa',
  'Frontal (dorsolateral)': '#f59e0b',
  'Frontal (mesial/SMA)': '#fbbf24',
  'Frontal (orbitofrontal)': '#fde68a',
  'Multifocal': '#ef4444',
  'Occipital': '#22c55e',
  'Parietal': '#06b6d4',
  'Insular': '#ec4899',
};
const zoneColor = z => ZONE_COLORS[z] || '#6b7280';

const drugColor = s =>
  s.includes('Drug-resistant') ? '#ef4444' :
  s.includes('Partial') ? '#f59e0b' :
  s.includes('Drug-responsive') ? '#22c55e' : '#6b7280';

const freqColor = f =>
  f === 'Daily' ? '#ef4444' :
  f === 'Weekly' ? '#f59e0b' :
  f === 'Monthly' ? '#3b82f6' :
  f === 'Yearly' ? '#22c55e' :
  f.includes('free') ? '#10b981' : '#6b7280';

/* ─── stat card ─────────────────────────────────────────────────────────── */
function StatCard({ label, value, color = '#3b82f6', sub }) {
  return (
    <div className="col-6 col-md mb-2">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="h5 mb-0 fw-bold" style={{ color }}>{value}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

/* ─── horizontal bar ────────────────────────────────────────────────────── */
function HBar({ label, count, total, color }) {
  const pct = total ? Math.round((count / total) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-bold">{count} <span className="text-muted">({pct}%)</span></span>
      </div>
      <div style={{ background: '#e5e7eb', borderRadius: 4, height: 10 }}>
        <div style={{ width: `${pct}%`, background: color || '#3b82f6', borderRadius: 4, height: 10 }} />
      </div>
    </div>
  );
}

/* ─── badge ─────────────────────────────────────────────────────────────── */
function Chip({ label, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 12,
      fontSize: '0.7rem', fontWeight: 600, color: '#fff',
      background: color || '#6b7280', marginRight: 4, marginBottom: 4,
    }}>{label}</span>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   OVERVIEW TAB
═══════════════════════════════════════════════════════════════════════════ */
function OverviewTab({ ov }) {
  const kpis = ov.kpis || {};
  const total = kpis.total_patients || 0;

  return (
    <div>
      {/* KPI row */}
      <div className="row row-cols-2 row-cols-md-4 g-2 mb-4">
        <StatCard label="Patients" value={total} color="#3b82f6" />
        <StatCard label="Focal Epilepsy" value={kpis.focal_epilepsy}
          sub={`${kpis.focal_pct}%`} color="#8b5cf6" />
        <StatCard label="Generalized" value={kpis.generalized_epilepsy} color="#06b6d4" />
        <StatCard label="Drug-Resistant" value={kpis.drug_resistant}
          sub={`${kpis.drug_resistant_pct}%`} color="#ef4444" />
        <StatCard label="Surgery Candidates" value={kpis.surgery_candidates} color="#f59e0b" />
        <StatCard label="Seizure-Free" value={kpis.seizure_free} color="#22c55e" />
        <StatCard label="Avg Age at Onset" value={`${kpis.avg_age_at_onset}y`} color="#6366f1" />
      </div>

      <div className="row g-3">
        {/* Onset zone distribution */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">Onset Zone Distribution</div>
            <div className="card-body">
              {(ov.onset_zone_distribution || []).map(d => (
                <HBar key={d.zone} label={d.zone} count={d.count} total={total} color={zoneColor(d.zone)} />
              ))}
            </div>
          </div>
        </div>

        {/* Seizure type frequency */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">Seizure Type Frequency</div>
            <div className="card-body">
              {(ov.seizure_type_frequency || []).map((d, i) => {
                const colors = ['#3b82f6','#8b5cf6','#ef4444','#f59e0b','#22c55e','#06b6d4','#ec4899','#6366f1','#14b8a6','#f97316','#a3e635','#84cc16'];
                return (
                  <HBar key={d.type} label={d.type} count={d.count}
                    total={Math.max(...(ov.seizure_type_frequency || []).map(x => x.count))} color={colors[i % colors.length]} />
                );
              })}
            </div>
          </div>
        </div>

        {/* Age at onset histogram */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Age at Onset Distribution</div>
            <div className="card-body">
              <div className="d-flex align-items-end gap-2" style={{ height: 120 }}>
                {(ov.age_at_onset_histogram || []).map(d => {
                  const maxH = Math.max(...(ov.age_at_onset_histogram || []).map(x => x.count));
                  const h = Math.round((d.count / maxH) * 100);
                  return (
                    <div key={d.bucket} className="d-flex flex-column align-items-center flex-fill">
                      <div className="small fw-bold text-muted">{d.count}</div>
                      <div style={{ height: `${h}%`, background: '#3b82f6', borderRadius: '4px 4px 0 0', width: '100%', minHeight: 4 }} />
                      <div className="small text-muted mt-1">{d.bucket}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Drug responsiveness */}
        <div className="col-12 col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Drug Responsiveness</div>
            <div className="card-body">
              {(ov.drug_responsiveness_distribution || []).map(d => (
                <HBar key={d.status} label={d.status} count={d.count} total={total} color={drugColor(d.status)} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   ETIOLOGY & SYNDROME TAB
═══════════════════════════════════════════════════════════════════════════ */
function EtiologyTab({ ov }) {
  const total = (ov.kpis || {}).total_patients || 0;
  const ETI_COLORS = ['#3b82f6','#8b5cf6','#ef4444','#f59e0b','#22c55e','#06b6d4','#ec4899','#6366f1','#f97316','#84cc16'];
  const SYN_COLORS = ['#3b82f6','#8b5cf6','#ef4444','#f59e0b','#22c55e','#06b6d4','#ec4899','#6366f1','#14b8a6','#f97316'];

  return (
    <div className="row g-3">
      <div className="col-12 col-md-6">
        <div className="card shadow-sm h-100">
          <div className="card-header fw-bold">Etiology Distribution</div>
          <div className="card-body">
            {(ov.etiology_distribution || []).map((d, i) => (
              <HBar key={d.etiology} label={d.etiology} count={d.count} total={total} color={ETI_COLORS[i % ETI_COLORS.length]} />
            ))}
          </div>
        </div>
      </div>
      <div className="col-12 col-md-6">
        <div className="card shadow-sm h-100">
          <div className="card-header fw-bold">Epilepsy Syndrome Distribution</div>
          <div className="card-body">
            {(ov.syndrome_distribution || []).map((d, i) => (
              <HBar key={d.syndrome} label={d.syndrome} count={d.count} total={total} color={SYN_COLORS[i % SYN_COLORS.length]} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   EEG & MRI TAB
═══════════════════════════════════════════════════════════════════════════ */
function EegMriTab({ ov }) {
  const EEG_COLORS = ['#3b82f6','#8b5cf6','#ef4444','#f59e0b','#22c55e','#06b6d4','#ec4899','#6366f1','#f97316','#14b8a6','#a3e635','#84cc16','#fbbf24'];
  const MRI_COLORS = ['#3b82f6','#6366f1','#22c55e','#f59e0b','#ef4444','#ec4899','#8b5cf6','#06b6d4','#f97316','#14b8a6','#a3e635','#84cc16'];
  const maxEEG = Math.max(...(ov.eeg_distribution || []).map(d => d.count));
  const maxMRI = Math.max(...(ov.mri_distribution || []).map(d => d.count));

  return (
    <div className="row g-3">
      <div className="col-12 col-md-6">
        <div className="card shadow-sm h-100">
          <div className="card-header fw-bold">EEG Interictal Patterns</div>
          <div className="card-body">
            {(ov.eeg_distribution || []).map((d, i) => (
              <HBar key={d.pattern} label={d.pattern} count={d.count} total={maxEEG} color={EEG_COLORS[i % EEG_COLORS.length]} />
            ))}
          </div>
        </div>
      </div>
      <div className="col-12 col-md-6">
        <div className="card shadow-sm h-100">
          <div className="card-header fw-bold">MRI Structural Findings</div>
          <div className="card-body">
            {(ov.mri_distribution || []).map((d, i) => (
              <HBar key={d.finding} label={d.finding} count={d.count} total={maxMRI} color={MRI_COLORS[i % MRI_COLORS.length]} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   FREQUENCY & DRUG RESPONSE TAB
═══════════════════════════════════════════════════════════════════════════ */
function FrequencyTab({ ov }) {
  const total = (ov.kpis || {}).total_patients || 0;
  return (
    <div className="row g-3">
      <div className="col-12 col-md-6">
        <div className="card shadow-sm h-100">
          <div className="card-header fw-bold">Seizure Frequency</div>
          <div className="card-body">
            {(ov.frequency_distribution || []).map(d => (
              <div key={d.frequency} className="mb-3">
                <div className="d-flex justify-content-between mb-1">
                  <Chip label={d.frequency} color={freqColor(d.frequency)} />
                  <span className="fw-bold">{d.count}</span>
                </div>
                <div style={{ background: '#e5e7eb', borderRadius: 4, height: 10 }}>
                  <div style={{ width: `${Math.round((d.count / total) * 100)}%`, background: freqColor(d.frequency), borderRadius: 4, height: 10 }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="col-12 col-md-6">
        <div className="card shadow-sm h-100">
          <div className="card-header fw-bold">Drug Responsiveness Detail</div>
          <div className="card-body">
            {(ov.drug_responsiveness_distribution || []).map(d => (
              <div key={d.status} className="mb-3">
                <div className="d-flex justify-content-between mb-1">
                  <span className="small">{d.status}</span>
                  <span className="fw-bold">{d.count} <span className="text-muted small">({d.pct}%)</span></span>
                </div>
                <div style={{ background: '#e5e7eb', borderRadius: 4, height: 12 }}>
                  <div style={{ width: `${d.pct}%`, background: drugColor(d.status), borderRadius: 4, height: 12 }} />
                </div>
              </div>
            ))}
            <hr />
            <div className="small text-muted">
              <div><span className="fw-bold" style={{ color: '#ef4444' }}>Drug-resistant:</span> Failed ≥2 AEDs at adequate doses — warrants surgical evaluation</div>
              <div className="mt-1"><span className="fw-bold" style={{ color: '#f59e0b' }}>Partial response:</span> Reduced seizure frequency but not seizure-free</div>
              <div className="mt-1"><span className="fw-bold" style={{ color: '#22c55e' }}>Drug-responsive:</span> Seizure-free or near-free on current regimen</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   PER PATIENT TAB
═══════════════════════════════════════════════════════════════════════════ */
function PerPatientTab({ bd }) {
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState('patient_id');
  const [sortAsc, setSortAsc] = useState(true);

  const patients = (bd?.per_patient || []).filter(p =>
    !search || JSON.stringify(p).toLowerCase().includes(search.toLowerCase())
  );

  const sorted = [...patients].sort((a, b) => {
    const av = a[sortKey] ?? '';
    const bv = b[sortKey] ?? '';
    return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
  });

  const toggleSort = key => {
    if (sortKey === key) setSortAsc(a => !a);
    else { setSortKey(key); setSortAsc(true); }
  };

  const Th = ({ k, label }) => (
    <th style={{ cursor: 'pointer', whiteSpace: 'nowrap' }} onClick={() => toggleSort(k)}>
      {label} {sortKey === k ? (sortAsc ? '▲' : '▼') : ''}
    </th>
  );

  const latColor = l =>
    l === 'Left' ? '#3b82f6' : l === 'Right' ? '#ef4444' :
    l === 'Bilateral' ? '#8b5cf6' : '#6b7280';

  return (
    <div>
      <div className="mb-3">
        <input className="form-control form-control-sm" style={{ maxWidth: 300 }}
          placeholder="Search patients…" value={search} onChange={e => setSearch(e.target.value)} />
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="table table-hover table-sm small align-middle">
          <thead className="table-dark">
            <tr>
              <Th k="patient_id" label="Patient" />
              <Th k="onset_zone" label="Onset Zone" />
              <Th k="syndrome" label="Syndrome" />
              <Th k="etiology" label="Etiology" />
              <Th k="drug_responsiveness" label="Drug Response" />
              <Th k="seizure_frequency" label="Frequency" />
              <Th k="age_at_onset" label="Onset Age" />
              <Th k="lateralization" label="Lateralization" />
              <th>Seizure Types</th>
              <th>Surgery</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(p => (
              <tr key={p.patient_id}>
                <td><span className="badge bg-primary">{p.patient_id}</span></td>
                <td><Chip label={p.onset_zone} color={zoneColor(p.onset_zone)} /></td>
                <td style={{ maxWidth: 160, whiteSpace: 'normal', fontSize: '0.72rem' }}>{p.syndrome}</td>
                <td style={{ maxWidth: 140, whiteSpace: 'normal', fontSize: '0.72rem' }}>{p.etiology}</td>
                <td>
                  <Chip label={p.drug_responsiveness?.split(' ')[0] || p.drug_responsiveness}
                    color={drugColor(p.drug_responsiveness || '')} />
                </td>
                <td><Chip label={p.seizure_frequency} color={freqColor(p.seizure_frequency)} /></td>
                <td className="text-center">{p.age_at_onset}y</td>
                <td><Chip label={p.lateralization || 'N/A'} color={latColor(p.lateralization)} /></td>
                <td style={{ maxWidth: 180, whiteSpace: 'normal', fontSize: '0.72rem' }}>{p.seizure_types}</td>
                <td style={{ fontSize: '0.72rem', color: p.surgery_candidacy?.includes('candidate') ? '#22c55e' : p.surgery_candidacy?.includes('Further') ? '#f59e0b' : '#6b7280' }}>
                  {p.surgery_candidacy}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-muted small">{sorted.length} of {patients.length} patients</div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   DEFINITIONS TAB
═══════════════════════════════════════════════════════════════════════════ */
function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading definitions…</div>;

  const Section = ({ title, data }) => (
    <div className="mb-4">
      <h6 className="fw-bold border-bottom pb-1">{title}</h6>
      <div className="row g-2">
        {Object.entries(data || {}).map(([k, v]) => (
          <div key={k} className="col-12 col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-body py-2">
                <div className="fw-bold small mb-1">{k}</div>
                <div className="text-muted" style={{ fontSize: '0.78rem' }}>{v}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div>
      <Section title="Seizure Type Definitions" data={defs.seizure_type_definitions} />
      <Section title="Onset Zone Descriptions" data={defs.onset_zone_descriptions} />
      <Section title="Etiology Categories" data={defs.etiology_categories} />
      <Section title="Drug Responsiveness Criteria" data={defs.drug_responsiveness_criteria} />
      <Section title="Surgery Candidacy Criteria" data={defs.surgery_candidacy_criteria} />
      {defs.ilae_classification_note && (
        <div className="alert alert-info small">
          <strong>ILAE Note:</strong> {defs.ilae_classification_note}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   ROOT PAGE
═══════════════════════════════════════════════════════════════════════════ */
export default function SeizureMetadataPage() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/seizure-metadata/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/seizure-metadata/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/seizure-metadata/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const TABS = [
    { id: 'overview',   label: 'Overview' },
    { id: 'etiology',   label: 'Etiology & Syndrome' },
    { id: 'eeg_mri',    label: 'EEG & MRI' },
    { id: 'frequency',  label: 'Drug Response & Frequency' },
    { id: 'patients',   label: 'Per Patient' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f9e0; Seizure Metadata (ILAE Classification)</h3>
      <p className="text-muted small">
        ILAE-structured seizure classification for {(ov.kpis || {}).total_patients} patients —
        onset zones, seizure types, etiology, syndrome, EEG/MRI findings, drug response &amp; surgery candidacy.
      </p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview'   && <OverviewTab ov={ov} />}
      {tab === 'etiology'   && <EtiologyTab ov={ov} />}
      {tab === 'eeg_mri'    && <EegMriTab ov={ov} />}
      {tab === 'frequency'  && <FrequencyTab ov={ov} />}
      {tab === 'patients'   && <PerPatientTab bd={bd} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  );
}
