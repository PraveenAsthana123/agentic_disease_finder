'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Breakdown', 'Etiology Detail', 'Treatments', 'Definitions'];
const COLOR = '#4a148c';   // deep purple — distinct from GLRA1 teal, GLRB dark-teal, SLC6A5 green, GPHN brown
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
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const kpis = data.kpis || {};
  const etiologies = data.etiology_distribution || [];
  const treatments = data.treatments_summary || [];
  const monitoring = data.monitoring_summary || [];
  const lifecycle = data.lifecycle || [];
  const thresholds = data.thresholds || [];
  const cis = data.contraindications_summary || [];
  const maxEtio = Math.max(...etiologies.map(e => e.pct || 0), 1);

  return (
    <div>
      <div className="alert alert-warning py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}` }}>
        <strong>ARHGEF9 (Xq11.1) — Collybistin · OMIM 300429/300855 · 5th of 5-gene Hyperekplexia Panel:</strong>{' '}
        Collybistin (CB) is a CDC42-GEF that <strong>activates CDC42 → membrane-targets GPHN</strong>.
        ARHGEF9 LOF → GPHN trapped cytoplasmic → iPSD never assembles → both GlyR AND GABA<sub>A</sub>R
        clustering fail (same end-result as GPHN LOF but GPHN protein is NORMAL).{' '}
        <strong>RAREST of 5-gene panel (&lt;1%); X-linked</strong> — hemizygous males severe DEE + hyperekplexia;
        het females variable by XCI skewing.{' '}
        <span className="text-danger fw-bold">
          5-gene panel + Xq11.1 MLPA mandatory. EEG mandatory. XCI assay mandatory for het females.
          POLG before VPA. Vigevano manoeuvre training universal.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Apnoeic Events" value={`${kpis.apnoeic_events_pct}%`} color="#dc3545" />
        <KPI label="Rigid-Baby" value={`${kpis.rigid_baby_pct}%`} color="#dc3545" />
        <KPI label="Epileptic Sz" value={`${kpis.epileptic_seizures_pct}%`} color="#6610f2" />
        <KPI label="Intellect Disab" value={`${kpis.intellectual_disability_pct}%`} color="#6f42c1" />
        <KPI label="On CLZ" value={`${kpis.on_clonazepam_pct}%`} color={COLOR} />
        <KPI label="ASM for Sz" value={`${kpis.asm_for_seizures_pct}%`} color="#fd7e14" />
        <KPI label="Manoeuvre Trained" value={`${kpis.forward_flexion_trained_pct}%`} color="#198754" />
        <KPI label="Nose-Tap +" value={`${kpis.nose_tap_positive_pct}%`} color={COLOR} />
        <KPI label="XCI Tested" value={`${kpis.xci_tested_pct}%`} color="#0d6efd" />
        <KPI label="EEG Abnormal" value={`${kpis.eeg_abnormal_pct}%`} color="#dc3545" />
        <KPI label="Xq11 MLPA" value={`${kpis.xq11_mlpa_pct}%`} color="#0d6efd" />
        <KPI label="ASD Diagnosis" value={`${kpis.asd_pct}%`} color="#6f42c1" />
        <KPI label="POLG Tested" value={`${kpis.polg_tested_pct}%`} color="#198754" />
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
              Treatment Lines
            </div>
            <div className="card-body">
              {treatments.map((t, i) => (
                <div key={i} className="mb-2 pb-1 border-bottom small">
                  <span className="fw-bold">{t.drug}</span>
                  <div className="text-muted">{t.level.substring(0, 80)}{t.level.length > 80 ? '…' : ''}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Clinical Thresholds
            </div>
            <div className="card-body">
              {thresholds.map((t, i) => (
                <div key={i} className="mb-2 small border-bottom pb-1">
                  <span className="fw-semibold">{t.parameter}:</span>{' '}
                  <span className="text-primary">{t.threshold}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small text-danger" style={{ background: '#fff5f5' }}>
              Contraindications (NEVER DO without screening)
            </div>
            <div className="card-body">
              {cis.map((c, i) => (
                <div key={i} className="mb-1 small text-danger">⛔ {c}</div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Monitoring Schedule
        </div>
        <div className="card-body">
          <div className="table-responsive">
            <table className="table table-sm table-hover small mb-0">
              <thead><tr><th>Timepoint</th><th>Action</th></tr></thead>
              <tbody>
                {monitoring.map((m, i) => (
                  <tr key={i}><td className="fw-semibold">{m.timepoint}</td><td>{m.action}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Disease Lifecycle Windows
        </div>
        <div className="card-body d-flex flex-wrap gap-2">
          {lifecycle.map((l, i) => (
            <div key={i} className="card border p-2 small" style={{ minWidth: 200, borderLeft: `3px solid ${COLOR}` }}>
              <div className="fw-bold">{l.stage}</div>
              <div className="text-muted small">{l.events.substring(0, 80)}{l.events.length > 80 ? '…' : ''}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;
  const byCategory = data.by_category || [];
  const summary = data.summary || {};

  return (
    <div>
      <div className="row g-3 mb-3">
        {[
          ['Apnoeic Events', `${summary.apnoeic_pct}%`, '#dc3545'],
          ['Rigid Baby', `${summary.rigid_baby_pct}%`, '#dc3545'],
          ['Epileptic Sz', `${summary.epileptic_seizures_pct}%`, '#6610f2'],
          ['Intellect Disab', `${summary.intellectual_disability_pct}%`, '#6f42c1'],
          ['EEG Abnormal', `${summary.eeg_abnormal_pct}%`, '#dc3545'],
          ['ASD Diagnosis', `${summary.asd_pct}%`, '#6f42c1'],
          ['XCI Tested', `${summary.xci_tested_pct}%`, '#0d6efd'],
          ['Xq11 MLPA', `${summary.xq11_mlpa_pct}%`, '#0d6efd'],
          ['POLG Tested', `${summary.polg_tested_pct}%`, '#198754'],
        ].map(([label, value, color], i) => (
          <KPI key={i} label={label} value={value} color={color} />
        ))}
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Per-Etiology Breakdown (40 patients total)
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead>
              <tr>
                <th>Category</th><th>N</th><th>Apnoea%</th><th>Rigid%</th>
                <th>Epileptic%</th><th>ID%</th><th>CLZ%</th>
                <th>EEG Abn%</th><th>ASD%</th><th>XCI%</th>
              </tr>
            </thead>
            <tbody>
              {byCategory.map((row, i) => (
                <tr key={i}>
                  <td>
                    <span className="badge" style={{ background: COLOR, fontSize: '0.65rem' }}>
                      {row.category.substring(0, 28)}
                    </span>
                  </td>
                  <td className="fw-semibold">{row.n}</td>
                  <td>{row.apnoeic_pct}%</td>
                  <td>{row.rigid_baby_pct}%</td>
                  <td>{row.epileptic_pct}%</td>
                  <td>{row.id_pct}%</td>
                  <td>{row.clz_pct}%</td>
                  <td>{row.eeg_abnormal_pct}%</td>
                  <td>{row.asd_pct}%</td>
                  <td>{row.xci_tested_pct}%</td>
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
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const details = data.etiology_details || [];

  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Etiology Detail — 4 Categories</h6>
      {details.map((e, i) => (
        <div key={i} className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${COLOR}` }}>
          <div className="card-header py-2" style={{ background: LIGHT }}>
            <span className="fw-bold">{e.category}</span>
            <span className="ms-2 badge bg-secondary">{e.inheritance}</span>
          </div>
          <div className="card-body py-2 small">
            <div className="mb-1"><span className="fw-semibold">Typical variant:</span>{' '}
              <code>{e.typical_variant}</code></div>
            <div className="mb-1"><span className="fw-semibold">Functional deficit:</span>{' '}
              {e.functional_deficit}</div>
            <div className="text-muted">{e.description}</div>
          </div>
        </div>
      ))}

      <div className="alert alert-info small mt-3">
        <strong>X-linked Panel Note:</strong> For male patients, ARHGEF9 hemizygous variants are
        confirmed by hemizygous state (males carry one X). For female patients, heterozygous
        ARHGEF9 variants require XCI analysis (AR methylation / HUMARA assay) to determine
        if unfavourable skewing (&gt;70% mutant X) renders them clinically affected.
        Xq11.1 MLPA must accompany sequence analysis to detect exonic deletions not seen
        by NGS panel alone.
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const treatments = data.treatments || [];
  const cis = data.contraindications || [];

  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Treatment Lines</h6>
      {treatments.map((t, i) => (
        <div key={i} className={`card shadow-sm mb-3 ${t.contraindication_flag ? 'border-danger' : ''}`}>
          <div className="card-header py-2" style={{ background: t.contraindication_flag ? '#fff5f5' : LIGHT }}>
            <span className="fw-bold">{t.drug}</span>
            <span className="ms-2 badge" style={{ background: COLOR }}>{t.level.split(' —')[0]}</span>
          </div>
          <div className="card-body py-2 small">
            <div className="mb-1"><span className="fw-semibold">Mechanism:</span> {t.mechanism}</div>
            <div className="mb-1"><span className="fw-semibold">Dose:</span> {t.dose}</div>
            <div><span className="fw-semibold">Note:</span>{' '}
              <span className={t.contraindication_flag ? 'text-danger fw-bold' : 'text-primary'}>
                {t.note}
              </span>
            </div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3 text-danger">Contraindications</h6>
      {cis.map((c, i) => (
        <div key={i} className="alert alert-danger py-2 small mb-2">
          <span className="fw-bold text-danger">⛔ {c.drug}</span>
          <span className="ms-2 badge bg-danger">{c.level}</span>
          <div className="mt-1 text-dark">{c.reason}</div>
          <div className="text-danger small">Risk: {c.risk}</div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const definitions = data.definitions || [];
  const ddx = data.key_ddx || [];
  const mandatory = data.mandatory_workup || [];
  const freqs = data.five_gene_panel_frequencies || {};
  const standards = data.standards || [];

  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Core Concepts ({definitions.length})</h6>
      {definitions.map((d, i) => (
        <div key={i} className="card shadow-sm mb-2">
          <div className="card-header py-1 small fw-bold" style={{ background: LIGHT }}>{d.term}</div>
          <div className="card-body py-2 small text-muted">{d.definition}</div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: COLOR }}>5-Gene Panel Frequencies</h6>
      <div className="d-flex flex-wrap gap-2 mb-4">
        {Object.entries(freqs).map(([gene, freq], i) => (
          <span key={i} className="badge fs-6" style={{ background: gene === 'ARHGEF9' ? COLOR : '#6c757d' }}>
            {gene}: {freq}
          </span>
        ))}
      </div>

      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Mandatory Workup</h6>
      <ul className="small mb-4">
        {mandatory.map((m, i) => <li key={i} className="mb-1">{m}</li>)}
      </ul>

      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Key Differential Diagnoses</h6>
      <ul className="small mb-4">
        {ddx.map((d, i) => <li key={i} className="mb-1 text-muted">{d}</li>)}
      </ul>

      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Clinical Standards</h6>
      <ul className="small mb-4">
        {standards.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
      </ul>
    </div>
  );
}

export default function ARHGEF9Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/arhgef9/overview`).then(r => r.json()),
      fetch(`${API}/api/arhgef9/breakdown`).then(r => r.json()),
      fetch(`${API}/api/arhgef9/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            🧬 ARHGEF9 Hyperekplexia / DEE (Collybistin — CDC42-GEF — Xq11.1)
          </h4>
          <div className="text-muted small">
            Collybistin / Xq11.1 · OMIM Gene 300429 · Disease 300855 · 480aa ·
            DH-domain (CDC42-GEF) + PH-domain (membrane-targeting) + SH3-domain (autoinhibition) ·
            X-linked · Hemizygous males severe DEE+Hyperekplexia · Het females XCI-governed ·
            RAREST of 5-gene panel (&lt;1%) · Upstream of GPHN membrane targeting ·
            40-patient cohort seed-503
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger py-2 small">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <EtiologyTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
