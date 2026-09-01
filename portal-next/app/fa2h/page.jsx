'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Leukodystrophy & Iron', 'Treatments', 'Definitions'];
const COLOR = '#33691e';   // olive green — FA2H/FAHN/NBIA3 (lipid/myelin green)
const LIGHT = '#f1f8e9';

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
  const highlights = data.clinical_highlights || [];
  const cis = data.contraindications || [];
  const thresholds = data.thresholds || [];
  const maxEtio = Math.max(...etiologies.map(e => e.pct || 0), 1);

  return (
    <div>
      <div className="alert py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
        <strong>FA2H (16q23.1) — 490aa · ER-membrane · FAD-binding Rossmann fold · HX3H di-iron motif · OMIM 611026/612319 · FAHN/NBIA3:</strong>{' '}
        4th most common NBIA (~5-10%). AR biallelic FA2H → 2-hydroxy sphingolipid (HFA-GalC) deficiency → myelin instability → leukodystrophy.{' '}
        <strong className="text-danger">Leukodystrophy EARLIEST + MOST PROMINENT MRI feature (precedes GP/SN iron). Spastic paraplegia DOMINANT early motor.</strong>{' '}
        NO eye-of-tiger (key DDx PKAN/NBIA1). GP+SN iron present but MILD early (secondary to myelin/axon degeneration).{' '}
        <span className="fw-bold" style={{ color: COLOR }}>
          3 phenotypes: FAHN-Classic 50% · HSP-Ataxia-Dystonia 35% · Complex-SPG 15%.
          PHT AVOID (leukodystrophy CNS depression). VGB AVOID (visual field + optic risk).
          Baclofen first-line Level C. POLG mandatory before VPA. GPi-DBS Level D (3 case reports). Deferiprone investigational.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="FAHN-Classic" value={kpis.n_fahn_classic} color="#dc3545" />
        <KPI label="HSP-Ataxia-Dystonia" value={kpis.n_hsp_ataxia_dystonia} color="#e65100" />
        <KPI label="Complex-SPG" value={kpis.n_complex_spg} color="#6f42c1" />
        <KPI label="Leukodystrophy" value={`${kpis.leukodystrophy_pct}%`} color="#dc3545" />
        <KPI label="GP Iron" value={`${kpis.gp_iron_pct}%`} color="#e65100" />
        <KPI label="Thin CC" value={`${kpis.thin_cc_pct}%`} color="#e65100" />
        <KPI label="Cerebellar Atrophy" value={`${kpis.cerebellar_atrophy_pct}%`} color="#e65100" />
        <KPI label="Spastic Paraplegia" value={`${kpis.spastic_paraplegia_pct}%`} color="#dc3545" />
        <KPI label="Ataxia" value={`${kpis.ataxia_pct}%`} color={COLOR} />
        <KPI label="Dystonia" value={`${kpis.dystonia_pct}%`} color={COLOR} />
        <KPI label="Has Seizures" value={`${kpis.has_seizures_pct}%`} color="#e65100" />
        <KPI label="Lost Ambulation" value={`${kpis.ambulation_lost_pct}%`} color="#dc3545" />
        <KPI label="Optic Atrophy" value={`${kpis.optic_atrophy_pct}%`} color={COLOR} />
        <KPI label="Baclofen" value={`${kpis.baclofen_pct ?? '—'}`} color="#0d6efd" />
        <KPI label="POLG Tested" value={`${kpis.polg_tested_pct ?? '—'}%`} color="#0d6efd" />
      </div>

      {/* Etiology Distribution */}
      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
              Etiology Distribution (Seed-523)
            </div>
            <div className="card-body">
              {etiologies.map((e, i) => (
                <Bar key={i} label={`${e.etiology} (n=${e.n})`} value={e.pct} max={maxEtio} />
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
              Phenotype Onset (mean yr)
            </div>
            <div className="card-body">
              <Bar label={`FAHN-Classic (n=${kpis.n_fahn_classic})`} value={kpis.fahn_classic_mean_onset_yr} max={20} color="#dc3545" />
              <Bar label={`HSP-Ataxia-Dystonia (n=${kpis.n_hsp_ataxia_dystonia})`} value={kpis.hsp_mean_onset_yr} max={20} color="#e65100" />
              <Bar label={`Complex-SPG (n=${kpis.n_complex_spg})`} value={kpis.complex_spg_mean_onset_yr} max={20} color="#6f42c1" />
              <div className="small text-muted mt-2">FAHN-Classic earliest onset (~3-5yr). Complex-SPG adolescent (~10-15yr).</div>
            </div>
          </div>
        </div>
      </div>

      {/* Clinical Highlights */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          Clinical Feature Prevalence (40 patients, seed-523)
        </div>
        <div className="card-body">
          <div className="row">
            {highlights.map((h, i) => (
              <div key={i} className="col-md-6 mb-2">
                <div className="d-flex justify-content-between small">
                  <span className="fw-semibold">{h.finding}</span>
                  <span className="badge" style={{ background: COLOR, color: '#fff' }}>{h.pct}%</span>
                </div>
                <div className="text-muted" style={{ fontSize: '0.75rem' }}>{h.note}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Contraindications */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold bg-danger text-white">
          Drug Contraindications / Avoidance (FA2H/FAHN specific)
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-light">
              <tr><th>Drug</th><th>Level</th><th>Reason</th><th>Alternative</th></tr>
            </thead>
            <tbody>
              {cis.map((ci, i) => (
                <tr key={i}>
                  <td className="fw-bold text-danger">{ci.drug}</td>
                  <td><span className="badge bg-danger">{ci.level}</span></td>
                  <td className="small">{ci.reason}</td>
                  <td className="small text-success">{ci.alternative}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Thresholds */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: '#37474f', color: '#fff' }}>
          Clinical Decision Thresholds
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-light">
              <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold small">{t.parameter}</td>
                  <td className="small text-warning fw-bold">{t.threshold}</td>
                  <td className="small">{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const phenotypes = data.phenotype_breakdown || [];
  const etiologies = data.etiology_breakdown || [];
  const patients = data.per_patient || [];
  const ddx = data.ddx_table || [];
  const monitoring = data.monitoring || [];

  return (
    <div>
      {/* Phenotype Breakdown */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          Phenotype Breakdown
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-light">
              <tr>
                <th>Phenotype</th><th>N</th><th>%</th><th>Mean Onset</th>
                <th>Leuko%</th><th>GP Iron%</th><th>Cerebel%</th>
                <th>Ataxia%</th><th>Dystonia%</th><th>Seizures%</th><th>Amb Lost%</th><th>DR%</th>
              </tr>
            </thead>
            <tbody>
              {phenotypes.map((ph, i) => (
                <tr key={i}>
                  <td className="fw-bold">{ph.phenotype}</td>
                  <td>{ph.n}</td><td>{ph.pct}%</td>
                  <td>{ph.mean_onset_yr}yr</td>
                  <td>{ph.leukodystrophy_pct}%</td>
                  <td>{ph.gp_iron_pct}%</td>
                  <td>{ph.cerebellar_atrophy_pct}%</td>
                  <td>{ph.ataxia_pct}%</td>
                  <td>{ph.dystonia_pct}%</td>
                  <td>{ph.has_seizures_pct}%</td>
                  <td>{ph.ambulation_lost_pct}%</td>
                  <td className="text-danger">{ph.drug_resistant_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* DDx Table */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: '#37474f', color: '#fff' }}>
          Differential Diagnosis (FA2H/FAHN vs other NBIA + Leukodystrophies)
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-light">
              <tr><th>Condition</th><th>Key Differentiator</th><th>MRI Clue</th><th>Clinical Clue</th></tr>
            </thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i}>
                  <td className="fw-bold">{d.condition}</td>
                  <td className="small">{d.key_differentiator}</td>
                  <td className="small">{d.mri_clue}</td>
                  <td className="small">{d.clinical_clue}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Monitoring Schedule */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          Monitoring Schedule
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-light">
              <tr><th>Item</th><th>Frequency</th><th>Notes</th></tr>
            </thead>
            <tbody>
              {monitoring.map((m, i) => (
                <tr key={i}>
                  <td className="fw-semibold small">{m.item}</td>
                  <td className="small">{m.freq}</td>
                  <td className="small text-muted">{m.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Per-Patient Table (scrollable) */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          Per-Patient Summary (40 patients, seed-523)
        </div>
        <div className="card-body p-0" style={{ overflowX: 'auto' }}>
          <table className="table table-sm table-striped table-bordered mb-0" style={{ minWidth: 1400 }}>
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Phenotype</th><th>Etiology</th>
                <th>Onset</th><th>Age</th><th>Dur</th>
                <th>Leuko</th><th>GP Fe</th><th>SN Fe</th><th>CC</th><th>Cereb</th>
                <th>Spastic</th><th>Ataxia</th><th>Dysarth</th><th>Dystonia</th><th>Optic</th>
                <th>Neuropathy</th><th>Ambul-Lost</th><th>Cogn</th>
                <th>Sz</th><th>DR</th><th>Bac</th><th>BTX</th><th>Trihex</th><th>DBS</th><th>POLG</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{p.id}</td>
                  <td className="small">{p.phenotype}</td>
                  <td className="small">{p.etiology?.replace(/_/g, ' ')}</td>
                  <td className="small">{p.onset_yr}yr</td>
                  <td className="small">{p.current_age}</td>
                  <td className="small">{p.disease_duration_yr}yr</td>
                  <td className="small">{p.leukodystrophy ? '✅' : '—'}</td>
                  <td className="small">{p.gp_iron ? '🔴' : '—'}</td>
                  <td className="small">{p.sn_iron ? '🔴' : '—'}</td>
                  <td className="small">{p.thin_cc ? '⚠️' : '—'}</td>
                  <td className="small">{p.cerebellar_atrophy ? '⚠️' : '—'}</td>
                  <td className="small">{p.spastic_paraplegia ? '✅' : '—'}</td>
                  <td className="small">{p.ataxia ? '✅' : '—'}</td>
                  <td className="small">{p.dysarthria ? '✅' : '—'}</td>
                  <td className="small">{p.dystonia ? `✅ ${p.dystonia_severity || ''}` : '—'}</td>
                  <td className="small">{p.optic_atrophy ? '⚠️' : '—'}</td>
                  <td className="small">{p.axonal_neuropathy ? '⚠️' : '—'}</td>
                  <td className="small">{p.ambulation_lost ? '🔴' : '—'}</td>
                  <td className="small">{p.cognitive_decline ? '⚠️' : '—'}</td>
                  <td className="small">{p.has_seizures ? '⚡' : '—'}</td>
                  <td className="small">{p.drug_resistant ? '🔴 DR' : '—'}</td>
                  <td className="small">{p.baclofen ? '✅' : '—'}</td>
                  <td className="small">{p.btx ? '✅' : '—'}</td>
                  <td className="small">{p.trihexyphenidyl ? '✅' : '—'}</td>
                  <td className="small">{p.dbs ? '🧠' : '—'}</td>
                  <td className="small">{p.polg_tested ? '✅' : '⚠️'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function LeukodystrophyTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const phenotypes = data.phenotype_breakdown || [];
  const etiologies = data.etiology_breakdown || [];
  const seizures = data.seizure_breakdown || [];

  return (
    <div>
      {/* MRI Key Info Banner */}
      <div className="alert alert-warning mb-4">
        <strong>🧠 MRI Sequence Priority in FA2H/FAHN:</strong>
        <ol className="mb-0 mt-2 small">
          <li><strong>T2/FLAIR:</strong> Leukodystrophy (bilateral WM hyperintensity) — EARLIEST finding — periventricular + deep WM</li>
          <li><strong>SWI/T2*:</strong> GP + SN iron accumulation — present but MILD early; worsens with age</li>
          <li><strong>T1:</strong> Thin corpus callosum (80%); NO T1 halo sign (DDx BPAN)</li>
          <li><strong>FLAIR:</strong> Cerebellar volume loss — appears after spastic onset, progressive</li>
          <li><strong>DWI:</strong> Active demyelination in acute phases</li>
        </ol>
        <div className="mt-2 small text-danger fw-bold">
          Key DDx: Leukodystrophy is ABSENT in PKAN (only eye-of-tiger GP), BPAN (only iron+T1 halo), MPAN (only iron, no WM). Leukodystrophy + GP iron = FAHN or SPG11 (but SPG11 has NO iron on SWI).
        </div>
      </div>

      {/* Etiology Breakdown Table */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          Etiology Breakdown
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-light">
              <tr><th>Etiology</th><th>N</th><th>%</th><th>FAHN-Classic%</th><th>HSP-AD%</th><th>Leuko%</th><th>DR%</th></tr>
            </thead>
            <tbody>
              {etiologies.map((e, i) => (
                <tr key={i}>
                  <td className="fw-bold">{e.etiology}</td>
                  <td>{e.n}</td><td>{e.pct}%</td>
                  <td>{e.fahn_classic_pct}%</td>
                  <td>{e.hsp_ataxia_dystonia_pct}%</td>
                  <td>{e.leukodystrophy_pct}%</td>
                  <td className="text-danger">{e.drug_resistant_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Seizure Breakdown */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: '#e65100', color: '#fff' }}>
          Seizure Breakdown (among patients with seizures)
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-light">
              <tr><th>Seizure Type</th><th>N</th><th>%</th><th>DR%</th></tr>
            </thead>
            <tbody>
              {seizures.map((s, i) => (
                <tr key={i}>
                  <td className="fw-bold">{s.type}</td>
                  <td>{s.n}</td><td>{s.pct}%</td>
                  <td className="text-danger">{s.drug_resistant_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card-footer small text-muted">
          PHT AVOID (leukodystrophy CNS depression). VGB AVOID (visual field + optic risk). Preferred: LEV, CLB, LCM.
        </div>
      </div>

      {/* 2-OH Sphingolipid Pathway Summary */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          FA2H 2-Hydroxy Sphingolipid Pathway — Leukodystrophy Mechanism
        </div>
        <div className="card-body small">
          <div className="row">
            <div className="col-md-6">
              <h6 className="fw-bold">Normal FA2H Function</h6>
              <ul>
                <li>FA2H + FAD + O₂ → 2-hydroxylation of fatty acyl-CoA</li>
                <li>2-OH fatty acid + ceramide → 2-hydroxy ceramide</li>
                <li>2-OH ceramide + galactosyltransferase → 2-OH galactocerebroside (HFA-GalC)</li>
                <li>HFA-GalC + sulphotransferase → 2-OH sulfatide</li>
                <li>HFA-GalC + sulfatide → tighter myelin bilayer → myelin stability</li>
                <li>2-OH lipid rafts → paranodal Caspr/contactin complex → node of Ranvier integrity</li>
              </ul>
            </div>
            <div className="col-md-6">
              <h6 className="fw-bold text-danger">FA2H LOF Consequences</h6>
              <ul>
                <li>HFA-GalC deficiency → loose myelin bilayer → <strong>demyelination = leukodystrophy</strong></li>
                <li>Node of Ranvier disruption → axon degeneration (independent of demyelination)</li>
                <li>Lipid raft disruption → metal transport failure → GP/SN iron accumulation (secondary)</li>
                <li>Mitochondrial membrane dysfunction (secondary) → POLG vulnerability</li>
                <li>Iron accumulation follows myelin loss: leukodystrophy appears FIRST on MRI</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const treatments = data.treatment_summary || [];
  const cis = data.contraindications || [];
  const lifecycle = data.lifecycle || {};

  return (
    <div>
      {/* Treatment Table */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          Treatment Protocols — FA2H/FAHN (2026)
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-light">
              <tr><th>Drug / Intervention</th><th>Indication</th><th>Level</th><th>Dose</th><th>Notes</th></tr>
            </thead>
            <tbody>
              {treatments.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.drug}</td>
                  <td className="small">{t.indication}</td>
                  <td>
                    <span className={`badge ${t.level.includes('Investigational') ? 'bg-warning text-dark' : t.level.includes('B') ? 'bg-primary' : t.level.includes('C') ? 'bg-success' : t.level.includes('D') ? 'bg-secondary' : 'bg-info text-dark'}`}>
                      {t.level}
                    </span>
                  </td>
                  <td className="small">{t.dose}</td>
                  <td className="small text-muted">{t.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Contraindications */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold bg-danger text-white">
          Drug Contraindications / Avoidance
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-light">
              <tr><th>Drug</th><th>Level</th><th>Reason</th><th>Evidence</th><th>Alternative</th></tr>
            </thead>
            <tbody>
              {cis.map((ci, i) => (
                <tr key={i}>
                  <td className="fw-bold text-danger">{ci.drug}</td>
                  <td><span className="badge bg-danger">{ci.level}</span></td>
                  <td className="small">{ci.reason}</td>
                  <td className="small text-muted">{ci.evidence}</td>
                  <td className="small text-success">{ci.alternative}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Disease Lifecycle */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: '#37474f', color: '#fff' }}>
          Disease Lifecycle / Care Pathway
        </div>
        <div className="card-body">
          {Object.values(lifecycle).map((phase, i) => (
            <div key={i} className="mb-3">
              <h6 className="fw-bold" style={{ color: COLOR }}>{phase.label}</h6>
              <p className="small text-muted mb-0">{phase.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const defs = data.definitions || [];
  const standards = data.standards || [];
  const refs = data.references || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          Clinical Definitions & Concepts ({defs.length})
        </div>
        <div className="card-body">
          {defs.map((d, i) => (
            <div key={i} className="mb-3 border-bottom pb-2">
              <div className="fw-bold small" style={{ color: COLOR }}>{d.term}</div>
              <div className="text-muted small fw-semibold">{d.full}</div>
              <div className="small">{d.detail}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: '#37474f', color: '#fff' }}>
          Standards & Guidelines
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-light">
              <tr><th>Standard</th><th>Relevance</th></tr>
            </thead>
            <tbody>
              {standards.map((s, i) => (
                <tr key={i}>
                  <td className="fw-semibold small">{s.standard}</td>
                  <td className="small text-muted">{s.relevance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: '#37474f', color: '#fff' }}>
          Key References
        </div>
        <div className="card-body">
          {refs.map((r, i) => (
            <div key={i} className="mb-3">
              <div className="small fw-semibold">{r.citation}</div>
              <div className="small text-success">{r.key_finding}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function FA2HPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/fa2h/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(e => setError('Backend unavailable: ' + e.message));
    fetch(`${API}/api/fa2h/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(() => {});
    fetch(`${API}/api/fa2h/definitions`)
      .then(r => r.json()).then(setDefinitions)
      .catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h2 className="fw-bold mb-0" style={{ color: COLOR }}>
          🧬 FA2H FAHN / NBIA3 — Fatty Acid Hydroxylase-Associated Neurodegeneration
        </h2>
        <div className="text-muted small">
          FA2H (490aa · 16q23.1) · ER-membrane FAD-dependent 2-hydroxylase · 4th most common NBIA (~5-10%) ·
          OMIM Gene 611026 / Disease 612319 · AR biallelic · Seed-523 (40 patients)
        </div>
        <div className="small mt-1">
          <span className="badge me-1" style={{ background: '#dc3545' }}>FAHN-Classic 50%</span>
          <span className="badge me-1" style={{ background: '#e65100' }}>HSP-Ataxia-Dystonia 35%</span>
          <span className="badge me-1" style={{ background: '#6f42c1' }}>Complex-SPG 15%</span>
          <span className="badge me-1 bg-danger">Leukodystrophy EARLIEST MRI</span>
          <span className="badge me-1 bg-warning text-dark">NO Eye-of-Tiger (DDx PKAN)</span>
          <span className="badge me-1 bg-dark">PHT AVOID</span>
          <span className="badge me-1 bg-dark">VGB AVOID</span>
          <span className="badge me-1 bg-dark">POLG Mandatory Before VPA</span>
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active fw-bold' : ''}`}
              style={tab === t ? { borderBottomColor: COLOR, color: COLOR } : {}}
              onClick={() => setTab(t)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Patients & Phenotype' && <PatientsTab data={breakdown} />}
      {tab === 'Leukodystrophy & Iron' && <LeukodystrophyTab data={breakdown} />}
      {tab === 'Treatments' && <TreatmentsTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
