'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'CoA Pathway & Iron', 'Treatments', 'Definitions'];
const COLOR = '#4a148c';   // deep purple — COASY/CoPAN/NBIA6 (CoA pathway — purple for rare/ultra-rare)
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
  const highlights = data.clinical_highlights || [];
  const cis = data.contraindications || [];
  const thresholds = data.thresholds || [];
  const maxEtio = Math.max(...etiologies.map(e => e.pct || 0), 1);

  return (
    <div>
      <div className="alert py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
        <strong>COASY (17q21.2) — 579aa bifunctional · PPAT (aa1-200) + DPCK (aa231-579) · OMIM 609686/CoPAN 615643 · NBIA6:</strong>{' '}
        Very rare NBIA (~1-2%; ~25-30 patients worldwide 2026). AR biallelic COASY → CoA biosynthesis block (final 2 steps, downstream PANK2) → TCA/FAO failure → GP/SN iron.{' '}
        <strong className="text-danger">GP iron PROMINENT from early disease (SWI/T2* — uniform, NOT eye-of-tiger). NO leukodystrophy (critical DDx FAHN/NBIA3). NO eye-of-tiger (critical DDx PKAN/NBIA1).</strong>{' '}
        BOTH spasticity + dystonia prominent simultaneously (unlike FAHN: spasticity dominant, dystonia late).{' '}
        <span className="fw-bold" style={{ color: COLOR }}>
          3 phenotypes: Classic-CoPAN 65% · Neuropsychiatric-CoPAN 25% · Late-onset-CoPAN 10%.
          Seizures 60-70%. PHT AVOID (dystonia aggravation). POLG mandatory before VPA.
          Pantothenate + Deferiprone investigational. GPi-DBS Level D.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Classic-CoPAN" value={kpis.n_classic_copan} color="#dc3545" />
        <KPI label="Neuropsychiatric" value={kpis.n_neuropsychiatric_copan} color="#e65100" />
        <KPI label="Late-onset" value={kpis.n_late_onset_copan} color="#6f42c1" />
        <KPI label="GP Iron (PROMINENT)" value={`${kpis.gp_iron_pct}%`} color="#dc3545" />
        <KPI label="SN Iron" value={`${kpis.sn_iron_pct}%`} color="#e65100" />
        <KPI label="NO Leukodystrophy" value={`${kpis.no_leukodystrophy_pct}%`} color="#198754" />
        <KPI label="Spastic Paraplegia" value={`${kpis.spastic_paraplegia_pct}%`} color="#dc3545" />
        <KPI label="Dystonia" value={`${kpis.dystonia_pct}%`} color="#dc3545" />
        <KPI label="Dysarthria" value={`${kpis.dysarthria_pct}%`} color={COLOR} />
        <KPI label="Seizures" value={`${kpis.has_seizures_pct}%`} color="#e65100" />
        <KPI label="Cogn Decline" value={`${kpis.cognitive_decline_pct}%`} color={COLOR} />
        <KPI label="OCD / Neuropsych" value={`${kpis.ocd_pct}%`} color="#e65100" />
        <KPI label="Lost Ambulation" value={`${kpis.ambulation_lost_pct}%`} color="#dc3545" />
        <KPI label="Axonal Neuropathy" value={`${kpis.axonal_neuropathy_pct}%`} color={COLOR} />
        <KPI label="POLG Tested" value={`${kpis.polg_tested_pct}%`} color="#0d6efd" />
      </div>

      {/* Etiology Distribution */}
      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
              Etiology Distribution (Seed-525)
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
              <Bar label={`Classic-CoPAN (n=${kpis.n_classic_copan})`} value={kpis.classic_mean_onset_yr} max={25} color="#dc3545" />
              <Bar label={`Neuropsychiatric-CoPAN (n=${kpis.n_neuropsychiatric_copan})`} value={kpis.npsy_mean_onset_yr} max={25} color="#e65100" />
              <Bar label={`Late-onset-CoPAN (n=${kpis.n_late_onset_copan})`} value={kpis.late_mean_onset_yr} max={25} color="#6f42c1" />
              <div className="small text-muted mt-2">Classic-CoPAN earliest (5-10yr). Neuropsychiatric (10-15yr). Late-onset (&gt;15yr).</div>
            </div>
          </div>
        </div>
      </div>

      {/* Clinical Highlights */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          Clinical Feature Prevalence (40 patients, seed-525)
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
          Drug Contraindications / Avoidance (COASY/CoPAN specific)
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
                  <td><span className="badge bg-danger">{ci.severity}</span></td>
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
                <th>GP Iron%</th><th>SN Iron%</th><th>Spastic%</th>
                <th>Dystonia%</th><th>CognDecl%</th><th>Seizures%</th><th>OCD%</th><th>Amb Lost%</th><th>DR%</th>
              </tr>
            </thead>
            <tbody>
              {phenotypes.map((ph, i) => (
                <tr key={i}>
                  <td className="fw-bold">{ph.phenotype}</td>
                  <td>{ph.n}</td><td>{ph.pct}%</td>
                  <td>{ph.mean_onset_yr}yr</td>
                  <td>{ph.gp_iron_pct}%</td>
                  <td>{ph.sn_iron_pct}%</td>
                  <td>{ph.spastic_paraplegia_pct}%</td>
                  <td>{ph.dystonia_pct}%</td>
                  <td>{ph.cognitive_decline_pct}%</td>
                  <td>{ph.has_seizures_pct}%</td>
                  <td>{ph.ocd_pct}%</td>
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
          Differential Diagnosis (COASY/CoPAN vs other NBIA + Movement Disorders)
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
          Per-Patient Summary (40 patients, seed-525)
        </div>
        <div className="card-body p-0" style={{ overflowX: 'auto' }}>
          <table className="table table-sm table-striped table-bordered mb-0" style={{ minWidth: 1600 }}>
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Phenotype</th><th>Etiology</th>
                <th>Onset</th><th>Age</th><th>Dur</th>
                <th>GP Fe</th><th>SN Fe</th><th>Leuko</th><th>CC</th><th>Cereb</th>
                <th>Spastic</th><th>Dystonia</th><th>Dysarth</th><th>Ataxia</th>
                <th>Optic</th><th>Neuropathy</th><th>Ambul-Lost</th><th>Cogn</th>
                <th>OCD</th><th>Psych</th><th>Sz</th><th>DR</th>
                <th>Bac</th><th>Trihex</th><th>BTX</th><th>DBS</th><th>POLG</th><th>Panto</th>
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
                  <td className="small">{p.gp_iron ? '🔴' : '—'}</td>
                  <td className="small">{p.sn_iron ? '🔴' : '—'}</td>
                  <td className="small">{p.leukodystrophy ? '⚠️' : '✅'}</td>
                  <td className="small">{p.thin_cc ? '⚠️' : '—'}</td>
                  <td className="small">{p.cerebellar_atrophy ? '⚠️' : '—'}</td>
                  <td className="small">{p.spastic_paraplegia ? '✅' : '—'}</td>
                  <td className="small">{p.dystonia ? `✅ ${p.dystonia_severity || ''}` : '—'}</td>
                  <td className="small">{p.dysarthria ? '✅' : '—'}</td>
                  <td className="small">{p.ataxia ? '✅' : '—'}</td>
                  <td className="small">{p.optic_atrophy ? '⚠️' : '—'}</td>
                  <td className="small">{p.axonal_neuropathy ? '⚠️' : '—'}</td>
                  <td className="small">{p.ambulation_lost ? '🔴' : '—'}</td>
                  <td className="small">{p.cognitive_decline ? '⚠️' : '—'}</td>
                  <td className="small">{p.ocd ? '🧠' : '—'}</td>
                  <td className="small">{p.psychiatric ? '⚠️' : '—'}</td>
                  <td className="small">{p.has_seizures ? '⚡' : '—'}</td>
                  <td className="small">{p.drug_resistant ? '🔴 DR' : '—'}</td>
                  <td className="small">{p.baclofen ? '✅' : '—'}</td>
                  <td className="small">{p.trihexyphenidyl ? '✅' : '—'}</td>
                  <td className="small">{p.btx ? '✅' : '—'}</td>
                  <td className="small">{p.dbs ? '🧠' : '—'}</td>
                  <td className="small">{p.polg_tested ? '✅' : '⚠️'}</td>
                  <td className="small">{p.pantothenate_trial ? '🔬' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function CoAPathwayTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const etiologies = data.etiology_breakdown || [];
  const seizures = data.seizure_breakdown || [];

  return (
    <div>
      {/* MRI Key Info Banner */}
      <div className="alert alert-warning mb-4">
        <strong>🧠 MRI Sequence Priority in COASY/CoPAN:</strong>
        <ol className="mb-0 mt-2 small">
          <li><strong>SWI/T2*:</strong> GP hypointensity — PROMINENT from early disease; UNIFORM (not central-bright) — key DDx from PKAN eye-of-tiger</li>
          <li><strong>SWI/T2*:</strong> SN iron — moderate; less than GP (GP&gt;SN pattern in CoPAN)</li>
          <li><strong>T2/FLAIR:</strong> NO leukodystrophy — ABSENT (white matter SPARED) — critical DDx from FAHN/NBIA3</li>
          <li><strong>T1:</strong> NO T1 halo sign (DDx BPAN/WDR45); thin corpus callosum variable (~40%)</li>
          <li><strong>MRS:</strong> May show reduced NAA/Cr in basal ganglia — metabolic failure signature</li>
        </ol>
        <div className="mt-2 small text-danger fw-bold">
          Key DDx: GP uniform hypointensity (SWI) + NO eye-of-tiger + NO leukodystrophy = CoPAN (COASY). PKAN has eye-of-tiger. FAHN has leukodystrophy. MPAN has optic atrophy 80%. BPAN has T1 halo + biphasic course.
        </div>
      </div>

      {/* CoA Pathway Box */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          CoA Biosynthesis Pathway — COASY Defect (Final 2 Steps)
        </div>
        <div className="card-body small">
          <div className="row">
            <div className="col-md-6">
              <h6 className="fw-bold">Normal CoA Biosynthesis (6 Steps)</h6>
              <ol>
                <li><strong>Pantothenate (Vit B5)</strong> → [<strong>PANK2</strong> — rate-limiting, PKAN gene]</li>
                <li>4&apos;-phosphopantothenate → [PPCS]</li>
                <li>4&apos;-phosphopantothenoyl-cysteine → [PPCDC]</li>
                <li>4&apos;-phosphopantetheine → [<strong>COASY-PPAT</strong> — Step 5]</li>
                <li>Dephospho-CoA → [<strong>COASY-DPCK</strong> — Step 6 = FINAL]</li>
                <li className="text-success fw-bold">→ Coenzyme A (CoA) — cofactor for &gt;100 reactions</li>
              </ol>
              <div className="alert alert-info small py-1 mt-2">
                COASY blocks Steps 5+6. PANK2 (PKAN) blocks Step 1.
                Both → CoA deficiency → GP/SN iron — but different upstream/downstream effects.
              </div>
            </div>
            <div className="col-md-6">
              <h6 className="fw-bold text-danger">COASY LOF Consequences</h6>
              <ul>
                <li>Dephospho-CoA accumulates (toxic — PPAT feedback inhibition)</li>
                <li>CoA critically deficient → <strong>TCA cycle failure</strong> (acetyl-CoA, succinyl-CoA)</li>
                <li>Fatty acid β-oxidation collapse → lipid dysregulation in GP/SN</li>
                <li>Acetylcholine synthesis reduced (CoA required for acetylation)</li>
                <li>GP/SN iron accumulation → metal homeostasis disruption secondary to CoA/lipid failure</li>
                <li>Mitochondrial dysfunction → secondary POLG vulnerability (VPA risk)</li>
              </ul>
              <div className="alert alert-danger small py-1 mt-2">
                Pantothenate supplementation: enters pathway through PANK2 — blocked at COASY.
                Only works in hypomorphic (partial-loss) COASY variants. NOT expected to help null/truncating.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Etiology Breakdown Table */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          Etiology Breakdown (COASY variant classes)
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-light">
              <tr><th>Etiology</th><th>N</th><th>%</th><th>Classic-CoPAN%</th><th>NeuropsychCoPAN%</th><th>GP Iron%</th><th>DR%</th></tr>
            </thead>
            <tbody>
              {etiologies.map((e, i) => (
                <tr key={i}>
                  <td className="fw-bold">{e.etiology}</td>
                  <td>{e.n}</td><td>{e.pct}%</td>
                  <td>{e.classic_copan_pct}%</td>
                  <td>{e.neuropsychiatric_copan_pct}%</td>
                  <td>{e.gp_iron_pct}%</td>
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
          PHT AVOID (dystonia aggravation — CoA pathway). POLG mandatory before VPA. Preferred: LEV, CLB, LCM.
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
          Treatment Protocols — COASY/CoPAN (2026)
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
              <tr><th>Drug</th><th>Severity</th><th>Reason</th><th>Alternative</th></tr>
            </thead>
            <tbody>
              {cis.map((ci, i) => (
                <tr key={i}>
                  <td className="fw-bold text-danger">{ci.drug}</td>
                  <td><span className="badge bg-danger">{ci.severity}</span></td>
                  <td className="small">{ci.reason}</td>
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

export default function COASYPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/coasy/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(e => setError('Backend unavailable: ' + e.message));
    fetch(`${API}/api/coasy/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(() => {});
    fetch(`${API}/api/coasy/definitions`)
      .then(r => r.json()).then(setDefinitions)
      .catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h2 className="fw-bold mb-0" style={{ color: COLOR }}>
          🧬 COASY CoPAN / NBIA6 — CoA Synthase Protein-Associated Neurodegeneration
        </h2>
        <div className="text-muted small">
          COASY (579aa · 17q21.2) · Bifunctional PPAT (aa1-200) + DPCK (aa231-579) · Very rare NBIA (~1-2%; ~25-30 patients worldwide) ·
          OMIM Gene 609686 / Disease CoPAN 615643 · AR biallelic · Seed-525 (40 patients)
        </div>
        <div className="small mt-1">
          <span className="badge me-1" style={{ background: '#dc3545' }}>Classic-CoPAN 65%</span>
          <span className="badge me-1" style={{ background: '#e65100' }}>Neuropsychiatric-CoPAN 25%</span>
          <span className="badge me-1" style={{ background: '#6f42c1' }}>Late-onset-CoPAN 10%</span>
          <span className="badge me-1 bg-danger">GP Iron PROMINENT Early</span>
          <span className="badge me-1 bg-success">NO Leukodystrophy (DDx FAHN)</span>
          <span className="badge me-1 bg-warning text-dark">NO Eye-of-Tiger (DDx PKAN)</span>
          <span className="badge me-1 bg-dark">PHT AVOID</span>
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
      {tab === 'CoA Pathway & Iron' && <CoAPathwayTab data={breakdown} />}
      {tab === 'Treatments' && <TreatmentsTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
