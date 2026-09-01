'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Iron Distribution & MRI', 'Treatments', 'Definitions'];
const COLOR = '#b71c1c';   // deep red — FTL/Neuroferritinopathy (iron accumulation + chorea dominant)
const LIGHT = '#ffebee';

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
        <strong>FTL (19q13.33) — 175aa Ferritin Light Chain · OMIM 134790/Neuroferritinopathy 606159 · NBIA7 · AD ONLY:</strong>{' '}
        The ONLY autosomal dominant NBIA. Heterozygous FTL exon 4 frameshifts (c.460InsA most common ~50%) destroy the E-helix 4-fold iron channel.{' '}
        <strong className="text-danger">LOW serum ferritin (&lt;30 ng/mL) PATHOGNOMONIC — only NBIA with a biochemical diagnostic marker.</strong>{' '}
        Caudate+putamen iron DOMINANT and earliest (NOT GP-first — critical DDx from PKAN). Cavitations (cystic signal voids) PATHOGNOMONIC in advanced disease.{' '}
        <span className="fw-bold" style={{ color: COLOR }}>
          Chorea DOMINANT early (distinguishes from all other NBIA). 3 phenotypes: Choreic-predominant 60% · Mixed Hyperkinetic 30% · Parkinsonism-predominant 10%.
          PHT/CBZ AVOID (worsens chorea). Tetrabenazine/Deutetrabenazine Level C. Seizures RARE (&lt;10%).
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Choreic-predominant" value={kpis.n_choreic} color="#dc3545" />
        <KPI label="Mixed Hyperkinetic" value={kpis.n_mixed} color="#e65100" />
        <KPI label="Parkinsonism-pred." value={kpis.n_parkinsonism} color="#6f42c1" />
        <KPI label="Low Ferritin (PATHOGN)" value={`${kpis.low_ferritin_pct}%`} color="#dc3545" />
        <KPI label="Caudate/Put Iron" value={`${kpis.caudate_putamen_iron_pct}%`} color="#dc3545" />
        <KPI label="SN Iron" value={`${kpis.sn_iron_pct}%`} color="#e65100" />
        <KPI label="GP Iron" value={`${kpis.gp_iron_pct}%`} color="#e65100" />
        <KPI label="Cavitations (adv)" value={`${kpis.cavitations_pct}%`} color="#dc3545" />
        <KPI label="Chorea (dominant)" value={`${kpis.chorea_pct}%`} color="#dc3545" />
        <KPI label="Dystonia" value={`${kpis.dystonia_pct}%`} color="#e65100" />
        <KPI label="Parkinsonism" value={`${kpis.parkinsonism_pct}%`} color={COLOR} />
        <KPI label="Dysarthria" value={`${kpis.dysarthria_pct}%`} color={COLOR} />
        <KPI label="Dysphagia" value={`${kpis.dysphagia_pct}%`} color="#e65100" />
        <KPI label="Cognitive Decline" value={`${kpis.cognitive_decline_pct}%`} color={COLOR} />
        <KPI label="Seizures (RARE)" value={`${kpis.has_seizures_pct}%`} color="#198754" />
      </div>

      {/* Etiology Distribution */}
      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
              Etiology Distribution (Seed-527)
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
              <Bar label={`Choreic-predominant (n=${kpis.n_choreic})`} value={kpis.choreic_mean_onset_yr} max={70} color="#dc3545" />
              <Bar label={`Mixed Hyperkinetic (n=${kpis.n_mixed})`} value={kpis.mixed_mean_onset_yr} max={70} color="#e65100" />
              <Bar label={`Parkinsonism-predominant (n=${kpis.n_parkinsonism})`} value={kpis.parkinson_mean_onset_yr} max={70} color="#6f42c1" />
              <div className="small text-muted mt-2">Adult-onset only. Choreic earliest (25-45yr). Parkinsonism-predominant latest (&gt;50yr).</div>
            </div>
          </div>
        </div>
      </div>

      {/* Clinical Highlights */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          Clinical Feature Prevalence (40 patients, seed-527)
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
          Drug Contraindications / Avoidance (FTL / Neuroferritinopathy specific)
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
                <th>Chorea%</th><th>Dystonia%</th><th>Parkins%</th>
                <th>Cavitations%</th><th>Low Fe%</th><th>CognDecl%</th><th>Amb Lost%</th><th>Sz%</th>
              </tr>
            </thead>
            <tbody>
              {phenotypes.map((ph, i) => (
                <tr key={i}>
                  <td className="fw-bold">{ph.phenotype}</td>
                  <td>{ph.n}</td><td>{ph.pct}%</td>
                  <td>{ph.mean_onset_yr}yr</td>
                  <td>{ph.chorea_pct}%</td>
                  <td>{ph.dystonia_pct}%</td>
                  <td>{ph.parkinsonism_pct}%</td>
                  <td>{ph.cavitations_pct}%</td>
                  <td>{ph.low_ferritin_pct}%</td>
                  <td>{ph.cognitive_decline_pct}%</td>
                  <td>{ph.ambulation_lost_pct}%</td>
                  <td className="text-success">{ph.has_seizures_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* DDx Table */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: '#37474f', color: '#fff' }}>
          Differential Diagnosis (FTL Neuroferritinopathy vs other NBIA + Movement Disorders)
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

      {/* Per-Patient Table */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          Per-Patient Summary (40 patients, seed-527)
        </div>
        <div className="card-body p-0" style={{ overflowX: 'auto' }}>
          <table className="table table-sm table-striped table-bordered mb-0" style={{ minWidth: 1800 }}>
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Phenotype</th><th>Etiology</th>
                <th>Onset</th><th>Age</th><th>Dur</th>
                <th>Fe (ng/mL)</th><th>Ca/Pu Fe</th><th>SN Fe</th><th>GP Fe</th><th>Cereb Fe</th><th>Cavit</th>
                <th>Chorea</th><th>Dystonia</th><th>Parkins</th><th>Ataxia</th>
                <th>Dysarth</th><th>Dysphagia</th><th>Amb-Lost</th>
                <th>Cogn</th><th>Psych</th><th>Sz</th><th>DR</th>
                <th>TBZ</th><th>DTBZ</th><th>CLB</th><th>LDOPA</th><th>DBS</th><th>Defer</th><th>POLG</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{p.id}</td>
                  <td className="small">{p.phenotype}</td>
                  <td className="small">{p.etiology?.replace(/ /g, '-').slice(0, 18)}</td>
                  <td className="small">{p.onset_yr}yr</td>
                  <td className="small">{p.current_age}</td>
                  <td className="small">{p.disease_duration_yr}yr</td>
                  <td className="small" style={{ color: p.low_ferritin ? '#dc3545' : '#198754' }}>
                    {p.ferritin_value} {p.low_ferritin ? '🔴' : ''}
                  </td>
                  <td className="small">{p.caudate_putamen_iron ? '🔴' : '—'}</td>
                  <td className="small">{p.sn_iron ? '🔴' : '—'}</td>
                  <td className="small">{p.gp_iron ? '🔴' : '—'}</td>
                  <td className="small">{p.cerebellar_iron ? '⚠️' : '—'}</td>
                  <td className="small">{p.cavitations ? '⚠️CAVIT' : '—'}</td>
                  <td className="small">{p.chorea ? `✅ ${p.chorea_severity || ''}` : '—'}</td>
                  <td className="small">{p.dystonia ? `✅ ${p.dystonia_severity || ''}` : '—'}</td>
                  <td className="small">{p.parkinsonism ? '✅' : '—'}</td>
                  <td className="small">{p.cerebellar_ataxia ? '✅' : '—'}</td>
                  <td className="small">{p.dysarthria ? '✅' : '—'}</td>
                  <td className="small">{p.dysphagia ? '⚠️' : '—'}</td>
                  <td className="small">{p.ambulation_lost ? '🔴' : '—'}</td>
                  <td className="small">{p.cognitive_decline ? '⚠️' : '—'}</td>
                  <td className="small">{p.psychiatric ? '⚠️' : '—'}</td>
                  <td className="small">{p.has_seizures ? '⚡' : '—'}</td>
                  <td className="small">{p.drug_resistant ? '🔴 DR' : '—'}</td>
                  <td className="small">{p.tetrabenazine ? '✅' : '—'}</td>
                  <td className="small">{p.deutetrabenazine ? '✅' : '—'}</td>
                  <td className="small">{p.clonazepam ? '✅' : '—'}</td>
                  <td className="small">{p.levodopa ? '✅' : '—'}</td>
                  <td className="small">{p.dbs ? '🧠' : '—'}</td>
                  <td className="small">{p.deferiprone_trial ? '🔬' : '—'}</td>
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

function IronMRITab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const etiologies = data.etiology_breakdown || [];
  const seizures = data.seizure_breakdown || [];

  return (
    <div>
      {/* MRI Priority Banner */}
      <div className="alert alert-danger mb-4">
        <strong>🧠 FTL Neuroferritinopathy — Iron Distribution vs Other NBIA (KEY DIFFERENTIATOR):</strong>
        <div className="row mt-2 small">
          <div className="col-md-6">
            <table className="table table-sm table-bordered table-light mb-0">
              <thead><tr><th>NBIA</th><th>Iron Dominant Site</th><th>Pathognomonic MRI</th></tr></thead>
              <tbody>
                <tr className="table-danger"><td><strong>FTL (NBIA7)</strong></td><td><strong>CAUDATE + PUTAMEN first</strong></td><td>Cavitations in GP/putamen (advanced)</td></tr>
                <tr><td>PKAN (NBIA1)</td><td>GP DOMINANT</td><td>Eye-of-tiger sign (T2 GP)</td></tr>
                <tr><td>MPAN (NBIA4)</td><td>GP + SN</td><td>Optic nerve + nerve biopsy</td></tr>
                <tr><td>BPAN (NBIA5)</td><td>SN + GP</td><td>T1 halo sign (WM around SN)</td></tr>
                <tr><td>PLAN (NBIA2)</td><td>GP late / cerebellar earliest</td><td>Cerebellar atrophy + spheroid bodies</td></tr>
                <tr><td>FAHN (NBIA3)</td><td>GP+SN mild / WM first</td><td>Leukodystrophy + GP/SN mild</td></tr>
                <tr><td>CoPAN (NBIA6)</td><td>GP prominent early</td><td>Uniform GP hypointensity (no EoT)</td></tr>
              </tbody>
            </table>
          </div>
          <div className="col-md-6">
            <h6 className="fw-bold">FTL MRI Sequence Priority:</h6>
            <ol className="mb-0">
              <li><strong>SWI / T2*:</strong> Caudate + putamen hypointensity — EARLIEST. GP less prominent than caudate. Dentate nucleus variable.</li>
              <li><strong>T2 (TSE/FSE):</strong> Caudate/putamen hypointensity + look for cavitations (focal signal voids = cystic changes) in GP/putamen → PATHOGNOMONIC advanced.</li>
              <li><strong>T1:</strong> NO T1 halo sign (DDx BPAN). GP cavitations may appear as T1 hyperintense foci if proteinaceous.</li>
              <li><strong>DWI:</strong> Restricted diffusion in active iron zones (early disease).</li>
              <li><strong>MRS:</strong> Reduced NAA/Cr in caudate/putamen — metabolic failure.</li>
            </ol>
            <div className="alert alert-warning small py-1 mt-2">
              Low serum ferritin (&lt;30 ng/mL) + adult chorea + caudate/putamen iron on SWI = FTL until proven otherwise. Order FTL genetic panel immediately.
            </div>
          </div>
        </div>
      </div>

      {/* Iron Evolution Timeline */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          FTL Iron Accumulation Timeline (disease duration)
        </div>
        <div className="card-body">
          <div className="row g-3">
            {[
              { phase: "0-3yr", findings: "Caudate + putamen T2*/SWI hypointensity (earliest). Serum ferritin already low. Chorea onset.", bg: "#ffcdd2" },
              { phase: "3-8yr", findings: "SN iron develops. GP iron mild. Cerebellar dentate in ~40%. Chorea + emerging dystonia. Cognitive decline onset.", bg: "#ef9a9a" },
              { phase: "8-15yr", findings: "GP iron prominent. Cavitations begin (GP, then putamen) — cystic signal voids on T2/GRE. Parkinsonism emerges.", bg: "#e57373" },
              { phase: ">15yr", findings: "Cavitations extensive — pathognomonic. Thalamus iron. Severe dystonia + parkinsonism. Dysphagia/aspiration dominant disability.", bg: "#b71c1c" },
            ].map((ph, i) => (
              <div key={i} className="col-md-3">
                <div className="card border-0" style={{ background: ph.bg }}>
                  <div className="card-body py-2">
                    <div className="fw-bold small" style={{ color: i < 2 ? '#333' : '#fff' }}>Phase: {ph.phase}</div>
                    <div className="small" style={{ color: i < 2 ? '#555' : '#ffe0e0' }}>{ph.findings}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* FTL Molecular Mechanism */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          FTL Molecular Mechanism — Why Iron Accumulates + Serum Ferritin Paradoxically LOW
        </div>
        <div className="card-body small">
          <div className="row">
            <div className="col-md-6">
              <h6 className="fw-bold">Normal Ferritin Function</h6>
              <ol>
                <li><strong>FTH1 (heavy chain):</strong> Ferroxidase activity — Fe2+ → Fe3+ oxidation</li>
                <li><strong>FTL (light chain):</strong> Iron nucleation — Glu60/Glu61 coordinate Fe3+ at core</li>
                <li>24-subunit heteropolymer (FTH1+FTL): stores up to 4,500 Fe3+ ions</li>
                <li>4-fold channel (E-helix, aa 130-175): iron ENTRY/EXIT pore</li>
                <li>Hepatocytes secrete ferritin → <strong>serum ferritin = iron stores indicator</strong></li>
              </ol>
            </div>
            <div className="col-md-6">
              <h6 className="fw-bold text-danger">FTL Frameshift LOF</h6>
              <ol>
                <li>Aberrant C-terminal peptide <strong>replaces E-helix</strong> → 4-fold channel DESTROYED</li>
                <li>Mutant FTL incorporates into 24-mers → entire complex iron storage impaired</li>
                <li>Cytosolic free Fe2+ rises → <strong>Fenton chemistry</strong> → OH• radical → oxidative death</li>
                <li>Hepatocyte ferritin secretion impaired → <strong>SERUM FERRITIN LOW</strong> (paradox)</li>
                <li>Brain accumulates iron → caudate/putamen/SN/dentate — neurodegeneration</li>
              </ol>
              <div className="alert alert-danger small py-1 mt-2">
                Paradox: HIGH tissue iron + LOW serum ferritin. Only NBIA with biochemical marker.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Etiology Breakdown */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold" style={{ background: COLOR, color: '#fff' }}>
          Etiology Breakdown (FTL variant classes)
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-light">
              <tr><th>Etiology</th><th>N</th><th>%</th><th>Choreic%</th><th>Mixed%</th><th>Cavitations%</th><th>DR%</th></tr>
            </thead>
            <tbody>
              {etiologies.map((e, i) => (
                <tr key={i}>
                  <td className="fw-bold">{e.etiology}</td>
                  <td>{e.n}</td><td>{e.pct}%</td>
                  <td>{e.choreic_pct}%</td>
                  <td>{e.mixed_pct}%</td>
                  <td>{e.cavitations_pct}%</td>
                  <td className="text-danger">{e.drug_resistant_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Seizure Breakdown */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold bg-success text-white">
          Seizure Breakdown (among rare patients with seizures — &lt;10% overall)
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
          PHT/CBZ AVOID (worsens chorea). Preferred if AED needed: LEV (first-line); CLB (second-line). Myoclonic jerks may be choreic — EEG-EMG correlation before AED.
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
          Treatment Protocols — FTL Neuroferritinopathy (2026)
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

export default function FTLPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/ftl/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(e => setError('Backend unavailable: ' + e.message));
    fetch(`${API}/api/ftl/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(() => {});
    fetch(`${API}/api/ftl/definitions`)
      .then(r => r.json()).then(setDefinitions)
      .catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h2 className="fw-bold mb-0" style={{ color: COLOR }}>
          🧬 FTL Neuroferritinopathy / NBIA7 — Ferritin Light Chain Neurodegeneration
        </h2>
        <div className="text-muted small">
          FTL (175aa · 19q13.33) · E-helix 4-fold channel frameshifts · ONLY AD NBIA (~50-70 families worldwide) ·
          OMIM Gene 134790 / Disease Neuroferritinopathy 606159 · Seed-527 (40 patients)
        </div>
        <div className="small mt-1">
          <span className="badge me-1" style={{ background: '#dc3545' }}>Choreic-predominant 60%</span>
          <span className="badge me-1" style={{ background: '#e65100' }}>Mixed Hyperkinetic 30%</span>
          <span className="badge me-1" style={{ background: '#6f42c1' }}>Parkinsonism-predominant 10%</span>
          <span className="badge me-1 bg-danger">LOW Ferritin PATHOGNOMONIC</span>
          <span className="badge me-1 bg-warning text-dark">Caudate/Putamen Iron First (NOT GP)</span>
          <span className="badge me-1 bg-dark">Cavitations Advanced</span>
          <span className="badge me-1 bg-success">Seizures RARE &lt;10%</span>
          <span className="badge me-1 bg-danger">PHT/CBZ AVOID</span>
          <span className="badge me-1 bg-primary">ONLY AD NBIA</span>
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
      {tab === 'Iron Distribution & MRI' && <IronMRITab data={breakdown} />}
      {tab === 'Treatments' && <TreatmentsTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
