'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Chorea & Imaging', 'Treatments', 'Definitions'];
const COLOR = '#4a148c';   // deep purple — OPA3/Costeff (mitochondrial OMM, chorea, optic atrophy)
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

function Alert({ variant, text }) {
  const bg = variant === 'danger' ? '#ffebee' : variant === 'warning' ? '#fff8e1' : variant === 'success' ? '#e8f5e9' : '#f3e5f5';
  const border = variant === 'danger' ? '#c62828' : variant === 'warning' ? '#f57f17' : variant === 'success' ? '#2e7d32' : COLOR;
  return (
    <div className="mb-2 p-2 rounded small" style={{ background: bg, borderLeft: `4px solid ${border}` }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview...</div>;
  const kpis = data.kpis || {};
  const highlights = data.clinical_highlights || [];
  const cis = data.contraindications || [];
  const thresholds = data.thresholds || [];

  return (
    <div>
      {/* Identity */}
      <SectionCard title="🧬 Disease Identity">
        <div className="row g-2 small">
          <div className="col-md-6"><strong>Disease:</strong> {data.disease}</div>
          <div className="col-md-6"><strong>Gene:</strong> {data.gene?.split(';')[0]}</div>
          <div className="col-md-4"><strong>Chromosome:</strong> {data.chromosome}</div>
          <div className="col-md-4"><strong>OMIM Gene:</strong> {data.omim_gene} &nbsp; <strong>Disease:</strong> {data.omim_disease}</div>
          <div className="col-md-4"><strong>Inheritance:</strong> {data.inheritance?.split(';')[0]}</div>
          <div className="col-md-6"><strong>Prevalence:</strong> {data.prevalence}</div>
          <div className="col-md-6"><strong>First described:</strong> {data.first_described}</div>
        </div>
      </SectionCard>

      {/* KPIs */}
      <SectionCard title="📊 Cohort KPIs (n=40, seed-535)">
        <div className="row g-2">
          <KPI label="Total patients" value={kpis.n_patients} color={COLOR} />
          <KPI label="Classic Costeff" value={kpis.n_classic} color={COLOR} />
          <KPI label="Chorea-Dominant" value={kpis.n_chorea} color="#7b1fa2" />
          <KPI label="Optic-Dominant" value={kpis.n_optic} color="#ab47bc" />
          <KPI label="Optic atrophy %" value="100%" color="#c62828" />
          <KPI label="Chorea %" value={`${kpis.chorea_pct}%`} color={COLOR} />
          <KPI label="Spasticity %" value={`${kpis.spasticity_pct}%`} color="#5e35b1" />
          <KPI label="Seizures %" value={`${kpis.seizures_pct}%`} color="#6a1b9a" />
          <KPI label="Mean onset (mo)" value={kpis.mean_optic_onset_mo} color={COLOR} />
          <KPI label="Mean 3-MGA" value={`${kpis.mean_mga}`} color="#7b1fa2" />
          <KPI label="UHDRS chorea" value={kpis.mean_uhdrs_chorea} color={COLOR} />
          <KPI label="Mean VA (LogMAR)" value={kpis.mean_va_logmar} color="#c62828" />
        </div>
      </SectionCard>

      {/* Alerts */}
      <SectionCard title="⚡ Critical Clinical Alerts">
        <Alert variant="info" text="🎯 CHOREA DOMINANT (not dystonia) — primary movement disorder; UHDRS TMS chorea subscore at every visit; tetrabenazine/deutetrabenazine for significant chorea" />
        <Alert variant="danger" text="🚫 CBZ/OXC/PHT AVOID — sodium channel blockers paradoxically worsen choreic movements; CBZ trap: may reduce seizures but worsens chorea" />
        <Alert variant="warning" text="⚠️ VPA RELATIVE CAUTION — NOT absolute CI (unlike MECR); OPA3 lacks lipoic acid pathway disruption BUT monitor ammonia/LFTs; POLG screen mandatory" />
        <Alert variant="warning" text="💉 CYP2D6 GENOTYPE MANDATORY before tetrabenazine/deutetrabenazine — poor metabolisers need 50% dose reduction; depression PHQ-9 monitoring q6M" />
        <Alert variant="success" text="✅ NO GP IRON on MRI — key DDx from MECR/NBIA; if GP iron found → reconsider OPA3 diagnosis; SWI mandatory in all 3-MGA-uria patients" />
        <Alert variant="info" text="🔬 3-MGA 100% — Type III (Costeff); level 40-200 mmol/mol creatinine; higher than MECR Type IV (20-100); shared biomarker, different mechanism" />
      </SectionCard>

      {/* Clinical Highlights */}
      <SectionCard title="🩺 Clinical Findings (% of cohort)">
        {highlights.map((h, i) => (
          <div key={i} className="mb-3">
            <Bar label={h.finding} value={h.pct} max={100} />
            <div className="text-muted small ms-1">{h.note}</div>
          </div>
        ))}
      </SectionCard>

      {/* Thresholds */}
      <SectionCard title="📏 Action Thresholds">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead><tr><th>Metric</th><th>Threshold</th><th>Action</th></tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{t.metric}</td>
                  <td><span className="badge" style={{ background: LIGHT, color: COLOR }}>{t.threshold}</span></td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const pheno = data.phenotype_breakdown || [];
  const variants = data.variant_breakdown || [];
  const treatments = data.treatment_breakdown || [];
  const sex = data.sex_distribution || [];

  return (
    <div>
      <SectionCard title="🏥 Phenotype Breakdown">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>Phenotype</th><th>N</th><th>%</th>
                <th>Optic onset (mo)</th><th>Chorea %</th>
                <th>Spasticity %</th><th>Seizures %</th>
                <th>Mean 3-MGA</th><th>UHDRS chorea</th>
              </tr>
            </thead>
            <tbody>
              {pheno.map((p, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{p.phenotype}</td>
                  <td>{p.n}</td>
                  <td>{p.pct}%</td>
                  <td>{p.mean_optic_onset_mo}</td>
                  <td>{p.chorea_pct}%</td>
                  <td>{p.spasticity_pct}%</td>
                  <td>{p.seizures_pct}%</td>
                  <td>{p.mean_mga}</td>
                  <td>{p.mean_uhdrs_chorea}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="🧬 Variant Distribution">
            {variants.map((v, i) => (
              <div key={i} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span className="text-truncate" style={{ maxWidth: '75%' }} title={v.variant}>{v.variant}</span>
                  <span className="text-muted">{v.n} ({v.pct}%)</span>
                </div>
                <div className="progress" style={{ height: 10 }}>
                  <div className="progress-bar" style={{ width: `${v.pct}%`, backgroundColor: COLOR }} />
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="💊 Treatment Distribution">
            {treatments.map((t, i) => (
              <div key={i} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{t.treatment}</span>
                  <span className="text-muted">{t.n} ({t.pct}%)</span>
                </div>
                <div className="progress" style={{ height: 10 }}>
                  <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: '#7b1fa2' }} />
                </div>
              </div>
            ))}
          </SectionCard>
          <SectionCard title="⚧ Sex Distribution">
            {sex.map((s, i) => (
              <div key={i} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{s.sex}</span><span className="text-muted">{s.n} ({s.pct}%)</span>
                </div>
                <div className="progress" style={{ height: 10 }}>
                  <div className="progress-bar" style={{ width: `${s.pct}%`, backgroundColor: i === 0 ? '#ab47bc' : '#5e35b1' }} />
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

function ImagingTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const mga = data.mga_breakdown || [];
  const uhdrs = data.uhdrs_breakdown || [];
  const va = data.va_breakdown || [];

  return (
    <div>
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="🧪 3-MGA-uria Distribution (mmol/mol creatinine)">
            <Alert variant="info" text="OPA3 Type III pattern: 40-200 range (higher than MECR Type IV 20-100)" />
            {mga.map((m, i) => (
              <div key={i} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{m.range} mmol/mol Cr</span>
                  <span className="text-muted">{m.n} ({m.pct}%)</span>
                </div>
                <div className="progress" style={{ height: 10 }}>
                  <div className="progress-bar" style={{ width: `${m.pct}%`, backgroundColor: COLOR }} />
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="🕺 UHDRS Chorea Subscore Distribution">
            <Alert variant="info" text="Action threshold: UHDRS chorea >8 → initiate tetrabenazine/deutetrabenazine" />
            {uhdrs.map((u, i) => (
              <div key={i} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{u.category}</span>
                  <span className="text-muted">{u.n} ({u.pct}%)</span>
                </div>
                <div className="progress" style={{ height: 10 }}>
                  <div className="progress-bar" style={{ width: `${u.pct}%`, backgroundColor: '#7b1fa2' }} />
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="👁️ Visual Acuity Distribution (LogMAR)">
        <Alert variant="success" text="Action: LogMAR >0.5 (6/18 equivalent) → low-vision aids + mobility assessment + educational support" />
        {va.map((v, i) => (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span>{v.category}</span>
              <span className="text-muted">{v.n} ({v.pct}%)</span>
            </div>
            <div className="progress" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${v.pct}%`, backgroundColor: '#c62828' }} />
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🧲 Brain MRI Summary — OPA3 vs MECR DDx">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ background: LIGHT }}>
              <tr><th>MRI Feature</th><th>OPA3 Costeff</th><th>MECR MEPAN</th><th>PKAN (NBIA1)</th></tr>
            </thead>
            <tbody>
              <tr><td className="fw-semibold">GP iron (SWI)</td><td className="text-success">ABSENT ✅</td><td className="text-danger">75-85% ❌</td><td className="text-danger">Eye-of-Tiger ❌</td></tr>
              <tr><td className="fw-semibold">Leukodystrophy</td><td className="text-success">ABSENT ✅</td><td className="text-success">ABSENT ✅</td><td className="text-success">ABSENT ✅</td></tr>
              <tr><td className="fw-semibold">Cerebellar atrophy</td><td className="text-success">Uncommon ✅</td><td className="text-warning">60-70% ⚠️</td><td className="text-success">Rare ✅</td></tr>
              <tr><td className="fw-semibold">Normal MRI</td><td className="text-success">~75% ✅</td><td className="text-danger">Rare ❌</td><td className="text-danger">Never ❌</td></tr>
              <tr><td className="fw-semibold">Eye-of-Tiger sign</td><td className="text-success">ABSENT ✅</td><td className="text-success">ABSENT ✅</td><td className="text-danger">PATHOGNOMONIC ❌</td></tr>
            </tbody>
          </table>
        </div>
        <Alert variant="success" text="Key rule: PRESENCE of GP iron on SWI in a 3-MGA-uria patient → MECR/NBIA, NOT OPA3. ABSENCE of GP iron + chorea + optic atrophy + 3-MGA → OPA3 first." />
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;
  const cis = data.contraindications || [];

  return (
    <div>
      <SectionCard title="🚫 Drug Safety & Contraindications">
        {cis.map((ci, i) => {
          const sev = ci.severity;
          const variant =
            sev.includes('ABSOLUTE') ? 'danger'
            : sev.includes('AVOID') ? 'warning'
            : sev.includes('CAUTION') ? 'warning'
            : sev.includes('PREFERRED') || sev.includes('FIRST-LINE') ? 'success'
            : 'info';
          return (
            <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${variant === 'danger' ? '#c62828' : variant === 'warning' ? '#f57f17' : variant === 'success' ? '#2e7d32' : COLOR}` }}>
              <div className="card-body py-2">
                <div className="d-flex justify-content-between align-items-start">
                  <strong className="small">{ci.drug}</strong>
                  <span className="badge ms-2" style={{ background: variant === 'danger' ? '#c62828' : variant === 'warning' ? '#f57f17' : variant === 'success' ? '#2e7d32' : COLOR, fontSize: '0.7rem' }}>
                    {sev}
                  </span>
                </div>
                <p className="small text-muted mt-1 mb-1">{ci.reason}</p>
                {ci.alternative && <p className="small mb-0"><strong>Alternative:</strong> {ci.alternative}</p>}
              </div>
            </div>
          );
        })}
      </SectionCard>

      <SectionCard title="📋 Treatment Protocol Summary">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Treatment</th><th>Indication</th><th>Level</th><th>Notes</th></tr></thead>
            <tbody>
              <tr><td>Deutetrabenazine</td><td>Chorea (preferred)</td><td>C</td><td>6-48mg/day; lower side-effect profile; CYP2D6 required</td></tr>
              <tr><td>Tetrabenazine</td><td>Chorea</td><td>C</td><td>12.5-50mg/day; CYP2D6 mandatory; depression monitoring q6M</td></tr>
              <tr><td>LEV</td><td>Seizures (first-line)</td><td>B</td><td>Renal excretion; no mito interactions; 500-3000mg/day</td></tr>
              <tr><td>Baclofen</td><td>Spasticity</td><td>C</td><td>5-20mg TDS; GABA-B; monitor respiratory depression</td></tr>
              <tr><td>CLB</td><td>Seizures (second-line)</td><td>C</td><td>Adjunct to LEV; minimal mito risk</td></tr>
              <tr><td>DHA supplementation</td><td>Supportive</td><td>D</td><td>500-1000mg/day; OMM membrane stabilisation; anecdotal</td></tr>
              <tr><td>Low-vision aids</td><td>Optic atrophy</td><td>A</td><td>Mandatory when VA LogMAR &gt;0.5; mobility + education support</td></tr>
              <tr><td>Physiotherapy</td><td>Spasticity + gait</td><td>A</td><td>Ongoing; gait training; spasticity prevention contractures</td></tr>
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🔬 Diagnostic Workup Checklist">
        <ul className="small">
          <li>✅ Urine organic acids: 3-MGA elevated (&gt;40 mmol/mol creatinine typical)</li>
          <li>✅ Ophthalmology: VEP + ERG + OCT (optic atrophy; ERG NORMAL in OPA3)</li>
          <li>✅ Brain MRI (SWI mandatory): NO GP iron; usually normal or mild periventricular T2</li>
          <li>✅ OPA3 gene sequencing (WES/targeted panel): biallelic pathogenic variants confirm</li>
          <li>✅ POLG screening: mandatory before VPA consideration</li>
          <li>✅ CYP2D6 genotyping: before tetrabenazine/deutetrabenazine initiation</li>
          <li>✅ Neurological exam: UHDRS TMS chorea subscore + Ashworth spasticity scale baseline</li>
          <li>✅ Cognitive assessment: IQ/adaptive functioning; educational support planning</li>
          <li>✅ Plasma amino acids: normal (DDx NKH — glycine normal in OPA3)</li>
          <li>✅ PHQ-9 depression screening: baseline before tetrabenazine; repeat q6M</li>
        </ul>
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  const [open, setOpen] = useState(null);
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const defs = data.definitions || [];

  return (
    <div>
      <SectionCard title="📚 OPA3 / Costeff Syndrome — Definitions & Concepts">
        {defs.map((d, i) => (
          <div key={i} className="card mb-2 shadow-sm">
            <div
              className="card-header d-flex justify-content-between align-items-center py-2 px-3"
              style={{ cursor: 'pointer', background: open === i ? LIGHT : '#fff' }}
              onClick={() => setOpen(open === i ? null : i)}
            >
              <span className="fw-semibold small" style={{ color: COLOR }}>{d.term}</span>
              <span className="text-muted small">{open === i ? '▲' : '▼'}</span>
            </div>
            {open === i && (
              <div className="card-body py-2 px-3">
                <p className="small fw-semibold mb-1">{d.full}</p>
                <p className="small text-muted mb-0" style={{ whiteSpace: 'pre-wrap' }}>{d.detail}</p>
              </div>
            )}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

export default function OPA3Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/opa3/overview`).then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/opa3/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/opa3/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 OPA3 — Costeff Syndrome / 3-MGA Type III
        </h4>
        <p className="text-muted small mb-0">
          OPA3-179aa-OMM · 19q13.2-q13.3 · AR biallelic LOF · Iraqi Jewish founder p.Gln105* ·
          Chorea + Optic Atrophy + 3-MGA · NO GP iron (DDx MECR) · OMIM Gene: 606580 · Disease: 258501
        </p>
      </div>

      {error && (
        <div className="alert alert-danger small">API error: {error}. Check backend is running on port 8010.</div>
      )}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <ImagingTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={overview} />}
      {tab === 4 && <DefinitionsTab data={defs} />}
    </div>
  );
}
