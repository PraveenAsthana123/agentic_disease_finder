'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#263238';   // dark blue-grey slate — CLN10 / Congenital NCL / CTSD
const ACCENT2 = '#880e4f';   // deep-rose — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep-orange — urgent alerts / warnings
const ACCENT4 = '#1565c0';   // deep-blue — safe treatments / monitoring

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
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header fw-bold" style={{ backgroundColor: '#eceff1', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

// ── TAB 1: Overview ──────────────────────────────────────────────────────────
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Alert
        variant="danger"
        text="⛔ ABSOLUTE CI: VGB (Vigabatrin) — CLN10 non-congenital has 90-95% progressive retinal NCL; VGB = CATASTROPHIC combined blindness. INFANTILE SPASMS TRAP: CLN10 West syndrome → standard protocol prescribes VGB → ABSOLUTE CI → use ACTH/prednisolone instead. CBZ / Oxcarbazepine / Phenytoin / Fosphenytoin — ABSOLUTE CI ALL CLN10 PHENOTYPES. NEONATAL SE TRAP: standard neonatal SE protocol uses fosphenytoin → ABSOLUTE CI in CLN10 neonate → IV PHB 20 mg/kg + IV LEV 40-60 mg/kg (NOT fosphenytoin). Tiagabine (TGB) — ABSOLUTE CI: NCSE risk."
      />
      <Alert
        variant="warning"
        text="⚠ CONGENITAL CLN10 — MOST SEVERE NCL: seizures from birth, microcephaly, fatal <1 year. HIGH RISK: GBP/Pregabalin (myoclonus worsening). LTG monotherapy (insufficient myoclonus cover — adjunct to VPA only). AED taper: NEVER — progressive NCL. VPA neonatal caution: hepatic immaturity <4 weeks → PHB preferred neonatally. POLG1 exclusion mandatory before VPA in infants <2y. Photosensitivity 40%."
      />
      <Alert
        variant="info"
        text="🧠 GRODs EM = CLN1 (PPT1) OR CLN10 (CTSD) — BOTH MUST BE TESTED: (1) PPT1 enzyme DBS first (1-3 days). (2) If normal/borderline → CTSD WES. Borderline PPT1 does NOT confirm CLN1 — CTSD activates PPT1, so CLN10 can reduce PPT1 secondarily. No standardised CTSD DBS enzyme assay → WES is primary CTSD diagnostic. VPA SAFE (lysosomal, NOT mitochondrial). CTSD ERT conceptually feasible (soluble lysosomal enzyme — CLN2 cerliponase precedent). All CLN10 → BDSRA/NCL Resource enrolment mandatory."
      />
      <Alert
        variant="success"
        text="✅ DIAGNOSIS STEPS: (1) EM skin biopsy → GRODs (NCL confirmed — days). (2) PPT1 enzyme DBS (1-3 days) — if clearly low → CLN1; if normal/borderline → step 3. (3) CTSD WES / NCL gene panel. Turkish consanguineous → p.Trp383Stop PCR first. Neonatal: GRODs skin biopsy + PPT1 DBS + CTSD WES simultaneously from day 1. (4) Ophthalmology ERG + VEP. (5) POLG1 exclusion before VPA (infants <2y). (6) Congenital CLN10: palliative care team from birth — ACP (ventilation, resuscitation, comfort goals) from day of diagnosis. (7) BDSRA registry enrolment."
      />

      <div className="row mb-4">
        <KPI label="Cohort" value={`${ov.cohort_size} pts`} color={ACCENT} />
        <KPI label="Congenital %" value={`${ov.congenital_pct}%`} color={ACCENT2} />
        <KPI label="Late-Infantile %" value={`${ov.late_infantile_pct}%`} color={ACCENT3} />
        <KPI label="Juv/Adult %" value={`${ov.juvenile_adult_pct}%`} color={ACCENT4} />
        <KPI label="Congen Onset" value={`Day ${ov.mean_onset_seizure_days_congenital}`} color={ACCENT2} />
        <KPI label="LINCL Onset" value={`${ov.mean_onset_seizure_years_lincl}y`} color={ACCENT3} />
        <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Retinal NCL" value={`${ov.retinal_degeneration_pct}%`} color={ACCENT2} />
        <KPI label="GRODs on EM" value={`${ov.grods_em_pct}%`} color={ACCENT} />
        <KPI label="Microcephaly" value={`${ov.microcephaly_congenital_pct}%`} color={ACCENT2} />
        <KPI label="Congen Survival" value={`~${ov.mean_survival_months_congenital}mo`} color={ACCENT2} />
        <KPI label="LINCL Survival" value={`~${ov.mean_survival_years_lincl}y`} color={ACCENT3} />
      </div>

      <SectionCard title="Gene & Mechanism">
        <p className="small mb-1"><strong>Gene:</strong> {ov.gene}</p>
        <p className="small mb-1"><strong>Protein:</strong> {ov.protein}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {ov.inheritance}</p>
        <p className="small mb-1"><strong>OMIM:</strong> {ov.omim}</p>
        <p className="small mb-1"><strong>Disease (Three Phenotypes):</strong> {ov.disease}</p>
        <p className="small mb-1"><strong>Mechanism:</strong> {ov.mechanism}</p>
      </SectionCard>

      <SectionCard title="⚠ GRODs EM = CLN1 OR CLN10 — Test PPT1 DBS First, Then CTSD WES" borderColor={ACCENT3}>
        <p className="small mb-0">{ov.grods_em_critical}</p>
      </SectionCard>

      <SectionCard title="No Disease-Modifying Therapy (CTSD ERT Preclinical — CLN2 Cerliponase Precedent)" borderColor={ACCENT2}>
        <p className="small mb-0">{ov.no_disease_modifying_therapy}</p>
      </SectionCard>

      {ov.key_pharmacological_distinctions && (
        <SectionCard title="Key Pharmacological Distinctions (Congenital / Late-Infantile / Juvenile CLN10)" borderColor={ACCENT4}>
          {Object.entries(ov.key_pharmacological_distinctions).map(([k, v]) => (
            <div key={k} className="mb-3">
              <div className="fw-bold small" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}</div>
              <div className="small text-muted">{v}</div>
            </div>
          ))}
        </SectionCard>
      )}
    </div>
  );
}

// ── TAB 2: Patients & Etiology ────────────────────────────────────────────────
function PatientsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="6-Class Etiology (40-patient CLN10 Cohort — Congenital / Late-Infantile / Juvenile)">
        {(bk.etiologies || []).map((et, i) => (
          <div key={i} className="mb-4 p-3 border rounded">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="fw-bold small">{et.class}</span>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{et.pct}% (n={et.count})</span>
            </div>
            <PctBar label="" pct={et.pct} color={ACCENT} />
            <p className="small text-muted mb-1">{et.description}</p>
            <p className="small mb-1"><strong>Mechanism:</strong> {et.gene_mechanism}</p>
            <div className="small"><strong>Key variants:</strong> {(et.key_variants || []).join(' · ')}</div>
          </div>
        ))}
      </SectionCard>

      {bk.lifecycle_stages && (
        <SectionCard title="6 Lifecycle Stages (CLN10 Disease Trajectory — All Phenotypes)" borderColor={ACCENT4}>
          {bk.lifecycle_stages.map((st, i) => (
            <div key={i} className="mb-3">
              <div className="fw-bold small" style={{ color: ACCENT }}>{st.stage}</div>
              <p className="small text-muted mb-0">{st.description}</p>
            </div>
          ))}
        </SectionCard>
      )}
    </div>
  );
}

// ── TAB 3: Seizures & Triggers ────────────────────────────────────────────────
function SeizuresTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="5 Seizure Types (Congenital / Late-Infantile / Juvenile CLN10)">
        {(bk.seizure_types || []).map((sz, i) => (
          <div key={i} className="mb-4 p-3 border rounded">
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-bold small">{sz.type}</span>
              <span className="badge" style={{ backgroundColor: ACCENT2 }}>{sz.pct}%</span>
            </div>
            <PctBar label="" pct={sz.pct} color={ACCENT2} />
            <p className="small text-muted mb-1">{sz.description}</p>
            <p className="small mb-1"><strong>EEG:</strong> {sz.eeg}</p>
            <p className="small mb-1"><strong>Semiology:</strong> {sz.semiology}</p>
            <div className="alert alert-info py-1 mb-0" style={{ fontSize: 12 }}>
              <strong>Clinical tip:</strong> {sz.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="8 Seizure Triggers" borderColor={ACCENT3}>
        {(bk.triggers || []).map((tr, i) => (
          <div key={i} className="mb-3 p-2 border rounded">
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-bold small">{tr.trigger}</span>
              <span className="badge bg-warning text-dark">{tr.pct}%</span>
            </div>
            <PctBar label="" pct={tr.pct} color={ACCENT3} />
            <p className="small text-muted mb-1">{tr.description}</p>
            <p className="small mb-0"><strong>Management:</strong> {tr.management}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 4: Treatments ─────────────────────────────────────────────────────────
function TreatmentsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="8 Treatments (Congenital / Late-Infantile / Juvenile CLN10)">
        {(bk.treatments || []).map((tx, i) => (
          <div key={i} className="mb-4 p-3 border rounded">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold">{tx.drug}</span>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{tx.level}</span>
            </div>
            <p className="small mb-1"><strong>Dose:</strong> {tx.dose}</p>
            <p className="small mb-1"><strong>MOA:</strong> {tx.moa}</p>
            <p className="small mb-1"><strong>Efficacy:</strong> {tx.efficacy}</p>
            <p className="small mb-1"><strong>Monitoring:</strong> {tx.monitoring}</p>
            <div className="alert alert-primary py-1 mb-0" style={{ fontSize: 12 }}>
              <strong>CLN10 Note:</strong> {tx.cln10_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="7 Contraindications" borderColor={ACCENT2}>
        {(bk.contraindications || []).map((ci, i) => (
          <div key={i} className="mb-3 p-2 border rounded border-danger">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold small">{ci.drug}</span>
              <span className={`badge ${ci.risk_level.includes('ABSOLUTE') ? 'bg-danger' : 'bg-warning text-dark'}`}>
                {ci.risk_level}
              </span>
            </div>
            <p className="small text-muted mb-1">{ci.reason}</p>
            <p className="small mb-0"><strong>Alternative:</strong> {ci.alternative}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="14 Monitoring Items" borderColor={ACCENT4}>
        {(bk.monitoring || []).map((m, i) => (
          <div key={i} className="mb-2 p-2 border-bottom">
            <div className="fw-bold small" style={{ color: ACCENT }}>{m.item}</div>
            <div className="small text-muted">{m.detail}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 5: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ df }) {
  if (!df) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="15 Key Concepts">
        {(df.concepts || []).map((c, i) => (
          <div key={i} className="mb-3 p-2 border-bottom">
            <div className="fw-bold small" style={{ color: ACCENT }}>{c.concept}</div>
            <p className="small text-muted mb-1">{c.definition}</p>
            <div className="small text-secondary"><strong>Standards:</strong> {c.standard}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="12 Clinical Thresholds" borderColor={ACCENT3}>
        {(df.thresholds || []).map((t, i) => (
          <div key={i} className="mb-2 p-2 border-bottom">
            <div className="small fw-bold">{t.threshold}</div>
            <div className="small text-muted">{t.standard}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="12 Standards & Guidelines" borderColor={ACCENT4}>
        {(df.standards || []).map((s, i) => (
          <div key={i} className="mb-2 p-2 border-bottom">
            <div className="small fw-bold">{s.standard}</div>
            <div className="small text-muted">{s.detail}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="6 References" borderColor={ACCENT}>
        {(df.references || []).map((r, i) => (
          <div key={i} className="mb-2 p-2 border-bottom">
            <div className="small fw-bold">{r.ref}</div>
            <div className="small text-muted">{r.citation}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function CLN10Page() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/cln10/overview`).then(r => r.json()).then(setOv).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2 || tab === 3) {
      if (!bk) fetch(`${API}/api/cln10/breakdown`).then(r => r.json()).then(setBk).catch(e => setErr(String(e)));
    }
    if (tab === 4) {
      if (!df) fetch(`${API}/api/cln10/definitions`).then(r => r.json()).then(setDf).catch(e => setErr(String(e)));
    }
  }, [tab, bk, df]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div
          className="rounded-circle d-flex align-items-center justify-content-center text-white fw-bold"
          style={{ width: 52, height: 52, backgroundColor: ACCENT, fontSize: 22, flexShrink: 0 }}
        >
          🧬
        </div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
            CLN10 Epilepsy — Congenital NCL / Cathepsin D Deficiency (CTSD)
          </h4>
          <div className="small text-muted">
            CTSD (11p15.5) · Lysosomal Aspartic Endopeptidase · Congenital (Most Severe NCL — Fatal &lt;1y) ·
            GRODs EM (Identical to CLN1 — Test PPT1 DBS First Then CTSD WES) ·
            CTSD Activates PPT1 (Secondary CLN1 Reduction) · VGB Absolute CI ·
            Fosphenytoin Absolute CI (Neonatal SE Trap) · VPA Safe · 40-pt Cohort
          </div>
        </div>
      </div>

      {err && <div className="alert alert-danger">{err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT, borderBottomWidth: 2 } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <PatientsTab bk={bk} />}
      {tab === 2 && <SeizuresTab bk={bk} />}
      {tab === 3 && <TreatmentsTab bk={bk} />}
      {tab === 4 && <DefinitionsTab df={df} />}
    </div>
  );
}
