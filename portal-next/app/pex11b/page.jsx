'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1b5e20';   // deep green — peroxisomal disease / fission tier
const ACCENT2 = '#b71c1c';   // dark red — HIGH RISK CI / VPA / VGB retinopathy
const ACCENT3 = '#e65100';   // deep orange — CAUTION / fasting caution / conditional CI
const ACCENT4 = '#1565c0';   // deep blue — treatments / LEV / ACTH
const ACCENT5 = '#4a148c';   // deep purple — PEX11B fission mechanism / DRP1 / tier hierarchy
const ACCENT6 = '#00695c';   // teal — IF signature / photosensitivity / fission tier 4

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
  const numPct = typeof pct === 'string' ? parseInt(pct) : pct;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${numPct}%`, backgroundColor: color }} />
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
    <div className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span className="badge me-1" style={{ backgroundColor: color, color: '#fff', fontSize: 11 }}>{text}</span>
  );
}

function CICard({ drug, level, reason, alternative }) {
  const color = level?.includes('NOT INDICATED') || level?.includes('NOT AVAILABLE')
    ? '#546e7a'
    : level?.includes('HIGH RISK') || level?.includes('ABSOLUTE')
    ? ACCENT2 : ACCENT3;
  return (
    <div className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold small">{drug}</span>
          <Badge text={level?.split('(')[0]?.trim().split(' ').slice(0, 4).join(' ')} color={color} />
        </div>
        <p className="small text-danger mb-1">{reason}</p>
        {alternative && <p className="small text-muted mb-0"><strong>Alternative:</strong> {alternative}</p>}
      </div>
    </div>
  );
}

function TreatmentCard({ drug, class: cls, evidence, dose, moa, monitoring, ci }) {
  return (
    <div className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold small">{drug}</span>
          <Badge text={cls?.split('—')[0]?.trim()} color={ACCENT4} />
        </div>
        <p className="small mb-1"><strong>Evidence:</strong> {evidence}</p>
        {dose && <p className="small mb-1"><strong>Dose:</strong> {dose}</p>}
        {moa && <p className="small mb-1"><strong>MOA:</strong> {moa}</p>}
        {monitoring && <p className="small mb-1 text-muted"><strong>Monitor:</strong> {monitoring}</p>}
        {ci && <p className="small mb-0 text-warning"><strong>CI:</strong> {ci}</p>}
      </div>
    </div>
  );
}

// ── Overview Tab ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading overview…</div>;
  const d = data;
  return (
    <div>
      <Alert
        text="⚠ VPA — HIGH RISK (DUAL CI in PEX11B — not triple as in ZSD): (1) hepatotoxicity + (2) carnitine depletion. The third ZSD mechanism (peroxisomal beta-oxidation inhibition) is less severe here (import intact, VLCFA-oxidising enzymes ARE in peroxisomes). Still HIGH RISK. POLG1 MANDATORY (CPIC Grade A). LEV first-line. NEVER VPA without POLG1 exclusion."
        variant="warning"
      />
      <Alert
        text="⚠ VGB (Vigabatrin) — RELATIVE CI (CONDITIONAL): Retinopathy present in ~62% of PEX11B cohort. If RP confirmed: VGB ABSOLUTE CI (additive irreversible VF loss). If RP not yet assessed: treat as relative CI. ACTH Level A preferred for IS — always safer choice in PEX11B."
        variant="warning"
      />
      <Alert
        text="ℹ Fasting / NPO — CAUTION (NOT extreme hazard as in ZSD): PEX11B retains per-organelle alpha-oxidation (phytanic borderline not severe). IV dextrose for fasts >4h is prudent but NOT the emergency mandate of ZSD (where ALL alpha-oxidation fails → acute phytanic/pristanic neurotoxic surge)."
        variant="info"
      />
      <Alert
        text={`🔬 PEX11B = FISSION TIER — 4th mechanistic tier of peroxisomal epilepsy. PEX11B (247 aa, 1q22) drives DRP1-mediated peroxisome fission. LOF → elongated/tubular peroxisomes. CRITICAL IF DISTINCTION: catalase PEROXISOMAL (import IS intact) — the OPPOSITE of all ZSD where catalase is CYTOPLASMIC. Plasmalogens NORMAL or borderline (not severely low as in ZSD). VLCFA borderline (not frank). Photosensitivity HALLMARK (71%). SNHL 78%. ~10–15 cases worldwide 2026. OMIM *${d.omim_gene} / #${d.omim_disease}.`}
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="Severe Neonatal %" value={`${d.severe_neonatal_pct}%`} color={ACCENT2} />
        <KPI label="Moderate Infantile IS %" value={`${d.moderate_infantile_pct}%`} color={ACCENT3} />
        <KPI label="Mild Childhood %" value={`${d.mild_childhood_pct}%`} color={ACCENT} />
        <KPI label="Drug Resistant %" value={`${d.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="Photosensitivity %" value={`${d.photosensitivity_pct}%`} color={ACCENT6} />
        <KPI label="SNHL %" value={`${d.hearing_loss_pct}%`} color={ACCENT3} />
        <KPI label="Retinopathy %" value={`${d.retinopathy_pct}%`} color={ACCENT2} />
        <KPI label="VLCFA borderline %" value={`${d.vlcfa_elevated_pct}%`} color={ACCENT3} />
        <KPI label="OMIM Gene" value={`*${d.omim_gene}`} color={ACCENT5} />
        <KPI label="Locus" value={d.locus} color={ACCENT} />
      </div>

      <SectionCard title="Disease Summary — PEX11B / Peroxisome Fission Defect (PBD9B)" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-1"><strong>Allele Spectrum:</strong> {d.common_variant}</p>
        <p className="small mb-0">{d.disease_mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Profile" borderColor={ACCENT4}>
            <PctBar label="Seizures (overall cohort)" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Drug-resistant seizures" pct={d.drug_resistance_pct} color={ACCENT2} />
            <PctBar label="Photosensitivity — HALLMARK (71%)" pct={d.photosensitivity_pct} color={ACCENT6} />
            <PctBar label="Sensorineural hearing loss (SNHL)" pct={d.hearing_loss_pct} color={ACCENT3} />
            <PctBar label="Retinopathy (RP)" pct={d.retinopathy_pct} color={ACCENT2} />
            <PctBar label="Severe neonatal class (null/null)" pct={d.severe_neonatal_pct} color={ACCENT2} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Biochemical Profile — PARTIAL (Fission Tier vs ZSD)" borderColor={ACCENT4}>
            <PctBar label="VLCFA borderline (not frank elevation)" pct={d.vlcfa_elevated_pct} color={ACCENT3} />
            <PctBar label="Plasmalogens LOW (18% — near-normal vs ZSD 100%)" pct={d.plasmalogen_low_pct} color={ACCENT6} />
            <PctBar label="Catalase PEROXISOMAL (100% — import intact)" pct={d.catalase_peroxisomal_pct} color={ACCENT} />
            <PctBar label="Moderate infantile IS (modal class)" pct={d.moderate_infantile_pct} color={ACCENT3} />
            <div className="mt-2 small text-muted">
              <strong>NBS:</strong> {d.nbs_positive_rate}
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="PEX11B — FISSION TIER — 4-Tier IF Hierarchy for Peroxisomal Epilepsy" borderColor={ACCENT5}>
        <div className="row small">
          <div className="col-md-3">
            <p className="fw-bold mb-1" style={{ color: '#546e7a' }}>Tier 1 — Membrane-Biogenesis (PEX3/PEX16/PEX19 LOF)</p>
            <ul className="mb-0">
              <li>PEX16: recruits PEX3 from ER vesicles</li>
              <li>PEX3: docking receptor for PEX19</li>
              <li>PEX19: inserts Class I PMPs (PEX14, PMP70)</li>
              <li>LOF → no membrane (ghost peroxisomes)</li>
              <li><strong>IF: PMP70− + PEX14− + catalase cytoplasmic</strong></li>
            </ul>
          </div>
          <div className="col-md-3">
            <p className="fw-bold mb-1" style={{ color: ACCENT3 }}>Tier 2 / 3 — Importomer + Receptor (PEX13/PEX14/PEX5 LOF)</p>
            <ul className="mb-0">
              <li>PEX14: central scaffold; PEX13: SH3 docking</li>
              <li>PEX5: cytosolic PTS1 receptor</li>
              <li>LOF → matrix import fails globally</li>
              <li>Membrane PRESENT; import ABSENT</li>
              <li><strong>IF: PMP70+ + PEX14± + catalase CYTOPLASMIC</strong></li>
            </ul>
          </div>
          <div className="col-md-3">
            <p className="fw-bold mb-1" style={{ color: ACCENT6 }}>TIER 4 — FISSION TIER (PEX11B LOF — THIS GENE)</p>
            <ul className="mb-0">
              <li>PEX11B: drives peroxisome fission via DRP1</li>
              <li>LOF → elongated/tubular peroxisomes</li>
              <li>Membrane PRESENT; importomer PRESENT</li>
              <li>Matrix import IS INTACT per organelle</li>
              <li><strong>IF: PMP70+ + PEX14+ + catalase PEROXISOMAL</strong></li>
              <li className="text-success fw-bold">catalase PEROXISOMAL = unique Tier 4 marker</li>
            </ul>
          </div>
          <div className="col-md-3">
            <p className="fw-bold mb-1" style={{ color: ACCENT5 }}>CRITICAL Biochemical Distinction</p>
            <ul className="mb-0">
              <li>ZSD (all tiers 1–3): plasmalogens SEVERELY LOW</li>
              <li>PEX11B Tier 4: plasmalogens NORMAL/borderline</li>
              <li>ZSD: VLCFA frankly elevated</li>
              <li>PEX11B: VLCFA borderline only</li>
              <li>ABCD1 (X-ALD): normal plasmalogens, NO epilepsy at onset</li>
              <li><strong>Plasmalogens = single discriminating test</strong></li>
            </ul>
          </div>
        </div>
        <div className="mt-2 small">
          <Badge text="p.Ile195Thr (partial fission)" color={ACCENT5} /> C-terminal DRP1-docking domain; ~40% residual fission → moderate phenotype.
          <Badge text="Photosensitivity HALLMARK" color={ACCENT6} className="ms-2" /> Highest photosensitivity rate among peroxisomal epilepsies (71%); photo-EEG mandatory.
          <Badge text="Catalase peroxisomal" color={ACCENT} className="ms-2" /> Confirms import intact — single most important IF differentiator from ZSD.
        </div>
      </SectionCard>

      <SectionCard title="Key Pharmacological Distinctions — PEX11B vs ZSD" borderColor={ACCENT2}>
        <Alert text="VPA HIGH RISK (DUAL CI — hepatotoxicity + carnitine depletion; less severe than ZSD triple CI but still HIGH). POLG1 mandatory." variant="warning" />
        <Alert text="VGB CONDITIONAL CI — only absolute if retinopathy CONFIRMED (62%); ACTH Level A preferred for IS in all PEX11B." variant="warning" />
        <Alert text="PHT/CBZ/OXC RELATIVE CI (mild DHA depletion risk; less severe than ZSD since DHA synthesis near-normal). Replace with LEV on diagnosis." variant="warning" />
        <Alert text="Fasting CAUTION (not extreme hazard). DHA Level C/experimental (not Level B as in ZSD — DHA near-normal). No ERT / No HSCT / Lorenzo's Oil NOT indicated (different mechanism)." variant="info" />
      </SectionCard>

      <SectionCard title="Key Concepts" borderColor={ACCENT5}>
        <ul className="list-unstyled mb-0">
          {d.key_concepts?.map((c, i) => (
            <li key={i} className="mb-1 small">
              <span className="me-1" style={{ color: ACCENT5 }}>▸</span>{c}
            </li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="Reference Standards" borderColor={ACCENT4}>
        <ul className="mb-0">
          {d.standards?.map((s, i) => <li key={i} className="small">{s}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Patients & Etiology Tab ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { etiologies = [], patients = [] } = data;
  const ETIOL_COLORS = {
    'Severe': ACCENT2,
    'Moderate': ACCENT3,
    'Mild': ACCENT,
    'Atypical': ACCENT6,
  };
  const getColor = (name) => {
    for (const [key, color] of Object.entries(ETIOL_COLORS)) {
      if (name?.includes(key)) return color;
    }
    return ACCENT;
  };
  return (
    <div>
      <Alert
        text="ℹ PEX11B SPECTRUM: null/null → Severe neonatal (35%) · null/p.Ile195Thr → Moderate infantile IS (45%, MODAL) · mild/mild → Mild childhood (15%) · Atypical late-onset (5%). MODAL class = IS + hypsarrhythmia. Biochemically PARTIAL: VLCFA borderline, plasmalogens NORMAL/borderline (not severely low as in ZSD). Catalase PEROXISOMAL in all classes (import intact). Photosensitivity hallmark (71%). SNHL (78%)."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>PEX11B Phenotypic Classes — 4 Classes (40 Patients)</h6>
      {etiologies?.map((e, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${getColor(e.name)}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-2">
              <h6 className="fw-bold mb-0" style={{ color: getColor(e.name) }}>{e.name}</h6>
              <div>
                <Badge text={`${e.pct}%`} color={getColor(e.name)} />
                <Badge text={`n=${e.n}`} color="#555" />
                <Badge text={e.sex} color="#888" />
              </div>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: getColor(e.name) }} />
            </div>
            <div className="row small">
              <div className="col-md-6">
                <p className="mb-1"><strong>Onset:</strong> {e.onset_age}</p>
                <p className="mb-1"><strong>Seizure risk:</strong> {e.seizure_risk}</p>
                <p className="mb-1"><strong>EEG:</strong> {e.eeg}</p>
              </div>
              <div className="col-md-6">
                <p className="mb-1"><strong>MRI:</strong> {e.mri}</p>
                <div className="d-flex gap-2 flex-wrap mt-1">
                  {!e.hsct_eligible && <Badge text="HSCT not indicated" color="#888" />}
                  {!e.ert_available && <Badge text="No ERT (integral PMP)" color="#888" />}
                  {!e.dha_supplement && <Badge text="DHA: Level C only" color="#aaa" />}
                </div>
              </div>
            </div>
            <p className="small text-muted mb-0 mt-2">{e.variant_detail}</p>
          </div>
        </div>
      ))}

      <SectionCard title="Patient Table (40 patients)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead>
              <tr>
                <th>ID</th><th>Phenotype</th><th>Sex</th><th>Onset</th>
                <th>SNHL</th><th>Photo</th><th>RP</th><th>VLCFA</th><th>Plasmal.</th>
                <th>Drug Res.</th><th>AEDs</th>
              </tr>
            </thead>
            <tbody>
              {patients?.map((p, i) => (
                <tr key={i}>
                  <td><code>{p.id}</code></td>
                  <td><Badge text={p.phenotype?.split(' ')[0] || p.phenotype} color={getColor(p.phenotype)} /></td>
                  <td>{p.sex}</td>
                  <td className="small">{p.onset}</td>
                  <td>{p.snhl ? <span className="text-warning fw-bold">Yes</span> : <span className="text-muted">—</span>}</td>
                  <td>{p.photosensitive ? <span className="text-danger fw-bold">Yes</span> : <span className="text-muted">—</span>}</td>
                  <td>{p.retinopathy ? <span className="text-danger fw-bold">Yes</span> : <span className="text-muted">—</span>}</td>
                  <td><Badge text={p.vlcfa} color={p.vlcfa === 'Borderline' ? ACCENT3 : ACCENT} /></td>
                  <td><Badge text={p.plasmalogens} color={p.plasmalogens === 'Normal' ? ACCENT : ACCENT3} /></td>
                  <td>{p.drug_resistant ? <span className="text-danger fw-bold">Yes</span> : <span className="text-muted">No</span>}</td>
                  <td className="small">{p.current_drugs?.join(' + ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Seizures & Triggers Tab ───────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types = {}, triggers = [], monitoring = [], thresholds = [], lifecycle = [] } = data;
  const seizureList = Object.entries(seizure_types).map(([name, v]) => ({ name, ...v }));
  return (
    <>
      <SectionCard title="Seizure Types (PEX11B — Fission Tier: Neonatal Myoclonic / IS / Focal / GTCS)" borderColor={ACCENT2}>
        {seizureList.map((st, i) => (
          <div key={i} className="mb-3">
            <PctBar label={`${st.name} (n=${st.n})`} pct={st.pct} color={ACCENT2} />
            <div className="small text-muted">{st.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers — Photic Stimulation HALLMARK (71%)" borderColor={ACCENT3}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3">
            <PctBar label={t.trigger} pct={t.pct} color={t.trigger.includes('Photic') ? ACCENT6 : ACCENT3} />
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Protocol (12 Parameters)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Parameter</th><th>Frequency</th><th>Target</th><th>Action</th></tr></thead>
            <tbody>
              {monitoring.map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold small">{m.parameter}</td>
                  <td className="small"><Badge text={m.frequency} color={ACCENT4} /></td>
                  <td className="small">{m.target}</td>
                  <td className="small text-muted">{m.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Clinical Thresholds & Action Points" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold small">{t.parameter}</td>
                  <td><Badge text={t.value} color={ACCENT3} /></td>
                  <td className="small text-muted">{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Disease Lifecycle (6 Stages)" borderColor={ACCENT}>
        {lifecycle.map((l, i) => (
          <div key={i} className="mb-3 pb-3 border-bottom">
            <div className="fw-bold small" style={{ color: ACCENT }}>{l.stage}</div>
            <div className="small text-muted">{l.features}</div>
            <div className="small mt-1"><strong>Action:</strong> {l.action}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Treatments Tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <>
      <Alert
        text="ℹ LEV = first-line AED (all PEX11B phenotypic classes). ACTH Level A for infantile spasms (preferred over VGB given retinopathy in 62%). DHA Level C/experimental only if plasma DHA documented below range (unlike ZSD where DHA Level B universally). No ERT (integral peroxisomal membrane protein — no M6P route). No HSCT (non-inflammatory fission defect). Lorenzo's Oil NOT indicated (different mechanism from X-ALD). Gene therapy (AAV9-PEX11B) preclinical."
        variant="info"
      />
      <SectionCard title="Treatments (AEDs + Supportive)" borderColor={ACCENT4}>
        {data.treatments?.map((t, i) => (
          <TreatmentCard key={i} {...t} />
        ))}
      </SectionCard>
      <SectionCard title="Contraindications & Hazards" borderColor={ACCENT2}>
        {data.contraindications?.map((c, i) => (
          <CICard key={i} {...c} />
        ))}
      </SectionCard>
    </>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading definitions…</div>;
  const termEntries = data.terms ? Object.entries(data.terms) : [];
  return (
    <>
      <SectionCard title="Key Concepts (15)" borderColor={ACCENT5}>
        <ul className="list-unstyled mb-0">
          {data.key_concepts?.map((c, i) => (
            <li key={i} className="mb-2 small">
              <span className="me-1" style={{ color: ACCENT5 }}>▸</span>{c}
            </li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="Glossary — PEX11B / Fission Tier Terms (8 definitions)" borderColor={ACCENT6}>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th style={{ width: '28%' }}>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {termEntries.map(([term, def], i) => (
                <tr key={i}>
                  <td className="fw-bold small" style={{ color: ACCENT5 }}>{term}</td>
                  <td className="small">{def}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function PEX11BPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/pex11b/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3) {
      fetch(`${API}/api/pex11b/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/pex11b/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4">
        <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
          PEX11B Epilepsy — Peroxisome Fission Defect (PBD9B)
        </h2>
        <p className="text-muted small mb-2">
          PEX11B (1q22) · FISSION TIER — 4th mechanistic tier of peroxisomal epilepsy ·
          247 aa integral peroxisomal membrane protein · Drives DRP1/FIS1/MFF-mediated peroxisome fission ·
          LOF → elongated/tubular peroxisomes (cannot divide) → reduced organelle number ·
          CRITICAL IF: PMP70+ + PEX14+ + catalase PEROXISOMAL (import intact) — OPPOSITE of all ZSD where catalase is cytoplasmic ·
          Plasmalogens NORMAL or borderline (severely low in ZSD) · VLCFA borderline (frank in ZSD) ·
          NBS equivocal (not frank elevation as in ZSD) · Photosensitivity HALLMARK (71%) · SNHL 78% · Retinopathy 62% ·
          AR biallelic LOF · p.Ile195Thr (C-terminal DRP1-docking domain, partial fission, moderate phenotype) · p.Leu145Pro (TM domain, severe) ·
          ~10–15 cases worldwide 2026 · OMIM *603637 / #614862 ·
          Severe neonatal 35% · Moderate infantile IS 45% (modal) · Mild childhood 15% · Atypical 5% ·
          VPA HIGH RISK (dual CI) · VGB conditional CI (RP-dependent) · LEV first-line · ACTH Level A IS ·
          Fasting CAUTION (not extreme hazard) · HSCT NOT indicated · No ERT · Gene therapy preclinical · 40 patients
        </p>
        {err && <div className="alert alert-danger small">{err}</div>}
      </div>

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
