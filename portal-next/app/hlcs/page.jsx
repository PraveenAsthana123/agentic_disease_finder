'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#00695c';   // deep teal — biotin / HLCS ligase / four-carboxylase activation
const ACCENT2 = '#b71c1c';   // dark red — acute neonatal crisis / lethal if untreated
const ACCENT3 = '#e65100';   // deep orange — metabolic acidosis / organic acid accumulation
const ACCENT4 = '#1565c0';   // deep blue — biotin treatment / dramatic response
const ACCENT5 = '#6a1b9a';   // deep purple — biotinidase NORMAL (key diagnostic)
const ACCENT6 = '#1b5e20';   // dark green — KEY NEGATIVES / differential
const ACCENT7 = '#37474f';   // dark slate — variant data / gene card
const ACCENT8 = '#4a148c';   // deep violet — four simultaneous carboxylase blocks

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

function Section({ title, children, color = ACCENT }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: 11 }}>{text}</span>
  );
}

// ─── TAB 0: OVERVIEW ───────────────────────────────────────────────────────

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const d = data;
  const k = d.kpis || {};

  return (
    <div>
      <Alert
        text={`⚠️ HLCS (Multiple Carboxylase Deficiency — Neonatal): master biotin ligase blocks ALL FOUR carboxylases simultaneously (PC + PCC + MCC + ACC). Biotin 10–40 mg/day LEVEL A — start immediately on suspicion. Biotinidase activity NORMAL (vs LOW in BTD). Raw egg white (avidin): ABSOLUTE CI.`}
        variant="danger"
      />

      <Section title="Gene & Disease Overview" color={ACCENT7}>
        <div className="row g-2 mb-2">
          {[
            ['Gene', d.gene], ['Protein', d.protein],
            ['Locus', d.locus], ['Length', d.aa_length],
            ['Cofactor', d.cofactor], ['Inheritance', d.inheritance],
            ['OMIM Gene', d.omim_gene], ['OMIM Disease', d.omim_disease],
          ].map(([k2, v]) => (
            <div className="col-12 col-md-6" key={k2}>
              <span className="fw-semibold me-1" style={{ color: ACCENT7 }}>{k2}:</span>
              <span className="small">{v}</span>
            </div>
          ))}
        </div>
        <div className="mt-2 p-2 rounded" style={{ background: '#f5f5f5' }}>
          <strong>Mechanism:</strong> <span className="small">{d.mechanism}</span>
        </div>
        <div className="mt-2 p-2 rounded" style={{ background: '#e8f5e9' }}>
          <strong style={{ color: ACCENT5 }}>Pathognomonic Pattern:</strong>{' '}
          <span className="small">{d.pathognomonic_pattern}</span>
        </div>
        <div className="mt-2 p-2 rounded" style={{ background: '#e3f2fd' }}>
          <strong style={{ color: ACCENT4 }}>Key Distinguishing Feature:</strong>{' '}
          <span className="small">{d.key_distinguishing_feature}</span>
        </div>
      </Section>

      <Section title="Cohort KPIs — 40 HLCS Patients" color={ACCENT}>
        <div className="row">
          <KPI label="Cohort N" value={k.cohort_n} color={ACCENT} />
          <KPI label="Avg Lactate (mmol/L)" value={k.avg_lactate_mmol} color={ACCENT2} />
          <KPI label="Avg Ammonia (µmol/L)" value={k.avg_ammonia_umol} color={ACCENT3} />
          <KPI label="Skin Rash %" value={`${k.skin_rash_periorificial_pct}%`} color={ACCENT3} />
          <KPI label="Alopecia %" value={`${k.alopecia_pct}%`} color={ACCENT3} />
          <KPI label="Seizures %" value={`${k.seizures_pct}%`} color={ACCENT2} />
          <KPI label="Biotin Responsive %" value={`${k.biotin_responsive_pct}%`} color={ACCENT4} />
          <KPI label="NBS Detected %" value={`${k.nbs_detected_pct}%`} color={ACCENT6} />
          <KPI label="Neuro Sequelae %" value={`${k.neuro_sequelae_pct}%`} color={ACCENT2} />
          <KPI label="Carnitine Deficient %" value={`${k.carnitine_deficient_pct}%`} color={ACCENT8} />
          <KPI label="Delayed Dx %" value={`${k.delayed_diagnosis_pct}%`} color={ACCENT2} />
          <KPI label="Male %" value={`${k.male_pct}%`} color={ACCENT7} />
        </div>
      </Section>

      <Section title="Phenotype Distribution" color={ACCENT}>
        {(d.phenotype_distribution || []).map(p => (
          <PctBar key={p.label} label={p.label} pct={p.pct} color={p.color} />
        ))}
      </Section>

      <Section title="Four Biotin-Dependent Carboxylases — ALL Blocked in HLCS Deficiency" color={ACCENT8}>
        <Alert text="HLCS LOF → none of the four apocarboxylases can be biotinylated → all four are inactive simultaneously. Each block creates its own metabolic crisis and biomarker stream." variant="warning" />
        {(d.four_carboxylases_blocked || []).map((e, i) => (
          <div key={i} className="mb-3 p-2 rounded border">
            <div className="fw-bold" style={{ color: ACCENT8 }}>{i + 1}. {e.enzyme}</div>
            <div className="small mt-1"><strong>Reaction:</strong> <span className="font-monospace">{e.reaction}</span></div>
            <div className="small mt-1 text-danger"><strong>HLCS Block:</strong> {e.block_consequence}</div>
            <div className="small mt-1"><strong>Biomarker:</strong> <span style={{ color: ACCENT3 }}>{e.biomarker}</span></div>
          </div>
        ))}
      </Section>

      <Section title="HLCS vs BTD — Two Causes of Multiple Carboxylase Deficiency" color={ACCENT5}>
        {d.hlcs_vs_btd_comparison && (
          <div>
            <Alert text={`Shared: ${d.hlcs_vs_btd_comparison.shared}`} variant="info" />
            <div className="row">
              <div className="col-md-6">
                <div className="p-2 rounded mb-2" style={{ background: '#e8f5e9' }}>
                  <div className="fw-bold" style={{ color: ACCENT }}>HLCS Deficiency</div>
                  {Object.entries(d.hlcs_vs_btd_comparison.hlcs || {}).map(([k2, v]) => (
                    <div key={k2} className="small"><strong>{k2.replace(/_/g,' ')}:</strong> {v}</div>
                  ))}
                </div>
              </div>
              <div className="col-md-6">
                <div className="p-2 rounded mb-2" style={{ background: '#fff3e0' }}>
                  <div className="fw-bold" style={{ color: ACCENT3 }}>BTD Deficiency</div>
                  {Object.entries(d.hlcs_vs_btd_comparison.btd || {}).map(([k2, v]) => (
                    <div key={k2} className="small"><strong>{k2.replace(/_/g,' ')}:</strong> {v}</div>
                  ))}
                </div>
              </div>
            </div>
            <Alert text={`Diagnostic Test: ${d.hlcs_vs_btd_comparison.diagnostic_test}`} variant="primary" />
          </div>
        )}
      </Section>

      <Section title="High-Risk Situations" color={ACCENT2}>
        {(d.high_risk_situations || []).map((r, i) => (
          <div key={i} className="mb-2 p-2 rounded border border-danger">
            <span className="badge me-2" style={{ backgroundColor: r.risk.includes('ABSOLUTE') ? '#b71c1c' : r.risk.includes('EXTREME') ? '#e65100' : '#f57f17' }}>
              {r.risk}
            </span>
            <strong>{r.situation}</strong>
            <div className="small text-muted mt-1">{r.detail}</div>
          </div>
        ))}
      </Section>
    </div>
  );
}

// ─── TAB 1: PATIENTS & PHENOTYPE ────────────────────────────────────────────

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const bm = data.biomarkers || [];
  const pts = data.patients_sample || [];

  return (
    <div>
      <Section title="Biomarkers — Four-Carboxylase Block Pattern" color={ACCENT3}>
        <Alert text="HLCS deficiency creates FOUR simultaneous metabolic blocks. Recognition of the combined pattern is diagnostic: C5-OH (NBS) + 3-OH-isovalerate + methylcitrate + lactic acidosis + NORMAL biotinidase." variant="warning" />
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead className="table-light">
              <tr>
                <th>Biomarker</th><th>Normal</th><th>HLCS Range</th><th>Significance</th>
              </tr>
            </thead>
            <tbody>
              {bm.map((b, i) => (
                <tr key={i}>
                  <td className="fw-semibold small">{b.name}</td>
                  <td className="small text-muted">{b.normal}</td>
                  <td className="small" style={{ color: b.hlcs_range?.includes('NORMAL') ? ACCENT6 : ACCENT3 }}>
                    {b.hlcs_range}
                  </td>
                  <td className="small text-muted">{b.significance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Key Variants" color={ACCENT7}>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead className="table-light">
              <tr><th>Variant</th><th>Molecular Effect</th><th>Phenotype</th></tr>
            </thead>
            <tbody>
              {(data.key_variants || []).map((v, i) => (
                <tr key={i}>
                  <td><span className="badge" style={{ backgroundColor: ACCENT7, fontFamily: 'monospace' }}>{v.variant}</span></td>
                  <td className="small">{v.effect}</td>
                  <td className="small" style={{ color: ACCENT3 }}>{v.phenotype}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Patient Cohort Sample (first 10 of 40)" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Sex</th><th>Phenotype</th><th>Onset (mo)</th>
                <th>Lactate</th><th>NH₃</th><th>C5-OH NBS</th>
                <th>Rash</th><th>Alopecia</th><th>Seizures</th>
                <th>Biotin (mg)</th><th>Responsive</th><th>NBS Det.</th><th>Neuro Seq.</th><th>Variant</th>
              </tr>
            </thead>
            <tbody>
              {pts.map(p => (
                <tr key={p.id}>
                  <td className="small">{p.id}</td>
                  <td className="small">{p.sex}</td>
                  <td className="small">{p.phenotype}</td>
                  <td className="small text-center">{p.onset_age_months}</td>
                  <td className="small" style={{ color: p.lactate_mmol > 4 ? ACCENT2 : 'inherit' }}>{p.lactate_mmol}</td>
                  <td className="small" style={{ color: p.ammonia_umol > 80 ? ACCENT3 : 'inherit' }}>{p.ammonia_umol}</td>
                  <td className="small" style={{ color: ACCENT3 }}>{(p.c5oh_acylcarnitine_nmol/1000).toFixed(1)} µM</td>
                  <td className="small text-center">{p.skin_rash_periorificial ? '✓' : '–'}</td>
                  <td className="small text-center">{p.alopecia ? '✓' : '–'}</td>
                  <td className="small text-center" style={{ color: p.seizures_present ? ACCENT2 : 'inherit' }}>{p.seizures_present ? '✓' : '–'}</td>
                  <td className="small text-center">{p.biotin_dose_mg}</td>
                  <td className="small text-center" style={{ color: p.biotin_responsive ? ACCENT6 : ACCENT2 }}>{p.biotin_responsive ? '✓' : '✗'}</td>
                  <td className="small text-center" style={{ color: p.nbs_detected ? ACCENT6 : ACCENT2 }}>{p.nbs_detected ? '✓' : '✗'}</td>
                  <td className="small text-center" style={{ color: p.neuro_sequelae ? ACCENT2 : ACCENT6 }}>{p.neuro_sequelae ? '✓' : '–'}</td>
                  <td className="small" style={{ fontFamily: 'monospace', fontSize: 10 }}>{p.variant}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

// ─── TAB 2: SEIZURES & TRIGGERS ─────────────────────────────────────────────

function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;

  return (
    <div>
      <Section title="Seizure Types in HLCS Deficiency" color={ACCENT2}>
        <Alert text="Seizures arise from metabolic acidosis + hypoglycemia + hyperammonemia + energy failure. Biotin corrects all mechanisms — seizures typically cease within 48–72 hours of biotin initiation." variant="warning" />
        {(data.seizure_types || []).map((s, i) => (
          <PctBar key={i} label={s.type} pct={s.pct} color={i < 2 ? ACCENT2 : i < 5 ? ACCENT3 : ACCENT7} />
        ))}
      </Section>

      <Section title="Metabolic Crisis Triggers" color={ACCENT3}>
        {(data.trigger_types || []).map((t, i) => (
          <PctBar key={i} label={t.trigger} pct={t.pct} color={i === 0 ? ACCENT2 : i < 3 ? ACCENT3 : ACCENT4} />
        ))}
      </Section>

      <Section title="High-Risk Drugs / Situations" color={ACCENT2}>
        {(data.high_risk_drugs || []).map((d2, i) => (
          <div key={i} className="mb-2 p-2 rounded border border-danger">
            <span className="badge me-2" style={{
              backgroundColor: d2.risk.includes('ABSOLUTE') ? '#b71c1c' : d2.risk.includes('EXTREME') ? '#e65100' : d2.risk.includes('CAUTION') ? '#f57f17' : '#9e9e9e'
            }}>
              {d2.risk}
            </span>
            <strong>{d2.drug}</strong>
            <div className="small text-muted mt-1">{d2.mechanism}</div>
          </div>
        ))}
      </Section>
    </div>
  );
}

// ─── TAB 3: TREATMENTS ──────────────────────────────────────────────────────

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;

  return (
    <div>
      <Alert text="Biotin 10–40 mg/day PO is LEVEL A — start immediately on clinical suspicion. Corrects ALL FOUR carboxylase deficiencies simultaneously. Response: metabolic acidosis 24–48 h; seizures 48–72 h; skin rash 1–2 weeks; alopecia 2–8 weeks." variant="info" />
      <Section title="Treatment Evidence" color={ACCENT4}>
        {(data.treatments || []).map((t, i) => (
          <div key={i} className="mb-3 p-3 rounded border">
            <div className="d-flex align-items-center mb-1">
              <span className="badge me-2" style={{ backgroundColor: t.color || ACCENT4 }}>Level {t.level}</span>
              <span className="fw-semibold">{t.drug}</span>
              <span className="ms-auto badge" style={{ backgroundColor: ACCENT6 }}>
                Response {t.response_pct}%
              </span>
            </div>
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </Section>
    </div>
  );
}

// ─── TAB 4: DEFINITIONS ─────────────────────────────────────────────────────

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const gc = data.gene_card || {};
  const kc = data.key_concepts || [];
  const dt = data.diagnostic_thresholds || {};
  const dd = data.differential_diagnosis || [];

  return (
    <div>
      <Section title="Gene Card" color={ACCENT7}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <tbody>
              {Object.entries(gc).map(([k2, v]) => (
                <tr key={k2}>
                  <th className="small" style={{ width: '28%', color: ACCENT7 }}>{k2}</th>
                  <td className="small">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Key Concepts" color={ACCENT}>
        {kc.map((c, i) => (
          <div key={i} className="mb-3">
            <div className="fw-semibold" style={{ color: ACCENT }}>{c.term}</div>
            <div className="small text-muted mt-1">{c.definition}</div>
          </div>
        ))}
      </Section>

      <Section title="Diagnostic Thresholds" color={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <tbody>
              {Object.entries(dt).map(([k2, v]) => (
                <tr key={k2}>
                  <th className="small" style={{ width: '35%', color: ACCENT3 }}>{k2.replace(/_/g, ' ')}</th>
                  <td className="small">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Differential Diagnosis" color={ACCENT6}>
        {dd.map((d2, i) => (
          <div key={i} className="mb-3 p-2 rounded border">
            <div className="fw-semibold" style={{ color: ACCENT6 }}>{d2.disease}</div>
            <div className="small text-muted mt-1">{d2.distinguishing_features}</div>
          </div>
        ))}
      </Section>
    </div>
  );
}

// ─── ROOT COMPONENT ─────────────────────────────────────────────────────────

export default function HLCSPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [ov, br, df] = await Promise.all([
          fetch(`${API}/api/hlcs/overview`).then(r => r.json()),
          fetch(`${API}/api/hlcs/breakdown`).then(r => r.json()),
          fetch(`${API}/api/hlcs/definitions`).then(r => r.json()),
        ]);
        setOverview(ov);
        setBreakdown(br);
        setDefinitions(df);
      } catch (e) {
        setError(e.message);
      }
    };
    load();
  }, []);

  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold" style={{ color: ACCENT }}>
          🧬 HLCS Epilepsy Dashboard
        </h4>
        <p className="text-muted small mb-1">
          Holocarboxylase Synthetase Deficiency — Multiple Carboxylase Deficiency (Neonatal) |
          OMIM <em>*609018 / #253270</em> | 21q22.13 | AR | 40-patient cohort
        </p>
        <div>
          <Badge text="Master Biotin Ligase" color={ACCENT} />
          <Badge text="4 Carboxylases Blocked (PC+PCC+MCC+ACC)" color={ACCENT8} />
          <Badge text="Biotin Level A — URGENT" color={ACCENT4} />
          <Badge text="Biotinidase NORMAL" color={ACCENT5} />
          <Badge text="Raw Egg White ABSOLUTE CI" color={ACCENT2} />
          <Badge text="NBS: C5-OH Elevated" color={ACCENT3} />
          <Badge text="AR / 21q22.13" color={ACCENT7} />
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
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
