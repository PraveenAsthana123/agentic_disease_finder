'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#1a237e';   // deep indigo — membrane arm integral subunit, 4-TM helix
const LIGHT = '#e8eaf6';

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

function Bar({ label, value, color = COLOR }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ variant, text }) {
  const bg     = variant === 'danger'  ? '#ffebee' : variant === 'warning' ? '#fff8e1'
               : variant === 'success' ? '#e8f5e9' : LIGHT;
  const border = variant === 'danger'  ? '#c62828' : variant === 'warning' ? '#f57f17'
               : variant === 'success' ? '#2e7d32' : COLOR;
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
        {title && <h6 className="fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>}
        {children}
      </div>
    </div>
  );
}

// ── Tab: Overview ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ff = data.feature_frequencies_pct || {};
  const bf = data.biochemical_fingerprint || {};
  const p  = data.protein || {};

  return (
    <>
      <div className="row g-3 mb-4">
        <KPI label="Gene"           value={data.gene}         color={COLOR} />
        <KPI label="Also known as"  value="B14.7 / 4-TM"      color={COLOR} />
        <KPI label="OMIM Gene"      value={`*${data.omim_gene}`}  color={COLOR} />
        <KPI label="Chromosome"     value={data.chromosome}   color={COLOR} />
        <KPI label="Inheritance"    value={data.inheritance}  color={COLOR} />
        <KPI label="Protein (mature)" value={`${p.size_kda} kDa`} color={COLOR} />
      </div>

      <SectionCard title="🧬 Gene & Protein">
        <p className="small mb-1"><strong>Full name:</strong> {data.gene_full_name}</p>
        <p className="small mb-1"><strong>Also known as:</strong> {data.also_known_as}</p>
        <p className="small mb-1"><strong>Fold / domain:</strong> {p.fold}</p>
        <p className="small mb-1"><strong>Module:</strong> {p.module}</p>
        <p className="small mb-2"><strong>Function:</strong> {p.function}</p>
        <Alert variant="info" text={data.key_pathway_note} />
      </SectionCard>

      <SectionCard title="🔬 Biochemical Fingerprint">
        {Object.entries(bf).map(([k, v]) => (
          <p key={k} className="small mb-1">
            <strong>{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}:</strong> {v}
          </p>
        ))}
        <Alert variant="success" text={`NDUFA11 is ISOLATED Complex I deficiency: CI ${data.cohort?.ci_activity_mean_pct ?? '~12'}% mean (range ${data.cohort?.ci_activity_range_pct ?? '5–20%'}), CII NORMAL, CIV NORMAL. Same biochemical fingerprint as other CI-Leigh nuclear subunit mutations; clinical distinguisher is the 4-TM helix membrane architecture and membrane arm location.`} />
      </SectionCard>

      <SectionCard title="📊 Cardinal Features (40-patient cohort)">
        <Bar label={`Psychomotor regression ${ff.psychomotor_regression}% (CARDINAL)`} value={ff.psychomotor_regression} />
        <Bar label={`Leigh MRI bilateral putamen/brainstem ${ff.leigh_mri}%`}           value={ff.leigh_mri} />
        <Bar label={`Hypotonia ${ff.hypotonia}%`}                                       value={ff.hypotonia} />
        <Bar label={`Lactic acidosis ${ff.lactic_acidosis}%`}                           value={ff.lactic_acidosis} />
        <Bar label={`Seizures ${ff.seizures}%`}                                         value={ff.seizures} color="#7986cb" />
        <Bar label={`Respiratory compromise ${ff.respiratory_compromise}%`}             value={ff.respiratory_compromise} color="#7986cb" />
        <Bar label={`Ataxia ${ff.ataxia}%`}                                             value={ff.ataxia} color="#9fa8da" />
        <Bar label={`Dystonia ${ff.dystonia}%`}                                         value={ff.dystonia} color="#9fa8da" />
      </SectionCard>

      <SectionCard title="🚫 KEY Differential Diagnosis Negatives">
        <Bar label={`NO peripheral neuropathy ${ff.peripheral_neuropathy}% (DDx NDUFS1 50%)`} value={ff.peripheral_neuropathy} color="#ef9a9a" />
        <Bar label={`NO olfactory bulb MRI ${ff.olfactory_bulb_lesions}% (DDx NDUFS4 58%)`}   value={ff.olfactory_bulb_lesions} color="#ef9a9a" />
        <Bar label={`NO leukodystrophy ${ff.leukodystrophy}% (DDx NDUFV1 45%)`}               value={ff.leukodystrophy} color="#ef9a9a" />
        <Bar label={`NO HCM ${ff.hcm}% (DDx NDUFV2 80%, SCO2 100%)`}                          value={ff.hcm} color="#ef9a9a" />
        <Bar label={`NO hepatopathy ${ff.hepatopathy}% (DDx POLG 80%, DGUOK 90%)`}            value={ff.hepatopathy} color="#ef9a9a" />
        <Alert variant="warning" text="NDUFA11 4-TM helix BN-PAGE: ABSENT CI (cleaner membrane arm disintegration). CONTRAST: peripheral arm sub-assembly intermediates in NDUFA2/NDUFS5/NDUFA12/NDUFA9. Similar to NDUFB3 but at PP-PD inter-module boundary (different location, opposite TM character)." />
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Features ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ff = data.feature_frequencies || {};
  const od = data.outcome_distribution || {};
  const md = data.mutation_distribution || {};
  const rd = data.region_distribution || {};
  const sd = data.sex_distribution || {};
  const hist = data.ci_activity_histogram || {};

  return (
    <>
      <SectionCard title={`👥 Cohort Overview (n=${data.n}, seed-635)`}>
        <div className="row g-2">
          <div className="col-6 col-md-3">
            <div className="border rounded p-2 text-center small">
              <div className="fw-bold" style={{ color: COLOR }}>M: {sd.M} / F: {sd.F}</div>
              <div className="text-muted">Sex</div>
            </div>
          </div>
          {Object.entries(od).map(([k, v]) => (
            <div key={k} className="col-6 col-md-3">
              <div className="border rounded p-2 text-center small">
                <div className="fw-bold" style={{ color: COLOR }}>{v}</div>
                <div className="text-muted">{k}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🔬 CI Activity Distribution">
        {(hist.bins || []).map((bin, i) => (
          <Bar key={bin} label={`CI ${bin}: ${hist.counts?.[i] ?? 0} patients`}
               value={Math.round((hist.counts?.[i] ?? 0) / data.n * 100)} />
        ))}
      </SectionCard>

      <SectionCard title="📈 Feature Frequencies">
        {Object.entries(ff).sort((a, b) => b[1].pct - a[1].pct).map(([k, v]) => (
          <Bar key={k} label={`${k.replace(/_/g, ' ')} (${v.count}/${data.n})`} value={v.pct} />
        ))}
      </SectionCard>

      <SectionCard title="🧬 Mutation Distribution">
        {Object.entries(md).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
          <Bar key={k} label={`${k} (n=${v})`} value={Math.round(v / data.n * 100)} />
        ))}
      </SectionCard>

      <SectionCard title="🌍 Region Distribution">
        {Object.entries(rd).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
          <Bar key={k} label={`${k} (n=${v})`} value={Math.round(v / data.n * 100)} color="#5c6bc0" />
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments & DDx ──────────────────────────────────────────────────────
function TreatmentsTab({ overview }) {
  if (!overview) return <p className="text-muted">Loading…</p>;
  const abs  = overview.absolute_contraindications || [];
  const cont = overview.contraindicated || [];
  const pref = overview.preferred_treatments || [];
  const ddx  = overview.key_ddx || [];

  return (
    <>
      <SectionCard title="🚨 Absolute Contraindications" borderColor="#c62828">
        {abs.map((t, i) => <Alert key={i} variant="danger"   text={t} />)}
      </SectionCard>
      <SectionCard title="⛔ Contraindicated" borderColor="#e65100">
        {cont.map((t, i) => <Alert key={i} variant="warning" text={t} />)}
      </SectionCard>
      <SectionCard title="✅ Preferred / Level C Treatments" borderColor="#2e7d32">
        {pref.map((t, i) => <Alert key={i} variant="success" text={t} />)}
      </SectionCard>
      <SectionCard title="🔍 Key DDx (NDUFA11 is NEGATIVE for all below)">
        {ddx.map((d, i) => (
          <div key={i} className="mb-3 p-2 rounded border small">
            <div className="fw-bold" style={{ color: COLOR }}>{d.feature}</div>
            <div className="text-muted">{d.significance}</div>
            {d.target_freq_pct > 0 && (
              <Bar label={`${d.target_gene}: ${d.target_freq_pct}%`} value={d.target_freq_pct} color="#ef9a9a" />
            )}
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  const sections = [
    { key: 'pharmacology',      title: '💊 Pharmacology' },
    { key: 'gene_concepts',     title: '🧬 Gene Concepts' },
    { key: 'disease_concepts',  title: '🏥 Disease Concepts' },
    { key: 'prescribing_safety', title: '📋 Prescribing Safety' },
  ];

  return (
    <>
      {sections.map(({ key, title }) => (
        <SectionCard key={key} title={title}>
          {(data[key] || []).map((item, i) => (
            <div key={i} className="mb-3 p-2 border rounded small">
              <div className="fw-bold" style={{ color: COLOR }}>{item.term}</div>
              <div className="text-muted small mb-1">[{item.category}]</div>
              <div style={{ whiteSpace: 'pre-line' }}>{item.detail}</div>
            </div>
          ))}
        </SectionCard>
      ))}
    </>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function NDUFA11Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ndufa11/overview`).then(r => r.json()),
      fetch(`${API}/api/ndufa11/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ndufa11/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <div className="alert alert-danger m-4">{error}</div>;

  return (
    <div className="container-fluid py-4">
      <div className="mb-3">
        <h4 className="fw-bold" style={{ color: COLOR }}>
          🧬 NDUFA11 — Leigh Syndrome Isolated Complex I Deficiency
        </h4>
        <p className="text-muted small mb-0">
          B14.7 Subunit · PP-Module/PD-Module Boundary · 4-TM Helix Integral Membrane Scaffold ·
          19q13.33 · OMIM *612638 / #256000 · AR Biallelic · seed-635
        </p>
      </div>

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab   data={overview}     />}
      {tab === 1 && <PatientsTab   data={breakdown}    />}
      {tab === 2 && <TreatmentsTab overview={overview} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
