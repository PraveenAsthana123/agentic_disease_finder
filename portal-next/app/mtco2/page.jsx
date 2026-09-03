'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
// Deep copper-blue — CuA binuclear copper centre / electron acceptor from cytochrome c
const COLOR  = '#1a237e';   // deep indigo-blue (copper-electron theme)
const LIGHT  = '#e8eaf6';
const COLOR2 = '#283593';
const COLOR3 = '#b71c1c';   // danger / absolute CI
const COLOR4 = '#e65100';   // warning / contraindication
const COLOR5 = '#1b5e20';   // success / treatments / normal

function KPI({ label, value, color = COLOR }) {
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

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const k = data.kpis || {};
  const pheno = data.phenotype_distribution || [];
  const concepts = data.key_concepts || [];
  const refs = data.references || [];

  return (
    <div>
      {/* Gene header */}
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `5px solid ${COLOR}` }}>
        <h5 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 MT-CO2 — CuA Binuclear Copper Center / CIV-Leigh / Multisystem CIV Neuropathy / MELAS-CIV
        </h5>
        <p className="mb-1 small">
          <strong>OMIM Gene:</strong> *{data.omim_gene} &nbsp;|&nbsp;
          <strong>Alias:</strong> {data.alias} &nbsp;|&nbsp;
          <strong>Genome:</strong> {data.chromosome}
        </p>
        <p className="mb-1 small">
          <strong>Protein:</strong> {data.protein_size} &nbsp;|&nbsp;
          <strong>Inheritance:</strong> {data.inheritance} &nbsp;|&nbsp;
          <strong>Cohort:</strong> {data.cohort_n} patients (seed {data.seed})
        </p>
        <p className="mb-0 small fw-semibold" style={{ color: COLOR }}>
          🔬 MT-CO2 contains the CuA BINUCLEAR COPPER CENTER — the ONLY ETC site receiving electrons directly from
          cytochrome c. CuA→Heme a (MT-CO1)→Heme a3-CuB (MT-CO1)→O₂. 227 aa / 26 kDa / 2 TM helices = SMALLEST
          mtDNA CIV subunit. SCO1/SCO2 are CuA copper chaperones.
          <span style={{ color: '#7b1fa2' }}> SCO2 deficiency → 100% HCM (KEY DDx — MT-CO2 has NO HCM);
          SCO1 deficiency → hepatopathy (KEY DDx — MT-CO2 has NO hepatopathy).</span>
        </p>
      </div>

      {/* Critical warnings */}
      <Alert variant="danger"
        text="⚠️ PROPOFOL: ABSOLUTE CI — inhibits MT-CO1 heme a3-CuB downstream of CuA (MT-CO2) + PRIS. In MT-CO2 deficiency, CuA→heme a electron delivery already impaired; propofol compounds the CIV double-hit. Use SEVOFLURANE." />
      <Alert variant="danger"
        text="⚠️ LINEZOLID / CHLORAMPHENICOL: ABSOLUTE CI — blocks mt ribosome 23S rRNA → prevents MT-CO2 synthesis. All 3 mtDNA CIV subunits (CO1/CO2/CO3) require mt ribosome translation." />
      <Alert variant="danger"
        text="⚠️ METFORMIN + VPA + KD: ABSOLUTE CI — CI inhibition + CoA sequestration + β-oxidation exacerbating CuA bottleneck." />
      <Alert variant="warning"
        text="⚠️ SCO2 DDx: CIV-Leigh + HCM → suspect SCO2 (AR 22q13.33), NOT MT-CO2 (maternal). Echocardiography is MANDATORY at diagnosis. MT-CO2 → NO HCM." />
      <Alert variant="warning"
        text="⚠️ WES MISSES MT-CO2: Dedicated mtDNA sequencing in MUSCLE required. Blood underestimates heteroplasmy 15–30%. Large deletions (KSS) need long-read or Southern blot." />

      {/* KPI row */}
      <div className="row mb-3">
        <KPI label="Seizures (%)" value={`${k.seizures_pct}%`} color={COLOR} />
        <KPI label="Hypotonia (%)" value={`${k.hypotonia_pct}%`} color={COLOR2} />
        <KPI label="Lactic Acidosis" value={`${k.lactic_acidosis_pct}%`} color={COLOR3} />
        <KPI label="Leigh MRI (%)" value={`${k.leigh_mri_pct}%`} color={COLOR} />
        <KPI label="Neuropathy (%)" value={`${k.neuropathy_pct}%`} color={COLOR4} />
        <KPI label="Respiratory (%)" value={`${k.respiratory_pct}%`} color={COLOR3} />
        <KPI label="Mean CIV %" value={`${k.mean_civ_pct}%`} color={COLOR3} />
        <KPI label="Median Onset" value={`${k.median_onset_mo}mo`} color={COLOR2} />
        <KPI label="HCM" value="0% ✓" color={COLOR5} />
        <KPI label="Hepatopathy" value="0% ✓" color={COLOR5} />
      </div>

      {/* Phenotype distribution */}
      <SectionCard title="Phenotype Distribution (40-patient Cohort, seed 747)">
        {pheno.map((p, i) => (
          <Bar key={i} label={p.class} value={p.pct}
            color={p.class.includes('Leigh infantile') ? COLOR3
                 : p.class.includes('neuropathy') ? '#6a1b9a'
                 : p.class.includes('MELAS') ? COLOR4
                 : p.class.includes('KSS') ? '#37474f' : COLOR2} />
        ))}
        <p className="text-muted small mt-2 mb-0">
          CIV residual range: infantile Leigh 3–15% | neuropathy 5–18% | MELAS-CIV 8–22% | KSS 8–35% | mild myopathy 30–55%.
          All point-mutation phenotypes → ISOLATED CIV (CI/CII/CIII normal). Large deletion → combined CI+CIV.
        </p>
      </SectionCard>

      {/* Seizure types */}
      <SectionCard title="Seizure Types (MT-CO2 Cohort)">
        {(data.seizure_types || []).map((s, i) => (
          <Bar key={i} label={s.type} value={s.pct} color={COLOR} />
        ))}
      </SectionCard>

      {/* Triggers */}
      <SectionCard title="Metabolic Decompensation Triggers" borderColor={COLOR4}>
        {(data.triggers || []).map((t, i) => (
          <Bar key={i} label={t.trigger} value={t.pct} color={COLOR4} />
        ))}
      </SectionCard>

      {/* Key concepts */}
      <SectionCard title="Key Clinical Concepts" borderColor={COLOR2}>
        {concepts.map((c, i) => (
          <div key={i} className="mb-3 pb-3" style={{ borderBottom: i < concepts.length - 1 ? '1px solid #e0e0e0' : 'none' }}>
            <div className="fw-semibold small mb-1" style={{ color: COLOR }}>{c.concept}</div>
            <div className="text-muted small">{c.detail}</div>
          </div>
        ))}
      </SectionCard>

      {/* References */}
      <SectionCard title="Key References" borderColor={COLOR2}>
        <ul className="mb-0 small">
          {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Tab: Patients & Features ──────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const pts  = data.patients  || [];
  const vars = data.variants  || [];

  return (
    <div>
      {/* Variant detail cards */}
      <SectionCard title="Pathogenic Variants — MT-CO2 CuA Domain & TM Helices">
        {vars.map((v, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: LIGHT }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold small" style={{ color: COLOR }}>{v.variant}</span>
              <span className="badge" style={{ background: COLOR, fontSize: '0.7rem' }}>{v.freq_pct}%</span>
            </div>
            <div className="small text-muted mb-1">
              <strong>AA change:</strong> {v.amino_acid} &nbsp;|&nbsp;
              <strong>Impact:</strong> {v.structural_impact}
            </div>
            <div className="small mb-1">
              <strong>Modal phenotype:</strong>{' '}
              <span style={{ color: COLOR3 }}>{v.modal_phenotype}</span>
            </div>
            <div className="small text-muted">{v.detail}</div>
            <div className="small mt-1 text-muted">
              <strong>In cohort (n={data.cohort_n}):</strong> {v.n_in_cohort} patients
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Patient table */}
      <SectionCard title={`Patient Cohort (n=${data.cohort_n}, seed 747)`}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr style={{ background: COLOR, color: '#fff' }}>
                <th>ID</th><th>Phenotype (short)</th><th>Variant</th>
                <th>CIV%</th><th>Onset(mo)</th>
                <th>Sz</th><th>Hypo</th><th>LacAc</th>
                <th>LeighMRI</th><th>SLE</th><th>Neuropathy</th>
                <th>HCM</th><th>Hepato</th>
              </tr>
            </thead>
            <tbody>
              {pts.map((p, i) => (
                <tr key={i}>
                  <td><strong>{p.id}</strong></td>
                  <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {p.phenotype.split('(')[0].trim()}
                  </td>
                  <td className="text-muted">{p.variant}</td>
                  <td style={{ color: p.civ_pct < 15 ? COLOR3 : COLOR4 }}>{p.civ_pct}%</td>
                  <td>{p.onset_mo}</td>
                  <td>{p.seizure ? '✓' : '—'}</td>
                  <td>{p.hypotonia ? '✓' : '—'}</td>
                  <td>{p.lactic_ac ? '✓' : '—'}</td>
                  <td>{p.leigh_mri ? '✓' : '—'}</td>
                  <td>{p.sle_mri ? '✓' : '—'}</td>
                  <td>{p.neuropathy ? <span style={{ color: '#6a1b9a' }}>✓</span> : '—'}</td>
                  <td><span style={{ color: COLOR5 }}>✗</span></td>
                  <td><span style={{ color: COLOR5 }}>✗</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-muted small mt-1 mb-0">
          HCM = 0% (NO HCM — KEY DDx from SCO2 which has 100% HCM) |
          Hepatopathy = 0% (NO liver disease — KEY DDx from SCO1) |
          Peripheral neuropathy most prominent in m.8249G&gt;A CuA-loop variant.
        </p>
      </SectionCard>
    </div>
  );
}

// ── Tab: Treatments & DDx ─────────────────────────────────────────────────────
function TreatmentTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const tx    = data.treatments        || [];
  const ci    = data.contraindications || [];
  const mon   = data.monitoring        || [];

  const evidColor = (ev) =>
    ev.includes('Mandatory') ? COLOR3 : ev.includes('Level A') ? '#1b5e20'
    : ev.includes('Level B') ? COLOR : ev.includes('Level C') ? COLOR4 : '#607d8b';

  return (
    <div>
      {/* DDx box — most important in MT-CO2 */}
      <SectionCard title="Critical DDx — CIV-Leigh Gene Differentiation" borderColor="#7b1fa2">
        <div className="row">
          <div className="col-md-6">
            <Alert variant="danger" text="SCO2 (22q13.33 AR): CIV-Leigh + 100% HCM — if HCM present → SCO2, NOT MT-CO2. Echocardiography MANDATORY." />
            <Alert variant="danger" text="SCO1 (17p13.2 AR): CIV-Leigh + severe hepatopathy (LFTs >10× ULN) — hepatopathy → SCO1, NOT MT-CO2. LFTs at diagnosis." />
            <Alert variant="warning" text="COX20 (FAM36A, 1p33 AR): MT-CO2-specific chaperone; CIV-Leigh + ataxia; AR (not maternal) — inheritance pattern is the DDx pivot." />
          </div>
          <div className="col-md-6">
            <Alert variant="warning" text="MT-CO1: 12 TM helices vs MT-CO2's 2 TM. CO1 is CIV Assembly Nucleus (12 TM, Heme a/a3-CuB). CO2 is CuA electron acceptor (2 TM, IMS domain). BSN = CO1 phenotype (not CO2)." />
            <Alert variant="warning" text="MELAS DDx (MT-TL1 m.3243A>G): SLE + CIV reduction → MT-CO2 MELAS-CIV overlap. MT-TL1 MELAS has multi-complex reduction; MT-CO2 has ISOLATED CIV. RC enzymology differentiates." />
            <Alert text="NDUFA4 (7p21.3 AR): contacts COX2 IMS loop (CuA zone) directly; CIV-Leigh; AR; no HCM; isolated CIV — but AR not maternal; pArg52Cys-TM-helix vs MT-CO2 maternal CuA mutations." />
          </div>
        </div>
        <div className="mt-2 small text-muted">
          <strong>Biochemistry DDx first branch:</strong> Isolated CIV (point mutation) vs Combined CI+CIV (large deletion) vs All-ETC-low (depletion POLG/TWNK).
        </div>
      </SectionCard>

      {/* Treatments */}
      <SectionCard title="Treatments — MT-CO2 CIV Deficiency" borderColor={COLOR5}>
        {tx.map((t, i) => (
          <div key={i} className="mb-3 pb-3" style={{ borderBottom: i < tx.length - 1 ? '1px solid #e8f5e9' : 'none' }}>
            <div className="d-flex justify-content-between align-items-start">
              <span className="fw-bold small">{t.name}</span>
              <span className="badge" style={{ background: evidColor(t.evidence), fontSize: '0.7rem' }}>
                {t.evidence}
              </span>
            </div>
            <div className="text-muted small mt-1">{t.notes}</div>
          </div>
        ))}
      </SectionCard>

      {/* Contraindications */}
      <SectionCard title="Contraindications — ABSOLUTE & Relative" borderColor={COLOR3}>
        {ci.map((c, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#ffebee' }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold small" style={{ color: COLOR3 }}>{c.drug}</span>
              <span className="badge bg-danger" style={{ fontSize: '0.7rem' }}>{c.class.split('—')[0].trim()}</span>
            </div>
            <div className="text-muted small">{c.reason}</div>
          </div>
        ))}
      </SectionCard>

      {/* Monitoring */}
      <SectionCard title="Monitoring Protocol" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead>
              <tr style={{ background: LIGHT }}>
                <th>Monitoring Item</th><th>Protocol</th>
              </tr>
            </thead>
            <tbody>
              {mon.map((m, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ color: COLOR }}>{m.item}</td>
                  <td className="text-muted">{m.protocol}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const glossary = data.glossary  || [];
  const refs     = data.references || [];

  return (
    <div>
      <SectionCard title="MT-CO2 Glossary — CuA, SCO1/SCO2, COX20, CIV Assembly">
        {glossary.map((g, i) => (
          <div key={i} className="mb-3 pb-3" style={{ borderBottom: i < glossary.length - 1 ? '1px solid #e0e0e0' : 'none' }}>
            <div className="fw-bold small mb-1" style={{ color: COLOR }}>{g.term}</div>
            <div className="text-muted small">{g.definition}</div>
          </div>
        ))}
      </SectionCard>
      <SectionCard title="References" borderColor={COLOR2}>
        <ul className="mb-0 small">
          {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Root page ─────────────────────────────────────────────────────────────────
export default function MTCO2Page() {
  const [tab,        setTab]        = useState(0);
  const [overview,   setOverview]   = useState(null);
  const [breakdown,  setBreakdown]  = useState(null);
  const [definitions,setDefinitions]= useState(null);
  const [error,      setError]      = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mtco2/overview`).then(r => r.json()),
      fetch(`${API}/api/mtco2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mtco2/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            🧬 MT-CO2 — CuA Binuclear Copper Center
          </h4>
          <p className="text-muted small mb-0">
            Cytochrome c Oxidase Subunit II · 227 aa / 26 kDa · 2 TM helices · H-strand m.7586–8269 · OMIM *516040
            · PRIMARY electron acceptor from cytochrome c · SCO1/SCO2 copper chaperones
          </p>
        </div>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR, fontWeight: 600 } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab    data={overview}    />}
      {tab === 1 && <PatientsTab    data={breakdown}   />}
      {tab === 2 && <TreatmentTab   data={breakdown}   />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
