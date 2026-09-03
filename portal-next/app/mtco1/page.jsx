'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
// Very dark teal — CIV catalytic core / oxygen-reduction / Assembly Nucleus
const COLOR  = '#004d40';
const LIGHT  = '#e0f2f1';
const COLOR2 = '#00695c';
const COLOR3 = '#b71c1c';   // danger / absolute CI / BSN
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
          🧬 MT-CO1 — CIV Assembly Nucleus / Bilateral Striatal Necrosis / CIV-Leigh / LHON-plus
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
          🔬 MT-CO1 is the CIV ASSEMBLY NUCLEUS — all CIV assembly factors (COA3/MITRAC, COX14, COA6, PET100, SURF1)
          load onto nascent MT-CO1 co-translationally. Contains Heme a (electron relay) + Heme a3-CuB (O₂→H₂O terminal site).
          Propofol DIRECTLY inhibits heme a3-CuB — ABSOLUTE CI.
          <span style={{ color: COLOR3 }}> Bilateral Striatal Necrosis (m.6930G>A) is the MOST DISTINCTIVE MT-CO1 phenotype —
          bilateral putamen/caudate T2 hyperintensity, NOT classic Leigh brainstem pattern.</span>
        </p>
      </div>

      {/* Critical warnings */}
      <Alert variant="danger"
        text="⚠️ PROPOFOL: ABSOLUTE CI — directly inhibits MT-CO1 heme a3-CuB (O₂-reduction site) + causes PRIS. Use SEVOFLURANE for all anaesthesia." />
      <Alert variant="danger"
        text="⚠️ LINEZOLID / CHLORAMPHENICOL: ABSOLUTE CI — directly blocks MT-CO1 synthesis at mitoribosome (23S rRNA). MT-CO1 loss collapses entire CIV assembly nucleus." />
      <Alert variant="danger"
        text="⚠️ METFORMIN + VPA: ABSOLUTE CI — CI inhibition + CoA sequestration + POLG inhibition compound CIV failure." />
      <Alert variant="warning"
        text="⚠️ BSN MIMIC — SLC19A3 (BTBGD): Bilateral striatal necrosis clinically identical to m.6930G>A. Give THIAMINE + BIOTIN EMPIRICALLY before mtDNA result — BTBGD is rapidly fatal without but fully reversible with early treatment." />
      <Alert variant="warning"
        text="⚠️ WES MISSES MT-CO1: Dedicated mtDNA sequencing in MUSCLE required. Blood underestimates heteroplasmy 15-30%. Large deletions need long-read or Southern blot." />

      {/* KPI row */}
      <div className="row mb-3">
        <KPI label="Seizures (%)" value={`${k.seizures_pct}%`} color={COLOR} />
        <KPI label="Hypotonia (%)" value={`${k.hypotonia_pct}%`} color={COLOR2} />
        <KPI label="Lactic Acidosis" value={`${k.lactic_acidosis_pct}%`} color={COLOR3} />
        <KPI label="Leigh MRI (%)" value={`${k.leigh_mri_pct}%`} color={COLOR} />
        <KPI label="BSN MRI (%)" value={`${k.bsn_mri_pct}%`} color={COLOR3} />
        <KPI label="Optic Atrophy" value={`${k.optic_atrophy_pct}%`} color={COLOR4} />
        <KPI label="Mean CIV %" value={`${k.mean_civ_pct}%`} color={COLOR3} />
        <KPI label="Median Onset" value={`${k.median_onset_mo}mo`} color={COLOR2} />
      </div>

      {/* Phenotype distribution */}
      <SectionCard title="Phenotype Distribution (40-patient Cohort)">
        {pheno.map((p, i) => (
          <Bar key={i} label={p.class} value={p.pct}
            color={p.class.includes('BSN') ? COLOR3 : p.class.includes('LHON') ? '#6a1b9a'
                 : p.class.includes('MELAS') ? COLOR4 : p.class.includes('KSS') ? '#37474f' : COLOR} />
        ))}
        <p className="text-muted small mt-2 mb-0">
          BSN (m.6930G>A) = most distinctive MT-CO1 phenotype — bilateral putamen/caudate, NOT Leigh brainstem.
          LHON-plus (m.7444G>A stop read-through) = optic atrophy + SNHL dual phenotype.
        </p>
      </SectionCard>

      {/* Variant overview bars */}
      <SectionCard title="Variant Frequency (Population Estimate)">
        {(data.top_variants || []).slice(0, 5).map(([v, n], i) => (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span className="font-monospace">{v}</span>
              <span className="text-muted">{n} pts</span>
            </div>
            <div className="progress" style={{ height: 10 }}>
              <div className="progress-bar"
                style={{ width: `${(n / (data.cohort_n || 40)) * 100}%`, backgroundColor: COLOR }} />
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Seizure types & triggers */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Seizure Types" borderColor={COLOR2}>
            {(data.seizure_types || []).map((s, i) => (
              <Bar key={i} label={s.type} value={s.pct} color={COLOR2} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Common Triggers" borderColor={COLOR4}>
            {(data.triggers || []).map((t, i) => (
              <Bar key={i} label={t.trigger} value={t.pct} color={COLOR4} />
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Key concepts */}
      <SectionCard title="Key Clinical Concepts">
        {concepts.slice(0, 6).map((c, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: LIGHT }}>
            <div className="fw-semibold small mb-1" style={{ color: COLOR }}>{c.concept}</div>
            <div className="small text-muted">{c.detail}</div>
          </div>
        ))}
      </SectionCard>

      {/* References */}
      <SectionCard title="Key References" borderColor={COLOR2}>
        <ol className="small mb-0 ps-3">
          {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Tab: Patients & Features ──────────────────────────────────────────────────
function PatientsTab({ data }) {
  const [sort, setSort] = useState('phenotype');
  if (!data) return <p className="text-muted">Loading…</p>;

  const pts = [...(data.patients || [])].sort((a, b) => {
    if (sort === 'civ_pct') return a.civ_pct - b.civ_pct;
    if (sort === 'onset_mo') return a.onset_mo - b.onset_mo;
    return (a[sort] || '').toString().localeCompare((b[sort] || '').toString());
  });

  const variants = data.variants || [];

  return (
    <div>
      <SectionCard title="Variant Details">
        {variants.map((v, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: LIGHT }}>
            <div className="d-flex justify-content-between align-items-start">
              <div>
                <span className="badge me-2" style={{ background: COLOR, color: '#fff' }}>
                  {v.variant}
                </span>
                <span className="badge me-2" style={{ background: COLOR2, color: '#fff' }}>
                  {v.amino_acid}
                </span>
                <span className="badge text-dark me-1"
                  style={{ background: '#ffe082' }}>
                  {v.freq_pct}% population
                </span>
                <span className="badge text-white" style={{ background: '#78909c' }}>
                  {v.n_in_cohort} pts in cohort
                </span>
              </div>
              <span className="small text-muted ms-2">{v.modal_phenotype}</span>
            </div>
            <div className="small text-muted mt-1">{v.structural_impact}</div>
            <div className="small mt-1">{v.detail}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="40-Patient Cohort">
        <div className="mb-2 d-flex gap-2 flex-wrap">
          {['phenotype','variant','civ_pct','onset_mo'].map(s => (
            <button key={s} className="btn btn-sm"
              style={{ background: sort === s ? COLOR : '#e0e0e0', color: sort === s ? '#fff' : '#333' }}
              onClick={() => setSort(s)}>Sort: {s}</button>
          ))}
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr>
                <th>ID</th><th>Phenotype</th><th>Variant</th><th>AA</th>
                <th>CIV%</th><th>Onset(mo)</th>
                <th>Sz</th><th>Hypo</th><th>Lactic</th>
                <th>BSN-MRI</th><th>Leigh-MRI</th><th>OA</th><th>SNHL</th><th>Dystonia</th>
              </tr>
            </thead>
            <tbody>
              {pts.map(p => (
                <tr key={p.id}>
                  <td className="fw-bold">{p.id}</td>
                  <td style={{ maxWidth: 160, overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}
                    title={p.phenotype}>{p.phenotype}</td>
                  <td className="font-monospace">{p.variant}</td>
                  <td className="font-monospace small">{p.amino_acid}</td>
                  <td className={p.civ_pct < 10 ? 'text-danger fw-bold' : ''}>{p.civ_pct}%</td>
                  <td>{p.onset_mo}</td>
                  <td>{p.seizure ? '✓' : '–'}</td>
                  <td>{p.hypotonia ? '✓' : '–'}</td>
                  <td>{p.lactic_ac ? '✓' : '–'}</td>
                  <td className={p.bsn_mri ? 'text-danger fw-bold' : ''}>{p.bsn_mri ? '✓' : '–'}</td>
                  <td>{p.leigh_mri ? '✓' : '–'}</td>
                  <td>{p.optic_atrophy ? '✓' : '–'}</td>
                  <td>{p.snhl ? '✓' : '–'}</td>
                  <td>{p.dystonia ? '✓' : '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Remaining key concepts */}
      <SectionCard title="Additional Key Concepts" borderColor={COLOR2}>
        {(data.key_concepts || []).slice(6).map((c, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: LIGHT }}>
            <div className="fw-semibold small mb-1" style={{ color: COLOR2 }}>{c.concept}</div>
            <div className="small text-muted">{c.detail}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Treatments & DDx ─────────────────────────────────────────────────────
function TreatmentTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const txs   = data.treatments || [];
  const cis   = data.contraindications || [];
  const mons  = data.monitoring || [];

  const ciColor = c => c === 'ABSOLUTE' ? COLOR3 : c === 'CONTRAINDICATED — MELAS-Leigh phenotype' ? '#c62828'
                    : c === 'CONTRAINDICATED' ? '#e53935' : c === 'HIGH CAUTION' ? COLOR4 : '#78909c';

  return (
    <div>
      <SectionCard title="Treatments" borderColor={COLOR5}>
        {txs.map((t, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f1f8e9' }}>
            <div className="d-flex align-items-start gap-2">
              <span className="badge" style={{ background: COLOR5, color: '#fff', minWidth: 28 }}>
                {t.evidence}
              </span>
              <div>
                <div className="fw-semibold small">{t.name}</div>
                <div className="small text-muted">{t.notes}</div>
              </div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications" borderColor={COLOR3}>
        {cis.map((c, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#fff8f8' }}>
            <div className="d-flex align-items-start gap-2">
              <span className="badge text-white" style={{ background: ciColor(c.class), minWidth: 90 }}>
                {c.class}
              </span>
              <div>
                <div className="fw-semibold small">{c.drug}</div>
                <div className="small text-muted">{c.reason}</div>
              </div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Protocol" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: COLOR2, color: '#fff' }}>
              <tr><th>Monitoring Item</th><th>Protocol</th></tr>
            </thead>
            <tbody>
              {mons.map((m, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ minWidth: 200 }}>{m.item}</td>
                  <td>{m.protocol}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* DDx summary */}
      <SectionCard title="Differential Diagnosis — Key DDx Pivots" borderColor={COLOR4}>
        {[
          ['SCO2 (22q13.33)',    'HCM ~100% — MT-CO1 CIV-Leigh has NO HCM; Echo mandatory to confirm absence'],
          ['SCO1 (17p13.1)',     'Hepatopathy 100% neonatal onset — MT-CO1 has NO hepatopathy; LFTs normal'],
          ['COX10 (17p12)',      'Renal tubulopathy ~40% — MT-CO1 point mutations have NO tubulopathy; large deletion may'],
          ['SURF1 (9q34.2)',     'Most common CIV-Leigh assembly factor; deeper CIV absence (70-80%); Leigh-only, no BSN'],
          ['SLC19A3 / BTBGD',   'Bilateral striatal necrosis IDENTICAL to m.6930G>A BSN on MRI; TREATABLE — give thiamine + biotin empirically NOW'],
          ['NDUFA4 (7p21.3)',   'CIV structural subunit (despite NDUFA name); isolated CIV; AR biallelic nuclear; no BSN phenotype'],
          ['MT-ND4/ND1/ND6',   'LHON primary variants — optic atrophy without SNHL; CI deficiency not CIV; H-strand encoding for ND4/ND1'],
          ['Wilson disease',     'BSN mimic on MRI — check KF rings, serum ceruloplasmin, 24h urine copper; AR, not maternal'],
          ['MELAS (MT-TL1)',     'Stroke-like + BG lesions — tRNA-Leu; pan-OXPHOS (not isolated CIV); cortical asymmetric vs MT-CO1 striatal symmetric'],
        ].map(([d, n], i) => (
          <div key={i} className="mb-2 p-2 rounded" style={{ background: '#fff8e1' }}>
            <span className="fw-semibold small" style={{ color: COLOR4 }}>{d}: </span>
            <span className="small">{n}</span>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const glossary = data.glossary || [];
  const refs     = data.references || [];

  return (
    <div>
      <SectionCard title="Glossary">
        {glossary.map((g, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: LIGHT }}>
            <div className="fw-bold small mb-1" style={{ color: COLOR }}>{g.term}</div>
            <div className="small text-muted">{g.definition}</div>
          </div>
        ))}
      </SectionCard>
      <SectionCard title="References" borderColor={COLOR2}>
        <ol className="small mb-0 ps-3">
          {refs.map((r, i) => <li key={i} className="mb-2">{r}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Root page ─────────────────────────────────────────────────────────────────
export default function MTCO1Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOv]     = useState(null);
  const [breakdown, setBd]    = useState(null);
  const [definitions, setDef] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mtco1/overview`).then(r => r.json()),
      fetch(`${API}/api/mtco1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mtco1/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, def]) => { setOv(ov); setBd(bd); setDef(def); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="container py-5 text-center">
      <div className="spinner-border" style={{ color: COLOR }} />
      <p className="mt-2 text-muted">Loading MT-CO1 dashboard…</p>
    </div>
  );
  if (error) return (
    <div className="container py-4">
      <div className="alert alert-danger">Error: {error}</div>
    </div>
  );

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded text-white" style={{ background: COLOR }}>
        <h4 className="mb-0 fw-bold">
          🧬 MT-CO1 — CIV Assembly Nucleus / Bilateral Striatal Necrosis / CIV-Leigh / LHON-plus
        </h4>
        <p className="mb-0 small opacity-75">
          514 aa · 57 kDa · 12 TM helices · Heme a + Heme a3-CuB (O₂→H₂O terminal site) ·
          mtDNA H-strand m.5904–7445 · OMIM *516030 · Maternal inheritance ·
          Assembly nucleus for all CIV subunits
        </p>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab    data={overview}     />}
      {tab === 1 && <PatientsTab    data={breakdown}    />}
      {tab === 2 && <TreatmentTab   data={breakdown}    />}
      {tab === 3 && <DefinitionsTab data={definitions}  />}
    </div>
  );
}
