'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// PME Atlas color palette — epilepsy / neurology
const COLOR  = '#4a148c';  // deep purple — epilepsy/neurology
const LIGHT  = '#f3e5f5';  // purple tint
const COLOR2 = '#b71c1c';  // deep red — severe/Lafora
const COLOR3 = '#e65100';  // orange — MERRF/mitochondrial
const COLOR4 = '#1565c0';  // blue — AMRF/renal
const COLOR5 = '#2e7d32';  // green — GOSR2
const COLOR6 = '#37474f';  // blue-grey — KCNC1/AD
const COLOR7 = '#795548';  // brown — PRICKLE1/PCP

const GENE_COLORS = {
  CSTB:    '#4a148c',  // ULD — most common N.Europe PME (deep purple)
  EPM2A:   '#b71c1c',  // Lafora type 1 — severe/fatal (deep red)
  NHLRC1:  '#c62828',  // Lafora type 2 — severe/fatal (red)
  'MT-TK': '#e65100',  // MERRF — mitochondrial (orange)
  SCARB2:  '#1565c0',  // AMRF — renal (blue)
  GOSR2:   '#2e7d32',  // North Sea PME — elevated CK (green)
  KCNC1:   '#37474f',  // EPM7 — AD/de novo (blue-grey)
  PRICKLE1:'#795548',  // EPM1B — PCP (brown)
};

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function BarRow({ label, pct, color = COLOR, note }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between mb-1">
        <span className="small fw-semibold">{label}</span>
        <span className="small text-muted">{typeof pct === 'number' ? `${pct}%` : pct}{note ? ` — ${note}` : ''}</span>
      </div>
      {typeof pct === 'number' && (
        <div className="progress" style={{ height: 8 }}>
          <div className="progress-bar" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color }} />
        </div>
      )}
    </div>
  );
}

function AlertBox({ type = 'info', title, children }) {
  const icons = { danger: '🚨', warning: '⚠️', info: 'ℹ️', success: '✅' };
  return (
    <div className={`alert alert-${type} py-2 px-3 mb-3`}>
      <strong>{icons[type]} {title}</strong>
      <div className="small mt-1">{children}</div>
    </div>
  );
}

function Loading() {
  return <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /><div className="mt-2 text-muted small">Loading PME-Atlas…</div></div>;
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger">{msg}</div>;
}

// ── Tab: Overview ─────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  return (
    <div>
      <AlertBox type="danger" title="DRUG-TO-AVOID (ALL PME) — Absolute Rule">
        CARBAMAZEPINE · OXCARBAZEPINE · PHENYTOIN · LAMOTRIGINE · VIGABATRIN · GABAPENTIN · PREGABALIN
        — all WORSEN myoclonus; risk of myoclonic status epilepticus. Never prescribe for PME.
      </AlertBox>
      <AlertBox type="info" title="Clinical Pearl">
        {ov.clinical_pearl}
      </AlertBox>

      <h6 className="fw-bold mb-2" style={{ color: COLOR }}>Cohort KPIs — {ov.total_patients} Patients (8×40, Seeds {ov.seeds?.[0]}–{ov.seeds?.[ov.seeds?.length - 1]})</h6>
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={ov.total_patients} color={COLOR} />
        <KPI label="Genes" value={ov.gene_count} color={COLOR} />
        <KPI label="Avg Onset" value={`${ov.avg_onset_y}y`} color={COLOR2} />
        <KPI label="Avg Dx Delay" value={`${ov.avg_dx_delay_y}y`} color={COLOR3} />
        <KPI label="On Valproate" value={`${ov.valproate_pct}%`} color={COLOR} />
        <KPI label="On LEV" value={`${ov.levetiracetam_pct}%`} color={COLOR} />
        <KPI label="On Clonazepam" value={`${ov.clonazepam_pct}%`} color={COLOR} />
        <KPI label="On Perampanel" value={`${ov.perampanel_pct}%`} color={COLOR2} />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>Gene-Specific Hallmarks (% of 320)</div>
            <div className="card-body">
              <BarRow label="Lafora bodies on skin biopsy (EPM2A/NHLRC1)" pct={ov.lafora_bodies_pct} color={COLOR2} note={`${ov.lafora_bodies_n} pts`} />
              <BarRow label="Cerebellar atrophy on MRI (KCNC1/MT-TK)" pct={ov.cerebellar_atrophy_pct} color={COLOR6} note={`${ov.cerebellar_atrophy_n} pts`} />
              <BarRow label="Ragged red fibers — muscle biopsy (MT-TK)" pct={ov.ragged_red_pct} color={COLOR3} note={`${ov.ragged_red_n} pts`} />
              <BarRow label="Renal involvement / nephrotic syndrome (SCARB2)" pct={ov.renal_pct} color={COLOR4} note={`${ov.renal_n} pts`} />
              <BarRow label="Elevated serum CK (GOSR2)" pct={ov.elevated_ck_pct} color={COLOR5} note={`${ov.elevated_ck_n} pts`} />
              <BarRow label="Scoliosis (GOSR2)" pct={ov.scoliosis_pct} color={COLOR5} note={`${ov.scoliosis_n} pts`} />
              <BarRow label="Hearing loss (MT-TK / MERRF)" pct={ov.hearing_loss_pct} color={COLOR3} note={`${ov.hearing_loss_n} pts`} />
              <BarRow label="Cardiomyopathy (MT-TK / MERRF)" pct={ov.cardiomyopathy_pct} color={COLOR3} note={`${ov.cardiomyopathy_n} pts`} />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>General PME Features (% of 320)</div>
            <div className="card-body">
              <BarRow label="Photosensitivity" pct={ov.photosensitivity_pct} color={COLOR} note={`${ov.photosensitivity_n} pts`} />
              <BarRow label="Still ambulatory" pct={ov.ambulatory_pct} color={COLOR5} note={`${ov.ambulatory_n} pts`} />
              <div className="mt-3 mb-2 fw-semibold small" style={{ color: COLOR }}>AED Use Across All PME</div>
              <BarRow label="Valproate" pct={ov.valproate_pct} color={COLOR} />
              <BarRow label="Levetiracetam" pct={ov.levetiracetam_pct} color={COLOR} />
              <BarRow label="Clonazepam" pct={ov.clonazepam_pct} color={COLOR} />
              <BarRow label="Zonisamide" pct={ov.zonisamide_pct} color={COLOR6} />
              <BarRow label="Perampanel (Lafora-predominant)" pct={ov.perampanel_pct} color={COLOR2} />
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>PME Classification by Pathogenic Mechanism</div>
        <div className="card-body">
          <div className="row g-2">
            {ov.pme_groups && Object.entries(ov.pme_groups).map(([grp, n]) => (
              <div className="col-md-6 col-lg-4" key={grp}>
                <div className="border rounded p-2 small">
                  <div className="fw-bold" style={{ color: COLOR }}>{n} pts</div>
                  <div className="text-muted">{grp}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Tab: Gene Table ───────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;
  return (
    <div>
      <AlertBox type="warning" title="Key Differentiators">
        Skin biopsy Lafora bodies = EPM2A/NHLRC1 · Ragged red fibers + maternal = MT-TK ·
        PME + renal = SCARB2 · Elevated CK + scoliosis = GOSR2 · AD/de novo + cerebellar atrophy = KCNC1 ·
        AR ULD-like CSTB-negative = PRICKLE1
      </AlertBox>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle small">
          <thead style={{ background: LIGHT }}>
            <tr>
              <th style={{ color: COLOR }}>Gene</th>
              <th>Disease / Subtype</th>
              <th>Locus</th>
              <th>Inh.</th>
              <th>Onset (y)</th>
              <th>Dx Delay (y)</th>
              <th>Myoclonus /10</th>
              <th>Severe %</th>
              <th>Ambulatory %</th>
              <th>Hallmark Finding</th>
              <th>Treatment Pearl</th>
            </tr>
          </thead>
          <tbody>
            {breakdown.map(g => (
              <tr key={g.gene}>
                <td>
                  <span className="badge rounded-pill" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR, fontSize: '0.75rem' }}>
                    {g.gene}
                  </span>
                </td>
                <td className="fw-semibold" style={{ color: GENE_COLORS[g.gene] || COLOR, maxWidth: 180, fontSize: '0.72rem' }}>
                  {g.subtype}
                  <div className="text-muted fw-normal" style={{ fontSize: '0.68rem' }}>OMIM #{g.omim_disease}</div>
                </td>
                <td className="text-muted">{g.locus}</td>
                <td>
                  <span className={`badge ${g.inheritance.includes('Dominant') ? 'bg-warning text-dark' : g.inheritance.includes('Mitochondrial') ? 'bg-danger' : 'bg-secondary'}`}>
                    {g.inheritance.includes('Dominant') ? 'AD' : g.inheritance.includes('Mitochondrial') ? 'Mito' : 'AR'}
                  </span>
                </td>
                <td>{g.avg_onset_y}</td>
                <td>{g.avg_dx_delay_y}</td>
                <td>
                  <div className="progress" style={{ height: 6, minWidth: 50 }}>
                    <div className="progress-bar" style={{ width: `${Math.min(g.avg_myoclonus_score * 10, 100)}%`, backgroundColor: GENE_COLORS[g.gene] || COLOR }} />
                  </div>
                  <span className="text-muted" style={{ fontSize: '0.7rem' }}>{g.avg_myoclonus_score}</span>
                </td>
                <td><span className={`badge ${g.severe_pct > 60 ? 'bg-danger' : g.severe_pct > 30 ? 'bg-warning text-dark' : 'bg-success'}`}>{g.severe_pct}%</span></td>
                <td><span className={`badge ${g.ambulatory_pct > 70 ? 'bg-success' : g.ambulatory_pct > 40 ? 'bg-warning text-dark' : 'bg-danger'}`}>{g.ambulatory_pct}%</span></td>
                <td style={{ maxWidth: 140, fontSize: '0.7rem' }}>
                  <span className="fw-semibold">{g.hallmark_pct}%</span> — {g.hallmark_label}
                </td>
                <td style={{ maxWidth: 180, fontSize: '0.68rem' }} className="text-muted">{g.treatment_pearl}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="row g-3 mt-2">
        {breakdown.map(g => (
          <div className="col-md-6 col-lg-3" key={g.gene}>
            <div className="card shadow-sm h-100">
              <div className="card-header py-1 px-2 fw-bold small" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR, color: '#fff' }}>
                {g.gene} — {g.protein.split('(')[0].trim()}
              </div>
              <div className="card-body py-2 px-2" style={{ fontSize: '0.72rem' }}>
                <div className="text-muted mb-1">{g.aa} · {g.locus} · OMIM #{g.omim_disease}</div>
                <div className="mb-1"><span className="fw-semibold">Mutation:</span> {g.mutation_type}</div>
                <BarRow label="VPA" pct={g.vpa_pct} color={COLOR} />
                <BarRow label="LEV" pct={g.lev_pct} color={COLOR} />
                <BarRow label="Clonazepam" pct={g.clonazepam_pct} color={COLOR} />
                <div className="mt-1 text-muted" style={{ fontSize: '0.68rem' }}>{g.treatment_pearl}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Tab: Clinical Atlas ───────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;

  const GROUPS = [
    {
      title: 'AR Glycogen/Repeat PME — Lafora Disease + Unverricht-Lundborg',
      color: COLOR2,
      genes: ['CSTB', 'EPM2A', 'NHLRC1'],
      notes: [
        'CSTB (ULD/EPM1): dodecamer repeat 5\'UTR → cystatin B silencing. Best prognosis PME. Piracetam effective.',
        'EPM2A (Lafora type 1): laforin phosphatase loss → Lafora bodies. SKIN BIOPSY pathognomonic. Fatal.',
        'NHLRC1 (Lafora type 2): malin E3 ligase loss → same pathway as EPM2A. Clinically identical; genetics differentiate.',
        'KEY: Skin biopsy (axilla/groin) with PAS stain → Lafora bodies = diagnostic for EPM2A/NHLRC1.',
        'Lafora TREATMENT: VPA + ZNS + perampanel; investigational: anti-sense oligos targeting PTG/GYS1.',
      ],
    },
    {
      title: 'Mitochondrial PME — MERRF (MT-TK)',
      color: COLOR3,
      genes: ['MT-TK'],
      notes: [
        'MT-TK: m.8344A>G (>80%) → defective mt-tRNA-Lys → OXPHOS failure (Complexes I/III/IV/V).',
        'MATERNAL INHERITANCE: no male-to-male transmission. Test maternal relatives.',
        'RAGGED RED FIBERS (Gomori trichrome): subsarcolemmal mitochondrial proliferation. COX-negative fibers.',
        'Multi-system: myoclonus + GTCS + ataxia + HEARING LOSS + short stature + CARDIOMYOPATHY + lipomas.',
        'MERRF treatment: CoQ10 + riboflavin. VPA CAUTION (Complex I inhibition → carnitine supplement). NO metformin.',
      ],
    },
    {
      title: 'AR Lysosomal/Golgi PME — AMRF + North Sea PME',
      color: COLOR4,
      genes: ['SCARB2', 'GOSR2'],
      notes: [
        'SCARB2 (AMRF): LIMP-2 loss → GBA mislocalisation + podocyte dysfunction. PME + NEPHROTIC SYNDROME unique.',
        'AMRF: ACTION MYOCLONUS predominantly; typically NO TONIC-CLONIC seizures (unlike other PMEs).',
        'AMRF: FOOT TREMOR early; renal failure often drives prognosis; LEV dose-adjust for CrCl.',
        'GOSR2 (North Sea PME): p.Gly144Trp founder (Dutch/Danish/British). ELEVATED CK + SCOLIOSIS distinctive.',
        'GOSR2: Golgi SNARE defect → glycosylation failure → neuronal + myopathic component (explains elevated CK).',
      ],
    },
    {
      title: 'AD/De Novo + AR PCP PME — KCNC1 + PRICKLE1',
      color: COLOR6,
      genes: ['KCNC1', 'PRICKLE1'],
      notes: [
        'KCNC1 (EPM7): p.Arg320His de novo AD — Kv3.1 channel; cerebellar atrophy MRI; NO DEMENTIA; LEV highly effective.',
        'KCNC1 is the ONLY AD (usually de novo) PME — all others AR or mitochondrial.',
        'KCNC1 prognosis: best among PMEs with cognitive involvement; patients maintain function for decades.',
        'PRICKLE1 (EPM1B): AR; Wnt/PCP signalling; ULD-like phenotype. Test when CSTB repeat expansion negative.',
        'PCP pathway: PRICKLE1 normally suppresses REST (transcriptional repressor of neuronal genes).',
      ],
    },
  ];

  return (
    <div>
      <AlertBox type="danger" title="PME Backbone Treatment (ALL) + Drug-to-Avoid (ALL)">
        BACKBONE: Valproate + Clonazepam + Levetiracetam (+ Zonisamide, Perampanel as add-on) |
        AVOID: CBZ · OXC · PHT · LTG · VGB · GBP · PGB — sodium channel blockers/GABA-T inhibitors worsen myoclonus
      </AlertBox>

      {GROUPS.map(grp => {
        const geneData = breakdown.filter(g => grp.genes.includes(g.gene));
        return (
          <div className="card shadow-sm mb-4" key={grp.title}>
            <div className="card-header fw-semibold" style={{ backgroundColor: grp.color, color: '#fff' }}>
              {grp.title}
            </div>
            <div className="card-body">
              <div className="row g-3 mb-3">
                {geneData.map(g => (
                  <div className="col-md-4" key={g.gene}>
                    <div className="border rounded p-2 h-100" style={{ borderColor: grp.color + '66' }}>
                      <div className="fw-bold small mb-1" style={{ color: GENE_COLORS[g.gene] || grp.color }}>
                        {g.gene} — {g.protein.split('(')[0].trim()}
                      </div>
                      <div style={{ fontSize: '0.72rem' }} className="text-muted mb-2">{g.subtype}</div>
                      <BarRow label={`Hallmark: ${g.hallmark_label}`} pct={g.hallmark_pct} color={grp.color} />
                      <BarRow label="Severe" pct={g.severe_pct} color={COLOR2} />
                      <BarRow label="Still ambulatory" pct={g.ambulatory_pct} color={COLOR5} />
                      <div className="mt-2 small"><span className="fw-semibold">Onset:</span> {g.avg_onset_y}y · <span className="fw-semibold">Delay:</span> {g.avg_dx_delay_y}y</div>
                    </div>
                  </div>
                ))}
              </div>
              <ul className="small mb-0 ps-3">
                {grp.notes.map((n, i) => <li key={i} className="mb-1">{n}</li>)}
              </ul>
            </div>
          </div>
        );
      })}

      <div className="card shadow-sm mb-3" style={{ borderColor: COLOR }}>
        <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>
          Genetic Testing Algorithm — Step-by-Step PME Workup
        </div>
        <div className="card-body small">
          <div className="row g-2">
            {[
              { step: '1', label: 'N.European + AR PME', action: 'CSTB repeat-primed PCR / Southern blot (dodecamer expansion)', color: COLOR },
              { step: '2', label: 'Skin biopsy Lafora bodies', action: 'EPM2A + NHLRC1 sequencing (PAS-positive sweat gland ducts)', color: COLOR2 },
              { step: '3', label: 'Maternal inheritance + elevated lactate', action: 'mtDNA m.8344A>G hotspot (MT-TK) → full mtGenome', color: COLOR3 },
              { step: '4', label: 'PME + renal / nephrotic', action: 'SCARB2 sequencing + CNV analysis (AMRF)', color: COLOR4 },
              { step: '5', label: 'Elevated CK + scoliosis + N.European', action: 'GOSR2 p.Gly144Trp hotspot sequencing', color: COLOR5 },
              { step: '6', label: 'AD or de novo + cerebellar atrophy MRI', action: 'KCNC1 p.Arg320His hotspot → trio WES', color: COLOR6 },
              { step: '7', label: 'AR PME, ULD-like, panels negative', action: 'PRICKLE1 sequencing / PME gene panel', color: COLOR7 },
              { step: '8', label: 'All above negative', action: 'Comprehensive PME panel / trio WES / WGS (KCTD7, LYST, MFSD8…)', color: '#666' },
            ].map(s => (
              <div className="col-md-6 col-lg-3" key={s.step}>
                <div className="border rounded p-2 h-100">
                  <div className="fw-bold" style={{ color: s.color }}>Step {s.step}</div>
                  <div className="fw-semibold small">{s.label}</div>
                  <div className="text-muted" style={{ fontSize: '0.7rem' }}>{s.action}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const [open, setOpen] = useState(null);
  const { definitions } = data;
  return (
    <div>
      <p className="text-muted small mb-3">
        {definitions.length} clinical definitions — click to expand.
      </p>
      <div className="accordion" id="pme-defs">
        {definitions.map((d, i) => (
          <div className="accordion-item" key={i}>
            <h2 className="accordion-header">
              <button
                className={`accordion-button ${open === i ? '' : 'collapsed'} small fw-semibold`}
                type="button"
                onClick={() => setOpen(open === i ? null : i)}
                style={{ color: COLOR }}
              >
                {d.term}
              </button>
            </h2>
            {open === i && (
              <div className="accordion-collapse show">
                <div className="accordion-body small" style={{ whiteSpace: 'pre-line', lineHeight: 1.7 }}>
                  {d.definition}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────
export default function PMEAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [ov, bd, df] = await Promise.all([
          fetch(`${API}/api/pme-atlas/overview`).then(r => r.json()),
          fetch(`${API}/api/pme-atlas/breakdown`).then(r => r.json()),
          fetch(`${API}/api/pme-atlas/definitions`).then(r => r.json()),
        ]);
        setOverview(ov);
        setBreakdown(bd);
        setDefinitions(df);
      } catch (e) {
        setError(e.message);
      }
    };
    load();
  }, []);

  if (error) return <div className="container py-4"><ErrorMsg msg={error} /></div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <span style={{ fontSize: '2rem' }}>🧠</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>PME-Atlas — Progressive Myoclonic Epilepsy</h4>
          <div className="text-muted small">
            Complete 8-Gene Atlas · CSTB · EPM2A · NHLRC1 · MT-TK · SCARB2 · GOSR2 · KCNC1 · PRICKLE1
            · 320 patients (8×40, seeds 1006–1013)
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t}>
            <button className={`nav-link ${tab === t ? 'active fw-semibold' : ''}`} onClick={() => setTab(t)}
              style={tab === t ? { color: COLOR, borderBottom: `2px solid ${COLOR}` } : {}}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* Tab Content */}
      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Gene Table' && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
