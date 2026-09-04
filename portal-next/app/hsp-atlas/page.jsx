'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// HSP Atlas color palette — neurology / spastic paraplegia
const COLOR  = '#1a237e';  // deep navy — neurological
const LIGHT  = '#e8eaf6';  // indigo tint
const COLOR2 = '#b71c1c';  // deep red — severe / AR complex
const COLOR3 = '#e65100';  // orange — mitochondrial (SPG7)
const COLOR4 = '#1b5e20';  // deep green — treatable (CYP7B1)
const COLOR5 = '#4a148c';  // purple — ZFYVE26/SPG15
const COLOR6 = '#37474f';  // blue-grey — AD forms
const COLOR7 = '#6d4c41';  // brown — KIF1A

const GENE_COLORS = {
  SPAST:   '#1a237e',  // SPG4 — most common AD pure (deep navy)
  ATL1:    '#283593',  // SPG3A — childhood AD pure (indigo)
  REEP1:   '#3949ab',  // SPG31 — AD ER-shaping (medium indigo)
  SPG11:   '#b71c1c',  // AR complex — thin CC, cognitive (deep red)
  SPG7:    '#e65100',  // mitochondrial spastic-ataxia (orange)
  CYP7B1:  '#1b5e20',  // treatable — oxysterol (deep green)
  ZFYVE26: '#4a148c',  // Kjellin — macular dystrophy (purple)
  KIF1A:   '#6d4c41',  // SPG30 — severe de novo (brown)
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
  return <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /><div className="mt-2 text-muted small">Loading HSP-Atlas…</div></div>;
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger">{msg}</div>;
}

// ── Tab: Overview ─────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const cf = ov.complex_features_prevalence || {};
  const sev = ov.severity || {};

  return (
    <div>
      <AlertBox type="info" title="First-Line Diagnostic Rule">
        SPAST sequencing + MLPA deletion analysis — covers ~40% of all HSP families.
        Brain MRI (sagittal T1) mandatory in all AR cases: thin corpus callosum → SPG11/ZFYVE26.
        Plasma oxysterol panel (25-OH-cholesterol) for all AR-HSP → elevated = CYP7B1/SPG5 (treatable).
      </AlertBox>
      <AlertBox type="success" title="Only Biochemically Treatable HSP">
        CYP7B1 (SPG5): elevated 25-OH-cholesterol + 27-OH-cholesterol (oxysterol panel) →
        Atorvastatin/Lovastatin trial (Level B) reduces oxysterol burden. First HSP with a rational metabolic therapy.
      </AlertBox>

      <h6 className="fw-bold mb-2" style={{ color: COLOR }}>
        Cohort KPIs — {ov.total_patients} Patients (8×40, Seeds {ov.seed_range})
      </h6>
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={ov.total_patients} color={COLOR} />
        <KPI label="Genes" value={ov.genes_covered} color={COLOR} />
        <KPI label="Mean Onset" value={`${ov.mean_onset_age_y}y`} color={COLOR2} />
        <KPI label="Severe" value={`${sev.severe_pct}%`} color={COLOR2} />
        <KPI label="Moderate" value={`${sev.moderate_pct}%`} color={COLOR3} />
        <KPI label="Mild" value={`${sev.mild_pct}%`} color={COLOR4} />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>
              Complicated Features (% of 320 patients)
            </div>
            <div className="card-body">
              <BarRow label="Thin corpus callosum on MRI" pct={cf.thin_corpus_callosum_pct} color={COLOR2} note="SPG11/ZFYVE26" />
              <BarRow label="Cognitive impairment" pct={cf.cognitive_impairment_pct} color={COLOR2} note="SPG11/ZFYVE26/KIF1A" />
              <BarRow label="Cerebellar signs" pct={cf.cerebellar_signs_pct} color={COLOR3} note="SPG7 predominant" />
              <BarRow label="Peripheral neuropathy" pct={cf.peripheral_neuropathy_pct} color={COLOR6} note="SPG11/KIF1A" />
              <BarRow label="Epilepsy" pct={cf.epilepsy_pct} color={COLOR5} note="KIF1A/SPG30 predominant" />
              <BarRow label="Optic atrophy" pct={cf.optic_atrophy_pct} color={COLOR3} note="SPG7 (~25% of SPG7 pts)" />
              <BarRow label="Pigmentary maculopathy" pct={cf.pigmentary_maculopathy_pct} color={COLOR5} note="ZFYVE26 PATHOGNOMONIC" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>
              HSP Classification by Mechanism
            </div>
            <div className="card-body">
              {ov.hsp_type_breakdown && Object.entries(ov.hsp_type_breakdown).map(([grp, n]) => (
                <div key={grp} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-semibold">{grp}</span>
                    <span className="small text-muted">{n} pts</span>
                  </div>
                  <div className="progress" style={{ height: 8 }}>
                    <div className="progress-bar" style={{
                      width: `${Math.round(n / (ov.total_patients || 320) * 100)}%`,
                      backgroundColor: grp.includes('Treatable') ? COLOR4 : grp.includes('Ataxia') ? COLOR3 : grp.includes('Complex') ? COLOR2 : COLOR,
                    }} />
                  </div>
                </div>
              ))}
              <div className="mt-3">
                <div className="fw-semibold small mb-2" style={{ color: COLOR }}>Inheritance Groups</div>
                {ov.inheritance_breakdown && Object.entries(ov.inheritance_breakdown).map(([k, genes]) => (
                  <div key={k} className="mb-1 small">
                    <span className="fw-semibold text-muted">{k.replace(/_/g, ' ')}: </span>
                    {genes.map(g => (
                      <span key={g} className="badge me-1" style={{ backgroundColor: GENE_COLORS[g] || COLOR, fontSize: '0.7rem' }}>{g}</span>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>
          Key Teaching Points
        </div>
        <div className="card-body">
          <ul className="small mb-0 ps-3">
            {(ov.key_teaching_points || []).map((pt, i) => (
              <li key={i} className="mb-1">{pt}</li>
            ))}
          </ul>
        </div>
      </div>

      {ov.drug_alerts && ov.drug_alerts.length > 0 && (
        <div className="card shadow-sm mb-3" style={{ borderColor: '#f57f17' }}>
          <div className="card-header fw-semibold" style={{ backgroundColor: '#fff8e1', color: '#e65100' }}>
            ⚠️ Drug Alerts
          </div>
          <div className="card-body">
            <ul className="small mb-0 ps-3">
              {ov.drug_alerts.map((a, i) => <li key={i} className="mb-1">{a}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Tab: Gene Table ───────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const { genes } = data;
  return (
    <div>
      <AlertBox type="warning" title="Key Differentiators">
        Thin CC + cognitive = SPG11 · Thin CC + macular dystrophy = ZFYVE26 ·
        Oxysterol elevated = CYP7B1 (treat with statin) · Cerebellar ataxia dominant = SPG7 ·
        Severe + de novo + child = KIF1A · Childhood onset mild AD = ATL1 · Most common AD = SPAST
      </AlertBox>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle small">
          <thead style={{ background: LIGHT }}>
            <tr>
              <th style={{ color: COLOR }}>Gene</th>
              <th>SPG / Disease</th>
              <th>Locus</th>
              <th>Type</th>
              <th>Inh.</th>
              <th>Onset (y)</th>
              <th>n=40 Severe%</th>
              <th>Thin CC%</th>
              <th>Cognit%</th>
              <th>Cerebell%</th>
              <th>Macular%</th>
              <th>Oxysterol%</th>
            </tr>
          </thead>
          <tbody>
            {genes.map(g => {
              const cf = g.complex_features || {};
              const sev = g.severity_distribution || {};
              const onset = g.onset_range_y || [];
              return (
                <tr key={g.gene}>
                  <td>
                    <span className="badge rounded-pill" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR, fontSize: '0.75rem' }}>
                      {g.gene}
                    </span>
                  </td>
                  <td className="fw-semibold" style={{ color: GENE_COLORS[g.gene] || COLOR, maxWidth: 160, fontSize: '0.72rem' }}>
                    {g.hsp_group}
                    <div className="text-muted fw-normal" style={{ fontSize: '0.68rem' }}>
                      {g.protein} · OMIM #{g.omim_disease}
                    </div>
                  </td>
                  <td className="text-muted">{g.locus}</td>
                  <td>
                    <span className={`badge ${g.hsp_type === 'pure' ? 'bg-primary' : 'bg-danger'}`} style={{ fontSize: '0.7rem' }}>
                      {g.hsp_type === 'pure' ? 'Pure' : 'Complex'}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${g.inheritance?.includes('Dominant') ? 'bg-warning text-dark' : 'bg-secondary'}`} style={{ fontSize: '0.7rem' }}>
                      {g.inheritance?.includes('Dominant') ? 'AD' : 'AR'}
                    </span>
                  </td>
                  <td className="text-muted" style={{ fontSize: '0.7rem' }}>
                    {onset[0] && onset[1] ? `${onset[0]}–${onset[1]}` : '—'}
                  </td>
                  <td>
                    <span className={`badge ${sev.severe_pct > 40 ? 'bg-danger' : sev.severe_pct > 20 ? 'bg-warning text-dark' : 'bg-success'}`}>
                      {sev.severe_pct}%
                    </span>
                  </td>
                  <td><span className={`badge ${cf.thin_corpus_callosum_pct > 50 ? 'bg-danger' : cf.thin_corpus_callosum_pct > 10 ? 'bg-warning text-dark' : 'bg-light text-dark'}`}>{cf.thin_corpus_callosum_pct}%</span></td>
                  <td><span className={`badge ${cf.cognitive_impairment_pct > 40 ? 'bg-danger' : cf.cognitive_impairment_pct > 10 ? 'bg-warning text-dark' : 'bg-light text-dark'}`}>{cf.cognitive_impairment_pct}%</span></td>
                  <td><span className={`badge ${cf.cerebellar_signs_pct > 40 ? 'bg-warning text-dark' : 'bg-light text-dark'}`}>{cf.cerebellar_signs_pct}%</span></td>
                  <td><span className={`badge ${cf.pigmentary_maculopathy_pct > 50 ? 'bg-danger' : 'bg-light text-dark'}`}>{cf.pigmentary_maculopathy_pct}%</span></td>
                  <td><span className={`badge ${cf.pigmentary_maculopathy_pct > 50 ? 'bg-light text-dark' : g.gene === 'CYP7B1' ? 'bg-success' : 'bg-light text-dark'}`}>{g.gene === 'CYP7B1' ? '>90%' : '<5%'}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="row g-3 mt-2">
        {genes.map(g => {
          const cf = g.complex_features || {};
          const sev = g.severity_distribution || {};
          return (
            <div className="col-md-6 col-lg-3" key={g.gene}>
              <div className="card shadow-sm h-100">
                <div className="card-header py-1 px-2 fw-bold small" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR, color: '#fff' }}>
                  {g.gene} — {g.protein}
                </div>
                <div className="card-body py-2 px-2" style={{ fontSize: '0.72rem' }}>
                  <div className="text-muted mb-1">{g.aa} · {g.locus} · OMIM #{g.omim_disease}</div>
                  <div className="mb-1"><span className="fw-semibold">Group:</span> {g.hsp_group}</div>
                  <BarRow label="Severe" pct={sev.severe_pct} color={GENE_COLORS[g.gene] || COLOR} />
                  <BarRow label="Thin CC" pct={cf.thin_corpus_callosum_pct} color={COLOR2} />
                  <BarRow label="Cognitive" pct={cf.cognitive_impairment_pct} color={COLOR2} />
                  <BarRow label="Cerebellar" pct={cf.cerebellar_signs_pct} color={COLOR3} />
                  {g.gene === 'ZFYVE26' && <BarRow label="Macular" pct={cf.pigmentary_maculopathy_pct} color={COLOR5} />}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Tab: Clinical Atlas ───────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const { genes } = data;

  const GROUPS = [
    {
      title: 'AD Pure HSP — ER-Shaping Pathway (SPAST · ATL1 · REEP1)',
      color: COLOR,
      geneList: ['SPAST', 'ATL1', 'REEP1'],
      notes: [
        'SPAST (SPG4): most common HSP (~40%); haploinsufficiency; microtubule severing; MLPA mandatory for deletions.',
        'ATL1 (SPG3A): childhood onset (<10 y) distinguishes from SPAST; milder; atlastin GTPase fuses ER tubules.',
        'REEP1 (SPG31): ER tubule shaping (same pathway); also causes distal HMN5B (dHMN overlap families).',
        'All three share ER tubule shaping pathway — long CST axons most vulnerable (length-dependent).',
        'TREATMENT: Baclofen (oral) first-line; intrathecal baclofen for severe; tizanidine/botox adjuncts; physiotherapy lifelong.',
        'INTRA-FAMILIAL VARIABILITY (SPAST): same mutation → asymptomatic carrier vs wheelchair-dependent sibling.',
      ],
    },
    {
      title: 'AR Complex HSP — Autophagy-Lysosomal (SPG11 · ZFYVE26)',
      color: COLOR2,
      geneList: ['SPG11', 'ZFYVE26'],
      notes: [
        'SPG11: THIN CORPUS CALLOSUM + periventricular "ears" PATHOGNOMONIC — do brain MRI first, then genetics.',
        'ZFYVE26 (Kjellin): thin CC + PIGMENTARY MACULAR DYSTROPHY — combination distinguishes from SPG11.',
        'Both: spatacsin (SPG11) and spastizin (ZFYVE26) co-operate in lysosomal tubulation/autophagy.',
        'SPG11: most common AR-CSHSP (~20%); cognitive impairment + dysarthria characteristic.',
        'ZFYVE26: ERG + OCT mandatory (cone-rod dystrophy on ERG); earlier onset + faster progression than SPG11.',
        'MANDATORY INVESTIGATIONS: sagittal T1 MRI (CC thickness) + ERG/OCT (macular) in all AR-CSHSP.',
      ],
    },
    {
      title: 'Biochemically Treatable HSP — Oxysterol (CYP7B1/SPG5)',
      color: COLOR4,
      geneList: ['CYP7B1'],
      notes: [
        'CYP7B1 (SPG5): ONLY HSP with a measurable plasma biomarker AND rational metabolic therapy.',
        'Deficiency → 25-OH-cholesterol and 27-OH-cholesterol accumulate (oxysterol panel MANDATORY).',
        'Atorvastatin/lovastatin 40-80 mg/day: reduces oxysterol precursor load — Level B evidence.',
        'CHENODEOXYCHOLIC ACID (CDCA) supplementation investigational — bile acid pathway bypass.',
        'Test ALL AR-HSP: plasma oxysterol panel (25-OH-cholesterol); CYP7B1 if elevated.',
        'Ongoing: gene therapy / enzyme replacement research. Only treatable HSP subtype currently.',
      ],
    },
    {
      title: 'Mitochondrial Spastic-Ataxia (SPG7) · Severe AD De Novo (KIF1A)',
      color: COLOR3,
      geneList: ['SPG7', 'KIF1A'],
      notes: [
        'SPG7 (Paraplegin): CEREBELLAR ATAXIA often > spasticity — frequently misdiagnosed as SCA; include in ATAXIA panels.',
        'SPG7: optic atrophy ~25% (optic exam + OCT mandatory); muscle biopsy ragged-red fibres ~20%.',
        'SPG7: p.Ala510Val European founder — single heterozygote common; need compound heterozygous for diagnosis.',
        'KIF1A (SPG30): kinesin-3 anterograde axonal transport; SEVERE COMPLEX HSP; de novo dominant ~50% of cases.',
        'KIF1A: cerebral atrophy + thin CC + epilepsy + intellectual disability; progressive and severe.',
        'SPG7 TREATMENT: CoQ10 + riboflavin (mitochondrial support); avoid metformin; baclofen for spasticity.',
      ],
    },
  ];

  const geneMap = Object.fromEntries((genes || []).map(g => [g.gene, g]));

  return (
    <div>
      <AlertBox type="info" title="Pure HSP vs Complex HSP — The Key Split">
        PURE: lower-limb spasticity ± mild bladder/vibration ± variable upper limb (SPAST, ATL1, REEP1) |
        COMPLEX: major additional features — thin CC, cognitive impairment, macular dystrophy, cerebellar ataxia,
        peripheral neuropathy, epilepsy (SPG11, ZFYVE26, CYP7B1, SPG7, KIF1A)
      </AlertBox>

      {GROUPS.map(grp => {
        const geneData = grp.geneList.map(gn => geneMap[gn]).filter(Boolean);
        return (
          <div className="card shadow-sm mb-4" key={grp.title}>
            <div className="card-header fw-semibold" style={{ backgroundColor: grp.color, color: '#fff' }}>
              {grp.title}
            </div>
            <div className="card-body">
              <div className="row g-3 mb-3">
                {geneData.map(g => {
                  const cf = g.complex_features || {};
                  const sev = g.severity_distribution || {};
                  return (
                    <div className="col-md-4" key={g.gene}>
                      <div className="border rounded p-2 h-100" style={{ borderColor: grp.color + '66' }}>
                        <div className="fw-bold small mb-1" style={{ color: GENE_COLORS[g.gene] || grp.color }}>
                          {g.gene} — {g.protein}
                        </div>
                        <div style={{ fontSize: '0.72rem' }} className="text-muted mb-2">{g.hsp_group}</div>
                        <BarRow label="Severe" pct={sev.severe_pct} color={COLOR2} />
                        <BarRow label="Thin CC" pct={cf.thin_corpus_callosum_pct} color={grp.color} />
                        <BarRow label="Cognitive" pct={cf.cognitive_impairment_pct} color={grp.color} />
                        <BarRow label="Cerebellar" pct={cf.cerebellar_signs_pct} color={COLOR3} />
                        <BarRow label="Macular" pct={cf.pigmentary_maculopathy_pct} color={COLOR5} />
                        <div className="mt-2 small">
                          <span className="fw-semibold">Onset: </span>
                          {g.onset_range_y?.[0]}–{g.onset_range_y?.[1]} y
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
              <ul className="small mb-0 ps-3">
                {grp.notes.map((note, i) => <li key={i} className="mb-1">{note}</li>)}
              </ul>
            </div>
          </div>
        );
      })}

      <div className="card shadow-sm mb-3" style={{ borderColor: COLOR }}>
        <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>
          Genetic Testing Algorithm — Step-by-Step HSP Workup
        </div>
        <div className="card-body small">
          <div className="row g-2">
            {[
              { step: '1', label: 'AD HSP or isolated case', action: 'SPAST sequencing + MLPA deletion (covers ~40%)', color: COLOR },
              { step: '2', label: 'Childhood onset + AD pure', action: 'ATL1 sequencing (SPAST negative)', color: '#283593' },
              { step: '3', label: 'AD pure + adult onset', action: 'REEP1 sequencing (SPAST/ATL1 negative)', color: '#3949ab' },
              { step: '4', label: 'AR + brain MRI thin CC', action: 'SPG11 sequencing (most common AR-CSHSP)', color: COLOR2 },
              { step: '5', label: 'AR + thin CC + macular', action: 'ZFYVE26 sequencing (Kjellin syndrome)', color: COLOR5 },
              { step: '6', label: 'AR + plasma oxysterol elevated', action: 'CYP7B1 sequencing → atorvastatin trial', color: COLOR4 },
              { step: '7', label: 'AR + cerebellar ataxia + optic', action: 'SPG7 (check p.Ala510Val; muscle biopsy)', color: COLOR3 },
              { step: '8', label: 'Severe + de novo + child onset', action: 'KIF1A sequencing (trio WES if panel negative)', color: COLOR7 },
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
  const defs = data.definitions || [];
  return (
    <div>
      <p className="text-muted small mb-3">
        {defs.length} clinical definitions — click to expand.
      </p>
      <div className="accordion" id="hsp-defs">
        {defs.map((d, i) => (
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
export default function HSPAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [ov, bd, df] = await Promise.all([
          fetch(`${API}/api/hsp-atlas/overview`).then(r => r.json()),
          fetch(`${API}/api/hsp-atlas/breakdown`).then(r => r.json()),
          fetch(`${API}/api/hsp-atlas/definitions`).then(r => r.json()),
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
        <span style={{ fontSize: '2rem' }}>🧬</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>HSP-Atlas — Hereditary Spastic Paraplegia</h4>
          <div className="text-muted small">
            Complete 8-Gene Atlas · SPAST · ATL1 · REEP1 · SPG11 · SPG7 · CYP7B1 · ZFYVE26 · KIF1A
            · 320 patients (8×40, seeds 1014–1021)
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
