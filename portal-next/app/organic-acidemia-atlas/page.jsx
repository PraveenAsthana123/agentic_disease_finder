'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#2e7d32';  // deep green — organic acid / amino acid metabolism
const LIGHT  = '#e8f5e9';
const COLOR2 = '#b71c1c';  // KD contraindicated (danger)
const COLOR3 = '#1565c0';  // OHCbl responsive (treatment)
const COLOR4 = '#e65100';  // VPA risk
const COLOR5 = '#6a1b9a';  // betaine / glycine treatment
const COLOR6 = '#37474f';  // NBS detected
const COLOR7 = '#004d40';  // benign / asymptomatic
const COLOR8 = '#880e4f';  // retinopathy

const CLASS_COLORS = {
  propionyl_coa_carboxylase_alpha:       '#2e7d32',
  propionyl_coa_carboxylase_beta:        '#388e3c',
  methylmalonyl_coa_mutase:              '#b71c1c',
  methylmalonyl_coa_mutase_cobalamin_a:  '#1565c0',
  methylmalonyl_coa_mutase_cobalamin_b:  '#1976d2',
  cobalamin_processing_cblc:             '#880e4f',
  isovaleryl_coa_dehydrogenase:          '#e65100',
  hmg_coa_lyase:                         '#f57f17',
  methylcrotonyl_coa_carboxylase_alpha:  '#558b2f',
  methylcrotonyl_coa_carboxylase_beta:   '#689f38',
};

const CLASS_LABELS = {
  propionyl_coa_carboxylase_alpha:       'Propionyl-CoA carboxylase α — Propionic Acidemia type A (PCCA)',
  propionyl_coa_carboxylase_beta:        'Propionyl-CoA carboxylase β — Propionic Acidemia type B (PCCB)',
  methylmalonyl_coa_mutase:              'Methylmalonyl-CoA mutase — Classic MMA mut0/mut− (MUT)',
  methylmalonyl_coa_mutase_cobalamin_a:  'Mitochondrial cobalamin reductase — MMA cblA, OHCbl responsive (MMAA)',
  methylmalonyl_coa_mutase_cobalamin_b:  'Adenosylcobalamin synthase — MMA cblB, variable OHCbl response (MMAB)',
  cobalamin_processing_cblc:             'Cytosolic cobalamin decyanase/reductase — Combined MMA+HCU, cblC (MMACHC)',
  isovaleryl_coa_dehydrogenase:          'Isovaleryl-CoA dehydrogenase — Isovaleric Acidemia, glycine Rx (IVD)',
  hmg_coa_lyase:                         'HMG-CoA lyase — Ketogenesis+leucine block, KD absolute CI (HMGCL)',
  methylcrotonyl_coa_carboxylase_alpha:  '3-MCC carboxylase α — 3-MCC deficiency, usually benign (MCCC1)',
  methylcrotonyl_coa_carboxylase_beta:   '3-MCC carboxylase β — 3-MCC deficiency, usually benign (MCCC2)',
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

function BarRow({ label, pct, color = COLOR }) {
  return (
    <div className="mb-1">
      <div className="d-flex justify-content-between mb-0" style={{ fontSize: '0.78rem' }}>
        <span>{label}</span><span className="fw-semibold">{pct}%</span>
      </div>
      <div className="progress" style={{ height: '7px' }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

export default function OrganicAcademiaAtlasPage() {
  const [tab, setTab]           = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]         = useState(null);
  const [loading, setLoading]   = useState(true);
  const [err, setErr]           = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/organic-acidemia-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/organic-acidemia-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/organic-acidemia-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center"><div className="spinner-border" style={{ color: COLOR }} /><p className="mt-2">Loading Organic-Acidemia-Atlas…</p></div>;
  if (err)     return <div className="p-4 alert alert-danger">Error: {err}</div>;

  const ac = overview?.aggregate_clinical || {};
  const ds = overview?.drug_safety || {};
  const genes = breakdown?.genes || [];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${COLOR} 0%, #1b5e20 100%)` }}>
        <h4 className="mb-1 fw-bold">&#x1f9ec; Organic-Acidemia-Atlas — Complete 10-Gene Organic Acidemia Atlas</h4>
        <p className="mb-1 small">
          PCCA(PA-A) · PCCB(PA-B) · MUT(classic-MMA) · MMAA(cblA) · MMAB(cblB) · MMACHC(cblC) ·
          IVD(IVA) · HMGCL(HMG-CoA-lyase) · MCCC1(3-MCC-α) · MCCC2(3-MCC-β) |&nbsp;
          400-patient aggregate (10×40, seeds 823–832)
        </p>
        <p className="mb-0 small opacity-75">
          PA-DCM-30-45pct · MUT-Renal-Damage · cblA-OHCbl-GT80pct-Response · cblC-Combined-MMA-HCU-Retinopathy-80pct ·
          IVA-Sweaty-Feet-Glycine-Rx · HMGCL-KD-ABSOLUTE-CI-NO-Ketogenesis · 3-MCC-Usually-Benign-NOT-Biotin-Responsive ·
          VPA-HIGH-RISK-ALL · GIR-8-12-Emergency · Zero-Protein-24-48h-Crisis
        </p>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active fw-bold' : ''}`}
              style={tab === t ? { borderBottomColor: COLOR, color: COLOR } : {}}
              onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'Overview' && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Genes"          value={overview?.n_genes}            color={COLOR} />
            <KPI label="Patients"       value={overview?.n_patients}         color={COLOR} />
            <KPI label="Encephalopathy" value={`${ac.encephalopathy_pct}%`}  color={COLOR2} />
            <KPI label="Hyperammonemia" value={`${ac.hyperammonemia_pct}%`}  color={COLOR2} />
            <KPI label="Hypoglycaemia"  value={`${ac.hypoglycaemia_pct}%`}   color={COLOR} />
            <KPI label="NBS Detected"   value={`${ac.nbs_positive_pct}%`}    color={COLOR6} />
            <KPI label="Lactic Acidosis"value={`${ac.lactic_acidosis_pct}%`} color={COLOR4} />
            <KPI label="Retinopathy"    value={`${ac.retinopathy_pct}%`}     color={COLOR8} />
            <KPI label="Renal Involve." value={`${ac.renal_involvement_pct}%`}color={COLOR2} />
            <KPI label="Neonatal Crisis"value={`${ac.neonatal_crisis_pct}%`} color={COLOR2} />
          </div>

          <div className="row g-3">
            {/* Clinical bars */}
            <div className="col-md-5">
              <div className="card shadow-sm">
                <div className="card-header fw-bold small" style={{ background: LIGHT }}>
                  Aggregate Clinical Profile (400 patients, 10 genes)
                </div>
                <div className="card-body py-2 px-3">
                  <BarRow label="Encephalopathy"    pct={ac.encephalopathy_pct}    color={COLOR2} />
                  <BarRow label="Hyperammonemia"    pct={ac.hyperammonemia_pct}    color={COLOR2} />
                  <BarRow label="Hypoglycaemia"     pct={ac.hypoglycaemia_pct}     color={COLOR} />
                  <BarRow label="Lactic Acidosis"   pct={ac.lactic_acidosis_pct}   color={COLOR4} />
                  <BarRow label="Neonatal Crisis"   pct={ac.neonatal_crisis_pct}   color={COLOR2} />
                  <BarRow label="Epilepsy"          pct={ac.epilepsy_pct}          color={COLOR} />
                  <BarRow label="Hepatopathy"       pct={ac.hepatopathy_pct}       color={COLOR} />
                  <BarRow label="Renal Involvement" pct={ac.renal_involvement_pct} color={COLOR2} />
                  <BarRow label="Retinopathy (cblC)"pct={ac.retinopathy_pct}       color={COLOR8} />
                  <BarRow label="DCM (PA only)"     pct={ac.dcm_pct}               color={COLOR2} />
                  <BarRow label="tHcy elevated (cblC)" pct={ac.homocysteine_elevated_pct} color={COLOR5} />
                </div>
              </div>
            </div>

            {/* Gene classes */}
            <div className="col-md-7">
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold small" style={{ background: LIGHT }}>
                  Metabolic Pathway Classes
                </div>
                <div className="card-body py-2">
                  {Object.entries(overview?.gene_classes || {}).map(([cls, genes]) => (
                    <div key={cls} className="mb-2">
                      <div className="fw-semibold small text-capitalize" style={{ color: COLOR }}>
                        {cls.replace(/_/g, ' ')}
                      </div>
                      <div className="d-flex flex-wrap gap-1 mt-1">
                        {genes.map(g => (
                          <span key={g} className="badge rounded-pill text-white" style={{ background: CLASS_COLORS[Object.keys(CLASS_COLORS).find(k => k.includes(g.toLowerCase().replace('_',''))) || 'propionyl_coa_carboxylase_alpha'] || COLOR }}>{g}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Drug safety */}
              <div className="card shadow-sm">
                <div className="card-header fw-bold small" style={{ background: '#ffebee' }}>
                  Drug Safety &amp; Treatment Rules
                </div>
                <div className="card-body py-2 small">
                  <div className="mb-1"><strong style={{ color: COLOR2 }}>KD CONTRAINDICATED:</strong>{' '}{ds.kd_contraindicated_genes?.join(', ')} — {ds.kd_note?.split('(')[0]}</div>
                  <div className="mb-1"><strong style={{ color: COLOR3 }}>OHCbl Trial Mandatory:</strong>{' '}{ds.ohcobl_trial_mandatory}</div>
                  <div className="mb-1"><strong style={{ color: COLOR5 }}>Betaine Rx:</strong>{' '}{ds.betaine_treatment_genes?.join(', ')} (cblC combined MMA+HCU)</div>
                  <div className="mb-1"><strong style={{ color: COLOR5 }}>Glycine Rx:</strong>{' '}{ds.glycine_treatment_genes?.join(', ')} (isovalerylglycine conjugation)</div>
                  <div className="mb-1"><strong style={{ color: COLOR7 }}>Biotin NOT responsive:</strong>{' '}{ds.biotin_NOT_responsive}</div>
                  <div className="mb-1"><strong style={{ color: COLOR4 }}>VPA Risk:</strong>{' '}{ds.vpa_risk}</div>
                  <div className="mb-1"><strong style={{ color: COLOR }}>Emergency Protocol:</strong>{' '}{ds.protein_zero_emergency}</div>
                  <div className="mb-1"><strong>GIR:</strong>{' '}{ds.gir_emergency}</div>
                  <div><strong>Ammonia:</strong>{' '}{ds.ammonia_scavengers}</div>
                </div>
              </div>
            </div>
          </div>

          {/* NBS markers */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              NBS Acylcarnitine Markers — Which Gene Elevates Which Marker
            </div>
            <div className="card-body py-2">
              <div className="row g-2 small">
                {Object.entries(overview?.nbs_markers || {}).filter(([k]) => k !== 'note').map(([marker, genes]) => (
                  <div key={marker} className="col-md-4">
                    <strong style={{ color: COLOR }}>{marker.replace(/_/g, ' ')}:</strong>{' '}
                    {Array.isArray(genes) ? genes.join(', ') : genes}
                  </div>
                ))}
              </div>
              <div className="mt-2 text-muted small">{overview?.nbs_markers?.note}</div>
            </div>
          </div>

          {/* Key teaching points */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Key Teaching Points
            </div>
            <div className="card-body py-2">
              <ol className="small mb-0">
                {(overview?.key_teaching_points || []).map((pt, i) => (
                  <li key={i} className="mb-1">{pt}</li>
                ))}
              </ol>
            </div>
          </div>
        </div>
      )}

      {/* ── GENE TABLE TAB ── */}
      {tab === 'Gene Table' && (
        <div className="table-responsive">
          <table className="table table-sm table-bordered table-hover" style={{ fontSize: '0.75rem' }}>
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr>
                <th>Gene</th><th>Class</th><th>Locus</th><th>aa/kDa</th>
                <th>Enceph%</th><th>NH₃%</th><th>HG%</th><th>Renal%</th><th>DCM%</th><th>Retina%</th>
                <th>KD CI</th><th>OHCbl</th><th>NBS Marker</th>
              </tr>
            </thead>
            <tbody>
              {genes.map(g => (
                <tr key={g.gene}>
                  <td>
                    <span className="badge text-white" style={{ background: CLASS_COLORS[g.gene_class] || COLOR }}>{g.gene}</span>
                    <div className="text-muted" style={{ fontSize: '0.7rem' }}>{g.alias.split('—')[0].trim()}</div>
                  </td>
                  <td style={{ maxWidth: 140 }}><span style={{ fontSize: '0.7rem' }}>{CLASS_LABELS[g.gene_class] || g.gene_class}</span></td>
                  <td>{g.locus}</td>
                  <td>{g.aa} / {g.kDa}</td>
                  <td><strong style={{ color: g.clinical_rates.encephalopathy_pct >= 60 ? COLOR2 : COLOR }}>{g.clinical_rates.encephalopathy_pct}%</strong></td>
                  <td><strong style={{ color: g.clinical_rates.hyperammonemia_pct >= 60 ? COLOR2 : COLOR }}>{g.clinical_rates.hyperammonemia_pct}%</strong></td>
                  <td>{g.clinical_rates.hypoglycaemia_pct}%</td>
                  <td style={{ color: g.clinical_rates.renal_involvement_pct >= 30 ? COLOR2 : 'inherit' }}>{g.clinical_rates.renal_involvement_pct}%</td>
                  <td style={{ color: g.clinical_rates.dcm_pct >= 20 ? COLOR2 : 'inherit' }}>{g.clinical_rates.dcm_pct}%</td>
                  <td style={{ color: g.clinical_rates.retinopathy_pct >= 50 ? COLOR8 : 'inherit' }}>{g.clinical_rates.retinopathy_pct}%</td>
                  <td>
                    {g.kd_contraindicated
                      ? <span className="badge bg-danger">CI</span>
                      : <span className="badge bg-secondary">—</span>}
                  </td>
                  <td>
                    {g.ohcobl_response
                      ? <span className="badge" style={{ background: COLOR3, color: '#fff' }}>Responsive</span>
                      : <span className="badge bg-secondary">—</span>}
                  </td>
                  <td style={{ maxWidth: 120, fontSize: '0.68rem' }}>{g.nbs_marker?.split(';')[0]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── CLINICAL ATLAS TAB ── */}
      {tab === 'Clinical Atlas' && (
        <div className="row g-3">
          {genes.map(g => (
            <div key={g.gene} className="col-12">
              <div className="card shadow-sm">
                <div className="card-header d-flex justify-content-between align-items-center py-2"
                  style={{ background: CLASS_COLORS[g.gene_class] || COLOR, color: '#fff' }}>
                  <span className="fw-bold">{g.gene} — {g.alias.split('—')[0].trim()}</span>
                  <span className="badge bg-white" style={{ color: CLASS_COLORS[g.gene_class] || COLOR }}>
                    {g.locus} · {g.aa}
                  </span>
                </div>
                <div className="card-body py-2 small">
                  <div className="row g-2">
                    <div className="col-md-8">
                      <p className="mb-1"><strong>Disease:</strong> {g.disease}</p>
                      <p className="mb-1"><strong>Hallmarks:</strong> {g.hallmark}</p>
                      <p className="mb-1"><strong>DDx:</strong> {g.key_ddx}</p>
                      <p className="mb-1"><strong>Onset:</strong> {g.onset_pattern}</p>
                      <p className="mb-1"><strong>MRI:</strong> {g.mri_pattern}</p>
                      <p className="mb-1"><strong>Founder variant:</strong> {g.founder_variant}</p>
                      <p className="mb-0"><strong>Acute Rx:</strong> {g.acute_treatment}</p>
                    </div>
                    <div className="col-md-4">
                      <div className="mb-2">
                        <div className="fw-semibold mb-1">Clinical Rates</div>
                        <BarRow label="Encephalopathy"    pct={g.clinical_rates.encephalopathy_pct}    color={CLASS_COLORS[g.gene_class] || COLOR} />
                        <BarRow label="Hyperammonemia"    pct={g.clinical_rates.hyperammonemia_pct}    color={CLASS_COLORS[g.gene_class] || COLOR} />
                        <BarRow label="Hypoglycaemia"     pct={g.clinical_rates.hypoglycaemia_pct}     color={CLASS_COLORS[g.gene_class] || COLOR} />
                        <BarRow label="Lactic Acidosis"   pct={g.clinical_rates.lactic_acidosis_pct}   color={CLASS_COLORS[g.gene_class] || COLOR} />
                        <BarRow label="Neonatal Crisis"   pct={g.clinical_rates.neonatal_crisis_pct}   color={CLASS_COLORS[g.gene_class] || COLOR} />
                        <BarRow label="Renal Involvement" pct={g.clinical_rates.renal_involvement_pct} color={CLASS_COLORS[g.gene_class] || COLOR} />
                        <BarRow label="DCM"               pct={g.clinical_rates.dcm_pct}               color={CLASS_COLORS[g.gene_class] || COLOR} />
                        <BarRow label="Retinopathy"       pct={g.clinical_rates.retinopathy_pct}       color={CLASS_COLORS[g.gene_class] || COLOR} />
                      </div>
                      <div className="d-flex flex-wrap gap-1">
                        <span className="badge" style={{ background: g.kd_contraindicated ? '#b71c1c' : '#388e3c', color: '#fff' }}>
                          {g.kd_contraindicated ? 'KD CI' : 'KD OK'}
                        </span>
                        {g.ohcobl_response && <span className="badge" style={{ background: COLOR3, color: '#fff' }}>OHCbl Resp.</span>}
                        {g.glycine_conjugation && <span className="badge" style={{ background: COLOR5, color: '#fff' }}>Glycine Rx</span>}
                        {g.betaine_treatment && <span className="badge" style={{ background: COLOR5, color: '#fff' }}>Betaine Rx</span>}
                        {g.biotin_response === false && g.gene_class.includes('carboxylase') && (
                          <span className="badge bg-warning text-dark">NOT Biotin-Rx</span>
                        )}
                        {g.nbs_detected && <span className="badge" style={{ background: COLOR6, color: '#fff' }}>NBS</span>}
                      </div>
                      <div className="mt-2 small text-muted">{g.nbs_marker}</div>
                      {g.vpa_risk && (
                        <div className="mt-1 small"><strong style={{ color: COLOR4 }}>VPA:</strong> {g.vpa_risk.split('—')[0]}</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'Definitions' && (
        <div>
          <h6 className="fw-bold mb-3" style={{ color: COLOR }}>
            Clinical Terminology — Organic Acidemia Atlas ({defs?.definitions?.length} terms)
          </h6>
          {(defs?.definitions || []).map((d, i) => (
            <div key={i} className="card shadow-sm mb-2">
              <div className="card-header py-1 fw-bold small" style={{ background: LIGHT, color: COLOR }}>
                {d.term}
              </div>
              <div className="card-body py-2 small">{d.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
