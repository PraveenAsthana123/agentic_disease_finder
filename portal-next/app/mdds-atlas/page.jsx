'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#b71c1c';   // deep red — mtDNA depletion / hepatic danger
const LIGHT  = '#ffebee';
const COLOR2 = '#e65100';   // myopathic cluster
const COLOR3 = '#6a1b9a';   // encephalomyopathic cluster
const COLOR4 = '#0277bd';   // MNGIE
const COLOR5 = '#2e7d32';   // PEO/adult-onset
const COLOR6 = '#c62828';   // hepatocerebral POLG
const COLOR7 = '#880e4f';   // special therapies
const COLOR8 = '#e65100';   // warnings
const COLOR9 = '#1565c0';   // WES utility

const CLASS_COLORS = {
  hepatocerebral:       '#b71c1c',
  hepatocerebral_polg:  '#c62828',
  myopathic:            '#e65100',
  encephalomyopathic:   '#6a1b9a',
  mngie:                '#0277bd',
  peo_adult:            '#2e7d32',
};

const CLASS_LABELS = {
  hepatocerebral:       'Hepatocerebral',
  hepatocerebral_polg:  'POLG (Wide Spectrum)',
  myopathic:            'Myopathic',
  encephalomyopathic:   'Encephalomyopathic',
  mngie:                'MNGIE (GI Dysmotility)',
  peo_adult:            'PEO / Adult-Onset',
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

function Badge({ text, color = COLOR }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: '0.72rem' }}>
      {text}
    </span>
  );
}

function ClassBadge({ cls }) {
  return (
    <span className="badge me-1" style={{ backgroundColor: CLASS_COLORS[cls] || COLOR, fontSize: '0.68rem' }}>
      {CLASS_LABELS[cls] || cls}
    </span>
  );
}

function SpecialTherapyBadge({ gene }) {
  if (!gene) return null;
  if (gene.nucleoside_therapy) return (
    <span className="badge bg-success me-1" style={{ fontSize: '0.65rem' }}>Nucleoside Tx (TK2)</span>
  );
  if (gene.sct_curative) return (
    <span className="badge bg-primary me-1" style={{ fontSize: '0.65rem' }}>SCT Curative (MNGIE)</span>
  );
  if (gene.liver_transplant) return (
    <span className="badge bg-warning text-dark me-1" style={{ fontSize: '0.65rem' }}>Liver Tx (Hepatic Only)</span>
  );
  return null;
}

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const agg   = data.aggregate_clinical || {};
  const drug  = data.drug_contraindications || {};
  const wes   = data.wes_utility || {};
  const genes = data.genes_by_class || {};
  const rules = data.key_rules || {};
  const spec  = data.special_therapies || {};

  return (
    <>
      {/* POLG VPA CRITICAL ALERT */}
      <div className="alert alert-danger border-danger shadow-sm mb-4" role="alert"
        style={{ borderLeft: `6px solid ${COLOR}` }}>
        <h6 className="fw-bold mb-1" style={{ color: COLOR }}>
          &#9888; POLG VPA ABSOLUTE CONTRAINDICATION — FATAL ALPERS HEPATOTOXICITY
        </h6>
        <div className="small mb-1">
          <strong>NEVER prescribe VPA without excluding POLG.</strong> {drug.vpa_polg_absolute_ci?.rule}
        </div>
        <div className="small text-muted">
          <strong>Alternative: </strong>{drug.vpa_polg_absolute_ci?.alternative}
        </div>
      </div>

      {/* Atlas banner */}
      <SectionCard title="MDDS-Atlas — Complete 13-Gene mtDNA Depletion Syndrome Atlas">
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">Function: </span>{data.function}</div>
            <div><span className="fw-semibold">Pathway: </span>{data.pathway}</div>
            <div><span className="fw-semibold">Total genes: </span>{data.n_genes} nuclear-encoded</div>
            <div><span className="fw-semibold">Cohort: </span>{data.cohort_formula}</div>
            <div className="alert alert-warning py-1 px-2 mt-2 small">
              <strong>WES NOTE:</strong> {wes.nuclear_genes_detected} — but {wes.mtdna_copy_number_separate_assay}
            </div>
          </div>
          <div className="col-12 col-md-6">
            {Object.entries(genes).map(([cls, glist]) => (
              <div key={cls} className="mb-1">
                <ClassBadge cls={cls} />
                <span className="small ms-1">{(glist || []).join(' · ')}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="row g-2 mt-2">
          <KPI label="Hepatocerebral"    value="4"              color={COLOR}  />
          <KPI label="Myopathic"         value="2"              color={COLOR2} />
          <KPI label="Encephalomyo."     value="3"              color={COLOR3} />
          <KPI label="MNGIE"             value="1"              color={COLOR4} />
          <KPI label="PEO / Adult"       value="3"              color={COLOR5} />
          <KPI label="Total Genes"       value={data.n_genes}   color={COLOR}  />
          <KPI label="Total Patients"    value={data.n_patients} color={COLOR} />
        </div>
        <div className="alert alert-info py-1 px-2 mt-2 small">
          <strong>BTBGD Exclusion Mandatory:</strong> {rules.btbgd_mandatory_exclusion}
        </div>
      </SectionCard>

      {/* mtDNA qPCR note */}
      <SectionCard title="Diagnostic Note — WES vs mtDNA Copy Number" borderColor={COLOR9}>
        <div className="row g-2 small">
          <div className="col-12 col-md-6">
            <div className="alert alert-warning py-2 px-3 mb-2">
              <strong>&#9888; mtDNA qPCR IS A SEPARATE ASSAY FROM WES</strong><br/>
              {wes.mtdna_copy_number_separate_assay}
            </div>
            <div className="small"><span className="fw-semibold">Preferred tissue: </span>{wes.muscle_biopsy_preferred}</div>
          </div>
          <div className="col-12 col-md-6">
            <div className="small mb-1"><span className="fw-semibold">Nuclear genes (WES): </span>{wes.nuclear_genes_detected}</div>
            <div className="small"><span className="fw-semibold">Multiple deletions: </span>{wes.southern_blot_deletions}</div>
          </div>
        </div>
      </SectionCard>

      {/* Aggregate phenotype rates */}
      <SectionCard title="Aggregate Clinical Phenotype Rates (520 patients, 13 genes × 40)">
        <div className="row g-2">
          <KPI label="Hepatopathy"       value={`${agg.hepatopathy_pct}%`}        color={COLOR}  />
          <KPI label="Myopathy"          value={`${agg.myopathy_pct}%`}           color={COLOR2} />
          <KPI label="Encephalopathy"    value={`${agg.encephalopathy_pct}%`}     color={COLOR3} />
          <KPI label="CPEO"              value={`${agg.cpeo_pct}%`}               color={COLOR5} />
          <KPI label="Epilepsy"          value={`${agg.epilepsy_pct}%`}           color={COLOR6} />
          <KPI label="Ataxia"            value={`${agg.ataxia_pct}%`}             color={COLOR5} />
          <KPI label="Lactic Acidosis"   value={`${agg.lactic_acidosis_pct}%`}    color={COLOR8} />
          <KPI label="SNHL"              value={`${agg.snhl_pct}%`}               color={COLOR3} />
          <KPI label="HCM"               value={`${agg.hcm_pct}%`}               color={COLOR}  />
          <KPI label="Renal"             value={`${agg.renal_pct}%`}              color={COLOR9} />
          <KPI label="GI Dysmotility"    value={`${agg.gi_dysmotility_pct}%`}    color={COLOR4} />
          <KPI label="Resp. Failure"     value={`${agg.respiratory_failure_pct}%`} color={COLOR8} />
        </div>
      </SectionCard>

      {/* Special therapies */}
      <SectionCard title="Special Disease-Modifying Therapies" borderColor={COLOR7}>
        <div className="row g-3 small">
          <div className="col-12 col-md-4">
            <div className="card h-100" style={{ borderTop: `3px solid #2e7d32` }}>
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: '#2e7d32' }}>TK2 — Nucleoside Supplementation</h6>
                <div><span className="fw-semibold">Therapy: </span>{spec.nucleoside_supplementation_tk2?.therapy}</div>
                <div><span className="fw-semibold">Mechanism: </span>{spec.nucleoside_supplementation_tk2?.mechanism}</div>
                <div className="text-muted"><em>{spec.nucleoside_supplementation_tk2?.status}</em></div>
              </div>
            </div>
          </div>
          <div className="col-12 col-md-4">
            <div className="card h-100" style={{ borderTop: `3px solid ${COLOR4}` }}>
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: COLOR4 }}>TYMP/MNGIE — Allogeneic SCT</h6>
                <div><span className="fw-semibold">Therapy: </span>{spec.sct_curative_mngie?.therapy}</div>
                <div><span className="fw-semibold">Mechanism: </span>{spec.sct_curative_mngie?.mechanism}</div>
                <div className="text-muted"><em>{spec.sct_curative_mngie?.status}</em></div>
              </div>
            </div>
          </div>
          <div className="col-12 col-md-4">
            <div className="card h-100" style={{ borderTop: `3px solid ${COLOR8}` }}>
              <div className="card-body">
                <h6 className="fw-bold" style={{ color: COLOR8 }}>DGUOK + MPV17 — Liver Transplant</h6>
                <div><span className="fw-semibold">Genes: </span>{(spec.liver_transplant_hepatocerebral?.genes || []).join(', ')}</div>
                <div className="alert alert-warning py-1 px-2 mt-1 small">
                  <strong>CAVEAT: </strong>{spec.liver_transplant_hepatocerebral?.caveat}
                </div>
                <div className="text-muted"><em>{spec.liver_transplant_hepatocerebral?.status}</em></div>
              </div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Drug contraindications */}
      <SectionCard title="Drug Contraindications — ALL MDDS Diseases" borderColor={COLOR}>
        <div className="alert alert-danger py-2 px-3 mb-2 small">
          <strong>VPA ABSOLUTE CI in POLG:</strong> {drug.vpa_polg_absolute_ci?.rule}
          <br /><em>Alternative: {drug.vpa_polg_absolute_ci?.alternative}</em>
        </div>
        <div className="row g-2 small">
          <div className="col-12 col-md-6">
            <div className="alert alert-warning py-1 px-2 mb-2">
              <strong>VPA HIGH RISK (all 12 other):</strong> {drug.vpa_all_mdds_high_risk}
            </div>
            <div className="alert alert-danger py-1 px-2 mb-2">
              <strong>Metformin ABSOLUTE CI:</strong> {drug.metformin_absolute_ci?.rule}
            </div>
            <div className="alert alert-warning py-1 px-2 mb-2">
              <strong>KD in POLG:</strong> {drug.kd_polg_contraindicated}
            </div>
          </div>
          <div className="col-12 col-md-6">
            <div className="alert alert-warning py-1 px-2 mb-2">
              <strong>Linezolid AVOID:</strong> {drug.linezolid_avoid}
            </div>
            <div className="alert alert-warning py-1 px-2 mb-2">
              <strong>Chloramphenicol AVOID:</strong> {drug.chloramphenicol_avoid}
            </div>
            <div className="alert alert-warning py-1 px-2 mb-2">
              <strong>Aminoglycosides AVOID:</strong> {drug.aminoglycosides_avoid?.rule}
            </div>
            <div className="alert alert-secondary py-1 px-2 mb-2">
              <strong>Metronidazole (MNGIE only):</strong> {drug.metronidazole_tymp_avoid}
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Key clinical rules */}
      <SectionCard title="Key Clinical Rules" borderColor={COLOR8}>
        {Object.entries(rules).map(([k, v]) => (
          <div key={k} className="mb-2 small border-bottom pb-1">
            <span className="fw-semibold" style={{ color: COLOR8 }}>{k.replace(/_/g, ' ')}:</span>
            <span className="ms-2">{v}</span>
          </div>
        ))}
      </SectionCard>

      {/* Gene notes */}
      <SectionCard title="Gene-Specific Clinical Notes" borderColor={COLOR6}>
        {[
          { label: 'POLG', note: data.polg_note, color: COLOR6 },
          { label: 'TYMP', note: data.tymp_note, color: COLOR4 },
          { label: 'TK2', note: data.tk2_note, color: '#2e7d32' },
          { label: 'DGUOK', note: data.dguok_note, color: COLOR },
        ].map(({ label, note, color }) => (
          <div key={label} className="mb-2 small border-bottom pb-2">
            <Badge text={label} color={color} />
            <span className="ms-2">{note}</span>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Gene Table ────────────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const genes = data.genes || [];
  const [sort, setSort] = useState('gene');
  const [filter, setFilter] = useState('');
  const [cls, setCls] = useState('');

  const sorted = [...genes]
    .filter(g => (!filter || g.gene.toLowerCase().includes(filter.toLowerCase()) || (g.phenotype || '').toLowerCase().includes(filter.toLowerCase())))
    .filter(g => (!cls || g.gene_class === cls))
    .sort((a, b) => {
      if (sort === 'hepatopathy') return b.hepatopathy_pct - a.hepatopathy_pct;
      if (sort === 'myopathy') return b.myopathy_pct - a.myopathy_pct;
      if (sort === 'lactic') return b.lactic_ac_pct - a.lactic_ac_pct;
      return a.gene.localeCompare(b.gene);
    });

  const classes = [...new Set(genes.map(g => g.gene_class))];

  return (
    <>
      <div className="row g-2 mb-3">
        <div className="col-md-4">
          <input className="form-control form-control-sm" placeholder="Filter gene / phenotype…"
            value={filter} onChange={e => setFilter(e.target.value)} />
        </div>
        <div className="col-md-3">
          <select className="form-select form-select-sm" value={cls} onChange={e => setCls(e.target.value)}>
            <option value="">All classes</option>
            {classes.map(c => <option key={c} value={c}>{CLASS_LABELS[c] || c}</option>)}
          </select>
        </div>
        <div className="col-md-5 d-flex gap-2 align-items-center flex-wrap">
          <small className="text-muted">Sort:</small>
          {['gene', 'hepatopathy', 'myopathy', 'lactic'].map(s => (
            <button key={s} className={`btn btn-sm ${sort === s ? 'btn-danger' : 'btn-outline-secondary'}`}
              onClick={() => setSort(s)} style={{ fontSize: '0.72rem' }}>
              {s === 'gene' ? 'A–Z' : s === 'hepatopathy' ? 'Hepatopathy%' : s === 'myopathy' ? 'Myopathy%' : 'Lactate%'}
            </button>
          ))}
        </div>
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle" style={{ fontSize: '0.78rem' }}>
          <thead className="table-dark">
            <tr>
              <th>Gene</th>
              <th>Cluster</th>
              <th>Locus</th>
              <th>Key Phenotype</th>
              <th>OMIM</th>
              <th>Onset</th>
              <th>Hepato%</th>
              <th>Myo%</th>
              <th>CPEO%</th>
              <th>Lactate%</th>
              <th>VPA CI</th>
              <th>Special Rx</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(g => (
              <tr key={g.gene}>
                <td><strong style={{ color: CLASS_COLORS[g.gene_class] || COLOR }}>{g.gene}</strong>
                  <div className="text-muted" style={{ fontSize: '0.62rem' }}>{g.alias?.split('/')[0]}</div>
                </td>
                <td><ClassBadge cls={g.gene_class} /></td>
                <td style={{ fontSize: '0.68rem' }}>{g.locus}</td>
                <td style={{ fontSize: '0.7rem', maxWidth: '160px' }}>
                  <div className="text-truncate">{g.phenotype?.split('—')[0]?.split('—')[0]?.substring(0, 60)}</div>
                </td>
                <td style={{ fontSize: '0.68rem' }}>#{g.omim_gene}</td>
                <td style={{ fontSize: '0.68rem', maxWidth: '80px' }}>
                  <div className="text-truncate">{g.onset_pattern?.substring(0, 40)}</div>
                </td>
                <td><span style={{ color: g.hepatopathy_pct > 50 ? COLOR : '#555' }}>{g.hepatopathy_pct}%</span></td>
                <td><span style={{ color: g.myopathy_pct > 60 ? COLOR2 : '#555' }}>{g.myopathy_pct}%</span></td>
                <td>{g.cpeo_pct}%</td>
                <td><span style={{ color: g.lactic_ac_pct > 70 ? COLOR8 : '#555' }}>{g.lactic_ac_pct}%</span></td>
                <td style={{ fontSize: '0.68rem' }}>
                  {g.vpa_ci?.includes('ABSOLUTE') ? (
                    <span className="badge bg-danger">ABSOLUTE CI</span>
                  ) : g.vpa_ci?.includes('HIGH RISK') ? (
                    <span className="badge bg-warning text-dark">HIGH RISK</span>
                  ) : (
                    <span className="badge bg-secondary">AVOID</span>
                  )}
                </td>
                <td>
                  <SpecialTherapyBadge gene={g} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-muted small mt-1">Showing {sorted.length} of {genes.length} genes</div>
    </>
  );
}

// ── Tab: Clinical Atlas ────────────────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const genes = data.genes || [];
  const [selected, setSelected] = useState(null);
  const gene = selected ? genes.find(g => g.gene === selected) : null;

  return (
    <div className="row g-3">
      <div className="col-md-3">
        <div className="list-group" style={{ maxHeight: '80vh', overflowY: 'auto' }}>
          {genes.map(g => (
            <button key={g.gene}
              className={`list-group-item list-group-item-action py-1 px-2 ${selected === g.gene ? 'active' : ''}`}
              style={{ fontSize: '0.78rem', borderLeft: `4px solid ${CLASS_COLORS[g.gene_class] || COLOR}` }}
              onClick={() => setSelected(g.gene)}>
              <strong>{g.gene}</strong>
              <div className="opacity-75" style={{ fontSize: '0.65rem' }}>{CLASS_LABELS[g.gene_class] || g.gene_class}</div>
              {g.nucleoside_therapy && <span className="badge bg-success" style={{ fontSize: '0.55rem' }}>Nucleoside Rx</span>}
              {g.sct_curative && <span className="badge bg-primary" style={{ fontSize: '0.55rem' }}>SCT</span>}
              {g.liver_transplant && <span className="badge bg-warning text-dark" style={{ fontSize: '0.55rem' }}>Liver Tx</span>}
            </button>
          ))}
        </div>
      </div>
      <div className="col-md-9">
        {!gene ? (
          <div className="alert alert-info">Select a gene from the list to view the clinical atlas.</div>
        ) : (
          <>
            <div className="card mb-3 shadow-sm" style={{ borderTop: `4px solid ${CLASS_COLORS[gene.gene_class] || COLOR}` }}>
              <div className="card-body">
                <h5 className="fw-bold" style={{ color: CLASS_COLORS[gene.gene_class] || COLOR }}>
                  {gene.gene} — {gene.alias?.split('/')[0]}
                </h5>
                <div className="row g-2 small mb-2">
                  <div className="col-6"><span className="fw-semibold">Size: </span>{gene.aa} / {gene.kDa}</div>
                  <div className="col-6"><span className="fw-semibold">Locus: </span>{gene.locus}</div>
                  <div className="col-6"><span className="fw-semibold">OMIM: </span>#{gene.omim_gene}</div>
                  <div className="col-6"><span className="fw-semibold">Inheritance: </span>{gene.inheritance}</div>
                  <div className="col-12"><span className="fw-semibold">Onset: </span>{gene.onset_pattern}</div>
                  <div className="col-12"><span className="fw-semibold">MRI: </span>{gene.mri_pattern}</div>
                </div>
                <div className="alert alert-light py-1 px-2 small"><strong>Phenotype:</strong> {gene.phenotype}</div>
                <div className="alert alert-secondary py-1 px-2 small" style={{ fontSize: '0.75rem' }}>
                  <strong>Hallmark:</strong> {gene.hallmark}
                </div>
                <div className="alert alert-light py-1 px-2 small" style={{ fontSize: '0.75rem' }}>
                  <strong>Key DDx:</strong> {gene.key_ddx}
                </div>
                {gene.vpa_ci?.includes('ABSOLUTE') ? (
                  <div className="alert alert-danger py-1 px-2 small fw-bold">
                    &#9888; VPA: {gene.vpa_ci}
                  </div>
                ) : (
                  <div className="alert alert-warning py-1 px-2 small">
                    <strong>VPA:</strong> {gene.vpa_ci}
                  </div>
                )}
                {gene.nucleoside_therapy && (
                  <div className="alert alert-success py-1 px-2 small">
                    <strong>&#10003; Nucleoside Supplementation Therapy available</strong> — thymidine + deoxycytidine oral bypass
                  </div>
                )}
                {gene.sct_curative && (
                  <div className="alert alert-primary py-1 px-2 small">
                    <strong>&#10003; Allogeneic SCT — potentially curative</strong> (MNGIE: restores TP enzyme in myeloid cells)
                  </div>
                )}
                {gene.liver_transplant && (
                  <div className="alert alert-warning py-1 px-2 small">
                    <strong>Liver Transplant available</strong> — corrects hepatic disease ONLY; neurological disease continues
                  </div>
                )}
                <div className="small mt-2"><strong>Founder variants:</strong> {gene.founder_variant}</div>
              </div>
            </div>
            {/* Phenotype bars */}
            <div className="card shadow-sm">
              <div className="card-body">
                <h6 className="fw-bold mb-3" style={{ color: CLASS_COLORS[gene.gene_class] || COLOR }}>
                  Phenotype Rates — {gene.cohort_n} patients
                </h6>
                {[
                  { label: 'Hepatopathy', val: gene.hepatopathy_pct, c: COLOR },
                  { label: 'Myopathy', val: gene.myopathy_pct, c: COLOR2 },
                  { label: 'Encephalopathy', val: gene.encephalopathy_pct, c: COLOR3 },
                  { label: 'CPEO', val: gene.cpeo_pct, c: COLOR5 },
                  { label: 'Epilepsy', val: gene.epilepsy_pct, c: COLOR6 },
                  { label: 'Ataxia', val: gene.ataxia_pct, c: COLOR5 },
                  { label: 'Lactic Acidosis', val: gene.lactic_ac_pct, c: COLOR8 },
                  { label: 'SNHL', val: gene.snhl_pct, c: COLOR3 },
                  { label: 'HCM', val: gene.hcm_pct, c: COLOR },
                  { label: 'Renal', val: gene.renal_pct, c: COLOR9 },
                  { label: 'GI Dysmotility', val: gene.gi_pct, c: COLOR4 },
                  { label: 'Cognitive', val: gene.cognitive_pct, c: '#37474f' },
                  { label: 'Respiratory Failure', val: gene.respiratory_pct, c: COLOR8 },
                ].map(({ label, val, c }) => (
                  <div key={label} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{label}</span><span className="fw-semibold">{val}%</span>
                    </div>
                    <div className="progress" style={{ height: '8px' }}>
                      <div className="progress-bar" style={{ width: `${val}%`, backgroundColor: c }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const terms = data.terms || data || [];
  const [search, setSearch] = useState('');

  // Support both array of {term, definition} and object {key: value}
  const entries = Array.isArray(terms)
    ? terms.map(t => [t.term, t.definition])
    : Object.entries(terms);

  const filtered = entries.filter(([k, v]) =>
    !search || k.toLowerCase().includes(search.toLowerCase()) || v.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <>
      <input className="form-control form-control-sm mb-3" placeholder="Search definitions…"
        value={search} onChange={e => setSearch(e.target.value)} />
      {filtered.map(([k, v]) => (
        <div key={k} className="mb-3 pb-2 border-bottom small">
          <span className="fw-bold" style={{ color: COLOR }}>{k}</span>
          <p className="mb-0 text-secondary mt-1">{v}</p>
        </div>
      ))}
      <div className="text-muted small">{filtered.length} of {entries.length} definitions shown</div>
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MDDSAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/mdds-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/mdds-atlas/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/mdds-atlas/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  if (error) return (
    <div className="container py-4">
      <div className="alert alert-danger">Error loading MDDS atlas: {error}</div>
      <Link href="/" className="btn btn-sm btn-outline-secondary">&#8592; Back</Link>
    </div>
  );

  return (
    <div className="container-fluid py-3" style={{ maxWidth: '1400px' }}>
      {/* Header */}
      <div className="d-flex align-items-center mb-3 gap-3">
        <Link href="/" className="btn btn-sm btn-outline-secondary">&#8592; Back</Link>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            &#x1f9ec; MDDS-Atlas
          </h4>
          <div className="text-muted small">
            Complete 13-Gene mtDNA Depletion Syndrome Atlas
            &middot; {overview?.n_genes || 13} genes &middot; {overview?.n_patients || 520} patients
            &middot; <span className="fw-semibold" style={{ color: COLOR }}>VPA ABSOLUTE CI: POLG (Alpers Hepatotoxicity)</span>
            &middot; <span className="fw-semibold" style={{ color: '#0277bd' }}>Nucleoside Rx: TK2 &middot; SCT Curative: TYMP</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active fw-bold' : ''}`}
              style={tab === t ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* Tab content */}
      {tab === 'Overview'       && <OverviewTab      data={overview}    />}
      {tab === 'Gene Table'     && <GeneTableTab     data={breakdown}   />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown}   />}
      {tab === 'Definitions'    && <DefinitionsTab   data={definitions} />}
    </div>
  );
}
