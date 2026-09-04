'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#4a148c';   // deep purple — mt-tRNA synthetase / aminoacylation
const LIGHT  = '#f3e5f5';
const COLOR2 = '#6a1b9a';   // leukoencephalopathy cluster
const COLOR3 = '#7b1fa2';   // Perrault (SNHL + POI)
const COLOR4 = '#8e24aa';   // epilepsy encephalopathy (VPA CI)
const COLOR5 = '#9c27b0';   // PCH / cerebellar
const COLOR6 = '#ab47bc';   // MLASA
const COLOR7 = '#b71c1c';   // drug CIs / ABSOLUTE
const COLOR8 = '#e65100';   // warnings
const COLOR9 = '#1565c0';   // WES utility

const CLASS_COLORS = {
  leukoencephalopathy: COLOR2,
  perrault: COLOR3,
  epilepsy_encephalopathy: COLOR4,
  pch: COLOR5,
  mlasa: COLOR6,
  multisystem: '#5e35b1',
};

const CLASS_LABELS = {
  leukoencephalopathy: 'Leukoencephalopathy',
  perrault: 'Perrault Syndrome',
  epilepsy_encephalopathy: 'Epilepsy Encephalopathy (VPA CI)',
  pch: 'PCH / Cerebellar',
  mlasa: 'MLASA',
  multisystem: 'Multisystem / Rare',
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

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const agg   = data.aggregate_clinical || {};
  const drug  = data.drug_contraindications || {};
  const wes   = data.wes_utility || {};
  const pheno = data.hallmark_phenotypes || {};
  const genes = data.genes_by_class || {};
  const rules = data.key_rules || {};

  return (
    <>
      {/* Atlas banner */}
      <SectionCard title="mtARS-Deficiency-Atlas — Complete 17-Gene Mitochondrial Aminoacyl-tRNA Synthetase Deficiency Reference">
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">Function: </span>{data.function}</div>
            <div><span className="fw-semibold">Pathway: </span>{data.pathway}</div>
            <div><span className="fw-semibold">Total genes: </span>{data.n_genes} nuclear-encoded (all AR)</div>
            <div><span className="fw-semibold">Cohort: </span>{data.cohort_formula}</div>
            <div className="alert alert-danger py-1 px-2 mt-2 small">
              <strong>CRITICAL:</strong> {rules.vpa_absolute_ci_3_genes}
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
          <KPI label="Leukodystrophy"  value="4"             color={COLOR2} />
          <KPI label="Perrault"        value="2"             color={COLOR3} />
          <KPI label="Epilepsy (VPA!)" value="2"             color={COLOR4} />
          <KPI label="PCH / Cereb."    value="2"             color={COLOR5} />
          <KPI label="MLASA"           value="1"             color={COLOR6} />
          <KPI label="Multisystem"     value="6"             color="#5e35b1" />
          <KPI label="Total Genes"     value={data.n_genes}  color={COLOR}  />
          <KPI label="Total Patients"  value={data.n_patients} color={COLOR} />
        </div>
        <div className="alert alert-warning py-1 px-2 mt-2 small">
          <strong>WES NOTE:</strong> {data.mars2_cnv_caveat}
        </div>
        <div className="alert alert-info py-1 px-2 mt-1 small">
          <strong>BTBGD Exclusion Mandatory:</strong> {rules.btbgd_exclusion_mandatory}
        </div>
      </SectionCard>

      {/* Aggregate phenotype rates */}
      <SectionCard title="Aggregate Clinical Phenotype Rates (680 patients, 17 genes × 40)">
        <div className="row g-2">
          <KPI label="SNHL"               value={`${agg.snhl_pct}%`}               color={COLOR3} />
          <KPI label="POI (females)"      value={`${agg.poi_pct}%`}               color={COLOR3} />
          <KPI label="Epilepsy"           value={`${agg.epilepsy_pct}%`}          color={COLOR4} />
          <KPI label="Ataxia"             value={`${agg.ataxia_pct}%`}            color={COLOR5} />
          <KPI label="Lactic Acidosis"    value={`${agg.lactic_acidosis_pct}%`}   color={COLOR8} />
          <KPI label="HCM"                value={`${agg.hcm_pct}%`}              color={COLOR7} />
          <KPI label="Myopathy"           value={`${agg.myopathy_pct}%`}          color={COLOR6} />
          <KPI label="Periph. Neuropathy" value={`${agg.peripheral_neuropathy_pct}%`} color="#5e35b1" />
        </div>
      </SectionCard>

      {/* Drug contraindications */}
      <SectionCard title="Drug Contraindications — ALL mtARS Diseases" borderColor={COLOR7}>
        <div className="alert alert-danger py-2 px-3 mb-2 small">
          <strong>VPA ABSOLUTE CI in {(data.vpa_absolute_ci_genes || []).join(' + ')}:</strong> {drug.vpa_absolute_ci_genes?.rule}
          <br /><em>Alternative: {drug.vpa_absolute_ci_genes?.alternative}</em>
        </div>
        <div className="row g-2 small">
          <div className="col-12 col-md-6">
            <div className="alert alert-warning py-1 px-2 mb-2">
              <strong>VPA HIGH RISK (all 17):</strong> {drug.vpa_high_risk_all}
            </div>
            <div className="alert alert-danger py-1 px-2 mb-2">
              <strong>Chloramphenicol ABSOLUTE CI:</strong> {drug.chloramphenicol_absolute_ci?.rule}
            </div>
          </div>
          <div className="col-12 col-md-6">
            <div className="alert alert-warning py-1 px-2 mb-2">
              <strong>Aminoglycosides AVOID:</strong> {drug.aminoglycosides_avoid?.rule}
            </div>
            <div className="alert alert-warning py-1 px-2 mb-2">
              <strong>Linezolid AVOID:</strong> {drug.linezolid_avoid}
            </div>
            <div className="alert alert-warning py-1 px-2 mb-2">
              <strong>Metformin AVOID:</strong> {drug.metformin_avoid}
            </div>
            <div className="alert alert-secondary py-1 px-2 mb-2">
              <strong>Statins:</strong> {drug.statins_caution}
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Hallmark phenotypes */}
      <SectionCard title="Hallmark MRI + Clinical Phenotypes by Gene Group" borderColor={COLOR2}>
        {Object.entries(pheno).map(([key, val]) => (
          <div key={key} className="mb-2 small border-bottom pb-2">
            <Badge text={key.replace(/_/g, ' / ')} color={COLOR2} />
            <span className="ms-2">{val}</span>
          </div>
        ))}
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

      {/* WES utility */}
      <SectionCard title="WES / Genetic Diagnostic Utility" borderColor={COLOR9}>
        <div className="row g-2 small">
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">Detects 16/17: </span>{wes.detects_all_17 ? '✅ Yes (16 reliably by WES)' : ''}</div>
            <div><span className="fw-semibold">All nuclear: </span>{wes.nuclear_encoded_all ? '✅ All 17 nuclear-encoded' : ''}</div>
            <div><span className="fw-semibold">MARS2 caveat: </span>{wes.mars2_cnv_note}</div>
          </div>
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">MT-tRNA exclusion: </span>{wes.mt_tRNA_exclusion}</div>
            <div><span className="fw-semibold">MT panel needed? </span>{wes.mt_panel_note}</div>
          </div>
        </div>
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
    .filter(g => (!filter || g.gene.toLowerCase().includes(filter.toLowerCase()) || (g.phenotype_summary || '').toLowerCase().includes(filter.toLowerCase())))
    .filter(g => (!cls || g.gene_class === cls))
    .sort((a, b) => {
      if (sort === 'onset') return a.median_onset_months - b.median_onset_months;
      if (sort === 'epilepsy') return b.epilepsy_pct - a.epilepsy_pct;
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
        <div className="col-md-5 d-flex gap-2 align-items-center">
          <small className="text-muted">Sort:</small>
          {['gene', 'onset', 'epilepsy', 'lactic'].map(s => (
            <button key={s} className={`btn btn-sm ${sort === s ? 'btn-primary' : 'btn-outline-secondary'}`}
              onClick={() => setSort(s)} style={{ fontSize: '0.72rem' }}>
              {s === 'gene' ? 'A–Z' : s === 'onset' ? 'Onset' : s === 'epilepsy' ? 'Epilepsy%' : 'Lactate%'}
            </button>
          ))}
        </div>
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle" style={{ fontSize: '0.78rem' }}>
          <thead className="table-dark">
            <tr>
              <th>Gene</th>
              <th>Class</th>
              <th>tRNA</th>
              <th>Locus</th>
              <th>Syndrome / OMIM</th>
              <th>Onset</th>
              <th>SNHL%</th>
              <th>Epilepsy%</th>
              <th>Ataxia%</th>
              <th>Lactic%</th>
              <th>HCM%</th>
              <th>VPA CI</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(g => (
              <tr key={g.gene}>
                <td><strong style={{ color: COLOR }}>{g.gene}</strong>
                  <div className="text-muted" style={{ fontSize: '0.65rem' }}>{g.alias?.split('/')[0]}</div>
                </td>
                <td><ClassBadge cls={g.gene_class} /></td>
                <td style={{ fontSize: '0.68rem' }}>{g.tRNA_charges?.split('(')[0].trim()}</td>
                <td style={{ fontSize: '0.68rem' }}>{g.locus}</td>
                <td style={{ fontSize: '0.7rem' }}>
                  <div>{g.phenotype_summary?.split('—')[0]}</div>
                  <div className="text-muted">OMIM #{g.disease_omim}</div>
                </td>
                <td style={{ fontSize: '0.68rem' }}>
                  {g.median_onset_months < 12
                    ? `${g.median_onset_months}mo`
                    : `${Math.round(g.median_onset_months / 12)}y`}
                </td>
                <td><span style={{ color: g.snhl_pct > 50 ? COLOR3 : '#555' }}>{g.snhl_pct}%</span></td>
                <td><span style={{ color: g.epilepsy_pct > 60 ? COLOR4 : '#555' }}>{g.epilepsy_pct}%</span></td>
                <td>{g.ataxia_pct}%</td>
                <td><span style={{ color: g.lactic_ac_pct > 70 ? COLOR8 : '#555' }}>{g.lactic_ac_pct}%</span></td>
                <td><span style={{ color: g.hcm_pct > 30 ? COLOR7 : '#555' }}>{g.hcm_pct}%</span></td>
                <td style={{ fontSize: '0.68rem' }}>
                  {g.vpa_ci?.includes('ABSOLUTE') ? (
                    <span className="badge bg-danger">ABSOLUTE CI</span>
                  ) : g.vpa_ci?.includes('HIGH RISK') ? (
                    <span className="badge bg-warning text-dark">HIGH RISK</span>
                  ) : (
                    <span className="badge bg-secondary">AVOID</span>
                  )}
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
                  <div className="col-6"><span className="fw-semibold">Charges: </span>{gene.tRNA_charges}</div>
                  <div className="col-6"><span className="fw-semibold">Locus: </span>{gene.locus}</div>
                  <div className="col-6"><span className="fw-semibold">Size: </span>{gene.aa} / {gene.kDa}</div>
                  <div className="col-6"><span className="fw-semibold">OMIM: </span>#{gene.disease_omim}</div>
                  <div className="col-12"><span className="fw-semibold">Onset: </span>{gene.onset_pattern}</div>
                  <div className="col-12"><span className="fw-semibold">MRI: </span>{gene.mri_pattern}</div>
                </div>
                <div className="alert alert-light py-1 px-2 small"><strong>Phenotype:</strong> {gene.phenotype_summary}</div>
                <div className="alert alert-secondary py-1 px-2 small" style={{ fontSize: '0.75rem' }}>
                  <strong>Hallmark:</strong> {gene.hallmark}
                </div>
                <div className="alert alert-light py-1 px-2 small" style={{ fontSize: '0.75rem' }}>
                  <strong>Key DDx:</strong> {gene.key_ddx}
                </div>
                <div className="alert alert-warning py-1 px-2 small">
                  <strong>VPA:</strong> {gene.vpa_ci}
                </div>
                <div className="small"><strong>Founder variants:</strong> {gene.founder_variant}</div>
              </div>
            </div>
            {/* Phenotype bars */}
            <div className="card shadow-sm">
              <div className="card-body">
                <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Phenotype Rates — {gene.cohort_n} patients</h6>
                {[
                  { label: 'SNHL', val: gene.snhl_pct, c: COLOR3 },
                  { label: 'POI (females)', val: gene.poi_pct, c: COLOR3 },
                  { label: 'Epilepsy', val: gene.epilepsy_pct, c: COLOR4 },
                  { label: 'Ataxia', val: gene.ataxia_pct, c: COLOR5 },
                  { label: 'Spasticity', val: gene.spasticity_pct, c: '#546e7a' },
                  { label: 'Myopathy', val: gene.myopathy_pct, c: COLOR6 },
                  { label: 'Lactic Acidosis', val: gene.lactic_ac_pct, c: COLOR8 },
                  { label: 'HCM', val: gene.hcm_pct, c: COLOR7 },
                  { label: 'Cognitive Impairment', val: gene.cognitive_pct, c: '#37474f' },
                  { label: 'Periph. Neuropathy', val: gene.pn_pct, c: '#5e35b1' },
                  { label: 'Cerebellar Atrophy', val: gene.cerebellar_atrophy_pct, c: COLOR5 },
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
  const terms = data.terms || {};
  const [search, setSearch] = useState('');
  const filtered = Object.entries(terms).filter(([k, v]) =>
    !search || k.toLowerCase().includes(search.toLowerCase()) || v.toLowerCase().includes(search.toLowerCase())
  );
  return (
    <>
      <input className="form-control form-control-sm mb-3" placeholder="Search definitions…"
        value={search} onChange={e => setSearch(e.target.value)} />
      {filtered.map(([k, v]) => (
        <div key={k} className="mb-3 pb-2 border-bottom small">
          <span className="fw-bold" style={{ color: COLOR }}>{k.replace(/_/g, ' ')}</span>
          <p className="mb-0 text-secondary mt-1">{v}</p>
        </div>
      ))}
      <div className="text-muted small">{filtered.length} of {Object.keys(terms).length} definitions shown</div>
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MtARSDeficiencyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/mtars-deficiency-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/mtars-deficiency-atlas/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/mtars-deficiency-atlas/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  if (error) return (
    <div className="container py-4">
      <div className="alert alert-danger">Error loading mtARS atlas: {error}</div>
      <Link href="/" className="btn btn-sm btn-outline-secondary">← Back</Link>
    </div>
  );

  return (
    <div className="container-fluid py-3" style={{ maxWidth: '1400px' }}>
      {/* Header */}
      <div className="d-flex align-items-center mb-3 gap-3">
        <Link href="/" className="btn btn-sm btn-outline-secondary">← Back</Link>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            🧬 mtARS-Deficiency-Atlas
          </h4>
          <div className="text-muted small">
            Complete 17-Gene Mitochondrial Aminoacyl-tRNA Synthetase Deficiency Reference
            · {overview?.n_genes || 17} genes · {overview?.n_patients || 680} patients
            · <span className="fw-semibold" style={{ color: COLOR7 }}>VPA ABSOLUTE CI: FARS2 · PARS2 · VARS2</span>
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
      {tab === 'Overview'      && <OverviewTab      data={overview}    />}
      {tab === 'Gene Table'    && <GeneTableTab     data={breakdown}   />}
      {tab === 'Clinical Atlas'&& <ClinicalAtlasTab data={breakdown}   />}
      {tab === 'Definitions'   && <DefinitionsTab   data={definitions} />}
    </div>
  );
}
