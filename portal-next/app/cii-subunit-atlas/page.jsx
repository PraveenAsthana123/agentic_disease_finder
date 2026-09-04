'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#b71c1c';   // deep red — CII / SDH / oncometabolite pathway
const LIGHT  = '#ffebee';
const COLOR2 = '#1565c0';   // Leigh genes (SDHA, SDHAF1)
const COLOR3 = '#e65100';   // highest malignant risk (SDHB)
const COLOR4 = '#6a1b9a';   // head-neck PGL (SDHC, SDHD, SDHAF2)
const COLOR5 = '#2e7d32';   // assembly factors
const COLOR6 = '#37474f';   // imprinted genes
const COLOR7 = '#880e4f';   // imprinting rule
const COLOR8 = '#c62828';   // danger / CI
const COLOR9 = '#1b5e20';   // safe / note

const CLASS_COLORS = {
  fp_subunit_ar_leigh_ad_pgl5:      '#1565c0',   // SDHA
  ip_subunit_ad_pgl4:               '#e65100',   // SDHB — highest malignant
  cytb_large_ad_pgl3:               '#6a1b9a',   // SDHC
  cytb_small_ad_pgl1_imprinted:     '#880e4f',   // SDHD — imprinted
  assembly_factor_ar_leigh:         '#2e7d32',   // SDHAF1
  assembly_factor_ad_pgl2_imprinted:'#37474f',   // SDHAF2 — imprinted
};

const CLASS_LABELS = {
  fp_subunit_ar_leigh_ad_pgl5:      'Fp subunit — FAD-binding, catalytic (SDHA, AR biallelic=Leigh, AD mono=PGL5)',
  ip_subunit_ad_pgl4:               'Ip subunit — 3 Fe-S clusters (SDHB, AD, PGL4, HIGHEST MALIGNANT RISK)',
  cytb_large_ad_pgl3:               'Cytochrome b560 large anchor (SDHC, AD, PGL3, head-neck + GIST)',
  cytb_small_ad_pgl1_imprinted:     'Cytochrome b560 small anchor, MATERNALLY IMPRINTED (SDHD, PGL1, most common)',
  assembly_factor_ar_leigh:         'Assembly factor — Fe-S chaperone (SDHAF1, AR biallelic = Leigh + leukodystrophy)',
  assembly_factor_ad_pgl2_imprinted:'Assembly factor — FAD insertion, MATERNALLY IMPRINTED (SDHAF2, PGL2, rare)',
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

function RuleCard({ title, text, color = COLOR8 }) {
  return (
    <div className="card mb-2 border-0 shadow-sm">
      <div className="card-body py-2 px-3">
        <div className="fw-semibold small" style={{ color }}>{title}</div>
        <div style={{ fontSize: '0.78rem' }}>{text}</div>
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading overview…</div>;
  const cl = data.aggregate_clinical || {};
  const ci = data.drug_contraindications || {};
  const bs = data.bimodal_spectrum || {};
  const arch = data.complex_architecture || {};
  const ihc = data.ihc_guide || {};
  const imp = data.imprinting_guide || {};
  const kr = data.key_rules || {};
  const wu = data.wes_utility || {};

  return (
    <div>
      {/* Atlas info */}
      <div className="alert mb-3" style={{ backgroundColor: LIGHT, borderColor: COLOR, borderLeft: `4px solid ${COLOR}` }}>
        <div className="fw-bold mb-1" style={{ color: COLOR }}>{data.atlas_subtitle}</div>
        <div style={{ fontSize: '0.83rem' }}>{data.description}</div>
      </div>

      {/* BIMODAL SPECTRUM banner */}
      <div className="alert mb-3 py-2" style={{ backgroundColor: '#fff3e0', borderLeft: `4px solid ${COLOR3}` }}>
        <div className="fw-bold">&#x1f9ec; BIMODAL DISEASE SPECTRUM — Two entirely distinct disease categories</div>
        <div className="row mt-1">
          <div className="col-md-6">
            <div className="small fw-semibold" style={{ color: COLOR2 }}>&#x1f9e0; Biallelic loss → Leigh syndrome / CII deficiency (metabolic)</div>
            <div className="small">{(bs.leigh_genes || []).join(' · ')}</div>
          </div>
          <div className="col-md-6">
            <div className="small fw-semibold" style={{ color: COLOR3 }}>&#x1fa80; Monoallelic haploinsufficiency → PGL/PHEO syndromes (neoplastic)</div>
            <div className="small">{(bs.pgl_genes || []).join(' · ')}</div>
          </div>
        </div>
      </div>

      {/* Imprinting alert */}
      <div className="alert alert-warning mb-3 py-2">
        <div className="fw-bold">&#x1f9ec; MATERNAL IMPRINTING — SDHD (PGL1) + SDHAF2 (PGL2): PATERNAL TRANSMISSION ONLY</div>
        <div className="small mt-1">
          Maternal SDHD/SDHAF2 carriers do NOT develop tumours · Test the father first · Daughters of affected males are at risk
        </div>
      </div>

      {/* IHC banner */}
      <div className="alert alert-info mb-3 py-2">
        <div className="fw-bold">&#x1f52c; SDHB IHC loss = universal SDH-tumour marker (all 6 genotypes) · Order SDHB + SDHA IHC together</div>
        <div className="small mt-1">{ihc.protocol}</div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Genes" value={data.n_genes} color={COLOR} />
        <KPI label="Patients" value={data.n_patients} color={COLOR} />
        <KPI label="Paraganglioma" value={`${cl.paraganglioma_pct}%`} color={COLOR3} />
        <KPI label="PHEO" value={`${cl.pheochromocytoma_pct}%`} color={COLOR3} />
        <KPI label="Leigh (biallelic)" value={`${cl.leigh_encephalopathy_pct}%`} color={COLOR2} />
        <KPI label="Renal/RCC" value={`${cl.renal_rcc_pct}%`} color={COLOR4} />
        <KPI label="Epilepsy (all)" value={`${cl.epilepsy_pct}%`} color={COLOR} />
        <KPI label="HCM (all)" value={`${cl.hcm_pct}%`} color={COLOR8} />
      </div>

      {/* Complex architecture */}
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-header py-1 fw-bold small" style={{ background: LIGHT, color: COLOR }}>CII Complex Architecture</div>
            <div className="card-body py-2">
              {Object.entries(arch).map(([k, v]) => (
                <div key={k} className="mb-1" style={{ fontSize: '0.78rem' }}>
                  <span className="fw-semibold text-capitalize">{k.replace(/_/g, ' ')}: </span>{v}
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          {/* Leigh vs PGL clinical split */}
          <div className="card shadow-sm mb-3">
            <div className="card-header py-1 fw-bold small" style={{ background: '#e3f2fd', color: COLOR2 }}>Leigh Genes (SDHA, SDHAF1) — Aggregate</div>
            <div className="card-body py-2">
              <BarRow label="Encephalopathy" pct={cl.leigh_encephalopathy_pct} color={COLOR2} />
              <BarRow label="Epilepsy" pct={cl.leigh_epilepsy_pct} color={COLOR2} />
              <BarRow label="Lactic Acidosis" pct={cl.leigh_lactic_ac_pct} color={COLOR2} />
              <BarRow label="HCM" pct={cl.hcm_pct} color={COLOR8} />
              <BarRow label="Ataxia" pct={cl.ataxia_pct} color={COLOR2} />
            </div>
          </div>
        </div>
      </div>

      {/* Drug rules */}
      <h6 className="fw-bold mb-2" style={{ color: COLOR }}>Pharmacology Rules (Leigh genes: SDHA biallelic, SDHAF1 biallelic)</h6>
      <div className="row mb-3">
        <div className="col-md-6">
          <RuleCard title="Metformin — ABSOLUTE CI (Leigh genes)" text={ci.metformin_ci} />
          <RuleCard title="KD — CONTRAINDICATED (Leigh genes)" text={ci.kd_ci} />
        </div>
        <div className="col-md-6">
          <RuleCard title="VPA — AVOID (Leigh genes)" text={ci.vpa_ci} />
          <RuleCard title="Propofol — CAUTION / AVOID" text={ci.propofol_avoid} />
        </div>
      </div>

      {/* Imprinting guide */}
      <h6 className="fw-bold mb-2" style={{ color: COLOR7 }}>Maternal Imprinting — SDHD + SDHAF2</h6>
      <div className="row mb-3">
        <div className="col-md-6">
          <RuleCard title="SDHD (PGL1, 11q23.1)" text={imp.sdhd_imprinting} color={COLOR7} />
        </div>
        <div className="col-md-6">
          <RuleCard title="SDHAF2 (PGL2, 11q13.1)" text={imp.sdhaf2_imprinting} color={COLOR7} />
        </div>
      </div>
      <RuleCard title="Clinical Rule for Imprinted Genes" text={imp.clinical_rule} color={COLOR7} />

      {/* IHC guide */}
      <h6 className="fw-bold mb-2 mt-3" style={{ color: COLOR }}>SDHB / SDHA IHC Guide</h6>
      <div className="row mb-3">
        <div className="col-md-6"><RuleCard title="SDHB IHC (universal marker)" text={ihc.sdhb_ihc} /></div>
        <div className="col-md-6"><RuleCard title="SDHA IHC (discriminator)" text={ihc.sdha_ihc} /></div>
      </div>

      {/* Key rules */}
      <h6 className="fw-bold mb-2" style={{ color: COLOR }}>Key Clinical Rules</h6>
      {Object.entries(kr).map(([k, v]) => (
        <RuleCard key={k} title={k.replace(/_/g, ' ').toUpperCase()} text={v} />
      ))}

      {/* BTBGD */}
      <div className="alert mt-3 py-2" style={{ backgroundColor: '#f3e5f5', borderLeft: `4px solid #6a1b9a` }}>
        <div className="fw-bold small">&#x26a0;&#xfe0f; BTBGD Mandatory Exclusion</div>
        <div style={{ fontSize: '0.78rem' }}>{data.btbgd_exclusion}</div>
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data || !data.genes) return <div className="text-muted">Loading gene table…</div>;
  const [sel, setSel] = useState(null);
  const g = sel !== null ? data.genes[sel] : null;

  return (
    <div className="row">
      <div className="col-md-4">
        <div className="list-group">
          {data.genes.map((gene, i) => (
            <button
              key={gene.gene}
              className={`list-group-item list-group-item-action py-2 ${sel === i ? 'active' : ''}`}
              style={sel === i ? { backgroundColor: CLASS_COLORS[gene.gene_class] || COLOR, borderColor: CLASS_COLORS[gene.gene_class] || COLOR } : {}}
              onClick={() => setSel(i)}
            >
              <div className="fw-bold">{gene.gene}</div>
              <div className="small">{gene.aa} · {gene.locus}</div>
              <div style={{ fontSize: '0.7rem' }}>
                {gene.biallelic_leigh && <span className="badge me-1" style={{ background: COLOR2 }}>Leigh</span>}
                {gene.monoallelic_pgl && <span className="badge me-1" style={{ background: COLOR3 }}>PGL</span>}
                {gene.imprinted_maternal && <span className="badge" style={{ background: COLOR7 }}>Imprinted</span>}
              </div>
            </button>
          ))}
        </div>
      </div>
      <div className="col-md-8">
        {!g && <div className="text-muted mt-3">Select a gene to view details</div>}
        {g && (
          <div>
            <div className="d-flex align-items-center mb-2 mt-2">
              <span className="fw-bold fs-5 me-2" style={{ color: CLASS_COLORS[g.gene_class] || COLOR }}>{g.gene}</span>
              <span className="text-muted small">{g.alias}</span>
            </div>
            <div className="mb-1 small"><b>Locus:</b> {g.locus} · <b>OMIM:</b> {g.omim_gene} · <b>Size:</b> {g.aa} ({g.kDa})</div>
            <div className="mb-1 small"><b>Inheritance:</b> {g.inheritance}</div>
            <div className="mb-2 small" style={{ fontSize: '0.78rem' }}>{g.disease}</div>

            <div className="mb-2">
              {g.biallelic_leigh && <span className="badge me-1" style={{ background: COLOR2 }}>Biallelic → Leigh syndrome</span>}
              {g.monoallelic_pgl && <span className="badge me-1" style={{ background: COLOR3 }}>Monoallelic → PGL/PHEO</span>}
              {g.imprinted_maternal && <span className="badge me-1" style={{ background: COLOR7 }}>Maternally Imprinted</span>}
              {g.kd_ci && <span className="badge me-1 bg-danger">KD CI</span>}
              {g.metformin_ci && <span className="badge me-1 bg-danger">Metformin CI</span>}
            </div>

            <div className="small mb-1"><b>PGL type:</b> {g.pgl_type}</div>
            <div className="small mb-1"><b>Malignant risk:</b> {g.malignant_risk_pct}%</div>
            <div className="small mb-1"><b>SDHA IHC lost:</b> {g.sdha_ihc_lost ? 'Yes' : 'No'} · <b>SDHB IHC lost:</b> {g.sdhb_ihc_lost ? 'Yes' : 'No'}</div>
            <div className="small mb-2"><b>VPA:</b> {g.vpa_ci}</div>

            <h6 className="fw-bold mt-2" style={{ fontSize: '0.82rem' }}>Hallmark</h6>
            <div style={{ fontSize: '0.78rem' }} className="mb-2">{g.hallmark}</div>

            <h6 className="fw-bold mt-2" style={{ fontSize: '0.82rem' }}>Key DDx</h6>
            <div style={{ fontSize: '0.78rem' }} className="mb-2">{g.key_ddx}</div>

            <h6 className="fw-bold mt-2" style={{ fontSize: '0.82rem' }}>Founder Variants</h6>
            <div style={{ fontSize: '0.78rem' }} className="mb-2">{g.founder_variant}</div>

            <h6 className="fw-bold mt-2" style={{ fontSize: '0.82rem' }}>Onset Pattern</h6>
            <div style={{ fontSize: '0.78rem' }} className="mb-2">{g.onset_pattern}</div>

            <h6 className="fw-bold mt-2" style={{ fontSize: '0.82rem' }}>MRI / Imaging Pattern</h6>
            <div style={{ fontSize: '0.78rem' }} className="mb-2">{g.mri_pattern}</div>
          </div>
        )}
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data || !data.genes) return <div className="text-muted">Loading clinical atlas…</div>;

  return (
    <div>
      <p className="text-muted small mb-3">Per-gene phenotype rates from 40-patient cohort simulation (seed-stable). Leigh genes (SDHA/SDHAF1): neurological rates dominate. PGL genes (SDHB/C/D/AF2): tumour rates dominate, neurological rates reflect secondary effects.</p>
      <div className="table-responsive">
        <table className="table table-bordered table-sm" style={{ fontSize: '0.75rem' }}>
          <thead>
            <tr style={{ backgroundColor: LIGHT }}>
              <th>Gene</th>
              <th>Phenotype</th>
              <th>PGL%</th>
              <th>PHEO%</th>
              <th>RCC%</th>
              <th>Enceph%</th>
              <th>Epilepsy%</th>
              <th>Lactic%</th>
              <th>HCM%</th>
              <th>Ataxia%</th>
              <th>SNHL%</th>
              <th>Myo%</th>
              <th>Resp%</th>
              <th>Malignant%</th>
            </tr>
          </thead>
          <tbody>
            {data.genes.map(g => (
              <tr key={g.gene}>
                <td>
                  <span className="fw-bold" style={{ color: CLASS_COLORS[g.gene_class] || COLOR }}>{g.gene}</span>
                  {g.imprinted_maternal && <span className="ms-1 badge" style={{ background: COLOR7, fontSize: '0.6rem' }}>Impr</span>}
                </td>
                <td>
                  {g.biallelic_leigh && <span className="badge" style={{ background: COLOR2, fontSize: '0.65rem' }}>Leigh</span>}
                  {g.monoallelic_pgl && <span className="badge" style={{ background: COLOR3, fontSize: '0.65rem' }}>{g.pgl_type.split(' ')[0]}</span>}
                </td>
                <td className="text-center" style={{ color: g.paraganglioma_pct > 40 ? COLOR3 : 'inherit', fontWeight: g.paraganglioma_pct > 40 ? 'bold' : 'normal' }}>{g.paraganglioma_pct}</td>
                <td className="text-center">{g.pheochromocytoma_pct}</td>
                <td className="text-center">{g.rcc_pct}</td>
                <td className="text-center" style={{ color: g.encephalopathy_pct > 60 ? COLOR2 : 'inherit', fontWeight: g.encephalopathy_pct > 60 ? 'bold' : 'normal' }}>{g.encephalopathy_pct}</td>
                <td className="text-center">{g.epilepsy_pct}</td>
                <td className="text-center">{g.lactic_ac_pct}</td>
                <td className="text-center" style={{ color: g.hcm_pct > 40 ? COLOR8 : 'inherit' }}>{g.hcm_pct}</td>
                <td className="text-center">{g.ataxia_pct}</td>
                <td className="text-center">{g.snhl_pct}</td>
                <td className="text-center">{g.myopathy_pct}</td>
                <td className="text-center">{g.respiratory_pct}</td>
                <td className="text-center fw-bold" style={{ color: g.malignant_risk_pct > 20 ? COLOR3 : g.malignant_risk_pct > 0 ? '#e65100' : '#2e7d32' }}>
                  {g.malignant_risk_pct > 0 ? `${g.malignant_risk_pct}%` : 'None'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Gene class legend */}
      <div className="mt-3">
        <div className="fw-bold small mb-2">Gene Class Legend</div>
        <div className="row">
          {Object.entries(CLASS_LABELS).map(([k, v]) => (
            <div key={k} className="col-md-6 mb-1" style={{ fontSize: '0.75rem' }}>
              <span className="me-2" style={{ display: 'inline-block', width: '10px', height: '10px', borderRadius: '50%', backgroundColor: CLASS_COLORS[k] || COLOR }} />
              <b>{k.split('_').slice(0,2).join(' ')}:</b> {v}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading definitions…</div>;
  const defs = Array.isArray(data) ? data : [];
  return (
    <div>
      {defs.map((d, i) => (
        <div key={i} className="card mb-2 border-0 shadow-sm">
          <div className="card-body py-2 px-3">
            <div className="fw-bold small" style={{ color: COLOR }}>{d.term}</div>
            <div style={{ fontSize: '0.78rem' }}>{d.definition}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function CIISubunitAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [ov, bd, df] = await Promise.all([
          fetch(`${API}/api/cii-subunit-atlas/overview`).then(r => r.json()),
          fetch(`${API}/api/cii-subunit-atlas/breakdown`).then(r => r.json()),
          fetch(`${API}/api/cii-subunit-atlas/definitions`).then(r => r.json()),
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

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <Link href="/" className="btn btn-sm btn-outline-secondary me-3">&#8592; Home</Link>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>&#x1f9ec; CII-Subunit-Atlas</h4>
          <div className="text-muted small">Complete 6-Gene Nuclear-Encoded Complex II (Succinate Dehydrogenase) Deficiency Reference · 240 patients · Seeds 807–812</div>
        </div>
      </div>

      {error && <div className="alert alert-danger">Error: {error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'      && <OverviewTab data={overview} />}
      {tab === 'Gene Table'    && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'   && <DefinitionsTab data={definitions} />}
    </div>
  );
}
