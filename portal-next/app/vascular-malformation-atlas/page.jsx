'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ENG:    '#880e4f',  // deep crimson — HHT1, PAVM, bevacizumab
  ACVRL1: '#1a237e',  // deep navy — HHT2, hepatic AVM, PAH
  SMAD4:  '#1b5e20',  // deep green — HHT-JPS, CRC, aortic root
  KRIT1:  '#4a148c',  // deep purple — CCM1, popcorn lesion, Hispanic founder
  CCM2:   '#006064',  // dark teal — CCM2, malcavernin, de novo 20%
  PDCD10: '#b71c1c',  // deep red — CCM3 most severe, meningioma
  TEK:    '#37474f',  // dark slate — VM, LIC, sirolimus, LMWH
  RASA1:  '#e65100',  // deep orange — CM-AVM, Parkes Weber, never embolize
};

const GENE_DISEASE = {
  ENG:    'HHT1 / Osler-Weber-Rendu (AD) — Endoglin; PAVM Screen Contrast Echo; IV Iron Preferred; Bevacizumab',
  ACVRL1: 'HHT2 (AD) — ALK1; Hepatic AVM > PAVM; PAH 8%; AVOID Hepatic Artery Embolization; Bevacizumab',
  SMAD4:  'HHT3+JPS (AD) — SMAD4 FIRST if HHT + Polyps; CRC 40% by 40; Colonoscopy Age 15; Aortic Root Echo',
  KRIT1:  'CCM1 (AD 2-hit) — KRIT1; Popcorn Lesion SWI/GRE; Hispanic Founder p.Arg51Gln; Annual MRI',
  CCM2:   'CCM2 (AD 2-hit) — Malcavernin; De Novo 20%; Central CCM Complex Scaffold; MRI 1st-degree Relatives',
  PDCD10: 'CCM3 Most Severe (AD 2-hit) — PDCD10; Meningioma Association; Spinal/Cutaneous Cavernomas; Earlier Surgery',
  TEK:    'Venous Malformation (AD/somatic) — TIE2; LIC D-dimer; Sirolimus 0.8mg/m²; LMWH Pre-Procedure; Avoid Warfarin',
  RASA1:  'CM-AVM/Parkes Weber (AD 2-hit) — RASA1; Fast-Flow CM; NEVER Embolize Alone; Limb Overgrowth; Sirolimus',
};

const HHT_GENES = ['ENG', 'ACVRL1', 'SMAD4'];
const CCM_GENES  = ['KRIT1', 'CCM2', 'PDCD10'];
const VM_GENES   = ['TEK'];
const AVM_GENES  = ['RASA1'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Hereditary Vascular Malformation atlas…</p>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger m-4"><strong>Error:</strong> {msg}</div>;
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-sm-4 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-body text-center p-2" style={{ borderTop: `4px solid ${color}` }}>
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function AlertBadge({ text, color = '#b71c1c' }) {
  return (
    <span className="badge me-1 mb-1" style={{ background: color, fontSize: '0.7rem' }}>
      {text}
    </span>
  );
}

/* ── OVERVIEW TAB ── */
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats;

  const statItems = [
    { key: 'epistaxis',                    label: 'Epistaxis (HHT)',           color: '#880e4f' },
    { key: 'pulmonary_avm',                label: 'Pulmonary AVM',             color: '#1a237e' },
    { key: 'hepatic_avm',                  label: 'Hepatic AVM',               color: '#006064' },
    { key: 'cerebral_avm',                 label: 'Cerebral AVM',              color: '#1b5e20' },
    { key: 'gi_telangiectasia',            label: 'GI Telangiectasia',         color: '#4e342e' },
    { key: 'seizures',                     label: 'Seizures (CCM)',             color: '#4a148c' },
    { key: 'intracerebral_haemorrhage',    label: 'ICH (CCM)',                  color: '#b71c1c' },
    { key: 'multiple_lesions',             label: 'Multiple Cavernomas',        color: '#37474f' },
    { key: 'compressible_blue_mass',       label: 'Venous Malformation',       color: '#37474f' },
    { key: 'localized_intravascular_coagulopathy', label: 'LIC (VM)',          color: '#e65100' },
    { key: 'pink_red_capillary_malformation', label: 'CM (RASA1)',             color: '#e65100' },
    { key: 'limb_overgrowth',              label: 'Parkes Weber Overgrowth',   color: '#880e4f' },
  ].filter(item => s[item.key] !== undefined);

  return (
    <div>
      <div className="alert border-0 mb-4" style={{ background: '#fce4ec' }}>
        <h5 className="mb-1">🧬 {data.atlas}</h5>
        <div className="text-muted small">{data.subtitle} · {data.total_patients} patients (8×40, seeds {data.seed_range})</div>
      </div>

      {/* Top Alerts */}
      <div className="alert border-0 mb-4" style={{ background: '#fff3e0' }}>
        <h6 className="mb-2 fw-bold" style={{ color: '#e65100' }}>⚠️ Critical Clinical Alerts</h6>
        {(data.top_alerts || []).map((a, i) => (
          <div key={i} className="d-flex mb-1">
            <span className="me-2" style={{ color: '#b71c1c' }}>▶</span>
            <small><strong>{a.split(':')[0]}:</strong>{a.includes(':') ? a.substring(a.indexOf(':') + 1) : ''}</small>
          </div>
        ))}
      </div>

      {/* Aggregate KPIs */}
      <h6 className="text-uppercase text-muted mb-3 small">Aggregate Cohort Statistics</h6>
      <div className="row g-2 mb-4">
        {statItems.map(({ key, label, color }) => (
          <KPI key={key} label={label} value={`${s[key] ?? 0}%`} color={color} />
        ))}
        <KPI label="Total Genes" value="8" color="#455a64" />
        <KPI label="Total Patients" value={data.total_patients} color="#546e7a" />
      </div>

      {/* Gene badge strip */}
      <h6 className="text-uppercase text-muted mb-3 small">8 Vascular Malformation Genes Covered</h6>
      <div className="mb-4">
        {Object.entries(GENE_DISEASE).map(([gene, disease]) => (
          <div key={gene} className="d-flex align-items-start mb-2">
            <span className="badge me-2 mt-1" style={{ background: GENE_COLORS[gene], minWidth: 72 }}>{gene}</span>
            <small className="text-muted">{disease}</small>
          </div>
        ))}
      </div>

      {/* Group breakdown */}
      <h6 className="text-uppercase text-muted mb-3 small">Disease Group Classification</h6>
      <div className="row g-3 mb-4">
        {[
          { label: 'HHT / Telangiectasia', genes: HHT_GENES, color: '#880e4f', desc: 'Osler-Weber-Rendu + JPS' },
          { label: 'Cerebral Cavernoma (CCM)', genes: CCM_GENES, color: '#4a148c', desc: 'Multiple cavernous malformations' },
          { label: 'Venous Malformation', genes: VM_GENES, color: '#37474f', desc: 'TIE2 · LIC · Sirolimus' },
          { label: 'CM-AVM (RASA1)', genes: AVM_GENES, color: '#e65100', desc: 'Parkes Weber · Never embolize alone' },
        ].map(({ label, genes, color, desc }) => (
          <div key={label} className="col-md-3">
            <div className="card border-0 shadow-sm h-100" style={{ borderTop: `4px solid ${color}` }}>
              <div className="card-body p-3">
                <div className="fw-bold small mb-1" style={{ color }}>{label}</div>
                <div className="text-muted" style={{ fontSize: '0.72rem' }}>{desc}</div>
                <div className="mt-2">
                  {genes.map(g => (
                    <span key={g} className="badge me-1" style={{ background: GENE_COLORS[g], fontSize: '0.65rem' }}>{g}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── GENE TABLE TAB ── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const rows = Object.values(data);

  return (
    <div>
      <h6 className="text-uppercase text-muted mb-3 small">Per-Gene Summary Table</h6>
      <div className="table-responsive">
        <table className="table table-sm table-bordered align-middle" style={{ fontSize: '0.78rem' }}>
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Protein</th><th>aa</th><th>Locus</th>
              <th>Inheritance</th><th>Disease</th><th>n</th><th>Primary Treatment</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(row => (
              <tr key={row.gene}>
                <td><span className="badge" style={{ background: GENE_COLORS[row.gene] }}>{row.gene}</span></td>
                <td>{row.protein}</td>
                <td>{row.aa}</td>
                <td>{row.locus}</td>
                <td>{row.inheritance?.split('—')[0]?.trim()}</td>
                <td className="text-muted small">{GENE_DISEASE[row.gene]?.split(' — ')[0]}</td>
                <td>{row.n_patients}</td>
                <td className="text-muted small">{row.primary_treatment}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Hallmarks per gene */}
      <h6 className="text-uppercase text-muted mt-4 mb-3 small">Clinical Hallmarks &amp; Treatment Alerts</h6>
      <div className="row g-3">
        {rows.map(row => (
          <div key={row.gene} className="col-md-6">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-header py-2" style={{ background: GENE_COLORS[row.gene] }}>
                <span className="text-white fw-bold small">{row.gene}</span>
                <span className="text-white ms-2 opacity-75 small">· {row.protein}</span>
              </div>
              <div className="card-body p-3">
                <div className="mb-2">
                  {(row.hallmarks || []).slice(0, 4).map((h, i) => (
                    <div key={i} className="d-flex mb-1">
                      <span className="me-1 text-muted">•</span>
                      <small>{h}</small>
                    </div>
                  ))}
                </div>
                <hr className="my-2" />
                <div className="small text-muted fw-bold mb-1">Treatment Alerts</div>
                {(row.treatment_alerts || []).slice(0, 3).map((a, i) => (
                  <div key={i} className="d-flex mb-1">
                    <span className="me-1" style={{ color: '#b71c1c' }}>⚠</span>
                    <small>{a}</small>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const rows = Object.values(data);

  return (
    <div>
      <h6 className="text-uppercase text-muted mb-3 small">Clinical Statistics per Gene</h6>
      {rows.map(row => {
        const stats = row.stats || {};
        const statEntries = Object.entries(stats).filter(([, v]) => typeof v === 'number');
        return (
          <div key={row.gene} className="card border-0 shadow-sm mb-4">
            <div className="card-header d-flex align-items-center py-2" style={{ background: GENE_COLORS[row.gene] }}>
              <span className="text-white fw-bold">{row.gene}</span>
              <span className="text-white ms-2 small opacity-75">— {row.organ_system}</span>
            </div>
            <div className="card-body p-3">
              <div className="row g-2 mb-3">
                {statEntries.map(([key, val]) => (
                  <div key={key} className="col-6 col-sm-4 col-md-3">
                    <div className="d-flex justify-content-between border-bottom pb-1">
                      <small className="text-muted text-capitalize">{key.replace(/_/g, ' ')}</small>
                      <small className="fw-bold" style={{ color: GENE_COLORS[row.gene] }}>{val}%</small>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mb-2">
                <span className="small fw-bold text-muted">Aetiology Distribution: </span>
                {(row.etiology_distribution || []).map((e, i) => (
                  <AlertBadge key={i} text={`${Math.round(e.fraction * 100)}% ${e.etiology.split('(')[0].trim()}`} color={GENE_COLORS[row.gene]} />
                ))}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── DEFINITIONS TAB ── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;

  return (
    <div>
      {/* Classification */}
      <h6 className="text-uppercase text-muted mb-3 small">Disease Classification</h6>
      <div className="row g-3 mb-4">
        {Object.entries(data.classification || {}).map(([group, subtypes]) => (
          <div key={group} className="col-md-4">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-header py-2 bg-dark text-white small fw-bold">
                {group.replace(/_/g, ' ').toUpperCase()}
              </div>
              <div className="card-body p-3">
                {Object.entries(subtypes).map(([subtype, desc]) => (
                  <div key={subtype} className="mb-2">
                    <div className="small fw-bold text-muted">{subtype.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: '0.72rem' }} className="text-muted">{desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Key Diagnostic Rules */}
      <h6 className="text-uppercase text-muted mb-3 small">Key Diagnostic Rules</h6>
      <div className="row g-3 mb-4">
        {Object.entries(data.key_diagnostic_rules || {}).map(([key, rule]) => (
          <div key={key} className="col-md-6">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body p-3">
                <div className="fw-bold small mb-1" style={{ color: '#b71c1c' }}>
                  {key.replace(/_/g, ' ')}
                </div>
                <div style={{ fontSize: '0.75rem' }} className="text-muted">{rule}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Treatment Hierarchy */}
      <h6 className="text-uppercase text-muted mb-3 small">Treatment Hierarchy by Disease Group</h6>
      <div className="row g-3 mb-4">
        {Object.entries(data.treatment_hierarchy || {}).map(([group, steps]) => (
          <div key={group} className="col-md-6">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-header py-2 bg-dark text-white small fw-bold">{group}</div>
              <div className="card-body p-2">
                {steps.map((step, i) => (
                  <div key={i} className="d-flex mb-1">
                    <span className="me-2 text-muted small">{i + 1}.</span>
                    <small>{step.replace(/^\d+\.\s*/, '')}</small>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── MAIN PAGE ── */
export default function VascularMalformationAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/vascular-malformation-atlas`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="mb-0" style={{ color: '#880e4f' }}>🩸 Hereditary Vascular Malformation Atlas</h4>
        <div className="text-muted small">
          8-gene reference · ENG · ACVRL1 · SMAD4 · KRIT1 · CCM2 · PDCD10 · TEK · RASA1
          · HHT / CCM / VM / CM-AVM · 320 patients (8×40, seeds 1382-1389)
        </div>
      </div>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active' : ''}`}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'       && <OverviewTab data={overview} />}
      {tab === 'Gene Table'     && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'    && <DefinitionsTab data={definitions} />}
    </div>
  );
}
