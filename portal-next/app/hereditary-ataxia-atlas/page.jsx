'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  FXN:   '#4a148c',  // deep purple — FRDA, GAA repeat, cardiomyopathy
  APTX:  '#1a237e',  // deep navy — AOA1, hypoalbuminaemia, Japan/Portugal
  SETX:  '#1b5e20',  // deep green — AOA2, AFP hallmark, European
  ATM:   '#b71c1c',  // deep red — A-T, cancer, radiosensitivity
  SACS:  '#006064',  // dark teal — ARSACS, spastic ataxia, Quebec founder
  ANO10: '#37474f',  // dark slate — SCAR10, adult onset, slowly progressive
  ADCK3: '#e65100',  // deep orange — ARCA2, CoQ10 deficiency, elevated CK
  ABHD12:'#880e4f',  // deep crimson — PHARC, polyneuropathy+SNHL+RP+cataract
};

const GENE_DISEASE = {
  FXN:   'Friedreich Ataxia (AR) — FXN; GAA Repeat Expansion; Standard Panels MISS; Cardiomyopathy 80%; Omaveloxolone FDA-2023',
  APTX:  'AOA1 (AR) — Aprataxin; Oculomotor Apraxia; Hypoalbuminaemia; Hypercholesterolaemia; Japan/Portugal Founder',
  SETX:  'AOA2 (AR) — Senataxin; AFP >10 mcg/L Hallmark; No Telangiectasia; Slowly Progressive; European/N.African',
  ATM:   'Ataxia-Telangiectasia (AR) — ATM Kinase; IgA Deficiency; Cancer 35%; Radiosensitivity; Carrier Breast Risk 2×',
  SACS:  'ARSACS (AR) — Sacsin; Spastic Ataxia + UMN Signs; Retinal NFL Hypermyelination; Quebec Founder p.Asp2521Asn',
  ANO10: 'SCAR10 (AR) — Anoctamin-10; Adult Onset 25y; Slowly Progressive; Pure Cerebellar; Vermis Atrophy',
  ADCK3: 'ARCA2 / CoQ10 Deficiency (AR) — ADCK3/COQ8A; Elevated CK; Exercise Intolerance; CoQ10 Trial Mandatory',
  ABHD12:'PHARC (AR) — ABHD12; Polyneuropathy + SNHL + Ataxia + RP + Cataract; Endocannabinoid; Iran/Palestine Founder',
};

const DNA_REPAIR_GENES = ['FXN', 'APTX', 'SETX', 'ATM'];
const MITOCHONDRIAL_GENES = ['SACS', 'ADCK3'];
const ION_LIPID_GENES = ['ANO10', 'ABHD12'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Hereditary Ataxia Atlas…</p>
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
    { key: 'gait_ataxia',                    label: 'Gait Ataxia (FRDA)',          color: '#4a148c' },
    { key: 'hypertrophic_cardiomyopathy',    label: 'Cardiomyopathy (FRDA)',        color: '#4a148c' },
    { key: 'oculomotor_apraxia',             label: 'Oculomotor Apraxia',           color: '#1a237e' },
    { key: 'hypoalbuminaemia',               label: 'Hypoalbuminaemia (AOA1)',       color: '#1a237e' },
    { key: 'elevated_afp',                   label: 'Elevated AFP (AOA2/AT)',        color: '#1b5e20' },
    { key: 'oculocutaneous_telangiectasia',  label: 'Telangiectasia (AT)',           color: '#b71c1c' },
    { key: 'iga_deficiency',                 label: 'IgA Deficiency (AT)',           color: '#b71c1c' },
    { key: 'spastic_ataxia',                 label: 'Spastic Ataxia (ARSACS)',       color: '#006064' },
    { key: 'retinal_nerve_fibre_hypermyelination', label: 'Retinal Hypermyelination', color: '#006064' },
    { key: 'elevated_creatine_kinase',       label: 'Elevated CK (ADCK3)',          color: '#e65100' },
    { key: 'polyneuropathy',                 label: 'Polyneuropathy (PHARC)',        color: '#880e4f' },
    { key: 'sensorineural_hearing_loss',     label: 'SNHL (PHARC)',                 color: '#880e4f' },
  ].filter(item => s[item.key] !== undefined);

  return (
    <div>
      <div className="alert border-0 mb-4" style={{ background: '#ede7f6' }}>
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
      <h6 className="text-uppercase text-muted mb-3 small">8 Hereditary Ataxia Genes Covered</h6>
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
          { label: 'DNA Repair Ataxias', genes: DNA_REPAIR_GENES, color: '#4a148c', desc: 'FRDA · AOA1 · AOA2 · Ataxia-Telangiectasia' },
          { label: 'Mitochondrial Ataxias', genes: MITOCHONDRIAL_GENES, color: '#006064', desc: 'ARSACS (sacsin) · ARCA2 (CoQ10 deficiency)' },
          { label: 'Ion Channel / Lipid Ataxias', genes: ION_LIPID_GENES, color: '#37474f', desc: 'SCAR10 (ANO10) · PHARC (ABHD12 endocannabinoid)' },
        ].map(({ label, genes, color, desc }) => (
          <div key={label} className="col-md-4">
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
      <h6 className="text-uppercase text-muted mb-3 small">Treatment Hierarchy by Disease</h6>
      <div className="row g-3 mb-4">
        {Object.entries(data.treatment_hierarchy || {}).map(([group, steps]) => (
          <div key={group} className="col-md-6">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-header py-2 bg-dark text-white small fw-bold">{group.replace(/_/g, ' / ')}</div>
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
export default function HereditaryAtaxiaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/hereditary-ataxia-atlas`;
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
        <h4 className="mb-0" style={{ color: '#4a148c' }}>🧬 Hereditary Ataxia Atlas</h4>
        <div className="text-muted small">
          8-gene reference · FXN · APTX · SETX · ATM · SACS · ANO10 · ADCK3 · ABHD12
          · FRDA / AOA1 / AOA2 / AT / ARSACS / SCAR10 / ARCA2 / PHARC · 320 patients (8×40, seeds 1390-1397)
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
