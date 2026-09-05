'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  COL5A1: '#1a237e',  // deep navy — Classical EDS-1, AD, atrophic scars
  COL5A2: '#283593',  // medium navy — Classical EDS-2, AD, digenic
  COL3A1: '#b71c1c',  // deep red — Vascular EDS, MOST LETHAL
  TNXB:   '#4a148c',  // deep purple — Classical-like, AR, adrenal screen
  ADAMTS2:'#827717',  // dark amber — Dermatosparaxis, AR, rarest
  PLOD1:  '#1b5e20',  // deep green — kEDS-1, LP:HP pathognomonic
  FKBP14: '#006064',  // dark teal — kEDS-2, normal LP:HP, SNHL
  COL1A2: '#37474f',  // dark slate — Cardiac-Valvular, AR biallelic
};

const GENE_DISEASE = {
  COL5A1: 'Classical EDS Type 1 — Atrophic Scarring + Gorlin Sign (AD)',
  COL5A2: 'Classical EDS Type 2 — Digenic with COL5A1 (AD)',
  COL3A1: 'Vascular EDS — MOST LETHAL, No Elective Surgery, Celiprolol (AD)',
  TNXB:   'Classical-like EDS — Adrenal Insufficiency Screen Mandatory (AR)',
  ADAMTS2:'Dermatosparaxis EDS — Rarest EDS, Sagging Redundant Skin at Birth (AR)',
  PLOD1:  'Kyphoscoliotic EDS Type 1 — Urine LP:HP Ratio Pathognomonic (AR)',
  FKBP14: 'Kyphoscoliotic EDS Type 2 — Normal LP:HP, SNHL 50% (AR)',
  COL1A2: 'Cardiac-Valvular EDS — Biallelic Null, Annual Echo Mandatory (AR)',
};

const CLASSICAL_GENES   = ['COL5A1', 'COL5A2'];
const VASCULAR_GENES    = ['COL3A1'];
const CLASSICAL_LIKE    = ['TNXB'];
const RARE_GENES        = ['ADAMTS2'];
const KYPHOSCOLIOTIC    = ['PLOD1', 'FKBP14'];
const VALVULAR_GENES    = ['COL1A2'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading EDS atlas…</p>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger m-4"><strong>Error:</strong> {msg}</div>;
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-3 mb-3">
      <div className="card h-100 shadow-sm border-0">
        <div className="card-body text-center p-3">
          <div className="fw-bold fs-3" style={{ color: color || '#1a237e' }}>{value}</div>
          <div className="small text-muted">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 px-3 mb-2`} style={{ fontSize: '0.85rem' }}>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const agg = ov.aggregate_stats || {};
  const kpis = ov.kpis || [];
  return (
    <div>
      <h4 className="fw-bold mb-1" style={{ color: '#1a237e' }}>{ov.atlas_name}</h4>
      <p className="text-muted mb-3">{ov.subtitle} · {ov.n_patients} patients · {ov.gene_count} genes · Seeds {ov.seeds}</p>

      {/* KPI row from API */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={ov.n_patients} color="#1a237e" />
        <KPI label="Genes" value={ov.gene_count} color="#880e4f" />
        {kpis.slice(0, 6).map((k, i) => (
          <KPI key={i} label={k.label} value={k.value} color={k.color || '#1a237e'} />
        ))}
      </div>

      {/* Drug alerts */}
      {(ov.drug_alerts || []).length > 0 && (
        <div className="mb-4">
          <h6 className="fw-bold text-danger mb-2">⚠ CRITICAL DRUG / PROCEDURE ALERTS</h6>
          {ov.drug_alerts.map((p, i) => (
            <Alert key={i} text={p} variant="danger" />
          ))}
        </div>
      )}

      {/* Critical rules */}
      {(ov.critical_rules || []).length > 0 && (
        <div className="mb-4">
          <h6 className="fw-bold text-secondary mb-2">KEY CLINICAL RULES</h6>
          {ov.critical_rules.map((p, i) => (
            <Alert key={i} text={p} variant="warning" />
          ))}
        </div>
      )}

      {/* Gene summary table */}
      <h6 className="fw-bold text-secondary mb-2">GENE SUMMARY</h6>
      <div className="table-responsive">
        <table className="table table-bordered table-sm align-middle mb-0" style={{ fontSize: '0.82rem' }}>
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Protein</th><th>Size</th><th>Locus</th>
              <th>Inheritance</th><th>Phenotype</th><th>Hallmark</th>
            </tr>
          </thead>
          <tbody>
            {(ov.gene_summary || []).map(g => (
              <tr key={g.gene}>
                <td><span className="badge" style={{ background: GENE_COLORS[g.gene] || '#333', color: '#fff' }}>{g.gene}</span></td>
                <td className="fw-semibold">{g.protein}</td>
                <td>{g.aa}</td>
                <td><code>{g.locus}</code></td>
                <td>{g.inheritance}</td>
                <td>{g.phenotype_short}</td>
                <td>{g.hallmark_short}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h5 className="fw-bold mb-3" style={{ color: '#1a237e' }}>Per-Gene Cohort Statistics</h5>
      <div className="table-responsive">
        <table className="table table-bordered table-sm align-middle" style={{ fontSize: '0.82rem' }}>
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Disease</th><th>Locus</th><th>aa</th>
              <th>Inheritance</th><th>N</th><th>Seed</th>
              <th>Age Mean</th><th>Female %</th>
            </tr>
          </thead>
          <tbody>
            {data.map(g => (
              <tr key={g.gene}>
                <td>
                  <span className="badge" style={{ background: GENE_COLORS[g.gene] || '#333', color: '#fff' }}>{g.gene}</span>
                </td>
                <td style={{ maxWidth: 200 }}>{GENE_DISEASE[g.gene] || g.gene}</td>
                <td><code>{g.locus}</code></td>
                <td>{g.aa}</td>
                <td>{g.inheritance?.split(';')[0]}</td>
                <td className="text-center fw-bold">{g.cohort_stats?.n}</td>
                <td className="text-center">{g.cohort_stats?.seed}</td>
                <td className="text-center">{g.cohort_stats?.age_mean}</td>
                <td className="text-center">{g.cohort_stats?.female_pct}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Patient sample tables */}
      {data.map(g => (
        <div key={g.gene} className="mb-4">
          <h6 className="fw-bold mt-4 mb-2" style={{ color: GENE_COLORS[g.gene] || '#333' }}>
            {g.gene} — {GENE_DISEASE[g.gene]} ({g.cohort_stats?.n} patients)
          </h6>
          <div className="table-responsive">
            <table className="table table-sm table-bordered" style={{ fontSize: '0.78rem' }}>
              <thead style={{ background: GENE_COLORS[g.gene], color: '#fff' }}>
                <tr>
                  <th>ID</th><th>Age</th><th>Sex</th>
                  {(g.gene === 'COL5A1' || g.gene === 'COL5A2') && <>
                    <th>Atrophic Scars</th><th>Hyperextensibility</th><th>Gorlin Sign</th><th>Beighton ≥5</th><th>Subluxations</th>
                  </>}
                  {g.gene === 'COL3A1' && <>
                    <th>Vascular Event</th><th>Bowel Perf.</th><th>Celiprolol</th><th>CTA Done</th><th>No Elective Sx</th>
                  </>}
                  {g.gene === 'TNXB' && <>
                    <th>Hyperextensibility</th><th>Adrenal Screen</th><th>AI Present</th><th>Beighton ≥5</th><th>No Atrophic Scars</th>
                  </>}
                  {g.gene === 'ADAMTS2' && <>
                    <th>Sagging Skin</th><th>Skin Fragility</th><th>Hyperextensibility</th><th>Bruising</th><th>Beighton ≥5</th>
                  </>}
                  {g.gene === 'PLOD1' && <>
                    <th>LP:HP Elevated</th><th>Scoliosis</th><th>Ocular Fragility</th><th>Beighton ≥5</th><th>Cobb Angle</th>
                  </>}
                  {g.gene === 'FKBP14' && <>
                    <th>LP:HP Normal</th><th>SNHL</th><th>Scoliosis</th><th>Beighton ≥5</th><th>Hearing Aid</th>
                  </>}
                  {g.gene === 'COL1A2' && <>
                    <th>MV Regurgitation</th><th>AV Regurgitation</th><th>Annual Echo</th><th>Surgery Done</th><th>Valve Replace</th>
                  </>}
                </tr>
              </thead>
              <tbody>
                {(g.patients || []).slice(0, 10).map(p => (
                  <tr key={p.patient_id}>
                    <td>{p.patient_id}</td>
                    <td>{p.age_at_dx}</td>
                    <td>{p.sex}</td>
                    {(g.gene === 'COL5A1' || g.gene === 'COL5A2') && <>
                      <td>{p.atrophic_scarring ? '✓' : '—'}</td>
                      <td>{p.skin_hyperextensibility ? '✓' : '—'}</td>
                      <td>{p.gorlin_sign ? '✓' : '—'}</td>
                      <td>{p.beighton_score >= 5 ? '✓' : '—'}</td>
                      <td>{p.joint_subluxations ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'COL3A1' && <>
                      <td>{p.vascular_event ? '✓' : '—'}</td>
                      <td>{p.bowel_perforation ? '✓' : '—'}</td>
                      <td>{p.celiprolol_prescribed ? '✓' : '—'}</td>
                      <td>{p.annual_cta_done ? '✓' : '—'}</td>
                      <td>{p.no_elective_surgery ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'TNXB' && <>
                      <td>{p.skin_hyperextensibility ? '✓' : '—'}</td>
                      <td>{p.adrenal_screen_done ? '✓' : '—'}</td>
                      <td>{p.adrenal_insufficiency ? '✓' : '—'}</td>
                      <td>{p.beighton_score >= 5 ? '✓' : '—'}</td>
                      <td>{!p.atrophic_scarring ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'ADAMTS2' && <>
                      <td>{p.sagging_redundant_skin ? '✓' : '—'}</td>
                      <td>{p.skin_fragility ? '✓' : '—'}</td>
                      <td>{p.skin_hyperextensibility ? '✓' : '—'}</td>
                      <td>{p.bruising_at_birth ? '✓' : '—'}</td>
                      <td>{p.beighton_score >= 5 ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'PLOD1' && <>
                      <td>{p.lp_hp_ratio_elevated ? '✓' : '—'}</td>
                      <td>{p.scoliosis ? '✓' : '—'}</td>
                      <td>{p.ocular_fragility ? '✓' : '—'}</td>
                      <td>{p.beighton_score >= 5 ? '✓' : '—'}</td>
                      <td>{p.cobb_angle ?? '—'}°</td>
                    </>}
                    {g.gene === 'FKBP14' && <>
                      <td>{!p.lp_hp_ratio_elevated ? '✓' : '—'}</td>
                      <td>{p.sensorineural_hearing_loss ? '✓' : '—'}</td>
                      <td>{p.scoliosis ? '✓' : '—'}</td>
                      <td>{p.beighton_score >= 5 ? '✓' : '—'}</td>
                      <td>{p.hearing_aid ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'COL1A2' && <>
                      <td>{p.mitral_valve_regurgitation ? '✓' : '—'}</td>
                      <td>{p.aortic_valve_regurgitation ? '✓' : '—'}</td>
                      <td>{p.annual_echo_done ? '✓' : '—'}</td>
                      <td>{p.valve_surgery_done ? '✓' : '—'}</td>
                      <td>{p.valve_replacement ? '✓' : '—'}</td>
                    </>}
                  </tr>
                ))}
              </tbody>
            </table>
            {(g.patients || []).length > 10 && (
              <p className="text-muted small ms-1">Showing 10 of {g.patients.length} patients</p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      {data.map(g => (
        <div key={g.gene} className="card mb-4 border-0 shadow-sm">
          <div className="card-header d-flex align-items-center" style={{ background: GENE_COLORS[g.gene] || '#333', color: '#fff' }}>
            <span className="fw-bold fs-5 me-3">{g.gene}</span>
            <span>{g.protein} · {g.aa} · {g.locus} · {g.inheritance?.split(';')[0]}</span>
          </div>
          <div className="card-body p-3">
            <div className="row g-3">
              <div className="col-md-6">
                <h6 className="fw-bold text-secondary">GENE CLASS</h6>
                <p style={{ fontSize: '0.82rem' }}>{g.gene_class}</p>
              </div>
              <div className="col-md-6">
                <h6 className="fw-bold text-secondary">PHENOTYPE</h6>
                <p style={{ fontSize: '0.82rem' }}>{g.phenotype}</p>
              </div>
              <div className="col-md-6">
                <h6 className="fw-bold text-secondary">KEY HALLMARKS</h6>
                <ul style={{ fontSize: '0.82rem' }} className="mb-0">
                  {(g.key_hallmarks || []).map((h, i) => <li key={i}>{h}</li>)}
                </ul>
              </div>
              <div className="col-md-6">
                <h6 className="fw-bold text-secondary">TREATMENT ALERTS</h6>
                <ul style={{ fontSize: '0.82rem' }} className="mb-0">
                  {(g.treatment_alerts || []).map((t, i) => <li key={i}>{t}</li>)}
                </ul>
              </div>
              {(g.special_features || []).length > 0 && (
                <div className="col-12">
                  <h6 className="fw-bold text-secondary">SPECIAL FEATURES</h6>
                  <ul style={{ fontSize: '0.82rem' }} className="mb-0">
                    {g.special_features.map((f, i) => <li key={i}>{f}</li>)}
                  </ul>
                </div>
              )}
              <div className="col-12">
                <h6 className="fw-bold text-secondary">KEY DDx</h6>
                <ul style={{ fontSize: '0.82rem' }} className="mb-0">
                  {(g.ddx || []).map((d, i) => <li key={i}>{d}</li>)}
                </ul>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.definitions || [];
  return (
    <div>
      <h5 className="fw-bold mb-3" style={{ color: '#1a237e' }}>Clinical Definitions — EDS Atlas</h5>
      {defs.map((d, i) => (
        <div key={i} className="card mb-3 border-0 shadow-sm">
          <div className="card-body p-3">
            <h6 className="fw-bold mb-1" style={{ color: '#1a237e' }}>{d.term}</h6>
            <p className="text-muted mb-2" style={{ fontSize: '0.85rem' }}><em>{d.short}</em></p>
            <p style={{ fontSize: '0.82rem' }} className="mb-2">{d.detail}</p>
            {d.clinical_rule && (
              <div className="alert alert-warning py-1 px-2 mb-0" style={{ fontSize: '0.8rem' }}>
                <strong>Clinical Rule:</strong> {d.clinical_rule}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function EdsAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/eds-atlas`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  const renderTab = () => {
    switch (tab) {
      case 'Overview':       return <OverviewTab data={overview} />;
      case 'Gene Table':     return <GeneTableTab data={breakdown} />;
      case 'Clinical Atlas': return <ClinicalAtlasTab data={breakdown} />;
      case 'Definitions':    return <DefinitionsTab data={definitions} />;
      default:               return null;
    }
  };

  return (
    <div className="container-fluid py-3 px-4" style={{ maxWidth: 1400 }}>
      {/* Page header */}
      <div className="d-flex align-items-center mb-3">
        <span style={{ fontSize: '2rem', marginRight: 12 }}>🧬</span>
        <div>
          <h3 className="mb-0 fw-bold" style={{ color: '#1a237e' }}>EDS-Atlas</h3>
          <small className="text-muted">
            Complete 8-Gene Ehlers-Danlos Syndromes Atlas ·{' '}
            {Object.entries(GENE_COLORS).map(([g, c]) => (
              <span key={g} className="badge me-1" style={{ background: c, color: '#fff' }}>{g}</span>
            ))}
            · 320 patients (8×40, seeds 1326–1333)
          </small>
        </div>
      </div>

      {/* Category bar */}
      <div className="d-flex gap-2 mb-3 flex-wrap">
        <span className="badge" style={{ background: '#1a237e', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          🔬 Classical EDS: COL5A1 · COL5A2
        </span>
        <span className="badge" style={{ background: '#b71c1c', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          ⚠ Vascular EDS (MOST LETHAL): COL3A1
        </span>
        <span className="badge" style={{ background: '#4a148c', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          🔵 Classical-like: TNXB
        </span>
        <span className="badge" style={{ background: '#1b5e20', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          🟢 Kyphoscoliotic: PLOD1 · FKBP14
        </span>
        <span className="badge" style={{ background: '#827717', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          🟡 Dermatosparaxis: ADAMTS2
        </span>
        <span className="badge" style={{ background: '#37474f', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          ⬛ Cardiac-Valvular: COL1A2
        </span>
      </div>

      {/* Tab navigation */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active fw-semibold' : ''}`}
              onClick={() => setTab(t)}
              style={tab === t ? { color: '#1a237e', borderBottomColor: '#1a237e', borderBottomWidth: 2 } : {}}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {renderTab()}
    </div>
  );
}
