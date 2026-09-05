'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ANOS1:  '#1a237e',  // deep navy — Kallmann type 1, XLR, mirror movements
  FGFR1:  '#4a148c',  // deep purple — Kallmann type 2, AD, craniofacial
  GNRHR:  '#01579b',  // deep cerulean — normosmic IHH, AR, pulsatile GnRH
  KISS1R: '#006064',  // deep teal — normosmic IHH, KP54 test
  FMR1:   '#880e4f',  // deep magenta — FXPOI, premutation, female
  FOXL2:  '#bf360c',  // deep orange-brown — BPES, ptosis, POI
  BMP15:  '#558b2f',  // deep green — XLD, oocyte factor, POI
  PROKR2: '#37474f',  // dark slate — Kallmann type 3, sleep/obesity
};

const GENE_DISEASE = {
  ANOS1:  'Kallmann Syndrome Type 1 — Anosmia + HH + Mirror Movements (XLR)',
  FGFR1:  'Kallmann Type 2 + Normosmic CHH Spectrum — Craniofacial (AD)',
  GNRHR:  'Normosmic IHH Type 7 — Pulsatile GnRH Diagnostic + Therapeutic (AR)',
  KISS1R: 'Normosmic IHH Type 15 — Kisspeptin-54 Test Pathognomonic (AR)',
  FMR1:   'FXPOI — Premutation (55-200 CGG) Premature Ovarian Insufficiency',
  FOXL2:  'BPES Type I — Ptosis + Blepharophimosis + Epicanthus Inversus + POI (AD)',
  BMP15:  'POI — Oocyte Paracrine Factor, X-Linked Dominant Haploinsufficiency',
  PROKR2: 'Kallmann Type 3 + Normosmic IHH — Hypersomnia + Obesity Clue (AR/Digenic)',
};

const HH_GENES  = ['ANOS1', 'FGFR1', 'GNRHR', 'KISS1R', 'PROKR2'];
const POI_GENES = ['FMR1', 'FOXL2', 'BMP15'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Reproductive Disorders atlas…</p>
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
  const agg = ov.aggregate_clinical || {};
  const cat = ov.category_summary || {};
  return (
    <div>
      <h4 className="fw-bold mb-1" style={{ color: '#1a237e' }}>{ov.atlas_name}</h4>
      <p className="text-muted mb-3">{ov.subtitle} · {ov.n_patients} patients · {ov.gene_count} genes · Seeds {ov.seeds}</p>

      {/* KPI row */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={ov.n_patients} color="#1a237e" />
        <KPI label="Genes" value={ov.gene_count} color="#880e4f" />
        <KPI label="HH/Kallmann Genes" value={HH_GENES.length} color="#01579b" />
        <KPI label="POI Genes" value={POI_GENES.length} color="#880e4f" />
        <KPI label="Anosmia Rate (atlas)" value={`${cat.anosmia_rate_atlas_wide_pct ?? '—'}%`} color="#1a237e" />
        <KPI label="Mirror Mvmt (ANOS1)" value={`${agg.mirror_movements_anos1_pct ?? '—'}%`} color="#1a237e" />
        <KPI label="FGFR1 Reversal Rate" value={`${agg.spontaneous_reversal_fgfr1_pct ?? '—'}%`} color="#4a148c" />
        <KPI label="GnRH Pump Fertility" value={`${agg.gnrh_pump_fertility_rate_pct ?? '—'}%`} color="#01579b" />
      </div>

      {/* Clinical pearls */}
      {(ov.key_clinical_pearls || []).length > 0 && (
        <div className="mb-4">
          <h6 className="fw-bold text-secondary mb-2">KEY CLINICAL PEARLS</h6>
          {ov.key_clinical_pearls.map((p, i) => (
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
                  {g.gene === 'ANOS1' && <><th>Anosmia</th><th>Mirror Mvmt</th><th>Cryptorchidism</th><th>Abs.OB MRI</th><th>Fertility</th></>}
                  {g.gene === 'FGFR1' && <><th>Anosmia</th><th>Cleft Palate</th><th>Dental Agenesis</th><th>Mirror Mvmt</th><th>Reversal</th></>}
                  {g.gene === 'GNRHR' && <><th>Anosmia</th><th>Partial LOF</th><th>GnRH Pump</th><th>LH Surge</th><th>Fertility</th></>}
                  {g.gene === 'KISS1R' && <><th>Anosmia</th><th>KP54 Test</th><th>LH→KP54</th><th>GnRH Pump</th><th>Fertility</th></>}
                  {g.gene === 'FMR1' && <><th>CGG Repeats</th><th>FSH (IU/L)</th><th>AMH (ng/mL)</th><th>HRT</th><th>Cog. Normal</th></>}
                  {g.gene === 'FOXL2' && <><th>BPES Type</th><th>POI</th><th>FSH (IU/L)</th><th>Ptosis Repair</th><th>Age at Repair</th></>}
                  {g.gene === 'BMP15' && <><th>FSH (IU/L)</th><th>AMH (ng/mL)</th><th>HRT</th><th>Donor Oocyte</th><th>DEXA Done</th></>}
                  {g.gene === 'PROKR2' && <><th>Anosmia</th><th>Hypersomnia</th><th>Obese (BMI>30)</th><th>Digenic</th><th>GnRH Pump</th></>}
                </tr>
              </thead>
              <tbody>
                {(g.patients || []).slice(0, 10).map(p => (
                  <tr key={p.patient_id}>
                    <td>{p.patient_id}</td>
                    <td>{p.age_at_dx}</td>
                    <td>{p.sex}</td>
                    {g.gene === 'ANOS1' && <>
                      <td>{p.anosmia ? '✓' : '—'}</td>
                      <td>{p.mirror_movements ? '✓' : '—'}</td>
                      <td>{p.cryptorchidism ? '✓' : '—'}</td>
                      <td>{p.absent_olfactory_bulbs_mri ? '✓' : '—'}</td>
                      <td>{p.successful_fertility ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'FGFR1' && <>
                      <td>{p.anosmia ? '✓' : '—'}</td>
                      <td>{p.cleft_palate ? '✓' : '—'}</td>
                      <td>{p.dental_agenesis ? '✓' : '—'}</td>
                      <td>{p.mirror_movements ? '✓' : '—'}</td>
                      <td>{p.spontaneous_reversal ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'GNRHR' && <>
                      <td>—</td>
                      <td>{p.partial_lof_fertile_eunuch ? '✓' : '—'}</td>
                      <td>{p.pulsatile_gnrh_therapy ? '✓' : '—'}</td>
                      <td>{p.lh_surge_on_gnrh_pump ? '✓' : '—'}</td>
                      <td>{p.fertility_achieved ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'KISS1R' && <>
                      <td>—</td>
                      <td>{p.kp54_stimulation_test_done ? '✓' : '—'}</td>
                      <td>{p.lh_response_to_kp54 ? '✓' : '✗'}</td>
                      <td>{p.pulsatile_gnrh_therapy ? '✓' : '—'}</td>
                      <td>{p.fertility_achieved ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'FMR1' && <>
                      <td>{p.cgg_repeat_count}</td>
                      <td>{p.fsh_iu_L}</td>
                      <td>{p.amh_ng_mL}</td>
                      <td>{p.hrt_started ? '✓' : '—'}</td>
                      <td>{p.cognitive_normal ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'FOXL2' && <>
                      <td>{p.bpes_type}</td>
                      <td>{p.poi_present ? '✓' : '—'}</td>
                      <td>{p.fsh_iu_L ?? '—'}</td>
                      <td>{p.ptosis_surgery_done ? '✓' : '—'}</td>
                      <td>{p.age_at_ptosis_repair ?? '—'}</td>
                    </>}
                    {g.gene === 'BMP15' && <>
                      <td>{p.fsh_iu_L}</td>
                      <td>{p.amh_ng_mL}</td>
                      <td>{p.hrt_started ? '✓' : '—'}</td>
                      <td>{p.donor_oocyte_ivf ? '✓' : '—'}</td>
                      <td>{p.dexa_done ? '✓' : '—'}</td>
                    </>}
                    {g.gene === 'PROKR2' && <>
                      <td>{p.anosmia ? '✓' : '—'}</td>
                      <td>{p.hypersomnia ? '✓' : '—'}</td>
                      <td>{p.obesity_bmi_over_30 ? '✓' : '—'}</td>
                      <td>{p.digenic_second_variant ? '✓' : '—'}</td>
                      <td>{p.gnrh_pump_done ? '✓' : '—'}</td>
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
      <h5 className="fw-bold mb-3" style={{ color: '#1a237e' }}>Clinical Definitions — Reproductive Disorders Atlas</h5>
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

export default function ReproductiveDisordersAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/reproductive-disorders-atlas`;
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
          <h3 className="mb-0 fw-bold" style={{ color: '#1a237e' }}>Reproductive-Disorders-Atlas</h3>
          <small className="text-muted">
            Complete 8-Gene Hereditary Reproductive Disorders Atlas ·{' '}
            {Object.entries(GENE_COLORS).map(([g, c]) => (
              <span key={g} className="badge me-1" style={{ background: c, color: '#fff' }}>{g}</span>
            ))}
            · 320 patients (8×40, seeds 1318–1325)
          </small>
        </div>
      </div>

      {/* Category bar */}
      <div className="d-flex gap-2 mb-3 flex-wrap">
        <span className="badge" style={{ background: '#1a237e', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          🔬 HH/Kallmann/Normosmic IHH: ANOS1 · FGFR1 · GNRHR · KISS1R · PROKR2
        </span>
        <span className="badge" style={{ background: '#880e4f', color: '#fff', fontSize: '0.8rem', padding: '6px 10px' }}>
          🩺 Premature Ovarian Insufficiency: FMR1 · FOXL2 · BMP15
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
