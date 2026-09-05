'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  SPTLC1:  '#1a237e',  // deep navy — HSAN1A, deoxySL toxicity, L-serine, AVOID vincristine
  ELP1:    '#880e4f',  // deep crimson — HSAN3 FD, Ashkenazi, TUDCA, autonomic crises
  NTRK1:   '#b71c1c',  // deep red — HSAN4 CIPA, hyperthermia kills, anhidrosis
  NGFB:    '#1b5e20',  // deep green — HSAN5, selective deep pain loss, Norwegian founder
  FAM134B: '#4e342e',  // deep brown — HSAN2B, ER-reticulophagy, neonatal mutilations
  DNMT1:   '#006064',  // dark teal — HSAN1E, sensory + SNHL + dementia triad
  WNK1:    '#37474f',  // dark slate — HSAN2A, HSN2-exon, standard panels MISS it
  PRDM12:  '#4a148c',  // deep purple — HSAN8, pain insensitivity, sweating PRESERVED
};

const GENE_DISEASE = {
  SPTLC1:  'HSAN1A (AD) — Serine Palmitoyltransferase; Deoxy-SL Toxicity; L-serine Rx; VINCRISTINE CI',
  ELP1:    'HSAN3/FD (AR) — Familial Dysautonomia; Ashkenazi c.2204+6T>C >99.5%; TUDCA; Autonomic Crisis',
  NTRK1:   'HSAN4/CIPA (AR) — TrkA Receptor; Pain + Anhidrosis; HYPERTHERMIA KILLS; Dental/Charcot',
  NGFB:    'HSAN5 (AR) — NGF Beta; Selective Deep Pain Loss; Superficial Touch PRESERVED; Norwegian R221W',
  FAM134B: 'HSAN2B (AR) — RETREG1; ER-Reticulophagy; Neonatal Onset; Self-Mutilation; Most Severe HSAN2',
  DNMT1:   'HSAN1E (AD) — DNA Methyltransferase 1; TRIAD: Sensory + SNHL + Dementia; RFTS Domain',
  WNK1:    'HSAN2A (AR) — HSN2 Isoform; STANDARD PANELS MISS HSN2-EXON; Pan-Modal Sensory Loss',
  PRDM12:  'HSAN8 (AR) — PR Domain TF; Pain Insensitivity; SWEATING PRESERVED (DDx from CIPA)',
};

const AD_GENES     = ['SPTLC1', 'DNMT1'];
const AR_GENES     = ['ELP1', 'NTRK1', 'NGFB', 'FAM134B', 'WNK1', 'PRDM12'];
const ANHIDROTIC   = ['NTRK1', 'FAM134B'];
const SWEATING_OK  = ['NGFB', 'PRDM12'];
const ASHKENAZI    = ['ELP1'];
const L_SERINE_GENE = ['SPTLC1'];
const TUDCA_GENE   = ['ELP1'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Hereditary Sensory & Autonomic Neuropathy atlas…</p>
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

  return (
    <div>
      <div className="alert border-0 mb-4" style={{ background: '#e8eaf6' }}>
        <h5 className="mb-1">🧬 {data.atlas}</h5>
        <div className="text-muted small">{data.subtitle} · {data.total_patients} patients (8×40, seeds {data.seed_range})</div>
      </div>

      {/* Aggregate KPIs */}
      <h6 className="text-uppercase text-muted mb-3 small">Aggregate Cohort Statistics</h6>
      <div className="row g-2 mb-4">
        <KPI label="Sensory Loss" value={`${s.sensory_loss_pct}%`} color="#1a237e" />
        <KPI label="Autonomic Dysfn" value={`${s.autonomic_dysfunction_pct}%`} color="#880e4f" />
        <KPI label="Pain Insensitivity" value={`${s.pain_insensitivity_pct}%`} color="#b71c1c" />
        <KPI label="Anhidrosis" value={`${s.anhidrosis_pct}%`} color="#006064" />
        <KPI label="Plantar Ulcers" value={`${s.plantar_ulcers_pct}%`} color="#37474f" />
        <KPI label="Self-Mutilations" value={`${s.mutilations_pct}%`} color="#4e342e" />
        <KPI label="Cognitive Decline" value={`${s.cognitive_decline_pct}%`} color="#1b5e20" />
        <KPI label="Hearing Loss (SNHL)" value={`${s.hearing_loss_pct}%`} color="#4a148c" />
        <KPI label="Hyperthermia" value={`${s.hyperthermia_pct}%`} color="#bf360c" />
        <KPI label="GI Dysmotility" value={`${s.gi_dysmotility_pct}%`} color="#424242" />
        <KPI label="Total Genes" value="8" color="#455a64" />
        <KPI label="Total Patients" value={data.total_patients} color="#546e7a" />
      </div>

      {/* Key DDx Anchors */}
      <h6 className="text-uppercase text-muted mb-2 small">Key Clinical DDx Anchors</h6>
      <div className="card border-0 shadow-sm mb-4">
        <div className="card-body p-3">
          {data.key_ddx_anchor.map((k, i) => (
            <div key={i} className="d-flex align-items-start mb-2">
              <span className="me-2 mt-1" style={{ color: '#b71c1c', fontWeight: 'bold' }}>▶</span>
              <small>{k}</small>
            </div>
          ))}
        </div>
      </div>

      {/* Gene Cards */}
      <h6 className="text-uppercase text-muted mb-3 small">Gene Summary</h6>
      <div className="row g-3">
        {data.genes_summary.map((g) => (
          <div key={g.gene} className="col-12 col-md-6">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-header text-white py-2 px-3" style={{ background: GENE_COLORS[g.gene] }}>
                <div className="d-flex justify-content-between align-items-center">
                  <span className="fw-bold">{g.gene}</span>
                  <span className="small opacity-75">{g.locus} · {g.aa.split(' ')[0]} aa</span>
                </div>
                <div className="small opacity-90">{GENE_DISEASE[g.gene]?.split('—')[0].trim()}</div>
              </div>
              <div className="card-body p-3">
                <div className="row g-1 mb-2">
                  {g.sensory_loss_pct  > 0 && <div className="col-6"><small className="text-muted">Sensory Loss:</small> <strong>{g.sensory_loss_pct}%</strong></div>}
                  {g.autonomic_pct     > 0 && <div className="col-6"><small className="text-muted">Autonomic:</small> <strong>{g.autonomic_pct}%</strong></div>}
                  {g.pain_insensitivity_pct > 0 && <div className="col-6"><small className="text-muted">Pain Insens:</small> <strong>{g.pain_insensitivity_pct}%</strong></div>}
                  {g.anhidrosis_pct    > 0 && <div className="col-6"><small className="text-muted">Anhidrosis:</small> <strong>{g.anhidrosis_pct}%</strong></div>}
                  {g.plantar_ulcers_pct > 0 && <div className="col-6"><small className="text-muted">Plantar Ulcers:</small> <strong>{g.plantar_ulcers_pct}%</strong></div>}
                  {g.mutilations_pct   > 0 && <div className="col-6"><small className="text-muted">Mutilations:</small> <strong>{g.mutilations_pct}%</strong></div>}
                  {g.cognitive_decline_pct > 0 && <div className="col-6"><small className="text-muted">Dementia:</small> <strong>{g.cognitive_decline_pct}%</strong></div>}
                  {g.hearing_loss_pct  > 0 && <div className="col-6"><small className="text-muted">SNHL:</small> <strong>{g.hearing_loss_pct}%</strong></div>}
                  {g.hyperthermia_pct  > 0 && <div className="col-6"><small className="text-muted">Hyperthermia:</small> <strong>{g.hyperthermia_pct}%</strong></div>}
                </div>
                <div className="mt-2">
                  {L_SERINE_GENE.includes(g.gene) && <AlertBadge text="L-SERINE Rx" color="#1a237e" />}
                  {TUDCA_GENE.includes(g.gene)    && <AlertBadge text="TUDCA" color="#880e4f" />}
                  {ANHIDROTIC.includes(g.gene)    && <AlertBadge text="ANHIDROTIC" color="#b71c1c" />}
                  {SWEATING_OK.includes(g.gene)   && <AlertBadge text="SWEATING OK" color="#1b5e20" />}
                  {ASHKENAZI.includes(g.gene)     && <AlertBadge text="ASHKENAZI ONLY" color="#880e4f" />}
                  {g.gene === 'WNK1' && <AlertBadge text="HSN2-EXON ONLY" color="#37474f" />}
                  {g.gene === 'NTRK1' && <AlertBadge text="HYPERTHERMIA LETHAL" color="#b71c1c" />}
                  {g.gene === 'DNMT1' && <AlertBadge text="DEMENTIA+SNHL TRIAD" color="#006064" />}
                </div>
                <div className="mt-2">
                  {g.hallmarks.slice(0, 2).map((h, i) => (
                    <div key={i} className="small text-muted">• {h}</div>
                  ))}
                </div>
                <div className="mt-2 p-2 rounded" style={{ background: '#fff3e0', fontSize: '0.72rem' }}>
                  <strong>⚠ Alert:</strong> {g.top_treatment_alert}
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
  return (
    <div>
      <h6 className="text-uppercase text-muted mb-3 small">8-Gene HSAN Reference Table</h6>
      <div className="table-responsive">
        <table className="table table-sm table-bordered align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Protein (aa)</th><th>Locus</th><th>HSAN Type</th>
              <th>Inheritance</th><th>Key Feature</th><th>Therapy</th><th>OMIM</th>
            </tr>
          </thead>
          <tbody>
            {data.map((g) => (
              <tr key={g.gene}>
                <td>
                  <span className="badge" style={{ background: GENE_COLORS[g.gene] }}>{g.gene}</span>
                </td>
                <td><small>{g.protein.split('(')[0].trim()} ({g.aa.split(' ')[0]} aa)</small></td>
                <td><small>{g.locus}</small></td>
                <td><small>{GENE_DISEASE[g.gene]?.split('(')[1]?.split(')')[0]}</small></td>
                <td>
                  <span className={`badge ${AD_GENES.includes(g.gene) ? 'bg-warning text-dark' : 'bg-info text-dark'}`}>
                    {AD_GENES.includes(g.gene) ? 'AD' : 'AR'}
                  </span>
                </td>
                <td><small>{g.hallmarks[0]?.slice(0, 70)}…</small></td>
                <td>
                  <span className="badge" style={{ background: GENE_COLORS[g.gene], fontSize: '0.7rem' }}>
                    {L_SERINE_GENE.includes(g.gene) ? 'L-serine' :
                     TUDCA_GENE.includes(g.gene) ? 'TUDCA' : 'Supportive'}
                  </span>
                </td>
                <td><small><a href={`https://omim.org/entry/${g.omim_disease}`} target="_blank" rel="noreferrer">#{g.omim_disease}</a></small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* DDx panel: anhidrosis */}
      <div className="row g-3 mt-3">
        <div className="col-md-6">
          <div className="card border-0 shadow-sm">
            <div className="card-header text-white" style={{ background: '#b71c1c' }}>
              <small className="fw-bold">🌡 ANHIDROTIC (hyperthermia risk)</small>
            </div>
            <div className="card-body p-2">
              {data.filter(g => ANHIDROTIC.includes(g.gene)).map(g => (
                <div key={g.gene} className="d-flex align-items-center mb-1">
                  <span className="badge me-2" style={{ background: GENE_COLORS[g.gene] }}>{g.gene}</span>
                  <small className="text-muted">{GENE_DISEASE[g.gene]?.split('—')[0].trim()}</small>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card border-0 shadow-sm">
            <div className="card-header text-white" style={{ background: '#1b5e20' }}>
              <small className="fw-bold">💧 SWEATING PRESERVED (no hyperthermia crisis)</small>
            </div>
            <div className="card-body p-2">
              {data.filter(g => SWEATING_OK.includes(g.gene)).map(g => (
                <div key={g.gene} className="d-flex align-items-center mb-1">
                  <span className="badge me-2" style={{ background: GENE_COLORS[g.gene] }}>{g.gene}</span>
                  <small className="text-muted">{GENE_DISEASE[g.gene]?.split('—')[0].trim()}</small>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h6 className="text-uppercase text-muted mb-3 small">Per-Gene Clinical Profile</h6>
      {data.map((g) => (
        <div key={g.gene} className="card border-0 shadow-sm mb-4">
          <div className="card-header text-white py-2 px-3" style={{ background: GENE_COLORS[g.gene] }}>
            <div className="d-flex justify-content-between">
              <span className="fw-bold">{g.gene} — {g.protein.split('(')[0].trim()}</span>
              <span className="small opacity-75">{g.locus} · {g.aa.split(' ')[0]} aa · {AD_GENES.includes(g.gene) ? 'AD' : 'AR'}</span>
            </div>
            <div className="small opacity-90">{GENE_DISEASE[g.gene]?.split('—')[1]?.trim()}</div>
          </div>
          <div className="card-body p-3">
            <div className="row g-3">
              {/* Stats */}
              <div className="col-md-4">
                <h6 className="small text-uppercase text-muted mb-2">Cohort Stats (n={g.n_patients})</h6>
                <table className="table table-sm table-borderless mb-0">
                  <tbody>
                    <tr><td className="text-muted small pe-2">Sensory Loss</td><td><strong>{g.sensory_loss_pct}%</strong></td></tr>
                    <tr><td className="text-muted small pe-2">Autonomic</td><td><strong>{g.autonomic_pct}%</strong></td></tr>
                    <tr><td className="text-muted small pe-2">Pain Insensitivity</td><td><strong>{g.pain_insensitivity_pct}%</strong></td></tr>
                    <tr><td className="text-muted small pe-2">Anhidrosis</td><td><strong>{g.anhidrosis_pct}%</strong></td></tr>
                    <tr><td className="text-muted small pe-2">Plantar Ulcers</td><td><strong>{g.plantar_ulcers_pct}%</strong></td></tr>
                    <tr><td className="text-muted small pe-2">Mutilations</td><td><strong>{g.mutilations_pct}%</strong></td></tr>
                    {g.cognitive_decline_pct > 0 && <tr><td className="text-muted small pe-2">Dementia</td><td><strong>{g.cognitive_decline_pct}%</strong></td></tr>}
                    {g.hearing_loss_pct > 0 && <tr><td className="text-muted small pe-2">SNHL</td><td><strong>{g.hearing_loss_pct}%</strong></td></tr>}
                    {g.hyperthermia_pct > 0 && <tr><td className="text-muted small pe-2">Hyperthermia</td><td><strong>{g.hyperthermia_pct}%</strong></td></tr>}
                    <tr><td className="text-muted small pe-2">Onset (avg)</td><td><strong>{g.avg_age_at_onset}y</strong></td></tr>
                    <tr><td className="text-muted small pe-2">Diag delay</td><td><strong>{g.avg_diagnosis_delay_years}y</strong></td></tr>
                  </tbody>
                </table>
              </div>
              {/* Hallmarks */}
              <div className="col-md-4">
                <h6 className="small text-uppercase text-muted mb-2">Clinical Hallmarks</h6>
                {g.hallmarks.map((h, i) => (
                  <div key={i} className="d-flex align-items-start mb-1">
                    <span className="me-2" style={{ color: GENE_COLORS[g.gene] }}>•</span>
                    <small>{h}</small>
                  </div>
                ))}
              </div>
              {/* Treatment Alerts */}
              <div className="col-md-4">
                <h6 className="small text-uppercase text-muted mb-2">Treatment Alerts</h6>
                {g.treatment_alerts.map((a, i) => (
                  <div key={i} className="mb-1 p-2 rounded" style={{ background: i === 0 ? '#ffebee' : '#f5f5f5', fontSize: '0.72rem' }}>
                    {i === 0 && <strong>⚠ </strong>}{a}
                  </div>
                ))}
              </div>
            </div>
            {/* Etiology distribution */}
            <div className="mt-3">
              <h6 className="small text-uppercase text-muted mb-2">Variant Distribution</h6>
              <div className="d-flex flex-wrap gap-2">
                {Object.entries(g.etiology_distribution).map(([et, cnt]) => (
                  <span key={et} className="badge" style={{ background: GENE_COLORS[g.gene], fontSize: '0.7rem' }}>
                    {et.length > 50 ? et.slice(0, 50) + '…' : et}: {cnt}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── DEFINITIONS TAB ── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const [open, setOpen] = useState(null);
  return (
    <div>
      <h6 className="text-uppercase text-muted mb-3 small">Clinical Definitions & Mechanisms</h6>
      <div className="accordion" id="hsanDefs">
        {data.definitions.map((d, i) => (
          <div key={i} className="accordion-item border-0 mb-2 shadow-sm">
            <h2 className="accordion-header">
              <button
                className={`accordion-button ${open === i ? '' : 'collapsed'} py-2 px-3`}
                style={{ fontSize: '0.85rem', fontWeight: open === i ? '600' : '400' }}
                onClick={() => setOpen(open === i ? null : i)}
              >
                {d.term}
              </button>
            </h2>
            {open === i && (
              <div className="accordion-body py-3 px-3">
                <p className="small mb-0" style={{ whiteSpace: 'pre-line', lineHeight: '1.6' }}>
                  {d.definition}
                </p>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Pharmacological Distinctions */}
      <h6 className="text-uppercase text-muted mb-3 mt-4 small">Pharmacological Distinctions</h6>
      <div className="card border-0 shadow-sm">
        <div className="card-body p-3">
          {data.pharmacological_distinctions.map((p, i) => (
            <div key={i} className="d-flex align-items-start mb-3">
              <span className="badge me-2 mt-1" style={{ background: '#1a237e', minWidth: 22, textAlign: 'center' }}>{i + 1}</span>
              <small style={{ lineHeight: '1.6' }}>{p}</small>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── MAIN PAGE ── */
export default function HSANAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hsan-atlas/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 'Gene Table' || tab === 'Clinical Atlas') {
      fetch(`${API}/api/hsan-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown)
        .catch(e => setError(e.message));
    }
    if (tab === 'Definitions') {
      fetch(`${API}/api/hsan-atlas/definitions`)
        .then(r => r.json()).then(setDefinitions)
        .catch(e => setError(e.message));
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4">
      <div className="row mb-3">
        <div className="col">
          <h4 className="mb-0">🧬 HSAN-Atlas</h4>
          <div className="text-muted small">
            Complete 8-Gene Hereditary Sensory &amp; Autonomic Neuropathy Atlas ·
            SPTLC1-HSAN1A · ELP1-FD/HSAN3 · NTRK1-CIPA/HSAN4 · NGFB-HSAN5 ·
            FAM134B-HSAN2B · DNMT1-HSAN1E · WNK1-HSAN2A · PRDM12-HSAN8 ·
            320 patients (8×40, seeds 1366–1373)
          </div>
        </div>
      </div>

      {error && <ErrorMsg msg={error} />}

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

      {tab === 'Overview'        && <OverviewTab      data={overview} />}
      {tab === 'Gene Table'      && <GeneTableTab     data={breakdown} />}
      {tab === 'Clinical Atlas'  && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'     && <DefinitionsTab   data={definitions} />}
    </div>
  );
}
