'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// Deafness Atlas color palette — auditory / cochlear
const COLOR  = '#0d47a1';  // deep blue — auditory system
const LIGHT  = '#e3f2fd';
const COLOR2 = '#b71c1c';  // dark red — CI mandatory / ANSD
const COLOR3 = '#e65100';  // orange — warning / aminoglycoside risk
const COLOR4 = '#1b5e20';  // dark green — CI curative
const COLOR5 = '#4a148c';  // dark purple — Usher / RP
const COLOR6 = '#880e4f';  // dark pink — Usher type 1
const COLOR7 = '#006064';  // teal — Usher type 2
const COLOR8 = '#37474f';  // blue-grey — Usher type 3 / progressive

const GENE_COLORS = {
  GJB2:    '#1565c0',  // blue — connexin / most common SNHL
  SLC26A4: '#00695c',  // teal — Pendred / thyroid / EVA
  OTOF:    '#c62828',  // red — ANSD / otoferlin
  MYO7A:   '#6a1b9a',  // purple — Usher 1B
  USH2A:   '#2e7d32',  // green — Usher 2A most common
  CDH23:   '#ad1457',  // dark pink — Usher 1D / tip-link
  PCDH15:  '#e65100',  // orange — Usher 1F / ankle-link
  CLRN1:   '#37474f',  // blue-grey — Usher 3A progressive
};

const USHER_BADGE_COLORS = {
  '1': '#6a1b9a',
  '2': '#2e7d32',
  '3': '#37474f',
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
        <span>{label}</span><span className="fw-semibold">{typeof pct === 'number' ? `${pct}%` : pct}</span>
      </div>
      {typeof pct === 'number' && (
        <div className="progress" style={{ height: '7px' }}>
          <div className="progress-bar" style={{ width: `${Math.min(pct,100)}%`, backgroundColor: color }} />
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
  return (
    <div className="text-center py-5">
      <div className="spinner-border" style={{ color: COLOR }} />
      <div className="mt-2 text-muted small">Loading Deafness Atlas…</div>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger">{msg}</div>;
}

// ── Tab: Overview ──────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ac = data.aggregate_clinical || {};
  const dc = data.drug_contraindications || {};
  const kr = data.key_rules || {};
  const ut = data.usher_types || {};

  return (
    <div>
      <h5 className="fw-bold mb-1" style={{ color: COLOR }}>{data.atlas_name}</h5>
      <p className="text-muted small mb-3">{data.atlas_subtitle}</p>
      <p className="mb-3">{data.description}</p>

      {/* Critical drug alerts */}
      {dc.aminoglycosides && (
        <AlertBox type="danger" title="AMINOGLYCOSIDES — HIGH RISK ALL 8 GENES">
          {dc.aminoglycosides}
        </AlertBox>
      )}
      {dc.loop_diuretics && (
        <AlertBox type="warning" title="LOOP DIURETICS — OTOTOXIC RISK">
          {dc.loop_diuretics}
        </AlertBox>
      )}
      {dc.cisplatin_caution && (
        <AlertBox type="warning" title="CISPLATIN/CARBOPLATIN — OTOTOXIC RISK">
          {dc.cisplatin_caution}
        </AlertBox>
      )}

      {/* Critical rules */}
      {Object.keys(kr).length > 0 && (
        <div className="alert alert-secondary py-2 px-3 mb-3">
          <strong>📋 Critical Clinical Rules</strong>
          <ul className="mb-0 mt-1 small">
            {Object.entries(kr).map(([k, v]) => <li key={k}>{v}</li>)}
          </ul>
        </div>
      )}

      {/* NBS utility */}
      {data.nbs_utility && (
        <AlertBox type="info" title="Newborn Hearing Screening (NBS)">
          {data.nbs_utility}
        </AlertBox>
      )}

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Genes" value={data.n_genes} color={COLOR} />
        <KPI label="Patients" value={data.n_patients} color={COLOR} />
        <KPI label="Profound SNHL" value={`${ac.profound_snhl_pct}%`} color={COLOR2} />
        <KPI label="RP (Usher)" value={`${ac.rp_pct}%`} color={COLOR5} />
        <KPI label="Vestibular Dysfn" value={`${ac.vestibular_dysfunction_pct}%`} color={COLOR3} />
        <KPI label="CI Performed" value={`${ac.ci_performed_pct}%`} color={COLOR4} />
        <KPI label="ANSD (OTOF)" value={`${ac.ansd_pct}%`} color={COLOR2} />
        <KPI label="Congenital Onset" value={`${ac.congenital_onset_pct}%`} color={COLOR} />
        <KPI label="NBS Detected" value={`${ac.nbs_detected_pct}%`} color={COLOR4} />
        <KPI label="Hearing Aid" value={`${ac.hearing_aid_pct}%`} color={COLOR7} />
      </div>

      <div className="row g-3">
        {/* Aggregate clinical */}
        <div className="col-md-4">
          <div className="card shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Aggregate Clinical (320 pts)</h6>
              <BarRow label="SNHL (all)" pct={ac.snhl_pct} color={COLOR} />
              <BarRow label="Profound SNHL" pct={ac.profound_snhl_pct} color={COLOR2} />
              <BarRow label="Retinitis Pigmentosa" pct={ac.rp_pct} color={COLOR5} />
              <BarRow label="Vestibular Dysfunction" pct={ac.vestibular_dysfunction_pct} color={COLOR3} />
              <BarRow label="Cochlear Implant" pct={ac.ci_performed_pct} color={COLOR4} />
              <BarRow label="Hearing Aid" pct={ac.hearing_aid_pct} color={COLOR7} />
              <BarRow label="ANSD Pattern" pct={ac.ansd_pct} color={COLOR2} />
              <BarRow label="Congenital Onset" pct={ac.congenital_onset_pct} color={COLOR} />
              <BarRow label="NBS Detected" pct={ac.nbs_detected_pct} color={COLOR4} />
            </div>
          </div>
        </div>

        {/* Usher syndrome classification */}
        <div className="col-md-4">
          <div className="card shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3" style={{ color: COLOR5 }}>Usher Syndrome Types</h6>
              {Object.entries(ut).map(([type, genes]) => (
                <div key={type} className="mb-3">
                  <div className="fw-bold small mb-1" style={{ color: USHER_BADGE_COLORS[type.split('_')[1]] || COLOR5 }}>
                    Usher Type {type.split('_')[1]}
                  </div>
                  <div className="d-flex flex-wrap gap-1">
                    {(Array.isArray(genes) ? genes : []).map(g => (
                      <span key={g} className="badge" style={{ backgroundColor: GENE_COLORS[g] || COLOR5 }}>{g}</span>
                    ))}
                  </div>
                  <div className="text-muted small mt-1">
                    {type.includes('1') ? 'Profound congenital SNHL + RP + absent vestibular' :
                     type.includes('2') ? 'Moderate SNHL + RP + normal vestibular' :
                     'Progressive postlingual SNHL + RP + variable vestibular'}
                  </div>
                </div>
              ))}
              <div className="mt-2 small text-muted">
                <strong>Non-Usher:</strong>{' '}
                <span className="badge" style={{ backgroundColor: GENE_COLORS['GJB2'] }}>GJB2</span>{' '}
                <span className="badge" style={{ backgroundColor: GENE_COLORS['SLC26A4'] }}>SLC26A4</span>{' '}
                <span className="badge" style={{ backgroundColor: GENE_COLORS['OTOF'] }}>OTOF</span>{' '}
                — pure SNHL (no RP)
              </div>
            </div>
          </div>
        </div>

        {/* Drug CIs */}
        <div className="col-md-4">
          <div className="card shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3" style={{ color: COLOR2 }}>Drug Contraindications</h6>
              {Object.entries(dc).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: '#fff8f6', borderLeft: `4px solid ${COLOR2}` }}>
                  <div className="fw-semibold small text-danger">{k.replace(/_/g, ' ').toUpperCase()}</div>
                  <div className="small">{typeof v === 'string' ? v : JSON.stringify(v)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* CI gene-specific rates */}
      {data.gene_ci_pct && (
        <div className="card shadow-sm mt-3">
          <div className="card-body">
            <h6 className="fw-bold mb-3" style={{ color: COLOR4 }}>🦻 Cochlear Implant Rate by Gene (%)</h6>
            <div className="row">
              {Object.entries(data.gene_ci_pct).map(([g, v]) => (
                <div key={g} className="col-6 col-md-3 mb-2">
                  <BarRow label={g} pct={v} color={GENE_COLORS[g] || COLOR} />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Tab: Gene Table ────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || [];

  return (
    <div>
      <h5 className="fw-bold mb-3" style={{ color: COLOR }}>Per-Gene Breakdown — Hereditary Deafness &amp; Usher Atlas</h5>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-dark">
            <tr>
              <th>Gene</th>
              <th>Alias / Disease</th>
              <th>Locus</th>
              <th>Profound SNHL %</th>
              <th>RP %</th>
              <th>Vestibular %</th>
              <th>CI %</th>
              <th>NBS %</th>
              <th>Usher Type</th>
              <th>ANSD</th>
            </tr>
          </thead>
          <tbody>
            {genes.map((g) => (
              <tr key={g.gene}>
                <td>
                  <span className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</span>
                </td>
                <td className="small text-muted" style={{ maxWidth: 160 }}>
                  <span title={g.alias}>{g.alias?.slice(0, 50)}</span>
                </td>
                <td><code className="small">{g.locus}</code></td>
                <td>
                  <span className={`badge ${g.profound_snhl_pct >= 70 ? 'bg-danger' : g.profound_snhl_pct >= 40 ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                    {g.profound_snhl_pct}%
                  </span>
                </td>
                <td>
                  <span className={`badge ${g.rp_pct >= 80 ? 'bg-danger' : g.rp_pct >= 40 ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                    {g.rp_pct}%
                  </span>
                </td>
                <td>
                  <span className={`badge ${g.vestibular_dysfunction_pct >= 80 ? 'bg-danger' : g.vestibular_dysfunction_pct >= 40 ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                    {g.vestibular_dysfunction_pct}%
                  </span>
                </td>
                <td>
                  <span className={`badge ${g.ci_performed_pct >= 70 ? 'bg-success' : g.ci_performed_pct >= 40 ? 'bg-info text-dark' : 'bg-secondary'}`}>
                    {g.ci_performed_pct}%
                  </span>
                </td>
                <td>
                  <span className={`badge ${g.nbs_detected_pct >= 80 ? 'bg-success' : 'bg-secondary'}`}>
                    {g.nbs_detected_pct}%
                  </span>
                </td>
                <td>
                  {g.usher_type
                    ? <span className="badge" style={{ backgroundColor: USHER_BADGE_COLORS[g.usher_type] || COLOR5 }}>Usher {g.usher_type}</span>
                    : <span className="badge bg-secondary">Non-Usher</span>}
                </td>
                <td>
                  {g.ansd_gene
                    ? <span className="badge bg-danger">ANSD</span>
                    : <span className="badge bg-secondary">—</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-gene cards */}
      <div className="row g-3 mt-2">
        {genes.map((g) => (
          <div key={g.gene} className="col-md-6 col-lg-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold py-2" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR, color: '#fff' }}>
                {g.gene} — {g.alias?.split('—')[0]?.trim() || g.alias?.slice(0, 30)}
              </div>
              <div className="card-body small">
                <p className="mb-1"><strong>Locus:</strong> {g.locus} · {g.aa} · {g.kDa}</p>
                <p className="mb-2"><strong>Phenotype:</strong> {g.phenotype?.slice(0, 120)}…</p>
                {g.hallmark && (
                  <p className="mb-2 text-muted"><strong>Hallmark:</strong> {g.hallmark?.slice(0, 120)}…</p>
                )}
                <div className="d-flex flex-wrap gap-1 mt-2">
                  {g.ansd_gene && <span className="badge bg-danger">ANSD</span>}
                  {g.ci_recommended && <span className="badge bg-success">CI Recommended</span>}
                  {g.usher_type && (
                    <span className="badge" style={{ backgroundColor: USHER_BADGE_COLORS[g.usher_type] }}>
                      Usher {g.usher_type}
                    </span>
                  )}
                  {g.rp_pct > 0 && <span className="badge bg-warning text-dark">RP</span>}
                </div>
                <div className="mt-2 text-danger small">
                  <strong>Drug Risk:</strong> {g.aminoglycoside_risk?.slice(0, 80)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Tab: Clinical Atlas ────────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || [];

  return (
    <div>
      <h5 className="fw-bold mb-3" style={{ color: COLOR }}>Clinical Atlas — Deafness &amp; Usher Syndrome · Pathomechanism &amp; Treatment</h5>
      {genes.map((g) => (
        <div key={g.gene} className="card shadow-sm mb-4">
          <div className="card-header fw-bold py-2" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR, color: '#fff' }}>
            {g.gene} ({g.locus}) — OMIM #{g.omim_gene} · {g.alias}
          </div>
          <div className="card-body">
            <div className="row g-3">
              <div className="col-md-7">
                <h6 className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>Disease &amp; Phenotype</h6>
                <p className="small mb-2">{g.disease}</p>
                <p className="small mb-2"><strong>Hallmark:</strong> {g.hallmark}</p>
                {g.founder_variant && (
                  <p className="small mb-2"><strong>Founder variant:</strong> {g.founder_variant}</p>
                )}
                {g.onset_pattern && (
                  <p className="small mb-2"><strong>Onset:</strong> {g.onset_pattern}</p>
                )}
                {g.key_ddx && (
                  <p className="small mb-2"><strong>Key DDx:</strong> {g.key_ddx}</p>
                )}
                {g.ansd_gene && (
                  <div className="alert alert-danger py-1 px-2 mb-2">
                    <strong className="small">🚨 ANSD: Normal OAE + Absent/Abnormal ABR</strong>
                    <div className="small">Hearing aid FAILS — CI curative (bypasses spiral ganglion)</div>
                  </div>
                )}
                {g.usher_type && (
                  <div className="alert alert-warning py-1 px-2 mb-2">
                    <strong className="small">👁️ Usher Type {g.usher_type}: Annual Retinal Monitoring MANDATORY</strong>
                    <div className="small">ERG diagnostic; ophthalmology + low-vision rehabilitation; driving assessment mandatory</div>
                  </div>
                )}
              </div>
              <div className="col-md-5">
                <h6 className="fw-bold">Per-Gene Stats</h6>
                <BarRow label="Profound SNHL" pct={g.profound_snhl_pct} color={GENE_COLORS[g.gene] || COLOR} />
                <BarRow label="Retinitis Pigmentosa" pct={g.rp_pct} color={COLOR5} />
                <BarRow label="Vestibular Dysfunction" pct={g.vestibular_dysfunction_pct} color={COLOR3} />
                <BarRow label="Cochlear Implant" pct={g.ci_performed_pct} color={COLOR4} />
                <BarRow label="Hearing Aid" pct={g.hearing_aid_pct} color={COLOR7} />
                <BarRow label="ANSD Pattern" pct={g.ansd_pct} color={COLOR2} />
                <BarRow label="Congenital Onset" pct={g.congenital_onset_pct} color={COLOR} />
                <BarRow label="NBS Detected" pct={g.nbs_detected_pct} color={COLOR4} />
                <div className="mt-2 d-flex flex-wrap gap-1">
                  {g.ci_recommended && <span className="badge bg-success">CI Recommended</span>}
                  {g.ansd_gene && <span className="badge bg-danger">ANSD</span>}
                  {g.usher_type && (
                    <span className="badge" style={{ backgroundColor: USHER_BADGE_COLORS[g.usher_type] }}>
                      Usher {g.usher_type}
                    </span>
                  )}
                  {g.rp_pct > 0 && <span className="badge bg-warning text-dark">RP</span>}
                  {g.vestibular_dysfunction_pct > 80 && <span className="badge bg-warning text-dark">Vestibular Absent</span>}
                </div>
                <div className="mt-2 small text-danger">
                  <strong>Aminoglycoside risk:</strong>
                  <div>{g.aminoglycoside_risk?.slice(0, 100)}…</div>
                </div>
                {g.loop_diuretic_risk && (
                  <div className="mt-1 small text-warning">
                    <strong>Loop diuretic:</strong> {g.loop_diuretic_risk?.slice(0, 80)}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const [q, setQ] = useState('');
  const filtered = q
    ? data.filter(d => d.term.toLowerCase().includes(q.toLowerCase()) || d.definition.toLowerCase().includes(q.toLowerCase()))
    : data;

  return (
    <div>
      <h5 className="fw-bold mb-3" style={{ color: COLOR }}>Clinical Terminology — Deafness &amp; Usher Atlas</h5>
      <input
        className="form-control mb-3"
        placeholder="Search terms…"
        value={q}
        onChange={e => setQ(e.target.value)}
      />
      <div className="row g-3">
        {filtered.map((d, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-body">
                <h6 className="fw-bold mb-2" style={{ color: COLOR }}>{d.term}</h6>
                <p className="small mb-0">{d.definition}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
      {filtered.length === 0 && <p className="text-muted">No matching terms.</p>}
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────
export default function DeafnessAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    const load = async (path, setter) => {
      try {
        const r = await fetch(`${API}${path}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        setter(await r.json());
      } catch (e) {
        setErr(e.message);
      }
    };
    load('/api/deafness-atlas/overview', setOverview);
    load('/api/deafness-atlas/breakdown', setBreakdown);
    load('/api/deafness-atlas/definitions', setDefinitions);
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2 flex-wrap">
        <span style={{ fontSize: 28 }}>👂</span>
        <div>
          <h4 className="fw-bold mb-0" style={{ color: COLOR }}>Deafness &amp; Usher Syndrome Atlas</h4>
          <div className="text-muted small">
            Complete 8-Gene Hereditary Deafness &amp; Usher Syndrome Atlas ·
            GJB2·SLC26A4·OTOF·MYO7A·USH2A·CDH23·PCDH15·CLRN1 · 320 patients (8×40, seeds 1126–1133)
          </div>
        </div>
      </div>

      {err && <ErrorMsg msg={err} />}

      {/* Tab navigation */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active fw-bold' : ''}`}
              style={tab === t ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(t)}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'       && <OverviewTab data={overview} />}
      {tab === 'Gene Table'     && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'    && <DefinitionsTab data={definitions} />}

      <div className="mt-4 text-muted small text-center">
        Deafness-Atlas · 8 genes · 320-patient aggregate · seeds 1126–1133 ·
        3 endpoints /api/deafness-atlas/overview|breakdown|definitions ·{' '}
        <Link href="/" className="text-muted">← Dashboard</Link>
      </div>
    </div>
  );
}
