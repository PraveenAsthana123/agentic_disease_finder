'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  CNGB3:  '#1565c0',  // deep blue — most common achromatopsia ACHM3
  CNGA3:  '#0288d1',  // mid blue — ACHM2 second most common
  GNAT2:  '#006064',  // dark teal — ACHM4 rare
  PDE6C:  '#2e7d32',  // deep green — ACHM5 incomplete more common
  PDE6H:  '#558b2f',  // olive green — ACHM6 mildest
  ATF6:   '#e65100',  // amber-orange — foveal hypoplasia DISTINCT
  KCNV2:  '#6a1b9a',  // deep purple — supernormal rod ERG PATHOGNOMONIC
  OPN1LW: '#c62828',  // deep red — BCM X-linked males only
};

const GENE_DISEASE = {
  CNGB3:  'Achromatopsia ACHM3 (AR) — MOST COMMON 50%; p.T383fsX European founder; complete; gene therapy Phase 2/3',
  CNGA3:  'Achromatopsia ACHM2 (AR) — 25%; p.R427W European founder; complete; gene therapy Phase 2',
  GNAT2:  'Achromatopsia ACHM4 (AR) — ~2-3%; cone transducin α2; typically complete; rare',
  PDE6C:  'Achromatopsia ACHM5 (AR) — incomplete form more common; PDE6 α\' cone subunit; photophobia prominent',
  PDE6H:  'Achromatopsia ACHM6 (AR) — mildest/incomplete; PDE6 γ\' cone subunit; residual color vision',
  ATF6:   'Achromatopsia ACHM7 (AR) — DISTINCT: foveal hypoplasia on OCT; ER stress TF; macular atrophy evolves',
  KCNV2:  'CDSRR (AR) — SUPERNORMAL ROD ERG PATHOGNOMONIC; Kv8.2 channel; PROGRESSIVE macular dystrophy',
  OPN1LW: 'Blue Cone Monochromatism BCM (XLR) — males only; WES MISSES; only S-cones preserved; OPN1LW/MW array',
};

const COMPLETE_ACHM_GENES   = ['CNGB3', 'CNGA3', 'GNAT2'];
const INCOMPLETE_ACHM_GENES = ['PDE6C', 'PDE6H', 'ATF6'];
const SPECIAL_GENES         = ['KCNV2', 'OPN1LW'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Hereditary Color Vision Disorder Atlas…</p>
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

function AlertBadge({ text, color = '#37474f' }) {
  return (
    <span className="badge me-1 mb-1" style={{ background: color, fontSize: '0.7rem' }}>
      {text}
    </span>
  );
}

/* ── OVERVIEW TAB ── */
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const m = data.aggregate_metrics || {};

  const statItems = [
    { key: 'va_worse_than_0_2_pct',      label: 'VA ≤ 0.2 (severely reduced)',          color: '#1565c0' },
    { key: 'total_color_blind_pct',       label: 'Total / severe colour blindness',       color: '#0288d1' },
    { key: 'nystagmus_pct',               label: 'Pendular Nystagmus (any)',              color: '#006064' },
    { key: 'photophobia_pct',             label: 'Photophobia (day-blindness)',           color: '#e65100' },
    { key: 'flat_photopic_erg_pct',       label: 'Flat Photopic ERG',                    color: '#1565c0' },
    { key: 'supernormal_rod_erg_pct',     label: 'Supernormal Rod ERG (KCNV2)',          color: '#6a1b9a' },
    { key: 'foveal_hypoplasia_pct',       label: 'Foveal Hypoplasia on OCT',             color: '#e65100' },
    { key: 'myopia_pct',                  label: 'Myopia (≥ −1.0 D)',                    color: '#c62828' },
    { key: 'progressive_macular_pct',     label: 'Progressive Macular Dystrophy',        color: '#6a1b9a' },
    { key: 'gene_therapy_eligible_pct',   label: 'Gene Therapy Trial Eligible',          color: '#2e7d32' },
  ];

  return (
    <div>
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={data.total_patients} color="#37474f" />
        <KPI label="Genes" value={data.genes_covered ? JSON.parse(data.genes_covered).length : 8} color="#37474f" />
        <KPI label="Complete Achromatopsia" value="3" color="#1565c0" />
        <KPI label="Incomplete / ATF6" value="3" color="#2e7d32" />
        <KPI label="Special Disorders" value="2" color="#6a1b9a" />
        <KPI label="Seeds" value={data.seeds} color="#37474f" />
      </div>

      <div className="alert alert-success mb-3">
        <strong>🧬 CNGB3 &amp; CNGA3 — GENE THERAPY TRIALS 2026:</strong> Refer ALL newly diagnosed complete achromatopsia patients for gene therapy trial eligibility assessment. CNGB3 (Phase 2/3: AGTC-402, RD-CURE) and CNGA3 (Phase 2: BTT-401) — most advanced photoreceptor gene therapy programmes. Enroll before secondary degeneration.
      </div>
      <div className="alert alert-danger mb-3">
        <strong>🔬 OPN1LW (BCM) — WES MISSES DIAGNOSIS:</strong> Blue Cone Monochromatism is caused by LCR (locus control region) deletion/rearrangement at Xq28. Standard WES/panel-NGS does NOT detect copy number changes here. Order dedicated OPN1LW/OPN1MW gene array analysis when BCM suspected (male, nystagmus, photophobia, residual blue-only colour).
      </div>
      <div className="alert alert-warning mb-3">
        <strong>⚠️ KCNV2 — PROGRESSIVE DISEASE (NOT STATIC):</strong> Unlike achromatopsia (stationary), KCNV2-CDSRR is PROGRESSIVE — macular dystrophy develops. Supernormal rod ERG on dark-adapted bright flash is PATHOGNOMONIC. Do NOT reassure families that vision is stable — annual monitoring mandatory.
      </div>
      <div className="alert alert-info mb-4">
        <strong>ℹ️ ATF6 vs CNGB3/CNGA3 on OCT:</strong> ATF6 shows FOVEAL HYPOPLASIA as primary OCT finding. CNGB3/CNGA3: fovea initially normal, then outer nuclear layer thinning. ATF6: ER stress-mediated cone differentiation failure — different mechanism, distinct OCT. FL-41 lenses for photophobia in ALL.
      </div>

      <h6 className="fw-bold mb-3">Aggregate Clinical Features (320 patients, 8 genes)</h6>
      <div className="row g-2 mb-4">
        {statItems.map(({ key, label, color }) => m?.[key] != null && (
          <div key={key} className="col-6 col-md-4 col-lg-3">
            <div className="card border-0 shadow-sm">
              <div className="card-body p-2" style={{ borderLeft: `4px solid ${color}` }}>
                <div className="fw-bold" style={{ color }}>{m[key]}%</div>
                <div className="text-muted small">{label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mb-2">Gene Classification</h6>
      <div className="row g-3 mb-4">
        <div className="col-md-4">
          <div className="card border-0 shadow-sm">
            <div className="card-header" style={{ background: '#1565c0', color: 'white' }}>
              <strong>Complete Achromatopsia (3 genes)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {COMPLETE_ACHM_GENES.map(g => (
                <li key={g} className="list-group-item py-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span>{' — '}
                  <span className="text-muted">{GENE_DISEASE[g].split('—')[1]?.split(';')[0]?.trim()}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card border-0 shadow-sm">
            <div className="card-header" style={{ background: '#2e7d32', color: 'white' }}>
              <strong>Incomplete / Macular (3 genes)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {INCOMPLETE_ACHM_GENES.map(g => (
                <li key={g} className="list-group-item py-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span>{' — '}
                  <span className="text-muted">{GENE_DISEASE[g].split('—')[1]?.split(';')[0]?.trim()}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card border-0 shadow-sm">
            <div className="card-header" style={{ background: '#6a1b9a', color: 'white' }}>
              <strong>Special Photoreceptor Disorders (2 genes)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {SPECIAL_GENES.map(g => (
                <li key={g} className="list-group-item py-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span>{' — '}
                  <span className="text-muted">{GENE_DISEASE[g].split('—')[1]?.split(';')[0]?.trim()}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <h6 className="fw-bold mb-2">Top Clinical Alerts</h6>
      <div className="mb-3">
        {(data.key_clinical_alerts || []).map((a, i) => (
          <AlertBadge key={i} text={a}
            color={
              a.includes('CNGB3') ? '#1565c0' :
              a.includes('CNGA3') ? '#0288d1' :
              a.includes('KCNV2') ? '#6a1b9a' :
              a.includes('OPN1LW') || a.includes('BCM') || a.includes('WES') ? '#c62828' :
              a.includes('ATF6') ? '#e65100' :
              a.includes('FL-41') ? '#2e7d32' :
              '#546e7a'
            } />
        ))}
      </div>

      <h6 className="fw-bold mb-2">Atlas Summary</h6>
      <p className="text-muted small">{data.atlas_summary}</p>

      <div className="row g-3">
        {Object.entries(data.gene_summary || {}).map(([gene, info]) => (
          <div key={gene} className="col-12 col-md-6">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body p-3" style={{ borderLeft: `5px solid ${GENE_COLORS[gene] || '#546e7a'}` }}>
                <div className="fw-bold small mb-1" style={{ color: GENE_COLORS[gene] }}>{gene}</div>
                <div className="text-muted" style={{ fontSize: '0.78rem' }}>
                  {info.disease_category || GENE_DISEASE[gene]}
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
  const genes = data.gene_breakdowns || [];

  return (
    <div className="table-responsive">
      <table className="table table-sm table-hover align-middle">
        <thead className="table-dark">
          <tr>
            <th>Gene</th><th>Disease</th><th>Locus</th>
            <th>Inheritance</th><th>Photopic ERG</th><th>OCT Finding</th><th>N Patients</th>
          </tr>
        </thead>
        <tbody>
          {genes.map(g => (
            <tr key={g.gene}>
              <td><span className="fw-bold" style={{ color: GENE_COLORS[g.gene] }}>{g.gene}</span></td>
              <td style={{ fontSize: '0.8rem', maxWidth: 200 }}>{g.disease_category}</td>
              <td><code style={{ fontSize: '0.75rem' }}>{g.locus}</code></td>
              <td>
                <span className={`badge ${
                  g.inheritance?.includes('AR') ? 'bg-success' :
                  g.inheritance?.includes('XLR') ? 'bg-danger' :
                  'bg-primary'}`}
                  style={{ fontSize: '0.65rem' }}>
                  {g.inheritance?.split(' ')[0]}
                </span>
              </td>
              <td style={{ fontSize: '0.78rem' }}>{g.morphology}</td>
              <td style={{ fontSize: '0.78rem' }}>{g.key_features?.[0] || '—'}</td>
              <td className="text-center">{g.n_patients}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.gene_breakdowns || [];
  const [selectedGene, setSelectedGene] = useState(genes[0]?.gene || '');
  const g = genes.find(x => x.gene === selectedGene);
  if (!g) return null;

  const statItems = [
    { key: 'va_poor_pct',           label: 'VA ≤ 0.2' },
    { key: 'total_color_blind_pct', label: 'Total color blind' },
    { key: 'nystagmus_pct',         label: 'Nystagmus' },
    { key: 'photophobia_pct',       label: 'Photophobia' },
    { key: 'flat_photopic_pct',     label: 'Flat photopic ERG' },
    { key: 'supernormal_rod_pct',   label: 'Supernormal rod ERG' },
    { key: 'foveal_hypoplasia_pct', label: 'Foveal hypoplasia OCT' },
    { key: 'myopia_pct',            label: 'Myopia' },
    { key: 'progressive_mac_pct',   label: 'Progressive macular' },
  ];

  return (
    <div className="row g-3">
      <div className="col-md-2">
        <div className="list-group list-group-flush">
          {genes.map(gene => (
            <button key={gene.gene}
              className={`list-group-item list-group-item-action py-1 px-2 ${selectedGene === gene.gene ? 'active' : ''}`}
              style={selectedGene === gene.gene ? { background: GENE_COLORS[gene.gene], borderColor: GENE_COLORS[gene.gene] } : {}}
              onClick={() => setSelectedGene(gene.gene)}>
              <span className="fw-bold small">{gene.gene}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="col-md-10">
        <div className="card border-0 shadow-sm">
          <div className="card-header" style={{ background: GENE_COLORS[g.gene], color: 'white' }}>
            <strong>{g.gene}</strong> — {g.disease_category} | {g.locus} | {g.inheritance?.split(' ')[0]}
          </div>
          <div className="card-body">
            <div className="row g-3 mb-3">
              <div className="col-md-6">
                <h6 className="fw-bold">Pathognomonic / Key Features</h6>
                <ul className="small mb-0">
                  <li>{g.pathognomonic}</li>
                  {(g.key_features || []).map((h, i) => <li key={i} className="mb-1">{h}</li>)}
                </ul>
              </div>
              <div className="col-md-6">
                <h6 className="fw-bold">Treatment / Management</h6>
                <p className="small mb-2">{g.treatment}</p>
                <h6 className="fw-bold">Key DDx</h6>
                <ul className="small mb-0">
                  {(g.key_ddx || []).map((d, i) => <li key={i}>{d}</li>)}
                </ul>
              </div>
            </div>

            <div className="mb-3">
              <h6 className="fw-bold">Feature Frequencies ({g.n_patients} patients)</h6>
              <div className="row g-1">
                {statItems.map(({ key, label }) => g[key] != null && (
                  <div key={key} className="col-6 col-md-4">
                    <div className="d-flex align-items-center gap-2 small">
                      <div style={{ width: 40, height: 8, borderRadius: 4, background: '#e0e0e0', position: 'relative', flexShrink: 0 }}>
                        <div style={{ width: `${g[key]}%`, height: '100%', borderRadius: 4, background: GENE_COLORS[g.gene] }} />
                      </div>
                      <span className="text-muted" style={{ fontSize: '0.7rem' }}>{label} <strong>{g[key]}%</strong></span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {g.systemic_involvement && (
              <div className="mb-3">
                <h6 className="fw-bold">Systemic / Syndromic</h6>
                <p className="small mb-0 text-muted">{g.systemic_involvement}</p>
              </div>
            )}

            <div className="mb-3">
              <h6 className="fw-bold">ERG Pattern</h6>
              <p className="small mb-0 text-muted">{g.morphology}</p>
            </div>

            {g.sample_patients?.length > 0 && (
              <div>
                <h6 className="fw-bold">Sample Patients</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-striped" style={{ fontSize: '0.75rem' }}>
                    <thead><tr><th>Age</th><th>VA</th><th>Nystagmus</th><th>Photophobia</th><th>ERG</th></tr></thead>
                    <tbody>
                      {g.sample_patients.slice(0, 5).map((p, i) => (
                        <tr key={i}>
                          <td>{p.age}y</td>
                          <td>{p.va?.toFixed(2) ?? '—'}</td>
                          <td>{p.nystagmus ? '✓' : '—'}</td>
                          <td>{p.photophobia ? '✓' : '—'}</td>
                          <td style={{ maxWidth: 200 }}>{p.erg_pattern}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── DEFINITIONS TAB ── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;

  return (
    <div>
      <h6 className="fw-bold mb-3">Gene Entries</h6>
      {Object.entries(data.gene_entries || {}).map(([gene, entry]) => (
        <div key={gene} className="mb-3 p-3 rounded" style={{ background: '#f8f9fa', borderLeft: `4px solid ${GENE_COLORS[gene] || '#37474f'}` }}>
          <div className="fw-bold small mb-1" style={{ color: GENE_COLORS[gene] || '#37474f' }}>{gene}</div>
          <div className="small text-muted">{typeof entry === 'string' ? entry : JSON.stringify(entry)}</div>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4">Clinical Glossary</h6>
      {Object.entries(data.cv_glossary || {}).map(([term, def]) => (
        <div key={term} className="mb-3 p-3 rounded" style={{ background: '#f8f9fa', borderLeft: '4px solid #546e7a' }}>
          <div className="fw-bold small mb-1" style={{ color: '#546e7a' }}>{term.replace(/_/g, ' ')}</div>
          <div className="small text-muted">{typeof def === 'string' ? def : JSON.stringify(def)}</div>
        </div>
      ))}
    </div>
  );
}

/* ── MAIN PAGE ── */
export default function HereditaryColorVisionDisorderAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-color-vision-disorder-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-color-vision-disorder-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-color-vision-disorder-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, def]) => { setOverview(ov); setBreakdown(bd); setDefinitions(def); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4">
      <div className="mb-4">
        <h4 className="fw-bold mb-1">🧬 Hereditary Color Vision &amp; Photoreceptor Disorder Atlas</h4>
        <p className="text-muted small mb-0">
          Complete 8-Gene Hereditary Color Vision &amp; Photoreceptor Reference —
          CNGB3 (ACHM3 most common) · CNGA3 (ACHM2) · GNAT2 (ACHM4) ·
          PDE6C (ACHM5) · PDE6H (ACHM6 mildest) · ATF6 (ACHM7 foveal hypoplasia) ·
          KCNV2 (CDSRR supernormal rod ERG) · OPN1LW (BCM WES misses) |
          320 patients · 8×40 · seeds 2422–2429
        </p>
      </div>

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>{t}</button>
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
