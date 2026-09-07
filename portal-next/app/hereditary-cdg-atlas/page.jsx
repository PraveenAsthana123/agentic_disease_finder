'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PMM2:    '#dc2626',  // red    — CDG-Ia; most common; no treatment
  MPI:     '#16a34a',  // green  — CDG-Ib; treatable; no neurology
  ALG6:    '#c2410c',  // orange — CDG-Ic; 2nd most common N-glyc CDG
  PGM1:    '#0d9488',  // teal   — CDG-PGM1; bifid uvula; treatable
  SLC35A2: '#7c3aed',  // purple — CDG-IIm; X-linked; epilepsy
  SLC35C1: '#0ea5e9',  // sky    — CDG-IIc/LAD-II; Bombay; treatable
  DOLK:    '#b45309',  // amber  — CDG-Im; DCM + ichthyosis
  COG7:    '#1e293b',  // slate  — CDG-IIe; severe neonatal
};

const GENE_DISEASE = {
  PMM2:    'AR CDG-Ia — PMM2-246aa — 16p13.2 — Phosphomannomutase-2 — Cerebellar-Hypoplasia-PATHOGNOMONIC — Inverted-Nipples-Fat-Pads — Protein-C-S-Reduced-Thrombosis — No-Treatment-Inositol-Trials',
  MPI:     'AR CDG-Ib — MPI-423aa — 15q24.1 — Mannose-Phosphate-Isomerase — NO-Neurology-KEY-DDx — Protein-Losing-Enteropathy-Hepatopathy — D-Mannose-1gkgday-CURATIVE',
  ALG6:    'AR CDG-Ic — ALG6-507aa — 1p31.3 — Glucosyltransferase-I — 2nd-Most-Common-N-Glyc-CDG — Milder-Than-PMM2 — LLO-Man9-Accumulation — No-Inverted-Nipples',
  PGM1:    'AR CDG-PGM1 — PGM1-562aa — 1p31.3 — Phosphoglucomutase-1 — Bifid-Uvula-PATHOGNOMONIC — DCM-Hepatopathy-Rhabdomyolysis — Galactose-0.5gkgday-TREATABLE',
  SLC35A2: 'XL-De-Novo CDG-IIm — SLC35A2-396aa — Xp11.23 — UDP-Galactose-Golgi-Transporter — First-X-Linked-CDG — De-Novo-Females-Males-Non-Viable — Epilepsy-ID-Dysmorphism',
  SLC35C1: 'AR CDG-IIc/LAD-II — SLC35C1-364aa — 11p11.2 — GDP-Fucose-Golgi-Transporter — Bombay-Blood-Group-PATHOGNOMONIC — Absent-Sialyl-LewisX — L-Fucose-500mgkgday-TREATABLE',
  DOLK:    'AR CDG-Im — DOLK-538aa — 9q34.11 — Dolichol-Kinase — DCM-Dominant-Feature — Ichthyosis-From-Birth — Hepatopathy — G301R-Irish-Traveller-Founder',
  COG7:    'AR CDG-IIe — COG7-841aa — 16p12.2 — COG-Complex-Subunit-7 — Wrinkled-Aged-Skin-PATHOGNOMONIC — Severe-Neonatal-Liver-Failure — High-Mortality — West-African-Founder',
};

function Loading() {
  return <div className="text-center py-5"><div className="spinner-border text-primary" role="status" /><p className="mt-2 text-muted">Loading CDG Atlas data…</p></div>;
}

function StatCard({ label, value, sub, color }) {
  return (
    <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${color || '#6c757d'}` }}>
      <div className="card-body py-2 px-3">
        <div className="fw-bold" style={{ fontSize: '1.4rem', color: color || '#333' }}>{value}</div>
        <div className="text-muted small">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const { aggregate_stats: s, gene_summary: gs, top_alerts: alerts, critical_treatment_alerts: tx } = data;
  return (
    <div>
      <div className="alert alert-primary mb-3">
        <strong>Hereditary CDG Atlas</strong> — {data.total_patients} patients across 8 CDG genes (seeds {data.seed_range}).<br />
        <small className="text-muted">{data.subtitle}</small>
      </div>

      {/* Aggregate stats */}
      <div className="row g-2 mb-3">
        <div className="col-6 col-md-3"><StatCard label="Total Patients" value={data.total_patients} color="#0d6efd" /></div>
        <div className="col-6 col-md-3"><StatCard label="Genes Covered" value={s.genes_covered} sub="AR×6 XL×1 DeNovo×1" color="#6610f2" /></div>
        <div className="col-6 col-md-3"><StatCard label="On Specific Rx" value={`${s.on_specific_treatment_pct}%`} sub="MPI/PGM1/SLC35C1" color="#198754" /></div>
        <div className="col-6 col-md-3"><StatCard label="Severe Cases" value={`${s.severity_severe_pct}%`} sub="includes COG7 neonatal" color="#dc3545" /></div>
        <div className="col-6 col-md-3"><StatCard label="Neurology" value={`${s.neurology_pct}%`} sub="absent in MPI" color="#0dcaf0" /></div>
        <div className="col-6 col-md-3"><StatCard label="Hepatopathy" value={`${s.hepatopathy_pct}%`} sub="MPI/PGM1/DOLK/COG7" color="#fd7e14" /></div>
        <div className="col-6 col-md-3"><StatCard label="Cardiac DCM" value={`${s.cardiac_pct}%`} sub="PGM1/DOLK dominant" color="#e83e8c" /></div>
        <div className="col-6 col-md-3"><StatCard label="Coagulopathy" value={`${s.coagulopathy_pct}%`} sub="Protein C/S — PMM2/MPI/COG7" color="#6f42c1" /></div>
      </div>

      {/* Transferrin IEF pattern breakdown */}
      <div className="card mb-3">
        <div className="card-header fw-bold">Transferrin IEF Pattern Distribution</div>
        <div className="card-body">
          <div className="row g-2">
            <div className="col-4">
              <div className="text-center p-2 rounded" style={{ background: '#fff3cd' }}>
                <div className="fw-bold">{s.transferrin_type_i_pct}%</div>
                <div className="small">Type I</div>
                <div style={{ fontSize: '0.65rem' }} className="text-muted">PMM2, MPI, ALG6, DOLK</div>
              </div>
            </div>
            <div className="col-4">
              <div className="text-center p-2 rounded" style={{ background: '#cff4fc' }}>
                <div className="fw-bold">{s.transferrin_type_ii_pct}%</div>
                <div className="small">Type II</div>
                <div style={{ fontSize: '0.65rem' }} className="text-muted">SLC35A2, SLC35C1, COG7</div>
              </div>
            </div>
            <div className="col-4">
              <div className="text-center p-2 rounded" style={{ background: '#d1e7dd' }}>
                <div className="fw-bold">{s.transferrin_mixed_pct}%</div>
                <div className="small">Mixed I/II</div>
                <div style={{ fontSize: '0.65rem' }} className="text-muted">PGM1 — PATHOGNOMONIC</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Top clinical alerts */}
      <div className="card mb-3 border-danger">
        <div className="card-header fw-bold text-danger">⚠ Top Clinical Alerts</div>
        <div className="card-body">
          <ul className="mb-0" style={{ fontSize: '0.8rem' }}>
            {(alerts || []).slice(0, 8).map((a, i) => (
              <li key={i}><code className="text-danger">{a.split(' — ')[0]}</code>{a.includes(' — ') ? ' — ' + a.split(' — ').slice(1).join(' — ') : ''}</li>
            ))}
          </ul>
        </div>
      </div>

      {/* Treatment alerts */}
      <div className="card mb-3 border-success">
        <div className="card-header fw-bold text-success">💊 Critical Treatment Alerts</div>
        <div className="card-body">
          <ul className="mb-0" style={{ fontSize: '0.8rem' }}>
            {(tx || []).map((a, i) => (
              <li key={i}><code className="text-success">{a.split(' — ')[0]}</code>{a.includes(' — ') ? ' — ' + a.split(' — ').slice(1).join(' — ') : ''}</li>
            ))}
          </ul>
        </div>
      </div>

      {/* Gene summary cards */}
      <div className="row g-2">
        {(gs || []).map(g => (
          <div key={g.gene} className="col-12 col-md-6">
            <div className="card h-100 shadow-sm" style={{ borderTop: `3px solid ${GENE_COLORS[g.gene] || '#888'}` }}>
              <div className="card-body py-2 px-3">
                <div className="fw-bold" style={{ color: GENE_COLORS[g.gene] }}>{g.gene}</div>
                <div style={{ fontSize: '0.72rem' }} className="text-muted mb-1">{g.disease_short}</div>
                <div style={{ fontSize: '0.7rem' }}><strong>Key:</strong> {g.key_finding}</div>
                <div style={{ fontSize: '0.7rem' }}><strong>Rx:</strong> {g.management_pearl}</div>
                <div className="mt-1 d-flex gap-2" style={{ fontSize: '0.65rem' }}>
                  <span className="badge bg-secondary">{g.chromosome}</span>
                  <span className="badge bg-light text-dark">{g.protein_size_aa} aa</span>
                  <span className="badge bg-warning text-dark">{g.severe_n}/{g.n_patients} severe</span>
                  {g.on_treatment_n > 0 && <span className="badge bg-success">Rx: {g.on_treatment_n}</span>}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown_by_gene: byGene } = data;
  const genes = Object.keys(byGene || {});
  return (
    <div>
      <div className="table-responsive">
        <table className="table table-sm table-bordered table-hover" style={{ fontSize: '0.72rem' }}>
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Disease</th><th>Chr</th><th>Size</th><th>Inherit</th>
              <th>Transferrin</th><th>N</th><th>Severe</th><th>Neurology</th>
              <th>Hepatopathy</th><th>Cardiac</th><th>On Rx</th><th>Pathognomonic</th>
            </tr>
          </thead>
          <tbody>
            {genes.map(g => {
              const row = byGene[g];
              return (
                <tr key={g}>
                  <td><span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span></td>
                  <td style={{ maxWidth: 120 }}><small>{GENE_DISEASE[g]?.split(' — ')[1] || ''}</small></td>
                  <td>{row.locus}</td>
                  <td>{row.protein_size}</td>
                  <td><small>{row.inheritance.replace('Autosomal recessive', 'AR').replace('X-linked de novo dominant', 'XL-dN').replace('(biallelic', '').replace('mutations)', '').trim()}</small></td>
                  <td>
                    {row.transferrin_type_i_n > 0 && <span className="badge bg-warning text-dark me-1">I({row.transferrin_type_i_n})</span>}
                    {row.transferrin_type_ii_n > 0 && <span className="badge bg-info text-dark me-1">II({row.transferrin_type_ii_n})</span>}
                    {row.transferrin_mixed_n > 0 && <span className="badge bg-success">Mix({row.transferrin_mixed_n})</span>}
                  </td>
                  <td>{row.n_patients}</td>
                  <td>{row.severe_n}</td>
                  <td>{row.neurology_n}</td>
                  <td>{row.hepatopathy_n}</td>
                  <td>{row.cardiac_n}</td>
                  <td>{row.on_treatment_n > 0 ? <span className="badge bg-success">{row.on_treatment_n}</span> : '—'}</td>
                  <td style={{ maxWidth: 120 }}><small>{row.pathognomonic?.split('=')[0]?.trim()}</small></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown_by_gene: byGene } = data;
  const genes = Object.keys(byGene || {});
  return (
    <div>
      {genes.map(g => {
        const row = byGene[g];
        return (
          <div key={g} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${GENE_COLORS[g] || '#888'}` }}>
            <div className="card-header py-2" style={{ background: `${GENE_COLORS[g]}18` }}>
              <span className="fw-bold" style={{ color: GENE_COLORS[g], fontSize: '1.1rem' }}>{g}</span>
              <span className="ms-2 text-muted small">{row.locus} · {row.protein_size} · {row.inheritance}</span>
            </div>
            <div className="card-body py-2 px-3">
              <div className="row g-2">
                <div className="col-md-6">
                  <div style={{ fontSize: '0.75rem' }}><strong>Protein:</strong> <span className="text-muted">{row.protein}</span></div>
                  <div style={{ fontSize: '0.75rem' }} className="mt-1"><strong>Age of onset:</strong> {row.age_of_onset}</div>
                  <div style={{ fontSize: '0.75rem' }}><strong>Key biomarker:</strong> {row.key_biomarker}</div>
                  <div style={{ fontSize: '0.75rem' }}><strong>Pathognomonic:</strong> {row.pathognomonic}</div>
                  <div style={{ fontSize: '0.75rem' }}><strong>Treatment:</strong> <span className="fw-bold text-success">{row.treatment}</span></div>
                </div>
                <div className="col-md-6">
                  <div className="d-flex flex-wrap gap-1 mb-2">
                    <span className="badge bg-primary">{row.n_patients} pts</span>
                    <span className="badge bg-danger">{row.severe_n} severe</span>
                    {row.neurology_n > 0 && <span className="badge bg-info text-dark">{row.neurology_n} neuro</span>}
                    {row.hepatopathy_n > 0 && <span className="badge bg-warning text-dark">{row.hepatopathy_n} liver</span>}
                    {row.cardiac_n > 0 && <span className="badge bg-danger">{row.cardiac_n} cardiac</span>}
                    {row.on_treatment_n > 0 && <span className="badge bg-success">{row.on_treatment_n} on Rx</span>}
                    {row.bifid_uvula_n > 0 && <span className="badge bg-teal text-dark" style={{ background: '#0d9488', color: '#fff' }}>{row.bifid_uvula_n} bifid uvula</span>}
                    {row.bombay_blood_group_n > 0 && <span className="badge bg-dark">{row.bombay_blood_group_n} Bombay</span>}
                    {row.cerebellar_hypoplasia_n > 0 && <span className="badge" style={{ background: '#dc2626', color: '#fff' }}>{row.cerebellar_hypoplasia_n} cerebellar hypo</span>}
                    {row.wrinkled_skin_n > 0 && <span className="badge bg-secondary">{row.wrinkled_skin_n} wrinkled skin</span>}
                    {row.ichthyosis_n > 0 && <span className="badge bg-warning text-dark">{row.ichthyosis_n} ichthyosis</span>}
                    {row.recurrent_infections_n > 0 && <span className="badge bg-danger">{row.recurrent_infections_n} infect</span>}
                  </div>
                  <div>
                    <strong style={{ fontSize: '0.72rem' }}>Critical flags:</strong>
                    <ul className="mb-0 mt-1" style={{ fontSize: '0.7rem' }}>
                      {(row.critical_flags || []).map((f, i) => (
                        <li key={i}><code className="text-danger">{f.split(' — ')[0]}</code>{f.includes(' — ') ? ' — ' + f.split(' — ').slice(1).join(' — ') : ''}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const { definitions: defs } = data;
  return (
    <div>
      {Object.entries(defs || {}).map(([title, body]) => (
        <div key={title} className="card mb-3 shadow-sm">
          <div className="card-header fw-bold py-2" style={{ background: '#f8f9fa' }}>{title}</div>
          <div className="card-body py-2">
            <p style={{ fontSize: '0.78rem', whiteSpace: 'pre-wrap', lineHeight: 1.6 }} className="mb-0">{body}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function HerediaryCDGAtlas() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [geneTable, setGeneTable] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-cdg-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/hereditary-cdg-atlas/breakdown`)
      .then(r => r.json()).then(setGeneTable).catch(() => {});
    fetch(`${API}/api/hereditary-cdg-atlas/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  if (error) return <div className="alert alert-danger m-3">Error: {error}</div>;

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-1">🧬 Hereditary CDG Atlas</h4>
      <p className="text-muted small mb-3">
        Complete 8-Gene Congenital Disorders of Glycosylation Atlas · 320 patients (8×40, seeds 1862–1869) ·
        PMM2 · MPI · ALG6 · PGM1 · SLC35A2 · SLC35C1 · DOLK · COG7
      </p>
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>
      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Gene Table' && <GeneTableTab data={geneTable} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={geneTable} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
