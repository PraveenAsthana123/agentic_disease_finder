'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#1a237e';  // deep indigo — amino acid metabolism
const LIGHT  = '#e8eaf6';
const COLOR2 = '#b71c1c';  // danger — lethal / no treatment / neurological
const COLOR3 = '#1b5e20';  // diet-controlled / treatment responsive
const COLOR4 = '#e65100';  // BH4/B6 responsive
const COLOR5 = '#006064';  // NBS
const COLOR6 = '#4a148c';  // transport defect
const COLOR7 = '#37474f';  // biomarker

const CLASS_COLORS = {
  hydroxylase:             '#1b5e20',
  dehydrogenase:           '#0d47a1',
  lyase:                   '#880e4f',
  hydrolase:               '#bf360c',
  transferase:             '#4a148c',
  decarboxylase:           '#c62828',
  transporter:             '#006064',
};

const CLASS_LABELS = {
  hydroxylase:   'Hydroxylase — PAH (phenylalanine-4-hydroxylase, PKU)',
  dehydrogenase: 'Dehydrogenase — BCKDHA (MSUD E1α), PHGDH (serine biosynthesis)',
  lyase:         'Lyase — CBS (cystathionine β-synthase, HCU type 1)',
  hydrolase:     'Hydrolase — FAH (fumarylacetoacetate hydrolase, HT1)',
  transferase:   'Transferase — TAT (tyrosine aminotransferase, HT2), OAT (gyrate atrophy)',
  decarboxylase: 'Decarboxylase — GLDC (glycine decarboxylase P-protein, NKH)',
  transporter:   'Transporter — SLC7A7 (y+LAT1, LPI), SLC6A19 (B0AT1, Hartnup)',
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

export default function AADisordersAtlasPage() {
  const [tab, setTab]             = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [err, setErr]             = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/aa-disorders-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/aa-disorders-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/aa-disorders-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center"><div className="spinner-border" style={{ color: COLOR }} /><p className="mt-2">Loading AA-Disorders-Atlas…</p></div>;
  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;

  const ac = overview?.aggregate_clinical || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-2 gap-3">
        <div style={{ width: 8, height: 48, background: COLOR, borderRadius: 4 }} />
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>AA-Disorders-Atlas — Amino Acid Disorders Atlas</h4>
          <small className="text-muted">
            10 genes · {overview?.n_patients} patients (10×40, seeds {overview?.seeds?.[0]}–{overview?.seeds?.[overview.seeds.length-1]}) ·
            PAH · BCKDHA · CBS · FAH · TAT · GLDC · OAT · SLC7A7 · PHGDH · SLC6A19
          </small>
        </div>
      </div>

      {/* Tab Nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === t ? ' active fw-semibold' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'Overview' && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Genes" value={overview?.n_genes} color={COLOR} />
            <KPI label="Patients" value={overview?.n_patients} color={COLOR} />
            <KPI label="Diet Controlled" value={`${ac.pct_diet_controlled}%`} color={COLOR3} />
            <KPI label="BH4-Responsive" value={`${ac.pct_bh4_responsive}%`} color={COLOR4} />
            <KPI label="Neurological" value={`${ac.pct_neurological}%`} color={COLOR2} />
            <KPI label="Deceased" value={`${ac.pct_deceased}%`} color={COLOR2} />
          </div>

          {/* Gene classes + aggregate */}
          <div className="row g-3 mb-4">
            <div className="col-md-6">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-header fw-semibold" style={{ background: COLOR, color: '#fff' }}>
                  Metabolic Pathway Distribution
                </div>
                <div className="card-body">
                  {overview?.gene_classes && Object.entries(overview.gene_classes).map(([cls, genes]) => (
                    <div key={cls} className="mb-2">
                      <span className="badge me-2" style={{ background: COLOR, fontSize: '0.7rem' }}>{cls.replace(/_/g, ' ')}</span>
                      <small>{genes.join(' · ')}</small>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-header fw-semibold" style={{ background: COLOR, color: '#fff' }}>
                  Aggregate Clinical Profile
                </div>
                <div className="card-body">
                  <BarRow label="Diet/treatment controlled" pct={ac.pct_diet_controlled} color={COLOR3} />
                  <BarRow label="BH4-responsive (PAH only)" pct={ac.pct_bh4_responsive} color={COLOR4} />
                  <BarRow label="Neurological involvement" pct={ac.pct_neurological} color={COLOR2} />
                  <BarRow label="Deceased (untreated/severe)" pct={ac.pct_deceased} color={COLOR2} />
                </div>
              </div>
            </div>
          </div>

          {/* Per-gene summary tiles */}
          <h6 className="fw-semibold mb-2" style={{ color: COLOR }}>Per-Gene Summary</h6>
          <div className="row g-3">
            {overview?.gene_summary?.map(g => (
              <div key={g.gene} className="col-md-6 col-lg-3">
                <div className="card h-100 shadow-sm border-0">
                  <div className="card-header py-2 fw-bold" style={{ background: CLASS_COLORS[g.gene_class] || COLOR, color: '#fff', fontSize: '0.85rem' }}>
                    {g.gene} <small className="opacity-75">{g.locus}</small>
                  </div>
                  <div className="card-body py-2 px-2" style={{ fontSize: '0.75rem' }}>
                    <div className="mb-1"><strong>NBS:</strong> {g.nbs_marker?.slice(0, 50)}</div>
                    <div className="mb-1"><strong>Biomarker:</strong> {g.key_biomarker?.slice(0, 50)}</div>
                    <div className="mb-1"><strong>Treatment:</strong> {g.diet_treatment?.slice(0, 50)}</div>
                    <div className="mb-1"><strong>Severity:</strong> {g.severity_spectrum?.slice(0, 50)}</div>
                    <BarRow label="Tx controlled" pct={g.pct_diet} color={COLOR3} />
                    <BarRow label="Neurological" pct={g.pct_neuro} color={COLOR2} />
                    <BarRow label="Deceased" pct={g.pct_deceased} color={COLOR2} />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Critical rules */}
          <div className="card border-0 shadow-sm mt-4">
            <div className="card-header fw-semibold" style={{ background: COLOR2, color: '#fff' }}>
              Critical Clinical Rules (Atlas-Wide)
            </div>
            <div className="card-body py-2">
              {overview?.critical_clinical_rules?.map((r, i) => (
                <div key={i} className="mb-2" style={{ fontSize: '0.8rem', borderLeft: `3px solid ${COLOR2}`, paddingLeft: 8 }}>
                  {r}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Gene Table ── */}
      {tab === 'Gene Table' && (
        <div className="table-responsive">
          <table className="table table-sm table-bordered table-hover" style={{ fontSize: '0.78rem' }}>
            <thead style={{ background: COLOR, color: '#fff', position: 'sticky', top: 0 }}>
              <tr>
                <th>Gene</th><th>Locus</th><th>Size</th><th>Class</th>
                <th>Phenotype (Short)</th>
                <th>Key Treatment</th>
                <th>NBS Marker</th><th>Key Biomarker</th>
                <th>Founder Variant</th><th>n</th>
                <th>Tx %</th><th>Neuro %</th><th>Deceased %</th>
              </tr>
            </thead>
            <tbody>
              {breakdown?.genes?.map(g => (
                <tr key={g.gene}>
                  <td><strong style={{ color: CLASS_COLORS[g.gene_class] || COLOR }}>{g.gene}</strong></td>
                  <td>{g.locus}</td>
                  <td>{g.aa}</td>
                  <td><span className="badge" style={{ background: CLASS_COLORS[g.gene_class] || COLOR, fontSize: '0.65rem' }}>{g.gene_class}</span></td>
                  <td style={{ maxWidth: 180, whiteSpace: 'normal' }}>{g.phenotype?.slice(0, 80)}…</td>
                  <td style={{ maxWidth: 130, whiteSpace: 'normal', fontSize: '0.7rem' }}>{g.diet_treatment?.slice(0, 60)}</td>
                  <td style={{ maxWidth: 120, whiteSpace: 'normal' }}>{g.nbs_marker?.slice(0, 50)}</td>
                  <td style={{ maxWidth: 120, whiteSpace: 'normal' }}>{g.key_biomarker?.slice(0, 50)}</td>
                  <td style={{ maxWidth: 130, whiteSpace: 'normal' }}>{g.founder_variant?.slice(0, 60)}</td>
                  <td>{g.n_patients}</td>
                  <td style={{ color: g.pct_diet > 60 ? COLOR3 : COLOR2 }}>{g.pct_diet}%</td>
                  <td style={{ color: g.pct_neuro > 50 ? COLOR2 : '#333' }}>{g.pct_neuro}%</td>
                  <td style={{ color: g.pct_deceased > 20 ? COLOR2 : '#333' }}>{g.pct_deceased}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Clinical Atlas ── */}
      {tab === 'Clinical Atlas' && (
        <div>
          {breakdown?.genes?.map(g => (
            <div key={g.gene} className="card mb-4 border-0 shadow-sm">
              <div className="card-header d-flex justify-content-between align-items-center py-2"
                style={{ background: CLASS_COLORS[g.gene_class] || COLOR, color: '#fff' }}>
                <div>
                  <strong>{g.gene}</strong>
                  <small className="ms-2 opacity-85">{g.alias}</small>
                </div>
                <small>{g.locus} · {g.aa} · OMIM {g.omim_gene}</small>
              </div>
              <div className="card-body py-2">
                <div className="row g-2">
                  <div className="col-md-4">
                    <div className="p-2 rounded" style={{ background: '#f8f9fa', fontSize: '0.78rem' }}>
                      <div className="fw-semibold mb-1">Inheritance</div>
                      <div>{g.inheritance}</div>
                    </div>
                  </div>
                  <div className="col-md-4">
                    <div className="p-2 rounded" style={{ background: '#fff3e0', fontSize: '0.78rem' }}>
                      <div className="fw-semibold mb-1">NBS / Biomarkers</div>
                      <div><strong>NBS:</strong> {g.nbs_marker}</div>
                      <div><strong>Key:</strong> {g.key_biomarker}</div>
                    </div>
                  </div>
                  <div className="col-md-4">
                    <div className="p-2 rounded" style={{ background: '#fce4ec', fontSize: '0.78rem' }}>
                      <div className="fw-semibold mb-1" style={{ color: COLOR2 }}>Critical CI</div>
                      <div>{g.critical_ci?.slice(0, 250)}</div>
                    </div>
                  </div>
                </div>
                <div className="mt-2" style={{ fontSize: '0.78rem' }}>
                  <div className="fw-semibold mb-1" style={{ color: CLASS_COLORS[g.gene_class] || COLOR }}>Hallmarks</div>
                  <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>{g.hallmark}</div>
                </div>
                <div className="mt-2" style={{ fontSize: '0.78rem' }}>
                  <div className="fw-semibold mb-1">Differential Diagnosis</div>
                  <div>{g.key_ddx}</div>
                </div>
                <div className="mt-2" style={{ fontSize: '0.78rem' }}>
                  <div className="fw-semibold mb-1" style={{ color: COLOR3 }}>Treatment</div>
                  <div>{g.diet_treatment}</div>
                </div>
                <div className="row g-2 mt-2">
                  <div className="col-sm-3">
                    <div className="p-2 rounded text-center" style={{ background: '#e8f5e9', fontSize: '0.75rem' }}>
                      <div className="fw-bold" style={{ color: COLOR3 }}>{g.pct_diet}%</div><small>Tx Controlled</small>
                    </div>
                  </div>
                  <div className="col-sm-3">
                    <div className="p-2 rounded text-center" style={{ background: '#fff3e0', fontSize: '0.75rem' }}>
                      <div className="fw-bold" style={{ color: COLOR4 }}>{g.pct_bh4}%</div><small>BH4/B6-Resp</small>
                    </div>
                  </div>
                  <div className="col-sm-3">
                    <div className="p-2 rounded text-center" style={{ background: '#fce4ec', fontSize: '0.75rem' }}>
                      <div className="fw-bold" style={{ color: COLOR2 }}>{g.pct_neuro}%</div><small>Neurological</small>
                    </div>
                  </div>
                  <div className="col-sm-3">
                    <div className="p-2 rounded text-center" style={{ background: '#fce4ec', fontSize: '0.75rem' }}>
                      <div className="fw-bold" style={{ color: COLOR2 }}>{g.pct_deceased}%</div><small>Deceased</small>
                    </div>
                  </div>
                </div>
                <div className="mt-2" style={{ fontSize: '0.75rem' }}>
                  <strong>Key variants: </strong>{g.key_variants?.join(' · ')}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'Definitions' && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-header fw-semibold" style={{ background: COLOR, color: '#fff' }}>
                  Atlas Scope
                </div>
                <div className="card-body" style={{ fontSize: '0.8rem' }}>
                  <div><strong>Atlas:</strong> {defs?.atlas}</div>
                  <div><strong>Genes in atlas:</strong> {defs?.aa_overview?.genes_in_atlas}</div>
                  <div><strong>Total known AA disorders:</strong> {defs?.aa_overview?.total_known_aa_disorders}</div>
                  <div><strong>Collective incidence:</strong> {defs?.aa_overview?.collective_incidence}</div>
                  <div><strong>Inheritance:</strong> {defs?.aa_overview?.inheritance_note}</div>
                  <div><strong>NBS note:</strong> {defs?.aa_overview?.nbs_note}</div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-header fw-semibold" style={{ background: COLOR, color: '#fff' }}>
                  Gene Class Legend
                </div>
                <div className="card-body" style={{ fontSize: '0.8rem' }}>
                  {Object.entries(CLASS_LABELS).map(([cls, label]) => (
                    <div key={cls} className="mb-2">
                      <span className="badge me-2" style={{ background: CLASS_COLORS[cls] || COLOR }}>{cls.replace(/_/g, ' ')}</span>
                      <small>{label}</small>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          <div className="row g-3">
            {defs?.definitions?.map((d, i) => (
              <div key={i} className="col-md-6">
                <div className="card border-0 shadow-sm h-100">
                  <div className="card-header py-2 fw-semibold" style={{ background: LIGHT, color: COLOR }}>
                    {d.term}
                  </div>
                  <div className="card-body py-2" style={{ fontSize: '0.8rem' }}>
                    {d.definition}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="mt-4 text-muted" style={{ fontSize: '0.72rem' }}>
        <Link href="/">← Back to Portal</Link> ·
        AA-Disorders-Atlas: {overview?.n_genes} genes · {overview?.n_patients} patients ·
        Seeds {overview?.seeds?.[0]}–{overview?.seeds?.[overview.seeds.length-1]} ·
        3 endpoints: /api/aa-disorders-atlas/overview|breakdown|definitions
      </div>
    </div>
  );
}
