'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#4a148c';  // deep purple — lysosomal / storage
const LIGHT  = '#f3e5f5';
const COLOR2 = '#b71c1c';  // danger — no-ERT / CRIM-negative / fatal
const COLOR3 = '#1b5e20';  // ERT available / gene therapy approved
const COLOR4 = '#e65100';  // SRT / miglustat
const COLOR5 = '#0d47a1';  // HSCT (pre-symptomatic window)
const COLOR6 = '#006064';  // NBS
const COLOR7 = '#37474f';  // biomarker / enzyme

const CLASS_COLORS = {
  glucosidase:              '#1b5e20',
  glycosidase:              '#0d47a1',
  sulfatase:                '#bf360c',
  phosphodiesterase:        '#880e4f',
  cholesterol_transporter:  '#4a148c',
};

const CLASS_LABELS = {
  glucosidase:              'Glycosidase — GBA (Gaucher β-glucocerebrosidase), GAA (Pompe acid maltase)',
  glycosidase:              'Glycosidase — GLA (Fabry α-galactosidase A, X-linked), GALC (Krabbe), HEXA (Tay-Sachs)',
  sulfatase:                'Sulfatase — ARSA (MLD arylsulfatase A)',
  phosphodiesterase:        'Phosphodiesterase — SMPD1 (Niemann-Pick A/B sphingomyelinase)',
  cholesterol_transporter:  'Cholesterol Transporter — NPC1 (Niemann-Pick C1 intracellular trafficking)',
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

export default function LSDAtlasPage() {
  const [tab, setTab]             = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [err, setErr]             = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/lsd-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/lsd-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/lsd-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center"><div className="spinner-border" style={{ color: COLOR }} /><p className="mt-2">Loading LSD-Atlas…</p></div>;
  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;

  const ac = overview?.aggregate_clinical || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-2 gap-3">
        <div style={{ width: 8, height: 48, background: COLOR, borderRadius: 4 }} />
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>LSD-Atlas — Lysosomal Storage Disorder Atlas</h4>
          <small className="text-muted">
            8 genes · {overview?.n_patients} patients (8×40, seeds {overview?.seeds?.[0]}–{overview?.seeds?.[overview.seeds.length-1]}) ·
            GBA · GLA · GAA · SMPD1 · NPC1 · GALC · ARSA · HEXA
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
            <KPI label="On ERT" value={`${ac.pct_ert_on}%`} color={COLOR3} />
            <KPI label="On SRT" value={`${ac.pct_srt_on}%`} color={COLOR4} />
            <KPI label="HSCT Done" value={`${ac.pct_hsct}%`} color={COLOR5} />
            <KPI label="Neurological" value={`${ac.pct_neurological}%`} color={COLOR2} />
            <KPI label="HCM" value={`${ac.pct_hcm}%`} color={COLOR2} />
            <KPI label="Splenomegaly" value={`${ac.pct_splenomegaly}%`} color={COLOR7} />
          </div>

          {/* Gene classes */}
          <div className="row g-3 mb-4">
            <div className="col-md-6">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-header fw-semibold" style={{ background: COLOR, color: '#fff' }}>
                  Gene Class Distribution
                </div>
                <div className="card-body">
                  {overview?.gene_classes && Object.entries(overview.gene_classes).map(([cls, genes]) => (
                    <div key={cls} className="mb-2">
                      <span className="badge me-2" style={{ background: CLASS_COLORS[cls] || COLOR }}>{cls.replace('_', ' ')}</span>
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
                  <BarRow label="On ERT" pct={ac.pct_ert_on} color={COLOR3} />
                  <BarRow label="On SRT (miglustat/eliglustat)" pct={ac.pct_srt_on} color={COLOR4} />
                  <BarRow label="HSCT Done" pct={ac.pct_hsct} color={COLOR5} />
                  <BarRow label="Neurological involvement" pct={ac.pct_neurological} color={COLOR2} />
                  <BarRow label="Cardiomyopathy (HCM)" pct={ac.pct_hcm} color={COLOR2} />
                  <BarRow label="Splenomegaly" pct={ac.pct_splenomegaly} color={COLOR7} />
                  <BarRow label="Deceased (progressive/untreated)" pct={ac.pct_deceased} color={COLOR2} />
                </div>
              </div>
            </div>
          </div>

          {/* Per-gene overview tiles */}
          <h6 className="fw-semibold mb-2" style={{ color: COLOR }}>Per-Gene Summary</h6>
          <div className="row g-3">
            {overview?.gene_summary?.map(g => (
              <div key={g.gene} className="col-md-6 col-lg-3">
                <div className="card h-100 shadow-sm border-0">
                  <div className="card-header py-2 fw-bold" style={{ background: CLASS_COLORS[g.gene_class] || COLOR, color: '#fff', fontSize: '0.85rem' }}>
                    {g.gene} <small className="opacity-75">{g.locus}</small>
                  </div>
                  <div className="card-body py-2 px-2" style={{ fontSize: '0.75rem' }}>
                    <div className="mb-1"><strong>NBS:</strong> {g.nbs_marker}</div>
                    <div className="mb-1"><strong>Biomarker:</strong> {g.key_biomarker}</div>
                    <div className="mb-1"><strong>ERT:</strong> <span style={{ color: g.ert_available?.startsWith('YES') ? COLOR3 : COLOR2 }}>{g.ert_available?.slice(0,40)}</span></div>
                    <div className="mb-1"><strong>Severity:</strong> {g.severity_spectrum?.slice(0,50)}</div>
                    <BarRow label="ERT on" pct={g.pct_ert} color={COLOR3} />
                    <BarRow label="Neuro" pct={g.pct_neuro} color={COLOR2} />
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
                <th>ERT</th><th>SRT</th><th>HSCT</th><th>Gene Therapy</th>
                <th>NBS Marker</th><th>Key Biomarker</th>
                <th>Founder Variant</th><th>n</th>
                <th>ERT %</th><th>Neuro %</th><th>HCM %</th><th>Deceased %</th>
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
                  <td style={{ color: g.ert_available?.startsWith('YES') ? COLOR3 : COLOR2, fontWeight: 600 }}>
                    {g.ert_available?.startsWith('YES') ? '✅ YES' : '❌ NO'}
                  </td>
                  <td style={{ color: g.srt_available?.startsWith('YES') ? COLOR4 : '#888' }}>
                    {g.srt_available?.startsWith('YES') ? '✅' : '—'}
                  </td>
                  <td>{g.hsct_role?.slice(0, 30)}</td>
                  <td style={{ maxWidth: 100, whiteSpace: 'normal', fontSize: '0.7rem' }}>{g.gene_therapy_status?.slice(0, 40)}</td>
                  <td style={{ maxWidth: 120, whiteSpace: 'normal' }}>{g.nbs_marker?.slice(0, 50)}</td>
                  <td style={{ maxWidth: 120, whiteSpace: 'normal' }}>{g.key_biomarker?.slice(0, 50)}</td>
                  <td style={{ maxWidth: 130, whiteSpace: 'normal' }}>{g.founder_variant?.slice(0, 60)}</td>
                  <td>{g.n_patients}</td>
                  <td>{g.pct_ert}%</td>
                  <td>{g.pct_neuro}%</td>
                  <td>{g.pct_hcm}%</td>
                  <td style={{ color: g.pct_deceased > 30 ? COLOR2 : '#333' }}>{g.pct_deceased}%</td>
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
                      <div>{g.critical_ci?.slice(0, 200)}</div>
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
                <div className="row g-2 mt-2">
                  <div className="col-sm-3">
                    <div className="p-2 rounded text-center" style={{ background: LIGHT, fontSize: '0.75rem' }}>
                      <div className="fw-bold" style={{ color: COLOR }}>{g.pct_ert}%</div><small>On ERT</small>
                    </div>
                  </div>
                  <div className="col-sm-3">
                    <div className="p-2 rounded text-center" style={{ background: '#e8f5e9', fontSize: '0.75rem' }}>
                      <div className="fw-bold" style={{ color: COLOR3 }}>{g.pct_srt}%</div><small>On SRT</small>
                    </div>
                  </div>
                  <div className="col-sm-3">
                    <div className="p-2 rounded text-center" style={{ background: '#e3f2fd', fontSize: '0.75rem' }}>
                      <div className="fw-bold" style={{ color: COLOR5 }}>{g.pct_hsct}%</div><small>HSCT</small>
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
                  <div><strong>Genes in atlas:</strong> {defs?.lsd_overview?.genes_in_atlas}</div>
                  <div><strong>Total known LSDs:</strong> {defs?.lsd_overview?.total_known_lsds}</div>
                  <div><strong>Collective incidence:</strong> {defs?.lsd_overview?.collective_incidence}</div>
                  <div><strong>Inheritance:</strong> {defs?.lsd_overview?.inheritance_note}</div>
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
        LSD-Atlas: {overview?.n_genes} genes · {overview?.n_patients} patients ·
        Seeds {overview?.seeds?.[0]}–{overview?.seeds?.[overview.seeds.length-1]} ·
        3 endpoints: /api/lsd-atlas/overview|breakdown|definitions
      </div>
    </div>
  );
}
