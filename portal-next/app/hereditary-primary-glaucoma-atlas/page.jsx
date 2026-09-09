'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-primary-glaucoma-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  MYOC:   '#1565c0',  // deep blue   — JOAG, highest IOP, trabecular meshwork
  CYP1B1: '#b71c1c',  // deep red    — PCG, buphthalmos, neonatal
  FOXC1:  '#e65100',  // deep orange — ARS3, iris hypoplasia, posterior embryotoxon
  PITX2:  '#880e4f',  // deep pink   — ARS1, iris processes, midface/dental
  PAX6:   '#2e7d32',  // deep green  — aniridia, LSCD, WAGR
  OPTN:   '#4a148c',  // deep purple — NTG, normal IOP, disc haemorrhages, ALS
  LTBP2:  '#f57f17',  // amber       — PCG + microspherophakia, Gulf Arab AR
  TEK:    '#00695c',  // teal        — PCG + absent Schlemm's canal, AD
};

const GENE_INFO = {
  MYOC:   { full: 'MYOC / 504aa',   locus: '1q24.3',  size: '504 aa / 55 kDa',   inh: 'AD', disease: 'JOAG / Adult POAG — HIGHEST IOP (30-50 mmHg) PATHOGNOMONIC in hereditary glaucoma; trabecular meshwork misfolding + ER stress → TM cell death; avoid topical steroids (steroid-response); p.Gln368STOP most common (1-2% POAG); p.Pro370Leu severe JOAG; trabeculectomy highly effective' },
  CYP1B1: { full: 'CYP1B1 / 543aa', locus: '2p22.2',  size: '543 aa / 60 kDa',   inh: 'AR', disease: 'PCG (Primary Congenital Glaucoma) — BUPHTHALMOS + HAAB STRIAE PATHOGNOMONIC; photophobia + epiphora + blepharospasm classic triad; goniotomy/trabeculotomy surgical first-line (AVOID brimonidine <2yr); p.Arg368His Arab/Turkish/Pakistani most common; most common PCG gene globally' },
  FOXC1:  { full: 'FOXC1 / 553aa',  locus: '6p25.3',  size: '553 aa / 60 kDa',   inh: 'AD', disease: 'ARS3 (Axenfeld-Rieger Syndrome type 3) — IRIS HYPOPLASIA + POSTERIOR EMBRYOTOXON (100%) PATHOGNOMONIC; iridocorneal adhesions; glaucoma 50-80%; dental hypodontia; MLPA mandatory (CNV 20%); pituitary (empty sella) in 10%; goniosynechialysis first angle surgery' },
  PITX2:  { full: 'PITX2 / 317aa',  locus: '4q25',    size: '317 aa / 35 kDa',   inh: 'AD', disease: 'ARS1 (Axenfeld-Rieger Syndrome type 1) — IRIDOCORNEAL ADHESIONS + POSTERIOR EMBRYOTOXON PATHOGNOMONIC; midface hypoplasia; hypodontia/peg teeth; umbilical stump residue; glaucoma 50-70%; cardiac septal defects rare (5%); goniosynechialysis first angle surgery' },
  PAX6:   { full: 'PAX6 / 422aa',   locus: '11p13',   size: '422 aa / 46 kDa',   inh: 'AD', disease: 'Aniridia + ARG — IRIS ABSENT PATHOGNOMONIC; LSCD keratopathy; foveal hypoplasia → nystagmus; cataract (80%); aniridia-related glaucoma (ARG) 30-50%; WAGR (11p13 deletion): Wilms tumour risk → CMA MANDATORY; tube shunt preferred (AVOID MMC near limbus)' },
  OPTN:   { full: 'OPTN / 577aa',   locus: '10p13',   size: '577 aa / 74 kDa',   inh: 'AD', disease: 'NTG (Normal Tension Glaucoma) — NORMAL IOP (<21 mmHg) + PROGRESSIVE VF LOSS PATHOGNOMONIC; disc haemorrhages more frequent than POAG; E50K = 10× NTG risk + ALS overlap (screen family); mitophagy pathway; brimonidine neuroprotection; 24h BP monitoring (nocturnal hypotension)' },
  LTBP2:  { full: 'LTBP2 / 1821aa', locus: '14q24.3', size: '1821 aa / 200 kDa', inh: 'AR', disease: 'PCG + Microspherophakia — SPHERICAL SUBLUXATED LENS + MEGALOCORNEA PATHOGNOMONIC; ectopia lentis; pupil block → acute angle closure (AVOID miotics/pilocarpine); Gulf Arab AR founder; Weill-Marchesani overlap; lens extraction specialist centre; goniotomy + LPI for pupil block' },
  TEK:    { full: 'TEK / 1124aa',   locus: '9p21.2',  size: '1124 aa / 125 kDa', inh: 'AD', disease: 'PCG + Schlemm\'s Canal Dysgenesis — SCHLEMM\'S CANAL ABSENT ON AS-OCT PATHOGNOMONIC; angiopoietin-TEK pathway; trabeculotomy preferred (not goniotomy, canal absent); early tube shunt; AD unlike CYP1B1 (AR); distinct from TEK-venous malformations (different mutations)' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 7px', fontSize: 11, fontWeight: 700, marginRight: 4,
    }}>{text}</span>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: '#fff', border: `2px solid ${color || '#e0e0e0'}`,
      borderRadius: 10, padding: '14px 18px', minWidth: 120, textAlign: 'center',
    }}>
      <div style={{ fontSize: 26, fontWeight: 800, color: color || '#333' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#555', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#888' }}>{sub}</div>}
    </div>
  );
}

function GeneCard({ gene, color, info, data }) {
  return (
    <div style={{
      border: `2px solid ${color}`, borderRadius: 10, padding: 16, marginBottom: 12,
      background: color + '08',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
        <span style={{ fontWeight: 800, fontSize: 18, color }}>{gene}</span>
        <Badge text={info.locus} color={color} />
        <Badge text={info.inh} color={color} />
        <Badge text={info.size} color="#555" />
      </div>
      <div style={{ fontSize: 13, color: '#333', lineHeight: 1.6 }}>{info.disease}</div>
      {data && (
        <div style={{ display: 'flex', gap: 12, marginTop: 10, flexWrap: 'wrap' }}>
          {data.avg_age_at_dx_yrs !== undefined && <span style={{ fontSize: 12, color: '#555' }}>Avg Age Dx: <b>{data.avg_age_at_dx_yrs}yr</b></span>}
          {data.avg_iop_mmhg !== undefined && <span style={{ fontSize: 12, color: '#555' }}>Avg IOP: <b>{data.avg_iop_mmhg} mmHg</b></span>}
          {data.n_patients !== undefined && <span style={{ fontSize: 12, color: '#555' }}>Patients: <b>{data.n_patients}</b></span>}
        </div>
      )}
    </div>
  );
}

export default function HeredPrimaryGlaucomaAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov); setBreakdown(bk); setDefs(df);
    }).catch(e => setError(e.message)).finally(() => setLoading(false));
  }, []);

  const TITLE = 'Hereditary Primary Glaucoma Atlas';
  const SUBTITLE = 'Complete 8-Gene Hereditary Glaucoma Atlas — MYOC · CYP1B1 · FOXC1 · PITX2 · PAX6 · OPTN · LTBP2 · TEK';

  return (
    <div style={{ fontFamily: 'Inter, sans-serif', background: '#f5f7fa', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1565c0 0%,#4a148c 100%)', color: '#fff', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 11, opacity: 0.8, marginBottom: 4 }}>🧬 Hereditary Disease Atlas Series</div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800 }}>{TITLE}</h1>
        <div style={{ fontSize: 13, opacity: 0.9, marginTop: 6 }}>{SUBTITLE}</div>
        <div style={{ fontSize: 12, opacity: 0.75, marginTop: 4 }}>320 synthetic patients · 8 genes · seeds 2262-2269</div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#fff', borderBottom: '2px solid #e0e0e0', display: 'flex', padding: '0 32px' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '12px 20px', border: 'none', background: 'none', cursor: 'pointer',
            fontWeight: tab === i ? 700 : 400, fontSize: 14,
            borderBottom: tab === i ? '3px solid #1565c0' : '3px solid transparent',
            color: tab === i ? '#1565c0' : '#555',
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 32px' }}>
        {loading && <div style={{ color: '#888', padding: 32 }}>Loading atlas data…</div>}
        {error && <div style={{ color: '#c62828', padding: 16 }}>Error: {error}</div>}

        {/* ── Overview Tab ── */}
        {tab === 0 && overview && (
          <div>
            {/* Stat Cards */}
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 28 }}>
              <StatCard label="Total Patients" value={overview.n_patients} sub="8 × 40" color="#1565c0" />
              <StatCard label="Glaucoma Genes" value={overview.n_genes} sub="Mendelian AD + AR" color="#4a148c" />
              <StatCard label="Seeds" value={overview.seed_range} sub="2262-2269" color="#2e7d32" />
              <StatCard label="Categories" value="6" sub="JOAG/PCG/ARS/Aniridia/NTG/Sphere" color="#e65100" />
              <StatCard label="Pathognomonic" value="8" sub="one per gene" color="#880e4f" />
            </div>

            {/* Glaucoma Category Table */}
            {overview.glaucoma_categories && (
              <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 24, border: '1px solid #e0e0e0' }}>
                <h3 style={{ margin: '0 0 14px', color: '#333' }}>Glaucoma Categories Covered</h3>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ background: '#f5f5f5' }}>
                      <th style={{ padding: '8px 12px', textAlign: 'left', fontSize: 12 }}>Category</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left', fontSize: 12 }}>Genes / Key Features</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(overview.glaucoma_categories).map(([cat, desc], idx) => (
                      <tr key={cat} style={{ background: idx % 2 === 0 ? '#fafafa' : '#fff' }}>
                        <td style={{ padding: '8px 12px', fontSize: 13, fontWeight: 600, color: '#1565c0' }}>{cat.replace(/_/g, ' ')}</td>
                        <td style={{ padding: '8px 12px', fontSize: 12, color: '#555' }}>{desc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Key Clinical Pearls */}
            {overview.key_clinical_pearls && (
              <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 24, border: '1px solid #e0e0e0' }}>
                <h3 style={{ margin: '0 0 14px', color: '#333' }}>Key Clinical Pearls</h3>
                {overview.key_clinical_pearls.map((pearl, i) => (
                  <div key={i} style={{ padding: '8px 0', borderBottom: i < overview.key_clinical_pearls.length - 1 ? '1px solid #f0f0f0' : 'none', fontSize: 13, lineHeight: 1.6 }}>
                    <span style={{ color: '#1565c0', fontWeight: 700 }}>•</span> {pearl}
                  </div>
                ))}
              </div>
            )}

            {/* Gene Summary Grid */}
            {overview.gene_summary && (
              <div style={{ background: '#fff', borderRadius: 10, padding: 20, border: '1px solid #e0e0e0', marginBottom: 24 }}>
                <h3 style={{ margin: '0 0 14px', color: '#333' }}>Gene Summary</h3>
                {overview.gene_summary.map(gs => {
                  const color = GENE_COLORS[gs.gene] || '#555';
                  return (
                    <div key={gs.gene} style={{ display: 'flex', alignItems: 'flex-start', gap: 12, padding: '10px 0', borderBottom: '1px solid #f5f5f5' }}>
                      <span style={{ fontWeight: 800, color, minWidth: 80, fontSize: 14 }}>{gs.gene}</span>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 12, color: '#888' }}>{gs.locus} · {gs.protein_size} · {gs.inheritance}</div>
                        <div style={{ fontSize: 12, color: '#444', marginTop: 3 }}>{gs.pathognomonic}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Diagnostic Algorithm */}
            {overview.diagnostic_algorithm && (
              <div style={{ background: '#fff', borderRadius: 10, padding: 20, border: '1px solid #e0e0e0' }}>
                <h3 style={{ margin: '0 0 14px', color: '#333' }}>Diagnostic Algorithm</h3>
                {Object.entries(overview.diagnostic_algorithm).map(([step, action]) => (
                  <div key={step} style={{ marginBottom: 10 }}>
                    <span style={{ fontSize: 12, fontWeight: 700, color: '#1565c0' }}>{step.replace(/_/g, ' ')}: </span>
                    <span style={{ fontSize: 12, color: '#444' }}>{action}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── Gene Table Tab ── */}
        {tab === 1 && (
          <div>
            {Object.entries(GENE_INFO).map(([gene, info]) => {
              const bkData = breakdown?.gene_breakdown?.[gene];
              return (
                <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene] || '#555'} info={info} data={bkData} />
              );
            })}
          </div>
        )}

        {/* ── Clinical Atlas Tab ── */}
        {tab === 2 && breakdown && (
          <div>
            <h3 style={{ color: '#333', marginBottom: 16 }}>Clinical Atlas — 8-Gene Hereditary Glaucoma Spectrum (320 patients)</h3>

            {/* Emergency Flags */}
            {breakdown.clinical_emergency_flags && (
              <div style={{ background: '#fff3e0', border: '2px solid #e65100', borderRadius: 10, padding: 16, marginBottom: 20 }}>
                <div style={{ fontWeight: 700, color: '#e65100', marginBottom: 8 }}>🚨 Clinical Emergency Flags</div>
                {breakdown.clinical_emergency_flags.map((flag, i) => (
                  <div key={i} style={{ fontSize: 13, color: '#333', padding: '4px 0' }}>• {flag}</div>
                ))}
              </div>
            )}

            {Object.entries(breakdown.gene_breakdown || {}).map(([gene, info]) => {
              const color = GENE_COLORS[gene] || '#555';
              return (
                <div key={gene} style={{ background: '#fff', border: `2px solid ${color}`, borderRadius: 10, padding: 20, marginBottom: 20 }}>
                  <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
                    <span style={{ fontWeight: 800, fontSize: 20, color }}>{gene}</span>
                    <Badge text={info.locus} color={color} />
                    <Badge text={info.protein_size} color="#555" />
                    <Badge text={`${info.n_patients} pts`} color={color} />
                    <Badge text={`Avg IOP ${info.avg_iop_mmhg} mmHg`} color={color} />
                    <Badge text={`Avg Dx ${info.avg_age_at_dx_yrs}yr`} color="#888" />
                  </div>

                  <div style={{ fontSize: 12, color: '#444', marginBottom: 12, lineHeight: 1.6 }}>
                    <b>PATHOGNOMONIC:</b> <span style={{ color: '#1565c0' }}>{info.pathognomonic}</span>
                  </div>

                  <div style={{ fontSize: 12, color: '#444', marginBottom: 12, lineHeight: 1.6 }}>
                    <b>Treatment:</b> {info.treatment_highlight}
                  </div>

                  {/* Glaucoma types */}
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#555', marginBottom: 6 }}>Glaucoma Type Distribution:</div>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {Object.entries(info.glaucoma_type_distribution || {}).slice(0, 5).map(([type, n]) => (
                        <span key={type} style={{ background: color + '18', color, border: `1px solid ${color}44`, borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>
                          {type}: {n}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Key features */}
                  {info.key_features && (
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 700, color: '#555', marginBottom: 6 }}>Key Features:</div>
                      {info.key_features.slice(0, 4).map((f, i) => (
                        <div key={i} style={{ fontSize: 12, color: '#444', padding: '3px 0' }}>• {f}</div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* ── Definitions Tab ── */}
        {tab === 3 && defs && (
          <div>
            {/* Gene Entries */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 20, border: '1px solid #e0e0e0' }}>
              <h3 style={{ margin: '0 0 14px', color: '#333' }}>Gene Definitions</h3>
              {Object.entries(defs.gene_entries || {}).map(([gene, entry]) => {
                const color = GENE_COLORS[gene] || '#555';
                return (
                  <details key={gene} style={{ borderBottom: '1px solid #f0f0f0', paddingBottom: 10, marginBottom: 10 }}>
                    <summary style={{ fontWeight: 700, color, cursor: 'pointer', fontSize: 14, padding: '4px 0' }}>{gene}</summary>
                    <div style={{ paddingTop: 10 }}>
                      <div style={{ fontSize: 12, color: '#666', marginBottom: 8, lineHeight: 1.5 }}><b>Inheritance:</b> {entry.inheritance_details?.slice(0, 400)}</div>
                      <div style={{ marginBottom: 8 }}>
                        <b style={{ fontSize: 12 }}>Key Features:</b>
                        {entry.key_features?.slice(0, 4).map((f, i) => (
                          <div key={i} style={{ fontSize: 12, color: '#444', padding: '2px 0' }}>• {f}</div>
                        ))}
                      </div>
                      <div style={{ fontSize: 12, color: '#444', lineHeight: 1.6 }}>
                        <b>Treatment:</b> {entry.treatment?.slice(0, 400)}…
                      </div>
                    </div>
                  </details>
                );
              })}
            </div>

            {/* Anatomy Glossary */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 20, border: '1px solid #e0e0e0' }}>
              <h3 style={{ margin: '0 0 14px', color: '#333' }}>Glaucoma Anatomy Glossary</h3>
              {Object.entries(defs.anatomy_glossary || {}).map(([term, def]) => (
                <div key={term} style={{ padding: '8px 0', borderBottom: '1px solid #f5f5f5' }}>
                  <span style={{ fontWeight: 700, color: '#1565c0', fontSize: 13 }}>{term}:</span>{' '}
                  <span style={{ fontSize: 12, color: '#555' }}>{def}</span>
                </div>
              ))}
            </div>

            {/* Syndrome Glossary */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 20, border: '1px solid #e0e0e0' }}>
              <h3 style={{ margin: '0 0 14px', color: '#333' }}>Syndrome Glossary</h3>
              {Object.entries(defs.syndrome_glossary || {}).map(([syndrome, def]) => (
                <div key={syndrome} style={{ padding: '8px 0', borderBottom: '1px solid #f5f5f5' }}>
                  <span style={{ fontWeight: 700, color: '#880e4f', fontSize: 13 }}>{syndrome}:</span>{' '}
                  <span style={{ fontSize: 12, color: '#555' }}>{def}</span>
                </div>
              ))}
            </div>

            {/* Treatment Glossary */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 20, border: '1px solid #e0e0e0' }}>
              <h3 style={{ margin: '0 0 14px', color: '#333' }}>Treatment Glossary</h3>
              {Object.entries(defs.treatment_glossary || {}).map(([term, def]) => (
                <div key={term} style={{ padding: '8px 0', borderBottom: '1px solid #f5f5f5' }}>
                  <span style={{ fontWeight: 700, color: '#2e7d32', fontSize: 13 }}>{term}:</span>{' '}
                  <span style={{ fontSize: 12, color: '#555' }}>{def}</span>
                </div>
              ))}
            </div>

            {/* Diagnostic Tests */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, border: '1px solid #e0e0e0' }}>
              <h3 style={{ margin: '0 0 14px', color: '#333' }}>Diagnostic Tests</h3>
              {Object.entries(defs.diagnostic_tests || {}).map(([test, def]) => (
                <div key={test} style={{ padding: '8px 0', borderBottom: '1px solid #f5f5f5' }}>
                  <span style={{ fontWeight: 700, color: '#e65100', fontSize: 13 }}>{test.replace(/_/g, ' ')}:</span>{' '}
                  <span style={{ fontSize: 12, color: '#555' }}>{def}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
