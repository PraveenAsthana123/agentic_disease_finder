'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-congenital-heart-disease-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  GATA4:  '#1565c0',  // deep blue   — isolated CHD (ASD/VSD/TOF), cardiac transcription factor
  TBX5:   '#b71c1c',  // deep red    — Holt-Oram, bilateral thumb + CHD
  'NKX2-5': '#e65100', // deep orange — ASD + progressive AV block
  NOTCH1: '#880e4f',  // deep pink   — BAV + aortopathy
  JAG1:   '#2e7d32',  // deep green  — Alagille, butterfly vertebrae
  CHD7:   '#4a148c',  // deep purple — CHARGE, coloboma + choanal atresia
  TFAP2B: '#f57f17',  // amber       — Char syndrome, PDA
  TBX1:   '#00695c',  // teal        — 22q11DS, conotruncal + hypocalcaemia
};

const GENE_INFO = {
  GATA4:  { full: 'GATA4 / 442aa',    locus: '8p23.1',  size: '442 aa / 50 kDa',  inh: 'AD', disease: 'Isolated CHD — ASD type II (40%), VSD (30%), TOF (20%); GATA4 haploinsufficiency; NO extracardia features (KEY DDx); incomplete penetrance ~85%; NKX2-5 interaction partner; p.Gly296Ser European variant' },
  TBX5:   { full: 'TBX5 / 518aa',     locus: '12q24.21',size: '518 aa / 58 kDa',  inh: 'AD', disease: 'Holt-Oram Syndrome — BILATERAL UPPER LIMB ANOMALY + HEART DEFECT PATHOGNOMONIC; thumb anomaly MOST SPECIFIC; ASD II (50%); 100% penetrance; limb severity ≠ heart severity; p.Tyr111Cys most common' },
  'NKX2-5':{ full: 'NKX2-5 / 324aa',  locus: '5q35.1',  size: '324 aa / 35 kDa',  inh: 'AD', disease: 'Isolated CHD + Progressive AV Block — ASD + PROGRESSIVE AV BLOCK PATHOGNOMONIC; block WORSENS after ASD repair; pacemaker 30-40% adulthood; 50% penetrance; LV noncompaction 5%; p.Arg25Cys homeodomain' },
  NOTCH1: { full: 'NOTCH1 / 2555aa',  locus: '9q34.3',  size: '2555 aa / 300 kDa',inh: 'AD', disease: 'BAV + Aortopathy — BICUSPID AORTIC VALVE + ASCENDING AORTIC DILATATION PATHOGNOMONIC; most common CHD (0.5-2%); calcific AS 5th decade; dissection risk independent of valve function; annual echo MANDATORY' },
  JAG1:   { full: 'JAG1 / 1218aa',    locus: '20p12.2', size: '1218 aa / 134 kDa',inh: 'AD', disease: 'Alagille Syndrome — BUTTERFLY VERTEBRAE 95% PATHOGNOMONIC (spine X-ray); posterior embryotoxon 78% PATHOGNOMONIC (slit-lamp); CHD 94% (PA stenosis/TOF); cholestatic liver 80%; 50% need liver transplant' },
  CHD7:   { full: 'CHD7 / 2997aa',    locus: '8q12.2',  size: '2997 aa / 340 kDa',inh: 'AD', disease: 'CHARGE Syndrome — COLOBOMA + HEART + CHOANAL ATRESIA PATHOGNOMONIC; semicircular canal APLASIA on MRI MOST PATHOGNOMONIC radiological finding; 75% conotruncal CHD; deafblind risk; ~95% de novo' },
  TFAP2B: { full: 'TFAP2B / 463aa',   locus: '6p24.3',  size: '463 aa / 52 kDa',  inh: 'AD', disease: 'Char Syndrome — PDA + FACIAL DYSMORPHISM + HAND ANOMALIES PATHOGNOMONIC; ductal tissue structurally abnormal; avoid indomethacin (low efficacy); catheter/surgical closure; ultra-rare ~30 families worldwide' },
  TBX1:   { full: 'TBX1 / 504aa',     locus: '22q11.21',size: '504 aa / 57 kDa',  inh: 'AD', disease: '22q11DS DiGeorge — CONOTRUNCAL CHD + HYPOCALCAEMIA + T-CELL LYMPHOPENIA PATHOGNOMONIC TRIAD; IAA-B/truncus/TOF-absent PV; most common microdeletion (1:4,000); schizophrenia 25%; check iCa ALL conotruncal CHD' },
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
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{ fontWeight: 800, fontSize: 18, color }}>{gene}</span>
        <Badge text={info.locus} color={color} />
        <Badge text={info.inh} color={color} />
        <Badge text={info.size} color="#555" />
      </div>
      <div style={{ fontSize: 13, color: '#333', lineHeight: 1.6 }}>{info.disease}</div>
      {data && (
        <div style={{ display: 'flex', gap: 12, marginTop: 10, flexWrap: 'wrap' }}>
          {data.avg_age_at_dx_yrs !== undefined && <span style={{ fontSize: 12, color: '#555' }}>Avg Age Dx: <b>{data.avg_age_at_dx_yrs}yr</b></span>}
          {data.n_patients !== undefined && <span style={{ fontSize: 12, color: '#555' }}>Patients: <b>{data.n_patients}</b></span>}
        </div>
      )}
    </div>
  );
}

export default function HCHDAtlasPage() {
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

  const TITLE = 'Hereditary Congenital Heart Disease Atlas';
  const SUBTITLE = 'Complete 8-Gene CHD Atlas — GATA4 · TBX5 · NKX2-5 · NOTCH1 · JAG1 · CHD7 · TFAP2B · TBX1';

  return (
    <div style={{ fontFamily: 'Inter, sans-serif', background: '#f5f7fa', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#880e4f 0%,#1565c0 100%)', color: '#fff', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 11, opacity: 0.8, marginBottom: 4 }}>🧬 Hereditary Disease Atlas Series</div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800 }}>{TITLE}</h1>
        <div style={{ fontSize: 13, opacity: 0.9, marginTop: 6 }}>{SUBTITLE}</div>
        <div style={{ fontSize: 12, opacity: 0.75, marginTop: 4 }}>320 synthetic patients · 8 genes · seeds 2254-2261</div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#fff', borderBottom: '2px solid #e0e0e0', display: 'flex', padding: '0 32px' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '12px 20px', border: 'none', background: 'none', cursor: 'pointer',
            fontWeight: tab === i ? 700 : 400, fontSize: 14,
            borderBottom: tab === i ? '3px solid #880e4f' : '3px solid transparent',
            color: tab === i ? '#880e4f' : '#555',
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
              <StatCard label="Total Patients" value={overview.n_patients} sub="8 × 40" color="#880e4f" />
              <StatCard label="CHD Genes" value={overview.n_genes} sub="Mendelian AD" color="#1565c0" />
              <StatCard label="Seeds" value={overview.seed_range} sub="2254-2261" color="#2e7d32" />
              <StatCard label="CHD Categories" value="6" sub="Septal/Valvular/Syndromic/Conotruncal/PDA/PA" color="#e65100" />
              <StatCard label="Pathognomonic" value="8" sub="one per gene" color="#4a148c" />
            </div>

            {/* CHD Category Table */}
            {overview.chd_categories && (
              <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 24, border: '1px solid #e0e0e0' }}>
                <h3 style={{ margin: '0 0 14px', color: '#333' }}>CHD Categories Covered</h3>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ background: '#f5f5f5' }}>
                      <th style={{ padding: '8px 12px', textAlign: 'left', fontSize: 12 }}>Category</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left', fontSize: 12 }}>Genes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(overview.chd_categories).map(([cat, genes], idx) => (
                      <tr key={cat} style={{ background: idx % 2 === 0 ? '#fafafa' : '#fff' }}>
                        <td style={{ padding: '8px 12px', fontSize: 13, fontWeight: 600, color: '#1565c0' }}>{cat.replace(/_/g, ' ')}</td>
                        <td style={{ padding: '8px 12px', fontSize: 12, color: '#555' }}>{genes}</td>
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
                    <span style={{ color: '#880e4f', fontWeight: 700 }}>•</span> {pearl}
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
            <h3 style={{ color: '#333', marginBottom: 16 }}>Clinical Atlas — 8-Gene CHD Spectrum (320 patients)</h3>

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
                  <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 12 }}>
                    <span style={{ fontWeight: 800, fontSize: 20, color }}>{gene}</span>
                    <Badge text={info.locus} color={color} />
                    <Badge text={info.protein_size} color="#555" />
                    <Badge text={`${info.n_patients} pts`} color={color} />
                    <Badge text={`Avg Dx ${info.avg_age_at_dx_yrs}yr`} color="#888" />
                  </div>

                  <div style={{ fontSize: 12, color: '#444', marginBottom: 12, lineHeight: 1.6 }}>
                    <b>PATHOGNOMONIC:</b> <span style={{ color: '#880e4f' }}>{info.pathognomonic}</span>
                  </div>

                  <div style={{ fontSize: 12, color: '#444', marginBottom: 12, lineHeight: 1.6 }}>
                    <b>Treatment:</b> {info.treatment_highlight}
                  </div>

                  {/* CHD types */}
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#555', marginBottom: 6 }}>CHD Type Distribution:</div>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {Object.entries(info.chd_type_distribution || {}).slice(0, 5).map(([chd, n]) => (
                        <span key={chd} style={{ background: color + '18', color, border: `1px solid ${color}44`, borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>
                          {chd}: {n}
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
              <h3 style={{ margin: '0 0 14px', color: '#333' }}>CHD Anatomy Glossary</h3>
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
