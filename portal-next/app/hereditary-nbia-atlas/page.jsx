'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-nbia-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'PANK2':   '#b71c1c',  // deep red — most common NBIA ~50%; eye-of-tiger PATHOGNOMONIC
  'PLA2G6':  '#4a148c',  // deep purple — PLAN; INAD; PARK14; spheroids EM PATHOGNOMONIC
  'WDR45':   '#880e4f',  // dark magenta — BPAN; females; de novo; biphasic PATHOGNOMONIC
  'C19orf12':'#1565c0',  // deep blue — MPAN; Polish founder; optic atrophy + neuropathy
  'FA2H':    '#1b5e20',  // dark green — FAHN/SPG35; leukodystrophy earliest; thin CC
  'ATP13A2': '#e65100',  // burnt orange — KRS/PARK9; juvenile Parkinson; levodopa-responsive
  'COASY':   '#4e342e',  // brown — CoPAN; CoA pathway same as PANK2; spasticity dominant
  'DCAF17':  '#006064',  // dark teal — WSS; UNIQUE multisystem; hypogonadism+alopecia+DM+SNHL
};

const GENE_INFO = {
  'PANK2':   { full: 'PANK2 / Pantothenate Kinase 2 / 570aa', locus: '2p13.3', size: '570 aa / 63 kDa', inh: 'AR', disease: 'PKAN (NBIA1) ~50% of all NBIA; Eye-of-Tiger T2-MRI PATHOGNOMONIC; dystonia dominant; deferiprone (B-PKAN 2022); GPi DBS palliation' },
  'PLA2G6':  { full: 'PLA2G6 / iPLA2-VIA / 806aa', locus: '22q13.1', size: '806 aa / 88 kDa', inh: 'AR', disease: 'PLAN (NBIA2/PARK14); Neuroaxonal spheroids EM PATHOGNOMONIC; optic ATROPHY; INAD infantile / atypical / PARK14 adult levodopa-responsive' },
  'WDR45':   { full: 'WDR45 / WIPI4 / 361aa', locus: 'Xp11.23', size: '361 aa / 40 kDa', inh: 'XLD de novo', disease: 'BPAN (NBIA5); Females; BIPHASIC PATHOGNOMONIC: childhood seizures → adult Parkinsonism+dementia; T1 halo sign; >95% de novo' },
  'C19orf12':{ full: 'C19orf12 / Mitochondrial 141aa', locus: '19q12', size: '141 aa / 16 kDa', inh: 'AR', disease: 'MPAN (NBIA4); Optic atrophy + motor neuropathy + psychiatric triad; Polish founder c.204_214del11; slowly progressive' },
  'FA2H':    { full: 'FA2H / Fatty Acid 2-Hydroxylase / 480aa', locus: '16q23.1', size: '480 aa / 55 kDa', inh: 'AR', disease: 'FAHN/SPG35; LEUKODYSTROPHY earliest+most prominent MRI; thin corpus callosum; spastic paraplegia dominant; GP iron MILD' },
  'ATP13A2': { full: 'ATP13A2 / P5-type ATPase / 1180aa', locus: '1p36.13', size: '1180 aa / 128 kDa', inh: 'AR', disease: 'KRS/PARK9; Juvenile Parkinson + supranuclear gaze palsy + pyramidal = TRIAD; LEVODOPA-RESPONSIVE (unique in NBIA); Jordanian founder' },
  'COASY':   { full: 'COASY / CoA Synthase / 579aa', locus: '17q21.2', size: '579 aa / 64 kDa', inh: 'AR', disease: 'CoPAN (NBIA6); CoA biosynthesis same pathway as PANK2 (downstream); GP iron MILD no eye-of-tiger; spasticity dominant; extremely rare' },
  'DCAF17':  { full: 'DCAF17 / DDB1-CUL4 Factor 17 / 520aa', locus: '2q31.1', size: '520 aa / 58 kDa', inh: 'AR', disease: 'WSS; UNIQUE: hypogonadism + alopecia + DM + SNHL + NBIA = pentad PATHOGNOMONIC; Gulf Arab p.Cys44Tyr founder; ubiquitin ligase' },
};

function GeneChip({ gene }) {
  const col = GENE_COLORS[gene] || '#555';
  return (
    <span style={{ background: col, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 12, fontWeight: 700, margin: '0 2px' }}>
      {gene}
    </span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 120 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8' }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HereditaryNBIAAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [selGene, setSelGene] = useState(null);

  useEffect(() => {
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' ? 'breakdown'
      : tab === 'Clinical Atlas' ? 'breakdown'
      : 'overview';
    setLoading(true); setErr(null);
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => {
        if (ep === 'overview') setOverview(data);
        else if (ep === 'breakdown') setBreakdown(data);
        else setDefinitions(data);
        setLoading(false);
      })
      .catch(e => { setErr(String(e)); setLoading(false); });
  }, [tab]);

  const bg = '#0f172a';
  const card = '#1e293b';
  const accent = '#b71c1c';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>

        {/* Header */}
        <div style={{ background: card, borderRadius: 12, padding: '20px 24px', marginBottom: 20, borderLeft: `4px solid ${accent}` }}>
          <h1 style={{ margin: 0, fontSize: 20, color: '#fca5a5' }}>🧠 Hereditary NBIA Atlas</h1>
          <p style={{ margin: '6px 0 0', color: '#94a3b8', fontSize: 13 }}>
            Neurodegeneration with Brain Iron Accumulation — Complete 8-Gene Reference &nbsp;|&nbsp;
            PANK2 · PLA2G6 · WDR45 · C19orf12 · FA2H · ATP13A2 · COASY · DCAF17 &nbsp;|&nbsp;
            320 patients · Seeds 2606–2613
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 10 }}>
            {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{ background: tab === t ? accent : card, color: tab === t ? '#fff' : '#94a3b8', border: 'none', borderRadius: 6, padding: '8px 18px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400 }}>
              {t}
            </button>
          ))}
        </div>

        {loading && <div style={{ color: '#94a3b8' }}>Loading…</div>}
        {err && <div style={{ color: '#ef4444' }}>Error: {err}</div>}

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 20 }}>
              <MetricCard label="Genes Covered" value={overview.n_genes} sub="NBIA subtypes" />
              <MetricCard label="Total Patients" value={overview.total_patients} sub={`Seeds ${overview.seeds}`} />
              <MetricCard label="GP Iron Overall" value={`${overview.aggregate_stats?.overall_gp_iron_pct}%`} sub="cross-gene" />
              <MetricCard label="Dystonia Overall" value={`${overview.aggregate_stats?.overall_dystonia_pct}%`} sub="cross-gene" />
              <MetricCard label="Eye-of-Tiger" value={`${overview.aggregate_stats?.overall_eye_of_tiger_pct}%`} warn sub="PANK2 only" />
            </div>

            {/* Key Distinctions */}
            <div style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 20, borderLeft: `3px solid ${accent}` }}>
              <div style={{ fontWeight: 700, color: '#fca5a5', marginBottom: 8 }}>🔑 Key Clinical Distinctions (PATHOGNOMONIC)</div>
              {(overview.key_clinical_distinctions || []).map((d, i) => (
                <div key={i} style={{ fontSize: 12, color: '#cbd5e1', padding: '4px 0', borderBottom: '1px solid #1e293b' }}>
                  • {d}
                </div>
              ))}
            </div>

            {/* Gene Summary Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
              {(overview.gene_summaries || []).map(gs => {
                const col = GENE_COLORS[gs.gene] || '#555';
                const info = GENE_INFO[gs.gene] || {};
                return (
                  <div key={gs.gene} style={{ background: card, borderRadius: 10, padding: 14, borderTop: `3px solid ${col}` }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                      <GeneChip gene={gs.gene} />
                      <span style={{ fontSize: 11, color: '#64748b' }}>{info.locus} · {info.inh}</span>
                    </div>
                    <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8, lineHeight: 1.5 }}>{info.disease}</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: 16, fontWeight: 700, color: '#38bdf8' }}>{gs.avg_onset_age}yr</div>
                        <div style={{ fontSize: 10, color: '#64748b' }}>Onset</div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: 16, fontWeight: 700, color: gs.eye_of_tiger_pct > 50 ? '#ef4444' : '#94a3b8' }}>
                          {gs.eye_of_tiger_pct}%
                        </div>
                        <div style={{ fontSize: 10, color: '#64748b' }}>Eye-Tiger</div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: 16, fontWeight: 700, color: '#38bdf8' }}>{gs.gp_iron_pct}%</div>
                        <div style={{ fontSize: 10, color: '#64748b' }}>GP Iron</div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && breakdown && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1e293b' }}>
                  {['Gene','Locus','Size','Inh','Disease','Onset','GP Iron','Eye-Tiger','Dystonia','Levodopa','n'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', borderBottom: '1px solid #334155' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.gene_breakdowns || []).map(g => (
                  <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b' }}
                    onClick={() => setSelGene(selGene === g.gene ? null : g.gene)}
                    style={{ borderBottom: '1px solid #1e293b', cursor: 'pointer', background: selGene === g.gene ? '#1e293b' : 'transparent' }}>
                    <td style={{ padding: '8px 10px' }}><GeneChip gene={g.gene} /></td>
                    <td style={{ padding: '8px 10px', color: '#64748b' }}>{g.locus}</td>
                    <td style={{ padding: '8px 10px', color: '#64748b' }}>{g.protein_size}</td>
                    <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.inheritance?.split(';')[0]}</td>
                    <td style={{ padding: '8px 10px', color: '#cbd5e1', maxWidth: 200 }}>{g.disease_category?.split(';')[0]}</td>
                    <td style={{ padding: '8px 10px', color: '#38bdf8' }}>{g.avg_onset_age}yr</td>
                    <td style={{ padding: '8px 10px', color: g.gp_iron_pct > 80 ? '#ef4444' : '#94a3b8' }}>{g.gp_iron_pct}%</td>
                    <td style={{ padding: '8px 10px', color: g.eye_of_tiger_pct > 50 ? '#ef4444' : '#64748b' }}>{g.eye_of_tiger_pct}%</td>
                    <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.dystonia_pct}%</td>
                    <td style={{ padding: '8px 10px', color: g.levodopa_response_pct > 50 ? '#4ade80' : '#64748b' }}>{g.levodopa_response_pct}%</td>
                    <td style={{ padding: '8px 10px', color: '#64748b' }}>{g.n_patients}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {selGene && breakdown.gene_breakdowns && (
              <div style={{ background: card, borderRadius: 10, padding: 16, marginTop: 16, borderLeft: `3px solid ${GENE_COLORS[selGene] || '#555'}` }}>
                {(() => {
                  const g = breakdown.gene_breakdowns.find(x => x.gene === selGene);
                  if (!g) return null;
                  return (
                    <>
                      <div style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 8 }}>{selGene} — Pathognomonic</div>
                      <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10 }}>{g.pathognomonic}</div>
                      <div style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 6 }}>Treatment</div>
                      <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10 }}>{g.treatment}</div>
                      <div style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 6 }}>Key DDx</div>
                      {(g.key_ddx || []).map((d, i) => <div key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 3 }}>• {d}</div>)}
                    </>
                  );
                })()}
              </div>
            )}
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 16 }}>
            {(breakdown.gene_breakdowns || []).map(g => {
              const col = GENE_COLORS[g.gene] || '#555';
              return (
                <div key={g.gene} style={{ background: card, borderRadius: 10, padding: 16, borderTop: `3px solid ${col}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
                    <GeneChip gene={g.gene} />
                    <span style={{ fontSize: 11, color: '#64748b' }}>{g.locus} · {g.protein_size}</span>
                  </div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8, lineHeight: 1.5 }}>
                    {g.disease_category?.split(';')[0]}
                  </div>
                  <div style={{ fontWeight: 700, color: '#fca5a5', fontSize: 11, marginBottom: 4 }}>Pathognomonic</div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8, lineHeight: 1.5 }}>
                    {g.pathognomonic?.split(';')[0]}
                  </div>
                  <div style={{ fontWeight: 700, color: '#fca5a5', fontSize: 11, marginBottom: 4 }}>Key Features</div>
                  {(g.key_features || []).slice(0, 3).map((f, i) => (
                    <div key={i} style={{ fontSize: 11, color: '#cbd5e1', marginBottom: 3 }}>• {f}</div>
                  ))}
                </div>
              );
            })}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            {/* NBIA Glossary */}
            <div style={{ marginBottom: 24 }}>
              <div style={{ fontWeight: 700, color: '#fca5a5', fontSize: 14, marginBottom: 12 }}>NBIA Glossary</div>
              {Object.entries(definitions.nbia_glossary || {}).map(([term, def]) => (
                <div key={term} style={{ background: card, borderRadius: 8, padding: 14, marginBottom: 10 }}>
                  <div style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 6 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6 }}>{def}</div>
                </div>
              ))}
            </div>
            {/* Per-Gene Definitions */}
            <div style={{ fontWeight: 700, color: '#fca5a5', fontSize: 14, marginBottom: 12 }}>Per-Gene Reference</div>
            {Object.entries(definitions.gene_entries || {}).map(([gene, entry]) => {
              const col = GENE_COLORS[gene] || '#555';
              return (
                <div key={gene} style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 12, borderLeft: `4px solid ${col}` }}>
                  <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10 }}>
                    <GeneChip gene={gene} />
                    <span style={{ fontSize: 12, color: '#64748b' }}>{entry.locus} · {entry.protein_size} · {entry.inheritance}</span>
                  </div>
                  <div style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 4 }}>{entry.disease_name}</div>
                  <div style={{ fontWeight: 700, color: '#94a3b8', fontSize: 12, marginTop: 10, marginBottom: 4 }}>Pathway</div>
                  <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6, marginBottom: 8 }}>{entry.disease_pathway}</div>
                  <div style={{ fontWeight: 700, color: '#94a3b8', fontSize: 12, marginBottom: 4 }}>Pathognomonic</div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6, marginBottom: 8 }}>{entry.pathognomonic}</div>
                  <div style={{ fontWeight: 700, color: '#94a3b8', fontSize: 12, marginBottom: 4 }}>Key Features</div>
                  {(entry.key_features || []).map((f, i) => <div key={i} style={{ fontSize: 11, color: '#94a3b8', marginBottom: 2 }}>• {f}</div>)}
                  <div style={{ fontWeight: 700, color: '#94a3b8', fontSize: 12, marginTop: 8, marginBottom: 4 }}>DDx</div>
                  {(entry.key_ddx || []).map((d, i) => <div key={i} style={{ fontSize: 11, color: '#64748b', marginBottom: 2 }}>• {d}</div>)}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
