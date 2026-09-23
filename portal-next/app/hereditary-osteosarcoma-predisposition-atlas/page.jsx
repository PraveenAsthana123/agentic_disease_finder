'use client';
import { useState, useEffect } from 'react';

const SLUG = 'hereditary-osteosarcoma-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  RB1:    '#10b981',
  TP53:   '#ef4444',
  RECQL4: '#f59e0b',
  BRCA2:  '#3b82f6',
  NF1:    '#8b5cf6',
  CDKN2A: '#6366f1',
  DICER1: '#ec4899',
  WRN:    '#f97316',
};

const GENE_INFO = {
  RB1:    { full: 'Retinoblastoma Protein',                              locus: '13q14.2',  size: '928aa',   inh: 'AD LOF' },
  TP53:   { full: 'Tumour Protein p53',                                  locus: '17p13.1',  size: '393aa',   inh: 'AD LOF' },
  RECQL4: { full: 'RecQ-like Helicase 4',                                locus: '8q24.12',  size: '1208aa',  inh: 'AR LOF' },
  BRCA2:  { full: 'Breast Cancer Type 2 Susceptibility',                 locus: '13q12.3',  size: '3418aa',  inh: 'AD LOF / AR FA-D1' },
  NF1:    { full: 'Neurofibromin',                                       locus: '17q11.2',  size: '2839aa',  inh: 'AD LOF' },
  CDKN2A: { full: 'Cyclin-Dependent Kinase Inhibitor 2A (p16/ARF)',      locus: '9p21.3',   size: '156aa',   inh: 'AD LOF' },
  DICER1: { full: 'DICER1 RNase III',                                    locus: '14q32.13', size: '1922aa',  inh: 'AD LOF' },
  WRN:    { full: 'Werner Syndrome Helicase',                            locus: '8p12',     size: '1432aa',  inh: 'AR LOF' },
};

function Badge({ label, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 6, padding: '2px 10px', fontSize: 12, fontWeight: 700,
      display: 'inline-block', whiteSpace: 'nowrap'
    }}>{label}</span>
  );
}

function GeneBar({ gene, value, max }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</span>
        <span>{value}</span>
      </div>
      <div style={{ background: '#e5e7eb', borderRadius: 4, height: 8 }}>
        <div style={{ background: GENE_COLORS[gene], width: `${pct}%`, height: 8, borderRadius: 4, transition: 'width 0.4s' }} />
      </div>
    </div>
  );
}

export default function HereditaryOsteosarcomaPredispositionAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => {
      setError(e.message);
      setLoading(false);
    });
  }, []);

  if (loading) return (
    <div style={{ padding: 48, textAlign: 'center', color: '#6b7280' }}>
      Loading Hereditary Osteosarcoma Predisposition Atlas…
    </div>
  );
  if (error) return (
    <div style={{ padding: 48, textAlign: 'center', color: '#ef4444' }}>
      Error: {error}
    </div>
  );

  const genes = Object.keys(GENE_COLORS);

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 16px', fontFamily: 'system-ui,sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <span style={{ fontSize: 32 }}>🧬</span>
          <div>
            <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, color: '#111827' }}>
              Hereditary Osteosarcoma Predisposition Atlas
            </h1>
            <div style={{ color: '#6b7280', fontSize: 14, marginTop: 4 }}>
              Complete 8-Gene Reference · RB1 · TP53 · RECQL4 · BRCA2 · NF1 · CDKN2A · DICER1 · WRN · Seeds 3310–3317
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <Badge label="Post-Irradiation Risk" color="#10b981" />
          <Badge label="Fanconi Anaemia" color="#3b82f6" />
          <Badge label="AVOID RADIATION: TP53 NF1 DICER1" color="#ef4444" />
          <Badge label="AVOID ALKYLATING: BRCA2 RECQL4" color="#f59e0b" />
          <Badge label="320 Patients · 8×40" color="#6b7280" />
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 2, marginBottom: 24, borderBottom: '2px solid #e5e7eb' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 20px', border: 'none', cursor: 'pointer', fontWeight: 600,
            background: tab === t ? '#2563eb' : 'transparent',
            color: tab === t ? '#fff' : '#374151',
            borderRadius: '6px 6px 0 0',
            borderBottom: tab === t ? '2px solid #2563eb' : '2px solid transparent',
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          {/* KPI cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 16, marginBottom: 32 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#2563eb' },
              { label: 'Complete Remission', value: `${overview.cr_pct}%`, color: '#10b981' },
              { label: 'Chemotherapy', value: `${overview.chemo_pct}%`, color: '#8b5cf6' },
              { label: 'HSCT Used', value: `${overview.hsct_pct}%`, color: '#f59e0b' },
              { label: 'Radiation Avoided', value: `${100 - overview.radiation_pct}%`, color: '#ef4444' },
              { label: 'Mean Age Dx', value: `${overview.mean_age_dx}yr`, color: '#6366f1' },
              { label: 'Relapse Rate', value: `${overview.relapse_pct}%`, color: '#ec4899' },
            ].map(k => (
              <div key={k.label} style={{
                background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12,
                padding: '16px 12px', textAlign: 'center', boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
              }}>
                <div style={{ fontSize: 26, fontWeight: 800, color: k.color }}>{k.value}</div>
                <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>{k.label}</div>
              </div>
            ))}
          </div>

          {/* Gene CR comparison */}
          <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 20, marginBottom: 24 }}>
            <h3 style={{ margin: '0 0 16px', fontSize: 15, fontWeight: 700 }}>Complete Remission Rate by Gene</h3>
            {overview.gene_summaries?.map(g => (
              <GeneBar key={g.gene} gene={g.gene} value={`${g.cr_pct}%`} max={100} />
            ))}
          </div>

          {/* Mean age at diagnosis */}
          <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 20, marginBottom: 24 }}>
            <h3 style={{ margin: '0 0 16px', fontSize: 15, fontWeight: 700 }}>Mean Age at Diagnosis by Gene</h3>
            {overview.gene_summaries?.map(g => (
              <GeneBar key={g.gene} gene={g.gene} value={`${g.mean_age_dx}yr`} max={40} />
            ))}
          </div>

          {/* Top tumour types */}
          {overview.top_tumour_types && (
            <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 20, marginBottom: 24 }}>
              <h3 style={{ margin: '0 0 16px', fontSize: 15, fontWeight: 700 }}>Top Tumour Types (All Genes)</h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {Object.entries(overview.top_tumour_types).map(([t, n]) => (
                  <span key={t} style={{
                    background: '#f3f4f6', borderRadius: 6, padding: '4px 12px',
                    fontSize: 12, fontWeight: 600, color: '#374151'
                  }}>{t} <strong>({n})</strong></span>
                ))}
              </div>
            </div>
          )}

          {/* Clinical note */}
          <div style={{ background: '#fffbeb', border: '1px solid #fbbf24', borderRadius: 12, padding: 16 }}>
            <div style={{ fontWeight: 700, color: '#92400e', marginBottom: 8 }}>⚠ Key Clinical Rules</div>
            <ul style={{ margin: 0, paddingLeft: 20, color: '#78350f', fontSize: 13, lineHeight: 1.7 }}>
              <li><strong>RB1:</strong> OS 40% lifetime HIGHEST hereditary OS risk. Bilateral RB PATHOGNOMONIC. AVOID post-RT OS site re-irradiation. CDK4/6i INACTIVE in RB1-null tumours. Annual bone scintigraphy/MRI surveillance.</li>
              <li><strong>TP53 LFS:</strong> OS anaplastic/pleomorphic PATHOGNOMONIC LFS. AVOID RADIATION ABSOLUTELY. WBMRI Toronto annually. Proton beam preferred if unavoidable.</li>
              <li><strong>RECQL4 RTS2:</strong> Poikiloderma 3–6 months PATHOGNOMONIC. OS 30–50%. AVOID alkylating in high-dose. Skin photoprotection MANDATORY lifelong.</li>
              <li><strong>BRCA2 FA-D1:</strong> DEB/MMC chromosomal fragility PATHOGNOMONIC. AVOID alkylating ABSOLUTELY. SIBLING DONOR EXCLUSION MANDATORY.</li>
              <li><strong>NF1:</strong> Café-au-lait macules 6+ PATHOGNOMONIC. MPNST 8–13%. OS 2–3×. Selumetinib FDA 2020 pediatric plexiform. AVOID whole-body radiation.</li>
              <li><strong>CDKN2A:</strong> p16 IHC loss PATHOGNOMONIC. CDK4/6i palbociclib investigational CDK4-amplified OS. Pancreatic adenocarcinoma 20× concurrent risk.</li>
              <li><strong>DICER1:</strong> PPB type I PATHOGNOMONIC. CT chest siblings &lt;8yr. AVOID radiation children. OS/mesenchymal 2–4× elevated.</li>
              <li><strong>WRN:</strong> Adult-onset bilateral cataracts PATHOGNOMONIC (onset 20s). Type 2 DM + scleroderma PATHOGNOMONIC Werner triad. Mesenchymal cancer predominance.</li>
            </ul>
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f9fafb' }}>
                {['Gene', 'Full Name', 'Locus', 'Size', 'Inheritance', 'Syndrome', 'OS Type', 'Key Risk'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 700, borderBottom: '2px solid #e5e7eb', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { gene: 'RB1',    syndrome: 'Hereditary Retinoblastoma',    osType: 'OS type',              risk: 'OS 40% lifetime HIGHEST' },
                { gene: 'TP53',   syndrome: 'Li-Fraumeni Syndrome',         osType: 'OS/STS/anaplastic',    risk: 'OS #1 cancer LFS' },
                { gene: 'RECQL4', syndrome: 'Rothmund-Thomson type 2',       osType: 'OS',                   risk: 'OS 30–50%' },
                { gene: 'BRCA2',  syndrome: 'HBOC / Fanconi FA-D1',         osType: 'OS/bone tumours',      risk: 'AVOID alkylating' },
                { gene: 'NF1',    syndrome: 'Neurofibromatosis 1',           osType: 'OS/MPNST',             risk: 'OS 2–3× elevated' },
                { gene: 'CDKN2A', syndrome: 'FAMMM',                         osType: 'OS/melanoma',          risk: 'CDK4/6i inactive at OS' },
                { gene: 'DICER1', syndrome: 'DICER1 syndrome',               osType: 'mesenchymal/OS',       risk: 'PPB PATHOGNOMONIC' },
                { gene: 'WRN',    syndrome: 'Werner syndrome',               osType: 'OS/mesenchymal',       risk: 'adult-onset PATHOGNOMONIC' },
              ].map((row, i) => (
                <tr key={row.gene} style={{ background: i % 2 === 0 ? '#fff' : '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                  <td style={{ padding: '10px 12px', fontWeight: 800, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                  <td style={{ padding: '10px 12px' }}>{GENE_INFO[row.gene]?.full}</td>
                  <td style={{ padding: '10px 12px', fontFamily: 'monospace' }}>{GENE_INFO[row.gene]?.locus}</td>
                  <td style={{ padding: '10px 12px', fontFamily: 'monospace' }}>{GENE_INFO[row.gene]?.size}</td>
                  <td style={{ padding: '10px 12px' }}>{GENE_INFO[row.gene]?.inh}</td>
                  <td style={{ padding: '10px 12px' }}>{row.syndrome}</td>
                  <td style={{ padding: '10px 12px' }}>
                    <Badge label={row.osType} color={GENE_COLORS[row.gene]} />
                  </td>
                  <td style={{ padding: '10px 12px', fontSize: 12 }}>{row.risk}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {breakdown.breakdown?.map(g => (
            <div key={g.gene} style={{
              background: '#fff', border: `2px solid ${GENE_COLORS[g.gene]}33`,
              borderLeft: `4px solid ${GENE_COLORS[g.gene]}`,
              borderRadius: 12, padding: 20, marginBottom: 20
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
                <div>
                  <span style={{ fontSize: 20, fontWeight: 800, color: GENE_COLORS[g.gene] }}>{g.gene}</span>
                  <span style={{ marginLeft: 8, fontSize: 13, color: '#6b7280' }}>{GENE_INFO[g.gene]?.full} · {g.locus}</span>
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <Badge label={`CR ${g.cr_pct}%`} color={g.cr_pct >= 65 ? '#10b981' : g.cr_pct >= 55 ? '#f59e0b' : '#ef4444'} />
                  <Badge label={`Radiation ${g.radiation_pct}%`} color={g.radiation_pct <= 5 ? '#10b981' : '#f59e0b'} />
                  <Badge label={`Age ${g.mean_age_dx}yr`} color="#6366f1" />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                <div>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 4 }}>Top Tumour Types</div>
                  {Object.entries(g.top_tumor_types || {}).map(([t, n]) => (
                    <div key={t} style={{ fontSize: 12, color: '#6b7280', marginBottom: 2 }}>• {t} ({n})</div>
                  ))}
                </div>
                <div>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 4 }}>Common Variants</div>
                  {Object.entries(g.top_variants || {}).map(([v, n]) => (
                    <div key={v} style={{ fontSize: 12, color: '#6b7280', marginBottom: 2, fontFamily: 'monospace' }}>• {v} ({n})</div>
                  ))}
                </div>
              </div>

              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 4 }}>Pathognomonic Features</div>
                <div style={{ fontSize: 12, color: '#374151', background: '#f0fdf4', padding: '8px 12px', borderRadius: 6 }}>
                  {g.pathognomonic}
                </div>
              </div>

              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 4 }}>Surveillance Protocol</div>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {g.surveillance_protocols?.map((s, i) => (
                    <li key={i} style={{ fontSize: 12, color: '#374151', marginBottom: 3 }}>{s}</li>
                  ))}
                </ul>
              </div>

              <div>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 6 }}>Key Distinctions</div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {g.key_distinctions?.map(d => (
                    <Badge key={d} label={d.replace(/-/g, ' ')} color={GENE_COLORS[g.gene]} />
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          {/* Key Rules */}
          <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 20, marginBottom: 24 }}>
            <h3 style={{ margin: '0 0 16px', fontSize: 15, fontWeight: 700 }}>Key Clinical Rules</h3>
            {Object.entries(definitions.key_rules || {}).map(([k, v]) => (
              <div key={k} style={{ marginBottom: 16 }}>
                <div style={{ fontWeight: 700, fontSize: 13, color: '#1d4ed8', marginBottom: 4 }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.6 }}>{v}</div>
              </div>
            ))}
          </div>

          {/* Cascade Testing Rule */}
          {definitions.cascade_testing_rule && (
            <div style={{ background: '#eff6ff', border: '1px solid #93c5fd', borderRadius: 12, padding: 16, marginBottom: 24 }}>
              <div style={{ fontWeight: 700, color: '#1d4ed8', marginBottom: 8 }}>Cascade Testing Protocol</div>
              <div style={{ fontSize: 13, color: '#1e3a8a', lineHeight: 1.7 }}>{definitions.cascade_testing_rule}</div>
            </div>
          )}

          {/* Per-gene definitions */}
          {Object.entries(definitions.definitions || {}).map(([gene, def]) => (
            <div key={gene} style={{
              background: '#fff', border: `1px solid ${GENE_COLORS[gene]}44`,
              borderLeft: `4px solid ${GENE_COLORS[gene]}`,
              borderRadius: 12, padding: 20, marginBottom: 16
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
                <div style={{ fontWeight: 800, fontSize: 18, color: GENE_COLORS[gene] }}>{gene}</div>
                <div style={{ display: 'flex', gap: 6 }}>
                  <Badge label={def.locus} color={GENE_COLORS[gene]} />
                  <Badge label={GENE_INFO[gene]?.size || ''} color="#6b7280" />
                </div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Inheritance</div>
                <div style={{ fontSize: 13, color: '#374151' }}>{def.inheritance}</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Cancer Risk</div>
                <div style={{ fontSize: 13, color: '#374151' }}>{def.cancer_risk}</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Pathognomonic</div>
                <div style={{ fontSize: 13, color: '#374151', background: '#f0fdf4', padding: '8px 10px', borderRadius: 6 }}>{def.pathognomonic}</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Surveillance</div>
                <div style={{ fontSize: 13, color: '#374151' }}>{def.surveillance_key}</div>
              </div>

              <div>
                <div style={{ fontSize: 11, fontWeight: 700, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>Key Distinctions</div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {def.key_distinctions?.map(d => (
                    <Badge key={d} label={d.replace(/-/g, ' ')} color={GENE_COLORS[gene]} />
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
