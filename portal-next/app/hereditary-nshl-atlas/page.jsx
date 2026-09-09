'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  GJB2:    '#1565c0',  // deep blue — DFNB1A most common AR NSHL
  SLC26A4: '#006064',  // dark teal — DFNB4/Pendred EVA
  OTOF:    '#e65100',  // amber — DFNB9 ANSD
  MYO15A:  '#2e7d32',  // deep green — DFNB3 short stereocilia
  TECTA:   '#6a1b9a',  // deep purple — DFNB21/DFNA8-12 dual inheritance
  KCNQ4:   '#c62828',  // deep red — DFNA2A progressive HF
  LHFPL5:  '#37474f',  // slate — DFNB67 mechanotransduction
  GJB6:    '#0277bd',  // mid blue — DFNB1B digenic
};

const GENE_DISEASE = {
  GJB2:    'DFNB1A (AR) — most common AR NSHL globally ~50%; p.35delG European founder; connexin K+ recycling; CI EXCELLENT',
  SLC26A4: 'DFNB4/Pendred (AR) — EVA on CT/MRI PATHOGNOMONIC; contact sports CONTRAINDICATED; thyroid goiter; p.H723R East Asian',
  OTOF:    'DFNB9-ANSD (AR) — present OAE + absent ABR; hearing aids ineffective; CI FIRST-LINE; p.Ile515Thr Iberian founder',
  MYO15A:  'DFNB3 (AR) — profound congenital SNHL; short stereocilia; EPS8 cargo; 3530 aa largest cochlear myosin',
  TECTA:   'DFNB21 (AR profound) / DFNA8-12 (AD mid-freq U-shape) — SAME GENE, OPPOSITE INHERITANCE, DIFFERENT PHENOTYPE',
  KCNQ4:   'DFNA2A (AD) — progressive HF SNHL; dominant-negative; onset 2nd-3rd decade; noise protection MANDATORY',
  LHFPL5:  'DFNB67 (AR) — profound congenital SNHL; TMC1/TMC2 mechanotransduction auxiliary; AAV gene therapy candidate',
  GJB6:    'DFNB1B (AR/digenic) — del(GJB6-D13S1830) 342 kb; digenic with GJB2; MLPA MANDATORY — exome misses deletion',
};

const AR_GENES  = ['GJB2', 'SLC26A4', 'OTOF', 'MYO15A', 'LHFPL5', 'GJB6'];
const AD_GENES  = ['KCNQ4'];
const DUAL_GENES = ['TECTA'];

export default function HereditaryNSHLAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [ov, bk, df] = await Promise.all([
          fetch(`${API}/api/hereditary-nshl-atlas/overview`).then(r => r.json()),
          fetch(`${API}/api/hereditary-nshl-atlas/breakdown`).then(r => r.json()),
          fetch(`${API}/api/hereditary-nshl-atlas/definitions`).then(r => r.json()),
        ]);
        setOverview(ov);
        setBreakdown(bk);
        setDefinitions(df);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#121212', minHeight: '100vh' }}>Loading Hereditary NSHL Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#121212', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#121212', minHeight: '100vh', color: '#fff', padding: '24px 32px', fontFamily: 'monospace' }}>
      <h1 style={{ color: '#29b6f6', fontSize: 22, marginBottom: 4 }}>
        🧬 Hereditary Non-Syndromic Hearing Loss Atlas
      </h1>
      <p style={{ color: '#90caf9', fontSize: 13, marginBottom: 20 }}>
        Complete 8-Gene Reference · GJB2 · SLC26A4 · OTOF · MYO15A · TECTA · KCNQ4 · LHFPL5 · GJB6 ·
        {overview && ` ${overview.total_patients} patients · seeds ${overview.seed_range}`}
      </p>

      {/* Gene legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
        {Object.entries(GENE_COLORS).map(([gene, col]) => (
          <span key={gene} style={{
            background: col, color: '#fff', padding: '3px 10px',
            borderRadius: 4, fontSize: 11, fontWeight: 700,
          }}>{gene}</span>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#1565c0' : '#1e1e1e',
            color: tab === t ? '#fff' : '#90caf9',
            border: '1px solid #1565c0', borderRadius: 4,
            padding: '6px 16px', cursor: 'pointer', fontSize: 12,
          }}>{t}</button>
        ))}
      </div>

      {/* ── Overview ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12, marginBottom: 24 }}>
            {[
              ['Total Patients', overview.total_patients],
              ['Genes', overview.genes_covered],
              ['Seed Range', overview.seed_range],
              ['Most Common', 'GJB2 DFNB1A (~50% AR NSHL)'],
            ].map(([k, v]) => (
              <div key={k} style={{ background: '#1e1e1e', border: '1px solid #333', borderRadius: 6, padding: 14 }}>
                <div style={{ color: '#64b5f6', fontSize: 11 }}>{k}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#fff' }}>{v}</div>
              </div>
            ))}
          </div>

          {/* Patient distribution */}
          <div style={{ background: '#1e1e1e', border: '1px solid #333', borderRadius: 6, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#64b5f6', marginBottom: 12, fontSize: 14 }}>Patients per Gene</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
              {overview.patients_per_gene && Object.entries(overview.patients_per_gene).map(([gene, n]) => (
                <div key={gene} style={{
                  background: GENE_COLORS[gene] || '#333',
                  padding: '8px 14px', borderRadius: 6, minWidth: 110,
                }}>
                  <div style={{ fontSize: 13, fontWeight: 700 }}>{gene}</div>
                  <div style={{ fontSize: 11, opacity: 0.85 }}>{n} patients</div>
                  <div style={{ fontSize: 10, opacity: 0.7, marginTop: 2 }}>{GENE_DISEASE[gene]?.split(';')[0]}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Inheritance classification */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 20 }}>
            {[
              ['AR Genes (Recessive)', AR_GENES, '#1565c0'],
              ['AD Gene (Dominant)', AD_GENES, '#c62828'],
              ['Dual AR/AD', DUAL_GENES, '#6a1b9a'],
            ].map(([label, genes, col]) => (
              <div key={label} style={{ background: '#1e1e1e', border: `1px solid ${col}`, borderRadius: 6, padding: 14 }}>
                <div style={{ color: col, fontSize: 12, fontWeight: 700, marginBottom: 8 }}>{label}</div>
                {genes.map(g => (
                  <div key={g} style={{ color: '#e0e0e0', fontSize: 11, marginBottom: 3 }}>• {g} — {GENE_DISEASE[g]?.split('—')[1]?.trim().slice(0, 60)}…</div>
                ))}
              </div>
            ))}
          </div>

          {/* Clinical pearls */}
          <div style={{ background: '#1e1e1e', border: '1px solid #1565c0', borderRadius: 6, padding: 16 }}>
            <h3 style={{ color: '#64b5f6', marginBottom: 10, fontSize: 14 }}>Clinical Pearls</h3>
            {overview.clinical_pearls?.map((p, i) => (
              <div key={i} style={{ color: '#e0e0e0', fontSize: 11, marginBottom: 6, borderLeft: '3px solid #1565c0', paddingLeft: 10 }}>
                {p}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Gene Table ── */}
      {tab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ background: '#1565c0' }}>
                {['Gene', 'Locus', 'Size', 'Inh.', 'Disease / DFNB/A', 'Key Feature', 'CI Outcome'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#fff', fontWeight: 700 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.values(breakdown.breakdown_by_gene || {}).map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#1a1a1a' : '#222' }}>
                  <td style={{ padding: '7px 10px', color: GENE_COLORS[g.gene] || '#fff', fontWeight: 700 }}>{g.gene}</td>
                  <td style={{ padding: '7px 10px', color: '#ccc' }}>{g.locus}</td>
                  <td style={{ padding: '7px 10px', color: '#ccc' }}>{g.protein_size}</td>
                  <td style={{ padding: '7px 10px', color: '#ccc' }}>{g.inheritance?.split(';')[0]?.split('(')[1]?.replace(')', '') || 'AR/AD'}</td>
                  <td style={{ padding: '7px 10px', color: '#e0e0e0' }}>{g.disease_category?.slice(0, 80)}…</td>
                  <td style={{ padding: '7px 10px', color: '#90caf9' }}>{g.key_features?.[0]?.slice(0, 70)}…</td>
                  <td style={{ padding: '7px 10px', color: '#a5d6a7' }}>{
                    g.gene === 'OTOF' ? 'EXCELLENT (ANSD)' :
                    g.gene === 'GJB2' ? 'EXCELLENT' :
                    g.gene === 'GJB6' ? 'EXCELLENT' :
                    g.gene === 'KCNQ4' ? 'Good (late-stage)' :
                    'Good'
                  }</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Clinical Atlas ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div style={{ display: 'grid', gap: 20 }}>
          {Object.values(breakdown.breakdown_by_gene || {}).map(g => (
            <div key={g.gene} style={{
              background: '#1a1a1a',
              border: `1px solid ${GENE_COLORS[g.gene] || '#333'}`,
              borderRadius: 8, padding: 20,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                <div>
                  <span style={{
                    background: GENE_COLORS[g.gene], color: '#fff',
                    padding: '3px 12px', borderRadius: 4, fontSize: 14, fontWeight: 700, marginRight: 10,
                  }}>{g.gene}</span>
                  <span style={{ color: '#90caf9', fontSize: 12 }}>{g.locus} · {g.protein_size}</span>
                </div>
                <span style={{ color: '#81d4fa', fontSize: 11 }}>{g.n_patients} patients</span>
              </div>

              <div style={{ color: '#ffe082', fontSize: 12, marginBottom: 8, fontStyle: 'italic' }}>
                {g.disease_category}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                <div>
                  <div style={{ color: '#64b5f6', fontSize: 11, marginBottom: 4 }}>Pathognomonic Features</div>
                  <div style={{ color: '#e0e0e0', fontSize: 11, lineHeight: 1.5 }}>{g.pathognomonic?.slice(0, 400)}…</div>
                </div>
                <div>
                  <div style={{ color: '#64b5f6', fontSize: 11, marginBottom: 4 }}>Treatment</div>
                  <div style={{ color: '#a5d6a7', fontSize: 11, lineHeight: 1.5 }}>{g.treatment?.slice(0, 350)}…</div>
                </div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#64b5f6', fontSize: 11, marginBottom: 4 }}>Key Features</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                  {g.key_features?.slice(0, 4).map((f, i) => (
                    <span key={i} style={{
                      background: '#1e2a3a', border: '1px solid #1565c0',
                      color: '#90caf9', padding: '2px 8px', borderRadius: 3, fontSize: 10,
                    }}>{f.slice(0, 60)}</span>
                  ))}
                </div>
              </div>

              <div>
                <div style={{ color: '#64b5f6', fontSize: 11, marginBottom: 4 }}>Key DDx</div>
                <div style={{ color: '#ef9a9a', fontSize: 11, lineHeight: 1.5 }}>{g.key_ddx?.slice(0, 300)}…</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'Definitions' && definitions && (
        <div style={{ display: 'grid', gap: 16 }}>
          {/* Key Clinical Definitions */}
          <div style={{ background: '#1e1e1e', border: '1px solid #1565c0', borderRadius: 6, padding: 18 }}>
            <h3 style={{ color: '#64b5f6', marginBottom: 12, fontSize: 14 }}>Key Clinical Definitions</h3>
            {Object.entries(definitions.key_clinical_definitions || {}).map(([k, v]) => (
              <div key={k} style={{ marginBottom: 14 }}>
                <div style={{ color: '#ffe082', fontSize: 12, fontWeight: 700, marginBottom: 3 }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ color: '#e0e0e0', fontSize: 11, lineHeight: 1.6 }}>{v}</div>
              </div>
            ))}
          </div>

          {/* Diagnostic Protocol */}
          <div style={{ background: '#1e1e1e', border: '1px solid #4caf50', borderRadius: 6, padding: 18 }}>
            <h3 style={{ color: '#81c784', marginBottom: 12, fontSize: 14 }}>Diagnostic Protocol</h3>
            {Object.entries(definitions.diagnostic_protocol || {}).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', gap: 10, marginBottom: 8 }}>
                <span style={{ color: '#4caf50', fontSize: 11, fontWeight: 700, minWidth: 60 }}>{k.replace('_', ' ').toUpperCase()}</span>
                <span style={{ color: '#e0e0e0', fontSize: 11 }}>{v}</span>
              </div>
            ))}
          </div>

          {/* CI Criteria */}
          <div style={{ background: '#1e1e1e', border: '1px solid #ff9800', borderRadius: 6, padding: 18 }}>
            <h3 style={{ color: '#ffb74d', marginBottom: 12, fontSize: 14 }}>Cochlear Implant Criteria</h3>
            {Object.entries(definitions.cochlear_implant_criteria || {}).map(([k, v]) => (
              <div key={k} style={{ marginBottom: 8 }}>
                <span style={{ color: '#ffb74d', fontSize: 11, fontWeight: 700 }}>{k.replace(/_/g, ' ')}: </span>
                <span style={{ color: '#e0e0e0', fontSize: 11 }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
