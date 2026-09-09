'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PAX3:  '#1565c0',  // deep blue — WS1 most common WS gene
  MITF:  '#4a148c',  // deep purple — WS2A master melanocyte regulator
  SOX10: '#b71c1c',  // deep red — WS4C/PCWH HSCR + neuropathy
  EDNRB: '#e65100',  // amber — WS4A AR Waardenburg-Shah
  EYA1:  '#2e7d32',  // deep green — BOR syndrome branchial + renal
  CHD7:  '#006064',  // dark teal — CHARGE most common syndromic SNHL
  TCOF1: '#37474f',  // slate — Treacher Collins absent malar/zygoma
  GATA3: '#880e4f',  // dark pink — HDR Barakat hypoparathyroidism
};

const GENE_DISEASE = {
  PAX3:  'WS1/WS3 (AD) — Dystopia canthorum W-index ≥1.95 PATHOGNOMONIC; SNHL ~57%; most common WS gene; white forelock ~45%',
  MITF:  'WS2A (AD) — NO dystopia canthorum (DDx WS1); most common WS2 ~40%; Tietz syndrome: severe alleles → complete albinism + profound SNHL',
  SOX10: 'WS4C/PCWH (AD) — HSCR + peripheral demyelinating neuropathy + WS; PCWH: alleles escaping NMD → dominant-negative (most severe)',
  EDNRB: 'WS4A (AR/AD) — Waardenburg-Shah; biallelic = HSCR + full WS4A; heterozygous = isolated HSCR only (no WS features); S305N Mennonite founder',
  EYA1:  'BOR syndrome (AD) — Branchial fistulae/cysts/tags + Mondini cochlea + Renal dysplasia; AVOID aminoglycosides (ototoxic + nephrotoxic)',
  CHD7:  'CHARGE (AD de novo) — SCC aplasia CT near-PATHOGNOMONIC; bilateral choanal atresia = neonatal airway EMERGENCY; absent VOR',
  TCOF1: 'Treacher Collins TCS1 (AD) — Absent malar/zygoma 3D CT PATHOGNOMONIC; predominantly CHL; BAHA FIRST-LINE bilateral atresia from 6m',
  GATA3: 'HDR/Barakat (AD) — Hypoparathyroidism + SNHL + Renal; CORRECT CALCIUM BEFORE audiometry; IV calcium for tetanic emergency',
};

const INHERITANCE = {
  PAX3: 'AD', MITF: 'AD', SOX10: 'AD', EDNRB: 'AR/AD',
  EYA1: 'AD', CHD7: 'AD (de novo)', TCOF1: 'AD', GATA3: 'AD',
};

export default function HereditorySyndrHLAtlasPage() {
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
          fetch(`${API}/api/hereditary-syndromic-hearing-loss-atlas/overview`).then(r => r.json()),
          fetch(`${API}/api/hereditary-syndromic-hearing-loss-atlas/breakdown`).then(r => r.json()),
          fetch(`${API}/api/hereditary-syndromic-hearing-loss-atlas/definitions`).then(r => r.json()),
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

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#121212', minHeight: '100vh' }}>Loading Hereditary Syndromic Hearing Loss Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#121212', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#121212', minHeight: '100vh', color: '#fff', padding: '24px 32px', fontFamily: 'monospace' }}>
      <h1 style={{ color: '#29b6f6', fontSize: 22, marginBottom: 4 }}>
        🧬 Hereditary Syndromic Hearing Loss Atlas
      </h1>
      <p style={{ color: '#90caf9', fontSize: 13, marginBottom: 20 }}>
        Complete 8-Gene Reference · PAX3 · MITF · SOX10 · EDNRB · EYA1 · CHD7 · TCOF1 · GATA3 ·
        {overview && ` ${overview.total_patients} patients · seeds ${overview.seed_range}`}
      </p>

      {/* Gene legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
        {Object.entries(GENE_COLORS).map(([gene, col]) => (
          <span key={gene} style={{
            background: col, color: '#fff', padding: '3px 10px',
            borderRadius: 4, fontSize: 11, fontWeight: 700,
          }}>{gene} · {INHERITANCE[gene]}</span>
        ))}
      </div>

      {/* Emergency alerts */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
        {[
          { label: '🚨 CHD7: Bilateral Choanal Atresia = Neonatal Airway EMERGENCY', col: '#b71c1c' },
          { label: '⚡ GATA3: IV Calcium Gluconate for Tetanic EMERGENCY', col: '#880e4f' },
          { label: '🚨 SOX10/EDNRB: HSCR Neonatal Surgical Assessment', col: '#e65100' },
          { label: '⚠ EYA1: AVOID Aminoglycosides (ototoxic + nephrotoxic)', col: '#2e7d32' },
        ].map(({ label, col }) => (
          <div key={label} style={{
            background: col + '33', border: `1px solid ${col}`,
            padding: '4px 10px', borderRadius: 4, fontSize: 11,
          }}>{label}</div>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 4, cursor: 'pointer', fontSize: 12,
            background: tab === t ? '#29b6f6' : '#1e1e1e',
            color: tab === t ? '#000' : '#aaa',
            border: tab === t ? 'none' : '1px solid #333',
            fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Genes Covered', value: overview.genes_covered },
              { label: 'Seed Range', value: overview.seed_range },
              { label: 'Patients / Gene', value: '40 per gene' },
            ].map(({ label, value }) => (
              <div key={label} style={{ background: '#1e1e1e', padding: 16, borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 22, fontWeight: 700, color: '#29b6f6' }}>{value}</div>
                <div style={{ fontSize: 11, color: '#aaa', marginTop: 4 }}>{label}</div>
              </div>
            ))}
          </div>

          {/* Clinical pearls */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#ffb300', fontSize: 13, margin: '0 0 12px' }}>🔑 Clinical Pearls — Syndromic Hearing Loss</h3>
            <ul style={{ margin: 0, padding: '0 0 0 18px', fontSize: 11, color: '#e0e0e0', lineHeight: 1.9 }}>
              {(overview.clinical_pearls || []).map((p, i) => (
                <li key={i}>{p}</li>
              ))}
            </ul>
          </div>

          {/* Hearing severity chart */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#29b6f6', fontSize: 13, margin: '0 0 12px' }}>Hearing Severity Distribution</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {Object.entries(overview.hearing_severity_distribution || {}).sort((a, b) => b[1] - a[1]).map(([sev, n]) => (
                <div key={sev} style={{
                  background: '#2a2a2a', padding: '6px 12px', borderRadius: 4,
                  fontSize: 11, color: '#e0e0e0',
                }}>
                  <span style={{ color: '#29b6f6', fontWeight: 700 }}>{n}</span> · {sev}
                </div>
              ))}
            </div>
          </div>

          {/* Patients per gene */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16 }}>
            <h3 style={{ color: '#29b6f6', fontSize: 13, margin: '0 0 12px' }}>Patients per Gene</h3>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {Object.entries(overview.patients_per_gene || {}).map(([gene, n]) => (
                <div key={gene} style={{
                  background: GENE_COLORS[gene] || '#444',
                  color: '#fff', padding: '6px 14px', borderRadius: 4, fontSize: 12, fontWeight: 700,
                }}>{gene}: {n}</div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ background: '#1e1e1e' }}>
                {['Gene', 'Locus', 'Size', 'Inh.', 'Syndrome', 'Key Distinguisher'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#29b6f6', borderBottom: '1px solid #333' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {SHL_GENES_TABLE.map((row, i) => (
                <tr key={row.gene} style={{ background: i % 2 === 0 ? '#1a1a1a' : '#1e1e1e' }}>
                  <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] || '#fff' }}>{row.gene}</td>
                  <td style={{ padding: '7px 10px', color: '#aaa' }}>{row.locus}</td>
                  <td style={{ padding: '7px 10px', color: '#ccc' }}>{row.size}</td>
                  <td style={{ padding: '7px 10px', color: '#ffb300' }}>{row.inh}</td>
                  <td style={{ padding: '7px 10px', color: '#e0e0e0' }}>{row.syndrome}</td>
                  <td style={{ padding: '7px 10px', color: '#80cbc4', fontSize: 10 }}>{row.key}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {Object.entries(breakdown.breakdown_by_gene || {}).map(([gene, data]) => (
            <div key={gene} style={{
              background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 16,
              borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}`,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <h3 style={{ color: GENE_COLORS[gene] || '#fff', margin: '0 0 4px', fontSize: 15 }}>
                    {gene} — {data.locus} · {data.protein_size} · {INHERITANCE[gene]}
                  </h3>
                  <div style={{ color: '#90caf9', fontSize: 11, marginBottom: 8 }}>{GENE_DISEASE[gene]}</div>
                </div>
                <span style={{
                  background: GENE_COLORS[gene] || '#444', color: '#fff',
                  padding: '2px 10px', borderRadius: 12, fontSize: 10,
                }}>{data.n_patients} pts</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 8 }}>
                <div>
                  <div style={{ color: '#ffb300', fontSize: 10, marginBottom: 4 }}>HEARING SEVERITY</div>
                  {Object.entries(data.hearing_severity_distribution || {}).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([sev, n]) => (
                    <div key={sev} style={{ fontSize: 10, color: '#ccc', marginBottom: 2 }}>
                      <span style={{ color: '#29b6f6', fontWeight: 700 }}>{n}</span> · {sev}
                    </div>
                  ))}
                </div>
                <div>
                  <div style={{ color: '#ffb300', fontSize: 10, marginBottom: 4 }}>MANAGEMENT</div>
                  {Object.entries(data.management_distribution || {}).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([mgmt, n]) => (
                    <div key={mgmt} style={{ fontSize: 10, color: '#ccc', marginBottom: 2 }}>
                      <span style={{ color: '#80cbc4', fontWeight: 700 }}>{n}</span> · {mgmt}
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ color: '#b0bec5', fontSize: 10, marginTop: 6 }}>
                <span style={{ color: '#ffb300' }}>Surgical urgency: </span>{data.surgical_urgency}
              </div>

              {/* Sample patients */}
              {data.sample_patients && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ color: '#aaa', fontSize: 10, marginBottom: 4 }}>Sample patients:</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {data.sample_patients.slice(0, 3).map(p => (
                      <div key={p.id} style={{
                        background: '#2a2a2a', padding: '4px 8px', borderRadius: 4, fontSize: 9,
                      }}>
                        <span style={{ color: GENE_COLORS[gene] || '#fff', fontWeight: 700 }}>{p.id}</span>
                        {' '}Age {p.age}{p.sex} · {p.hearing_severity} · {p.management}
                        {p.extra_events !== '—' && (
                          <span style={{ color: '#80cbc4' }}> · {p.extra_events.split(';')[0]}</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          {/* Emergencies */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <h3 style={{ color: '#ff5252', fontSize: 13, margin: '0 0 12px' }}>🚨 Emergency Protocols</h3>
            {(definitions.emergency_protocols || []).map((ep, i) => (
              <div key={i} style={{
                background: '#2a1a1a', border: '1px solid #b71c1c',
                padding: '6px 12px', borderRadius: 4, marginBottom: 6, fontSize: 11, color: '#ef9a9a',
              }}>{ep}</div>
            ))}
          </div>

          {/* Definitions */}
          {Object.entries(definitions.definitions || {}).map(([term, def]) => (
            <div key={term} style={{
              background: '#1e1e1e', borderRadius: 8, padding: 14, marginBottom: 10,
            }}>
              <div style={{ color: '#ffb300', fontWeight: 700, fontSize: 12, marginBottom: 6 }}>
                {term.replace(/_/g, ' ')}
              </div>
              <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.7 }}>{def}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Static gene table data
const SHL_GENES_TABLE = [
  { gene: 'PAX3',  locus: '2q36.1',  size: '505 aa/56 kDa',  inh: 'AD',         syndrome: 'Waardenburg Type 1/3 (WS1/WS3)',             key: 'W-index ≥1.95 PATHOGNOMONIC; SNHL ~57%; most common WS gene (~50%); white forelock ~45%' },
  { gene: 'MITF',  locus: '3p13',    size: '526 aa/58 kDa',  inh: 'AD',         syndrome: 'Waardenburg Type 2A (WS2A) / Tietz syndrome', key: 'NO dystopia canthorum (DDx WS1); most common WS2 ~40%; Tietz = severe alleles → complete albinism' },
  { gene: 'SOX10', locus: '22q13.1', size: '466 aa/52 kDa',  inh: 'AD',         syndrome: 'Waardenburg Type 4C / PCWH syndrome',          key: 'HSCR + demyelinating neuropathy; PCWH = NMD-escape alleles (dominant-negative)' },
  { gene: 'EDNRB', locus: '13q22.3', size: '442 aa/50 kDa',  inh: 'AR/AD',      syndrome: 'Waardenburg Type 4A (Waardenburg-Shah)',        key: 'AR biallelic = HSCR + full WS4A; heterozygous = isolated HSCR only (~20-30%); S305N Mennonite founder' },
  { gene: 'EYA1',  locus: '8q13.3',  size: '559 aa/61 kDa',  inh: 'AD',         syndrome: 'BOR syndrome / Branchiootic (BO)',              key: 'Branchial fistulae + Mondini cochlea + Renal dysplasia; AVOID aminoglycosides' },
  { gene: 'CHD7',  locus: '8q12.2',  size: '2997 aa/337 kDa',inh: 'AD de novo', syndrome: 'CHARGE syndrome',                              key: 'SCC aplasia CT near-PATHOGNOMONIC; bilateral choanal atresia = neonatal airway EMERGENCY; absent VOR' },
  { gene: 'TCOF1', locus: '5q33.1',  size: '1411 aa/152 kDa',inh: 'AD',         syndrome: 'Treacher Collins Syndrome (TCS1)',              key: 'Absent malar/zygoma 3D CT PATHOGNOMONIC; predominantly CHL; BAHA FIRST-LINE bilateral atresia' },
  { gene: 'GATA3', locus: '10p15.3', size: '444 aa/48 kDa',  inh: 'AD',         syndrome: 'HDR syndrome / Barakat syndrome',               key: 'Hypoparathyroidism + SNHL + Renal; CORRECT CALCIUM before audiometry; IV calcium for tetany EMERGENCY' },
];
