'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ANOS1:  '#1565c0',  // deep blue — KS1 XLR absent OB bimanual synkinesis
  FGFR1:  '#4a148c',  // deep purple — KS2 most common AD cleft palate
  PROKR2: '#006064',  // dark teal — KS3 GPCR digenic sleep
  PROK2:  '#e65100',  // amber — KS4 ligand circadian obesity
  CHD7:   '#b71c1c',  // deep red — CHARGE anosmia HH SCC aplasia emergency
  FGF8:   '#2e7d32',  // deep green — KS6 FGF ligand cerebellar vermis
  GNRHR:  '#37474f',  // slate — nIHH normosmic pump FAILS
  KISS1R: '#880e4f',  // dark pink — nIHH normosmic pump WORKS reversal
};

const GENE_DISEASE = {
  ANOS1:  'KS1 (XLR) — Absent olfactory bulbs MRI PATHOGNOMONIC; bimanual synkinesis 50%; unilateral renal agenesis 25%; cryptorchidism ~95%',
  FGFR1:  'KS2 (AD) — Most common AD Kallmann; cleft palate/lip 10-15%; digital anomalies; incomplete penetrance ~30-40%; FGF8-FGFR1 axis',
  PROKR2: 'KS3 (AR/dig) — GPCR; variable anosmia/hyposmia; sleep disorder; digenic with PROK2/FGFR1; PROKR2 het alone often insufficient',
  PROK2:  'KS4 (AR/dig) — Prokineticin 2 ligand 81aa; circadian disorder; obesity overlap; digenic with PROKR2; smallest KS protein',
  CHD7:   'CHARGE (AD de novo) — Anosmia + HH 60-80%; SCC aplasia CT near-PATHOGNOMONIC; bilateral choanal atresia = neonatal EMERGENCY',
  FGF8:   'KS6 (AD) — FGF8 ligand for FGFR1; same pathway as KS2; cleft palate; cerebellar vermis hypoplasia rare; incomplete penetrance',
  GNRHR:  'nIHH (AR) — Normosmic IHH; olfactory bulbs PRESENT; pulsatile GnRH pump DOES NOT WORK (pituitary receptor LOF) — use hCG+rFSH',
  KISS1R: 'nIHH (AR) — Normosmic IHH; kisspeptin receptor; pulsatile GnRH pump WORKS; reversal ~10-20%; puberty switch gene',
};

const INHERITANCE = {
  ANOS1: 'XLR', FGFR1: 'AD', PROKR2: 'AR/dig', PROK2: 'AR/dig',
  CHD7: 'AD de novo', FGF8: 'AD', GNRHR: 'AR', KISS1R: 'AR',
};

const KS_GROUP = {
  ANOS1: 'Kallmann', FGFR1: 'Kallmann', PROKR2: 'Kallmann', PROK2: 'Kallmann',
  CHD7: 'CHARGE/KS', FGF8: 'Kallmann', GNRHR: 'Normosmic IHH', KISS1R: 'Normosmic IHH',
};

export default function HeredKallmannIHHAtlasPage() {
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
          fetch(`${API}/api/hereditary-kallmann-ihh-atlas/overview`).then(r => r.json()),
          fetch(`${API}/api/hereditary-kallmann-ihh-atlas/breakdown`).then(r => r.json()),
          fetch(`${API}/api/hereditary-kallmann-ihh-atlas/definitions`).then(r => r.json()),
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

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#121212', minHeight: '100vh' }}>Loading Hereditary Kallmann / IHH Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#121212', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#121212', minHeight: '100vh', color: '#fff', padding: '24px 32px', fontFamily: 'monospace' }}>
      <h1 style={{ color: '#29b6f6', fontSize: 22, marginBottom: 4 }}>
        🧬 Hereditary Kallmann Syndrome / IHH Atlas
      </h1>
      <p style={{ color: '#90caf9', fontSize: 13, marginBottom: 20 }}>
        Complete 8-Gene Reference · ANOS1 · FGFR1 · PROKR2 · PROK2 · CHD7 · FGF8 · GNRHR · KISS1R ·
        {overview && ` ${overview.total_patients} patients · seeds ${overview.seed_range}`}
      </p>

      {/* Gene legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
        {Object.entries(GENE_COLORS).map(([gene, col]) => (
          <span key={gene} style={{
            background: col, color: '#fff', padding: '3px 10px',
            borderRadius: 4, fontSize: 11, fontWeight: 700,
          }}>{gene} · {INHERITANCE[gene]} · {KS_GROUP[gene]}</span>
        ))}
      </div>

      {/* Critical clinical alerts */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
        {[
          { label: '🚨 CHD7: Bilateral Choanal Atresia = Neonatal Airway EMERGENCY', col: '#b71c1c' },
          { label: '⛔ GNRHR: Pulsatile GnRH Pump DOES NOT WORK — use hCG + rFSH', col: '#37474f' },
          { label: '✅ KISS1R: Pulsatile GnRH Pump WORKS; Reversal ~10-20%', col: '#880e4f' },
          { label: '🧠 ANOS1: Bimanual Synkinesis 50% — PATHOGNOMONIC for KS1', col: '#1565c0' },
          { label: '👃 Normosmic IHH: Olfactory Bulbs PRESENT on MRI (DDx Kallmann)', col: '#006064' },
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

          {/* Critical clinical pearls */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#ffb300', fontSize: 13, margin: '0 0 12px' }}>🔑 Clinical Pearls — Kallmann Syndrome / IHH</h3>
            <ul style={{ margin: 0, padding: '0 0 0 18px', fontSize: 11, color: '#e0e0e0', lineHeight: 1.9 }}>
              {(overview.clinical_pearls || []).map((p, i) => (
                <li key={i}>{p}</li>
              ))}
            </ul>
          </div>

          {/* Phenotype distribution */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#29b6f6', fontSize: 13, margin: '0 0 12px' }}>Phenotype Distribution (IHH Severity)</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {Object.entries(overview.hearing_severity_distribution || {}).sort((a, b) => b[1] - a[1]).slice(0, 16).map(([pheno, n]) => (
                <div key={pheno} style={{
                  background: '#2a2a2a', padding: '6px 12px', borderRadius: 4,
                  fontSize: 11, color: '#e0e0e0',
                }}>
                  <span style={{ color: '#29b6f6', fontWeight: 700 }}>{n}</span> · {pheno}
                </div>
              ))}
            </div>
          </div>

          {/* Patients per gene */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 20 }}>
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

          {/* Kallmann vs normosmic IHH differentiation */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16 }}>
              <h3 style={{ color: '#29b6f6', fontSize: 13, margin: '0 0 10px' }}>🔴 Kallmann Syndrome (Anosmia + HH)</h3>
              <div style={{ fontSize: 11, color: '#e0e0e0', lineHeight: 1.8 }}>
                <div><span style={{ color: '#1565c0', fontWeight: 700 }}>ANOS1:</span> XLR · absent OB · synkinesis 50% · renal agenesis 25%</div>
                <div><span style={{ color: '#4a148c', fontWeight: 700 }}>FGFR1:</span> AD · cleft palate · digital anomalies · FGF pathway</div>
                <div><span style={{ color: '#006064', fontWeight: 700 }}>PROKR2:</span> AR/dig · GPCR · sleep disorder · variable anosmia</div>
                <div><span style={{ color: '#e65100', fontWeight: 700 }}>PROK2:</span> AR/dig · 81aa · circadian · obesity · ligand for PROKR2</div>
                <div><span style={{ color: '#b71c1c', fontWeight: 700 }}>CHD7:</span> CHARGE · SCC aplasia · coloboma · choanal atresia ⚠</div>
                <div><span style={{ color: '#2e7d32', fontWeight: 700 }}>FGF8:</span> AD · FGFR1 ligand · cerebellar vermis · same pathway KS2</div>
                <div style={{ marginTop: 8, color: '#80cbc4', fontSize: 10 }}>→ MRI: absent/hypoplastic olfactory bulbs</div>
                <div style={{ color: '#80cbc4', fontSize: 10 }}>→ GnRH pump WORKS (pituitary intact)</div>
              </div>
            </div>
            <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16 }}>
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '0 0 10px' }}>🟡 Normosmic IHH (No Anosmia)</h3>
              <div style={{ fontSize: 11, color: '#e0e0e0', lineHeight: 1.8 }}>
                <div><span style={{ color: '#37474f', fontWeight: 700 }}>GNRHR:</span> AR · pituitary GnRH-RESISTANT · pump FAILS · use hCG+FSH</div>
                <div><span style={{ color: '#880e4f', fontWeight: 700 }}>KISS1R:</span> AR · kisspeptin receptor · pump WORKS · reversal ~10-20%</div>
                <div style={{ marginTop: 8, color: '#80cbc4', fontSize: 10 }}>→ MRI: olfactory bulbs PRESENT and NORMAL</div>
                <div style={{ color: '#80cbc4', fontSize: 10 }}>→ Smell test NORMAL (UPSIT ≥34/40)</div>
                <div style={{ color: '#ffb300', fontSize: 10, marginTop: 4 }}>⚠ GNRHR: GnRH pump DOES NOT WORK — critical management difference</div>
                <div style={{ color: '#4caf50', fontSize: 10 }}>✓ KISS1R: pituitary + GNRHR intact → GnRH pump effective</div>
              </div>
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
                {['Gene', 'Locus', 'Size', 'Inh.', 'Type', 'Key Distinguisher'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#29b6f6', borderBottom: '1px solid #333' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {KS_IHH_GENES_TABLE.map((row, i) => (
                <tr key={row.gene} style={{ background: i % 2 === 0 ? '#1a1a1a' : '#1e1e1e' }}>
                  <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] || '#fff' }}>{row.gene}</td>
                  <td style={{ padding: '7px 10px', color: '#aaa' }}>{row.locus}</td>
                  <td style={{ padding: '7px 10px', color: '#ccc' }}>{row.size}</td>
                  <td style={{ padding: '7px 10px', color: '#ffb300' }}>{row.inh}</td>
                  <td style={{ padding: '7px 10px', color: row.type === 'Normosmic IHH' ? '#ffb300' : '#90caf9' }}>{row.type}</td>
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
                    {gene} — {data.locus} · {data.protein_size} · {INHERITANCE[gene]} · {KS_GROUP[gene]}
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
                  <div style={{ color: '#ffb300', fontSize: 10, marginBottom: 4 }}>IHH PHENOTYPE</div>
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

              {/* Key features */}
              <div style={{ marginTop: 8 }}>
                <div style={{ color: '#aaa', fontSize: 10, marginBottom: 4 }}>Key features:</div>
                <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 10, color: '#e0e0e0', lineHeight: 1.7 }}>
                  {(data.key_features || []).slice(0, 4).map((f, i) => <li key={i}>{f}</li>)}
                </ul>
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
                        {' '}Age {p.age}{p.sex} · {p.hearing_severity?.slice(0, 35)} · {p.management?.slice(0, 30)}
                        {p.extra_events !== '—' && (
                          <span style={{ color: '#80cbc4' }}> · {p.extra_events.split(';')[0]?.slice(0, 50)}</span>
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
const KS_IHH_GENES_TABLE = [
  { gene: 'ANOS1',  locus: 'Xp22.31',  size: '680 aa/~100 kDa', inh: 'XLR',        type: 'Kallmann KS1',   key: 'Absent olfactory bulbs MRI PATHOGNOMONIC; bimanual synkinesis 50%; unilateral renal agenesis 25%; cryptorchidism ~95%' },
  { gene: 'FGFR1',  locus: '8p11.23',  size: '822 aa/~92 kDa',  inh: 'AD',         type: 'Kallmann KS2',   key: 'Most common AD KS; cleft palate/lip 10-15%; digital anomalies; incomplete penetrance ~30-40%; FGF8-FGFR1 axis' },
  { gene: 'PROKR2', locus: '20p13',    size: '384 aa/~43 kDa',  inh: 'AR/dig',     type: 'Kallmann KS3',   key: 'GPCR for PROK2; variable hyposmia; sleep disorder; digenic with PROK2/FGFR1 — het alone often insufficient' },
  { gene: 'PROK2',  locus: '3p13',     size: '81 aa/~9 kDa',    inh: 'AR/dig',     type: 'Kallmann KS4',   key: 'Prokineticin 2 ligand 81aa; AVITGA motif; circadian disorder (SCN); obesity; digenic with PROKR2' },
  { gene: 'CHD7',   locus: '8q12.2',   size: '2997 aa/~337 kDa',inh: 'AD de novo', type: 'CHARGE/KS',      key: 'CHARGE: anosmia + HH 60-80%; SCC aplasia CT near-PATHOGNOMONIC; choanal atresia = neonatal EMERGENCY; coloboma' },
  { gene: 'FGF8',   locus: '10q24.32', size: '215 aa/~23 kDa',  inh: 'AD',         type: 'Kallmann KS6',   key: 'FGF8 ligand for FGFR1 (same KS2 pathway); cleft palate; cerebellar vermis hypoplasia rare; incomplete penetrance' },
  { gene: 'GNRHR',  locus: '4q13.2',   size: '328 aa/~37 kDa',  inh: 'AR',         type: 'Normosmic IHH',  key: 'Normosmic IHH; OB PRESENT; smell NORMAL; pituitary GnRH receptor LOF → pump FAILS; use hCG + rFSH for fertility' },
  { gene: 'KISS1R', locus: '19p13.3',  size: '398 aa/~45 kDa',  inh: 'AR',         type: 'Normosmic IHH',  key: 'Normosmic IHH; kisspeptin receptor; puberty switch; GnRH pump WORKS (pituitary intact); reversal ~10-20%' },
];
