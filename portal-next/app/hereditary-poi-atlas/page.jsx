'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  FMR1:  '#1565c0',  // deep blue — FXPOI premutation CGG WES misses
  FOXL2: '#6a1b9a',  // deep purple — BPES type I eyelid triad ptosis repair
  BMP15: '#00695c',  // dark teal — XLD oocyte BMP OHSS risk
  GDF9:  '#e65100',  // amber — AR/AD cumulin heterodimer ICSI
  NR5A1: '#b71c1c',  // deep red — SF-1 adrenal insufficiency Synacthen
  NOBOX: '#2e7d32',  // deep green — AR primary amenorrhoea puberty induction
  FIGLA: '#37474f',  // slate — AR streak gonads primordial assembly
  MCM8:  '#880e4f',  // dark pink — AR DNA repair cancer surveillance
};

const GENE_DISEASE = {
  FMR1:  'FXPOI (XLD-premutation) — TEST FIRST in ALL POI; 20-28% of 55-200 CGG carriers; WES MISSES repeat expansion; PCR + Southern blot',
  FOXL2: 'BPES type I (AD) — Blepharophimosis+Ptosis+Epicanthus inversus VISIBLE AT BIRTH; ptosis repair MANDATORY age 3-4y (amblyopia risk)',
  BMP15: 'XLD POI — Oocyte BMP; heterozygous females POI; hemizygous males fertile; OHSS risk paradox in IVF; low-dose FSH + GnRH agonist trigger',
  GDF9:  'AR/AD POI — Cumulin heterodimer with BMP15 (10x potency); AR biallelic→primary amenorrhoea; AD het→premature menopause; ICSI preferred',
  NR5A1: 'SF-1 (AD) — Master adrenal+gonadal regulator; Synacthen test MANDATORY; adrenal crisis risk; 46,XY NR5A1 LOF→46,XY DSD',
  NOBOX: 'AR POI — Oocyte homeobox TF; primordial→primary follicle transition blocked; primary amenorrhoea; FSH>40-100 IU/L; puberty induction age 11-12y',
  FIGLA: 'AR POI — bHLH TF; primordial follicle ASSEMBLY fails; streak gonads from birth; ZP1/ZP3/ZP4 absent; rarest identifiable AR POI cause',
  MCM8:  'AR POI — Meiotic DNA repair helicase; CANCER SURVEILLANCE from age 30y (colonoscopy+endometrial); avoid cisplatin; Ashkenazi founder p.Lys325Glu',
};

const INHERITANCE = {
  FMR1: 'XLD-premut', FOXL2: 'AD', BMP15: 'XLD', GDF9: 'AR/AD',
  NR5A1: 'AD', NOBOX: 'AR', FIGLA: 'AR', MCM8: 'AR',
};

const POI_GROUP = {
  FMR1:  'Toxic mRNA / Mitochondrial',
  FOXL2: 'Granulosa TF / Eyelid',
  BMP15: 'Oocyte BMP Paracrine',
  GDF9:  'Oocyte GDF Paracrine',
  NR5A1: 'Steroidogenesis / Adrenal',
  NOBOX: 'Oocyte Homeobox TF',
  FIGLA: 'Primordial Assembly TF',
  MCM8:  'Meiotic DNA Repair',
};

export default function HeredPOIAtlasPage() {
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
          fetch(`${API}/api/hereditary-poi-atlas/overview`).then(r => r.json()),
          fetch(`${API}/api/hereditary-poi-atlas/breakdown`).then(r => r.json()),
          fetch(`${API}/api/hereditary-poi-atlas/definitions`).then(r => r.json()),
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

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#121212', minHeight: '100vh' }}>Loading Hereditary POI Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#121212', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#121212', minHeight: '100vh', color: '#fff', padding: '24px 32px', fontFamily: 'monospace' }}>
      <h1 style={{ color: '#ec407a', fontSize: 22, marginBottom: 4 }}>
        🧬 Hereditary Primary Ovarian Insufficiency (POI) Atlas
      </h1>
      <p style={{ color: '#f48fb1', fontSize: 13, marginBottom: 20 }}>
        Complete 8-Gene Reference · FMR1 · FOXL2 · BMP15 · GDF9 · NR5A1 · NOBOX · FIGLA · MCM8 ·
        {overview && ` ${overview.total_patients} patients · seeds ${overview.seed_range}`}
      </p>

      {/* Gene legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
        {Object.entries(GENE_COLORS).map(([gene, col]) => (
          <span key={gene} style={{
            background: col, color: '#fff', padding: '3px 10px',
            borderRadius: 4, fontSize: 11, fontWeight: 700,
          }}>{gene} · {INHERITANCE[gene]} · {POI_GROUP[gene]}</span>
        ))}
      </div>

      {/* Critical clinical alerts */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
        {[
          { label: '🔬 FMR1 TEST FIRST: WES misses CGG repeats — use PCR + Southern blot in ALL POI', col: '#1565c0' },
          { label: '👁️ FOXL2: Ptosis repair MANDATORY age 3-4y — delayed → irreversible amblyopia', col: '#6a1b9a' },
          { label: '⚠ NR5A1: Synacthen test MANDATORY at diagnosis — adrenal crisis risk', col: '#b71c1c' },
          { label: '🔬 MCM8: Cancer surveillance from age 30y — colonoscopy + endometrial', col: '#880e4f' },
          { label: '💉 BMP15 IVF: OHSS paradox — low-dose FSH + GnRH agonist trigger', col: '#00695c' },
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
            background: tab === t ? '#ec407a' : '#1e1e1e',
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
                <div style={{ fontSize: 22, fontWeight: 700, color: '#ec407a' }}>{value}</div>
                <div style={{ fontSize: 11, color: '#aaa', marginTop: 4 }}>{label}</div>
              </div>
            ))}
          </div>

          {/* Clinical pearls */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#ffb300', fontSize: 13, margin: '0 0 12px' }}>🔑 Clinical Pearls — Primary Ovarian Insufficiency</h3>
            <ul style={{ margin: 0, padding: '0 0 0 18px', fontSize: 11, color: '#e0e0e0', lineHeight: 1.9 }}>
              {(overview.clinical_pearls || []).map((p, i) => (
                <li key={i}>{p}</li>
              ))}
            </ul>
          </div>

          {/* Phenotype distribution */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#ec407a', fontSize: 13, margin: '0 0 12px' }}>POI Phenotype Distribution</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {Object.entries(overview.hearing_severity_distribution || {}).sort((a, b) => b[1] - a[1]).slice(0, 18).map(([pheno, n]) => (
                <div key={pheno} style={{
                  background: '#2a2a2a', padding: '6px 12px', borderRadius: 4,
                  fontSize: 11, color: '#e0e0e0',
                }}>
                  <span style={{ color: '#ec407a', fontWeight: 700 }}>{n}</span> · {pheno}
                </div>
              ))}
            </div>
          </div>

          {/* Patients per gene */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#ec407a', fontSize: 13, margin: '0 0 12px' }}>Patients per Gene</h3>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {Object.entries(overview.patients_per_gene || {}).map(([gene, n]) => (
                <div key={gene} style={{
                  background: GENE_COLORS[gene] || '#444',
                  color: '#fff', padding: '6px 14px', borderRadius: 4, fontSize: 12, fontWeight: 700,
                }}>{gene}: {n}</div>
              ))}
            </div>
          </div>

          {/* POI mechanism overview */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16 }}>
              <h3 style={{ color: '#ec407a', fontSize: 13, margin: '0 0 10px' }}>🔬 Oocyte Paracrine / TF Axis</h3>
              <div style={{ fontSize: 11, color: '#e0e0e0', lineHeight: 1.8 }}>
                <div><span style={{ color: '#00695c', fontWeight: 700 }}>BMP15:</span> Oocyte BMP → granulosa SMAD1/5/8; XLD; OHSS paradox</div>
                <div><span style={{ color: '#e65100', fontWeight: 700 }}>GDF9:</span> Oocyte GDF → granulosa SMAD2/3; AR/AD; cumulin</div>
                <div><span style={{ color: '#37474f', fontWeight: 700 }}>FIGLA:</span> bHLH TF → ZP1/ZP3/ZP4 → primordial assembly</div>
                <div><span style={{ color: '#2e7d32', fontWeight: 700 }}>NOBOX:</span> Homeobox TF → GDF9/BMP15/ZP → follicle activation</div>
                <div style={{ marginTop: 8, color: '#80cbc4', fontSize: 10 }}>→ BMP15+GDF9 → CUMULIN (10x potency heterodimer)</div>
                <div style={{ color: '#80cbc4', fontSize: 10 }}>→ ICSI preferred if BMP15 or GDF9 LOF</div>
              </div>
            </div>
            <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16 }}>
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '0 0 10px' }}>🧬 Regulatory / DNA Repair / Systemic</h3>
              <div style={{ fontSize: 11, color: '#e0e0e0', lineHeight: 1.8 }}>
                <div><span style={{ color: '#1565c0', fontWeight: 700 }}>FMR1:</span> Premutation toxic mRNA → granulosa dysfunction → FXTAS</div>
                <div><span style={{ color: '#6a1b9a', fontWeight: 700 }}>FOXL2:</span> Granulosa TF → represses SOX9; eyelid morphogenesis</div>
                <div><span style={{ color: '#b71c1c', fontWeight: 700 }}>NR5A1:</span> SF-1 → adrenal + gonadal steroidogenesis; DSD risk</div>
                <div><span style={{ color: '#880e4f', fontWeight: 700 }}>MCM8:</span> Meiotic DSB repair → cancer instability; surveillance</div>
                <div style={{ marginTop: 8, color: '#ffb300', fontSize: 10 }}>⚠ FMR1: PCR first — WES misses CGG expansions</div>
                <div style={{ color: '#f48fb1', fontSize: 10 }}>⚠ NR5A1: Synacthen test — adrenal crisis without replacement</div>
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
                {['Gene', 'Locus', 'Size', 'Inh.', 'Mechanism', 'Key Distinguisher'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#ec407a', borderBottom: '1px solid #333' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {POI_GENES_TABLE.map((row, i) => (
                <tr key={row.gene} style={{ background: i % 2 === 0 ? '#1a1a1a' : '#1e1e1e' }}>
                  <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] || '#fff' }}>{row.gene}</td>
                  <td style={{ padding: '7px 10px', color: '#aaa' }}>{row.locus}</td>
                  <td style={{ padding: '7px 10px', color: '#ccc' }}>{row.size}</td>
                  <td style={{ padding: '7px 10px', color: '#ffb300' }}>{row.inh}</td>
                  <td style={{ padding: '7px 10px', color: '#90caf9' }}>{row.mechanism}</td>
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
                    {gene} — {data.locus} · {data.protein_size} · {INHERITANCE[gene]} · {POI_GROUP[gene]}
                  </h3>
                  <div style={{ color: '#f48fb1', fontSize: 11, marginBottom: 8 }}>{GENE_DISEASE[gene]}</div>
                </div>
                <span style={{
                  background: GENE_COLORS[gene] || '#444', color: '#fff',
                  padding: '2px 10px', borderRadius: 12, fontSize: 10,
                }}>{data.n_patients} pts</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 8 }}>
                <div>
                  <div style={{ color: '#ffb300', fontSize: 10, marginBottom: 4 }}>POI PHENOTYPE</div>
                  {Object.entries(data.hearing_severity_distribution || {}).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([sev, n]) => (
                    <div key={sev} style={{ fontSize: 10, color: '#ccc', marginBottom: 2 }}>
                      <span style={{ color: '#ec407a', fontWeight: 700 }}>{n}</span> · {sev}
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

              <div style={{ marginTop: 8 }}>
                <div style={{ color: '#aaa', fontSize: 10, marginBottom: 4 }}>Key features:</div>
                <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 10, color: '#e0e0e0', lineHeight: 1.7 }}>
                  {(data.key_features || []).slice(0, 4).map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </div>

              {data.sample_patients && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ color: '#aaa', fontSize: 10, marginBottom: 4 }}>Sample patients:</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {data.sample_patients.slice(0, 3).map(p => (
                      <div key={p.id} style={{
                        background: '#2a2a2a', padding: '4px 8px', borderRadius: 4, fontSize: 9,
                      }}>
                        <span style={{ color: GENE_COLORS[gene] || '#fff', fontWeight: 700 }}>{p.id}</span>
                        {' '}Age {p.age}·{p.sex} · {p.hearing_severity?.slice(0, 35)} · {p.management?.slice(0, 30)}
                        {p.extra_events !== '—' && (
                          <span style={{ color: '#80cbc4' }}> · {p.extra_events.split(';')[0]?.slice(0, 55)}</span>
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
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <h3 style={{ color: '#ff5252', fontSize: 13, margin: '0 0 12px' }}>🚨 Emergency Protocols</h3>
            {(definitions.emergency_protocols || []).map((ep, i) => (
              <div key={i} style={{
                background: '#2a1a1a', border: '1px solid #b71c1c',
                padding: '6px 12px', borderRadius: 4, marginBottom: 6, fontSize: 11, color: '#ef9a9a',
              }}>{ep}</div>
            ))}
          </div>

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
const POI_GENES_TABLE = [
  { gene: 'FMR1',  locus: 'Xq27.3',   size: '632 aa/~71 kDa',  inh: 'XLD-premut', mechanism: 'Toxic mRNA (premutation)', key: 'TEST FIRST all POI; 20-28% of 55-200 CGG carriers; WES misses repeats; PCR+Southern; FXTAS older carriers' },
  { gene: 'FOXL2', locus: '3q22.3',   size: '376 aa/~44 kDa',  inh: 'AD',          mechanism: 'Granulosa TF identity',    key: 'BPES type I: eyelid triad VISIBLE at birth; ptosis repair MANDATORY age 3-4y (amblyopia); BPES II: no POI' },
  { gene: 'BMP15', locus: 'Xp11.22',  size: '392 aa/~46 kDa',  inh: 'XLD',         mechanism: 'Oocyte BMP paracrine',     key: 'Heterozygous females POI; hemizygous males fertile; OHSS paradox IVF; low-dose FSH; GnRH agonist trigger' },
  { gene: 'GDF9',  locus: '5q31.1',   size: '454 aa/~52 kDa',  inh: 'AR/AD',       mechanism: 'Oocyte GDF cumulin',       key: 'AR biallelic→primary amenorrhoea; AD het→premature menopause; cumulin with BMP15 (10x); ICSI preferred' },
  { gene: 'NR5A1', locus: '9q33.3',   size: '461 aa/~52 kDa',  inh: 'AD',          mechanism: 'SF-1 steroidogenesis',     key: 'Synacthen MANDATORY; adrenal crisis risk; inhibin B earliest marker; 46,XY NR5A1 LOF→46,XY DSD' },
  { gene: 'NOBOX', locus: '7q35',     size: '672 aa/~75 kDa',  inh: 'AR',          mechanism: 'Oocyte homeobox TF',       key: 'Primordial→primary follicle transition blocked; primary amenorrhoea; FSH>40-100 IU/L; puberty induction 11-12y' },
  { gene: 'FIGLA', locus: '2p13.3',   size: '115 aa/~13 kDa',  inh: 'AR',          mechanism: 'Primordial follicle assembly', key: 'ZP1/ZP3/ZP4 absent; streak gonads from birth; rarest AR POI; 46,XX streak=low gonadoblastoma risk' },
  { gene: 'MCM8',  locus: '20p12.3',  size: '840 aa/~93 kDa',  inh: 'AR',          mechanism: 'Meiotic DSB DNA repair',   key: 'Cancer surveillance age 30y (colonoscopy+endometrial); avoid cisplatin; Ashkenazi p.Lys325Glu founder' },
];
