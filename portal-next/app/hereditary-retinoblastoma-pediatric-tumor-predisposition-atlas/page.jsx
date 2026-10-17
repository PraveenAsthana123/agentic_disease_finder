'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-retinoblastoma-pediatric-tumor-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'RB1':     '#b71c1c',   // deep red         -- Hereditary Rb; bilateral PATHOGNOMONIC; CDK4/6i INACTIVE
  'DICER1':  '#880e4f',   // deep crimson      -- DICER1 syndrome; PPB PATHOGNOMONIC <8yr
  'SMARCB1': '#4a148c',   // deep purple       -- RTPS1; AT/RT; radiation CI <36mo
  'SMARCA4': '#1565c0',   // deep blue         -- RTPS2; SCCOHT; hypercalcemia PATHOGNOMONIC
  'PTCH1':   '#2e7d32',   // forest green      -- Gorlin/NBCCS; radiation ABSOLUTELY CI; OKC jaw cysts
  'SUFU':    '#e65100',   // burnt orange      -- SHH-MB; Gorlin-like; vismodegib NOT effective
  'GPC3':    '#006064',   // dark teal         -- SGB; XLR overgrowth; AFP/USS surveillance
  'ALK':     '#33691e',   // olive green       -- Familial NB; F1174L/R1275Q; lorlatinib
};

const GENE_INFO = {
  'RB1':     { full: 'Retinoblastoma-protein / bilateral-Rb-45pct-germline PATHOGNOMONIC / trilateral-Rb-pinealoblastoma PATHOGNOMONIC / second-primary-osteosarcoma-30-35pct / CDK4/6i-INACTIVE-Rb-null', locus: '13q14.2', size: '928 aa / 110 kDa', inh: 'AD LOF' },
  'DICER1':  { full: 'DICER1-syndrome / PPB-PATHOGNOMONIC-child-<8yr / thyroid-nodule-adolescent / ovarian-SLCT-androgenic / 2nd-hit-somatic-hotspot-metal-binding-exons-24-25', locus: '14q32.13', size: '1922 aa / 218 kDa', inh: 'AD LOF' },
  'SMARCB1': { full: 'RTPS1 / AT/RT-brain / RTK-kidney / SMARCB1-BAF47-IHC-loss PATHOGNOMONIC / radiation-CONTRAINDICATED-<36-months / de-novo-40pct', locus: '22q11.23', size: '385 aa / 44 kDa', inh: 'AD LOF (INI1/BAF47)' },
  'SMARCA4': { full: 'RTPS2 / SCCOHT-young-women PATHOGNOMONIC / hypercalcemia-PATHOGNOMONIC / BRG1-IHC-loss / Coffin-Siris-syndrome / 43pct-germline-SCCOHT', locus: '19p13.2', size: '1647 aa / 185 kDa', inh: 'AD LOF (BRG1)' },
  'PTCH1':   { full: 'Gorlin-syndrome / multiple-BCC PATHOGNOMONIC / OKC-jaw-keratocysts PATHOGNOMONIC / falx-cerebri-calcification PATHOGNOMONIC / bifid-ribs / radiation-ABSOLUTELY-CONTRAINDICATED-massive-BCC-induction', locus: '9q22.32', size: '1447 aa / 162 kDa', inh: 'AD LOF' },
  'SUFU':    { full: 'SHH-medulloblastoma-predisposition / Gorlin-like-fewer-BCC-jaw-cysts / desmoplastic-nodular-MB / vismodegib-NOT-effective-SUFU / 25pct-MB-risk-higher-than-PTCH1', locus: '10q24.32', size: '484 aa / 55 kDa', inh: 'AD LOF' },
  'GPC3':    { full: 'Simpson-Golabi-Behmel-syndrome / XLR-LOF / overgrowth PATHOGNOMONIC / macrosomia-coarse-facies-polydactyly / embryonal-tumor-10pct (Wilms-hepatoblastoma-NB) / AFP-USS-every-3mo-8yr', locus: 'Xq26.2', size: '580 aa / 70 kDa', inh: 'XLR LOF' },
  'ALK':     { full: 'Familial-neuroblastoma / AD-GOF / F1174L-R1275Q-most-common / 75pct-penetrance / bilateral-adrenal-NB PATHOGNOMONIC / lorlatinib-preferred-F1174L (crizotinib-partially-resistant)', locus: '2p23.2', size: '1620 aa / 177 kDa', inh: 'AD GOF (RTK)' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#333', color: '#fff', padding: '2px 7px',
      borderRadius: 4, fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneCard({ gene, color, info, breakdown }) {
  const bg = color;
  const cohort = breakdown?.gene_data?.[gene] || {};
  const geneAtlas = breakdown?.atlas_genes?.find(g => g.gene === gene) || {};
  return (
    <div style={{ border: `2px solid ${bg}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ background: bg, color: '#fff', borderRadius: 6, padding: '6px 14px', fontWeight: 700, fontSize: 18 }}>{gene}</div>
        <div>
          <div style={{ fontWeight: 600 }}>{geneAtlas.syndrome?.split('/')[0] || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      <div style={{ marginBottom: 8 }}>
        <Badge text={`Tumor risk: ${(geneAtlas.bmf_risk || '').split(';')[0]}`} color={bg} />
      </div>
      {geneAtlas.pathognomonic && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ⚠ <b>Pathognomonic:</b> {geneAtlas.pathognomonic}
        </div>
      )}
      {geneAtlas.key_rule && (
        <div style={{ fontSize: 12, background: '#e8f5e9', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ✓ <b>Key rule:</b> {geneAtlas.key_rule}
        </div>
      )}
      {geneAtlas.key_avoid && (
        <div style={{ fontSize: 12, background: '#ffebee', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ✗ <b>Avoid:</b> {geneAtlas.key_avoid}
        </div>
      )}
      {geneAtlas.targeted_rx && (
        <div style={{ fontSize: 12, background: '#e3f2fd', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          💊 <b>Treatment:</b> {geneAtlas.targeted_rx}
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginTop: 10 }}>
        {[
          ['N', cohort.n],
          ['Avg age', cohort.mean_age != null ? cohort.mean_age + 'yr' : '—'],
          ['% Female', cohort.pct_female != null ? cohort.pct_female + '%' : '—'],
          ['Bilateral', gene === 'RB1' ? (cohort.bilateral_n != null ? `${cohort.bilateral_n}/40` : '—') : 'N/A'],
          ['Metastatic', cohort.metastatic_n != null ? `${cohort.metastatic_n}/40` : '—'],
          ['2nd primary', cohort.second_primary_n != null ? `${cohort.second_primary_n}/40` : '—'],
        ].map(([label, val]) => (
          <div key={label} style={{ background: '#f5f5f5', borderRadius: 4, padding: '6px 8px', fontSize: 12 }}>
            <div style={{ color: '#888', fontSize: 10 }}>{label}</div>
            <div style={{ fontWeight: 600 }}>{val ?? '—'}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function HerediaryRetinoblastomaPediatricTumorAtlasPage() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, fontFamily: 'monospace' }}>Loading Hereditary Retinoblastoma &amp; Pediatric Solid Tumor Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', maxWidth: 1100, margin: '0 auto', padding: 24 }}>
      <div style={{ background: 'linear-gradient(135deg,#b71c1c,#4a148c)', color: '#fff', borderRadius: 10, padding: '24px 32px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>🧬 Hereditary Retinoblastoma &amp; Pediatric Solid Tumor Predisposition Atlas</h1>
        <div style={{ marginTop: 8, opacity: 0.9, fontSize: 13 }}>
          Complete 8-Gene Reference — RB1 · DICER1 · SMARCB1 · SMARCA4 · PTCH1 · SUFU · GPC3 · ALK
        </div>
        <div style={{ marginTop: 6, opacity: 0.8, fontSize: 12 }}>
          320-patient aggregate (8 × 40, seeds 3702–3709) · Rb / DICER1-Syndrome / Rhabdoid / Gorlin / SGB / Familial-NB
        </div>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {(overview?.genes || []).map(g => (
            <span key={g} style={{ background: GENE_COLORS[g] || '#555', color: '#fff', borderRadius: 4, padding: '3px 10px', fontSize: 12, fontWeight: 600 }}>{g}</span>
          ))}
        </div>
      </div>

      {/* Critical alerts */}
      <div style={{ background: '#ffebee', border: '2px solid #c62828', borderRadius: 8, padding: '12px 16px', marginBottom: 16 }}>
        <b style={{ color: '#b71c1c' }}>⚠ CRITICAL RULES:</b>
        <ul style={{ margin: '6px 0 0 0', paddingLeft: 20, fontSize: 13 }}>
          <li><b>RB1:</b> Bilateral Rb = assume germline; EUA every 3-4 weeks; CDK4/6i <b>INACTIVE</b> in Rb-null tumors</li>
          <li><b>DICER1:</b> PPB in child &lt;8yr = test germline DICER1; 2nd hit = somatic hotspot (not full LOH)</li>
          <li><b>SMARCB1:</b> Radiation <b>CONTRAINDICATED</b> in infants &lt;36 months with AT/RT (catastrophic neurotoxicity)</li>
          <li><b>PTCH1:</b> Radiation <b>ABSOLUTELY CONTRAINDICATED</b> (hundreds BCCs in irradiated field — multiple fatalities)</li>
          <li><b>SUFU:</b> Vismodegib <b>NOT effective</b> in SUFU-germline MB (SUFU acts downstream of SMO)</li>
          <li><b>ALK F1174L:</b> Lorlatinib preferred over crizotinib (F1174L partially crizotinib-resistant)</li>
        </ul>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '2px solid #e0e0e0', marginBottom: 20 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setActiveTab(i)} style={{
            padding: '10px 22px', border: 'none', background: 'none', cursor: 'pointer',
            fontWeight: activeTab === i ? 700 : 400,
            borderBottom: activeTab === i ? '3px solid #b71c1c' : '3px solid transparent',
            color: activeTab === i ? '#b71c1c' : '#555', fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      {/* Overview tab */}
      {activeTab === 0 && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14, marginBottom: 20 }}>
            {[
              ['Total Patients', overview.total_patients],
              ['Genes', overview.gene_count],
              ['Seeds', overview.seeds],
              ['Tumor Categories', Object.keys(overview.clinical_categories || {}).length],
            ].map(([l, v]) => (
              <div key={l} style={{ background: '#fce4ec', borderRadius: 8, padding: '14px 18px', textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#b71c1c' }}>{v}</div>
                <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>{l}</div>
              </div>
            ))}
          </div>

          <h3>Clinical Categories</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20 }}>
            {Object.entries(overview.clinical_categories || {}).map(([cat, genes]) => (
              <div key={cat} style={{ background: '#f8f9fa', borderRadius: 8, padding: 14 }}>
                <div style={{ fontWeight: 600, textTransform: 'capitalize', marginBottom: 6 }}>{cat.replace(/_/g,' ')}</div>
                <div>{genes.map(g => <span key={g} style={{ background: GENE_COLORS[g] || '#555', color:'#fff', borderRadius:4, padding:'2px 8px', fontSize:12, marginRight:4 }}>{g}</span>)}</div>
              </div>
            ))}
          </div>

          <h3>Cohort Overview (per gene, n=40 each)</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f0f0f0' }}>
                {['Gene', 'Syndrome', 'Inheritance', 'Mean Age (yr)', '% Female'].map(h => (
                  <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(overview.cohort_stats || []).map(r => (
                <tr key={r.gene} style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '8px 12px' }}>
                    <span style={{ background: GENE_COLORS[r.gene] || '#555', color:'#fff', borderRadius:4, padding:'2px 8px', fontSize:12, fontWeight:600 }}>{r.gene}</span>
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 12 }}>{r.syndrome}</td>
                  <td style={{ padding: '8px 12px', fontSize: 12 }}>{r.inheritance}</td>
                  <td style={{ padding: '8px 12px' }}>{r.mean_age}</td>
                  <td style={{ padding: '8px 12px' }}>{r.pct_female}%</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h3 style={{ marginTop: 20 }}>Key Clinical Rules</h3>
          <ul style={{ fontSize: 13 }}>
            {(overview.key_clinical_rules || []).map((r, i) => (
              <li key={i} style={{ marginBottom: 6 }}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Gene Table tab */}
      {activeTab === 1 && breakdown && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f0f0f0' }}>
                {['Gene', 'Locus', 'Size', 'Inheritance', 'Syndrome', 'N', 'Mean Age', 'Bilateral/Met.', '2nd Primary'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #ddd', fontSize: 11 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown.gene_data || {}).map(([gene, d]) => (
                <tr key={gene} style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '7px 10px' }}>
                    <span style={{ background: GENE_COLORS[gene] || '#555', color:'#fff', borderRadius:4, padding:'2px 7px', fontSize:11, fontWeight:600 }}>{gene}</span>
                  </td>
                  <td style={{ padding: '7px 10px' }}>{GENE_INFO[gene]?.locus}</td>
                  <td style={{ padding: '7px 10px', fontSize: 11 }}>{GENE_INFO[gene]?.size}</td>
                  <td style={{ padding: '7px 10px', fontSize: 11 }}>{GENE_INFO[gene]?.inh}</td>
                  <td style={{ padding: '7px 10px', fontSize: 11 }}>{d.syndrome}</td>
                  <td style={{ padding: '7px 10px' }}>{d.n}</td>
                  <td style={{ padding: '7px 10px' }}>{d.mean_age}yr</td>
                  <td style={{ padding: '7px 10px' }}>{gene === 'RB1' ? `${d.bilateral_n}bil` : `${d.metastatic_n}met`}/40</td>
                  <td style={{ padding: '7px 10px' }}>{d.second_primary_n}/40</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ marginTop: 20 }}>
            <h3>Summary</h3>
            {breakdown.summary && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
                {[
                  ['Retinoblastoma', breakdown.summary.retinoblastoma_patients],
                  ['DICER1 Syndrome', breakdown.summary.dicer1_patients],
                  ['Rhabdoid Tumors', breakdown.summary.rhabdoid_patients],
                  ['Gorlin/Hedgehog', breakdown.summary.gorlin_hedgehog_patients],
                  ['SGB/Overgrowth', breakdown.summary.overgrowth_embryonal_patients],
                  ['Familial NB', breakdown.summary.familial_nb_patients],
                ].map(([l, v]) => (
                  <div key={l} style={{ background: '#fff3e0', borderRadius: 6, padding: '12px 14px' }}>
                    <div style={{ fontWeight: 600 }}>{l}</div>
                    <div style={{ fontSize: 22, color: '#e65100' }}>{v}</div>
                    <div style={{ fontSize: 11, color: '#888' }}>patients</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Clinical Atlas tab */}
      {activeTab === 2 && breakdown && (
        <div>
          {Object.keys(GENE_COLORS).map(gene => (
            <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={GENE_INFO[gene]} breakdown={breakdown} />
          ))}
        </div>
      )}

      {/* Definitions tab */}
      {activeTab === 3 && defs && (
        <div>
          <h3>Critical Clinical Rules</h3>
          {Object.entries(defs.critical_clinical_rules || {}).map(([rule, desc]) => (
            <div key={rule} style={{ background: '#fff3e0', border: '1px solid #ffe0b2', borderRadius: 6, padding: '10px 14px', marginBottom: 10 }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#e65100' }}>{rule.replace(/-/g, ' ')}</div>
              <div style={{ fontSize: 13, marginTop: 4 }}>{desc}</div>
            </div>
          ))}

          <h3 style={{ marginTop: 24 }}>Definitions</h3>
          {(defs.definitions || []).map((d, i) => (
            <div key={i} style={{ borderBottom: '1px solid #eee', padding: '10px 0' }}>
              <div style={{ fontWeight: 600, fontSize: 13 }}>{d.term}</div>
              <div style={{ fontSize: 13, color: '#444', marginTop: 4 }}>{d.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
