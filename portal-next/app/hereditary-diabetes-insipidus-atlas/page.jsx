'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-diabetes-insipidus-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'AVP':     '#1565c0',  // deep blue     — FNDI, central DI, desmopressin effective
  'AVPR2':   '#b71c1c',  // deep red      — NDI1, XLR, desmopressin completely unresponsive
  'AQP2':    '#00695c',  // dark teal     — NDI2, AR/AD, water channel
  'WFS1':    '#4a148c',  // deep purple   — Wolfram DIDMOAD, non-immune DM
  'CISD2':   '#e65100',  // deep orange   — Wolfram 2, NO DI, bleeding tendency
  'PCSK1':   '#2e7d32',  // dark green    — PC1/3 deficiency, malabsorption first
  'KCNJ1':   '#f57f17',  // amber         — Bartter 2, neonatal hyperkalemia
  'SLC12A1': '#006064',  // dark cyan     — Bartter 1, furosemide-like, nephrocalcinosis
};

const GENE_INFO = {
  'AVP': {
    full: 'AVP / Arginine Vasopressin Prepropeptide / 164aa',
    locus: '20p13',
    size: '164 aa (neuropeptide precursor encodes AVP + neurophysin II + copeptin; AD LOF: misfolded pro-AVP → ER stress → progressive magnocellular neuron death → FNDI onset age 2-10; AR biallelic: congenital severe non-progressive; MRI: posterior pituitary bright spot ABSENT pathognomonic; desmopressin EFFECTIVE; distinguishes from NDI; AD/AR LOF)',
    inh: 'AD/AR LOF',
    imprinting: null,
  },
  'AVPR2': {
    full: 'AVPR2 / V2 Vasopressin Receptor / 371aa',
    locus: 'Xq28',
    size: '371 aa (Gs-coupled GPCR in collecting duct; AVP→V2R→cAMP→PKA→AQP2 insertion; XLR: >200 mutations; males severely affected neonatal; females variable X-inactivation; desmopressin COMPLETELY UNRESPONSIVE (V2R absent/dysfunctional); treat: thiazide + amiloride + low-solute diet; GOF → NSIAD opposite phenotype; XLR)',
    inh: 'XLR',
    imprinting: null,
  },
  'AQP2': {
    full: 'AQP2 / Aquaporin-2 Water Channel / 271aa',
    locus: '12q13.12',
    size: '271 aa / 26 kDa (water channel apical collecting duct; PKA-Ser256 phosphorylation → apical insertion; AR biallelic: classic NDI2; AD C-terminal dominant-negative; desmopressin UNRESPONSIVE (V2R intact; AQP2 absent/non-trafficked); urine AQP2 absent = diagnostic; emerging: sildenafil + statins improve trafficking; AR/AD)',
    inh: 'AR/AD',
    imprinting: null,
  },
  'WFS1': {
    full: 'WFS1 / Wolframin / 890aa',
    locus: '4p16.1',
    size: '890 aa / 100 kDa (ER transmembrane 9-TM helices; ER Ca2+ homeostasis + UPR modulation; AR LOF → ER stress in beta cells + magnocellular neurons + retinal ganglion + cochlear cells; DIDMOAD sequence: optic atrophy age 6 → DM (NON-IMMUNE no autoantibodies) age 6 → DI (central) age 14 → deafness age 16; desmopressin EFFECTIVE; AR)',
    inh: 'AR LOF',
    imprinting: null,
  },
  'CISD2': {
    full: 'CISD2 / CDGSH Iron-Sulfur Domain 2 (ERIS/NAF-1) / 135aa',
    locus: '4q24',
    size: '135 aa ([2Fe-2S] domain; mitochondrial outer membrane + MAM; ER/mitochondrial Ca2+ homeostasis; WFS2: DM + optic atrophy (earlier onset age 3-4) + peripheral neuropathy + bleeding tendency (peptic ulcers + platelet hyperaggregability); NO DI in ~85%; Israeli Arab founder p.Trp45Ser; AR)',
    inh: 'AR LOF',
    imprinting: null,
  },
  'PCSK1': {
    full: 'PCSK1 / Proprotein Convertase Subtilisin/Kexin Type 1 (PC1/3) / 753aa',
    locus: '5q15',
    size: '753 aa / 83 kDa (serine protease; processes: pro-AVP→AVP, pro-POMC→ACTH, pro-insulin→insulin, pro-GLP-1, pro-GnRH, pro-TRH; neonatal malabsorption PATHOGNOMONIC AND FIRST; morbid obesity early-onset; central DI 50-60% desmopressin effective; ACTH deficiency; hyperproinsulinaemia biomarker; AR)',
    inh: 'AR LOF',
    imprinting: null,
  },
  'KCNJ1': {
    full: 'KCNJ1 / ROMK Kir1.1 Renal Outer Medullary K+ Channel / 391aa',
    locus: '11q24.3',
    size: '391 aa / 45 kDa (inwardly-rectifying K+ channel apical TAL + collecting duct; TAL: recycles K+ for NKCC2; CD: K+ secretion; Bartter type 2: polyhydramnios + neonatal PARADOXICAL TRANSIENT HYPERKALEMIA (ROMK absent in CD → K+ not secreted → initial hyperkalemia, resolves weeks); hypokalemia + alkalosis + hypercalciuria; PGE2 elevated; indomethacin; AR)',
    inh: 'AR LOF',
    imprinting: null,
  },
  'SLC12A1': {
    full: 'SLC12A1 / NKCC2 Na-K-2Cl Cotransporter Type 2 / 1099aa',
    locus: '15q21.1',
    size: '1099 aa / 120 kDa (apical TAL cotransporter; furosemide binding site; 25% total renal NaCl; LOF = permanent furosemide effect; Bartter type 1 MOST SEVERE: severe polyhydramnios early; premature <32 weeks; NO neonatal hyperkalemia; nephrocalcinosis 80%; hypokalemia + alkalosis + hypercalciuria; indomethacin; AR)',
    inh: 'AR LOF',
    imprinting: null,
  },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22',
      color,
      border: `1px solid ${color}55`,
      borderRadius: 4,
      padding: '2px 7px',
      fontSize: 11,
      fontWeight: 600,
      marginRight: 4,
    }}>{text}</span>
  );
}

export default function HreditaryDiabetesInsipidusAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedGene, setExpandedGene] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const endpoints = {
      'Overview': 'overview',
      'Gene Table': 'breakdown',
      'Clinical Atlas': 'breakdown',
      'Definitions': 'definitions',
    };
    const ep = endpoints[tab] || 'overview';
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(data => {
        if (tab === 'Overview') setOverview(data);
        else if (tab === 'Gene Table' || tab === 'Clinical Atlas') setBreakdown(data);
        else setDefinitions(data);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  const cardStyle = {
    background: '#fff',
    border: '1px solid #e0e0e0',
    borderRadius: 8,
    padding: 16,
    marginBottom: 14,
    boxShadow: '0 1px 3px rgba(0,0,0,0.07)',
  };

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto', padding: '20px 16px' }}>
      <div style={{ marginBottom: 18 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#1a1a2e', margin: '0 0 4px' }}>
          🧬 Hereditary Diabetes Insipidus Atlas
        </h1>
        <p style={{ color: '#555', fontSize: 13, margin: 0 }}>
          Complete 8-Gene DI Reference · AVP-AVPR2-AQP2-WFS1-CISD2-PCSK1-KCNJ1-SLC12A1 · 320 Patients · Seeds 2990-2997
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e0e0e0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 16px', border: 'none', cursor: 'pointer', fontWeight: tab === t ? 700 : 400,
            background: tab === t ? '#1565c0' : 'transparent',
            color: tab === t ? '#fff' : '#555',
            borderRadius: '6px 6px 0 0', fontSize: 13,
            borderBottom: tab === t ? '2px solid #1565c0' : 'none',
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#888', padding: 20 }}>Loading…</div>}
      {error && <div style={{ color: '#c00', padding: 20 }}>Error: {error}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 12, marginBottom: 20 }}>
            {[
              { label: 'Genes', value: overview.total_genes },
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Seed Range', value: overview.seed_range },
              { label: 'Central DI Genes', value: 'AVP, WFS1, PCSK1' },
              { label: 'NDI Genes', value: 'AVPR2 (XLR), AQP2' },
              { label: 'Bartter Genes', value: 'KCNJ1, SLC12A1' },
            ].map(k => (
              <div key={k.label} style={{ ...cardStyle, textAlign: 'center', padding: 14 }}>
                <div style={{ fontSize: 18, fontWeight: 800, color: '#1565c0' }}>{k.value}</div>
                <div style={{ fontSize: 11, color: '#777', marginTop: 2 }}>{k.label}</div>
              </div>
            ))}
          </div>

          {/* Gene chips */}
          <div style={{ ...cardStyle }}>
            <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>8 DI-Spectrum Genes — Loci &amp; Inheritance</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {(overview.genes || []).map(g => (
                <div key={g} style={{
                  background: (GENE_COLORS[g] || '#888') + '15',
                  border: `1.5px solid ${(GENE_COLORS[g] || '#888')}55`,
                  borderRadius: 8, padding: '8px 12px', minWidth: 140,
                }}>
                  <div style={{ fontWeight: 800, color: GENE_COLORS[g] || '#333', fontSize: 15 }}>{g}</div>
                  <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>{overview.gene_loci[g]}</div>
                  <div style={{ fontSize: 10, color: '#777' }}>{(overview.inheritance_modes[g] || '').slice(0, 60)}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Key clinical rules */}
          <div style={cardStyle}>
            <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 10, color: '#b71c1c' }}>
              🔑 Key Clinical Rules
            </h3>
            {(overview.key_clinical_rules || []).map((rule, i) => {
              const [head, ...rest] = rule.split(':');
              return (
                <div key={i} style={{ borderLeft: '3px solid #1565c0', paddingLeft: 10, marginBottom: 8, fontSize: 12 }}>
                  <strong style={{ color: '#1565c0' }}>{head}:</strong>{' '}
                  <span style={{ color: '#444' }}>{rest.join(':')}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1565c0', color: '#fff' }}>
                  {['Gene', 'Locus', 'Inheritance', 'Urine Osmol Nadir', 'Desmopressin Resp%', 'Neonatal Onset%', 'Mean Age Dx', 'n'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.genes.map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#f8f9fa' : '#fff', cursor: 'pointer' }}
                    onClick={() => setExpandedGene(expandedGene === g.gene ? null : g.gene)}>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>
                      {g.gene}
                    </td>
                    <td style={{ padding: '7px 10px', color: '#555' }}>{g.locus}</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{GENE_INFO[g.gene]?.inh || ''}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'right' }}>
                      {g.mean_urine_osmol_nadir != null ? g.mean_urine_osmol_nadir + ' mOsm' : 'N/A'}
                    </td>
                    <td style={{
                      padding: '7px 10px', textAlign: 'right',
                      color: g.desmopressin_responsive_pct >= 50 ? '#2e7d32' : '#b71c1c',
                      fontWeight: 700,
                    }}>
                      {g.desmopressin_responsive_pct}%
                    </td>
                    <td style={{ padding: '7px 10px', textAlign: 'right' }}>{g.neonatal_onset_pct}%</td>
                    <td style={{ padding: '7px 10px', textAlign: 'right' }}>{g.mean_age_dx}y</td>
                    <td style={{ padding: '7px 10px', textAlign: 'right', color: '#888' }}>{g.n_patients}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {expandedGene && (() => {
            const g = breakdown.genes.find(x => x.gene === expandedGene);
            if (!g) return null;
            return (
              <div style={{ ...cardStyle, marginTop: 16, borderLeft: `4px solid ${GENE_COLORS[expandedGene] || '#888'}` }}>
                <h3 style={{ color: GENE_COLORS[expandedGene] || '#333', marginTop: 0, fontSize: 15 }}>
                  {expandedGene} — {GENE_INFO[expandedGene]?.full}
                </h3>
                <div style={{ fontSize: 11, color: '#555', lineHeight: 1.6, marginBottom: 8, whiteSpace: 'pre-wrap' }}>
                  <strong>Protein/Function:</strong> {g.protein_size}
                </div>
                <div style={{ fontSize: 11, color: '#444', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                  <strong>Disease Category:</strong> {g.disease_category}
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {breakdown.genes.map(g => (
            <div key={g.gene} style={{ ...cardStyle, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#888'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                <h3 style={{ color: GENE_COLORS[g.gene] || '#333', margin: 0, fontSize: 15 }}>
                  {g.gene}
                  <span style={{ fontWeight: 400, fontSize: 12, color: '#555', marginLeft: 8 }}>
                    {g.locus} · {GENE_INFO[g.gene]?.inh}
                  </span>
                </h3>
                <div style={{ fontSize: 11, color: '#888' }}>n={g.n_patients} · mean age {g.mean_age_dx}y</div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(130px,1fr))', gap: 8, marginBottom: 10 }}>
                {[
                  { label: 'Urine Osmol Nadir', val: g.mean_urine_osmol_nadir != null ? g.mean_urine_osmol_nadir + ' mOsm' : 'N/A' },
                  { label: 'Desmopressin Resp%', val: g.desmopressin_responsive_pct + '%', highlight: g.desmopressin_responsive_pct >= 50 ? 'green' : 'red' },
                  { label: 'Neonatal Onset%', val: g.neonatal_onset_pct + '%' },
                  { label: 'Mean Age Dx', val: g.mean_age_dx + 'y' },
                ].map(m => (
                  <div key={m.label} style={{
                    background: m.highlight === 'green' ? '#e8f5e9' : m.highlight === 'red' ? '#ffebee' : '#f5f5f5',
                    borderRadius: 6, padding: '6px 10px', textAlign: 'center',
                    border: m.highlight === 'green' ? '1px solid #a5d6a7' : m.highlight === 'red' ? '1px solid #ef9a9a' : '1px solid #e0e0e0',
                  }}>
                    <div style={{ fontWeight: 700, color: m.highlight === 'green' ? '#2e7d32' : m.highlight === 'red' ? '#b71c1c' : '#333', fontSize: 14 }}>{m.val}</div>
                    <div style={{ fontSize: 10, color: '#777' }}>{m.label}</div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 11, color: '#555', lineHeight: 1.6, whiteSpace: 'pre-wrap', borderTop: '1px solid #f0f0f0', paddingTop: 8 }}>
                {(g.inheritance || '').slice(0, 900)}
              </div>
              <div style={{ marginTop: 8 }}>
                <strong style={{ fontSize: 11 }}>Common mutations: </strong>
                {Object.keys(g.mutation_breakdown || {}).slice(0, 4).map(m => (
                  <Badge key={m} text={m} color={GENE_COLORS[g.gene] || '#888'} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          {(definitions.definitions || []).map((d, i) => (
            <div key={i} style={{ ...cardStyle }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: '#1565c0', marginTop: 0, marginBottom: 8 }}>
                {d.term}
              </h3>
              <div style={{ display: 'flex', gap: 6, marginBottom: 8, flexWrap: 'wrap' }}>
                {(d.genes || []).map(g => (
                  <span key={g} style={{
                    background: (GENE_COLORS[g] || '#888') + '22',
                    color: GENE_COLORS[g] || '#333',
                    border: `1px solid ${(GENE_COLORS[g] || '#888')}55`,
                    borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 700,
                  }}>{g}</span>
                ))}
              </div>
              <div style={{ fontSize: 12, color: '#444', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
                {d.definition}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
