'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-nhej-ssbr-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'LIG4':   '#b71c1c',  // deep red — LIG4 Syndrome RS-SCID + pancytopenia + microcephaly + lymphoma + TBI-FATAL
  'DCLRE1C':'#e65100',  // deep orange — Artemis RS-SCID + Navajo founder + Omenn + cyclophosphamide-CI
  'PRKDC':  '#1b5e20',  // dark green — DNA-PKcs SCID ultra-rare + scaffold kinase + <30 cases worldwide
  'XRCC4':  '#0d47a1',  // deep blue — primordial microcephalic dwarfism + NO immunodeficiency (DDx LIG4)
  'NHEJ1':  '#4a148c',  // deep purple — XLF SCID milder + variable B cells + filament scaffold
  'PNKP':   '#006064',  // deep teal — MCSZ seizures+microcephaly / AOA4 ataxia+OMA — SAME GENE dual phenotype
  'APTX':   '#37474f',  // dark slate — AOA1 ataxia+OMA+hypoalbuminaemia + NORMAL AFP + Portuguese/Japanese
  'TDP1':   '#bf360c',  // burnt sienna — SCAN1 ataxia+neuropathy + camptothecin-ABSOLUTE-CI + NO OMA
};

const GENE_INFO = {
  'LIG4':   { full: 'LIG4 / DNA Ligase IV / 911aa', locus: '13q33.3', size: '911 aa / 102 kDa (ATP-dependent ligase; BRCT binds XRCC4; final NHEJ ligation)', inh: 'AR' },
  'DCLRE1C':{ full: 'DCLRE1C / Artemis / 692aa', locus: '10p13', size: '692 aa / 78 kDa (metallo-β-lactamase; hairpin-opening endonuclease; DNA-PKcs-activated)', inh: 'AR' },
  'PRKDC':  { full: 'PRKDC / DNA-PKcs / 4128aa', locus: '8q11.21', size: '4128 aa / 470 kDa (PIKK family Ser/Thr kinase; NHEJ scaffold + Artemis activator)', inh: 'AR' },
  'XRCC4':  { full: 'XRCC4 / X-ray Repair CC 4 / 336aa', locus: '5q14.2', size: '336 aa / 38 kDa (homodimer; LIG4 scaffold; XLF-XRCC4 filament; no catalytic activity)', inh: 'AR' },
  'NHEJ1':  { full: 'NHEJ1 / XLF / Cernunnos / 299aa', locus: '2q35', size: '299 aa / 33 kDa (XRCC4-paralogue; filament bridge; stimulates LIG4 incompatible-end ligation)', inh: 'AR' },
  'PNKP':   { full: 'PNKP / PNK3P / 521aa', locus: '19q13.33', size: '521 aa / 57 kDa (bifunctional: 5\'-kinase + 3\'-phosphatase; FHA domain; SSBR/NHEJ end-processing)', inh: 'AR' },
  'APTX':   { full: 'APTX / Aprataxin / 342aa', locus: '9p21.1', size: '342 aa / 38 kDa (FHA + HIT-Zn-finger; removes 5\'-adenylate abortive ligation dead-end)', inh: 'AR' },
  'TDP1':   { full: 'TDP1 / Tyrosyl-DNA Phosphodiesterase 1 / 608aa', locus: '14q31.3', size: '608 aa / 68 kDa (PLD superfamily; HxK motifs; hydrolyses 3\'-phosphotyrosyl TOP1cc)', inh: 'AR' },
};

function GeneChip({ gene, active, onClick }) {
  return (
    <button
      onClick={() => onClick(gene)}
      style={{
        background: active ? GENE_COLORS[gene] : '#263238',
        color: '#fff',
        border: `2px solid ${GENE_COLORS[gene]}`,
        borderRadius: 8,
        padding: '6px 14px',
        margin: 4,
        cursor: 'pointer',
        fontWeight: active ? 700 : 400,
        fontSize: 13,
        transition: 'all 0.15s',
      }}
    >
      {gene}
    </button>
  );
}

function StatBadge({ label, value, color }) {
  return (
    <div style={{ background: '#1e2a31', border: `1px solid ${color || '#37474f'}`, borderRadius: 8, padding: '10px 14px', minWidth: 100, textAlign: 'center', margin: 4 }}>
      <div style={{ color: color || '#90a4ae', fontSize: 10, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#fff', fontSize: 20, fontWeight: 700 }}>{value}</div>
    </div>
  );
}

function OverviewPanel({ data }) {
  if (!data) return null;
  const { summary = [], pathway_categories = [], critical_distinctions = [] } = data;
  return (
    <div>
      <h2 style={{ color: '#ef5350', marginBottom: 8 }}>Hereditary NHEJ &amp; SSBR Atlas — 8-Gene Reference</h2>
      <p style={{ color: '#90a4ae', marginBottom: 16 }}>
        LIG4 · DCLRE1C · PRKDC · XRCC4 · NHEJ1 · PNKP · APTX · TDP1 &mdash; {data.total_patients} patients (8 × 40), seeds {data.seeds}
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 24 }}>
        {summary.map(g => (
          <div key={g.gene} style={{ background: '#1e2a31', border: `2px solid ${GENE_COLORS[g.gene]}`, borderRadius: 10, padding: 12, minWidth: 170 }}>
            <div style={{ color: GENE_COLORS[g.gene], fontWeight: 700, fontSize: 15 }}>{g.gene}</div>
            <div style={{ color: '#90a4ae', fontSize: 10 }}>{g.locus} · {GENE_INFO[g.gene]?.inh}</div>
            <div style={{ color: '#cfd8dc', fontSize: 10, marginTop: 4 }}>{g.protein_size.split('(')[0].trim()}</div>
            <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3, fontSize: 10 }}>
              {g.pct_scid > 0 && <span style={{ color: '#ef9a9a' }}>SCID {g.pct_scid}%</span>}
              {g.pct_immunodeficiency > 10 && <span style={{ color: '#f48fb1' }}>ImmunoD {g.pct_immunodeficiency}%</span>}
              {g.pct_radiosensitivity > 20 && <span style={{ color: '#ffcc80' }}>RadSens {g.pct_radiosensitivity}%</span>}
              {g.pct_microcephaly > 10 && <span style={{ color: '#80cbc4' }}>Micro {g.pct_microcephaly}%</span>}
              {g.pct_ataxia > 10 && <span style={{ color: '#a5d6a7' }}>Ataxia {g.pct_ataxia}%</span>}
              {g.pct_seizures > 10 && <span style={{ color: '#ce93d8' }}>Seiz {g.pct_seizures}%</span>}
              {g.pct_oculomotor_apraxia > 10 && <span style={{ color: '#80deea' }}>OMA {g.pct_oculomotor_apraxia}%</span>}
              {g.pct_hsct > 10 && <span style={{ color: '#ffe082' }}>HSCT {g.pct_hsct}%</span>}
            </div>
          </div>
        ))}
      </div>

      <h3 style={{ color: '#90a4ae', marginBottom: 8 }}>Pathway Categories</h3>
      {pathway_categories.map((cat, i) => (
        <div key={i} style={{ background: '#1e2a31', border: '1px solid #37474f', borderRadius: 8, padding: 12, marginBottom: 10 }}>
          <div style={{ color: '#ef5350', fontWeight: 600, marginBottom: 4 }}>{cat.pathway}</div>
          <div style={{ display: 'flex', gap: 6, marginBottom: 6, flexWrap: 'wrap' }}>
            {cat.genes.map(g => <span key={g} style={{ background: GENE_COLORS[g], color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600 }}>{g}</span>)}
          </div>
          <div style={{ color: '#90a4ae', fontSize: 12 }}>{cat.note}</div>
        </div>
      ))}

      <h3 style={{ color: '#90a4ae', margin: '20px 0 8px' }}>Critical Clinical Distinctions</h3>
      {critical_distinctions.map((d, i) => (
        <div key={i} style={{ background: '#1e2a31', border: '1px solid #b71c1c', borderLeft: '4px solid #ef5350', borderRadius: 6, padding: '8px 12px', marginBottom: 6, color: '#cfd8dc', fontSize: 12 }}>
          {d}
        </div>
      ))}
    </div>
  );
}

function GeneTablePanel({ data }) {
  if (!data?.genes) return null;
  const cols = ['Gene', 'Locus', 'Size', 'Inh.', 'SCID%', 'ImmunoD%', 'RadSens%', 'Micro%', 'Pancyto%', 'Lymphoma%', 'Ataxia%', 'Seiz%', 'Neuropathy%', 'OMA%', 'Hypoalb%', 'Camptotech%', 'HSCT%'];
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 11 }}>
        <thead>
          <tr style={{ background: '#1e2a31' }}>
            {cols.map(c => <th key={c} style={{ padding: '8px 8px', color: '#90a4ae', textAlign: 'left', borderBottom: '2px solid #37474f', whiteSpace: 'nowrap' }}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {data.genes.map(g => (
            <tr key={g.gene} style={{ borderBottom: '1px solid #263238' }}>
              <td style={{ padding: '7px 8px', color: GENE_COLORS[g.gene], fontWeight: 700 }}>{g.gene}</td>
              <td style={{ padding: '7px 8px', color: '#cfd8dc' }}>{g.locus}</td>
              <td style={{ padding: '7px 8px', color: '#90a4ae', whiteSpace: 'nowrap' }}>{g.protein_size.split('(')[0].trim()}</td>
              <td style={{ padding: '7px 8px', color: '#cfd8dc' }}>{GENE_INFO[g.gene]?.inh}</td>
              <td style={{ padding: '7px 8px', color: g.pct_scid > 50 ? '#ef9a9a' : '#cfd8dc' }}>{g.pct_scid}%</td>
              <td style={{ padding: '7px 8px', color: g.pct_immunodeficiency > 50 ? '#f48fb1' : '#cfd8dc' }}>{g.pct_immunodeficiency}%</td>
              <td style={{ padding: '7px 8px', color: g.pct_radiosensitivity > 60 ? '#ffcc80' : '#cfd8dc' }}>{g.pct_radiosensitivity}%</td>
              <td style={{ padding: '7px 8px', color: g.pct_microcephaly > 50 ? '#80cbc4' : '#cfd8dc' }}>{g.pct_microcephaly}%</td>
              <td style={{ padding: '7px 8px', color: g.pct_pancytopenia > 30 ? '#ef9a9a' : '#cfd8dc' }}>{g.pct_pancytopenia}%</td>
              <td style={{ padding: '7px 8px', color: g.pct_lymphoma > 15 ? '#f48fb1' : '#cfd8dc' }}>{g.pct_lymphoma}%</td>
              <td style={{ padding: '7px 8px', color: g.pct_ataxia > 50 ? '#a5d6a7' : '#cfd8dc' }}>{g.pct_ataxia}%</td>
              <td style={{ padding: '7px 8px', color: g.pct_seizures > 40 ? '#ce93d8' : '#cfd8dc' }}>{g.pct_seizures}%</td>
              <td style={{ padding: '7px 8px', color: g.pct_peripheral_neuropathy > 50 ? '#80deea' : '#cfd8dc' }}>{g.pct_peripheral_neuropathy}%</td>
              <td style={{ padding: '7px 8px', color: g.pct_oculomotor_apraxia > 50 ? '#80deea' : '#cfd8dc' }}>{g.pct_oculomotor_apraxia}%</td>
              <td style={{ padding: '7px 8px', color: g.pct_hypoalbuminemia > 50 ? '#ffe082' : '#cfd8dc' }}>{g.pct_hypoalbuminemia}%</td>
              <td style={{ padding: '7px 8px', color: g.pct_camptothecin_sensitivity > 50 ? '#ff8a65' : '#cfd8dc' }}>{g.pct_camptothecin_sensitivity}%</td>
              <td style={{ padding: '7px 8px', color: g.pct_hsct > 50 ? '#ffe082' : '#cfd8dc' }}>{g.pct_hsct}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasPanel({ data }) {
  const [activeGene, setActiveGene] = useState('LIG4');
  if (!data?.genes) return null;
  const geneData = data.genes.find(g => g.gene === activeGene);
  if (!geneData) return null;
  const sections = [
    { label: 'Inheritance & Penetrance', key: 'inheritance', color: '#ef5350' },
    { label: 'Disease Category', key: 'disease_category', color: '#9c27b0' },
    { label: 'Disease Pathway', key: 'disease_pathway', color: '#1976d2' },
    { label: 'Pathognomonic / Diagnosis', key: 'pathognomonic', color: '#00897b' },
    { label: 'Treatment', key: 'treatment', color: '#f57c00' },
  ];
  return (
    <div>
      <div style={{ marginBottom: 16, display: 'flex', flexWrap: 'wrap' }}>
        {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} active={activeGene === g} onClick={setActiveGene} />)}
      </div>
      <div style={{ background: `${GENE_COLORS[activeGene]}22`, border: `2px solid ${GENE_COLORS[activeGene]}`, borderRadius: 10, padding: 16, marginBottom: 16 }}>
        <div style={{ color: GENE_COLORS[activeGene], fontSize: 20, fontWeight: 700 }}>{geneData.gene}</div>
        <div style={{ color: '#90a4ae', fontSize: 12 }}>{geneData.locus} · {geneData.protein_size}</div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 12 }}>
          <StatBadge label="SCID" value={`${geneData.pct_scid}%`} color="#ef9a9a" />
          <StatBadge label="ImmunoD" value={`${geneData.pct_immunodeficiency}%`} color="#f48fb1" />
          <StatBadge label="RadSens" value={`${geneData.pct_radiosensitivity}%`} color="#ffcc80" />
          <StatBadge label="Micro" value={`${geneData.pct_microcephaly}%`} color="#80cbc4" />
          <StatBadge label="Pancyto" value={`${geneData.pct_pancytopenia}%`} color="#ef9a9a" />
          <StatBadge label="Lymphoma" value={`${geneData.pct_lymphoma}%`} color="#f48fb1" />
          <StatBadge label="Ataxia" value={`${geneData.pct_ataxia}%`} color="#a5d6a7" />
          <StatBadge label="Seizures" value={`${geneData.pct_seizures}%`} color="#ce93d8" />
          <StatBadge label="Neuropathy" value={`${geneData.pct_peripheral_neuropathy}%`} color="#80deea" />
          <StatBadge label="OMA" value={`${geneData.pct_oculomotor_apraxia}%`} color="#80deea" />
          <StatBadge label="Hypoalb" value={`${geneData.pct_hypoalbuminemia}%`} color="#ffe082" />
          <StatBadge label="CamptoCI" value={`${geneData.pct_camptothecin_sensitivity}%`} color="#ff8a65" />
          <StatBadge label="HSCT" value={`${geneData.pct_hsct}%`} color="#ffe082" />
          <StatBadge label="n" value={geneData.n_patients} color="#90a4ae" />
        </div>
      </div>
      {sections.map(s => (
        <div key={s.key} style={{ background: '#1e2a31', border: `1px solid ${s.color}44`, borderLeft: `4px solid ${s.color}`, borderRadius: 8, padding: 14, marginBottom: 10 }}>
          <div style={{ color: s.color, fontWeight: 600, marginBottom: 6 }}>{s.label}</div>
          <pre style={{ color: '#cfd8dc', fontSize: 11, whiteSpace: 'pre-wrap', margin: 0, fontFamily: 'inherit', lineHeight: 1.6 }}>{geneData[s.key]}</pre>
        </div>
      ))}
      <div style={{ background: '#1e2a31', border: '1px solid #37474f', borderRadius: 8, padding: 14, marginTop: 12 }}>
        <div style={{ color: '#90a4ae', fontWeight: 600, marginBottom: 8 }}>Patient Sample ({geneData.patients?.length} patients)</div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 10 }}>
            <thead>
              <tr style={{ background: '#263238' }}>
                {['ID','Age','Sex','SCID','ImmunoD','RadSens','Micro','Growth','Pancyto','Lymphoma','Ataxia','Seiz','Neuro','OMA','Hypoalb','CamptoCI','HSCT'].map(h => (
                  <th key={h} style={{ padding: '4px 6px', color: '#90a4ae', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(geneData.patients || []).slice(0, 20).map(p => (
                <tr key={p.id} style={{ borderBottom: '1px solid #1e2a31' }}>
                  <td style={{ padding: '3px 6px', color: GENE_COLORS[geneData.gene] }}>{p.id}</td>
                  <td style={{ padding: '3px 6px', color: '#cfd8dc' }}>{p.age}</td>
                  <td style={{ padding: '3px 6px', color: '#cfd8dc' }}>{p.sex}</td>
                  <td style={{ padding: '3px 6px', color: p.scid ? '#ef9a9a' : '#546e7a' }}>{p.scid ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 6px', color: p.immunodeficiency ? '#f48fb1' : '#546e7a' }}>{p.immunodeficiency ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 6px', color: p.radiosensitivity ? '#ffcc80' : '#546e7a' }}>{p.radiosensitivity ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 6px', color: p.microcephaly ? '#80cbc4' : '#546e7a' }}>{p.microcephaly ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 6px', color: p.growth_retardation ? '#80cbc4' : '#546e7a' }}>{p.growth_retardation ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 6px', color: p.pancytopenia ? '#ef9a9a' : '#546e7a' }}>{p.pancytopenia ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 6px', color: p.lymphoma ? '#f48fb1' : '#546e7a' }}>{p.lymphoma ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 6px', color: p.ataxia ? '#a5d6a7' : '#546e7a' }}>{p.ataxia ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 6px', color: p.seizures ? '#ce93d8' : '#546e7a' }}>{p.seizures ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 6px', color: p.peripheral_neuropathy ? '#80deea' : '#546e7a' }}>{p.peripheral_neuropathy ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 6px', color: p.oculomotor_apraxia ? '#80deea' : '#546e7a' }}>{p.oculomotor_apraxia ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 6px', color: p.hypoalbuminemia ? '#ffe082' : '#546e7a' }}>{p.hypoalbuminemia ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 6px', color: p.camptothecin_sensitivity ? '#ff8a65' : '#546e7a' }}>{p.camptothecin_sensitivity ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 6px', color: p.hsct_performed ? '#ffe082' : '#546e7a' }}>{p.hsct_performed ? 'Y' : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data?.glossary) return null;
  return (
    <div>
      <h3 style={{ color: '#ef5350', marginBottom: 12 }}>Glossary — NHEJ &amp; SSBR Atlas</h3>
      {Object.entries(data.glossary).map(([term, def]) => (
        <div key={term} style={{ background: '#1e2a31', border: '1px solid #37474f', borderRadius: 8, padding: 12, marginBottom: 8 }}>
          <div style={{ color: '#ef5350', fontWeight: 600, marginBottom: 4 }}>{term}</div>
          <div style={{ color: '#cfd8dc', fontSize: 12, lineHeight: 1.6 }}>{def}</div>
        </div>
      ))}
      {data.standards && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ color: '#90a4ae', marginBottom: 8 }}>Standards &amp; Key References</h3>
          {data.standards.map((s, i) => (
            <div key={i} style={{ color: '#78909c', fontSize: 12, padding: '4px 0', borderBottom: '1px solid #263238' }}>{s}</div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function NHEJSSBRAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const base = `${API}/api/${SLUG}`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#fff', padding: 24, fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        <div style={{ marginBottom: 8, fontSize: 12, color: '#546e7a' }}>
          Hereditary Cancer Genetics &rsaquo; DNA Repair Disorders &rsaquo; <span style={{ color: '#ef5350' }}>NHEJ &amp; SSBR Atlas</span>
        </div>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#ef5350', marginBottom: 4 }}>
          &#x1f9ec; Hereditary NHEJ &amp; SSBR Atlas — 8-Gene End-Joining &amp; Break-Repair Reference
        </h1>
        <p style={{ color: '#78909c', marginBottom: 20, fontSize: 13 }}>
          LIG4 · DCLRE1C · PRKDC · XRCC4 · NHEJ1 · PNKP · APTX · TDP1 — Complete NHEJ/SSBR Clinical Reference
          · 320 patients (8 × 40) · Seeds 2750–2757
        </p>

        <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '2px solid #1e2a31', paddingBottom: 0 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              background: 'none', border: 'none', color: tab === t ? '#ef5350' : '#546e7a',
              borderBottom: tab === t ? '2px solid #ef5350' : '2px solid transparent',
              padding: '10px 18px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400,
              fontSize: 14, marginBottom: -2,
            }}>{t}</button>
          ))}
        </div>

        {loading && <div style={{ color: '#90a4ae', padding: 40, textAlign: 'center' }}>Loading NHEJ &amp; SSBR Atlas…</div>}
        {error && <div style={{ color: '#f44336', padding: 20 }}>Error: {error}</div>}
        {!loading && !error && (
          <>
            {tab === 'Overview' && <OverviewPanel data={overview} />}
            {tab === 'Gene Table' && <GeneTablePanel data={breakdown} />}
            {tab === 'Clinical Atlas' && <ClinicalAtlasPanel data={breakdown} />}
            {tab === 'Definitions' && <DefinitionsPanel data={definitions} />}
          </>
        )}
      </div>
    </div>
  );
}
