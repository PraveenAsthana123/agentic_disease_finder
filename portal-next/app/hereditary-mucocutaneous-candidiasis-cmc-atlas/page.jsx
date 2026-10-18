'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-mucocutaneous-candidiasis-cmc-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'STAT1':    '#b71c1c',  // deep red            -- AD GOF; most common CMC; JAKi responsive; autoimmune endo
  'IL17RA':   '#880e4f',  // deep magenta        -- AR; CMC1; complete IL-17 signal absent; severe + bacterial
  'IL17F':    '#e65100',  // burnt orange        -- AD; CMC3; haploinsufficiency; mild; Candida-only; azoles
  'TRAF3IP2': '#f57f17',  // amber               -- AR; CMC4; ACT1 adaptor; serum IL-17 NORMAL (diagnostic trap)
  'IL17RC':   '#1565c0',  // deep blue           -- AR; CMC5; selective; IL-17E spared; ultra-rare
  'RORC':     '#1a237e',  // deep navy           -- AR; CMC7; RORγt absent; BCG CI; triple susceptibility
  'CARD9':    '#4a148c',  // deep purple         -- AR; CNS Candida PATHOGNOMONIC; invasive; GM-CSF adjunct
  'CLEC7A':   '#004d40',  // dark teal           -- AD; Dectin-1 haploinsufficiency; Y238X; mucosal only
};

const GENE_INFO = {
  'STAT1':    { full: 'STAT1-GOF-750aa / AD-GOF / CMC-Most-Common-GT50pct / Th17-Suppressed-IFN-Hypersignalling / Autoimmune-Endo-60pct / Viral-HPV-HSV / JAKi-Ruxolitinib-First-Line', locus: '2q32.2',  size: '750 aa / 87 kDa',  inh: 'AD GOF' },
  'IL17RA':   { full: 'IL17RA-866aa / AR-CMC1 / Complete-IL17A-IL17F-IL17E-Signal-Absent / Severe-CMC-Staph-Skin / No-Autoimmunity / Azole-Prophylaxis-Lifelong', locus: '22q11.1', size: '866 aa / 95 kDa',  inh: 'AR' },
  'IL17F':    { full: 'IL17F-163aa / AD-CMC3 / Haploinsufficiency-Mild / Oral-Nail-Candida-Only / No-Bacterial-No-Viral / Excellent-Azole-Response', locus: '6p12.2',  size: '163 aa / 18 kDa',  inh: 'AD' },
  'TRAF3IP2': { full: 'TRAF3IP2-ACT1-574aa / AR-CMC4 / IL17-Adaptor-Absent / Serum-IL17-NORMAL-DIAGNOSTIC-TRAP / Functional-NF-kB-Assay-Required / Nail-Oral-Skin-Candida', locus: '6q21',    size: '574 aa / 64 kDa',  inh: 'AR' },
  'IL17RC':   { full: 'IL17RC-734aa / AR-CMC5 / Selective-IL17A-IL17F-Impaired / IL17E-via-IL17RA-RB-INTACT / Pure-CMC-No-Bacterial / Ultra-Rare-LT30-Cases', locus: '3p24.3',  size: '734 aa / 82 kDa',  inh: 'AR' },
  'RORC':     { full: 'RORC-RORgt-519aa / AR-CMC7 / Th17-Master-TF-Absent / No-Th17-No-ILC3 / CMC-Mycobacteria-Staph-TRIPLE / BCG-ABSOLUTELY-CONTRAINDICATED / HSCT', locus: '1q23.3',  size: '519 aa / 59 kDa',  inh: 'AR' },
  'CARD9':    { full: 'CARD9-536aa / AR / CLR-Signalosome-Absent / CNS-Candida-Meningitis-PATHOGNOMONIC / Invasive-Fungal / Dermatophyte-Subcutaneous / GM-CSF-Adjunct / L-AmB', locus: '9q34.3',  size: '536 aa / 61 kDa',  inh: 'AR' },
  'CLEC7A':   { full: 'CLEC7A-Dectin1-247aa / AD-Haploinsufficiency / Y238X-Polymorphic-South-Asian-5pct-European-3pct / Mucosal-Candida-Only / No-Invasive-Fungal / Azole-Responsive', locus: '12p13.2', size: '247 aa / 27 kDa',  inh: 'AD' },
};

const PHENOTYPE_ROWS = [
  { gene: 'STAT1',    group: 'GOF-Dysreg',    cmc: '✓✓✓', invasive: '±', endo: '✓✓',  viral: '✓✓',  myco: '−',   bacterial: '±',  cns_c: '−',  derm: '−',   jaki: '✓✓✓',unique: 'Autoimmune endocrinopathy (hypothyroidism ~60%) + viral (HPV/HSV) TRIAD; JAKi (ruxolitinib) first-line targeted therapy' },
  { gene: 'IL17RA',   group: 'IL17-Receptor', cmc: '✓✓✓', invasive: '±', endo: '−',   viral: '−',   myco: '−',   bacterial: '✓✓', cns_c: '−',  derm: '±',   jaki: '−',   unique: 'Complete IL-17 signal absent (all IL-17A/F/AF/E blocked); severe CMC + S. aureus skin; NO autoimmunity; lifelong azole prophylaxis' },
  { gene: 'IL17F',    group: 'IL17-Receptor', cmc: '✓✓',  invasive: '−', endo: '−',   viral: '−',   myco: '−',   bacterial: '−',  cns_c: '−',  derm: '−',   jaki: '−',   unique: 'Mildest CMC gene; oral/nail only; AD haploinsufficiency; excellent azole response; no surveillance needed beyond fungals' },
  { gene: 'TRAF3IP2', group: 'IL17-Receptor', cmc: '✓✓',  invasive: '±', endo: '−',   viral: '±',   myco: '−',   bacterial: '±',  cns_c: '−',  derm: '±',   jaki: '−',   unique: 'IL-17 serum level NORMAL (diagnostic trap); functional NF-kB assay in fibroblasts/keratinocytes required for diagnosis' },
  { gene: 'IL17RC',   group: 'IL17-Receptor', cmc: '✓✓',  invasive: '−', endo: '−',   viral: '−',   myco: '−',   bacterial: '−',  cns_c: '−',  derm: '−',   jaki: '−',   unique: 'Selective: IL-17E (anti-parasite, eosinophil axis) INTACT; pure CMC without bacterial susceptibility; ultra-rare' },
  { gene: 'RORC',     group: 'Th17-Dev',      cmc: '✓✓✓', invasive: '±', endo: '±',   viral: '±',   myco: '✓✓✓',bacterial: '✓✓', cns_c: '±',  derm: '✓',   jaki: '−',   unique: 'TRIPLE susceptibility: CMC + NTM/mycobacteria + Staph; BCG ABSOLUTELY CONTRAINDICATED; ILC3 absent → IBD-like; BROADEST phenotype' },
  { gene: 'CARD9',    group: 'CLR-Innate',    cmc: '✓✓',  invasive: '✓✓✓',endo: '−',  viral: '±',   myco: '±',   bacterial: '±',  cns_c: '✓✓✓',derm: '✓✓✓',jaki: '−',   unique: 'CNS Candida (meningitis/abscess) PATHOGNOMONIC; Trichosporon/Phialophora/Cladophialophora CNS; subcutaneous dermatophytosis; GM-CSF adjunct' },
  { gene: 'CLEC7A',   group: 'CLR-Innate',    cmc: '✓✓',  invasive: '−', endo: '−',   viral: '−',   myco: '−',   bacterial: '−',  cns_c: '−',  derm: '±',   jaki: '−',   unique: 'Y238X stop polymorphism; mucosal Candida ONLY (no invasive); recurrent vulvovaginal/oral candidiasis; azole-responsive' },
];

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#333', color: '#fff', padding: '2px 7px',
      borderRadius: 4, fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneCard({ gene, color, info, cohort }) {
  const groupLabel = gene === 'STAT1' ? 'STAT1 GOF / Immune Dysregulation'
                   : ['IL17RA','IL17F','TRAF3IP2','IL17RC'].includes(gene) ? 'IL-17 Receptor / Ligand Pathway'
                   : gene === 'RORC' ? 'Th17 Development (RORγt)'
                   : 'CLR Innate Fungal Sensing';
  const groupColor = gene === 'STAT1' ? '#b71c1c'
                   : ['IL17RA','IL17F','TRAF3IP2','IL17RC'].includes(gene) ? '#1565c0'
                   : gene === 'RORC' ? '#1a237e'
                   : '#4a148c';
  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ background: color, color: '#fff', borderRadius: 6, padding: '6px 14px', fontWeight: 700, fontSize: 18 }}>{gene}</div>
        <div style={{ background: groupColor, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>{groupLabel}</div>
        <div>
          <div style={{ fontWeight: 600, fontSize: 13 }}>{cohort?.syndrome || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      {cohort?.pathognomonic && (
        <div style={{ background: '#fff8e1', border: '1px solid #ffc107', borderRadius: 4, padding: '6px 10px', marginBottom: 8, fontSize: 12 }}>
          <strong>PATHOGNOMONIC/KEY:</strong> {cohort.pathognomonic}
        </div>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {cohort?.cmc_pct > 80 && <Badge text={`CMC ${cohort.cmc_pct}%`} color="#c62828" />}
        {cohort?.invasive_pct > 50 && <Badge text={`Invasive fungal ${cohort.invasive_pct}%`} color="#ad1457" />}
        {cohort?.autoimmune_endo_pct > 30 && <Badge text={`Autoimmune endo ${cohort.autoimmune_endo_pct}%`} color="#6a1b9a" />}
        {cohort?.viral_suscept_pct > 30 && <Badge text={`Viral suscept ${cohort.viral_suscept_pct}%`} color="#4527a0" />}
        {cohort?.mycobacterial_pct > 30 && <Badge text={`Mycobacterial ${cohort.mycobacterial_pct}%`} color="#1565c0" />}
        {cohort?.bacterial_pct > 30 && <Badge text={`Bacterial ${cohort.bacterial_pct}%`} color="#0277bd" />}
        {cohort?.cns_candida_pct > 30 && <Badge text={`CNS Candida ${cohort.cns_candida_pct}%`} color="#4a148c" />}
        {cohort?.dermatophyte_pct > 40 && <Badge text={`Dermatophyte ${cohort.dermatophyte_pct}%`} color="#37474f" />}
        {cohort?.jaki_response_pct > 50 && <Badge text={`JAKi response ${cohort.jaki_response_pct}%`} color="#2e7d32" />}
        {cohort?.azole_response_pct > 70 && <Badge text={`Azole response ${cohort.azole_response_pct}%`} color="#388e3c" />}
        <Badge text={`MIM gene: ${cohort?.mim_gene}`} color="#555" />
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div>Loading...</div>;
  const s = data.aggregate_stats || {};
  return (
    <div>
      <h2 style={{ color: '#b71c1c', marginBottom: 4 }}>{data.atlas}</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>{data.subtitle}</p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 12, marginBottom: 24 }}>
        {[
          { label: 'Total Patients', value: data.total_patients, color: '#b71c1c' },
          { label: 'CMC rate', value: `${s.cmc_pct}%`, color: '#c62828' },
          { label: 'Invasive fungal', value: `${s.invasive_pct}%`, color: '#ad1457' },
          { label: 'Autoimmune endo', value: `${s.autoimmune_endo_pct}%`, color: '#6a1b9a' },
          { label: 'Viral suscept', value: `${s.viral_suscept_pct}%`, color: '#4527a0' },
          { label: 'Mycobacterial', value: `${s.mycobacterial_pct}%`, color: '#1565c0' },
          { label: 'CNS Candida', value: `${s.cns_candida_pct}%`, color: '#4a148c' },
          { label: 'Azole response', value: `${s.azole_response_pct}%`, color: '#388e3c' },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ background: '#f9f9f9', border: `1px solid ${color}`, borderRadius: 8, padding: 14, textAlign: 'center' }}>
            <div style={{ fontSize: 22, fontWeight: 700, color }}>{value}</div>
            <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>{label}</div>
          </div>
        ))}
      </div>
      <h3 style={{ marginBottom: 8 }}>Cascade Testing Concepts</h3>
      <ul style={{ paddingLeft: 20 }}>
        {(data.cascade_testing_concepts || []).map((c, i) => (
          <li key={i} style={{ marginBottom: 6, fontSize: 13 }}>{c}</li>
        ))}
      </ul>
      <p style={{ fontSize: 12, color: '#888', marginTop: 12 }}>Seeds: {data.seeds} · IUIS: {data.iuis_category}</p>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <div>Loading...</div>;
  const cols = [
    { key: 'gene', label: 'Gene' },
    { key: 'locus', label: 'Locus' },
    { key: 'aa', label: 'aa' },
    { key: 'inheritance', label: 'Inh' },
    { key: 'cmc_pct', label: 'CMC%' },
    { key: 'invasive_pct', label: 'Invasive%' },
    { key: 'autoimmune_endo_pct', label: 'EndoAI%' },
    { key: 'mycobacterial_pct', label: 'Myco%' },
    { key: 'cns_candida_pct', label: 'CNS-C%' },
    { key: 'azole_response_pct', label: 'Azole%' },
    { key: 'jaki_response_pct', label: 'JAKi%' },
    { key: 'mean_onset_months', label: 'Onset(mo)' },
  ];
  const cohorts = (data.cohort_summaries || []);
  return (
    <div style={{ overflowX: 'auto' }}>
      <h3 style={{ marginBottom: 8 }}>Gene Cohort Comparison (8 × 40 patients = 320 aggregate)</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#b71c1c', color: '#fff' }}>
            {cols.map(c => <th key={c.key} style={{ padding: '6px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{c.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {cohorts.map((r, i) => (
            <tr key={r.gene} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa', borderBottom: '1px solid #eee' }}>
              {cols.map(c => (
                <td key={c.key} style={{ padding: '5px 10px', fontWeight: c.key === 'gene' ? 700 : 400, color: c.key === 'gene' ? (GENE_COLORS[r.gene] || '#333') : '#333' }}>
                  {r[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <h3 style={{ marginTop: 24, marginBottom: 8 }}>Phenotype Matrix</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#37474f', color: '#fff' }}>
            {['Gene','Group','CMC','Invasive','EndoAI','Viral','Myco','Bacterial','CNS-C','Derm','JAKi','Key Feature'].map(h => (
              <th key={h} style={{ padding: '6px 8px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {PHENOTYPE_ROWS.map((r, i) => (
            <tr key={r.gene} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa', borderBottom: '1px solid #eee' }}>
              <td style={{ padding: '5px 8px', fontWeight: 700, color: GENE_COLORS[r.gene] || '#333' }}>{r.gene}</td>
              <td style={{ padding: '5px 8px', fontSize: 11, color: '#555' }}>{r.group}</td>
              <td style={{ padding: '5px 8px' }}>{r.cmc}</td>
              <td style={{ padding: '5px 8px' }}>{r.invasive}</td>
              <td style={{ padding: '5px 8px' }}>{r.endo}</td>
              <td style={{ padding: '5px 8px' }}>{r.viral}</td>
              <td style={{ padding: '5px 8px' }}>{r.myco}</td>
              <td style={{ padding: '5px 8px' }}>{r.bacterial}</td>
              <td style={{ padding: '5px 8px' }}>{r.cns_c}</td>
              <td style={{ padding: '5px 8px' }}>{r.derm}</td>
              <td style={{ padding: '5px 8px' }}>{r.jaki}</td>
              <td style={{ padding: '5px 8px', fontSize: 11, color: '#555', maxWidth: 280 }}>{r.unique}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PathwayMapTab({ data }) {
  if (!data) return <div>Loading...</div>;
  const groups = data.diagnostic_groups || {};
  const pathog = data.pathognomonic_features || {};
  return (
    <div>
      <h3 style={{ marginBottom: 8 }}>Diagnostic Groups — CMC / Th17 / CLR Pathway</h3>
      {Object.entries(groups).map(([k, g]) => (
        <div key={k} style={{ border: '1px solid #ccc', borderRadius: 8, padding: 14, marginBottom: 14 }}>
          <div style={{ fontWeight: 700, color: '#b71c1c', marginBottom: 4 }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
          <div style={{ fontSize: 12, color: '#555', marginBottom: 4 }}><strong>Genes:</strong> {g.genes?.join(', ')}</div>
          <div style={{ fontSize: 12, marginBottom: 4 }}><strong>Mechanism:</strong> {g.mechanism}</div>
          <div style={{ fontSize: 12, marginBottom: 4 }}><strong>Shared feature:</strong> {g.shared_feature}</div>
          <div style={{ fontSize: 12 }}><strong>Key DDx:</strong> {g.key_ddx}</div>
        </div>
      ))}
      <h3 style={{ marginTop: 20, marginBottom: 8 }}>Pathognomonic / Key Features per Gene</h3>
      {Object.entries(pathog).map(([gene, feat]) => (
        <div key={gene} style={{ display: 'flex', gap: 10, marginBottom: 8, alignItems: 'flex-start' }}>
          <div style={{ minWidth: 100, background: GENE_COLORS[gene.split('_')[0]] || '#555', color: '#fff', borderRadius: 4, padding: '3px 8px', fontSize: 12, fontWeight: 700 }}>{gene.split('_')[0]}</div>
          <div style={{ fontSize: 12, color: '#333' }}>{feat}</div>
        </div>
      ))}
      <h3 style={{ marginTop: 20, marginBottom: 8 }}>Diagnostic Pathway</h3>
      <ol style={{ paddingLeft: 20 }}>
        {((data.gene_cohorts ? [] : [])).map((step, i) => (
          <li key={i} style={{ marginBottom: 6, fontSize: 13 }}>{step}</li>
        ))}
      </ol>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div>Loading...</div>;
  const defs = data.definitions || {};
  const distinguishers = data.key_distinguishers || {};
  const pathway = data.diagnostic_pathway || [];
  return (
    <div>
      <h3 style={{ marginBottom: 8 }}>Gene Definitions</h3>
      {Object.entries(defs).map(([gene, def]) => (
        <div key={gene} style={{ marginBottom: 14, padding: 12, border: `2px solid ${GENE_COLORS[gene] || '#ccc'}`, borderRadius: 6 }}>
          <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#333', marginBottom: 6, fontSize: 14 }}>{gene}</div>
          <div style={{ fontSize: 12, color: '#333' }}>{def}</div>
        </div>
      ))}
      <h3 style={{ marginTop: 20, marginBottom: 8 }}>Key Distinguishers</h3>
      {Object.entries(distinguishers).map(([k, v]) => (
        <div key={k} style={{ marginBottom: 8 }}>
          <span style={{ fontWeight: 600, color: '#b71c1c', fontSize: 12 }}>{k.replace(/_/g, ' ')}: </span>
          <span style={{ fontSize: 12 }}>{v}</span>
        </div>
      ))}
      <h3 style={{ marginTop: 20, marginBottom: 8 }}>Diagnostic Pathway</h3>
      <ol style={{ paddingLeft: 20 }}>
        {pathway.map((step, i) => (
          <li key={i} style={{ marginBottom: 6, fontSize: 13 }}>{step}</li>
        ))}
      </ol>
    </div>
  );
}

export default function HeredMucocutaneousCandidiasisCMCAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const endpoints = [
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ];
    Promise.all(endpoints)
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const activeData = tab === 'Overview' ? overview
                   : tab === 'Gene Table' ? breakdown
                   : tab === 'Pathway Map' ? breakdown
                   : definitions;

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui, sans-serif', maxWidth: 1200 }}>
      <div style={{ marginBottom: 16 }}>
        <h1 style={{ color: '#b71c1c', marginBottom: 4, fontSize: 20 }}>
          &#x1f9ec; Hereditary-Mucocutaneous-Candidiasis-CMC-Atlas
        </h1>
        <p style={{ color: '#555', fontSize: 13, margin: 0 }}>
          Complete 8-Gene Reference — STAT1·IL17RA·IL17F·TRAF3IP2·IL17RC·RORC·CARD9·CLEC7A ·
          Th17 Pathway / CLR Innate Sensing / Invasive Fungal Susceptibility · 320-Patient Aggregate · Seeds 3902–3909
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #b71c1c', paddingBottom: 4 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: '4px 4px 0 0', border: 'none', cursor: 'pointer',
            background: tab === t ? '#b71c1c' : '#f5f5f5',
            color: tab === t ? '#fff' : '#333', fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading atlas data…</div>}
      {error && <div style={{ padding: 16, background: '#ffebee', borderRadius: 6, color: '#c62828' }}>Error: {error}</div>}

      {!loading && !error && (
        <>
          {tab === 'Overview' && <OverviewTab data={overview} />}
          {tab === 'Gene Table' && <GeneTableTab data={breakdown} />}
          {tab === 'Pathway Map' && (
            <div>
              <PathwayMapTab data={breakdown} />
              <h3 style={{ marginTop: 24, marginBottom: 10 }}>Gene Cards</h3>
              {GENES.map(gene => (
                <GeneCard
                  key={gene}
                  gene={gene}
                  color={GENE_COLORS[gene]}
                  info={GENE_INFO[gene]}
                  cohort={(breakdown?.gene_cohorts || []).find(c => c.gene === gene)}
                />
              ))}
            </div>
          )}
          {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
        </>
      )}
    </div>
  );
}

const GENES = ["STAT1", "IL17RA", "IL17F", "TRAF3IP2", "IL17RC", "RORC", "CARD9", "CLEC7A"];
