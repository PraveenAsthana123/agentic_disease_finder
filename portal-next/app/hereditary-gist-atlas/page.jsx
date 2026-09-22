'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-gist-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'KIT':    '#b71c1c',  // deep red          — Hereditary GIST type 1; KIT IHC 95% PATHOGNOMONIC; exon 9 800mg
  'PDGFRA': '#1a237e',  // deep indigo       — D842V PATHOGNOMONIC; avapritinib FDA 2020; 91% ORR NAVIGATOR
  'SDHA':   '#e65100',  // deep orange       — SDH-deficient GIST; Carney Triad NOT hereditary; SDHB IHC PATHOGNOMONIC
  'SDHB':   '#1b5e20',  // deep green        — CSS; malignancy 30-50% HIGHEST; methoxytyramine; DOTATATE PET
  'SDHC':   '#4a148c',  // deep purple       — PGL3; malignancy 1-3% LOWEST; NOT imprinted; HNPGL
  'SDHD':   '#006064',  // dark teal         — PGL1; PATERNAL IMPRINTING; maternal NOT at risk; multilocal HNPGL
  'NF1':    '#880e4f',  // deep magenta      — NF1-GIST multifocal small bowel PATHOGNOMONIC; MEK inhibitor
  'MAX':    '#33691e',  // deep olive        — PGL5; PATERNAL IMPRINTING; bilateral adrenal PHEO; adrenaline
};

const GENE_INFO = {
  'KIT':    { full: 'HGIST1 / KIT-IHC-95%-PATHOGNOMONIC / Exon9-800mg / Exon11-Best-Response / Exon17-Ripretinib', locus: '4q12',    size: '976 aa / 110 kDa',  inh: 'AD GOF' },
  'PDGFRA': { full: 'HGIST2 / D842V-IMATINIB-RESISTANT-PATHOGNOMONIC / Avapritinib-FDA2020-91%-ORR / Epithelioid-Gastric', locus: '4q12',    size: '1089 aa / 122 kDa', inh: 'AD GOF' },
  'SDHA':   { full: 'SDH-Deficient-GIST / SDHB-IHC-Loss-PATHOGNOMONIC / Carney-Triad-NOT-Hereditary / Multifocal-Gastric-Young-Female', locus: '5p15.33', size: '664 aa / 70 kDa',   inh: 'AR/AD LOF' },
  'SDHB':   { full: 'CSS / Malignancy-30-50%-HIGHEST / Methoxytyramine-SDHB-Signature / DOTATATE-PET-Preferred', locus: '1p36.13', size: '280 aa / 30 kDa',   inh: 'AD LOF' },
  'SDHC':   { full: 'PGL3 / Malignancy-1-3%-LOWEST / NOT-Imprinted / HNPGL-Dominant / CSS-GIST-10%', locus: '1q23.3',  size: '169 aa / 15 kDa',   inh: 'AD LOF' },
  'SDHD':   { full: 'PGL1 / PATERNAL-IMPRINTING-Maternal-NOT-at-Risk / Multilocal-HNPGL-Bilateral-Carotid / Pulsatile-Tinnitus-First', locus: '11q23.1', size: '160 aa / 12 kDa',   inh: 'AD LOF (Paternal)' },
  'NF1':    { full: "NF1-GIST-Multifocal-Small-Bowel-PATHOGNOMONIC / KIT-PDGFRA-SDH-WT / Imatinib-POOR / MEK-Binimetinib-Selumetinib", locus: '17q11.2', size: '2839 aa / 319 kDa', inh: 'AD LOF' },
  'MAX':    { full: 'PGL5 / PATERNAL-IMPRINTING-Maternal-NOT-at-Risk / Bilateral-Adrenal-PHEO / Adrenaline-Secreting-Metanephrine', locus: '14q23.3', size: '160 aa / 17 kDa',   inh: 'AD LOF (Paternal)' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#1565c0', color: '#fff', borderRadius: 4,
      padding: '2px 8px', fontSize: 11, fontWeight: 700, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneBar({ gene, pct, color }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ fontWeight: 700, color }}>{gene}</span>
        <span style={{ color: '#333' }}>{pct}%</span>
      </div>
      <div style={{ background: '#e0e0e0', borderRadius: 4, height: 14 }}>
        <div style={{ background: color, width: `${pct}%`, height: '100%', borderRadius: 4, transition: 'width 0.6s ease' }} />
      </div>
    </div>
  );
}

export default function HreditaryGISTAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [ov, br, df] = await Promise.all([
          fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
          fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
          fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
        ]);
        setOverview(ov);
        setBreakdown(br);
        setDefinitions(df);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#666' }}>Loading Hereditary GIST Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#c62828' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'Inter, sans-serif', maxWidth: 1100, margin: '0 auto', padding: '24px 16px' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#b71c1c 100%)', borderRadius: 12, padding: '28px 32px', marginBottom: 24, color: '#fff' }}>
        <div style={{ fontSize: 13, opacity: 0.8, marginBottom: 6 }}>🧬 Hereditary Cancer Predisposition Atlas Series</div>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800 }}>Hereditary GIST Predisposition Atlas</h1>
        <div style={{ marginTop: 8, fontSize: 14, opacity: 0.9 }}>
          Complete 8-Gene Reference · KIT · PDGFRA · SDHA · SDHB · SDHC · SDHD · NF1 · MAX
        </div>
        <div style={{ marginTop: 6, fontSize: 13, opacity: 0.8 }}>
          320-Patient Aggregate · 8 × 40 · Seeds 3198-3205 · Gastrointestinal Stromal Tumor + Paraganglioma Predisposition
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <Badge text={`${overview?.total_patients || 320} Patients`} color="#0d47a1" />
          <Badge text={`${overview?.genes_n || 8} Genes`} color="#1b5e20" />
          <Badge text={`Highest Risk: ${overview?.highest_risk_gene || 'SDHB'} (${overview?.highest_risk_pct || ''}%)`} color="#b71c1c" />
          <Badge text="KIT-IHC-95%-GIST-PATHOGNOMONIC" color="#880e4f" />
          <Badge text="D842V-IMATINIB-RESISTANT" color="#006064" />
          <Badge text="SDHB-IHC-PATHOGNOMONIC-ALL-SDH" color="#4a148c" />
          <Badge text="PATERNAL-IMPRINTING-SDHD-MAX" color="#e65100" />
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e0e0e0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '10px 20px', border: 'none', background: tab === t ? '#1a237e' : 'transparent',
            color: tab === t ? '#fff' : '#555', borderRadius: '6px 6px 0 0', cursor: 'pointer',
            fontWeight: tab === t ? 700 : 400, fontSize: 14, transition: 'all 0.2s'
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          {/* Summary Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 16, marginBottom: 28 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#1565c0' },
              { label: 'Severe Event Rate', value: `${overview.severe_total_pct}%`, color: '#c62828' },
              { label: 'Highest Risk Gene', value: `${overview.highest_risk_gene} (${overview.highest_risk_pct}%)`, color: '#1b5e20' },
              { label: 'Genes in Atlas', value: overview.genes_n, color: '#6a1b9a' },
            ].map(c => (
              <div key={c.label} style={{ background: '#fff', borderRadius: 10, padding: '18px 20px', boxShadow: '0 2px 8px rgba(0,0,0,0.08)', borderLeft: `4px solid ${c.color}` }}>
                <div style={{ fontSize: 12, color: '#888', marginBottom: 4 }}>{c.label}</div>
                <div style={{ fontSize: 22, fontWeight: 800, color: c.color }}>{c.value}</div>
              </div>
            ))}
          </div>

          {/* Severe Event Rate Bar Chart */}
          <div style={{ background: '#fff', borderRadius: 10, padding: '20px 24px', boxShadow: '0 2px 8px rgba(0,0,0,0.08)', marginBottom: 24 }}>
            <h3 style={{ margin: '0 0 16px', fontSize: 16, color: '#333' }}>Severe Event Rate by Gene (% of 40-patient cohort)</h3>
            {overview.gene_summary?.map(r => (
              <GeneBar key={r.gene} gene={r.gene} pct={r.severe_pct} color={GENE_COLORS[r.gene] || '#1565c0'} />
            ))}
          </div>

          {/* Key Clinical Facts */}
          <div style={{ background: '#fff', borderRadius: 10, padding: '20px 24px', boxShadow: '0 2px 8px rgba(0,0,0,0.08)', marginBottom: 24 }}>
            <h3 style={{ margin: '0 0 16px', fontSize: 16, color: '#333' }}>Key Clinical Signals</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))', gap: 16 }}>
              {[
                { gene: 'KIT', title: 'KIT IHC 95% PATHOGNOMONIC', body: 'KIT (CD117) IHC positive >95% GIST = PATHOGNOMONIC surrogate. DOG1 more specific. Exon 9 → imatinib 800mg (NOT 400mg) — critical dosing pitfall.' },
                { gene: 'PDGFRA', title: 'D842V = IMATINIB RESISTANT', body: 'PDGFRA D842V PATHOGNOMONIC imatinib resistance. Avapritinib FDA 2020: NAVIGATOR trial 91% ORR = BEST ever GIST trial. Epithelioid/myxoid gastric GIST + KIT-weak → test D842V.' },
                { gene: 'SDHB', title: 'SDHB IHC PATHOGNOMONIC', body: 'SDHB IHC loss = PATHOGNOMONIC for ALL SDH-deficient GIST (any subunit A/B/C/D). SDHB is the canary — degrades when ANY SDH subunit lost.' },
                { gene: 'SDHB', title: 'SDHB Malignancy 30-50% HIGHEST', body: 'SDHB-PGL malignancy 30-50% = HIGHEST of all SDH genes. Plasma methoxytyramine (dopamine metabolite) elevated = SDHB signature. DOTATATE PET preferred.' },
                { gene: 'SDHD', title: 'PATERNAL IMPRINTING', body: 'SDHD + MAX: PATERNAL imprinting — maternal carriers NOT at risk. Paternal SDHD → full surveillance mandatory. Maternal SDHD → reassure.' },
                { gene: 'NF1', title: 'NF1-GIST Multifocal Small Bowel', body: 'Multifocal GIST in SMALL BOWEL (NOT gastric) = NF1-GIST PATHOGNOMONIC. KIT/PDGFRA/SDH-WT. Imatinib POOR response. MEK inhibitor (binimetinib/selumetinib) active.' },
              ].map(f => (
                <div key={f.gene + f.title} style={{ background: '#f8f9fa', borderRadius: 8, padding: '14px 16px', borderLeft: `3px solid ${GENE_COLORS[f.gene] || '#1565c0'}` }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: GENE_COLORS[f.gene], marginBottom: 6 }}>{f.gene}: {f.title}</div>
                  <div style={{ fontSize: 13, color: '#444', lineHeight: 1.5 }}>{f.body}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && overview && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', background: '#fff', borderRadius: 10, overflow: 'hidden', boxShadow: '0 2px 8px rgba(0,0,0,0.08)' }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene', 'Locus', 'Size', 'Inheritance', 'Severe Event %', 'Mean Age', 'Key Info'].map(h => (
                  <th key={h} style={{ padding: '12px 14px', textAlign: 'left', fontSize: 13, fontWeight: 700 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {overview.gene_summary?.map((row, i) => (
                <tr key={row.gene} style={{ background: i % 2 === 0 ? '#fff' : '#f8f9fa', borderBottom: '1px solid #e0e0e0' }}>
                  <td style={{ padding: '12px 14px', fontWeight: 800, color: GENE_COLORS[row.gene], fontSize: 14 }}>{row.gene}</td>
                  <td style={{ padding: '12px 14px', fontSize: 13, color: '#555' }}>{row.locus}</td>
                  <td style={{ padding: '12px 14px', fontSize: 12, color: '#666' }}>{GENE_INFO[row.gene]?.size || '—'}</td>
                  <td style={{ padding: '12px 14px', fontSize: 12, color: '#555' }}>{GENE_INFO[row.gene]?.inh || '—'}</td>
                  <td style={{ padding: '12px 14px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ background: '#e0e0e0', borderRadius: 4, height: 10, width: 80, flex: 'none' }}>
                        <div style={{ background: GENE_COLORS[row.gene], width: `${row.severe_pct}%`, height: '100%', borderRadius: 4 }} />
                      </div>
                      <span style={{ fontSize: 13, fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.severe_pct}%</span>
                    </div>
                  </td>
                  <td style={{ padding: '12px 14px', fontSize: 13, color: '#555' }}>{row.mean_age_onset}yr</td>
                  <td style={{ padding: '12px 14px', fontSize: 12, color: '#444', maxWidth: 280 }}>{GENE_INFO[row.gene]?.full || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(440px,1fr))', gap: 20 }}>
          {overview.gene_summary?.map(row => (
            <div key={row.gene} style={{ background: '#fff', borderRadius: 10, padding: '20px 22px', boxShadow: '0 2px 8px rgba(0,0,0,0.08)', borderTop: `4px solid ${GENE_COLORS[row.gene] || '#1565c0'}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                <div>
                  <span style={{ fontSize: 20, fontWeight: 900, color: GENE_COLORS[row.gene] }}>{row.gene}</span>
                  <span style={{ fontSize: 12, color: '#888', marginLeft: 8 }}>{row.locus} · {GENE_INFO[row.gene]?.size}</span>
                </div>
                <Badge text={`${row.severe_pct}% severe`} color={GENE_COLORS[row.gene]} />
              </div>
              <div style={{ fontSize: 12, color: '#555', marginBottom: 10 }}>
                <strong>Inheritance:</strong> {row.inheritance}
              </div>
              <div style={{ fontSize: 12, color: '#c62828', marginBottom: 10, fontWeight: 600 }}>
                🔴 Pathognomonic: {row.pathognomonic}
              </div>
              <div style={{ fontSize: 12, color: '#1b5e20', marginBottom: 10, fontWeight: 600 }}>
                🔬 Cancer Risk: {row.cancer_risk}
              </div>
              <div style={{ fontSize: 12, color: '#555', marginBottom: 12 }}>
                <strong>Surveillance:</strong> {row.surveillance_key}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {row.key_distinctions?.map(k => (
                  <Badge key={k} text={k} color={GENE_COLORS[row.gene]} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          {definitions.definitions?.map((def, i) => (
            <div key={i} style={{ background: '#fff', borderRadius: 10, padding: '20px 24px', boxShadow: '0 2px 8px rgba(0,0,0,0.08)', marginBottom: 16 }}>
              <div style={{ fontWeight: 800, fontSize: 14, color: '#1a237e', marginBottom: 12, borderBottom: '2px solid #e3f2fd', paddingBottom: 8 }}>
                {def.term}
              </div>
              <pre style={{ fontFamily: 'inherit', fontSize: 13, color: '#444', margin: 0, whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>
                {def.definition}
              </pre>
            </div>
          ))}
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: 32, padding: '16px 20px', background: '#f5f5f5', borderRadius: 8, fontSize: 12, color: '#888', textAlign: 'center' }}>
        Hereditary GIST Predisposition Atlas · KIT-PDGFRA-SDHA-SDHB-SDHC-SDHD-NF1-MAX ·
        320 patients (8 × 40) · Seeds 3198-3205 · API: /api/hereditary-gist-atlas/overview|breakdown|definitions
      </div>
    </div>
  );
}
