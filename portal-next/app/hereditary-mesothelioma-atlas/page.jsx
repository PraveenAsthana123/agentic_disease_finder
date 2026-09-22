'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-mesothelioma-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'BAP1':    '#1a237e',  // deep navy        -- TPDS primary; mesothelioma 30-60%; MBAITs PATHOGNOMONIC
  'BRCA2':   '#880e4f',  // deep magenta     -- HBOC; mesothelioma 2-5x; FA-D1 most severe
  'NF2':     '#1b5e20',  // deep green       -- bilateral VS 100% PATHOGNOMONIC; somatic 40-80% sporadic meso
  'CDKN2A':  '#e65100',  // deep orange      -- FAMM; melanoma 25-36%; 9p21 deletion 50-80% sporadic meso
  'TP53':    '#b71c1c',  // deep red         -- LFS; AVOID RADIATION ABSOLUTELY; WBMRI Toronto
  'SMARCB1': '#4a148c',  // deep purple      -- AT/RT infant PATHOGNOMONIC; MRT kidney; tazemetostat EZH2
  'MLH1':    '#006064',  // dark teal        -- Lynch1; MSI-H; pembrolizumab all histologies; CAPP2
  'ATM':     '#33691e',  // deep olive       -- A-T; RADIOSENSITIVITY ABSOLUTE biallelic; ceralasertib
};

const GENE_INFO = {
  'BAP1':    { full: 'BAP1-TPDS / Mesothelioma-30-60%-PRIMARY / Uveal-Melanoma-30-50% / MBAITs-PATHOGNOMONIC / AVOID-ASBESTOS-ABSOLUTELY', locus: '3p21.1',  size: '729 aa / 80 kDa',   inh: 'AD LOF' },
  'BRCA2':   { full: 'HBOC / Mesothelioma-2-5x / Olaparib-PARP / Cisplatin-Sensitive / FA-D1-Biallelic-MOST-SEVERE',                        locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF' },
  'NF2':     { full: 'Bilateral-VS-100%-PATHOGNOMONIC / Meningioma-50-75% / Somatic-NF2-40-80%-Sporadic-Meso / Bevacizumab-VS / YAP-Driver', locus: '22q12.2', size: '595 aa / 70 kDa',   inh: 'AD LOF' },
  'CDKN2A':  { full: 'FAMM / Melanoma-25-36%-PRIMARY / Pancreatic-20x / 9p21-Deletion-50-80%-Sporadic-Meso / CDK4-6-Inhibitors',             locus: '9p21.3',  size: '156 aa / 16 kDa',   inh: 'AD LOF' },
  'TP53':    { full: 'LFS / AVOID-RADIATION-ABSOLUTELY / WBMRI-Toronto-Annually / Sarcoma-50-60%-PRIMARY / R337H-Brazilian-Founder',          locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF' },
  'SMARCB1': { full: 'AT/RT-Infant-PATHOGNOMONIC / MRT-Kidney-Infant-PATHOGNOMONIC / SMARCB1-Null-IHC / Tazemetostat-EZH2-FDA / Schwannomatosis2', locus: '22q11.23', size: '385 aa / 44 kDa', inh: 'AD LOF' },
  'MLH1':    { full: 'Lynch1 / CRC-40-50% / Endometrial-40-50% / MSI-H-PATHOGNOMONIC / Pembrolizumab-ALL-Histologies / CAPP2-Aspirin-50%',   locus: '3p22.2',  size: '756 aa / 85 kDa',   inh: 'AD LOF' },
  'ATM':     { full: 'A-T-Biallelic / RADIOSENSITIVITY-ABSOLUTE / Mesothelioma-2-4x-Monoallelic / Ceralasertib-ATRi-Olaparib / HBOC-2',     locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD LOF / AR biallelic' },
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

export default function HreditaryMesotheliomaAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#666' }}>Loading Hereditary Mesothelioma Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#c62828' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'Inter, sans-serif', maxWidth: 1100, margin: '0 auto', padding: '24px 16px' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#b71c1c 100%)', borderRadius: 12, padding: '28px 32px', marginBottom: 24, color: '#fff' }}>
        <div style={{ fontSize: 13, opacity: 0.8, marginBottom: 6 }}>🧬 Hereditary Cancer Predisposition Atlas Series</div>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800 }}>Hereditary Mesothelioma Predisposition Atlas</h1>
        <div style={{ marginTop: 8, fontSize: 14, opacity: 0.9 }}>
          Complete 8-Gene Reference · BAP1 · BRCA2 · NF2 · CDKN2A · TP53 · SMARCB1 · MLH1 · ATM
        </div>
        <div style={{ marginTop: 6, fontSize: 13, opacity: 0.8 }}>
          320-Patient Aggregate · 8 × 40 · Seeds 3206-3213 · Mesothelioma + Multi-Cancer Predisposition
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <Badge text={`${overview?.total_patients || 320} Patients`} color="#0d47a1" />
          <Badge text={`${overview?.genes_n || 8} Genes`} color="#1b5e20" />
          <Badge text={`Highest Risk: ${overview?.highest_risk_gene || 'TP53'} (${overview?.highest_risk_pct || ''}%)`} color="#b71c1c" />
          <Badge text="BAP1-NULL-IHC-PATHOGNOMONIC" color="#880e4f" />
          <Badge text="AVOID-ASBESTOS-ABSOLUTELY" color="#1a237e" />
          <Badge text="CDKN2A-FISH-MESO-DIAGNOSTIC" color="#e65100" />
          <Badge text="RADIOSENSITIVITY-ABSOLUTE-ATM-TP53" color="#4a148c" />
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
                { gene: 'BAP1', title: 'BAP1-null IHC PATHOGNOMONIC', body: 'BAP1-TPDS: mesothelioma 30-60% PRIMARY + uveal melanoma 30-50% = PATHOGNOMONIC combination. MBAITs (dome-shaped BAP1-null skin lesions) PATHOGNOMONIC. Avoid asbestos ABSOLUTELY — synergistic with germline BAP1.' },
                { gene: 'NF2', title: 'Bilateral VS 100% PATHOGNOMONIC + Somatic NF2', body: 'NF2: bilateral vestibular schwannoma 100% by age 30yr = PATHOGNOMONIC. Somatic NF2 deletion 40-80% sporadic mesothelioma = MOST COMMON sporadic alteration. YAP/TAZ activation — FAK inhibitor defactinib trials.' },
                { gene: 'CDKN2A', title: 'CDKN2A FISH mesothelioma diagnostic', body: '9p21 homozygous deletion 50-80% sporadic mesothelioma. CDKN2A FISH (p16 deletion) = PATHOGNOMONIC for malignant mesothelioma vs reactive mesothelium. CDK4/6 inhibitor trials.' },
                { gene: 'TP53', title: 'AVOID RADIATION ABSOLUTELY (LFS)', body: 'Li-Fraumeni Syndrome: AVOID therapeutic radiation absolutely — radiation-induced secondary sarcoma lethal risk. WBMRI annually Toronto Protocol (no CT). R337H Brazilian founder 1/300 south Brazil.' },
                { gene: 'SMARCB1', title: 'AT/RT + MRT PATHOGNOMONIC + Tazemetostat', body: 'AT/RT (infant brain) + MRT (infant kidney) = SMARCB1 biallelic PATHOGNOMONIC. SMARCB1/INI1 null IHC PATHOGNOMONIC. Tazemetostat (EZH2 inhibitor) FDA-approved SMARCB1-deficient epithelioid sarcoma.' },
                { gene: 'ATM', title: 'RADIOSENSITIVITY ABSOLUTE (A-T biallelic)', body: 'Biallelic A-T: AVOID ALL THERAPEUTIC RADIATION — LETHAL. Cerebellar ataxia + telangiectasia PATHOGNOMONIC combination. Monoallelic ATM: mesothelioma 2-4x elevated. Ceralasertib (ATRi) + olaparib synthetic lethality.' },
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
        Hereditary Mesothelioma Predisposition Atlas · BAP1-BRCA2-NF2-CDKN2A-TP53-SMARCB1-MLH1-ATM ·
        320 patients (8 × 40) · Seeds 3206-3213 · API: /api/hereditary-mesothelioma-atlas/overview|breakdown|definitions
      </div>
    </div>
  );
}
