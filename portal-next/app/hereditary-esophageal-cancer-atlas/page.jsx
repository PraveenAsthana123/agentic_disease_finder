'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-esophageal-cancer-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'RHBDF2': '#7b1fa2',  // deep purple    -- tylosis PPK PATHOGNOMONIC; ESCC 100% penetrance
  'TP53':   '#b71c1c',  // deep red       -- LFS; AVOID RADIATION ABSOLUTELY; WBMRI Toronto
  'CDKN2A': '#4e342e',  // dark brown     -- FAMM/9p21 deletion ESCC/EAC
  'ATM':    '#1a237e',  // deep navy      -- A-T radiosensitivity ABSOLUTE; upper GI 2-3x
  'BRCA2':  '#1565c0',  // deep blue      -- HBOC; EAC 2-3x; olaparib; FA-D1 most severe
  'FANCA':  '#e65100',  // deep orange    -- FA type A; ESCC 400x RR HIGHEST; avoid aldehyde
  'MLH1':   '#1b5e20',  // dark green     -- Lynch1; MSI-H; pembrolizumab all histologies
  'PALB2':  '#006064',  // dark teal      -- HBOC-2; breast 53%; TBCRC048 82% ORR
};

const GENE_INFO = {
  'RHBDF2': { full: 'TOC-Howel-Evans / ESCC-100%-Penetrance-age65 / PPK-PATHOGNOMONIC / Lugol-Iodine-Annual / AVOID-TOBACCO-ALCOHOL-ABSOLUTELY', locus: '17q25.1', size: '817 aa / 92 kDa',   inh: 'AD GOF' },
  'TP53':   { full: 'LFS / AVOID-RADIATION-ABSOLUTELY / WBMRI-Toronto-Annually / Sarcoma-50-60%-PRIMARY / R337H-Brazilian-Founder / p53-Aberrant-ESCC-EAC',    locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF' },
  'CDKN2A': { full: 'FAMM / Melanoma-25-36%-PRIMARY / Pancreatic-20x / 9p21-Deletion-ESCC-EAC-50-80% / CDK4-6-Inhibitors / Barrett-EAC-Early-Event',             locus: '9p21.3',  size: '156 aa / 16 kDa',   inh: 'AD LOF' },
  'ATM':    { full: 'A-T-Biallelic / RADIOSENSITIVITY-ABSOLUTE / Upper-GI-Esophageal-2-3x-Monoallelic / Ceralasertib-ATRi-Olaparib / HBOC-2',                   locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD LOF / AR biallelic' },
  'BRCA2':  { full: 'HBOC / EAC-2-3x / Olaparib-PARP / Cisplatin-Sensitive / FA-D1-Biallelic-MOST-SEVERE / BRCA2-PALB2-BRCA1-Module',                           locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF' },
  'FANCA':  { full: 'FA-Type-A-Most-Common-60% / ESCC-400x-RR-HIGHEST-Solid-Tumour / AVOID-ALDEHYDE-ABSOLUTELY / BMF-Radial-Ray / DEB-Test-PATHOGNOMONIC',        locus: '16q24.3', size: '1455 aa / 163 kDa', inh: 'AR LOF' },
  'MLH1':   { full: 'Lynch1 / CRC-40-50% / EAC-ESCC-2-3x / MSI-H-PATHOGNOMONIC / Pembrolizumab-ALL-Histologies / CAPP2-Aspirin-50%',                            locus: '3p22.2',  size: '756 aa / 90 kDa',   inh: 'AD LOF' },
  'PALB2':  { full: 'HBOC-2 / Breast-53%-Lifetime / Upper-GI-EAC-2-3x-Emerging / Olaparib-TBCRC048-82%-ORR-HIGHEST / FA-N-Biallelic / BRCA2-Bridge',            locus: '16p12.2', size: '1186 aa / 131 kDa', inh: 'AD LOF' },
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

export default function HereditaryEsophagealCancerAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#666' }}>Loading Hereditary Esophageal Cancer Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#c62828' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'Inter, sans-serif', maxWidth: 1100, margin: '0 auto', padding: '24px 16px' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#7b1fa2 0%,#e65100 100%)', borderRadius: 12, padding: '28px 32px', marginBottom: 24, color: '#fff' }}>
        <div style={{ fontSize: 13, opacity: 0.8, marginBottom: 6 }}>&#x1f9ec; Hereditary Cancer Predisposition Atlas Series</div>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800 }}>Hereditary Esophageal Cancer Predisposition Atlas</h1>
        <div style={{ marginTop: 8, fontSize: 14, opacity: 0.9 }}>
          Complete 8-Gene Reference · RHBDF2 · TP53 · CDKN2A · ATM · BRCA2 · FANCA · MLH1 · PALB2
        </div>
        <div style={{ marginTop: 6, fontSize: 13, opacity: 0.8 }}>
          320-Patient Aggregate · 8 × 40 · Seeds 3214-3221 · Esophageal Cancer + Multi-Cancer Predisposition
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <Badge text={`${overview?.total_patients || 320} Patients`} color="#6a1b9a" />
          <Badge text={`${overview?.genes_n || 8} Genes`} color="#1b5e20" />
          <Badge text={`Highest Risk: ${overview?.highest_risk_gene || 'RHBDF2'} (${overview?.highest_risk_pct || ''}%)`} color="#b71c1c" />
          <Badge text="RHBDF2-PPK-PATHOGNOMONIC-ESCC-100PCT" color="#7b1fa2" />
          <Badge text="FANCA-ESCC-400X-RR-HIGHEST" color="#e65100" />
          <Badge text="AVOID-TOBACCO-ALCOHOL-ABSOLUTELY" color="#4e342e" />
          <Badge text="RADIOSENSITIVITY-ABSOLUTE-ATM-TP53" color="#1a237e" />
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e0e0e0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '10px 20px', border: 'none', background: tab === t ? '#7b1fa2' : 'transparent',
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
              { label: 'Total Patients', value: overview.total_patients, color: '#7b1fa2' },
              { label: 'Severe Event Rate', value: `${overview.severe_total_pct}%`, color: '#c62828' },
              { label: 'Highest Risk Gene', value: `${overview.highest_risk_gene} (${overview.highest_risk_pct}%)`, color: '#1b5e20' },
              { label: 'Genes in Atlas', value: overview.genes_n, color: '#e65100' },
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
                { gene: 'RHBDF2', title: 'PPK PATHOGNOMONIC + Lugol Iodine Annual', body: 'TOC/Howel-Evans: ESCC nearly 100% penetrance by age 65yr. PPK (diffuse non-epidermolytic) PATHOGNOMONIC onset childhood. Annual Lugol iodine endoscopy from age 30yr MANDATORY. AVOID tobacco/alcohol ABSOLUTELY — synergistic co-risk.' },
                { gene: 'FANCA', title: 'ESCC 400x RR HIGHEST solid tumour in FA', body: 'FA type A (most common 60%): ESCC 10-15% lifetime vs 0.04% general = ~400x RR. AVOID ALDEHYDE/ALCOHOL ABSOLUTELY (aldehydes directly damage FA pathway). DEB test chromosomal fragility PATHOGNOMONIC FA. Annual endoscopy from age 16yr. ESCC risk persists post-HSCT.' },
                { gene: 'TP53', title: 'AVOID RADIATION ABSOLUTELY (LFS)', body: 'Li-Fraumeni Syndrome: AVOID therapeutic radiation absolutely — radiation-induced secondary sarcoma lethal risk. WBMRI annually Toronto Protocol (no CT). R337H Brazilian founder 1/300 south Brazil. p53 most commonly mutated gene in ESCC (~80-90% sporadic).' },
                { gene: 'CDKN2A', title: '9p21 deletion ESCC/EAC + FAMM', body: '9p21.3 homozygous deletion: 50-80% sporadic ESCC; common early EAC/Barrett\'s event. Germline CDKN2A: melanoma 25-36% PRIMARY; pancreatic 20x. CDK4/6 inhibitor trials in CDKN2A-deleted upper GI cancers.' },
                { gene: 'ATM', title: 'RADIOSENSITIVITY ABSOLUTE (A-T biallelic)', body: 'Biallelic A-T: AVOID ALL THERAPEUTIC RADIATION — LETHAL. Cerebellar ataxia + telangiectasia PATHOGNOMONIC. Monoallelic ATM: upper GI/esophageal 2-3x elevated. Ceralasertib (ATRi) + olaparib synthetic lethality ATM-deficient tumours.' },
                { gene: 'PALB2', title: 'Olaparib TBCRC048 82% ORR HIGHEST', body: 'PALB2/HBOC-2: breast 53% lifetime; upper GI/EAC 2-3x emerging. BRCA2-bridge protein — HRD phenotype equivalent BRCA2-null. TBCRC048 82% ORR = HIGHEST PARP inhibitor ORR in germline breast cancer. FA-N biallelic severe.' },
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
              <tr style={{ background: '#7b1fa2', color: '#fff' }}>
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
                &#x1f534; Pathognomonic: {row.pathognomonic}
              </div>
              <div style={{ fontSize: 12, color: '#1b5e20', marginBottom: 10, fontWeight: 600 }}>
                &#x1f52c; Cancer Risk: {row.cancer_risk}
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
              <div style={{ fontWeight: 800, fontSize: 14, color: '#7b1fa2', marginBottom: 12, borderBottom: '2px solid #f3e5f5', paddingBottom: 8 }}>
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
        Hereditary Esophageal Cancer Predisposition Atlas · RHBDF2-TP53-CDKN2A-ATM-BRCA2-FANCA-MLH1-PALB2 ·
        320 patients (8 × 40) · Seeds 3214-3221 · API: /api/hereditary-esophageal-cancer-atlas/overview|breakdown|definitions
      </div>
    </div>
  );
}
