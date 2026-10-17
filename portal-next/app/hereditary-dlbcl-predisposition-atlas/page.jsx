'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-dlbcl-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'TP53':       '#b71c1c',  // deep red        — LFS DLBCL 5-8x; AVOID RADIATION ABSOLUTELY; WB-MRI Toronto
  'ATM':        '#1a237e',  // deep navy        — RADIOSENSITIVITY ABSOLUTE; del11q 40%; ceralasertib ATRi
  'BRCA2':      '#4a148c',  // deep purple      — HRD 60%; cisplatin preferred; olaparib maintenance; HBOC
  'CHEK2':      '#006064',  // dark teal        — 1100delC European; ATM additive 8-10x; breast 40%
  'PIK3CD':     '#1b5e20',  // deep green       — APDS1 EBV-driven DLBCL; Leniolisib FDA2023; sirolimus
  'KMT2D':      '#e65100',  // deep orange      — Kabuki1; H3K4me1 loss; tazemetostat EZH2 inhibitor
  'CARD11':     '#880e4f',  // deep magenta     — BENTA GOF NF-κB venetoclax; LOF IVIG; MALT1 inhibitor
  'TNFRSF13B':  '#bf360c',  // deep red-brown   — CVID-2 NHL 10-12x HIGHEST; IVIG mandatory; C104R founder
};

const GENE_INFO = {
  'TP53':      { full: 'LFS-DLBCL-5-8x / AVOID-RADIATION-ABSOLUTELY / Complex-Karyotype-60pct / WB-MRI-Toronto / APR-246-Eprenetapopt / Venetoclax-R-ICE',                    locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF heterozygous' },
  'ATM':       { full: 'RADIOSENSITIVITY-ABSOLUTE-no-RT / NHL-DLBCL-4-5x / del11q-40pct / Ceralasertib-ATRi-Synthetic-Lethality / RIC-HSCT-no-TBI',                            locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD LOF heterozygous' },
  'BRCA2':     { full: 'HBOC-NHL-DLBCL-2-3x / HRD-60pct / Cisplatin-Preferred / Olaparib-PARP-Maintenance / Concurrent-Breast-Ovarian-15-20pct',                               locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF heterozygous' },
  'CHEK2':     { full: '1100delC-European-1in200 / IVS2-Ashkenazi / NHL-DLBCL-2x / ATM-CHEK2-Additive-8-10x / Breast-Concurrent-40pct / Standard-R-CHOP',                      locus: '22q12.1', size: '543 aa / 61 kDa',   inh: 'AD LOF heterozygous' },
  'PIK3CD':    { full: 'APDS1-GOF / EBV-DLBCL-70pct / Leniolisib-FDA2023-PI3Kdelta / Sirolimus-Bridge / RIC-HSCT-Curative / CMV-EBV-Monitoring-Mandatory',                     locus: '1p36.22', size: '1044 aa / 119 kDa', inh: 'AD GOF heterozygous' },
  'KMT2D':     { full: 'Kabuki1-Syndrome / H3K4me1-Enhancer-Loss / B-cell-Lymphoma / Tazemetostat-EZH2-FDA2020 / Azacitidine-Epigenetic-Combo / Facial-Dysmorphology',          locus: '12q13.12', size: '5537 aa / 500 kDa', inh: 'AD LOF heterozygous' },
  'CARD11':    { full: 'BENTA-GOF-NF-kB / Venetoclax-Preferred-GOF / Ibrutinib-LESS-Effective-GOF / MALT1-Inhibitor-JNJ / LOF-Immunodeficiency-IVIG / Atopic-Eosinophilia',    locus: '7p22.2',  size: '1154 aa / 133 kDa', inh: 'AD LOF/GOF heterozygous' },
  'TNFRSF13B': { full: 'CVID-2-TACI / NHL-DLBCL-10-12x-HIGHEST / C104R-1in500-European / IVIG-Mandatory-IgG-7gL / Rituximab-Lenalidomide-R-CHOP / Granulomatous-30pct',       locus: '17p11.2', size: '293 aa / 31 kDa',   inh: 'AD (haploinsufficiency/DN)' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: `${color}18`, color, border: `1px solid ${color}55`,
      borderRadius: 4, fontSize: 10, padding: '2px 7px', marginLeft: 6, fontWeight: 600,
    }}>{text}</span>
  );
}

function card(label, value, sub, color) {
  return (
    <div style={{ background: '#fff', border: `2px solid ${color}33`, borderRadius: 10,
      padding: '14px 18px', minWidth: 140, flex: 1 }}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: 10, color: '#aaa' }}>{sub}</div>
    </div>
  );
}

export default function HereditaryDLBCLPredispositionAtlasPage() {
  const [tab, setTab]                 = useState('Overview');
  const [overview, setOverview]       = useState(null);
  const [breakdown, setBreakdown]     = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState('');
  const [expandedGene, setExpandedGene] = useState(null);
  const [expandedDef, setExpandedDef]   = useState(null);

  useEffect(() => {
    setLoading(true); setError('');
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const tabStyle = (t) => ({
    padding: '8px 18px', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600,
    fontSize: 13, background: tab === t ? '#b71c1c' : '#f5f5f5', color: tab === t ? '#fff' : '#555',
    marginRight: 6,
  });

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary DLBCL Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#f8f9fa', minHeight: '100vh', padding: '28px 24px' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#b71c1c 0%,#880e4f 50%,#4a148c 100%)',
        borderRadius: 14, padding: '24px 32px', marginBottom: 24, color: '#fff' }}>
        <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 6 }}>
          🧬 Hereditary DLBCL Predisposition Atlas
        </div>
        <div style={{ fontSize: 13, opacity: 0.88 }}>
          Complete 8-Gene Reference · TP53 · ATM · BRCA2 · CHEK2 · PIK3CD · KMT2D · CARD11 · TNFRSF13B
        </div>
        <div style={{ fontSize: 11, opacity: 0.70, marginTop: 4 }}>
          320-Patient Aggregate · 8×40 · Seeds 3582–3589 · Diffuse Large B-Cell Lymphoma Germline Predisposition
        </div>
      </div>

      {/* Tabs */}
      <div style={{ marginBottom: 20 }}>
        {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          {/* Summary Cards */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            {card('Total Patients', overview.total_patients, '8 genes × 40 each', '#b71c1c')}
            {card('CR (1st-line)', overview.cr_n, `${Math.round(overview.cr_n/overview.total_patients*100)}% complete response`, '#1a237e')}
            {card('HRD Positive', overview.hrd_positive, 'BRCA2/ATM synthetic lethality', '#4a148c')}
            {card('EBV+ DLBCL', overview.ebv_positive, 'PIK3CD/CARD11/TNFRSF13B driven', '#1b5e20')}
            {card('Complex Karyotype', overview.complex_karyotype, 'TP53/ATM high-risk', '#e65100')}
            {card('5-yr Alive', overview.five_yr_alive, `${Math.round(overview.five_yr_alive/overview.total_patients*100)}% overall survival`, '#006064')}
            {card('IVIG Required', overview.ivig_required, 'PIK3CD / CARD11 LOF / TNFRSF13B', '#880e4f')}
            {card('Avg IPI', overview.avg_ipi, `Avg LDH ratio ${overview.avg_ldh_ratio}`, '#bf360c')}
          </div>

          {/* Clinical Pearl */}
          {overview.clinical_pearl && (
            <div style={{ background: '#fff8e1', border: '1px solid #ffcc80', borderRadius: 10,
              padding: '14px 18px', marginBottom: 20, fontSize: 12, color: '#5d4037' }}>
              <strong>⚡ Clinical Pearl:</strong> {overview.clinical_pearl}
            </div>
          )}

          {/* Gene Summary Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 14 }}>
            {(overview.gene_summary || []).map(gs => {
              const color = GENE_COLORS[gs.gene] || '#555';
              const info  = GENE_INFO[gs.gene] || {};
              const expanded = expandedGene === gs.gene;
              return (
                <div key={gs.gene} style={{ background: '#fff', border: `2px solid ${color}33`,
                  borderRadius: 12, padding: '16px 18px', cursor: 'pointer' }}
                  onClick={() => setExpandedGene(expanded ? null : gs.gene)}>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontSize: 17, fontWeight: 800, color }}>{gs.gene}</span>
                    <Badge text={gs.inheritance.split(' ')[0]} color={color} />
                    <Badge text={info.locus || gs.locus} color={color} />
                    <span style={{ marginLeft: 'auto', fontSize: 11, color: '#aaa' }}>{info.size}</span>
                  </div>
                  <div style={{ fontSize: 11, color: '#666', marginBottom: 6 }}>{gs.syndrome}</div>
                  <div style={{ fontSize: 11, color: '#888', marginBottom: 8 }}>
                    Mean age Dx: <strong style={{ color }}>{gs.mean_age_dx} yr</strong> · n={gs.n}
                  </div>
                  <div style={{ fontSize: 11, background: `${color}0d`, borderRadius: 6,
                    padding: '6px 10px', color, marginBottom: 6 }}>
                    <strong>DLBCL Risk:</strong> {gs.dlbcl_risk}
                  </div>
                  {expanded && (
                    <div style={{ marginTop: 10 }}>
                      <div style={{ fontSize: 11, color: '#555', marginBottom: 6 }}>
                        <strong>Full info:</strong> {info.full}
                      </div>
                      <div style={{ fontSize: 11, background: '#fff3e0', borderRadius: 6,
                        padding: '6px 10px', color: '#bf360c', marginBottom: 6 }}>
                        <strong>KEY AVOID:</strong> {gs.key_avoid}
                      </div>
                      <div style={{ fontSize: 11, background: '#e8f5e9', borderRadius: 6,
                        padding: '6px 10px', color: '#1b5e20', marginBottom: 6 }}>
                        <strong>KEY RULE:</strong> {gs.key_rule}
                      </div>
                      <div style={{ fontSize: 11, background: '#e3f2fd', borderRadius: 6,
                        padding: '6px 10px', color: '#1a237e' }}>
                        <strong>TARGETED Rx:</strong> {gs.targeted_rx}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', background: '#fff',
            borderRadius: 10, overflow: 'hidden', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#b71c1c', color: '#fff' }}>
                {['Gene','Syndrome','Locus','Inh','n','Mean Age Dx','CR n','EBV+','HRD+','Complex K','ABC','CNS','IVIG n','5yr Alive','Stage 3-4'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 700, fontSize: 11 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown.gene_data || {}).map(([gene, d], idx) => {
                const color = GENE_COLORS[gene] || '#555';
                return (
                  <tr key={gene} style={{ background: idx % 2 === 0 ? '#fff' : '#fafafa',
                    borderBottom: '1px solid #f0f0f0' }}>
                    <td style={{ padding: '9px 12px', fontWeight: 800, color }}>{gene}</td>
                    <td style={{ padding: '9px 12px', fontSize: 10, maxWidth: 160 }}>{d.syndrome?.split('-').slice(0,3).join('-')}</td>
                    <td style={{ padding: '9px 12px' }}>{d.locus}</td>
                    <td style={{ padding: '9px 12px', fontSize: 10 }}>{d.inheritance?.split(' ').slice(0,2).join(' ')}</td>
                    <td style={{ padding: '9px 12px', fontWeight: 700 }}>{d.n}</td>
                    <td style={{ padding: '9px 12px', color }}>{d.avg_age_dx} yr</td>
                    <td style={{ padding: '9px 12px' }}>{d.cr_n}</td>
                    <td style={{ padding: '9px 12px' }}>{d.ebv_positive_n}</td>
                    <td style={{ padding: '9px 12px' }}>{d.hrd_positive_n}</td>
                    <td style={{ padding: '9px 12px' }}>{d.complex_karyotype_n}</td>
                    <td style={{ padding: '9px 12px' }}>{d.abc_subtype_n}</td>
                    <td style={{ padding: '9px 12px' }}>{d.cns_involved_n}</td>
                    <td style={{ padding: '9px 12px' }}>{d.ivig_required_n}</td>
                    <td style={{ padding: '9px 12px', fontWeight: 700, color: '#1b5e20' }}>{d.five_yr_alive_n}</td>
                    <td style={{ padding: '9px 12px' }}>{d.stage_3_4_n}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(380px,1fr))', gap: 16 }}>
          {Object.entries(breakdown.gene_data || {}).map(([gene, d]) => {
            const color = GENE_COLORS[gene] || '#555';
            return (
              <div key={gene} style={{ background: '#fff', border: `2px solid ${color}33`,
                borderRadius: 12, padding: '16px 18px' }}>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: 10 }}>
                  <span style={{ fontSize: 18, fontWeight: 800, color }}>{gene}</span>
                  <span style={{ marginLeft: 8, fontSize: 11, color: '#888' }}>{d.locus}</span>
                  <span style={{ marginLeft: 'auto', fontSize: 11, background: `${color}18`,
                    color, borderRadius: 4, padding: '2px 8px' }}>n={d.n}</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
                  {[
                    ['Avg Age Dx', `${d.avg_age_dx} yr`],
                    ['CR (1st-line)', d.cr_n],
                    ['EBV Positive', d.ebv_positive_n],
                    ['HRD Positive', d.hrd_positive_n],
                    ['Complex K', d.complex_karyotype_n],
                    ['ABC Subtype', d.abc_subtype_n],
                    ['CNS Involved', d.cns_involved_n],
                    ['5yr Alive', d.five_yr_alive_n],
                    ['IVIG Required', d.ivig_required_n],
                    ['Stage 3-4', d.stage_3_4_n],
                    ['Concurrent Breast', d.concurrent_breast_n],
                    ['Avg IPI', d.avg_ipi],
                  ].map(([k, v]) => (
                    <div key={k} style={{ background: '#f8f9fa', borderRadius: 6,
                      padding: '5px 8px', fontSize: 11 }}>
                      <div style={{ color: '#aaa', fontSize: 10 }}>{k}</div>
                      <div style={{ fontWeight: 700, color }}>{v}</div>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: 11, background: '#fff3e0', borderRadius: 6,
                  padding: '6px 10px', color: '#bf360c', marginBottom: 6 }}>
                  <strong>KEY AVOID:</strong> {d.key_avoid?.split(';')[0]}
                </div>
                <div style={{ fontSize: 11, background: '#e8f5e9', borderRadius: 6,
                  padding: '6px 10px', color: '#1b5e20', marginBottom: 6 }}>
                  <strong>KEY RULE:</strong> {d.key_rule?.split(';')[0]}
                </div>
                <div style={{ fontSize: 11, background: '#e3f2fd', borderRadius: 6,
                  padding: '6px 10px', color: '#1a237e' }}>
                  <strong>Rx:</strong> {d.targeted_rx?.split(';')[0]}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          {/* Key Clinical Distinctions */}
          {(definitions.key_clinical_distinctions || []).length > 0 && (
            <div style={{ background: '#fff', borderRadius: 12, padding: '18px 22px', marginBottom: 20 }}>
              <div style={{ fontSize: 15, fontWeight: 800, color: '#b71c1c', marginBottom: 12 }}>
                Key Clinical Distinctions — Hereditary DLBCL Predisposition
              </div>
              {definitions.key_clinical_distinctions.map((d, i) => (
                <div key={i} style={{ background: '#fce4ec', borderLeft: '4px solid #b71c1c',
                  borderRadius: 6, padding: '10px 14px', marginBottom: 8, fontSize: 12, color: '#4a148c' }}>
                  {d}
                </div>
              ))}
            </div>
          )}

          {/* Gene Definitions */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(380px,1fr))', gap: 14 }}>
            {(definitions.definitions || []).map((def, i) => {
              const gene = def.term?.split(' — ')[0];
              const color = GENE_COLORS[gene] || '#555';
              const expanded = expandedDef === i;
              return (
                <div key={i} style={{ background: '#fff', border: `2px solid ${color}22`,
                  borderRadius: 10, padding: '14px 18px', cursor: 'pointer' }}
                  onClick={() => setExpandedDef(expanded ? null : i)}>
                  <div style={{ fontWeight: 700, color, marginBottom: 6, fontSize: 13 }}>
                    {def.term}
                  </div>
                  {expanded && (
                    <pre style={{ fontSize: 11, color: '#555', whiteSpace: 'pre-wrap',
                      background: '#f8f9fa', borderRadius: 6, padding: '10px', marginTop: 8 }}>
                      {def.definition}
                    </pre>
                  )}
                  {!expanded && (
                    <div style={{ fontSize: 11, color: '#aaa' }}>Click to expand definition…</div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
