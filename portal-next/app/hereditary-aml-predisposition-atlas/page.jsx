'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-aml-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'CEBPA':  '#1a237e',  // deep navy      — Familial AML biallelic ELN-fav; CR 90%+; relapse = new clone; HSCT not mandatory CR1
  'RUNX1':  '#4a148c',  // deep purple    — FPD/AML; thrombocytopenia PATHOGNOMONIC; AML/MDS 30-50%; sibling donor CI HSCT
  'GATA2':  '#006064',  // dark teal      — GATA2 deficiency; MonoMAC; lymphoedema PATHOGNOMONIC; del7q; HSCT curative
  'DDX41':  '#1b5e20',  // deep green     — Familial MDS/AML; late onset 55-65yr HALLMARK; R525H; HMA+venetoclax preferred
  'TP53':   '#e65100',  // deep orange    — LFS; AML 10-20x; complex karyotype; AVOID RADIATION ABSOLUTELY; WBMRI Toronto
  'BRCA2':  '#880e4f',  // deep magenta   — HBOC-2/FA-D1; AML/ALL HIGHEST biallelic; AVOID ALKYLATING biallelic; olaparib HRD
  'FANCA':  '#bf360c',  // deep red-brown — FA-A most common 65%; AML 50x biallelic; AVOID ALKYLATING; fludarabine HSCT
  'NF1':    '#b71c1c',  // deep red       — NF1 syndrome; JMML 200-500x HIGHEST; café-au-lait PATHOGNOMONIC; selumetinib FDA2020
};

const GENE_INFO = {
  'CEBPA':  { full: 'Familial-AML-CEBPA / Biallelic-ELN-Favourable / CR-90pct / Relapse-New-Clone-Re-Salvageable / HSCT-Not-Mandatory-CR1',                          locus: '19q13.11', size: '358 aa / 42 kDa',   inh: 'AD biallelic GOF/LOF' },
  'RUNX1':  { full: 'FPD-AML / Thrombocytopenia-PATHOGNOMONIC-50-150 / AML-MDS-30-50pct-Lifetime / Sibling-Donor-CI-HSCT / RUNX1-RUNX1T1-t8-21-Somatic-DISTINCT',   locus: '21q22.12', size: '453 aa / 51 kDa',   inh: 'AD LOF' },
  'GATA2':  { full: 'GATA2-Deficiency / MonoMAC-Monocytopenia-NK-B-Lymphopenia / Lymphoedema-PATHOGNOMONIC / del7q-Monosomy7 / HSCT-Curative / HPV-Vaccine-MANDATORY', locus: '3q21.3',   size: '480 aa / 50 kDa',   inh: 'AD LOF' },
  'DDX41':  { full: 'Familial-MDS-AML-DDX41 / Late-Onset-55-65yr-HALLMARK / R525H-Most-Common-European / Biallelic-Somatic-10-20pct / HMA-Venetoclax-Preferred',      locus: '5q35.3',   size: '622 aa / 70 kDa',   inh: 'AD LOF' },
  'TP53':   { full: 'LFS / AML-10-20x / Complex-Karyotype-ELN-Adverse / AVOID-RADIATION-ABSOLUTELY / WBMRI-Toronto / Decitabine-APR-246-Eprenetapopt',                locus: '17p13.1',  size: '393 aa / 43 kDa',   inh: 'AD LOF' },
  'BRCA2':  { full: 'HBOC-2-Monoallelic / FA-D1-Biallelic / AML-ALL-HIGHEST-Biallelic / AVOID-ALKYLATING-ABSOLUTELY-Biallelic / Olaparib-HRD-FDA2020 / Bilateral-Wilms-PATHOGNOMONIC-FA-D1', locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF / AR biallelic' },
  'FANCA':  { full: 'FA-A-Most-Common-65pct / AML-MDS-50x-Biallelic-HIGHEST / AVOID-ALKYLATING-ABSOLUTELY / Fludarabine-RIC-HSCT / DEB-MMC-Test / Cafe-au-Lait-Thumb-PATHOGNOMONIC', locus: '16q24.3', size: '1455 aa / 163 kDa', inh: 'AR LOF' },
  'NF1':    { full: 'NF1-Syndrome / JMML-200-500x-HIGHEST-Paediatric / AML-5-10x-Adult / Cafe-au-Lait-6-PATHOGNOMONIC / Selumetinib-FDA2020 / JMML-HSCT-Curative',   locus: '17q11.2',  size: '2839 aa / 327 kDa', inh: 'AD LOF' },
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

export default function HereditaryAMLPredispositionAtlasPage() {
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
    fontSize: 13, background: tab === t ? '#1a237e' : '#f5f5f5', color: tab === t ? '#fff' : '#555',
    marginRight: 6,
  });

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary AML Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'Inter,system-ui,sans-serif', background: '#f8f9fa', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#283593 60%,#3949ab 100%)', color: '#fff', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 11, opacity: 0.75, marginBottom: 4 }}>Hereditary Cancer Predisposition Atlases</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>Hereditary AML Predisposition Atlas</h1>
        <div style={{ fontSize: 12, opacity: 0.85, marginTop: 6 }}>
          8-Gene AML/MDS/JMML Predisposition Reference — 320-Patient Aggregate (seeds 3566-3573)
        </div>
        <div style={{ fontSize: 11, opacity: 0.7, marginTop: 4 }}>
          CEBPA · RUNX1 · GATA2 · DDX41 · TP53 · BRCA2 · FANCA · NF1
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 14, flexWrap: 'wrap' }}>
          <Badge text="CEBPA biallelic ELN-fav CR 90%" color="#fff" />
          <Badge text="DDX41 late-onset 55-65yr" color="#fff" />
          <Badge text="GATA2 MonoMAC lymphoedema" color="#fff" />
          <Badge text="NF1 JMML 200-500x" color="#fff" />
          <Badge text="FA AVOID ALKYLATING ABSOLUTELY" color="#fff" />
          <Badge text="OLAPARIB HRD BRCA2/FANCA" color="#fff" />
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#fff', padding: '12px 32px', borderBottom: '1px solid #e8e8e8' }}>
        {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      <div style={{ padding: '24px 32px' }}>

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
              {card('Total Patients', overview.total_patients, `${overview.genes_n} genes × 40 pts`, '#1a237e')}
              {card('AML Events', `${overview.aml_total_n} (${overview.aml_total_pct}%)`, 'across 8 germline genes', '#c62828')}
              {card('MDS Events', `${overview.mds_total_n} (${overview.mds_total_pct}%)`, 'myelodysplastic syndrome', '#6a1b9a')}
              {card('JMML Events', `${overview.jmml_total_n} (${overview.jmml_total_pct}%)`, 'juvenile myelomonocytic', '#1b5e20')}
              {card('ELN-Favourable', `${overview.eln_fav_n} (${overview.eln_fav_pct}%)`, 'CEBPA biallelic dominant', '#1565c0')}
              {card('Complex Karyotype', `${overview.complex_karyotype_n} (${overview.complex_karyotype_pct}%)`, '≥3 chromosomal abnormalities', '#b71c1c')}
              {card('HRD-Positive', `${overview.hrd_positive_n} (${overview.hrd_positive_pct}%)`, 'BRCA2/FANCA HR-deficient', '#880e4f')}
              {card('HSCT Required', `${overview.hsct_needed_n} (${overview.hsct_needed_pct}%)`, 'allogeneic transplant needed', '#4a148c')}
            </div>
            <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
              {card('Mean Age at Dx', `${overview.mean_age_onset}yr`, 'range: 1yr (NF1-JMML) – 65yr (DDX41)', '#37474f')}
              {card('Biallelic Hit', `${overview.biallelic_hit_n} (${overview.biallelic_hit_pct}%)`, 'germline + somatic second hit', '#006064')}
            </div>

            {/* Pathway categories */}
            {overview.pathway_categories && (
              <div style={{ marginBottom: 20 }}>
                <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 12, color: '#1a237e' }}>Pathway Architecture</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {overview.pathway_categories.map((pw, i) => (
                    <div key={i} style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8, padding: '12px 16px' }}>
                      <div style={{ fontWeight: 700, color: '#1a237e', fontSize: 13, marginBottom: 4 }}>{pw.pathway}</div>
                      <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
                        {pw.genes.map(g => <span key={g} style={{ background: `${GENE_COLORS[g]}18`, color: GENE_COLORS[g], border: `1px solid ${GENE_COLORS[g]}55`, borderRadius: 4, fontSize: 11, padding: '2px 8px', fontWeight: 700 }}>{g}</span>)}
                      </div>
                      <div style={{ fontSize: 12, color: '#555' }}>{pw.mechanism}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Gene summary table */}
            {overview.gene_summary && (
              <div style={{ marginBottom: 20 }}>
                <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 10, color: '#1a237e' }}>Gene Summary</h3>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ background: '#1a237e', color: '#fff' }}>
                        <th style={{ padding: '8px 10px', textAlign: 'left' }}>Gene</th>
                        <th style={{ padding: '8px 10px' }}>Locus</th>
                        <th style={{ padding: '8px 10px' }}>N</th>
                        <th style={{ padding: '8px 10px' }}>AML %</th>
                        <th style={{ padding: '8px 10px' }}>MDS %</th>
                        <th style={{ padding: '8px 10px' }}>JMML %</th>
                        <th style={{ padding: '8px 10px' }}>Mean Age</th>
                        <th style={{ padding: '8px 10px' }}>Inheritance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {overview.gene_summary.map((gs, i) => (
                        <tr key={gs.gene} style={{ background: i % 2 === 0 ? '#f8f9fa' : '#fff' }}>
                          <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[gs.gene] }}>{gs.gene}</td>
                          <td style={{ padding: '7px 10px', textAlign: 'center', fontFamily: 'monospace', fontSize: 11 }}>{gs.locus}</td>
                          <td style={{ padding: '7px 10px', textAlign: 'center' }}>{gs.n}</td>
                          <td style={{ padding: '7px 10px', textAlign: 'center', color: '#c62828', fontWeight: 600 }}>{gs.aml_pct}%</td>
                          <td style={{ padding: '7px 10px', textAlign: 'center', color: '#6a1b9a', fontWeight: 600 }}>{gs.mds_pct}%</td>
                          <td style={{ padding: '7px 10px', textAlign: 'center', color: '#1b5e20', fontWeight: 600 }}>{gs.jmml_pct}%</td>
                          <td style={{ padding: '7px 10px', textAlign: 'center' }}>{gs.mean_age_onset}yr</td>
                          <td style={{ padding: '7px 10px', fontSize: 10, color: '#777' }}>{gs.inheritance}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Critical distinctions */}
            {overview.critical_distinctions && (
              <div>
                <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 10, color: '#1a237e' }}>Critical Clinical Distinctions</h3>
                {overview.critical_distinctions.map((d, i) => (
                  <div key={i} style={{ background: '#fff', border: '1px solid #e0e0e0', borderLeft: '4px solid #1a237e', borderRadius: 6, padding: '10px 14px', marginBottom: 8, fontSize: 12, color: '#333', lineHeight: 1.5 }}>{d}</div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && breakdown && (
          <div>
            <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 12, color: '#1a237e' }}>Per-Gene AML/MDS/JMML Breakdown</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#1a237e', color: '#fff' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left' }}>Gene</th>
                    <th style={{ padding: '8px 10px' }}>Locus</th>
                    <th style={{ padding: '8px 10px' }}>N</th>
                    <th style={{ padding: '8px 10px' }}>AML %</th>
                    <th style={{ padding: '8px 10px' }}>MDS %</th>
                    <th style={{ padding: '8px 10px' }}>JMML %</th>
                    <th style={{ padding: '8px 10px' }}>Age (yr)</th>
                    <th style={{ padding: '8px 10px' }}>Cplx Kary</th>
                    <th style={{ padding: '8px 10px' }}>HRD+</th>
                    <th style={{ padding: '8px 10px' }}>Biallelic</th>
                    <th style={{ padding: '8px 10px' }}>HSCT</th>
                    <th style={{ padding: '8px 10px' }}>Seed</th>
                  </tr>
                </thead>
                <tbody>
                  {breakdown.breakdown.map((b, i) => (
                    <tr key={b.gene} style={{ background: i % 2 === 0 ? '#f8f9fa' : '#fff' }}>
                      <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[b.gene] }}>
                        <button onClick={() => setExpandedGene(expandedGene === b.gene ? null : b.gene)}
                          style={{ background: 'none', border: 'none', cursor: 'pointer', fontWeight: 700, color: GENE_COLORS[b.gene], fontSize: 13 }}>
                          {b.gene} {expandedGene === b.gene ? '▲' : '▼'}
                        </button>
                      </td>
                      <td style={{ padding: '7px 10px', textAlign: 'center', fontFamily: 'monospace', fontSize: 11 }}>{b.locus}</td>
                      <td style={{ padding: '7px 10px', textAlign: 'center' }}>{b.n}</td>
                      <td style={{ padding: '7px 10px', textAlign: 'center', color: '#c62828', fontWeight: 600 }}>{b.aml_pct}%</td>
                      <td style={{ padding: '7px 10px', textAlign: 'center', color: '#6a1b9a', fontWeight: 600 }}>{b.mds_pct}%</td>
                      <td style={{ padding: '7px 10px', textAlign: 'center', color: '#1b5e20', fontWeight: 600 }}>{b.jmml_pct}%</td>
                      <td style={{ padding: '7px 10px', textAlign: 'center' }}>{b.mean_age_onset}</td>
                      <td style={{ padding: '7px 10px', textAlign: 'center' }}>{b.complex_karyotype_n}</td>
                      <td style={{ padding: '7px 10px', textAlign: 'center', color: '#880e4f', fontWeight: 600 }}>{b.hrd_positive_n}</td>
                      <td style={{ padding: '7px 10px', textAlign: 'center' }}>{b.biallelic_hit_n}</td>
                      <td style={{ padding: '7px 10px', textAlign: 'center', color: '#4a148c' }}>{b.hsct_needed_n}</td>
                      <td style={{ padding: '7px 10px', textAlign: 'center', fontFamily: 'monospace', fontSize: 10, color: '#999' }}>{b.seed}</td>
                    </tr>
                  ))}
                  {expandedGene && breakdown.breakdown.filter(b => b.gene === expandedGene).map(b => (
                    <tr key={`${b.gene}-detail`}>
                      <td colSpan={12} style={{ padding: '12px 16px', background: `${GENE_COLORS[b.gene]}08`, borderTop: `2px solid ${GENE_COLORS[b.gene]}33` }}>
                        <div style={{ fontWeight: 700, color: GENE_COLORS[b.gene], marginBottom: 8 }}>{b.gene} — Clinical Detail</div>
                        <div style={{ fontSize: 12, marginBottom: 6 }}><strong>Key Rule:</strong> {b.key_rule}</div>
                        <div style={{ fontSize: 12, marginBottom: 6 }}><strong>Pathognomonic:</strong> {b.pathognomonic}</div>
                        <div style={{ fontSize: 12, color: '#555' }}><strong>Surveillance:</strong> {b.surveillance_key}</div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && overview && (
          <div>
            <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 12, color: '#1a237e' }}>Gene Clinical Atlas</h3>
            {overview.genes_detail && overview.genes_detail.map((gd) => (
              <div key={gd.gene} style={{ background: '#fff', border: `2px solid ${GENE_COLORS[gd.gene]}33`, borderRadius: 10, marginBottom: 16, overflow: 'hidden' }}>
                <div style={{ background: `${GENE_COLORS[gd.gene]}12`, padding: '12px 16px', borderBottom: `1px solid ${GENE_COLORS[gd.gene]}22` }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div>
                      <span style={{ fontWeight: 800, fontSize: 16, color: GENE_COLORS[gd.gene] }}>{gd.gene}</span>
                      <span style={{ marginLeft: 10, fontSize: 12, color: '#666' }}>{gd.locus} · {gd.protein_size?.split('/').slice(0,2).join('/').split(' ').slice(0,4).join(' ')}</span>
                      <Badge text={gd.inheritance} color={GENE_COLORS[gd.gene]} />
                    </div>
                    <button onClick={() => setExpandedGene(expandedGene === gd.gene ? null : gd.gene)}
                      style={{ background: GENE_COLORS[gd.gene], color: '#fff', border: 'none', borderRadius: 6, padding: '5px 14px', cursor: 'pointer', fontSize: 12 }}>
                      {expandedGene === gd.gene ? 'Collapse' : 'Expand'}
                    </button>
                  </div>
                  <div style={{ fontSize: 12, color: '#555', marginTop: 6 }}>{gd.syndrome}</div>
                </div>
                {expandedGene === gd.gene && (
                  <div style={{ padding: '14px 16px' }}>
                    <div style={{ marginBottom: 10 }}>
                      <div style={{ fontWeight: 700, color: GENE_COLORS[gd.gene], fontSize: 12, marginBottom: 4 }}>AML/MDS Risk</div>
                      <div style={{ fontSize: 12, color: '#333', lineHeight: 1.5 }}>{gd.cancer_risk}</div>
                    </div>
                    <div style={{ marginBottom: 10 }}>
                      <div style={{ fontWeight: 700, color: '#c62828', fontSize: 12, marginBottom: 4 }}>Pathognomonic Signs</div>
                      <div style={{ fontSize: 12, color: '#333', lineHeight: 1.5 }}>{gd.pathognomonic}</div>
                    </div>
                    <div style={{ marginBottom: 10 }}>
                      <div style={{ fontWeight: 700, color: '#1b5e20', fontSize: 12, marginBottom: 4 }}>Key Rule</div>
                      <div style={{ fontSize: 12, color: '#333', lineHeight: 1.5 }}>{gd.key_rule}</div>
                    </div>
                    <div>
                      <div style={{ fontWeight: 700, color: '#b71c1c', fontSize: 12, marginBottom: 4 }}>Key Avoid</div>
                      <div style={{ fontSize: 12, color: '#333', lineHeight: 1.5 }}>{gd.key_avoid}</div>
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
            <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 12, color: '#1a237e' }}>Clinical Definitions</h3>
            {definitions.key_clinical_distinctions && (
              <div style={{ marginBottom: 20 }}>
                <h4 style={{ fontWeight: 700, fontSize: 13, color: '#555', marginBottom: 8 }}>Key Clinical Distinctions</h4>
                {definitions.key_clinical_distinctions.map((d, i) => (
                  <div key={i} style={{ background: '#fff', border: '1px solid #e0e0e0', borderLeft: '4px solid #1a237e', borderRadius: 6, padding: '10px 14px', marginBottom: 8, fontSize: 12, color: '#333', lineHeight: 1.5 }}>{d}</div>
                ))}
              </div>
            )}
            {definitions.definitions && definitions.definitions.map((d, i) => (
              <div key={i} style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8, marginBottom: 10, overflow: 'hidden' }}>
                <div style={{ background: '#f5f5f5', padding: '10px 14px', fontWeight: 700, fontSize: 13, cursor: 'pointer', display: 'flex', justifyContent: 'space-between' }}
                  onClick={() => setExpandedDef(expandedDef === i ? null : i)}>
                  <span>{d.term}</span>
                  <span>{expandedDef === i ? '▲' : '▼'}</span>
                </div>
                {expandedDef === i && (
                  <div style={{ padding: '12px 16px', fontSize: 12, color: '#333', whiteSpace: 'pre-line', lineHeight: 1.6 }}>{d.definition}</div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
