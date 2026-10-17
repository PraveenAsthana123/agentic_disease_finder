'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-nk-t-cell-lymphoma-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'SH2D1A':  '#1a237e',  // deep indigo      — XLP1; iNKT absent PATHOGNOMONIC; EBV-NK-T 50-80x HIGHEST
  'XIAP':    '#880e4f',  // deep crimson      — XLP2; IBD+HLH COMBINATION PATHOGNOMONIC; NOT EBV-exclusive
  'CD27':    '#004d40',  // deep teal         — CD27 deficiency; EBV-NK-T 30-50x; Rituximab INEFFECTIVE
  'ITK':     '#bf360c',  // deep burnt orange — ITK deficiency; EBV-NK-T 90pct; HLH 50pct
  'GATA2':   '#006064',  // dark cyan         — NK-lymphopenia PATHOGNOMONIC; EBV-CMV-Q3M MANDATORY
  'PIK3CD':  '#4a148c',  // deep purple       — APDS1; Leniolisib FDA2023; Live-vaccines CI
  'STAT5B':  '#33691e',  // olive green       — NK-LGL/NK-T; Cytopenia HALLMARK; Ruxolitinib
  'RASGRP1': '#e65100',  // deep amber        — RASGRP1 deficiency; EBV-T-NK-LPD; HSCT beneficial
};

const GENE_INFO = {
  'SH2D1A':  { full: 'XLP1-SAP-XLR-MALES-ONLY / iNKT-absent-PATHOGNOMONIC / EBV-NK-T-LPD-50-80x-HIGHEST / HSCT-MANDATORY-BEFORE-EBV / HLH-2004-etoposide', locus: 'Xq25', size: '128 aa / 15 kDa', inh: 'XLR LOF; males only' },
  'XIAP':    { full: 'XLP2-BIRC4-XLR-MALES-ONLY / IBD+HLH-COMBINATION-PATHOGNOMONIC / NOT-EBV-exclusive / Ruxolitinib-bridge-HSCT / SMAC-mimetics-experimental', locus: 'Xq25', size: '497 aa / 56 kDa', inh: 'XLR LOF; males only' },
  'CD27':    { full: 'EBV-NK-T-LPD-30-50x-HIGHEST / HLH-45pct / Rituximab-INEFFECTIVE-NK-CD20-negative / IVIG-mandatory-IgG<4 / HSCT-MANDATORY', locus: '12p13.31', size: '260 aa / 29 kDa', inh: 'AR LOF biallelic' },
  'ITK':     { full: 'EBV-NK-T-LPD-90pct / HLH-50pct / Rituximab-INEFFECTIVE-T-NK-cell / HSCT-MANDATORY-BEFORE-EBV / L-asparaginase-SMILE-DDGP', locus: '5q33.3', size: '620 aa / 72 kDa', inh: 'AR LOF biallelic' },
  'GATA2':   { full: 'NK-lymphopenia-<0.05-PATHOGNOMONIC / EBV-CMV-Q3M-MANDATORY / NK-T-lymphoma-EBV-CMV-driven / HSCT-CURATIVE-DO-NOT-DELAY / MDS-30-70x', locus: '3q21.3', size: '480 aa / 50 kDa', inh: 'AD LOF heterozygous' },
  'PIK3CD':  { full: 'APDS1-EBV-NK-T-lymphoma-10-15pct / Leniolisib-FDA2023-FIRST-LINE / LIVE-VACCINES-CI-ABSOLUTELY / Acyclovir-MANDATORY / IVIG-IgG<4', locus: '1p36.22', size: '1044 aa / 119 kDa', inh: 'AD GOF heterozygous' },
  'STAT5B':  { full: 'NK-LGL-NK-T-lymphoma / CYTOPENIA-HALLMARK / Ruxolitinib-JAKi-FIRST-LINE / METHOTREXATE-AVOID / LGL>2x10-9-L-DIAGNOSTIC', locus: '17q21.2', size: '787 aa / 92 kDa', inh: 'AD GOF heterozygous' },
  'RASGRP1': { full: 'EBV-T-NK-LPD-RAS-MAPK / SLE-like-autoimmunity / NK-cell-dysfunction / HSCT-BENEFICIAL / Ruxolitinib-cytokine-storm', locus: '15q14', size: '795 aa / 90 kDa', inh: 'AR LOF biallelic' },
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
          <div style={{ fontWeight: 600 }}>{geneAtlas.syndrome || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      <div style={{ marginBottom: 8 }}>
        <Badge text={`NK/T risk: ${(geneAtlas.nkt_risk || '').split(';')[0]}`} color={bg} />
      </div>
      {geneAtlas.pathognomonic && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ⚠ <b>Pathognomonic:</b> {geneAtlas.pathognomonic}
        </div>
      )}
      <div style={{ fontSize: 12, background: '#e8f5e9', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
        ✅ <b>Key rule:</b> {geneAtlas.key_rule}
      </div>
      <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
        🚫 <b>Avoid:</b> {geneAtlas.key_avoid}
      </div>
      {cohort.n && (
        <div style={{ display: 'flex', gap: 12, fontSize: 12, flexWrap: 'wrap', marginTop: 8 }}>
          <span>n={cohort.n}</span>
          <span>Mean age: {cohort.mean_age}yr</span>
          <span>Stage IV: {cohort.stage_iv}</span>
          <span>EBER+: {cohort.eber_pos}</span>
          <span>L-asp: {cohort.tx_l_asp}</span>
          <span>HSCT: {cohort.hsct_performed}</span>
          <span>HLH: {cohort.hlh_count}</span>
          <span>CR: {Math.round(cohort.cr_rate * 100)}%</span>
          <span>OS-2yr: {Math.round(cohort.os_2yr * 100)}%</span>
        </div>
      )}
    </div>
  );
}

export default function HereditaryNKTCellLymphomaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = async (endpoint, setter) => {
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${API}/api/${SLUG}/${endpoint}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setter(await r.json());
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  useEffect(() => {
    if (tab === 'Overview' && !overview) fetchData('overview', setOverview);
    if ((tab === 'Gene Table' || tab === 'Clinical Atlas') && !breakdown) fetchData('breakdown', setBreakdown);
    if (tab === 'Definitions' && !definitions) fetchData('definitions', setDefinitions);
  }, [tab]);

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto', fontFamily: 'sans-serif' }}>
      <h1 style={{ fontSize: 22, marginBottom: 4 }}>🧬 Hereditary NK/T-Cell Lymphoma Predisposition Atlas</h1>
      <div style={{ color: '#555', fontSize: 13, marginBottom: 16 }}>
        Complete 8-Gene Reference · SH2D1A · XIAP · CD27 · ITK · GATA2 · PIK3CD · STAT5B · RASGRP1 ·
        320 patients (8×40) · Seeds 3654–3661 · EBV-driven NK/T-cell lymphoproliferative disorders
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#1a237e' : '#eee', color: tab === t ? '#fff' : '#333',
            fontWeight: tab === t ? 700 : 400
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#888' }}>Loading…</div>}
      {error && <div style={{ color: 'red' }}>Error: {error}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px,1fr))', gap: 12, marginBottom: 20 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Genes', value: overview.n_genes },
              { label: 'Mean Age', value: `${overview.mean_age_overall} yr` },
              { label: 'EBER+ Rate', value: `${overview.ebv_distribution?.eber_pos_pct}%` },
              { label: 'Avg CR', value: `${overview.outcomes?.avg_cr_pct}%` },
              { label: 'Avg OS-2yr', value: `${overview.outcomes?.avg_os_2yr_pct}%` },
              { label: 'HLH Events', value: overview.outcomes?.total_hlh },
              { label: 'HSCT Done', value: `${overview.outcomes?.hsct_done_pct}%` },
            ].map(m => (
              <div key={m.label} style={{ background: '#f5f5f5', borderRadius: 8, padding: 14, textAlign: 'center' }}>
                <div style={{ fontSize: 22, fontWeight: 700, color: '#1a237e' }}>{m.value}</div>
                <div style={{ fontSize: 12, color: '#666' }}>{m.label}</div>
              </div>
            ))}
          </div>

          <h3 style={{ fontSize: 15, marginBottom: 8 }}>Stage Distribution (Ann Arbor)</h3>
          <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
            {['stage_i','stage_ii','stage_iii','stage_iv'].map(s => (
              <div key={s} style={{ background: '#e8eaf6', borderRadius: 6, padding: '8px 14px', fontSize: 13 }}>
                <b>{s.replace('_',' ').toUpperCase()}</b>: {overview.stage_distribution?.[s]}
              </div>
            ))}
          </div>

          <h3 style={{ fontSize: 15, marginBottom: 8 }}>Per-Gene Summary</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1a237e', color: '#fff' }}>
                  {['Gene','Locus','n','Mean Age','Stage IV%','EBER+%','L-Asp%','HSCT%','CR%','OS-2yr%','HLH','Key Rule'].map(h => (
                    <th key={h} style={{ padding: '6px 8px', textAlign: 'left' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview.per_gene || []).map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff' : '#f9f9f9' }}>
                    <td style={{ padding: '5px 8px', fontWeight: 700, color: GENE_COLORS[g.gene] }}>{g.gene}</td>
                    <td style={{ padding: '5px 8px' }}>{g.locus}</td>
                    <td style={{ padding: '5px 8px' }}>{g.n}</td>
                    <td style={{ padding: '5px 8px' }}>{g.mean_age}</td>
                    <td style={{ padding: '5px 8px' }}>{g.stage_iv_pct}%</td>
                    <td style={{ padding: '5px 8px' }}>{g.eber_pos_pct}%</td>
                    <td style={{ padding: '5px 8px' }}>{g.l_asp_pct}%</td>
                    <td style={{ padding: '5px 8px' }}>{g.hsct_pct}%</td>
                    <td style={{ padding: '5px 8px' }}>{g.cr_pct}%</td>
                    <td style={{ padding: '5px 8px' }}>{g.os_2yr_pct}%</td>
                    <td style={{ padding: '5px 8px' }}>{g.hlh_count}</td>
                    <td style={{ padding: '5px 8px', fontSize: 11 }}>{(g.key_rule || '').split(';')[0]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3 style={{ fontSize: 15, marginTop: 20, marginBottom: 8 }}>Key Clinical Distinctions</h3>
          <ul style={{ paddingLeft: 20 }}>
            {(overview.key_clinical_distinctions || []).map((d, i) => (
              <li key={i} style={{ fontSize: 12, marginBottom: 6, color: '#333' }}>{d}</li>
            ))}
          </ul>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <h3 style={{ fontSize: 15, marginBottom: 12 }}>Gene-Level Atlas Data</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1a237e', color: '#fff' }}>
                  {['Gene','Locus','Inheritance','Syndrome','NK/T Risk','Pathognomonic','Key Rule','Key Avoid','Surveillance','Targeted Rx'].map(h => (
                    <th key={h} style={{ padding: '6px 8px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.atlas_genes || []).map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff' : '#f5f5f5', verticalAlign: 'top' }}>
                    <td style={{ padding: '5px 8px', fontWeight: 700, color: GENE_COLORS[g.gene], whiteSpace: 'nowrap' }}>{g.gene}</td>
                    <td style={{ padding: '5px 8px', whiteSpace: 'nowrap' }}>{g.locus}</td>
                    <td style={{ padding: '5px 8px', whiteSpace: 'nowrap' }}>{g.inheritance}</td>
                    <td style={{ padding: '5px 8px' }}>{g.syndrome}</td>
                    <td style={{ padding: '5px 8px' }}>{g.nkt_risk}</td>
                    <td style={{ padding: '5px 8px' }}>{g.pathognomonic}</td>
                    <td style={{ padding: '5px 8px' }}>{g.key_rule}</td>
                    <td style={{ padding: '5px 8px' }}>{g.key_avoid}</td>
                    <td style={{ padding: '5px 8px' }}>{g.surveillance}</td>
                    <td style={{ padding: '5px 8px' }}>{g.targeted_rx}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          <h3 style={{ fontSize: 15, marginBottom: 12 }}>Clinical Atlas — Gene Cards</h3>
          {Object.entries(GENE_COLORS).map(([gene, color]) => (
            <GeneCard key={gene} gene={gene} color={color} info={GENE_INFO[gene]} breakdown={breakdown} />
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ fontSize: 15, marginBottom: 12 }}>Clinical Definitions</h3>
          {(definitions.definitions || []).map((d, i) => (
            <div key={i} style={{ marginBottom: 14, padding: '10px 14px', background: '#f9f9f9', borderRadius: 6, borderLeft: '4px solid #1a237e' }}>
              <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4, color: '#1a237e' }}>{d.term}</div>
              <div style={{ fontSize: 12, color: '#444', lineHeight: 1.6 }}>{d.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
