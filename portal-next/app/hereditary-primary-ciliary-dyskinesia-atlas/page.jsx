'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-primary-ciliary-dyskinesia-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'DNAH5':  '#1a237e',  // deep navy          -- most common; absent ODA PATHOGNOMONIC
  'DNAI1':  '#880e4f',  // deep crimson        -- CILD1; absent ODA
  'CCDC39': '#1b5e20',  // forest green        -- CILD14; IDA+MT; most severe
  'CCDC40': '#4e342e',  // dark brown          -- CILD15; IDA+MT
  'DNAI2':  '#006064',  // dark teal           -- CILD9; ODA defect
  'DNAH11': '#b71c1c',  // deep red            -- CILD7; NORMAL TEM PATHOGNOMONIC
  'RSPH4A': '#bf360c',  // deep orange-red     -- CILD11; central pair; situs RARE
  'CCNO':   '#4a148c',  // deep purple         -- RGMC; reduced cilia number
};

const GENE_INFO = {
  'DNAH5':  { full: 'Dynein-Heavy-Chain-5 / Most-Common-PCD-28-38pct / Absent-ODA-TEM-PATHOGNOMONIC / Situs-Inversus-48pct / Kartagener', locus: '5p15.2', size: '4624 aa / 512 kDa', inh: 'AR LOF' },
  'DNAI1':  { full: 'Dynein-Intermediate-Chain-1 / CILD1-9-16pct / Absent-ODA / Situs-Inversus-43pct / IVS1+2_1G>A-Founder', locus: '9p21.2', size: '699 aa / 79 kDa', inh: 'AR LOF' },
  'CCDC39': { full: 'Coiled-Coil-39 / CILD14 / IDA+MT-Transposition / Most-Severe-Bronchiectasis / Lowest-FEV1', locus: '3q26.33', size: '941 aa / 107 kDa', inh: 'AR LOF' },
  'CCDC40': { full: 'Coiled-Coil-40 / CILD15 / IDA+MT-Defect / Overlaps-CCDC39 / Same-Treatment', locus: '17q25.3', size: '1092 aa / 122 kDa', inh: 'AR LOF' },
  'DNAI2':  { full: 'Dynein-Intermediate-Chain-2 / CILD9 / Absent-Short-ODA / Clinically-Identical-DNAI1', locus: '17p13.1', size: '612 aa / 70 kDa', inh: 'AR LOF' },
  'DNAH11': { full: 'Dynein-Heavy-Chain-11 / CILD7 / NORMAL-TEM-PATHOGNOMONIC / Hyperkinetic-Stiff-Cilia / Waveform-Mandatory', locus: '7p21.3', size: '4624 aa / 520 kDa', inh: 'AR LOF' },
  'RSPH4A': { full: 'Radial-Spoke-Head-4A / CILD11 / Central-Pair-Defect / Situs-Inversus-RARE-lt5pct / Milder-Lung-Disease', locus: '6q22.1', size: '703 aa / 78 kDa', inh: 'AR LOF' },
  'CCNO':   { full: 'Cyclin-O / RGMC / Reduced-Cilia-Number / TEM-Near-Normal / Female-Subfertility-65pct', locus: '5q11.2', size: '269 aa / 31 kDa', inh: 'AR LOF' },
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
        <Badge text={`nNO: ${cohort.mean_nno_nl_min} nL/min`} color={bg} />
        <Badge text={`FEV1: ${cohort.mean_fev1_pct}%`} color={bg} />
        <Badge text={`Situs: ${cohort.situs_inversus_pct}%`} color={bg} />
        <Badge text={`TEM: ${geneAtlas.defect}`} color='#555' />
      </div>
      {geneAtlas.pathognomonic && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ⚠ <b>Pathognomonic:</b> {geneAtlas.pathognomonic}
        </div>
      )}
      {geneAtlas.key_rule && (
        <div style={{ fontSize: 12, background: '#e8f5e9', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          📋 <b>Key Rule:</b> {geneAtlas.key_rule}
        </div>
      )}
      {cohort.n && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8, marginTop: 8 }}>
          {[
            { label: 'Patients', val: cohort.n },
            { label: 'Mean Age Dx', val: cohort.mean_age_dx ? `${cohort.mean_age_dx} yr` : '—' },
            { label: '% Female', val: cohort.pct_female ? `${Math.round(cohort.pct_female * 100)}%` : '—' },
            { label: 'nNO (nL/min)', val: cohort.mean_nno_nl_min || '—' },
            { label: 'FEV1%', val: cohort.mean_fev1_pct ? `${cohort.mean_fev1_pct}%` : '—' },
            { label: 'Bronchiectasis', val: cohort.bronchiectasis_pct ? `${cohort.bronchiectasis_pct}%` : '—' },
          ].map(s => (
            <div key={s.label} style={{ background: '#f5f5f5', padding: '6px 8px', borderRadius: 4, textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#666' }}>{s.label}</div>
              <div style={{ fontWeight: 700, fontSize: 14 }}>{s.val}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function HeredPCDAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefs(df);
    }).catch(e => setError(e.message)).finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading PCD Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 24, fontFamily: 'system-ui,sans-serif' }}>
      <div style={{ background: 'linear-gradient(135deg,#1a237e,#4a148c)', color: '#fff', borderRadius: 10, padding: '20px 24px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>🫁 Hereditary Primary Ciliary Dyskinesia Atlas</h1>
        <div style={{ fontSize: 13, marginTop: 6, opacity: 0.9 }}>
          DNAH5 · DNAI1 · CCDC39 · CCDC40 · DNAI2 · DNAH11 · RSPH4A · CCNO — 320 patients · 8×40 · seeds 3726–3733
        </div>
        <div style={{ fontSize: 12, marginTop: 4, background: 'rgba(255,255,255,0.15)', borderRadius: 4, padding: '4px 10px', display: 'inline-block' }}>
          ⚠ DNAH11: Normal TEM PATHOGNOMONIC — Waveform Analysis Mandatory · nNO &lt;77 nL/min PATHOGNOMONIC · Kartagener triad
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#1a237e' : '#e8eaf6', color: tab === t ? '#fff' : '#333',
            fontWeight: tab === t ? 700 : 400
          }}>{t}</button>
        ))}
      </div>

      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 14, marginBottom: 20 }}>
            {[
              { label: 'Total Patients', val: overview.total_patients, color: '#1a237e' },
              { label: 'Kartagener Eligible', val: `${overview.key_stats?.kartagener_eligible_pct}%`, color: '#4a148c' },
              { label: 'With Bronchiectasis', val: `${overview.key_stats?.bronchiectasis_pct_overall}%`, color: '#1b5e20' },
              { label: 'Genes Covered', val: (overview.genes || []).length, color: '#006064' },
              { label: 'Mean nNO (nL/min)', val: overview.key_stats?.mean_nno_nl_min_overall, color: '#bf360c' },
              { label: 'Mean FEV1%', val: `${overview.key_stats?.mean_fev1_pct_overall}%`, color: '#b71c1c' },
            ].map(s => (
              <div key={s.label} style={{ background: s.color, color: '#fff', borderRadius: 8, padding: '14px 16px', textAlign: 'center' }}>
                <div style={{ fontSize: 11, opacity: 0.85 }}>{s.label}</div>
                <div style={{ fontSize: 24, fontWeight: 800 }}>{s.val}</div>
              </div>
            ))}
          </div>

          <div style={{ background: '#ffebee', border: '2px solid #b71c1c', borderRadius: 8, padding: 14, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, color: '#b71c1c', marginBottom: 4 }}>🚨 Critical PCD Rules</div>
            <div style={{ fontSize: 13 }}>
              <b>nNO Cutoff:</b> {overview.key_stats?.nno_rule}<br/>
              <b>DNAH11 TEM Rule:</b> {overview.key_stats?.dnah11_tem_rule}<br/>
              <b>Kartagener:</b> {overview.key_stats?.kartagener_triad_rule}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 12 }}>
            {(overview.genes || []).map(gene => {
              const gs = overview.gene_summaries?.[gene] || {};
              const color = GENE_COLORS[gene] || '#333';
              return (
                <div key={gene} style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 14 }}>
                  <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 6 }}>
                    <div style={{ background: color, color: '#fff', borderRadius: 5, padding: '4px 12px', fontWeight: 700, fontSize: 16 }}>{gene}</div>
                    <div style={{ fontSize: 12, color: '#555' }}>{gs.locus} · {gs.inheritance} · {gs.prevalence_pct_pcd}% PCD</div>
                  </div>
                  <div style={{ fontSize: 12 }}>{gs.syndrome || ''}</div>
                  <div style={{ fontSize: 11, marginTop: 4, color: '#777' }}>TEM: {gs.defect}</div>
                  {gs.pathognomonic && (
                    <div style={{ fontSize: 11, marginTop: 6, background: '#fff3e0', borderRadius: 3, padding: '3px 6px' }}>
                      ⚠ {gs.pathognomonic}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {tab === 'Gene Table' && breakdown && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene','Locus','Size','Inheritance','TEM Defect','n','Age Dx','nNO','FEV1%','Situs%','Bronch%'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown.gene_data || {}).map(([gene, d], i) => {
                const info = GENE_INFO[gene] || {};
                const color = GENE_COLORS[gene] || '#333';
                return (
                  <tr key={gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff' }}>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color }}>{gene}</td>
                    <td style={{ padding: '7px 10px' }}>{info.locus}</td>
                    <td style={{ padding: '7px 10px' }}>{info.size}</td>
                    <td style={{ padding: '7px 10px' }}>{info.inh}</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{d.tem_defect}</td>
                    <td style={{ padding: '7px 10px' }}>{d.n}</td>
                    <td style={{ padding: '7px 10px' }}>{d.mean_age_dx} yr</td>
                    <td style={{ padding: '7px 10px' }}>{d.mean_nno_nl_min}</td>
                    <td style={{ padding: '7px 10px' }}>{d.mean_fev1_pct}%</td>
                    <td style={{ padding: '7px 10px' }}>{d.situs_inversus_pct}%</td>
                    <td style={{ padding: '7px 10px' }}>{d.bronchiectasis_pct}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {Object.keys(GENE_COLORS).map(gene => (
            <GeneCard
              key={gene}
              gene={gene}
              color={GENE_COLORS[gene]}
              info={GENE_INFO[gene]}
              breakdown={breakdown}
            />
          ))}
        </div>
      )}

      {tab === 'Definitions' && defs && (
        <div>
          <div style={{ background: '#ffebee', border: '2px solid #c62828', borderRadius: 8, padding: 14, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, color: '#c62828', marginBottom: 8 }}>🚨 Critical Clinical Rules</div>
            {Object.entries(defs.critical_clinical_rules || {}).map(([k, v]) => (
              <div key={k} style={{ marginBottom: 8, paddingLeft: 8, borderLeft: '3px solid #e57373' }}>
                <div style={{ fontWeight: 700, color: '#c62828', marginBottom: 3 }}>{k.replace(/-/g,' ')}</div>
                <div style={{ fontSize: 13 }}>{v}</div>
              </div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 14 }}>
            {(defs.definitions || []).map(d => (
              <div key={d.term} style={{ border: '1px solid #e8eaf6', borderRadius: 8, padding: 14 }}>
                <div style={{ fontWeight: 700, color: '#1a237e', marginBottom: 6 }}>{d.term}</div>
                <div style={{ fontSize: 13 }}>{d.definition}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
