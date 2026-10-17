'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-skeletal-muscle-channelopathy-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'SCN4A':   '#1a237e',  // deep navy          -- Nav1.4; HyperKPP/PMC/HypoPP2
  'CACNA1S': '#880e4f',  // deep crimson        -- Cav1.1; HypoPP1-MOST-COMMON; MHS5
  'KCNJ2':   '#1b5e20',  // forest green        -- Kir2.1; ATS-LQT7; TRIAD; bidir VT
  'CLCN1':   '#4e342e',  // dark brown          -- ClC-1; Myotonia Congenita; warm-up
  'KCNJ18':  '#006064',  // dark teal           -- Kir2.6; Thyrotoxic PP; treat thyroid FIRST
  'RYR1':    '#b71c1c',  // deep red            -- RyR1; MHS1/CCD; volatile ABSOLUTE CI
  'DMPK':    '#bf360c',  // deep orange-red     -- CTG repeat; DM1; anticipation PATHOGNOMONIC
  'CNBP':    '#4a148c',  // deep purple         -- CCTG repeat; DM2-PROMM; proximal; no anticipation
};

const GENE_INFO = {
  'SCN4A':   { full: 'Nav1.4 / HyperKPP-type1 / Paramyotonia-Congenita / HypoPP-type2 / Normokalemic-PP / acetazolamide-mexiletine', locus: '17q23.3', size: '1836 aa / 200 kDa', inh: 'AD GOF/LOF' },
  'CACNA1S': { full: 'Cav1.1 / HypoPP-type1-MOST-COMMON / MHS5 / R528H-R1239H-hotspots / volatile+succinylcholine-ABSOLUTE-CI', locus: '1q32.1', size: '1873 aa / 210 kDa', inh: 'AD GOF' },
  'KCNJ2':   { full: 'Kir2.1 / ATS-LQT7 / TRIAD-periodic-paralysis+LQT7+dysmorphia-PATHOGNOMONIC / bidirectional-VT-PATHOGNOMONIC / avoid-class-I', locus: '17q24.3', size: '427 aa / 48 kDa', inh: 'AD LOF' },
  'CLCN1':   { full: 'ClC-1 / Myotonia-Congenita-Thomsen-AD-Becker-AR / warm-up-phenomenon-PATHOGNOMONIC / mexiletine-first-line', locus: '7q34', size: '988 aa / 108 kDa', inh: 'AD (Thomsen) / AR (Becker)' },
  'KCNJ18':  { full: 'Kir2.6 / Thyrotoxic-Periodic-Paralysis / treat-thyrotoxicosis-FIRST / propranolol-acute / NOT-K+-alone', locus: '17p11.2', size: '445 aa / 50 kDa', inh: 'AD LOF' },
  'RYR1':    { full: 'RyR1 / MHS1-CCD-MmD / volatile+succinylcholine-ABSOLUTELY-CI / dantrolene-IV-MANDATORY / IVCT-CHCT-GOLD-STANDARD', locus: '19q13.2', size: '5038 aa / 565 kDa', inh: 'AD (MHS1) / AR (CCD/MmD)' },
  'DMPK':    { full: 'CTG-repeat / DM1-Steinert / anticipation-PATHOGNOMONIC / cardiac-conduction-annual-ECG-Holter-MANDATORY / distal-weakness', locus: '19q13.32', size: '629 aa / 70 kDa', inh: 'AD CTG-repeat' },
  'CNBP':    { full: 'CCTG-repeat / DM2-PROMM / proximal-weakness / NO-anticipation / pain-prominent / milder-cardiac', locus: '19q13.11', size: '629 aa / 53 kDa', inh: 'AD CCTG-repeat' },
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
        <Badge text={`Age Dx: ${cohort.mean_age_dx} yr`} color={bg} />
        <Badge text={`Severity: ${cohort.mean_severity_score}/10`} color={bg} />
        <Badge text={`Dx Delay: ${cohort.mean_dx_delay_years} yr`} color={bg} />
        <Badge text={`Cardiac: ${cohort.cardiac_involvement_pct}%`} color='#555' />
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
      {geneAtlas.trigger && (
        <div style={{ fontSize: 12, background: '#e3f2fd', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ⚡ <b>Triggers:</b> {geneAtlas.trigger}
        </div>
      )}
      {cohort.n && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8, marginTop: 8 }}>
          {[
            { label: 'Patients', val: cohort.n },
            { label: 'Mean Age Dx', val: cohort.mean_age_dx ? `${cohort.mean_age_dx} yr` : '—' },
            { label: '% Female', val: cohort.pct_female ? `${Math.round(cohort.pct_female * 100)}%` : '—' },
            { label: 'Severity /10', val: cohort.mean_severity_score || '—' },
            { label: 'Dx Delay (yr)', val: cohort.mean_dx_delay_years || '—' },
            { label: 'Cardiac %', val: cohort.cardiac_involvement_pct ? `${cohort.cardiac_involvement_pct}%` : '—' },
          ].map(s => (
            <div key={s.label} style={{ background: '#f5f5f5', padding: '6px 8px', borderRadius: 4, textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#666' }}>{s.label}</div>
              <div style={{ fontWeight: 700, fontSize: 14 }}>{s.val}</div>
            </div>
          ))}
        </div>
      )}
      {cohort.clinical_facts && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontWeight: 700, fontSize: 12, color: bg, marginBottom: 4 }}>Clinical Facts</div>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {cohort.clinical_facts.slice(0, 5).map((f, i) => (
              <li key={i} style={{ fontSize: 12, marginBottom: 3 }}>{f}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function HeredSkeletalMuscleAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Skeletal Muscle Channelopathy Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 24, fontFamily: 'system-ui,sans-serif' }}>
      <div style={{ background: 'linear-gradient(135deg,#1a237e,#b71c1c)', color: '#fff', borderRadius: 10, padding: '20px 24px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>🧬 Hereditary Skeletal Muscle Channelopathy &amp; Myotonic Disorder Atlas</h1>
        <div style={{ fontSize: 13, marginTop: 6, opacity: 0.9 }}>
          SCN4A · CACNA1S · KCNJ2 · CLCN1 · KCNJ18 · RYR1 · DMPK · CNBP — 320 patients · 8×40 · seeds 3734–3741
        </div>
        <div style={{ fontSize: 12, marginTop: 4, background: 'rgba(255,255,255,0.15)', borderRadius: 4, padding: '4px 10px', display: 'inline-block' }}>
          ⚠ RYR1/CACNA1S: volatile+succinylcholine ABSOLUTE CI · KCNJ2-ATS: bidirectional VT PATHOGNOMONIC · DMPK-DM1: anticipation PATHOGNOMONIC · annual ECG MANDATORY
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
              { label: 'Genes Covered', val: (overview.genes || []).length, color: '#4a148c' },
              { label: 'Cardiac Involvement', val: `${overview.key_stats?.cardiac_involvement_pct_overall}%`, color: '#b71c1c' },
              { label: 'Respiratory Involvement', val: `${overview.key_stats?.respiratory_pct_overall}%`, color: '#1b5e20' },
              { label: 'Mean Severity /10', val: overview.key_stats?.mean_severity_score_overall, color: '#bf360c' },
              { label: 'Mean Dx Delay (yr)', val: overview.key_stats?.mean_dx_delay_years_overall, color: '#006064' },
            ].map(s => (
              <div key={s.label} style={{ background: s.color, color: '#fff', borderRadius: 8, padding: '14px 16px', textAlign: 'center' }}>
                <div style={{ fontSize: 11, opacity: 0.85 }}>{s.label}</div>
                <div style={{ fontSize: 24, fontWeight: 800 }}>{s.val}</div>
              </div>
            ))}
          </div>

          <div style={{ background: '#ffebee', border: '2px solid #b71c1c', borderRadius: 8, padding: 14, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, color: '#b71c1c', marginBottom: 4 }}>🚨 Critical Safety Rules</div>
            <div style={{ fontSize: 13 }}>
              <b>MH/Anesthesia:</b> {overview.key_stats?.mh_critical_rule}<br/>
              <b>ATS (KCNJ2):</b> {overview.key_stats?.ats_triad_rule}<br/>
              <b>DM1 (DMPK):</b> {overview.key_stats?.dm1_rule}<br/>
              <b>DM2 (CNBP):</b> {overview.key_stats?.dm2_rule}
            </div>
          </div>

          <div style={{ background: '#e8f5e9', border: '2px solid #1b5e20', borderRadius: 8, padding: 14, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, color: '#1b5e20', marginBottom: 4 }}>📋 Myotonia Differentiation Rules</div>
            <div style={{ fontSize: 13 }}>
              <b>Warm-up (CLCN1):</b> {overview.key_stats?.mc_warmup_rule}<br/>
              <b>Thyrotoxic PP (KCNJ18):</b> {overview.key_stats?.tpp_rule}
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
                    <div style={{ fontSize: 12, color: '#555' }}>{gs.locus} · {gs.inheritance}</div>
                  </div>
                  <div style={{ fontSize: 12 }}>{gs.syndrome || ''}</div>
                  <div style={{ fontSize: 11, marginTop: 4, color: '#777' }}>{gs.channel}</div>
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
                {['Gene','Locus','Size','Inheritance','Syndrome','n','Age Dx','Severity','Dx Delay','Cardiac%','Resp%'].map(h => (
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
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{info.inh}</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{d.syndrome}</td>
                    <td style={{ padding: '7px 10px' }}>{d.n}</td>
                    <td style={{ padding: '7px 10px' }}>{d.mean_age_dx} yr</td>
                    <td style={{ padding: '7px 10px' }}>{d.mean_severity_score}/10</td>
                    <td style={{ padding: '7px 10px' }}>{d.mean_dx_delay_years} yr</td>
                    <td style={{ padding: '7px 10px' }}>{d.cardiac_involvement_pct}%</td>
                    <td style={{ padding: '7px 10px' }}>{d.respiratory_pct}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div style={{ marginTop: 20 }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Gene Classification by Channel/Mechanism</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 10 }}>
              {[
                { label: 'Na+ Channel (Nav1.4)', genes: breakdown.cohort_summary?.sodium_channel_genes, color: '#1a237e' },
                { label: 'Ca2+ Channel (Cav1.1)', genes: breakdown.cohort_summary?.calcium_channel_genes, color: '#880e4f' },
                { label: 'K+ Channel (Kir2.x)', genes: breakdown.cohort_summary?.potassium_channel_genes, color: '#1b5e20' },
                { label: 'Cl- Channel (ClC-1)', genes: breakdown.cohort_summary?.chloride_channel_genes, color: '#4e342e' },
                { label: 'Ca2+ Release Channel (RyR1)', genes: breakdown.cohort_summary?.calcium_release_channel_genes, color: '#b71c1c' },
                { label: 'Trinucleotide Repeat (CTG/DM1)', genes: breakdown.cohort_summary?.trinucleotide_repeat_genes, color: '#bf360c' },
                { label: 'Tetranucleotide Repeat (CCTG/DM2)', genes: breakdown.cohort_summary?.tetranucleotide_repeat_genes, color: '#4a148c' },
                { label: 'MH Susceptibility', genes: breakdown.cohort_summary?.mh_susceptibility_genes, color: '#c62828' },
              ].map(row => (
                <div key={row.label} style={{ background: row.color, color: '#fff', borderRadius: 6, padding: '8px 12px' }}>
                  <div style={{ fontSize: 11, opacity: 0.85 }}>{row.label}</div>
                  <div style={{ fontWeight: 700 }}>{(row.genes || []).join(', ')}</div>
                </div>
              ))}
            </div>
          </div>
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
