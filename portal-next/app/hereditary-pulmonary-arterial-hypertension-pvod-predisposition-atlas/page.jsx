'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-pulmonary-arterial-hypertension-pvod-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'BMPR2':    '#1a237e',  // deep navy          -- most common; RHC mandatory
  'ACVRL1':   '#880e4f',  // deep crimson        -- HHT2; telangiectases PATHOGNOMONIC
  'ENG':      '#1b5e20',  // forest green        -- HHT1; cerebral AVM MRI mandatory
  'SMAD9':    '#4e342e',  // dark brown          -- BMP pathway; rare PAH
  'KCNK3':    '#006064',  // dark teal           -- TASK1 channel; vasoconstriction
  'CAV1':     '#bf360c',  // deep orange-red     -- lipodystrophy PATHOGNOMONIC
  'EIF2AK4':  '#b71c1c',  // deep red            -- PVOD; prostanoids ABSOLUTELY CI
  'TBX4':     '#4a148c',  // deep purple         -- pediatric PAH; small patella PATHOGNOMONIC
};

const GENE_INFO = {
  'BMPR2':   { full: 'BMP-Receptor-Type-2 / Most-Common-Familial-PAH-75pct / RHC-Mandatory-Echo-Insufficient / Penetrance-20pct-F-10pct-M / Sotatercept-FDA2024', locus: '2q33.1', size: '1038 aa / 115 kDa', inh: 'AD LOF' },
  'ACVRL1':  { full: 'ALK1-Activin-Receptor-Like-Kinase-1 / HHT2-Telangiectases-PATHOGNOMONIC / PAH-10pct-HHT2 / pAVM-CT-Mandatory / Bevacizumab-Epistaxis', locus: '12q13.13', size: '503 aa / 57 kDa', inh: 'AD LOF' },
  'ENG':     { full: 'Endoglin / HHT1-Most-Common-HHT / Cerebral-AVM-MRI-Mandatory / pAVM-30-50pct / PAH-2pct / Epistaxis-Childhood-Onset', locus: '9q34.11', size: '658 aa / 70 kDa', inh: 'AD LOF' },
  'SMAD9':   { full: 'SMAD9-MADH9 / BMP-Signalling-Downstream-BMPR2 / Rare-Familial-PAH-1-2pct / Clinically-Identical-BMPR2-PAH', locus: '13q13.3', size: '530 aa / 59 kDa', inh: 'AD LOF' },
  'KCNK3':   { full: 'TASK1-Potassium-Channel / Membrane-Depolarisation-Vasoconstriction / Rare-lt1pct / No-Extrapulmonary-Features / Activators-Investigational', locus: '2p23.3', size: '354 aa / 39 kDa', inh: 'AD LOF' },
  'CAV1':    { full: 'Caveolin-1-Lipid-Raft / Lipodystrophy-PATHOGNOMONIC / Partial-Lipodystrophy-Limb-Fat-Loss / Metabolic-Insulin-Resistance', locus: '7q31.2', size: '178 aa / 22 kDa', inh: 'AD LOF' },
  'EIF2AK4': { full: 'GCN2-Kinase / PVOD-PCH-Biallelic-AR / CT-GGO-Triad-PATHOGNOMONIC / Prostanoids-ABSOLUTELY-CI-Flash-Oedema / Lung-Transplant-Only-Cure', locus: '15q15.1', size: '1649 aa / 185 kDa', inh: 'AR LOF (biallelic)' },
  'TBX4':    { full: 'T-Box-TF-4 / Pediatric-PAH / Small-Patella-PATHOGNOMONIC / ICPS-Coxa-Vara / Better-VRT-Response-vs-BMPR2', locus: '17q23.2', size: '530 aa / 58 kDa', inh: 'AD LOF' },
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
        <Badge text={`PAH/PVOD: ${(geneAtlas.bmf_risk || '').split(';')[0]}`} color={bg} />
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
            { label: 'Mean Age Dx', val: cohort.mean_age ? `${cohort.mean_age} yr` : '—' },
            { label: '% Female', val: cohort.pct_female ? `${Math.round(cohort.pct_female * 100)}%` : '—' },
            { label: 'mPAP (mmHg)', val: cohort.mean_mpap_mmhg || '—' },
            { label: 'PVR (WU)', val: cohort.mean_pvr_wu || '—' },
            { label: '6MWD (m)', val: cohort.mean_6mwd_m || '—' },
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

export default function HeredPAHPVODAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading PAH/PVOD Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 24, fontFamily: 'system-ui,sans-serif' }}>
      <div style={{ background: 'linear-gradient(135deg,#1a237e,#4a148c)', color: '#fff', borderRadius: 10, padding: '20px 24px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>🫁 Hereditary PAH / PVOD Predisposition Atlas</h1>
        <div style={{ fontSize: 13, marginTop: 6, opacity: 0.9 }}>
          BMPR2 · ACVRL1 · ENG · SMAD9 · KCNK3 · CAV1 · EIF2AK4 · TBX4 — 320 patients · 8×40 · seeds 3718–3725
        </div>
        <div style={{ fontSize: 12, marginTop: 4, background: 'rgba(255,255,255,0.15)', borderRadius: 4, padding: '4px 10px', display: 'inline-block' }}>
          ⚠ RHC MANDATORY — Echo alone insufficient · EIF2AK4/PVOD: Prostanoids ABSOLUTELY CI · Sotatercept FDA 2024
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
              { label: 'PAH Patients', val: overview.key_stats?.total_pah_patients, color: '#4a148c' },
              { label: 'PVOD Patients', val: overview.key_stats?.total_pvod_patients, color: '#b71c1c' },
              { label: 'Genes Covered', val: (overview.genes || []).length, color: '#006064' },
              { label: 'Sotatercept FDA', val: overview.key_stats?.sotatercept_fda_year, color: '#1b5e20' },
              { label: 'VRT+ Overall %', val: `${overview.key_stats?.vrt_positive_overall_pct}%`, color: '#bf360c' },
            ].map(s => (
              <div key={s.label} style={{ background: s.color, color: '#fff', borderRadius: 8, padding: '14px 16px', textAlign: 'center' }}>
                <div style={{ fontSize: 11, opacity: 0.85 }}>{s.label}</div>
                <div style={{ fontSize: 24, fontWeight: 800 }}>{s.val}</div>
              </div>
            ))}
          </div>

          <div style={{ background: '#ffebee', border: '2px solid #b71c1c', borderRadius: 8, padding: 14, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, color: '#b71c1c', marginBottom: 4 }}>🚨 Critical PAH/PVOD Rules</div>
            <div style={{ fontSize: 13 }}>
              <b>RHC MANDATORY:</b> {overview.key_stats?.rhc_mandatory_rule}<br/>
              <b>PVOD Safety:</b> {overview.key_stats?.pvod_vasodilator_ci_rule}
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
                {['Gene','Locus','Size','Inheritance','Syndrome','n','Mean Age','% F','mPAP mmHg','PVR WU','6MWD m'].map(h => (
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
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{d.syndrome}</td>
                    <td style={{ padding: '7px 10px' }}>{d.n}</td>
                    <td style={{ padding: '7px 10px' }}>{d.mean_age} yr</td>
                    <td style={{ padding: '7px 10px' }}>{Math.round((d.pct_female || 0) * 100)}%</td>
                    <td style={{ padding: '7px 10px' }}>{d.mean_mpap_mmhg}</td>
                    <td style={{ padding: '7px 10px' }}>{d.mean_pvr_wu}</td>
                    <td style={{ padding: '7px 10px' }}>{d.mean_6mwd_m}</td>
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
