'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-congenital-myasthenic-syndrome-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'CHRNE': '#4a148c',  // deep purple    -- most common AR CMS; slow/fast channel distinction critical
  'COLQ':  '#b71c1c',  // dark red       -- endplate AChE; pupil ABSENT PATHOGNOMONIC; pyridostigmine ABSOLUTELY CI
  'RAPSN': '#1a237e',  // deep navy      -- clustering defect; N88K founder; neonatal apnea
  'DOK7':  '#880e4f',  // deep crimson   -- limb-girdle WITHOUT ptosis PATHOGNOMONIC; pyridostigmine ABSOLUTELY CI; salbutamol FIRST-LINE
  'AGRN':  '#1b5e20',  // dark green     -- distal-predominant ATYPICAL; salbutamol effective
  'MUSK':  '#e65100',  // burnt orange   -- distal+ocular+bulbar; distinguish autoimmune MG
  'SCN4A': '#004d40',  // dark teal      -- Nav1.4 NOT nicotinic; episodic apnea; acetazolamide
  'GFPT1': '#37474f',  // dark blue-grey -- glycosylation CMS; tubular aggregates PATHOGNOMONIC
};

const GENE_INFO = {
  'CHRNE': { full: 'AChR-epsilon / most-common-AR-CMS-40-50% / slow-channel-GOF-quinidine vs fast-channel-LOF-pyridostigmine / single-fiber-EMG-jitter-PATHOGNOMONIC', locus: '17p13.2', size: '493 aa / 54 kDa', inh: 'AR' },
  'COLQ':  { full: 'AChE-collagen-tail / endplate-AChE-deficiency / pupillary-reflex-ABSENT-PATHOGNOMONIC / pyridostigmine-ABSOLUTELY-CI / ephedrine-first-line', locus: '3p25.1', size: '461 aa / 54 kDa', inh: 'AR' },
  'RAPSN': { full: 'Receptor-associated-protein-synapse / AChR-clustering-defect / N88K-Northern-European-founder / neonatal-hypotonia+arthrogryposis / fever-apnea-HALLMARK / pyridostigmine-EFFECTIVE', locus: '11p11.2', size: '412 aa / 43 kDa', inh: 'AR' },
  'DOK7':  { full: 'Downstream-of-kinase-7 / limb-girdle-WITHOUT-ptosis-PATHOGNOMONIC / pyridostigmine-ABSOLUTELY-CI / salbutamol-4-8mg-TID-SPECIFIC-FIRST-LINE / misdiagnosed-LGMD', locus: '4p16.3', size: '504 aa / 56 kDa', inh: 'AR' },
  'AGRN':  { full: 'Agrin / NMJ-organizer / distal-predominant-ATYPICAL / salbutamol-ephedrine-effective / pyridostigmine-variable', locus: '1p36.33', size: '2045 aa / 225 kDa', inh: 'AR' },
  'MUSK':  { full: 'Muscle-specific-kinase / distal+ocular+bulbar-combined / distinguish-autoimmune-anti-MuSK-MG / ephedrine+3-4-DAP-preferred / pyridostigmine-variable', locus: '9q31.3', size: '869 aa / 97 kDa', inh: 'AR' },
  'SCN4A': { full: 'Nav1.4-sodium-channel / NOT-nicotinic-AChR / episodic-apnea / acetazolamide-first-line / pyridostigmine-NOT-indicated / AD-inheritance', locus: '17q23.3', size: '1836 aa / 211 kDa', inh: 'AD' },
  'GFPT1': { full: 'Glutamine-fructose-6-phosphate-transaminase / glycosylation-CMS / tubular-aggregates-biopsy-PATHOGNOMONIC / 3-4-DAP+pyridostigmine-EFFECTIVE / limb-girdle', locus: '2p13.1', size: '699 aa / 79 kDa', inh: 'AR' },
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
        <Badge text={`Resp crisis: ${cohort.respiratory_crisis_n}/40`} color='#6a1b9a' />
        <Badge text={`Ptosis/Oph: ${cohort.ptosis_ophthalmoplegia_n}/40`} color='#1b5e20' />
        <Badge text={`Pyr adverse: ${cohort.pyridostigmine_adverse_n}/40`} color='#b71c1c' />
      </div>
      {gene === 'CHRNE' && (
        <div style={{ fontSize: 12, background: '#ede7f6', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700 }}>
          🧬 <b>SUBTYPE POLARITY CRITICAL:</b> Slow-channel GOF → quinidine/fluoxetine (pyridostigmine CI). Fast-channel LOF → pyridostigmine. {cohort.slow_channel_n}/{cohort.n} slow-channel.
        </div>
      )}
      {gene === 'COLQ' && (
        <div style={{ fontSize: 12, background: '#ffebee', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#b71c1c' }}>
          🚫 <b>PYRIDOSTIGMINE ABSOLUTELY CONTRAINDICATED:</b> Absent pupillary reflex PATHOGNOMONIC ({cohort.pupil_abnormal_n || '~36'}/40). Ephedrine/salbutamol first-line.
        </div>
      )}
      {gene === 'RAPSN' && (
        <div style={{ fontSize: 12, background: '#e3f2fd', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700 }}>
          🧬 <b>N88K founder:</b> {cohort.n88k_n || '~26'}/{cohort.n} N88K positive. Arthrogryposis {cohort.arthrogryposis_n || '~14'}/40. Pyridostigmine EFFECTIVE. Fever → apnea emergency.
        </div>
      )}
      {gene === 'DOK7' && (
        <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#880e4f' }}>
          🚫 <b>PYRIDOSTIGMINE ABSOLUTELY CONTRAINDICATED:</b> {cohort.pyridostigmine_adverse_n}/40 worsened. Limb-girdle WITHOUT ptosis ({cohort.ptosis_ophthalmoplegia_n}/40 ptosis). Salbutamol FIRST-LINE response {cohort.salbutamol_response_n || '~34'}/40. Misdiagnosed LGMD {cohort.misdiagnosed_lgmd_n || '~22'}/40.
        </div>
      )}
      {gene === 'SCN4A' && cohort.episodic_apnea_n > 0 && (
        <div style={{ fontSize: 12, background: '#e0f2f1', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#004d40' }}>
          ⚡ <b>Nav1.4 NOT nicotinic AChR:</b> Episodic apnea {cohort.episodic_apnea_n}/40. Myotonic discharges {cohort.myotonic_discharge_n}/40. Acetazolamide response {cohort.acetazolamide_response_n}/40. Pyridostigmine NOT indicated.
        </div>
      )}
      {gene === 'GFPT1' && cohort.tubular_aggregates_n > 0 && (
        <div style={{ fontSize: 12, background: '#eceff1', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#37474f' }}>
          🔬 <b>TUBULAR AGGREGATES PATHOGNOMONIC:</b> {cohort.tubular_aggregates_n}/40 on biopsy. CK elevated {cohort.ck_elevated_n}/40. 3,4-DAP response {cohort.dap_response_n}/40. Glycosylation CMS — pyridostigmine EFFECTIVE (unlike DOK7/COLQ).
        </div>
      )}
      <div style={{ fontSize: 12, color: '#555', marginTop: 6 }}>
        <b>Key therapy:</b> {geneAtlas.key_therapy || ''}
      </div>
    </div>
  );
}

export default function HereditaryCMSAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); setLoading(false); })
      .catch(e => { setErr(String(e)); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#888' }}>Loading Hereditary CMS Atlas…</div>;
  if (err)     return <div style={{ padding: 40, color: 'red' }}>Error: {err}</div>;

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto' }}>
      <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>
        🧬 Hereditary Congenital Myasthenic Syndrome Atlas
      </h1>
      <div style={{ fontSize: 13, color: '#555', marginBottom: 16 }}>
        Complete 8-Gene CMS Reference · CHRNE · COLQ · RAPSN · DOK7 · AGRN · MUSK · SCN4A · GFPT1 ·
        Seeds {overview?.seeds} · {overview?.total_patients} patients
      </div>

      {/* Treatment polarity alert */}
      <div style={{ background: '#fff3e0', border: '2px solid #e65100', borderRadius: 8, padding: '10px 16px', marginBottom: 16, fontSize: 13 }}>
        <b>⚠️ CRITICAL TREATMENT POLARITY:</b> Pyridostigmine <b>ABSOLUTELY CONTRAINDICATED</b> in DOK7 (→ salbutamol) and COLQ (→ ephedrine) · 
        CONTRAINDICATED in slow-channel CHRNE (→ quinidine) · NOT indicated in SCN4A (→ acetazolamide) · 
        EFFECTIVE in fast-channel CHRNE, RAPSN, GFPT1
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{ padding: '6px 16px', borderRadius: 20, border: 'none', cursor: 'pointer',
              background: tab === t ? '#4a148c' : '#eee', color: tab === t ? '#fff' : '#333',
              fontWeight: tab === t ? 700 : 400, fontSize: 13 }}>
            {t}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Atlas Overview — {overview.genes?.join(' · ')}</h2>
          <p style={{ fontSize: 13, color: '#444', marginBottom: 16, lineHeight: 1.6 }}>{overview.description}</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {overview.gene_summaries?.map(g => (
              <div key={g.gene} style={{
                border: `2px solid ${GENE_COLORS[g.gene] || '#999'}`,
                borderRadius: 8, padding: 12
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <div style={{ background: GENE_COLORS[g.gene] || '#999', color: '#fff',
                    borderRadius: 5, padding: '3px 10px', fontWeight: 700, fontSize: 15 }}>{g.gene}</div>
                  <div style={{ fontSize: 11, color: '#666' }}>{g.locus} · {g.inheritance}</div>
                </div>
                <div style={{ fontSize: 12, color: '#333', marginBottom: 4 }}>{g.syndrome?.split('/').slice(0,2).join(' / ')}</div>
                <div style={{ fontSize: 11, color: '#666' }}>{g.pathognomonic?.substring(0, 100)}…</div>
                <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>{g.protein_size?.split('/').slice(0,2).join(' / ')}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Cohort Summary — 320 Patients (8 × 40)</h2>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#4a148c', color: '#fff' }}>
                {['Gene', 'Locus', 'Inheritance', 'Age Dx (yr)', 'Resp Crisis', 'Ptosis/Oph', 'Pyr Adverse', 'Pathognomonic'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {breakdown.atlas_genes?.map((g, i) => {
                const c = breakdown.gene_data?.[g.gene] || {};
                return (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] }}>{g.gene}</td>
                    <td style={{ padding: '8px 10px' }}>{g.locus}</td>
                    <td style={{ padding: '8px 10px' }}>{g.inheritance}</td>
                    <td style={{ padding: '8px 10px' }}>{c.mean_age_dx}</td>
                    <td style={{ padding: '8px 10px' }}>{c.respiratory_crisis_n}/40</td>
                    <td style={{ padding: '8px 10px' }}>{c.ptosis_ophthalmoplegia_n}/40</td>
                    <td style={{ padding: '8px 10px', color: c.pyridostigmine_adverse_n > 20 ? '#b71c1c' : '#333', fontWeight: c.pyridostigmine_adverse_n > 20 ? 700 : 400 }}>{c.pyridostigmine_adverse_n}/40</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#555' }}>{g.pathognomonic?.substring(0, 70)}…</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Clinical Gene Cards</h2>
          {breakdown.atlas_genes?.map(g => (
            <GeneCard key={g.gene} gene={g.gene} color={GENE_COLORS[g.gene]} info={GENE_INFO[g.gene]} breakdown={breakdown} />
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>CMS Definitions & Critical Rules</h2>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20 }}>
            {Object.entries(definitions.definitions || {}).map(([term, def]) => (
              <div key={term} style={{ border: '1px solid #ddd', borderRadius: 6, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4, color: '#4a148c' }}>{term}</div>
                <div style={{ fontSize: 12, color: '#555', lineHeight: 1.5 }}>{def}</div>
              </div>
            ))}
          </div>
          <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>Critical Clinical Rules ({Object.keys(definitions.critical_clinical_rules || {}).length})</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {Object.keys(definitions.critical_clinical_rules || {}).map(rule => (
              <span key={rule} style={{
                background: rule.includes('CONTRAINDICATED') || rule.includes('ABSOLUTELY') ? '#ffebee' :
                            rule.includes('PATHOGNOMONIC') ? '#e8f5e9' : '#f3e5f5',
                color: rule.includes('CONTRAINDICATED') || rule.includes('ABSOLUTELY') ? '#b71c1c' :
                       rule.includes('PATHOGNOMONIC') ? '#1b5e20' : '#4a148c',
                border: `1px solid ${rule.includes('CONTRAINDICATED') || rule.includes('ABSOLUTELY') ? '#ef9a9a' :
                         rule.includes('PATHOGNOMONIC') ? '#a5d6a7' : '#ce93d8'}`,
                borderRadius: 4, padding: '2px 8px', fontSize: 11
              }}>{rule}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
