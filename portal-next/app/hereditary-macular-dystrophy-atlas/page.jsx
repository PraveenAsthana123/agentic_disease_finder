'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-macular-dystrophy-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ABCA4:   '#1a237e',  // deep indigo       — Stargardt STGD1, bull's eye, dark choroid, Vitamin A CI
  BEST1:   '#b71c1c',  // deep red           — BVMD, egg-yolk, EOG Arden ratio, 5-stage
  PRPH2:   '#0277bd',  // steel blue         — Pattern dystrophy, butterfly pigment, AVMD, CACD
  TIMP3:   '#4a148c',  // deep purple        — Sorsby SFD, CNV haemorrhage, Bruch's deposits, Vitamin A +ve
  ELOVL4:  '#e65100',  // burnt orange       — STGD3, AD Stargardt-like, VLC-PUFA, biallelic ichthyosis
  C1QTNF5: '#004d40',  // dark teal          — LORD, iris crystalline deposits PATHOGNOMONIC, late-onset
  PRDM13:  '#1b5e20',  // dark green         — NCMD, non-progressive stationary, grade 0-3, exome misses
  EFEMP1:  '#37474f',  // blue-grey          — DHRD/ML, R345W single variant, honeycomb drusen nasal to disc
};

const GENE_INFO = {
  ABCA4:   { full: 'ABCA4 / ATP-binding cassette transporter A4 / 2273aa', locus: '1p22.1', size: '2273 aa / 250 kDa', inh: 'AR', disease: 'Stargardt disease type 1 (STGD1) — MOST COMMON HEREDITARY MACULAR DYSTROPHY; BULL\'S EYE MACULOPATHY + PISCIFORM FLECKS PATHOGNOMONIC; dark choroid sign on FFA (~80%); VITAMIN A SUPPLEMENTS ABSOLUTELY CONTRAINDICATED (accelerates A2E accumulation); full-field ERG initially NORMAL; onset childhood–early adulthood' },
  BEST1:   { full: 'BEST1 / Bestrophin-1 / 585aa', locus: '11q12.3', size: '585 aa / 68 kDa', inh: 'AD/AR', disease: 'Best vitelliform macular dystrophy (BVMD) — EGG-YOLK FOVEAL LESION PATHOGNOMONIC; EOG ARDEN RATIO <1.5 PATHOGNOMONIC (even in pre-vitelliform/asymptomatic carriers); 5-stage evolution (previtelliform→vitelliform→pseudohypopyon→vitelliruptive→atrophic/CNV); FAF intensely hyperfluorescent; full-field ERG NORMAL; AD + biallelic AR (ARB more severe)' },
  PRPH2:   { full: 'PRPH2 / Peripherin-2 (RDS) / 346aa', locus: '6p21.1', size: '346 aa / 39 kDa', inh: 'AD', disease: 'Pattern dystrophy — BUTTERFLY-SHAPED PIGMENT DYSTROPHY PATHOGNOMONIC; AVMD (adult vitelliform); CACD (central areolar); HIGHLY VARIABLE EXPRESSIVITY (same variant → different phenotypes in one family); digenic RP with ROM1; full-field ERG NORMAL unless RP phenotype; anti-VEGF if CNV; generally benign prognosis' },
  TIMP3:   { full: 'TIMP3 / Tissue inhibitor of metalloproteinase 3 / 211aa', locus: '22q12.3', size: '211 aa / 24 kDa', inh: 'AD', disease: 'Sorsby fundus dystrophy (SFD) — BILATERAL CHOROIDAL NEOVASCULARIZATION + HAEMORRHAGE IN 4TH DECADE PATHOGNOMONIC; Bruch\'s membrane thickening; nyctalopia (dark adaptation impaired); VITAMIN A MAY IMPROVE dark adaptation (OPPOSITE of STGD1); anti-VEGF RESPONSIVE; dominant-negative TIMP3 Cys-domain variants; acute visual loss' },
  ELOVL4:  { full: 'ELOVL4 / Elongation of very-long-chain fatty acids protein 4 / 314aa', locus: '6q14.1', size: '314 aa / 34 kDa', inh: 'AD/AR', disease: 'Stargardt-like macular dystrophy 3 (STGD3) — AD inheritance KEY DDx from ABCA4 (AR); yellow flecks + bull\'s eye, NO dark choroid; Vitamin A NOT contraindicated; VLC-PUFA deficiency mechanism; BIALLELIC (homozygous) = neonatal ichthyosis + seizures + profound ID; SCA34 variant (different alleles) = adult spinocerebellar ataxia' },
  C1QTNF5: { full: 'C1QTNF5 / C1q and TNF-related protein 5 (CTRP5) / 243aa', locus: '11q23.3', size: '243 aa / 27 kDa', inh: 'AD', disease: 'Late-onset retinal degeneration (LORD/L-ORD) — CRYSTALLINE IRIS DEPOSITS (glistening iris stroma crystals) PATHOGNOMONIC AND UNIQUE TO LORD; no other hereditary macular dystrophy causes iris crystals; drusenoid peripapillary + macular deposits; onset 6th decade; p.Ser163Arg (S163R) founder variant; anti-VEGF for CNV; slow progression' },
  PRDM13:  { full: 'PRDM13 / PR/SET domain 13 / 718aa', locus: '6q16.1', size: '718 aa / 81 kDa', inh: 'AD', disease: 'North Carolina Macular Dystrophy (NCMD/MCDR1) — NON-PROGRESSIVE (STATIONARY) MACULAR LESION FROM BIRTH PATHOGNOMONIC; grade 0 (drusen) to grade 3 (coloboma-like staphyloma); full-field ERG NORMAL; DIAGNOSTIC TRAP: coding exome MISSES regulatory variant (promoter/5\'UTR duplication) — request chromosomal microarray or dedicated PRDM13 promoter sequencing; excellent prognosis (non-progressive)' },
  EFEMP1:  { full: 'EFEMP1 / Fibulin-3 / 493aa', locus: '2p16.1', size: '493 aa / 55 kDa', inh: 'AD', disease: 'Doyne honeycomb retinal dystrophy (DHRD) / Malattia Leventinese (ML) — RADIAL HONEYCOMB DRUSEN NASAL TO OPTIC DISC PATHOGNOMONIC; p.Arg345Trp (R345W) = >95% of cases (single variant); drusen from 2nd-3rd decade (age-inappropriate); CNV ~50% by 7th decade — anti-VEGF responsive; complement dysregulation via FHL-1; two historic European families (Oxford + Swiss Alps) = same variant' },
};

function GeneChip({ gene }) {
  return (
    <span style={{
      background: GENE_COLORS[gene] || '#555',
      color: '#fff',
      borderRadius: 4,
      padding: '2px 8px',
      fontSize: 12,
      fontWeight: 700,
      marginRight: 4,
      display: 'inline-block',
    }}>{gene}</span>
  );
}

function MetricCard({ label, value, sub }) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', minWidth: 140, flex: '1 1 140px' }}>
      <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#f1f5f9', fontSize: 22, fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function MacularDystrophyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const ep = tab === 'Overview' || tab === 'Gene Table' ? 'overview'
             : tab === 'Clinical Atlas' ? 'breakdown'
             : 'definitions';
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(data => {
        if (ep === 'overview') setOverview(data);
        else if (ep === 'breakdown') setBreakdown(data);
        else setDefinitions(data);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  const dark = { background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', fontFamily: 'system-ui,sans-serif', padding: '24px 32px' };
  const tabBar = { display: 'flex', gap: 8, marginBottom: 24, borderBottom: '1px solid #334155', paddingBottom: 8 };
  const tabBtn = (active) => ({
    background: active ? '#3b82f6' : 'transparent',
    color: active ? '#fff' : '#94a3b8',
    border: 'none', borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontWeight: active ? 700 : 400,
  });

  return (
    <div style={dark}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9', marginBottom: 4 }}>
          🧬 Hereditary Macular Dystrophy Atlas
        </h1>
        <p style={{ color: '#94a3b8', fontSize: 14, margin: 0 }}>
          Complete 8-Gene Reference — ABCA4 · BEST1 · PRPH2 · TIMP3 · ELOVL4 · C1QTNF5 · PRDM13 · EFEMP1<br />
          320-Patient Aggregate Cohort (8×40) · Seeds 2406–2413 · Stargardt · Best · Pattern Dystrophy · Sorsby · LORD · NCMD · DHRD/ML
        </p>
      </div>

      {/* Gene chips */}
      <div style={{ marginBottom: 20, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
        {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
      </div>

      <div style={tabBar}>
        {TABS.map(t => (
          <button key={t} style={tabBtn(tab === t)} onClick={() => setTab(t)}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#94a3b8' }}>Loading…</div>}
      {error && <div style={{ color: '#f87171' }}>Error: {error}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
            <MetricCard label="Total Patients" value={overview.total_patients} sub="8 genes × 40" />
            <MetricCard label="VA < 6/18" value={`${overview.aggregate_metrics?.va_worse_than_6_18_pct ?? '--'}%`} sub="visual impairment" />
            <MetricCard label="CNV Present" value={`${overview.aggregate_metrics?.cnv_present_pct ?? '--'}%`} sub="choroidal neovascularisation" />
            <MetricCard label="Drusen/Deposits" value={`${overview.aggregate_metrics?.drusen_deposits_pct ?? '--'}%`} sub="Bruch's/subretinal deposits" />
            <MetricCard label="Anti-VEGF Rx" value={`${overview.aggregate_metrics?.anti_vegf_treatment_pct ?? '--'}%`} sub="intravitreal treatment" />
            <MetricCard label="Nyctalopia" value={`${overview.aggregate_metrics?.nyctalopia_pct ?? '--'}%`} sub="dark adaptation impaired" />
            <MetricCard label="EOG Abnormal" value={`${overview.aggregate_metrics?.eog_abnormal_pct ?? '--'}%`} sub="Arden ratio < 1.5" />
          </div>

          <h3 style={{ color: '#93c5fd', marginBottom: 12 }}>Gene Summary</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            {overview.gene_summary && Object.values(overview.gene_summary).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 14, minWidth: 260, flex: '1 1 260px', borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <GeneChip gene={g.gene} />
                  <span style={{ color: '#94a3b8', fontSize: 11 }}>{g.locus} · {g.protein_size} · {g.inheritance}</span>
                </div>
                <div style={{ color: '#e2e8f0', fontSize: 12, marginBottom: 6, lineHeight: 1.4 }}>
                  {g.disease_category}
                </div>
                <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>{g.onset_age}</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 6 }}>
                  <span style={{ color: '#f87171', fontSize: 11 }}>VA↓ {g.va_poor_pct}%</span>
                  <span style={{ color: '#fb923c', fontSize: 11 }}>CNV {g.cnv_pct}%</span>
                  <span style={{ color: '#fbbf24', fontSize: 11 }}>Drusen {g.drusen_pct}%</span>
                  <span style={{ color: '#34d399', fontSize: 11 }}>AntiVEGF {g.anti_vegf_pct}%</span>
                  <span style={{ color: '#60a5fa', fontSize: 11 }}>EOG↓ {g.eog_abnormal_pct}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && overview && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#1e293b' }}>
                {['Gene','Locus','Size','Inh','Disease','VA↓ %','CNV %','Drusen %','AntiVEGF %','EOG↓ %','Urgency'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', color: '#93c5fd', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {overview.gene_summary && Object.values(overview.gene_summary).map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b', borderBottom: '1px solid #334155' }}>
                  <td style={{ padding: '8px 10px' }}><GeneChip gene={g.gene} /></td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.locus}</td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.protein_size}</td>
                  <td style={{ padding: '8px 10px', color: '#fbbf24', whiteSpace: 'nowrap' }}>{g.inheritance}</td>
                  <td style={{ padding: '8px 10px', color: '#e2e8f0', maxWidth: 200 }}>{g.disease_category}</td>
                  <td style={{ padding: '8px 10px', color: '#f87171', textAlign: 'center' }}>{g.va_poor_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#fb923c', textAlign: 'center' }}>{g.cnv_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#fbbf24', textAlign: 'center' }}>{g.drusen_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#34d399', textAlign: 'center' }}>{g.anti_vegf_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#60a5fa', textAlign: 'center' }}>{g.eog_abnormal_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8', fontSize: 11 }}>{g.surgical_urgency}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ marginTop: 24 }}>
            <h3 style={{ color: '#93c5fd', marginBottom: 12 }}>Gene Reference Detail</h3>
            {Object.entries(GENE_INFO).map(([gene, info]) => (
              <div key={gene} style={{ background: '#1e293b', borderRadius: 8, padding: 14, marginBottom: 10, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <GeneChip gene={gene} />
                  <span style={{ color: '#e2e8f0', fontSize: 13, fontWeight: 600 }}>{info.full}</span>
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>· {info.locus} · {info.size} · {info.inh}</span>
                </div>
                <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5 }}>{info.disease}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {breakdown.gene_breakdowns?.map(g => (
            <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, padding: 18, marginBottom: 16, borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                <GeneChip gene={g.gene} />
                <span style={{ color: '#e2e8f0', fontWeight: 700 }}>{g.disease_category}</span>
                <span style={{ color: '#94a3b8', fontSize: 12 }}>· {g.locus} · {g.protein_size} · {g.inheritance}</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                <div>
                  <div style={{ color: '#fbbf24', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>PATHOGNOMONIC</div>
                  <div style={{ color: '#e2e8f0', fontSize: 12, lineHeight: 1.5 }}>{g.pathognomonic}</div>
                </div>
                <div>
                  <div style={{ color: '#60a5fa', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>KEY DDx</div>
                  <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5 }}>{g.key_ddx}</div>
                </div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#34d399', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>TREATMENT / MANAGEMENT</div>
                <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5, whiteSpace: 'pre-line' }}>{g.treatment}</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#a78bfa', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>KEY FEATURES</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {g.key_features?.map((f, i) => (
                    <span key={i} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 4, padding: '2px 8px', fontSize: 11, color: '#cbd5e1' }}>{f}</span>
                  ))}
                </div>
              </div>

              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', borderTop: '1px solid #334155', paddingTop: 10, marginTop: 8 }}>
                <span style={{ color: '#f87171', fontSize: 12 }}>VA↓ {g.va_poor_pct}%</span>
                <span style={{ color: '#fb923c', fontSize: 12 }}>CNV {g.cnv_pct}%</span>
                <span style={{ color: '#fbbf24', fontSize: 12 }}>Drusen {g.drusen_pct}%</span>
                <span style={{ color: '#34d399', fontSize: 12 }}>Anti-VEGF {g.anti_vegf_pct}%</span>
                <span style={{ color: '#60a5fa', fontSize: 12 }}>EOG↓ {g.eog_abnormal_pct}%</span>
                <span style={{ color: '#a78bfa', fontSize: 12 }}>Nyctalopia {g.nyctalopia_pct}%</span>
                <span style={{ color: '#94a3b8', fontSize: 12 }}>Consanguineous {g.consanguineous_pct}%</span>
                <span style={{ color: '#64748b', fontSize: 11 }}>Urgency: {g.surgical_urgency}</span>
              </div>

              {g.sample_patients?.length > 0 && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>Sample Patients (n=3)</div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {g.sample_patients.map(p => (
                      <div key={p.id} style={{ background: '#0f172a', borderRadius: 6, padding: '6px 10px', fontSize: 11, color: '#cbd5e1' }}>
                        <b>{p.id}</b> · VA↓: {p.va_poor ? 'Y' : 'N'} · CNV: {p.cnv ? 'Y' : 'N'} · AntiVEGF: {p.anti_vegf_treatment ? 'Y' : 'N'} · EOG↓: {p.eog_abnormal ? 'Y' : 'N'} · Nyct: {p.nyctalopia ? 'Y' : 'N'}
                      </div>
                    ))}
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
          <h3 style={{ color: '#93c5fd', marginBottom: 12 }}>Gene Definitions</h3>
          {Object.values(definitions.gene_entries || {}).map(e => (
            <div key={e.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 14, marginBottom: 12, borderLeft: `4px solid ${GENE_COLORS[e.gene] || '#555'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                <GeneChip gene={e.gene} />
                <span style={{ color: '#e2e8f0', fontSize: 13, fontWeight: 600 }}>{e.full_name}</span>
                <span style={{ color: '#94a3b8', fontSize: 11 }}>· {e.locus} · {e.protein_size} · {e.inheritance}</span>
              </div>
              <div style={{ color: '#fbbf24', fontSize: 11, marginBottom: 4 }}>{e.disease_name}</div>
              <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5, marginBottom: 6 }}>{e.disease_pathway}</div>
              <div style={{ color: '#a78bfa', fontSize: 11, marginBottom: 2, fontWeight: 700 }}>PATHOGNOMONIC</div>
              <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5, marginBottom: 6 }}>{e.pathognomonic}</div>
              <div style={{ color: '#34d399', fontSize: 11, fontWeight: 700, marginBottom: 2 }}>TREATMENT</div>
              <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5 }}>{e.treatment}</div>
            </div>
          ))}

          <h3 style={{ color: '#93c5fd', marginTop: 24, marginBottom: 12 }}>Macular Dystrophy Glossary</h3>
          {Object.entries(definitions.md_glossary || {}).map(([term, def]) => (
            <div key={term} style={{ background: '#1e293b', borderRadius: 8, padding: 14, marginBottom: 10 }}>
              <div style={{ color: '#fbbf24', fontSize: 12, fontWeight: 700, marginBottom: 6 }}>{term}</div>
              <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.6 }}>{def}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
