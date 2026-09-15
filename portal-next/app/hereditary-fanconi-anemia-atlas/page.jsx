'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-fanconi-anemia-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'FANCA':  '#b71c1c',  // deep red — most common FA (60-65%), founder alleles, HSCT primary
  'FANCC':  '#1565c0',  // deep blue — Ashkenazi IVS4+4A>T 1:89, STAT1 cytoplasmic inhibitor
  'FANCD2': '#7b1fa2',  // deep purple — central ICL hub, FANCD2-Ub gatekeeper Western blot
  'FANCG':  '#e65100',  // deep orange — 10% FA, SA Black founder, azoospermia, severe aplasia
  'BRCA2':  '#00695c',  // dark teal — FA-D1 biallelic (Wilms+MB+AML) + AD HBOC
  'PALB2':  '#37474f',  // dark slate — FA-N biallelic + AD 53% breast cancer, PARPi 82% ORR
  'BRIP1':  '#2e7d32',  // dark green — FA-J biallelic + AD ovarian ONLY (no breast) RRSO 45-50
  'FANCI':  '#4527a0',  // deep indigo — ID2 complex partner, K523-Ub, mutual FANCD2 stabiliser
};

const GENE_INFO = {
  'FANCA':  { full: 'FANCA / FA Complementation Group A / 1455aa', locus: '16q24.3', size: '1455 aa / 163 kDa (FA-CORE scaffold; nuclear import)', inh: 'AR' },
  'FANCC':  { full: 'FANCC / FA Complementation Group C / 558aa', locus: '9q22.32', size: '558 aa / 63 kDa (FA-CORE cytoplasmic; STAT1 inhibitor)', inh: 'AR' },
  'FANCD2': { full: 'FANCD2 / FA Complementation Group D2 / 1451aa', locus: '3p25.3', size: '1451 aa / 155 kDa (ID2 complex; K561 monoubiquitination)', inh: 'AR' },
  'FANCG':  { full: 'FANCG / XRCC9 / FA Complementation Group G / 622aa', locus: '9p13.3', size: '622 aa / 68 kDa (FA-CORE TPR scaffold; FANCA nuclear import)', inh: 'AR' },
  'BRCA2':  { full: 'BRCA2 / FANCD1 / Breast Cancer 2 / 3418aa', locus: '13q12.3', size: '3418 aa / 384 kDa (RAD51 loader; 8 BRC repeats; OB-folds)', inh: 'AR/AD' },
  'PALB2':  { full: 'PALB2 / FANCN / Partner And Localiser of BRCA2 / 1186aa', locus: '16p12.2', size: '1186 aa / 130 kDa (coiled-coil + WD40; BRCA1-BRCA2 bridge)', inh: 'AR/AD' },
  'BRIP1':  { full: 'BRIP1 / FANCJ / BACH1 / BRCA1-IP C-Helicase 1 / 1249aa', locus: '17q23.2', size: '1249 aa / 140 kDa (DEAH-box 5\'→3\' helicase; G4 resolver)', inh: 'AR/AD' },
  'FANCI':  { full: 'FANCI / FA Complementation Group I / 1328aa', locus: '15q26.1', size: '1328 aa / 147 kDa (ID2 complex; K523-Ub; ARM-solenoid)', inh: 'AR' },
};

function GeneChip({ gene, active, onClick }) {
  const col = GENE_COLORS[gene] || '#555';
  return (
    <span
      onClick={() => onClick && onClick(gene)}
      style={{
        background: col, color: '#fff', borderRadius: 4,
        padding: '3px 10px', fontSize: 12, fontWeight: 700,
        margin: '0 3px 4px 0', cursor: onClick ? 'pointer' : 'default',
        opacity: active === null || active === gene ? 1 : 0.45,
        border: active === gene ? '2px solid #fff' : '2px solid transparent',
        display: 'inline-block',
      }}
    >{gene}</span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 130 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8' }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function PctBar({ label, pct, color }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#94a3b8', marginBottom: 3 }}>
        <span>{label}</span><span>{pct}%</span>
      </div>
      <div style={{ background: '#334155', borderRadius: 4, height: 8 }}>
        <div style={{ background: color || '#38bdf8', width: `${pct}%`, height: 8, borderRadius: 4 }} />
      </div>
    </div>
  );
}

export default function HereditaryFanconiAnemiaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [activeGene, setActiveGene] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        if (tab === 'Overview' && !overview) {
          const r = await fetch(`${API}/api/${SLUG}/overview`);
          setOverview(await r.json());
        } else if ((tab === 'Gene Table' || tab === 'Clinical Atlas') && !breakdown) {
          const r = await fetch(`${API}/api/${SLUG}/breakdown`);
          setBreakdown(await r.json());
        } else if (tab === 'Definitions' && !definitions) {
          const r = await fetch(`${API}/api/${SLUG}/definitions`);
          setDefinitions(await r.json());
        }
      } catch (e) { setError(e.message); }
      setLoading(false);
    };
    fetchData();
  }, [tab]);

  const base = { fontFamily: 'monospace', background: '#0f172a', color: '#e2e8f0', minHeight: '100vh', padding: '0 0 60px 0' };
  const header = { background: 'linear-gradient(135deg,#4a044e 0%,#1e1b4b 60%,#0f172a 100%)', padding: '28px 32px 20px', borderBottom: '1px solid #334155' };

  return (
    <div style={base}>
      <div style={header}>
        <div style={{ fontSize: 11, color: '#a78bfa', letterSpacing: 2, marginBottom: 6 }}>🧬 HEREDITARY DISEASE ATLAS — ICL DNA REPAIR</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#fff', lineHeight: 1.3 }}>
          Hereditary Fanconi Anemia & DNA Crosslink Repair Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 6 }}>
          Complete 8-Gene ICL Repair Reference · FANCA-FANCC-FANCD2-FANCG-BRCA2(FANCD1)-PALB2(FANCN)-BRIP1(FANCJ)-FANCI · 320 patients · seeds 2726–2733
        </div>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} active={null} />)}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, padding: '16px 32px 0', borderBottom: '1px solid #1e293b' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#a78bfa' : 'transparent',
            color: tab === t ? '#fff' : '#94a3b8',
            border: 'none', borderRadius: '6px 6px 0 0',
            padding: '8px 18px', cursor: 'pointer', fontWeight: 600, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 32px' }}>
        {loading && <div style={{ color: '#94a3b8' }}>Loading…</div>}
        {error && <div style={{ color: '#ef4444' }}>Error: {error}</div>}

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 28 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40 synthetic cohort" />
              <MetricCard label="Genes Covered" value={overview.genes?.length || 8} sub="FA-A/C/D2/G/D1/N/J/I" />
              <MetricCard label="Pathway" value="ICL Repair" sub="FA-CORE → ID2 → HR" />
              <MetricCard label="Seed Range" value="2726–2733" sub="deterministic cohort" />
            </div>

            {/* Pathway categories */}
            {overview.pathway_categories?.map(cat => (
              <div key={cat.pathway} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: 16, marginBottom: 14 }}>
                <div style={{ fontWeight: 700, color: '#a78bfa', marginBottom: 6, fontSize: 14 }}>{cat.pathway}</div>
                <div style={{ marginBottom: 8 }}>
                  {cat.genes?.map(g => <GeneChip key={g} gene={g} active={null} />)}
                </div>
                <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6 }}>{cat.note}</div>
              </div>
            ))}

            {/* Gene summary table */}
            <div style={{ overflowX: 'auto', marginTop: 20 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene','Locus','Protein','N Patients','Onset (yr)','BMF%','AML%','SCC%','Wilms%','MB%','Thumb%','HSCT%'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', fontWeight: 600, borderBottom: '1px solid #334155' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summaries?.map((g, i) => (
                    <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                      <td style={{ padding: '7px 10px' }}><GeneChip gene={g.gene} active={null} /></td>
                      <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.locus}</td>
                      <td style={{ padding: '7px 10px', color: '#64748b', maxWidth: 180 }}>{g.protein_size}</td>
                      <td style={{ padding: '7px 10px', color: '#38bdf8' }}>{g.n_patients}</td>
                      <td style={{ padding: '7px 10px', color: '#fbbf24' }}>{g.mean_onset_years}</td>
                      <td style={{ padding: '7px 10px', color: '#ef4444' }}>{g.pct_bmf}%</td>
                      <td style={{ padding: '7px 10px', color: '#f97316' }}>{g.pct_aml_mds}%</td>
                      <td style={{ padding: '7px 10px', color: '#a78bfa' }}>{g.pct_scc}%</td>
                      <td style={{ padding: '7px 10px', color: '#34d399' }}>{g.pct_wilms}%</td>
                      <td style={{ padding: '7px 10px', color: '#60a5fa' }}>{g.pct_medulloblastoma}%</td>
                      <td style={{ padding: '7px 10px', color: '#fbbf24' }}>{g.pct_thumb_defect}%</td>
                      <td style={{ padding: '7px 10px', color: '#4ade80' }}>{g.pct_hsct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Critical distinctions */}
            {overview.critical_distinctions?.length > 0 && (
              <div style={{ marginTop: 24 }}>
                <div style={{ fontWeight: 700, color: '#a78bfa', marginBottom: 10, fontSize: 14 }}>⚡ Critical Clinical Distinctions</div>
                {overview.critical_distinctions.map((d, i) => (
                  <div key={i} style={{ background: '#1e293b', border: '1px solid #ef4444', borderRadius: 6, padding: '10px 14px', marginBottom: 8, fontSize: 12, color: '#fca5a5', lineHeight: 1.6 }}>
                    {d}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && breakdown && (
          <div>
            <div style={{ marginBottom: 14 }}>
              <span style={{ fontSize: 12, color: '#64748b', marginRight: 8 }}>Filter:</span>
              <span onClick={() => setActiveGene(null)} style={{ cursor: 'pointer', padding: '3px 10px', borderRadius: 4, background: activeGene === null ? '#a78bfa' : '#1e293b', color: '#fff', fontSize: 12, marginRight: 6 }}>All</span>
              {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} active={activeGene} onClick={setActiveGene} />)}
            </div>
            {breakdown.genes?.filter(g => activeGene === null || g.gene === activeGene).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', border: `2px solid ${GENE_COLORS[g.gene] || '#334155'}`, borderRadius: 10, padding: 20, marginBottom: 18 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                  <GeneChip gene={g.gene} active={null} />
                  <span style={{ color: '#94a3b8', fontSize: 13 }}>{g.locus}</span>
                  <span style={{ color: '#64748b', fontSize: 12 }}>{g.protein_size}</span>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 14 }}>
                  <MetricCard label="Patients" value={g.n_patients} />
                  <MetricCard label="BMF%" value={`${g.pct_bmf}%`} warn={g.pct_bmf > 85} />
                  <MetricCard label="AML/MDS%" value={`${g.pct_aml_mds}%`} warn={g.pct_aml_mds > 35} />
                  <MetricCard label="SCC%" value={`${g.pct_scc}%`} />
                  <MetricCard label="Wilms%" value={`${g.pct_wilms}%`} warn={g.pct_wilms > 15} />
                  <MetricCard label="HSCT%" value={`${g.pct_hsct}%`} />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                  <div>
                    <div style={{ color: '#a78bfa', fontWeight: 700, fontSize: 12, marginBottom: 4 }}>Inheritance</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.6 }}>{g.inheritance?.slice(0, 500)}</div>
                  </div>
                  <div>
                    <div style={{ color: '#ef4444', fontWeight: 700, fontSize: 12, marginBottom: 4 }}>⚠ Pathognomonic</div>
                    <div style={{ fontSize: 11, color: '#fca5a5', lineHeight: 1.6 }}>{g.pathognomonic?.slice(0, 500)}</div>
                  </div>
                  <div>
                    <div style={{ color: '#34d399', fontWeight: 700, fontSize: 12, marginBottom: 4 }}>Treatment</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.6 }}>{g.treatment?.slice(0, 500)}</div>
                  </div>
                  <div>
                    <div style={{ color: '#38bdf8', fontWeight: 700, fontSize: 12, marginBottom: 4 }}>Pathway</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.6 }}>{g.disease_pathway?.slice(0, 500)}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            <div style={{ marginBottom: 14 }}>
              {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} active={activeGene} onClick={setActiveGene} />)}
            </div>
            {breakdown.genes?.filter(g => activeGene === null || g.gene === activeGene).map(g => (
              <div key={g.gene} style={{ marginBottom: 28 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                  <GeneChip gene={g.gene} active={null} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus} · {g.protein_size}</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(220px,1fr))', gap: 10, marginBottom: 14 }}>
                  <PctBar label="Bone Marrow Failure" pct={g.pct_bmf} color="#ef4444" />
                  <PctBar label="AML / MDS" pct={g.pct_aml_mds} color="#f97316" />
                  <PctBar label="SCC (H&N/Oesoph)" pct={g.pct_scc} color="#a78bfa" />
                  <PctBar label="Wilms Tumor" pct={g.pct_wilms} color="#34d399" />
                  <PctBar label="Medulloblastoma" pct={g.pct_medulloblastoma} color="#60a5fa" />
                  <PctBar label="Thumb/Radial Defect" pct={g.pct_thumb_defect} color="#fbbf24" />
                  <PctBar label="Café-au-Lait Macules" pct={g.pct_cafe_au_lait} color="#fb923c" />
                  <PctBar label="Renal Anomaly" pct={g.pct_renal} color="#38bdf8" />
                  <PctBar label="HSCT Performed" pct={g.pct_hsct} color="#4ade80" />
                  <PctBar label="Diabetes Mellitus" pct={g.pct_diabetes} color="#f472b6" />
                  <PctBar label="Somatic Mosaicism" pct={g.pct_mosaicism} color="#94a3b8" />
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr style={{ background: '#1e293b' }}>
                        {['Patient','Sex','Onset','Age','BMF','AML','SCC','Wilms','MB','Thumb','CAL','Renal','HSCT','DM'].map(h => (
                          <th key={h} style={{ padding: '6px 8px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #334155' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {g.patients?.slice(0, 20).map((p, i) => (
                        <tr key={p.patient_id} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                          <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{p.patient_id}</td>
                          <td style={{ padding: '5px 8px' }}>{p.sex}</td>
                          <td style={{ padding: '5px 8px', color: '#fbbf24' }}>{p.age_onset_years}</td>
                          <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{p.age_current_years}</td>
                          <td style={{ padding: '5px 8px', color: p.bone_marrow_failure ? '#ef4444' : '#374151' }}>{p.bone_marrow_failure ? '✓' : '—'}</td>
                          <td style={{ padding: '5px 8px', color: p.aml_mds ? '#f97316' : '#374151' }}>{p.aml_mds ? '✓' : '—'}</td>
                          <td style={{ padding: '5px 8px', color: p.scc_head_neck_oesophageal ? '#a78bfa' : '#374151' }}>{p.scc_head_neck_oesophageal ? '✓' : '—'}</td>
                          <td style={{ padding: '5px 8px', color: p.wilms_tumor ? '#34d399' : '#374151' }}>{p.wilms_tumor ? '✓' : '—'}</td>
                          <td style={{ padding: '5px 8px', color: p.medulloblastoma ? '#60a5fa' : '#374151' }}>{p.medulloblastoma ? '✓' : '—'}</td>
                          <td style={{ padding: '5px 8px', color: p.thumb_radial_defect ? '#fbbf24' : '#374151' }}>{p.thumb_radial_defect ? '✓' : '—'}</td>
                          <td style={{ padding: '5px 8px', color: p.cafe_au_lait_macules ? '#fb923c' : '#374151' }}>{p.cafe_au_lait_macules ? '✓' : '—'}</td>
                          <td style={{ padding: '5px 8px', color: p.renal_anomaly ? '#38bdf8' : '#374151' }}>{p.renal_anomaly ? '✓' : '—'}</td>
                          <td style={{ padding: '5px 8px', color: p.hsct_performed ? '#4ade80' : '#374151' }}>{p.hsct_performed ? '✓' : '—'}</td>
                          <td style={{ padding: '5px 8px', color: p.diabetes_mellitus ? '#f472b6' : '#374151' }}>{p.diabetes_mellitus ? '✓' : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            {Object.entries(definitions.glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: '14px 18px', marginBottom: 12 }}>
                <div style={{ fontWeight: 700, color: '#a78bfa', marginBottom: 6, fontSize: 13 }}>{term}</div>
                <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7 }}>{def}</div>
              </div>
            ))}
            {definitions.standards?.length > 0 && (
              <div style={{ marginTop: 20 }}>
                <div style={{ fontWeight: 700, color: '#38bdf8', marginBottom: 10, fontSize: 14 }}>📚 Standards & References</div>
                {definitions.standards.map((s, i) => (
                  <div key={i} style={{ fontSize: 12, color: '#64748b', padding: '4px 0', borderBottom: '1px solid #1e293b' }}>{s}</div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
