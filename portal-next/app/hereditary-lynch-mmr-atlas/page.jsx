'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-lynch-mmr-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'MLH1':  '#b71c1c',  // deep red — most common Lynch, most penetrant CRC, BRAF-V600E discriminator
  'MSH2':  '#1565c0',  // deep blue — Muir-Torre+Lynch, urinary-tract highest, EPCAM-epigenetic link
  'MSH6':  '#7b1fa2',  // deep purple — endometrial bias 70%, MSI-L pitfall, attenuated Lynch
  'PMS2':  '#e65100',  // deep orange — CMMRD biallelic brain tumors, CALMs, most attenuated Lynch
  'EPCAM': '#00695c',  // dark teal — epigenetic MSH2 silencing, 3' deletion, CTE biallelic
  'MLH3':  '#37474f',  // dark slate — MutLγ meiotic MMR, modifier, male infertility
  'MSH3':  '#2e7d32',  // dark green — MutSβ AR polyposis, glioblastoma, dinucleotide MSI-specific
  'PMS1':  '#4527a0',  // deep indigo — MutLβ modifier, low penetrance, no endonuclease
};

const GENE_INFO = {
  'MLH1':  { full: 'MLH1 / MutL Homolog 1 / 793aa', locus: '3p22.2', size: '793 aa / 90 kDa (MutLα anchor; ATPase)', inh: 'AD/AR' },
  'MSH2':  { full: 'MSH2 / MutS Homolog 2 / 934aa', locus: '2p21', size: '934 aa / 100 kDa (MutSα+MutSβ shared)', inh: 'AD/AR' },
  'MSH6':  { full: 'MSH6 / MutS Homolog 6 / 1360aa', locus: '2p16.3', size: '1360 aa / 160 kDa (MutSα binding)', inh: 'AD' },
  'PMS2':  { full: 'PMS2 / PMS1 Homolog 2 / 862aa', locus: '7p22.1', size: '862 aa / 96 kDa (MutLα endonuclease)', inh: 'AD/AR' },
  'EPCAM': { full: 'EPCAM / Epithelial Cell Adhesion Molecule / 314aa', locus: '2p21', size: '314 aa / 35 kDa (transmembrane; 3\' deletion → MSH2 methylation)', inh: 'AD' },
  'MLH3':  { full: 'MLH3 / MutL Homolog 3 / 1453aa', locus: '14q24.3', size: '1453 aa / 165 kDa (MutLγ meiotic MMR)', inh: 'AR/AD' },
  'MSH3':  { full: 'MSH3 / MutS Homolog 3 / 1137aa', locus: '5q14.1', size: '1137 aa / 128 kDa (MutSβ IDL repair)', inh: 'AR' },
  'PMS1':  { full: 'PMS1 / PMS1 Homolog 1 / 932aa', locus: '2q31.1', size: '932 aa / 103 kDa (MutLβ modifier; no endonuclease)', inh: 'AD' },
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
        <span>{label}</span><span style={{ fontWeight: 700, color: '#e2e8f0' }}>{pct}%</span>
      </div>
      <div style={{ background: '#334155', borderRadius: 4, height: 8 }}>
        <div style={{ background: color || '#38bdf8', width: `${pct}%`, height: 8, borderRadius: 4 }} />
      </div>
    </div>
  );
}

export default function HeredLynchMMRAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [selGene, setSelGene] = useState(null);

  useEffect(() => {
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' ? 'breakdown'
      : tab === 'Clinical Atlas' ? 'breakdown'
      : 'overview';
    setLoading(true); setErr(null);
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => {
        if (ep === 'overview') setOverview(data);
        else if (ep === 'breakdown') setBreakdown(data);
        else setDefinitions(data);
        setLoading(false);
      })
      .catch(e => { setErr(String(e)); setLoading(false); });
  }, [tab]);

  const bg = '#0f172a';
  const card = '#1e293b';
  const accent = '#22c55e';  // green — Lynch/MMR cancer prevention theme

  const genes = ['MLH1', 'MSH2', 'MSH6', 'PMS2', 'EPCAM', 'MLH3', 'MSH3', 'PMS1'];

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: accent, margin: 0 }}>
          &#x1f9ec; Hereditary Lynch / MMR Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 6 }}>
          Complete 8-Gene Mismatch Repair Reference · MLH1 · MSH2 · MSH6 · PMS2 · EPCAM · MLH3 · MSH3 · PMS1
          · 320 patients (8×40) · seeds 2718–2725
        </div>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
          MutLα (MLH1+PMS2) · MutSα (MSH2+MSH6) · MutSβ (MSH2+MSH3) · MutLγ (MLH1+MLH3) · MutLβ (MLH1+PMS1) ·
          EPCAM-epigenetic-MSH2-silencing · Lynch/CMMRD/DMMR-Polyposis
        </div>
      </div>

      {/* Gene chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 20 }}>
        {genes.map(g => (
          <GeneChip key={g} gene={g} active={selGene} onClick={g2 => setSelGene(selGene === g2 ? null : g2)} />
        ))}
        {selGene && (
          <span onClick={() => setSelGene(null)} style={{ fontSize: 11, color: '#64748b', cursor: 'pointer', marginLeft: 8, alignSelf: 'center' }}>
            ✕ clear
          </span>
        )}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              background: tab === t ? accent : '#1e293b',
              color: tab === t ? '#0f172a' : '#94a3b8',
              border: 'none', borderRadius: 6, padding: '6px 16px', cursor: 'pointer',
              fontWeight: tab === t ? 700 : 400, fontSize: 13,
            }}
          >{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#64748b' }}>Loading…</div>}
      {err && <div style={{ color: '#ef4444' }}>Error: {err}</div>}

      {/* OVERVIEW */}
      {tab === 'Overview' && overview && !loading && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
            <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40, seeds 2718-2725" />
            <MetricCard label="Genes" value={overview.genes?.length} sub="Lynch/MMR atlas" />
            <MetricCard label="MMR Pathway Groups" value={overview.pathway_categories?.length} />
            <MetricCard label="Critical Distinctions" value={overview.critical_distinctions?.length} warn />
          </div>

          {/* Gene Summaries grid */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: accent, marginBottom: 12 }}>Gene Summary Table</div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene', 'Locus', 'Onset (yrs)', 'CRC%', 'EC%', 'Gastric%', 'Urinary%', 'Brain%', 'Sebaceous%', 'Polyposis%', 'CMMRD%', 'MSI-H%'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', color: '#64748b', textAlign: 'left', borderBottom: '1px solid #334155', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.gene_summaries || []).filter(gs => !selGene || gs.gene === selGene).map(gs => (
                    <tr key={gs.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                      <td style={{ padding: '6px 8px' }}>
                        <GeneChip gene={gs.gene} active={selGene} onClick={g => setSelGene(selGene === g ? null : g)} />
                      </td>
                      <td style={{ padding: '6px 8px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{gs.locus}</td>
                      <td style={{ padding: '6px 8px', color: '#e2e8f0', fontWeight: 700 }}>{gs.mean_onset_years}y</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_crc > 60 ? '#ef4444' : gs.pct_crc > 30 ? '#f59e0b' : '#64748b' }}>{gs.pct_crc}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_ec > 50 ? '#a78bfa' : gs.pct_ec > 20 ? '#818cf8' : '#64748b' }}>{gs.pct_ec}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_gastric > 8 ? '#f97316' : '#64748b' }}>{gs.pct_gastric}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_urinary > 15 ? '#38bdf8' : '#64748b' }}>{gs.pct_urinary}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_brain > 10 ? '#ef4444' : '#64748b' }}>{gs.pct_brain}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_sebaceous > 5 ? '#fbbf24' : '#64748b' }}>{gs.pct_sebaceous}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_polyposis > 50 ? '#22c55e' : '#64748b' }}>{gs.pct_polyposis}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_cmmrd > 15 ? '#ef4444' : '#64748b' }}>{gs.pct_cmmrd}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_msi_high > 80 ? '#22c55e' : gs.pct_msi_high > 50 ? '#f59e0b' : '#64748b' }}>{gs.pct_msi_high}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pathway Categories */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: accent, marginBottom: 10 }}>MMR Pathway Categories</div>
            {(overview.pathway_categories || []).map((pc, i) => (
              <div key={i} style={{ background: card, borderRadius: 8, padding: 12, marginBottom: 8, borderLeft: `4px solid ${accent}`, fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>
                <div style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 4 }}>{pc.pathway}</div>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 4 }}>
                  {(pc.genes || []).map(g => <GeneChip key={g} gene={g} />)}
                </div>
                <div style={{ color: '#94a3b8' }}>{pc.note}</div>
              </div>
            ))}
          </div>

          {/* Critical Distinctions */}
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#ef4444', marginBottom: 10 }}>
              ⚠ Critical Distinctions
            </div>
            {(overview.critical_distinctions || []).map((cd, i) => (
              <div key={i} style={{ background: '#1a0a0a', border: '1px solid #7f1d1d', borderRadius: 8, padding: 12, marginBottom: 8, fontSize: 12, color: '#fca5a5', lineHeight: 1.6 }}>
                {cd}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* GENE TABLE */}
      {tab === 'Gene Table' && breakdown && !loading && (
        <div>
          {(breakdown.genes || []).filter(g => !selGene || g.gene === selGene).map(g => (
            <div key={g.gene} style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 20, borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#64748b'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                <GeneChip gene={g.gene} />
                <span style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0' }}>{GENE_INFO[g.gene]?.full}</span>
              </div>
              <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 12 }}>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>Locus: <b style={{ color: '#e2e8f0' }}>{g.locus}</b></span>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>Inheritance: <b style={{ color: '#e2e8f0' }}>{GENE_INFO[g.gene]?.inh}</b></span>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>Patients: <b style={{ color: '#38bdf8' }}>{g.n_patients}</b></span>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>{g.protein_size}</span>
              </div>

              {/* Phenotype bars */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0 20px' }}>
                {[
                  { label: 'Colorectal Cancer', key: 'pct_crc', color: '#ef4444' },
                  { label: 'Endometrial Cancer', key: 'pct_ec', color: '#a78bfa' },
                  { label: 'Gastric Cancer', key: 'pct_gastric', color: '#f97316' },
                  { label: 'Ovarian Cancer', key: 'pct_ovarian', color: '#f472b6' },
                  { label: 'Urinary Tract Cancer', key: 'pct_urinary', color: '#38bdf8' },
                  { label: 'Brain Tumor', key: 'pct_brain', color: '#ef4444' },
                  { label: 'Sebaceous Neoplasm', key: 'pct_sebaceous', color: '#fbbf24' },
                  { label: 'Colorectal Polyposis', key: 'pct_polyposis', color: '#22c55e' },
                  { label: 'CMMRD Phenotype', key: 'pct_cmmrd', color: '#ef4444' },
                  { label: 'MSI-High', key: 'pct_msi_high', color: '#22c55e' },
                  { label: 'Immunotherapy Received', key: 'pct_immunotherapy', color: '#818cf8' },
                ].map(({ label, key, color }) => {
                  const pct = g[key] ?? 0;
                  return pct > 0 ? (
                    <PctBar key={key} label={label} pct={pct} color={color} />
                  ) : null;
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* CLINICAL ATLAS */}
      {tab === 'Clinical Atlas' && breakdown && !loading && (
        <div>
          {(breakdown.genes || []).filter(g => !selGene || g.gene === selGene).map(g => (
            <div key={g.gene} style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 24, borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#64748b'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                <GeneChip gene={g.gene} />
                <span style={{ fontSize: 16, fontWeight: 700, color: accent }}>{g.gene}</span>
                <span style={{ fontSize: 12, color: '#64748b' }}>{g.locus} · {g.protein_size}</span>
              </div>

              {[
                { title: 'Inheritance / Biology', content: g.inheritance },
                { title: 'Disease Category', content: g.disease_category },
                { title: 'MMR Pathway Mechanism', content: g.disease_pathway },
                { title: 'Pathognomonic Features', content: g.pathognomonic },
                { title: 'Treatment / Surveillance', content: g.treatment },
              ].map(({ title, content }) => (
                <div key={title} style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#38bdf8', marginBottom: 4 }}>{title}</div>
                  <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7, background: '#0f172a', borderRadius: 6, padding: '10px 14px' }}>
                    {content}
                  </div>
                </div>
              ))}

              {/* Patient sample table */}
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#64748b', marginBottom: 6 }}>
                  Patient Cohort Sample (first 10 of {g.n_patients})
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ fontSize: 10, borderCollapse: 'collapse', width: '100%' }}>
                    <thead>
                      <tr style={{ background: '#0f172a' }}>
                        {['ID', 'Sex', 'Onset', 'Age', 'CRC', 'EC', 'Gastric', 'Urinary', 'Brain', 'Sebac', 'Polyp', 'CMMRD', 'CALMs', 'MSI-H', 'IO'].map(h => (
                          <th key={h} style={{ padding: '4px 6px', color: '#475569', textAlign: 'left', borderBottom: '1px solid #1e293b' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(g.patients || []).slice(0, 10).map(p => (
                        <tr key={p.patient_id} style={{ borderBottom: '1px solid #1e293b' }}>
                          <td style={{ padding: '3px 6px', color: '#64748b' }}>{p.patient_id}</td>
                          <td style={{ padding: '3px 6px' }}>{p.sex}</td>
                          <td style={{ padding: '3px 6px', color: '#e2e8f0' }}>{p.age_onset_years}y</td>
                          <td style={{ padding: '3px 6px', color: '#94a3b8' }}>{p.age_current_years}y</td>
                          {[
                            [p.colorectal_cancer, '#ef4444'],
                            [p.endometrial_cancer, '#a78bfa'],
                            [p.gastric_cancer, '#f97316'],
                            [p.urinary_tract_cancer, '#38bdf8'],
                            [p.brain_tumor, '#ef4444'],
                            [p.sebaceous_neoplasm, '#fbbf24'],
                            [p.colorectal_polyposis, '#22c55e'],
                            [p.cmmrd_phenotype, '#ef4444'],
                            [p.cafe_au_lait_macules, '#f59e0b'],
                            [p.msi_high, '#22c55e'],
                            [p.immunotherapy_received, '#818cf8'],
                          ].map(([val, col], j) => (
                            <td key={j} style={{ padding: '3px 6px', color: val ? col : '#334155' }}>
                              {val ? '✓' : '·'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS */}
      {tab === 'Definitions' && definitions && !loading && (
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: accent, marginBottom: 16 }}>
            Lynch/MMR Glossary & Standards
          </div>
          {Object.entries(definitions.glossary || {}).map(([term, def]) => (
            <div key={term} style={{ background: card, borderRadius: 8, padding: 14, marginBottom: 12, borderLeft: '4px solid #334155' }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#38bdf8', marginBottom: 6 }}>{term}</div>
              <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7 }}>{def}</div>
            </div>
          ))}

          <div style={{ fontSize: 14, fontWeight: 700, color: accent, marginBottom: 12, marginTop: 24 }}>
            Standards & References
          </div>
          {(definitions.standards || []).map((s, i) => (
            <div key={i} style={{ background: '#1a2332', borderRadius: 6, padding: '8px 14px', marginBottom: 6, fontSize: 12, color: '#64748b' }}>
              {s}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
