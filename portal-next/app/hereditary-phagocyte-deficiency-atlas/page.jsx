'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-phagocyte-deficiency-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  CYBB:   '#b71c1c',  // deep red     — CGD-X1, most common CGD (65%), gp91phox XLR
  NCF1:   '#1565c0',  // deep blue    — CGD-AR2, GT deletion founder, most common AR CGD
  CYBA:   '#0d47a1',  // navy         — CGD-AR1, p22phox, mutual b558 destabilisation
  NCF2:   '#283593',  // dark indigo  — CGD-AR3, p67phox, rarest AR CGD
  ITGB2:  '#2e7d32',  // deep green   — LAD-I, CD18, most common LAD, delayed cord
  FERMT3: '#6a1b9a',  // deep purple  — LAD-III, Kindlin-3, LAD+bleeding dual defect
  ELANE:  '#e65100',  // burnt orange — SCN1/Cyclic Neutropenia, AD, G-CSF, 21-day cycle
  HAX1:   '#004d40',  // dark teal    — SCN3/Kostmann 1956, AR, neurological isoform B
};

const GENE_INFO = {
  CYBB:   { full: 'CYBB/gp91phox / 570aa', locus: 'Xp21.1', size: '570 aa / 91 kDa', inh: 'XLR',    disease: 'CGD-X1 — MOST COMMON CGD (~65% of all CGD); X-linked gp91phox deficiency; DHR ZERO OXIDATIVE BURST (males); female carriers bimodal; Aspergillus + Staphylococcus aureus + Serratia catalase-positive organisms; granuloma (GI/bladder); TRIPLE PROPHYLAXIS: TMP-SMX + itraconazole + IFN-gamma LIFELONG; HSCT curative' },
  NCF1:   { full: 'NCF1/p47phox / 390aa',  locus: '7q11.23', size: '390 aa / 47 kDa', inh: 'AR',     disease: 'CGD-AR2 — MOST COMMON AR CGD (~25%); GT deletion founder in NCF1B/NCF1C pseudogenes (85%); DHR markedly REDUCED (not zero) — DDx from CYBB; same clinical profile (Aspergillus, Staph, granuloma); same triple prophylaxis; pseudogene-aware sequencing required; milder than CYBB on average; HSCT if severe' },
  CYBA:   { full: 'CYBA/p22phox / 195aa',  locus: '16q24.2', size: '195 aa / 22 kDa', inh: 'AR',     disease: 'CGD-AR1 — p22phox stabilises gp91phox; BOTH gp91phox AND p22phox ABSENT on Western blot (mutual destabilisation); DHR zero (same as CYBB); Western blot discriminates: CYBB = p22phox present/gp91phox absent; CYBA = both absent; same clinical + prophylaxis as CYBB; consanguineous enriched; HSCT curative' },
  NCF2:   { full: 'NCF2/p67phox / 526aa',  locus: '1q25.3',  size: '526 aa / 67 kDa', inh: 'AR',     disease: 'CGD-AR3 — RAREST AR CGD (~2-5%); p67phox activation domain; CYTOCHROME b558 PRESENT on Western blot (gp91phox + p22phox) — key DDx: NCF2 has b558 present but cytosolic activator absent; enriched Middle East/North Africa; gene panel essential; same triple prophylaxis + HSCT option' },
  ITGB2:  { full: 'ITGB2/CD18 / 769aa',   locus: '21q22.3', size: '769 aa / 95 kDa', inh: 'AR',     disease: 'LAD-I — MOST COMMON LAD (>80%); CD18 absent (<1% = severe); ALL beta-2 integrins (LFA-1, Mac-1, p150.95) absent; DELAYED CORD SEPARATION >21 days + OMPHALITIS + LEUKOCYTOSIS >25×10⁹/L WITHOUT PUS = PATHOGNOMONIC; severe (<1% CD18) = death in infancy without HSCT; periodontitis in mild/moderate survivors; HSCT curative' },
  FERMT3: { full: 'FERMT3/Kindlin-3 / 667aa', locus: '11q13.1', size: '667 aa / 74 kDa', inh: 'AR', disease: 'LAD-III — Kindlin-3 activates BOTH beta-2 (neutrophils) AND beta-3 (platelets) integrins; LAD FEATURES + GLANZMANN-LIKE BLEEDING PATHOGNOMONIC COMBINATION; CD18 REDUCED (not absent) — DDx from LAD-I; platelet aggregation ABSENT (ADP/collagen/thrombin); HSCT corrects BOTH leukocyte AND platelet defects; NO aspirin/NSAIDs ever' },
  ELANE:  { full: 'ELANE/NE / 256aa',      locus: '19p13.3', size: '256 aa / 29 kDa', inh: 'AD',    disease: 'SCN1 / Cyclic Neutropenia — ELANE misfolding → ER stress → UPR → promyelocyte apoptosis; 21-DAY CYCLE ANC <200 + ORAL ULCERS + FEVER = Cyclic Neutropenia PATHOGNOMONIC (serial ANC 3×/week 6-8 wks); BM maturation arrest; G-CSF LIFELONG (target ANC >1000); MDS/AML risk 10-20% → ANNUAL BM biopsy + CSF3R/RUNX1 panel MANDATORY' },
  HAX1:   { full: 'HAX1 / 279aa',          locus: '1q21.3',  size: '279 aa / 31 kDa', inh: 'AR',    disease: 'SCN3/Kostmann Disease — ORIGINAL 1956 KOSTMANN SWEDISH PEDIGREE (first SCN ever described); HAX1 mitochondrial anti-apoptosis → HtrA2 → caspase-9; NEUROLOGICAL INVOLVEMENT (epilepsy + cognitive) = ISOFORM B TRUNCATION PATHOGNOMONIC — unique among SCN; G-CSF responsive; HSCT corrects neutropenia but NOT neurological (brain not replaced); annual BM biopsy mandatory' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 7px', fontSize: 11, fontWeight: 700, marginRight: 4,
    }}>{text}</span>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: '#fff', border: `2px solid ${color || '#e0e0e0'}`,
      borderRadius: 10, padding: '14px 18px', minWidth: 120, textAlign: 'center',
    }}>
      <div style={{ fontSize: 26, fontWeight: 800, color: color || '#333' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#555', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#888' }}>{sub}</div>}
    </div>
  );
}

export default function HeredPhagocyteAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov); setBreakdown(bk); setDefinitions(df);
    }).catch(e => setError(String(e))).finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#b71c1c', fontWeight: 700 }}>Loading Hereditary Phagocyte Deficiency Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#b71c1c' }}>Error: {error}</div>;
  if (!overview) return null;

  const genes = Object.keys(GENE_COLORS);

  return (
    <div style={{ padding: '24px 32px', fontFamily: 'system-ui, sans-serif', maxWidth: 1200 }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#b71c1c', marginBottom: 4 }}>
          🧬 Hereditary Phagocyte Deficiency Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#555' }}>
          Complete 8-Gene Atlas — CYBB · NCF1 · CYBA · NCF2 · ITGB2 · FERMT3 · ELANE · HAX1 —
          320 patients (8 × 40, seeds 2326-2333) | CGD · LAD · SCN
        </div>
      </div>

      {/* Stat cards */}
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 22 }}>
        <StatCard label="Total Patients" value={overview.n_patients} color="#b71c1c" />
        <StatCard label="Genes" value={overview.n_genes} color="#2e7d32" />
        <StatCard label="Seed Range" value="2326-2333" color="#e65100" />
        <StatCard label="CGD Genes" value={overview.cgd_genes?.length ?? 4} sub="CYBB · NCF1 · CYBA · NCF2" color="#1565c0" />
        <StatCard label="LAD Genes" value={overview.lad_genes?.length ?? 2} sub="ITGB2 · FERMT3" color="#6a1b9a" />
        <StatCard label="SCN Genes" value={overview.scn_genes?.length ?? 2} sub="ELANE · HAX1" color="#004d40" />
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '2px solid #e0e0e0' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', fontWeight: 700, fontSize: 13, cursor: 'pointer',
            background: tab === i ? '#b71c1c' : '#f5f5f5',
            color: tab === i ? '#fff' : '#333',
            border: 'none', borderRadius: '6px 6px 0 0',
          }}>{t}</button>
        ))}
      </div>

      {/* Tab 0 — Overview */}
      {tab === 0 && (
        <div>
          {/* Disease category map */}
          <div style={{ marginBottom: 24 }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, color: '#333', marginBottom: 12 }}>Disease Categories</h2>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              {Object.entries(overview.disease_categories || {}).map(([cat, catGenes]) => (
                <div key={cat} style={{ background: '#f9f9f9', border: '1.5px solid #ddd', borderRadius: 8, padding: '12px 16px', minWidth: 180 }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#b71c1c', marginBottom: 6 }}>{cat}</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {catGenes.map(g => (
                      <Badge key={g} text={g} color={GENE_COLORS[g] || '#555'} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Key clinical pearls */}
          <div style={{ marginBottom: 24 }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, color: '#333', marginBottom: 12 }}>Key Clinical Pearls</h2>
            {(overview.key_clinical_pearls || []).map((pearl, i) => (
              <div key={i} style={{
                background: '#fff8f0', border: '1.5px solid #ffcc80',
                borderRadius: 8, padding: '10px 14px', marginBottom: 8, fontSize: 12.5, lineHeight: 1.5,
              }}>
                <span style={{ fontWeight: 700, color: '#e65100' }}>Pearl {i + 1}:</span> {pearl}
              </div>
            ))}
          </div>

          {/* Pathway map */}
          <div style={{ marginBottom: 24 }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, color: '#333', marginBottom: 12 }}>Phagocyte Pathway Map</h2>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
              {Object.entries(overview.phagocyte_pathway_map || {}).map(([g, pathway]) => (
                <div key={g} style={{
                  background: (GENE_COLORS[g] || '#555') + '15',
                  border: `1.5px solid ${GENE_COLORS[g] || '#555'}55`,
                  borderRadius: 8, padding: '8px 12px', minWidth: 200, maxWidth: 280,
                }}>
                  <div style={{ fontWeight: 800, color: GENE_COLORS[g] || '#555', fontSize: 13 }}>{g}</div>
                  <div style={{ fontSize: 11.5, color: '#444', marginTop: 3 }}>{pathway}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Clinical emergency flags */}
          <div style={{ marginBottom: 24 }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, color: '#c62828', marginBottom: 12 }}>⚠️ Clinical Emergency Flags</h2>
            {(overview.clinical_emergency_flags || []).map((flag, i) => (
              <div key={i} style={{
                background: '#fff5f5', border: '1.5px solid #ef9a9a',
                borderRadius: 8, padding: '10px 14px', marginBottom: 8, fontSize: 12.5, lineHeight: 1.5,
              }}>
                <span style={{ fontWeight: 700, color: '#c62828' }}>🚨 Emergency {i + 1}:</span> {flag}
              </div>
            ))}
          </div>

          {/* Diagnostic algorithm */}
          <div style={{ marginBottom: 24 }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, color: '#333', marginBottom: 12 }}>Diagnostic Algorithm</h2>
            {(overview.diagnostic_algorithm || []).map((step, i) => (
              <div key={i} style={{
                background: '#f3f8ff', border: '1.5px solid #90caf9',
                borderRadius: 8, padding: '10px 14px', marginBottom: 8, fontSize: 12.5, lineHeight: 1.5,
              }}>
                <span style={{ fontWeight: 700, color: '#1565c0' }}>{step.split('—')[0]}</span>
                {step.includes('—') ? '—' + step.split('—').slice(1).join('—') : ''}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 1 — Gene Table */}
      {tab === 1 && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#b71c1c', color: '#fff' }}>
                {['Gene', 'Protein / Size', 'Locus', 'Inh.', 'Disease / Category', 'Key Feature'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {genes.map((g, ri) => {
                const info = GENE_INFO[g];
                const ovg = overview.gene_summary?.[g];
                return (
                  <tr key={g} style={{ background: ri % 2 === 0 ? '#fafafa' : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 800, color: GENE_COLORS[g] }}>{g}</td>
                    <td style={{ padding: '8px 10px' }}>{info.full}</td>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>{info.locus}</td>
                    <td style={{ padding: '8px 10px' }}><Badge text={info.inh} color={GENE_COLORS[g]} /></td>
                    <td style={{ padding: '8px 10px', maxWidth: 180, fontSize: 11 }}>{ovg?.phagocyte_category || ''}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, maxWidth: 220 }}>{(info.disease || '').substring(0, 180)}…</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Tab 2 — Clinical Atlas */}
      {tab === 2 && breakdown && (
        <div>
          {genes.map(g => {
            const gd = breakdown.gene_breakdown?.[g];
            if (!gd) return null;
            return (
              <div key={g} style={{
                border: `2px solid ${GENE_COLORS[g]}55`, borderRadius: 10,
                marginBottom: 20, overflow: 'hidden',
              }}>
                <div style={{ background: GENE_COLORS[g], color: '#fff', padding: '10px 16px', display: 'flex', gap: 12, alignItems: 'center' }}>
                  <span style={{ fontSize: 17, fontWeight: 800 }}>{g}</span>
                  <span style={{ fontSize: 12 }}>{gd.locus} · {gd.protein_size} · {gd.inheritance}</span>
                  <span style={{ fontSize: 12, marginLeft: 'auto' }}>{gd.n_patients} patients</span>
                </div>
                <div style={{ padding: '12px 16px', background: '#fafafa' }}>
                  <div style={{ fontSize: 12, color: '#444', marginBottom: 8, lineHeight: 1.5 }}>
                    <strong>Pathognomonic:</strong> {gd.pathognomonic}
                  </div>
                  <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', fontSize: 12 }}>
                    <div>
                      <strong>Complications:</strong>
                      {Object.entries(gd.complication_distribution || {}).slice(0, 5).map(([k, v]) => (
                        <div key={k} style={{ color: '#555' }}>• {k}: {v}</div>
                      ))}
                    </div>
                    <div>
                      <strong>Treatments:</strong>
                      {Object.entries(gd.treatment_distribution || {}).slice(0, 5).map(([k, v]) => (
                        <div key={k} style={{ color: '#555' }}>• {k}: {v}</div>
                      ))}
                    </div>
                    <div>
                      <strong>Metrics:</strong>
                      <div>Avg Dx delay: {gd.avg_diagnosis_delay_months} mo</div>
                      <div>Avg ANC nadir: {gd.avg_anc_nadir} cells/µL</div>
                      <div>Fungal: {gd.pct_fungal}%</div>
                      <div>Granuloma: {gd.pct_granuloma}%</div>
                      {gd.bleeding_risk && <div>Bleeding: {gd.pct_bleeding}%</div>}
                      {gd.neurological_risk && <div>Neurological: {gd.pct_neurological}%</div>}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Tab 3 — Definitions */}
      {tab === 3 && definitions && (
        <div>
          {/* Gene entries */}
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Gene Definitions</h2>
          {genes.map(g => {
            const entry = definitions.gene_entries?.[g];
            if (!entry) return null;
            return (
              <div key={g} style={{
                border: `1.5px solid ${GENE_COLORS[g]}55`, borderRadius: 8,
                marginBottom: 16, overflow: 'hidden',
              }}>
                <div style={{ background: GENE_COLORS[g] + '22', padding: '8px 14px', display: 'flex', gap: 10, alignItems: 'center' }}>
                  <span style={{ fontWeight: 800, color: GENE_COLORS[g], fontSize: 15 }}>{g}</span>
                  <span style={{ fontSize: 11.5, color: '#555' }}>{entry.locus} · {entry.protein_size} · {entry.inheritance} · {entry.disease_name}</span>
                </div>
                <div style={{ padding: '10px 14px', fontSize: 12 }}>
                  <div style={{ marginBottom: 6 }}><strong>Pathognomonic:</strong> {entry.pathognomonic}</div>
                  <div style={{ marginBottom: 6 }}><strong>Treatment:</strong> {entry.treatment}</div>
                  <div style={{ marginBottom: 4 }}><strong>Key Features:</strong></div>
                  {(entry.key_features || []).map((f, i) => (
                    <div key={i} style={{ color: '#444', marginLeft: 10 }}>• {f}</div>
                  ))}
                  <div style={{ marginTop: 6 }}><strong>Key DDx:</strong></div>
                  {(entry.key_ddx || []).map((d, i) => (
                    <div key={i} style={{ color: '#555', marginLeft: 10, marginTop: 2 }}>• {d}</div>
                  ))}
                </div>
              </div>
            );
          })}

          {/* Glossary */}
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12, marginTop: 24 }}>Phagocyte Deficiency Glossary</h2>
          {Object.entries(definitions.phagocyte_glossary || {}).map(([term, def]) => (
            <div key={term} style={{
              background: '#f5f5f5', border: '1.5px solid #ddd',
              borderRadius: 8, padding: '10px 14px', marginBottom: 10,
            }}>
              <div style={{ fontWeight: 700, color: '#b71c1c', marginBottom: 4 }}>{term}</div>
              <div style={{ fontSize: 12.5, color: '#444', lineHeight: 1.5 }}>{def}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
