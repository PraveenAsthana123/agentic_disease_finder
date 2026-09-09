'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-primary-immunodeficiency-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  IL2RG:   '#1565c0',  // deep blue    — SCID-X1, X-linked most common SCID
  ADA:     '#0d47a1',  // navy         — ADA-SCID, complete lymphopenia
  RAG1:    '#6a1b9a',  // deep purple  — RAG1-SCID / Omenn syndrome
  BTK:     '#b71c1c',  // deep red     — XLA Bruton, absent B cells
  AIRE:    '#2e7d32',  // deep green   — APECED/APS-1, candidiasis triad
  LRBA:    '#e65100',  // burnt orange — LRBA deficiency, abatacept
  CTLA4:   '#4a148c',  // dark purple  — CTLA4 haploinsufficiency, AD
  PIK3CD:  '#004d40',  // dark teal    — APDS, leniolisib FDA 2023
};

const GENE_INFO = {
  IL2RG:   { full: 'IL2RG / 369aa',   locus: 'Xq13.1',   size: '369 aa / 42 kDa',   inh: 'XLR', disease: 'X-linked SCID (SCID-X1) — T-B+NK- IMMUNOPHENOTYPE PATHOGNOMONIC (absent T + NK; normal but non-functional B cells); MOST COMMON SCID IN WESTERN COUNTRIES (45-50%); TREC absent on newborn screen; NO LIVE VACCINES EVER (BCG-osis = disseminated fatal BCG before diagnosis); Gene therapy: lentiviral IL2RG (CARTEYVA) FDA curative; HSCT preferred if matched sibling donor; without treatment median survival <1yr' },
  ADA:     { full: 'ADA / 363aa',     locus: '20q13.12',  size: '363 aa / 41 kDa',   inh: 'AR',  disease: 'ADA-SCID — T-B-NK- IMMUNOPHENOTYPE (complete absence all lymphocytes — WORST of all SCIDs); RIB-COSTAL JUNCTION CUPPING ("rachitic rosary") on CXR PATHOGNOMONIC; PEG-ADA (REVCOVI) enzyme replacement = bridge, non-curative; Strimvelis gene therapy FDA curative; bone marrow ADA activity + RBC dATP:ATP ratio = diagnostic; partial ADA deficiency: late-onset adult form' },
  RAG1:    { full: 'RAG1 / 1043aa',   locus: '11p13',     size: '1043 aa / 119 kDa', inh: 'AR',  disease: 'RAG1-SCID / Omenn Syndrome — COMPLETE NULL: T-B-NK+ SCID (NK preserved = KEY DDx from IL2RG/ADA); HYPOMORPHIC: OMENN SYNDROME = ERYTHRODERMA + ELEVATED IgE + EOSINOPHILIA + ABSENT NORMAL IMMUNOGLOBULINS PATHOGNOMONIC; RAG1+RAG2 head-to-head on 11p13 (test both); HSCT curative; no live vaccines; IVIG + prophylaxis pre-HSCT' },
  BTK:     { full: 'BTK / 659aa',     locus: 'Xq22.1',    size: '659 aa / 76 kDa',   inh: 'XLR', disease: 'X-linked Agammaglobulinemia (XLA / Bruton) — PERIPHERAL B CELLS <1% PATHOGNOMONIC (CD19/CD20 by flow); ALL IMMUNOGLOBULIN ISOTYPES VIRTUALLY ZERO; absent tonsils/adenoids; recurrent encapsulated bacteria (Strep pneumo, H. influenzae) from age 6-18mo when maternal IgG wanes; ENTEROVIRAL ENCEPHALITIS LIFE-THREATENING; IVIG lifelong (trough IgG >600-800 mg/dL); NO live vaccines (OPV strictly contraindicated)' },
  AIRE:    { full: 'AIRE / 552aa',    locus: '21q22.3',   size: '552 aa / 58 kDa',   inh: 'AR',  disease: 'APECED/APS-1 — MUCOCUTANEOUS CANDIDIASIS (FIRST, age 1-5yr) + HYPOPARATHYROIDISM (hypocalcaemia/tetany) + ADRENAL INSUFFICIENCY = TRIAD PATHOGNOMONIC (2 of 3 = diagnosis); Anti-IFN-ω antibodies = diagnostic biomarker (>95% sensitivity); Finnish founder p.Arg257Ter; CALCIUM MONITORING + calcitriol mandatory (hypocalcaemic seizures); ADRENAL CRISIS: hydrocortisone stress-dosing mandatory' },
  LRBA:    { full: 'LRBA / 2863aa',   locus: '4q31.3',    size: '2863 aa / 319 kDa', inh: 'AR',  disease: 'LRBA Deficiency — HYPOGAMMAGLOBULINEMIA + AUTOIMMUNITY (cytopenias/enteropathy/hepatitis/GLILD) COMBINED = LRBA PATHOGNOMONIC DDx from CVID; CTLA4 trafficking failure (CTLA4 fails to recycle to cell surface → T-reg dysfunction); ABATACEPT (CTLA4-Ig): HIGHLY EFFECTIVE, reverses autoimmunity within weeks = diagnostic response; IBD-like enteropathy prominent (50-70%); CTLA4 flow cytometry on T-regs: reduced expression = screening test' },
  CTLA4:   { full: 'CTLA4 / 223aa',   locus: '2q33.2',    size: '223 aa / 25 kDa',   inh: 'AD',  disease: 'CTLA4 Haploinsufficiency (CTLA4-H) — SPLENOMEGALY + LYMPHADENOPATHY + AUTOIMMUNE CYTOPENIAS + HYPOGAMMAGLOBULINEMIA + LYMPHOCYTIC INFILTRATION LUNGS/GUT/BRAIN PATHOGNOMONIC; GLILD (lymphocytic interstitial lung disease) — CHECK CTLA4 before immunosuppression for "sarcoid"; PARADOXICAL: recurrent infections despite splenomegaly; ABATACEPT: highly effective — dramatic response is diagnostic signal; AD haploinsufficiency — one normal allele insufficient' },
  PIK3CD:  { full: 'PIK3CD / 1044aa', locus: '1p36.22',   size: '1044 aa / 119 kDa', inh: 'AD GOF', disease: 'Activated PI3Kδ Syndrome (APDS/APDS1) — LYMPHOPROLIFERATION + EBV/CMV SUSCEPTIBILITY (chronic active infection, lymphoma up to 20%) + RECURRENT SINOPULMONARY INFECTIONS + HYPOGAMMAGLOBULINEMIA = PATHOGNOMONIC; LENIOLISIB (Joenja) FDA 2023 — first selective PI3Kδ inhibitor, FIRST TARGETED THERAPY FOR APDS; EBV viral load ANNUAL MONITORING MANDATORY (lymphoma risk); bronchiectasis by 2nd decade; IVIG + antiviral prophylaxis' },
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

export default function HeredPIDAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, color: '#1565c0', fontWeight: 700 }}>Loading Hereditary Primary Immunodeficiency Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#b71c1c' }}>Error: {error}</div>;
  if (!overview) return null;

  const genes = Object.keys(GENE_COLORS);

  return (
    <div style={{ padding: '24px 32px', fontFamily: 'system-ui, sans-serif', maxWidth: 1200 }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#1565c0', marginBottom: 4 }}>
          🧬 Hereditary Primary Immunodeficiency Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#555' }}>
          Complete 8-Gene Atlas — IL2RG · ADA · RAG1 · BTK · AIRE · LRBA · CTLA4 · PIK3CD —
          320 patients (8 × 40, seeds 2310-2317)
        </div>
      </div>

      {/* Stat cards */}
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 22 }}>
        <StatCard label="Total Patients" value={overview.n_patients} color="#1565c0" />
        <StatCard label="Genes" value={overview.n_genes} color="#2e7d32" />
        <StatCard label="Seed Range" value="2310-2317" color="#e65100" />
        <StatCard label="SCID Genes" value={overview.immunodeficiency_categories?.SCID?.length ?? 3} sub="IL2RG · ADA · RAG1" color="#b71c1c" />
        <StatCard label="Immune Dysreg" value={overview.immunodeficiency_categories?.['Immune dysregulation']?.length ?? 4} sub="AIRE · LRBA · CTLA4 · PIK3CD" color="#4a148c" />
        <StatCard label="Agammaglobulin" value={overview.immunodeficiency_categories?.Agammaglobulinemia?.length ?? 1} sub="BTK (XLA)" color="#004d40" />
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #e0e0e0' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', border: 'none', cursor: 'pointer', fontWeight: 700, fontSize: 13,
            background: tab === i ? '#1565c0' : '#f5f5f5',
            color: tab === i ? '#fff' : '#555',
            borderRadius: '6px 6px 0 0',
          }}>{t}</button>
        ))}
      </div>

      {/* Tab 0: Overview */}
      {tab === 0 && (
        <div>
          {/* Category groupings */}
          <div style={{ marginBottom: 20 }}>
            <h3 style={{ fontSize: 15, fontWeight: 700, color: '#1565c0', marginBottom: 10 }}>Disease Categories</h3>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              {overview.immunodeficiency_categories && Object.entries(overview.immunodeficiency_categories).map(([cat, gns]) => (
                <div key={cat} style={{ background: '#e3f2fd', borderRadius: 8, padding: '10px 16px', minWidth: 200 }}>
                  <div style={{ fontWeight: 700, color: '#1565c0', fontSize: 13, marginBottom: 6 }}>{cat}</div>
                  {gns.map(g => (
                    <div key={g} style={{ fontSize: 12, color: '#333', marginBottom: 2 }}>
                      <span style={{ background: GENE_COLORS[g] + '22', color: GENE_COLORS[g], borderRadius: 3, padding: '1px 5px', fontWeight: 700, marginRight: 4 }}>{g}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Key clinical pearls */}
          <div style={{ background: '#e8f5e9', borderRadius: 10, padding: '16px 20px', marginBottom: 20 }}>
            <h3 style={{ fontSize: 15, fontWeight: 700, color: '#2e7d32', marginBottom: 10 }}>🔑 Key Clinical Pearls</h3>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(overview.key_clinical_pearls || []).map((p, i) => (
                <li key={i} style={{ fontSize: 13, color: '#333', marginBottom: 6 }}>{p}</li>
              ))}
            </ul>
          </div>

          {/* Emergency flags */}
          {overview.clinical_emergency_flags && (
            <div style={{ background: '#fff3e0', borderRadius: 10, padding: '16px 20px', marginBottom: 20 }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: '#e65100', marginBottom: 10 }}>🚨 Clinical Emergency Flags</h3>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {overview.clinical_emergency_flags.map((f, i) => (
                  <li key={i} style={{ fontSize: 12, color: '#333', marginBottom: 5 }}>{f}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Diagnostic algorithm */}
          {overview.diagnostic_algorithm && (
            <div style={{ background: '#f3e5f5', borderRadius: 10, padding: '16px 20px' }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: '#6a1b9a', marginBottom: 10 }}>🔬 Diagnostic Algorithm</h3>
              <ol style={{ margin: 0, paddingLeft: 20 }}>
                {overview.diagnostic_algorithm.map((s, i) => (
                  <li key={i} style={{ fontSize: 12, color: '#333', marginBottom: 4 }}>{s}</li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}

      {/* Tab 1: Gene Table */}
      {tab === 1 && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1565c0', color: '#fff' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Gene</th>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Locus / Size</th>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Inheritance</th>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Disease / Key Feature</th>
              </tr>
            </thead>
            <tbody>
              {genes.map((g, i) => {
                const info = GENE_INFO[g];
                return (
                  <tr key={g} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff', borderBottom: '1px solid #e0e0e0' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 800, color: GENE_COLORS[g], fontSize: 14 }}>{g}</td>
                    <td style={{ padding: '10px 12px', color: '#555' }}>
                      <div style={{ fontWeight: 600 }}>{info.locus}</div>
                      <div style={{ fontSize: 11, color: '#888' }}>{info.size}</div>
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      <Badge text={info.inh} color={GENE_COLORS[g]} />
                    </td>
                    <td style={{ padding: '10px 12px', color: '#333', lineHeight: 1.5 }}>{info.disease}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Tab 2: Clinical Atlas */}
      {tab === 2 && breakdown && (
        <div>
          {genes.map(g => {
            const gb = breakdown.gene_breakdown?.[g];
            if (!gb) return null;
            return (
              <div key={g} style={{
                border: `2px solid ${GENE_COLORS[g]}44`,
                borderRadius: 10, marginBottom: 18, padding: '14px 18px',
                borderLeft: `5px solid ${GENE_COLORS[g]}`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <span style={{ fontSize: 18, fontWeight: 800, color: GENE_COLORS[g] }}>{g}</span>
                  <Badge text={GENE_INFO[g].inh} color={GENE_COLORS[g]} />
                  <Badge text={GENE_INFO[g].locus} color="#555" />
                  <span style={{ fontSize: 12, color: '#888' }}>{gb.n_patients} pts</span>
                </div>
                <div style={{ fontSize: 12, color: '#444', marginBottom: 10, lineHeight: 1.5 }}>
                  {gb.immunodeficiency_category}
                </div>
                <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', fontSize: 12 }}>
                  {gb.complication_distribution && (
                    <div>
                      <div style={{ fontWeight: 700, color: '#555', marginBottom: 4 }}>Complications</div>
                      {Object.entries(gb.complication_distribution).slice(0, 4).map(([k, v]) => (
                        <div key={k} style={{ color: '#444' }}>{k}: <b>{v}</b></div>
                      ))}
                    </div>
                  )}
                  {gb.immunoglobulin_distribution && (
                    <div>
                      <div style={{ fontWeight: 700, color: '#555', marginBottom: 4 }}>Ig Level</div>
                      {Object.entries(gb.immunoglobulin_distribution).slice(0, 4).map(([k, v]) => (
                        <div key={k} style={{ color: '#444' }}>{k}: <b>{v}</b></div>
                      ))}
                    </div>
                  )}
                  {gb.severity_distribution && (
                    <div>
                      <div style={{ fontWeight: 700, color: '#555', marginBottom: 4 }}>Severity</div>
                      {Object.entries(gb.severity_distribution).map(([k, v]) => (
                        <div key={k} style={{ color: '#444' }}>{k}: <b>{v}</b></div>
                      ))}
                    </div>
                  )}
                  {gb.treatment_distribution && (
                    <div>
                      <div style={{ fontWeight: 700, color: '#555', marginBottom: 4 }}>Treatments</div>
                      {Object.entries(gb.treatment_distribution).slice(0, 4).map(([k, v]) => (
                        <div key={k} style={{ color: '#444' }}>{k}: <b>{v}</b></div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Tab 3: Definitions */}
      {tab === 3 && definitions && (
        <div>
          <h3 style={{ fontSize: 15, fontWeight: 700, color: '#1565c0', marginBottom: 12 }}>Gene Entries</h3>
          {Object.entries(definitions.gene_entries || {}).map(([gene, entry]) => (
            <div key={gene} style={{
              border: `1px solid ${GENE_COLORS[gene] || '#ccc'}44`,
              borderRadius: 8, padding: '12px 16px', marginBottom: 14,
              borderLeft: `4px solid ${GENE_COLORS[gene] || '#ccc'}`,
            }}>
              <div style={{ fontWeight: 800, color: GENE_COLORS[gene] || '#333', fontSize: 15, marginBottom: 4 }}>{gene}</div>
              <div style={{ fontSize: 12, color: '#666', marginBottom: 6 }}>
                {entry.protein_size} · {entry.locus} · {entry.inheritance}
              </div>
              <div style={{ fontSize: 12, color: '#333', marginBottom: 4 }}><b>Disease:</b> {entry.disease_name}</div>
              <div style={{ fontSize: 12, color: '#333', marginBottom: 4 }}><b>Pathognomonic:</b> {entry.pathognomonic}</div>
              {entry.key_features && (
                <ul style={{ margin: '4px 0', paddingLeft: 18, fontSize: 12, color: '#444' }}>
                  {entry.key_features.slice(0, 4).map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              )}
            </div>
          ))}

          {definitions.immunology_glossary && (
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: '#2e7d32', marginBottom: 10, marginTop: 20 }}>Immunology Glossary</h3>
              {Object.entries(definitions.immunology_glossary).map(([term, def]) => (
                <div key={term} style={{ background: '#e8f5e9', borderRadius: 6, padding: '10px 14px', marginBottom: 10 }}>
                  <div style={{ fontWeight: 700, color: '#2e7d32', fontSize: 13, marginBottom: 4 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#333', lineHeight: 1.5 }}>{def}</div>
                </div>
              ))}
            </div>
          )}

          {definitions.treatment_glossary && (
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: '#e65100', marginBottom: 10, marginTop: 20 }}>Treatment Glossary</h3>
              {Object.entries(definitions.treatment_glossary).map(([term, def]) => (
                <div key={term} style={{ background: '#fff3e0', borderRadius: 6, padding: '10px 14px', marginBottom: 10 }}>
                  <div style={{ fontWeight: 700, color: '#e65100', fontSize: 13, marginBottom: 4 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#333', lineHeight: 1.5 }}>{def}</div>
                </div>
              ))}
            </div>
          )}

          {definitions.diagnostic_tests && (
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: '#6a1b9a', marginBottom: 10, marginTop: 20 }}>Diagnostic Tests</h3>
              {Object.entries(definitions.diagnostic_tests).map(([term, def]) => (
                <div key={term} style={{ background: '#f3e5f5', borderRadius: 6, padding: '10px 14px', marginBottom: 10 }}>
                  <div style={{ fontWeight: 700, color: '#6a1b9a', fontSize: 13, marginBottom: 4 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#333', lineHeight: 1.5 }}>{def}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
