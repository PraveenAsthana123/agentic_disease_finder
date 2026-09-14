'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-bile-acid-sterol-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'DHCR7':  '#1a237e',  // deep indigo — SLO most common sterol synth defect; 7-DHC elevated PATHOGNOMONIC; 2nd-3rd toe syndactyly 97%
  'CYP27A1':'#880e4f',  // deep magenta — CTX; cholestanol elevated PATHOGNOMONIC; Achilles xanthomas; CDCA reverses neurodegeneration
  'HSD3B7': '#1b5e20',  // dark green — CBAS1; neonatal cholestasis; NORMAL GGT; 3β-OH-Δ5 precursors; cholic acid cures
  'AKR1D1': '#b71c1c',  // deep red — CBAS2; allo-bile acids PATHOGNOMONIC; severe neonatal hepatitis; VPA ABSOLUTELY CI
  'CYP7B1': '#e65100',  // deep orange — CBAS3/SPG5; dual phenotype; oxysterols; CDCA; normal GGT
  'AMACR':  '#4a148c',  // deep purple — CBAS4; adult-onset; pristanic+THCA elevated; VLCFA NORMAL; dietary restriction
  'SC5D':   '#006064',  // dark teal — lathosterolosis; lathosterol elevated; ultra-rare <20 cases; overlap with SLO
  'EBP':    '#3e2723',  // dark brown — CDPX2; X-linked dominant; males lethal in utero; Blaschko ichthyosis; stippled epiphyses resolves
};

const GENE_INFO = {
  'DHCR7':  { full: 'DHCR7 / 7-Dehydrocholesterol Reductase / 475aa', locus: '11q13.4', size: '475 aa / 54 kDa (9-TM ER membrane)', inh: 'AR', disease: 'SMITH-LEMLI-OPITZ (SLO) — most common cholesterol synthesis defect (1:15,000-30,000); 7-DHC elevated PATHOGNOMONIC (>10 µg/mL; normal <5); 2nd-3rd toe syndactyly PATHOGNOMONIC (>97%); ASD features 60-70%; photosensitivity (7-DHC photodegradation → toxic oxysterols); 46XY genital ambiguity; low cholesterol; holoprosencephaly in severe; IVS8-1G>C most common European mutation; STATINS ABSOLUTELY CI (worsen cholesterol depletion); TREATMENT: cholesterol supplementation + photoprotection; PRENATAL CLUE: low maternal uE3 on triple screen → reflex 7-DHC testing' },
  'CYP27A1':{ full: 'CYP27A1 / Sterol 27-Hydroxylase / 531aa', locus: '2q35', size: '531 aa / 60 kDa (mitochondrial inner membrane CYP450)', inh: 'AR', disease: 'CEREBROTENDINOUS XANTHOMATOSIS (CTX) — cholestanol elevated PATHOGNOMONIC; Achilles tendon xanthomas PATHOGNOMONIC (cholestanol deposits); infantile diarrhoea + cataracts (first signs); dentate nucleus T2-hyperintensities PATHOGNOMONIC on MRI; progressive neurodegeneration (ataxia, dementia, epilepsy, pyramidal); Moroccan Jewish founder pGln403Arg (1:108 carrier); CDCA 750 mg/day REVERSES neurological deterioration if started early; average 15-20 year diagnostic delay; NBS not detected; statins add-on (reduce substrate)' },
  'HSD3B7': { full: 'HSD3B7 / 3β-Hydroxy-Δ5-C27-Steroid Oxidoreductase / 369aa', locus: '16p11.2', size: '369 aa / 42 kDa (microsomal ER, NAD-dependent)', inh: 'AR', disease: 'CBAS1 (CONGENITAL BILE ACID SYNTHESIS DEFECT TYPE 1) — 3β-hydroxy-Δ5 cholenoic acids in urine PATHOGNOMONIC (urine LSIMS/mass spectrometry); NORMAL GGT in cholestasis (key discriminator); neonatal cholestasis + fat-soluble vitamin deficiency; fat-soluble vitamin K → coagulopathy urgent; vitamin K IV if coagulopathy; CHOLIC ACID 5-15 mg/kg/day → EXCELLENT PROGNOSIS (nearly curative); UDCA NOT effective; normal serum bile acid panel does NOT exclude CBAS; diagnose by urine mass spectrometry' },
  'AKR1D1': { full: 'AKR1D1 / Δ4-3-Oxosteroid 5β-Reductase / 326aa', locus: '7q33', size: '326 aa / 37 kDa (cytoplasmic NADPH-dependent aldo-keto reductase)', inh: 'AR', disease: 'CBAS2 (CONGENITAL BILE ACID SYNTHESIS DEFECT TYPE 2) — allo-bile acids (5α-reduced stereoisomers) in urine PATHOGNOMONIC; MORE SEVERE than CBAS1 — severe neonatal hepatitis → cirrhosis; VPA ABSOLUTELY CONTRAINDICATED (hepatotoxic + impairs pathway); normal GGT in cholestasis; cholic acid treatment — prognosis worse than CBAS1; liver transplant more likely needed; fat-soluble vitamins urgently; LSIMS/MS-MS specialist lab required for diagnosis; standard serum bile acids mislead' },
  'CYP7B1': { full: 'CYP7B1 / Oxysterol 7α-Hydroxylase / 506aa', locus: '8q12.3', size: '506 aa / 57 kDa (microsomal ER CYP450)', inh: 'AR', disease: 'DUAL DISEASE — same biallelic LOF → neonatal liver failure (CBAS3) OR adult-onset hereditary spastic paraplegia (SPG5); CBAS3: severe neonatal liver failure + normal GGT + oxysterols (25-OH-cholesterol + 27-OH-cholesterol) elevated; SPG5: pure HSP + cerebellar ataxia + white matter lesions; ONLY HSP WITH BIOMARKER (plasma oxysterols) AND POTENTIAL TREATMENT (CDCA); oxysterol accumulation → corticospinal tract degeneration (neuronal apoptosis); CDCA treats both phenotypes (reduces oxysterol accumulation via FXR); plasma oxysterol panel mandatory in all HSP workups' },
  'AMACR':  { full: 'AMACR / 2-Methylacyl-CoA Racemase / 382aa', locus: '5p13.2', size: '382 aa / 42 kDa (peroxisomal + mitochondrial dual localisation)', inh: 'AR', disease: 'CBAS4 — ADULT-ONSET (unlike CBAS1-3 neonatal); pristanic acid + THCA/DHCA elevated (C27 bile acid intermediates); sensorimotor neuropathy + cerebellar ataxia + retinitis pigmentosa; VLCFA NORMAL (key DDx from X-ALD and ZSD — both have elevated VLCFA); liver disease variable; AMACR also cancer biomarker (overexpressed in prostate cancer — opposite mechanism from deficiency disease); TREATMENT: dietary pristanic/phytanic restriction (dairy fat, ruminant fat) + bile acid replacement; bile acid therapy: cholic acid/CDCA suppresses CYP7A1 → reduces THCA/DHCA production' },
  'SC5D':   { full: 'SC5D / Sterol-C5-Desaturase / 299aa', locus: '11q23.3', size: '299 aa / 35 kDa (ER membrane, FAD-dependent)', inh: 'AR', disease: 'LATHOSTEROLOSIS — EXTREMELY RARE (<20 cases worldwide); lathosterol elevated PATHOGNOMONIC (lathosterol:cholesterol ratio >0.05); 7-DHC LOW (SC5D is upstream of DHCR7 — cannot form 7-DHC without SC5D); cholesterol LOW; microcephaly + liver disease + cleft palate + cataracts; overlaps with SLO clinically; KEY DDx from SLO: SLO = 7-DHC HIGH; Lathosterolosis = lathosterol HIGH + 7-DHC LOW; full plasma sterol profile (GC-MS) required — not just 7-DHC; STATINS AVOID; cholesterol supplementation (very limited evidence)' },
  'EBP':    { full: 'EBP / Emopamil-Binding Protein / Δ8-Δ7 Sterol Isomerase / 230aa', locus: 'Xp11.23', size: '230 aa / 25 kDa (4-TM ER membrane)', inh: 'XLD', disease: 'CDPX2 (CHONDRODYSPLASIA PUNCTATA TYPE 2) — X-LINKED DOMINANT; hemizygous MALES VIRTUALLY ALWAYS LETHAL IN UTERO (rare mosaic males survive); virtually all survivors are FEMALE; 8-dehydrocholesterol elevated (8-DHC on GC-MS sterol profile); CLINICAL TRIAD: (1) ichthyosis following Blaschko\'s lines (X-inactivation mosaicism); (2) stippled epiphyses = calcified cartilaginous epiphyses — RESOLVES with age; (3) cataracts; normal VLCFA (DDx from RCDP/ZSD); DDx from warfarin embryopathy (same stippling + nasal hypoplasia; normal sterol profile in warfarin); 50% recurrence risk per pregnancy for affected females' },
};

function GeneChip({ gene }) {
  const col = GENE_COLORS[gene] || '#555';
  return (
    <span style={{ background: col, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 12, fontWeight: 700, margin: '0 2px' }}>
      {gene}
    </span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 120 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8' }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HeredBileAcidSterolAtlasPage() {
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
  const accent = '#38bdf8';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>
      <h1 style={{ color: accent, marginBottom: 4 }}>🧬 Hereditary Bile Acid Synthesis &amp; Sterol Biosynthesis Atlas</h1>
      <p style={{ color: '#94a3b8', marginBottom: 16, fontSize: 13 }}>
        8-Gene Reference: DHCR7 · CYP27A1 · HSD3B7 · AKR1D1 · CYP7B1 · AMACR · SC5D · EBP &nbsp;|&nbsp; 320 patients (8×40) &nbsp;|&nbsp; Seeds 2678–2685
        <br />
        <span style={{ color: '#64748b' }}>
          Sterol synthesis defects (DHCR7/SLO, SC5D/Lathosterolosis, EBP/CDPX2) &amp; Primary Bile Acid Synthesis Defects (HSD3B7/CBAS1, AKR1D1/CBAS2, CYP7B1/CBAS3+SPG5, AMACR/CBAS4) &amp; CYP27A1/CTX
        </span>
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{ background: tab === t ? accent : card, color: tab === t ? '#0f172a' : '#94a3b8',
              border: 'none', borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontWeight: 700, fontSize: 13 }}>
            {t}
          </button>
        ))}
      </div>

      {loading && <div style={{ color: '#64748b' }}>Loading…</div>}
      {err && <div style={{ color: '#ef4444' }}>Error: {err}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40 cohort" />
            <MetricCard label="Genes" value={overview.genes?.length} sub="Sterol + Bile Acid synthesis" />
            <MetricCard label="Seeds" value={`${overview.seeds?.[0]}–${overview.seeds?.[overview.seeds?.length-1]}`} sub="Deterministic" />
            <MetricCard label="NBS Detected" value="DHCR7 only*" sub="*expanded NBS only" warn />
            <MetricCard label="Dual-Phenotype" value="CYP7B1" sub="CBAS3 + SPG5" />
            <MetricCard label="XLD Male-Lethal" value="EBP" sub="CDPX2 hemizygous fatal" warn />
          </div>

          {/* Pathway classification */}
          <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: accent, marginBottom: 10 }}>Pathway Classification</h3>
            {overview.pathway_classification?.map((cls, i) => (
              <div key={i} style={{ marginBottom: 8, padding: '8px 12px', background: '#0f172a', borderRadius: 6, borderLeft: `3px solid ${accent}`, fontSize: 13 }}>
                {cls}
              </div>
            ))}
          </div>

          {/* Gene summary table */}
          <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 20, overflowX: 'auto' }}>
            <h3 style={{ color: accent, marginBottom: 10 }}>Gene Summary (320 Patients)</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
                  {['Gene','N','Onset (yrs)','Chol (mg/dL)','Liver%','Neuro%','NBS%','Tx Resp%','Path Marker%','GGT Normal'].map(h => (
                    <th key={h} style={{ padding: '4px 8px', textAlign: 'left' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {overview.gene_summaries?.map((s, i) => (
                  <tr key={s.gene} style={{ borderBottom: '1px solid #1e293b', background: i % 2 === 0 ? '#0f172a' : 'transparent' }}>
                    <td style={{ padding: '4px 8px' }}><GeneChip gene={s.gene} /></td>
                    <td style={{ padding: '4px 8px', color: '#94a3b8' }}>{s.n}</td>
                    <td style={{ padding: '4px 8px', color: '#94a3b8' }}>{s.mean_onset_years}</td>
                    <td style={{ padding: '4px 8px', color: s.mean_cholesterol_mg_dL < 100 ? '#ef4444' : '#94a3b8' }}>{s.mean_cholesterol_mg_dL}</td>
                    <td style={{ padding: '4px 8px', color: s.liver_involvement_pct >= 80 ? '#f97316' : '#94a3b8' }}>{s.liver_involvement_pct}%</td>
                    <td style={{ padding: '4px 8px', color: s.neurological_pct >= 80 ? '#a78bfa' : '#94a3b8' }}>{s.neurological_pct}%</td>
                    <td style={{ padding: '4px 8px', color: s.nbs_detected_pct === 0 ? '#ef4444' : '#22c55e' }}>{s.nbs_detected_pct}%</td>
                    <td style={{ padding: '4px 8px', color: s.treatment_responsive_pct >= 80 ? '#22c55e' : '#94a3b8' }}>{s.treatment_responsive_pct}%</td>
                    <td style={{ padding: '4px 8px', color: s.pathognomonic_marker_pct >= 90 ? '#38bdf8' : '#94a3b8' }}>{s.pathognomonic_marker_pct}%</td>
                    <td style={{ padding: '4px 8px', color: s.ggt_normal_in_cholestasis ? '#22c55e' : '#64748b' }}>
                      {s.ggt_normal_in_cholestasis ? '✓ Normal' : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Treatment key */}
          <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: accent, marginBottom: 10 }}>Primary Treatment by Gene</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 8 }}>
              {overview.primary_treatment && Object.entries(overview.primary_treatment).map(([gene, tx]) => (
                <div key={gene} style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', borderLeft: `3px solid ${GENE_COLORS[gene] || '#555'}` }}>
                  <div style={{ marginBottom: 4 }}><GeneChip gene={gene} /></div>
                  <div style={{ fontSize: 12, color: '#94a3b8' }}>{tx}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Special flags */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {overview.male_lethal_gene && (
              <div style={{ background: '#450a0a', border: '1px solid #7f1d1d', borderRadius: 8, padding: '10px 14px', fontSize: 12, maxWidth: 400 }}>
                <div style={{ color: '#f87171', fontWeight: 700 }}>⚠ Male Lethality</div>
                <div style={{ color: '#94a3b8', marginTop: 4 }}>{overview.male_lethal_gene}</div>
              </div>
            )}
            {overview.dual_phenotype_genes?.map(dp => (
              <div key={dp} style={{ background: '#1e1b4b', border: '1px solid #3730a3', borderRadius: 8, padding: '10px 14px', fontSize: 12, maxWidth: 400 }}>
                <div style={{ color: '#818cf8', fontWeight: 700 }}>⊕ Dual Phenotype Gene</div>
                <div style={{ color: '#94a3b8', marginTop: 4 }}>{dp}</div>
              </div>
            ))}
            {overview.most_common && (
              <div style={{ background: '#0c4a6e', border: '1px solid #0369a1', borderRadius: 8, padding: '10px 14px', fontSize: 12, maxWidth: 400 }}>
                <div style={{ color: '#38bdf8', fontWeight: 700 }}>Most Common</div>
                <div style={{ color: '#94a3b8', marginTop: 4 }}>{overview.most_common}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <div style={{ display: 'grid', gap: 16 }}>
            {breakdown.gene_entries?.map(entry => {
              const info = GENE_INFO[entry.gene] || {};
              return (
                <div key={entry.gene} style={{ background: card, borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[entry.gene] || '#555'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
                    <GeneChip gene={entry.gene} />
                    <span style={{ color: '#94a3b8', fontSize: 12 }}>{info.full || entry.gene}</span>
                    <span style={{ color: '#64748b', fontSize: 11 }}>{entry.locus || info.locus}</span>
                    <span style={{ background: '#164e63', color: '#7dd3fc', borderRadius: 4, padding: '1px 6px', fontSize: 11 }}>{info.inh}</span>
                  </div>
                  <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8, lineHeight: 1.5 }}>{info.disease}</div>
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                    {[
                      { l: 'N', v: entry.patients_n },
                      { l: 'Onset (yrs)', v: entry.mean_onset_years },
                      { l: 'Chol (mg/dL)', v: entry.mean_cholesterol_mg_dL },
                      { l: 'Liver%', v: `${entry.liver_involvement_pct}%` },
                      { l: 'Neuro%', v: `${entry.neurological_pct}%` },
                      { l: 'NBS%', v: `${entry.nbs_detected_pct}%` },
                      { l: 'Tx Resp%', v: `${entry.treatment_responsive_pct}%` },
                    ].map(({ l, v }) => (
                      <div key={l} style={{ background: '#0f172a', borderRadius: 4, padding: '4px 8px', fontSize: 11 }}>
                        <span style={{ color: '#64748b' }}>{l}: </span>
                        <span style={{ color: '#e2e8f0', fontWeight: 700 }}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
            {breakdown.gene_entries?.map(e => (
              <button key={e.gene} onClick={() => setSelGene(selGene === e.gene ? null : e.gene)}
                style={{ background: selGene === e.gene ? GENE_COLORS[e.gene] : card,
                  color: '#fff', border: `1px solid ${GENE_COLORS[e.gene] || '#555'}`,
                  borderRadius: 6, padding: '4px 12px', cursor: 'pointer', fontSize: 12, fontWeight: 700 }}>
                {e.gene}
              </button>
            ))}
          </div>
          {selGene && (() => {
            const entry = breakdown.gene_entries?.find(e => e.gene === selGene);
            if (!entry) return null;
            const info = GENE_INFO[selGene] || {};
            const sections = [
              { label: 'Protein / Locus / Size', text: `${info.full || selGene} | ${entry.locus || info.locus} | ${entry.protein_size}` },
              { label: 'Inheritance & Mechanism', text: entry.inheritance_summary },
              { label: 'Disease Features', text: entry.disease_category },
              { label: 'Metabolic Pathway', text: entry.disease_pathway },
              { label: 'Pathognomonic Findings', text: entry.pathognomonic },
              { label: 'Treatment', text: entry.treatment },
            ];
            return (
              <div style={{ background: card, borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[selGene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
                  <GeneChip gene={selGene} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{info.full}</span>
                  <span style={{ background: '#164e63', color: '#7dd3fc', borderRadius: 4, padding: '1px 6px', fontSize: 11 }}>{info.inh}</span>
                </div>
                {sections.map(({ label, text }) => (
                  <div key={label} style={{ marginBottom: 16 }}>
                    <div style={{ color: accent, fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{label}</div>
                    <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.7, whiteSpace: 'pre-wrap', background: '#0f172a', padding: '8px 12px', borderRadius: 6 }}>
                      {text?.split('; ').map((line, i) => (
                        <div key={i} style={{ marginBottom: 2 }}>• {line}</div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            );
          })()}
          {!selGene && (
            <div style={{ color: '#64748b', fontSize: 13 }}>Select a gene above to view full clinical atlas entry.</div>
          )}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ display: 'grid', gap: 12 }}>
            {Object.entries(definitions.glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: card, borderRadius: 8, padding: '12px 16px' }}>
                <div style={{ color: accent, fontWeight: 700, fontSize: 14, marginBottom: 6 }}>{term}</div>
                <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.7 }}>
                  {String(def).split('; ').map((line, i) => (
                    <div key={i} style={{ marginBottom: 2 }}>• {line}</div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
