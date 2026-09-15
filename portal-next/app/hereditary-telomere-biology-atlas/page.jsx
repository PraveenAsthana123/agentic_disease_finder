'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-telomere-biology-atlas';
const TABS = ['Overview', 'Gene Breakdown', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'DKC1':  '#b71c1c',  // deep red — X-linked DC; mucocutaneous triad PATHOGNOMONIC; most common DC (~40%)
  'TERC':  '#1a237e',  // deep navy — AD DC-2; telomerase RNA template; genetic anticipation HALLMARK
  'TERT':  '#004d40',  // dark teal — AD/AR; #1 familial IPF gene; telomerase reverse transcriptase
  'NHP2':  '#4a148c',  // deep purple — AR DC-5; H/ACA snoRNP; Val126Met; equal sex ratio DDx DKC1
  'NOP10': '#bf360c',  // dark orange — AR DC-4 ultra-rare; Arg34Trp; structural bridge NHP2-DKC1
  'WRAP53':'#006064',  // dark cyan — AR DC-6; Cajal body trafficking; normal TERC + short telomeres
  'ACD':   '#33691e',  // dark olive green — AD/AR; TPP1 TEL-patch; telomerase recruitment defect
  'PARN':  '#37474f',  // dark slate — AR HH/IPF; TERC deadenylase; PAPD5-inhibitor BCH001 target
};

const GENE_INFO = {
  'DKC1':  { full: 'DKC1 / Dyskerin / 514aa', locus: 'Xq28',    size: '514 aa / 58 kDa (H/ACA snoRNP catalytic subunit; pseudouridine synthase; TERC stability)', inh: 'XLR' },
  'TERC':  { full: 'TERC / Telomerase RNA / 451nt', locus: '3q26.2', size: '451 nt RNA (template CUAACCCUAAC; H/ACA box; CAB-box; pseudoknot/CR4-CR5)', inh: 'AD (haploinsufficiency)' },
  'TERT':  { full: 'TERT / Telomerase RT / 1132aa', locus: '5p15.33',size: '1132 aa / 127 kDa (TRBD+palm+finger+thumb; D712+D868 active site; androgen response element)', inh: 'AD / AR biallelic' },
  'NHP2':  { full: 'NHP2 / snoRNP-NHP2 / 153aa', locus: '5q35.3', size: '153 aa / 16 kDa (L7Ae motif; K-turn binding; H/ACA snoRNP scaffold)', inh: 'AR' },
  'NOP10': { full: 'NOP10 / snoRNP-NOP10 / 64aa',  locus: '15q14',  size: '64 aa / 7 kDa (smallest H/ACA snoRNP; NHP2-DKC1 bridge; Arg34Trp founder)', inh: 'AR' },
  'WRAP53':{ full: 'WRAP53 / TCAB1 / 548aa',        locus: '17p13.1',size: '548 aa / 62 kDa (WD40; CAB-box binding UGAG; Cajal body TERC trafficking)', inh: 'AR' },
  'ACD':   { full: 'ACD / TPP1 / 544aa',             locus: '16q22.1',size: '544 aa / 61 kDa (OB fold; TEL patch E169+E171 recruits TERT; POT1-TIN2 bridge)', inh: 'AD (TEL-patch) / AR biallelic (HH)' },
  'PARN':  { full: 'PARN / Deadenylase / 639aa',     locus: '16p13.12',size: '639 aa / 74 kDa (DEDDh ribonuclease; TERC 3\' oligo-A trimming; PAPD5 counter)', inh: 'AR' },
};

function GeneChip({ gene, active, onClick }) {
  return (
    <button
      onClick={() => onClick(gene)}
      style={{
        background: active ? GENE_COLORS[gene] : '#263238',
        color: '#fff',
        border: `2px solid ${GENE_COLORS[gene]}`,
        borderRadius: 8,
        padding: '6px 14px',
        margin: 4,
        cursor: 'pointer',
        fontWeight: active ? 700 : 400,
        fontSize: 13,
        transition: 'all 0.15s',
      }}
    >
      {gene}
    </button>
  );
}

function StatBadge({ label, value, color }) {
  return (
    <div style={{ background: '#1e2a31', border: `1px solid ${color || '#37474f'}`, borderRadius: 8, padding: '10px 14px', minWidth: 100, textAlign: 'center', margin: 4 }}>
      <div style={{ color: color || '#90a4ae', fontSize: 10, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#fff', fontSize: 20, fontWeight: 700 }}>{value}</div>
    </div>
  );
}

function Section({ title, children, color }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ color: color || '#90a4ae', fontWeight: 700, fontSize: 13, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>{title}</div>
      {children}
    </div>
  );
}

function ClinicalText({ text }) {
  if (!text) return null;
  return (
    <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12, fontSize: 12, color: '#cfd8dc', lineHeight: 1.7, whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
      {text}
    </div>
  );
}

export default function HTBAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [activeGene, setActiveGene] = useState('DKC1');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(String(e)); setLoading(false); });
  }, []);

  const genes = Object.keys(GENE_COLORS);

  if (loading) return <div style={{ background: '#102027', minHeight: '100vh', color: '#90a4ae', padding: 40, fontFamily: 'monospace' }}>Loading Hereditary-Telomere-Biology-Atlas…</div>;
  if (error) return <div style={{ background: '#102027', minHeight: '100vh', color: '#ef5350', padding: 40, fontFamily: 'monospace' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#102027', minHeight: '100vh', color: '#eceff1', fontFamily: 'monospace', padding: '24px 32px' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ fontSize: 22, fontWeight: 700, color: '#b71c1c', marginBottom: 4 }}>
          🧬 Hereditary-Telomere-Biology-Atlas
        </div>
        <div style={{ fontSize: 13, color: '#90a4ae' }}>
          Complete 8-Gene Dyskeratosis Congenita / Hoyeraal-Hreidarsson / IPF Spectrum Reference · 320 patients · seeds 2790-2797
        </div>
        <div style={{ fontSize: 11, color: '#546e7a', marginTop: 4 }}>
          DKC1 · TERC · TERT · NHP2 · NOP10 · WRAP53 · ACD · PARN
        </div>
      </div>

      {/* Gene chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', marginBottom: 20 }}>
        {genes.map(g => <GeneChip key={g} gene={g} active={activeGene === g} onClick={setActiveGene} />)}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#b71c1c' : '#1e2a31',
            color: '#fff', border: '1px solid #37474f', borderRadius: 6,
            padding: '6px 16px', cursor: 'pointer', fontSize: 13, fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', marginBottom: 20 }}>
            <StatBadge label="Total Patients" value={overview.total_patients} color="#b71c1c" />
            <StatBadge label="Genes" value={overview.genes?.length} color="#1a237e" />
            <StatBadge label="Avg Telomere Centile" value={`${overview.summary?.avg_telomere_centile_pct}%`} color="#004d40" />
            <StatBadge label="Seeds" value={overview.seeds} color="#37474f" />
          </div>

          <Section title="Telomere Maintenance Pathway (All 8 Steps)" color="#b71c1c">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12 }}>
              {overview.telomere_maintenance_pathway && Object.entries(overview.telomere_maintenance_pathway).map(([step, desc]) => (
                <div key={step} style={{ marginBottom: 8 }}>
                  <span style={{ color: '#b71c1c', fontWeight: 700, fontSize: 12 }}>{step.replace(/_/g,' ')}: </span>
                  <span style={{ color: '#cfd8dc', fontSize: 11 }}>{desc}</span>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Pathognomonic Signs by Gene" color="#1a237e">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12 }}>
              {overview.pathognomonic_signs && Object.entries(overview.pathognomonic_signs).map(([gene, sign]) => (
                <div key={gene} style={{ marginBottom: 8, display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                  <span style={{ color: GENE_COLORS[gene] || '#90a4ae', fontWeight: 700, fontSize: 12, minWidth: 70 }}>{gene}</span>
                  <span style={{ color: '#cfd8dc', fontSize: 11 }}>{sign}</span>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Critical Treatment Rules" color="#e65100">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12 }}>
              {overview.critical_treatment_rules && Object.entries(overview.critical_treatment_rules).map(([key, rule]) => {
                const isCI = rule.includes('CONTRAINDICATED') || rule.includes('CI');
                const isAlert = rule.includes('REDUCED') || rule.includes('investigational') || rule.includes('Investigational');
                return (
                  <div key={key} style={{ display: 'flex', gap: 10, marginBottom: 8, alignItems: 'flex-start' }}>
                    <span style={{
                      background: isCI ? '#b71c1c' : isAlert ? '#f57f17' : '#1b5e20',
                      color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 10, fontWeight: 700, minWidth: 80, textAlign: 'center'
                    }}>{isCI ? 'CI/CAUTION' : isAlert ? 'CAUTION' : 'KEY RULE'}</span>
                    <span style={{ color: '#90a4ae', fontWeight: 700, fontSize: 12, minWidth: 60 }}>{key.split('_')[0]}</span>
                    <span style={{ color: '#cfd8dc', fontSize: 11 }}>{rule}</span>
                  </div>
                );
              })}
            </div>
          </Section>

          <Section title="TERC Level Profiles (Diagnostic Clue)" color="#006064">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12 }}>
              {overview.terc_level_profiles && Object.entries(overview.terc_level_profiles).map(([profile, genes_desc]) => (
                <div key={profile} style={{ marginBottom: 8 }}>
                  <span style={{ color: '#006064', fontWeight: 700, fontSize: 12 }}>{profile.replace(/_/g,' ')}: </span>
                  <span style={{ color: '#cfd8dc', fontSize: 11 }}>{genes_desc}</span>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Per-Gene Summary" color="#1b5e20">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
              {overview.summary?.per_gene && Object.entries(overview.summary.per_gene).map(([gene, info]) => (
                <div key={gene} style={{
                  background: '#1e2a31', border: `1px solid ${GENE_COLORS[gene] || '#37474f'}`,
                  borderRadius: 8, padding: '10px 14px', minWidth: 200
                }}>
                  <div style={{ color: GENE_COLORS[gene], fontWeight: 700, fontSize: 14, marginBottom: 4 }}>{gene}</div>
                  <div style={{ color: '#90a4ae', fontSize: 11 }}>{info.locus} · {info.protein_size}</div>
                  <div style={{ color: '#cfd8dc', fontSize: 11, marginTop: 4 }}>n={info.n} · avg telo={info.avg_telomere_centile}%ile</div>
                  <div style={{ color: '#78909c', fontSize: 10, marginTop: 4 }}>{info.disease}</div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Cascade Testing" color="#f57f17">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 10, color: '#cfd8dc', fontSize: 12 }}>
              {overview.cascade_testing}
            </div>
          </Section>
        </div>
      )}

      {/* Gene Breakdown Tab */}
      {tab === 'Gene Breakdown' && breakdown && breakdown[activeGene] && (
        <div>
          <div style={{ marginBottom: 12 }}>
            <span style={{ color: GENE_COLORS[activeGene], fontSize: 18, fontWeight: 700 }}>{activeGene}</span>
            <span style={{ color: '#546e7a', fontSize: 12, marginLeft: 12 }}>{GENE_INFO[activeGene]?.locus} · {GENE_INFO[activeGene]?.inh}</span>
          </div>
          <div style={{ color: '#90a4ae', fontSize: 12, marginBottom: 16 }}>{GENE_INFO[activeGene]?.size}</div>

          <Section title="Key Facts" color={GENE_COLORS[activeGene]}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {breakdown[activeGene].key_facts?.map(f => (
                <span key={f} style={{ background: '#1e2a31', border: `1px solid ${GENE_COLORS[activeGene]}`, borderRadius: 4, padding: '3px 8px', fontSize: 10, color: '#cfd8dc' }}>{f}</span>
              ))}
            </div>
          </Section>

          <Section title="Inheritance & Gene Function" color="#90a4ae">
            <ClinicalText text={breakdown[activeGene].inheritance} />
          </Section>

          <Section title="Disease Category & Clinical Features" color={GENE_COLORS[activeGene]}>
            <ClinicalText text={breakdown[activeGene].disease_category} />
          </Section>

          <Section title="Disease Pathway & Mechanism" color="#1a237e">
            <ClinicalText text={breakdown[activeGene].disease_pathway} />
          </Section>

          <Section title="Pathognomonic Features & DDx" color="#b71c1c">
            <ClinicalText text={breakdown[activeGene].pathognomonic} />
          </Section>

          <Section title="Treatment" color="#1b5e20">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12, fontSize: 12, color: '#cfd8dc', lineHeight: 1.7 }}>
              {breakdown[activeGene].treatment}
            </div>
          </Section>

          <Section title="Sample Patients" color="#37474f">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#1e2a31' }}>
                    {['Patient ID', 'Age', 'Sex', 'Telomere Centile (%)', 'Key Phenotype'].map(h => (
                      <th key={h} style={{ padding: '6px 10px', color: '#90a4ae', textAlign: 'left', borderBottom: '1px solid #37474f' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown[activeGene].sample_patients?.map((p, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#1a2730' : '#1e2a31' }}>
                      <td style={{ padding: '5px 10px', color: '#cfd8dc' }}>{p.patient_id}</td>
                      <td style={{ padding: '5px 10px', color: '#90a4ae' }}>{p.age}</td>
                      <td style={{ padding: '5px 10px', color: '#90a4ae' }}>{p.sex}</td>
                      <td style={{ padding: '5px 10px', color: p.telomere_centile_pct < 1 ? '#ef5350' : '#ffb74d' }}>{p.telomere_centile_pct}%ile</td>
                      <td style={{ padding: '5px 10px', color: '#78909c', fontSize: 10 }}>{p.key_phenotype}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && overview && (
        <div>
          <Section title="8-Gene Telomere Biology Spectrum" color="#b71c1c">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 14 }}>
              <div style={{ color: '#78909c', fontSize: 11, marginBottom: 10, fontStyle: 'italic' }}>
                TERC → [DKC1+NHP2+NOP10: H/ACA snoRNP stabilises] → [PARN: trims 3′ oligo-A] → [WRAP53: Cajal body localisation] → TERT+TERC = active telomerase → [ACD/TPP1 TEL-patch: recruited to telomere] → TTGGGG added → [Shelterin/TRF1-TRF2-RAP1-TIN2-TPP1-POT1: caps telomere]
              </div>
              {genes.map(gene => (
                <div key={gene} style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 14, paddingBottom: 14, borderBottom: '1px solid #263238' }}>
                  <div style={{ minWidth: 70, color: GENE_COLORS[gene], fontWeight: 700, fontSize: 14 }}>{gene}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ color: '#90a4ae', fontSize: 11 }}>{GENE_INFO[gene]?.locus} · {GENE_INFO[gene]?.size?.split('(')[0]?.trim()} · {GENE_INFO[gene]?.inh}</div>
                    <div style={{ color: '#cfd8dc', fontSize: 12, marginTop: 4 }}>{overview.pathognomonic_signs?.[gene]}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="TERC Level Diagnostic Matrix" color="#006064">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
              {[
                { profile: 'TERC Low', genes: 'DKC1, NHP2, NOP10, PARN', mechanism: 'TERC destabilised or degraded', color: '#b71c1c', test: 'TRAP assay reduced; Northern blot low TERC' },
                { profile: 'TERC Normal (short telomere)', genes: 'WRAP53, ACD (TEL-patch)', mechanism: 'TERC present but not recruited to telomere/Cajal body', color: '#006064', test: 'TRAP assay normal; Cajal body co-localisation absent' },
                { profile: 'TERC Haploinsufficient', genes: 'TERC, TERT (AD)', mechanism: '50% TERC from one allele; insufficient over decades', color: '#1a237e', test: 'TRAP ~50% of normal; flow-FISH <10th centile' },
              ].map(({ profile, genes: g, mechanism, color, test }) => (
                <div key={profile} style={{ background: '#1e2a31', border: `1px solid ${color}`, borderRadius: 8, padding: '10px 14px', minWidth: 280 }}>
                  <div style={{ color, fontWeight: 700, fontSize: 13, marginBottom: 4 }}>{profile}</div>
                  <div style={{ color: '#cfd8dc', fontSize: 12 }}>Genes: {g}</div>
                  <div style={{ color: '#90a4ae', fontSize: 11, marginTop: 4 }}>{mechanism}</div>
                  <div style={{ color: '#546e7a', fontSize: 10, marginTop: 4 }}>Test: {test}</div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Treatment Matrix by Severity" color="#1b5e20">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12 }}>
              {[
                { severity: 'Mild (IPF/isolated AA adult)', genes: 'TERC, TERT, PARN hypomorphic', tx: 'Androgens (danazol); pirfenidone/nintedanib for IPF; annual surveillance', color: '#1b5e20' },
                { severity: 'Moderate (DC triad + BMF)', genes: 'DKC1, NHP2, NOP10, WRAP53, ACD-TEL-patch', tx: 'Androgens first; RIC-HSCT for severe BMF; TBI CI; oral leucoplakia surveillance', color: '#f57f17' },
                { severity: 'Severe (HH/SCID + cerebellar)', genes: 'DKC1-HH, TERT AR, ACD AR, PARN null, RTEL1', tx: 'Early HSCT-RIC before cerebellar damage irreversible; TBI ABSOLUTELY CI; immunoglobulin replacement', color: '#b71c1c' },
              ].map(({ severity, genes: g, tx, color }) => (
                <div key={severity} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #263238' }}>
                  <div style={{ color, fontWeight: 700, fontSize: 12, marginBottom: 4 }}>{severity}</div>
                  <div style={{ color: '#90a4ae', fontSize: 11 }}>Genes: {g}</div>
                  <div style={{ color: '#cfd8dc', fontSize: 11, marginTop: 4 }}>Treatment: {tx}</div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Biochemical Fingerprints by Gene" color="#37474f">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
              {[
                { gene: 'DKC1', marker: 'TERC reduced; short telomeres <1st centile', note: 'H/ACA snoRNP cannot stabilise TERC', color: '#b71c1c' },
                { gene: 'TERC', marker: 'TERC 50% (haploinsufficient)', note: 'Telomeres <10th centile; anticipation pedigree', color: '#1a237e' },
                { gene: 'TERT', marker: 'Telomerase activity ~50% AD; near-zero AR', note: 'IPF UIP pattern; TERT promoter androgen-responsive', color: '#004d40' },
                { gene: 'NHP2', marker: 'TERC reduced; rRNA pseudouridylation impaired', note: 'Val126Met; equal sex ratio; <1st centile', color: '#4a148c' },
                { gene: 'NOP10', marker: 'TERC reduced; Arg34Trp structural disruption', note: 'Ultra-rare; <1st centile; consanguinity', color: '#bf360c' },
                { gene: 'WRAP53', marker: 'TERC NORMAL; TRAP NORMAL; telomeres SHORT', note: 'Cajal body trafficking failure; F164L', color: '#006064' },
                { gene: 'ACD', marker: 'TERC normal; TERT present; processivity reduced', note: 'TEL-patch K170del; shelterin deprotection (AR)', color: '#33691e' },
                { gene: 'PARN', marker: 'TERC 3\' oligo-A tail; TERC reduced', note: 'Northern blot pathognomonic; PAPD5-inhibitor rescues', color: '#37474f' },
              ].map(({ gene, marker, note, color }) => (
                <div key={gene} style={{ background: '#1e2a31', border: `1px solid ${color}`, borderRadius: 8, padding: '10px 14px', minWidth: 220 }}>
                  <div style={{ color, fontWeight: 700, fontSize: 13, marginBottom: 4 }}>{gene}</div>
                  <div style={{ color: '#cfd8dc', fontSize: 11 }}>↑ {marker}</div>
                  <div style={{ color: '#546e7a', fontSize: 10, marginTop: 4 }}>{note}</div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          <Section title="Glossary" color="#b71c1c">
            <div style={{ columns: 1, gap: 20 }}>
              {Object.entries(definitions.terms || {}).map(([term, def]) => (
                <div key={term} style={{ background: '#1e2a31', borderRadius: 8, padding: '10px 14px', marginBottom: 10, breakInside: 'avoid' }}>
                  <div style={{ color: '#b71c1c', fontWeight: 700, fontSize: 12, marginBottom: 4 }}>{term.replace(/_/g, ' ')}</div>
                  <div style={{ color: '#cfd8dc', fontSize: 11, lineHeight: 1.6 }}>{def}</div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: 40, color: '#37474f', fontSize: 10, borderTop: '1px solid #1e2a31', paddingTop: 12 }}>
        Hereditary-Telomere-Biology-Atlas · 8-gene · 320 patients · seeds 2790-2797 · DKC1-TERC-TERT-NHP2-NOP10-WRAP53-ACD-PARN · Registered 2026-09-15
      </div>
    </div>
  );
}
